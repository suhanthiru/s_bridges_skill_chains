"""PPO over a frozen bridge: the action is the noise level eps_t.

The env step is one bridge integration step; the drift f_theta is frozen and
evaluated under no_grad.  The same PPO core is reused by the plain-PPO
baseline (action = u) in baselines.py.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
from env import BatchEnv, STEPS_PER_SKILL, N_SKILLS, DT
from bridge import LOG_EPS_MIN, LOG_EPS_MAX

SIGMA_D_SWEEP = (0.0, 0.02, 0.05, 0.10)


def mlp(d_in, d_out, hidden=128):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, d_out))


LOG_EPS_MID = 0.5 * (LOG_EPS_MIN + LOG_EPS_MAX)
LOG_EPS_HALF = 0.5 * (LOG_EPS_MAX - LOG_EPS_MIN)


def squash_log_eps(a):
    """Raw Gaussian sample -> log eps in [log 1e-3, log 1e-1] via tanh (no dead zones)."""
    return LOG_EPS_MID + LOG_EPS_HALF * torch.tanh(a)


class EpsPolicy(nn.Module):
    """pi_phi(x, k) -> Gaussian over a raw scalar, tanh-squashed to log eps in [log 1e-3, log 1e-1]."""
    d_act = 1

    def __init__(self):
        super().__init__()
        self.mu = mlp(2 + N_SKILLS, 1)
        self.log_std = nn.Parameter(torch.full((1,), math.log(1.0)))

    def features(self, obs):
        return obs[:, :2 + N_SKILLS]

    def dist(self, obs):
        mu = self.mu(self.features(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def mean_log_eps(self, obs):
        return squash_log_eps(self.mu(self.features(obs)).squeeze(1))

    @staticmethod
    def to_env(a):
        """Raw action -> (u, eps).  u is None: the bridge supplies it."""
        return None, squash_log_eps(a.squeeze(1)).exp()


class Critic(nn.Module):
    def __init__(self, d_in=2 + N_SKILLS):
        super().__init__()
        self.v = mlp(d_in, 1)
        self.d_in = d_in

    def forward(self, obs):
        return self.v(obs[:, :self.d_in]).squeeze(1)


class BridgeEnv:
    """BatchEnv driven by frozen per-skill drift nets; action = eps."""

    def __init__(self, nets, w, n, seed, device, sigma_d=None):
        self.nets = nets
        self.env = BatchEnv(w, sigma_d if sigma_d is not None else 0.0, n, seed, device,
                            sigma_d_choices=None if sigma_d is not None else list(SIGMA_D_SWEEP))
        self.device = device

    def reset(self, mask=None):
        return self.env.reset(mask)

    @torch.no_grad()
    def step(self, u, eps):
        e = self.env
        if u is None:
            u = torch.zeros(e.n, 2, device=self.device)
            tau = e.t.float() * DT
            for k, net in enumerate(self.nets):
                sel = e.k == k
                if sel.any():
                    u[sel] = net(e.x[sel], tau[sel], eps[sel].log())
        return e.step(u, eps)


def ppo_train(env, policy, critic, total_steps, seed, device, lr=3e-4, rollout=32, epochs=4,
              minibatch=4096, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.0, log=print):
    """Generic vectorised PPO.  env.step(u, eps); policy.to_env(a) -> (u, eps)."""
    torch.manual_seed(seed)
    n = env.env.n
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=lr)
    obs = env.reset()
    d_obs, d_act = obs.shape[1], policy.d_act
    n_updates = total_steps // (n * rollout)
    curve, ep_ret, ep_rets, ep_succ = [], torch.zeros(n, device=device), [], []
    t0 = time.time()
    for upd in range(n_updates):
        O = torch.zeros(rollout, n, d_obs, device=device)
        A = torch.zeros(rollout, n, d_act, device=device)
        LP = torch.zeros(rollout, n, device=device)
        R = torch.zeros(rollout, n, device=device)
        D = torch.zeros(rollout, n, device=device)
        V = torch.zeros(rollout + 1, n, device=device)
        with torch.no_grad():
            for t in range(rollout):
                dist = policy.dist(obs)
                a = dist.sample()
                u, eps = policy.to_env(a)
                O[t], A[t], LP[t], V[t] = obs, a, dist.log_prob(a).sum(1), critic(obs)
                obs, r, term, trunc, info = env.step(u, eps)
                done = term | trunc
                R[t], D[t] = r, done.float()
                ep_ret += r
                if done.any():
                    ep_rets += ep_ret[done].tolist()
                    ep_succ += info["success"][done].float().tolist()
                    ep_ret[done] = 0
                    obs = env.reset(done)
            V[rollout] = critic(obs)
        # GAE
        adv = torch.zeros_like(R)
        last = torch.zeros(n, device=device)
        for t in reversed(range(rollout)):
            nonterm = 1 - D[t]
            delta = R[t] + gamma * V[t + 1] * nonterm - V[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + V[:rollout]
        b_obs, b_a, b_lp = O.reshape(-1, d_obs), A.reshape(-1, d_act), LP.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        N = b_obs.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, minibatch):
                idx = perm[i:i + minibatch]
                dist = policy.dist(b_obs[idx])
                lp = dist.log_prob(b_a[idx]).sum(1)
                ratio = (lp - b_lp[idx]).exp()
                pg = -torch.min(ratio * b_adv[idx], ratio.clamp(1 - clip, 1 + clip) * b_adv[idx]).mean()
                vl = ((critic(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vl - ent_coef * dist.entropy().sum(1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
        if ep_rets:
            curve.append(dict(step=(upd + 1) * n * rollout, ret=float(np.mean(ep_rets[-200:])),
                              success=float(np.mean(ep_succ[-200:]))))
        if upd % max(1, n_updates // 10) == 0 and curve:
            log(f"  upd {upd}/{n_updates} ret={curve[-1]['ret']:.3f} succ={curve[-1]['success']:.3f} "
                f"({time.time() - t0:.0f}s)")
    return curve


@torch.no_grad()
def eps_heatmap(policy, device, grid=50):
    """Mean log eps over a grid of x for each skill k -> (3, grid, grid)."""
    g = (torch.arange(grid, device=device) + 0.5) / grid
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    x = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    out = []
    for k in range(N_SKILLS):
        k1h = torch.zeros(x.shape[0], N_SKILLS, device=device); k1h[:, k] = 1
        obs = torch.cat([x, k1h, torch.zeros(x.shape[0], 1, device=device)], 1)
        out.append(policy.mean_log_eps(obs).reshape(grid, grid).cpu().numpy())
    return np.stack(out)
