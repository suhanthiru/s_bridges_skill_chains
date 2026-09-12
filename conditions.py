"""Conditions wired to the terrain env.

BRIDGE-plain / BRIDGE-obs : iteration-0 bridge matching on demo endpoints.
TRACK                     : nearest demo replay + PD tracking.
DIFF                      : diffusion policy over 8-step action chunks (DDPM 50 train / DDIM 10 test).
PPO                       : goal-conditioned PPO on x ⊕ one-hot k ⊕ rays (rl.ppo_train core).
ORACLE                    : the demo MPC run live with pushes (demos.MPC).
"""
import math
import numpy as np
import torch
import torch.nn as nn
from env import DT, STEPS_PER_SKILL, N_SKILLS, clip_u
from bridge import sinusoidal
from rl import mlp
import terrain_env as T
import demos as D

CHUNK, EXEC = 8, 4


# ------------------------------------------------------------------ bridges
class TerrainDrift(nn.Module):
    """f(x, tau[, rays]) -> u.  4 x 256 SiLU, sinusoidal tau."""

    def __init__(self, use_obs, hidden=256, n_freq=16):
        super().__init__()
        self.use_obs, self.n_freq = use_obs, n_freq
        d_in = 2 + 2 * n_freq + (T.N_RAYS if use_obs else 0)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, tau, o=None):
        h = [x, sinusoidal(tau, self.n_freq)] + ([o] if self.use_obs else [])
        return self.net(torch.cat(h, 1))


def train_terrain_bridge(fields, x0, x1, lid, eps, use_obs, seed, device, steps=4000, batch=1024, lr=1e-3, log=print):
    """Iteration-0 bridge matching: Brownian bridge with reference noise eps between demo endpoints."""
    torch.manual_seed(seed)
    net = TerrainDrift(use_obs).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = x0.shape[0]
    for s in range(steps):
        idx = torch.randint(n, (batch,), device=device)
        a, b, l = x0[idx], x1[idx], lid[idx]
        tau = torch.rand(batch, device=device)
        xt = (1 - tau)[:, None] * a + tau[:, None] * b + (eps * tau * (1 - tau)).sqrt()[:, None] * torch.randn_like(a)
        target = (b - xt) / (1 - tau).clamp_min(0.02)[:, None]
        pred = net(xt, tau, T.rays(fields, xt, l) if use_obs else None)
        loss = ((pred - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            log(f"  step {s} loss {loss.item():.4f}")
    return net


class BridgeCtl:
    def __init__(self, nets, use_obs):
        self.nets, self.use_obs = nets, use_obs

    @torch.no_grad()
    def act(self, env, obs):
        u = torch.zeros(env.n, 2, device=env.device)
        tau = env.t.float() * DT
        for k, net in enumerate(self.nets):
            sel = env.k == k
            if sel.any():
                u[sel] = net(env.x[sel], tau[sel], obs[sel, 2 + N_SKILLS:] if self.use_obs else None)
        return u


# -------------------------------------------------------------------- TRACK
class TrackCtl:
    """Nearest demo (by start state) per skill as the nominal path; PD on position error + feed-forward."""

    def __init__(self, demo_states, demo_actions, kp, kd, device):
        self.S = [torch.tensor(s, device=device) for s in demo_states]  # per skill (N,101,2)
        self.A = [torch.tensor(a, device=device) for a in demo_actions]
        self.kp, self.kd, self.device = kp, kd, device
        self.ref_idx, self.e_prev = None, None

    @torch.no_grad()
    def act(self, env, obs):
        n = env.n
        if self.ref_idx is None:
            self.ref_idx = torch.zeros(n, dtype=torch.long, device=self.device)
            self.e_prev = torch.zeros(n, 2, device=self.device)
        u = torch.zeros(n, 2, device=self.device)
        for k in range(N_SKILLS):
            sel = env.k == k
            if not sel.any():
                continue
            start = sel & (env.t == 0)
            if start.any():
                dist = torch.cdist(env.x[start], self.S[k][:, 0])
                self.ref_idx[start] = dist.argmin(1); self.e_prev[start] = 0
            t = env.t[sel].clamp(max=STEPS_PER_SKILL - 1)
            idx = self.ref_idx[sel]
            x_ref, u_ff = self.S[k][idx, t + 1], self.A[k][idx, t]
            e = x_ref - env.x[sel]
            u[sel] = u_ff + self.kp * e + self.kd * (e - self.e_prev[sel]) / DT
            self.e_prev[sel] = e
        return u


# --------------------------------------------------------------------- DIFF
class DiffusionPolicy(nn.Module):
    """Conditional denoiser over an 8x2 action chunk; obs = x ⊕ one-hot k ⊕ rays."""

    def __init__(self, n_train=50, hidden=256):
        super().__init__()
        self.n_train = n_train
        self.net = nn.Sequential(nn.Linear(CHUNK * 2 + T.OBS_DIM + 32, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, CHUNK * 2))
        beta = torch.linspace(1e-4, 0.02, n_train)
        self.register_buffer("abar", torch.cumprod(1 - beta, 0))

    def forward(self, a_noisy, obs, t):
        return self.net(torch.cat([a_noisy, obs, sinusoidal(t.float() / self.n_train, 16)], 1))

    @torch.no_grad()
    def sample(self, obs, n_ddim=10):
        n = obs.shape[0]
        a = torch.randn(n, CHUNK * 2, device=obs.device)
        ts = torch.linspace(self.n_train - 1, 0, n_ddim + 1).long()[:-1]
        for i, t in enumerate(ts):
            ab = self.abar[t]; ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
            eps = self(a, obs, torch.full((n,), int(t), device=obs.device))
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        return a.reshape(n, CHUNK, 2)


def train_diffusion(fields, obs_all, act_all, seed, device, steps=20000, batch=1024, lr=3e-4, log=print):
    """obs_all: (N, OBS_DIM); act_all: (N, CHUNK, 2) windows from the demos."""
    torch.manual_seed(seed)
    pol = DiffusionPolicy().to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    N = obs_all.shape[0]
    for s in range(steps):
        idx = torch.randint(N, (batch,), device=device)
        a0 = act_all[idx].reshape(batch, -1); o = obs_all[idx]
        t = torch.randint(pol.n_train, (batch,), device=device)
        ab = pol.abar[t][:, None]
        noise = torch.randn_like(a0)
        loss = ((pol(ab.sqrt() * a0 + (1 - ab).sqrt() * noise, o, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 5000 == 0:
            log(f"  step {s} loss {loss.item():.4f}")
    return pol


def demo_windows(S, A, lids, fields, k, device):
    """All 8-step windows of the demos: obs at window start (x, one-hot k, rays), chunk of 8 actions."""
    S, A, lids = torch.tensor(S, device=device), torch.tensor(A, device=device), torch.tensor(lids, device=device)
    n = S.shape[0]
    starts = torch.arange(0, STEPS_PER_SKILL - CHUNK + 1, 2, device=device)
    x = S[:, starts].reshape(-1, 2)
    lid = lids.repeat_interleave(len(starts))
    k1h = torch.zeros(x.shape[0], N_SKILLS, device=device); k1h[:, k] = 1
    obs = torch.cat([x, k1h, T.rays(fields, x, lid)], 1)
    act = torch.stack([A[:, s:s + CHUNK] for s in starts.tolist()], 1).reshape(-1, CHUNK, 2)
    return obs, act


class DiffCtl:
    def __init__(self, pol):
        self.pol, self.buf, self.ptr = pol, None, EXEC

    @torch.no_grad()
    def act(self, env, obs):
        if self.buf is None or self.ptr >= EXEC:
            self.buf, self.ptr = self.pol.sample(obs), 0
        u = self.buf[:, self.ptr]; self.ptr += 1
        return u


# ---------------------------------------------------------------------- PPO
class TerrainPolicy(nn.Module):
    d_act = 2

    def __init__(self):
        super().__init__()
        self.mu = mlp(T.OBS_DIM, 2)
        self.log_std = nn.Parameter(torch.full((2,), math.log(0.5)))

    def dist(self, obs):
        mu = self.mu(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @staticmethod
    def to_env(a):
        return torch.tanh(a), None


class TerrainCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = mlp(T.OBS_DIM, 1)

    def forward(self, obs):
        return self.v(obs).squeeze(1)


class PPOEnvWrap:
    """Adapts TerrainEnv to rl.ppo_train's (u, eps) step signature."""

    def __init__(self, env):
        self.env = env

    def reset(self, mask=None):
        return self.env.reset(mask)

    def step(self, u, eps):
        return self.env.step(u)


class PPOCtl:
    def __init__(self, pol):
        self.pol = pol

    @torch.no_grad()
    def act(self, env, obs):
        return torch.tanh(self.pol.mu(obs))


class OracleCtl:
    def __init__(self, fields, lids, rough, seed):
        self.mpc = D.MPC(fields, lids, rough, seed=seed)
        self.goals, self.gen = None, torch.Generator(device=fields.device).manual_seed(seed + 1)

    @torch.no_grad()
    def act(self, env, obs):
        if self.goals is None:
            self.goals = torch.zeros(env.n, 2, device=env.device); self.gk = torch.full((env.n,), -1, device=env.device)
        new = env.k != self.gk
        if new.any():  # draw a fresh rho_k target sample at each skill start
            for k in range(N_SKILLS):
                sel = new & (env.k == k)
                if sel.any():
                    self.goals[sel] = T.goal_sample(k + 1, int(sel.sum()), env.device, self.gen)
            self.gk = env.k.clone()
        return self.mpc.act(env.x, self.goals)
