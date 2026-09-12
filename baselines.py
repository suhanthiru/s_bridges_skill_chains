"""Baselines.

D  : bridge-matching net at IPF iteration 0 (stochastic interpolant, independent
     coupling, no refinement) -> bridge.train_bridge(..., ipf_iters=0).  Same
     PPO eps-policy as condition C on top.
D' : deterministic-interpolant conditional flow matching (optional, weaker).
E  : plain PPO, Gaussian policy over u, no skills, same reward and disturbance
     schedule (kicks at steps 100 and 200).
"""
import math
import torch
import torch.nn as nn
from env import BatchEnv, N_SKILLS
from bridge import DriftNet, sample_pairs, sample_log_eps
from rl import mlp, SIGMA_D_SWEEP


def train_flow(rho_a, rho_b, wall, seed, device, steps=4000, batch=1024, lr=1e-3, penalty=10.0, n_pairs=20000):
    """Conditional flow matching: x_tau = (1-tau)x0 + tau x1, target x1 - x0.
    eps is fed as an input for signature parity but does not enter the target."""
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    net = DriftNet().to(device)
    x0, x1 = sample_pairs(rho_a, rho_b, wall, n_pairs, device, cpu_gen)
    log_eps = sample_log_eps(n_pairs, device, gen)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(n_pairs, (batch,), generator=gen, device=device)
        a, b, le = x0[idx], x1[idx], log_eps[idx]
        tau = torch.rand(batch, device=device)
        xt = (1 - tau).unsqueeze(1) * a + tau.unsqueeze(1) * b
        pred = net(xt, tau, le)
        mse = ((pred - (b - a)) ** 2).sum(1).mean()
        near = wall.near(xt, 0.02)
        pen = (torch.relu(pred[:, 0] * torch.sign(0.5 - xt[:, 0])) ** 2 * near.float()).mean()
        loss = mse + penalty * pen
        opt.zero_grad(); loss.backward(); opt.step()
    return net


class GaussianPolicy(nn.Module):
    """pi(u | x, t/300) for plain PPO; u = tanh(a) so |u| <= 1 per axis, then norm-clipped by env."""
    d_act = 2

    def __init__(self):
        super().__init__()
        self.mu = mlp(3, 2)
        self.log_std = nn.Parameter(torch.full((2,), math.log(0.5)))

    def features(self, obs):
        # obs = (x, one-hot k, tau); global time = (k + tau) / 3
        k = obs[:, 2:2 + N_SKILLS].argmax(1).float()
        return torch.cat([obs[:, :2], ((k + obs[:, -1]) / N_SKILLS).unsqueeze(1)], 1)

    def dist(self, obs):
        mu = self.mu(self.features(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @staticmethod
    def to_env(a):
        return torch.tanh(a), None


class PlainEnv:
    """BatchEnv with no bridge noise (eps = 0); the policy supplies u directly."""

    def __init__(self, w, n, seed, device, sigma_d=None):
        self.env = BatchEnv(w, sigma_d if sigma_d is not None else 0.0, n, seed, device,
                            sigma_d_choices=None if sigma_d is not None else list(SIGMA_D_SWEEP))
        self.device = device

    def reset(self, mask=None):
        return self.env.reset(mask)

    def step(self, u, eps):
        return self.env.step(u, torch.zeros(self.env.n, device=self.device))


class PlainCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = mlp(3, 1)
        self.feat = GaussianPolicy.features

    def forward(self, obs):
        return self.v(self.feat(self, obs)).squeeze(1)
