"""2D point robot in a unit square with a gapped wall.

Single-integrator dynamics, dt = 0.01.  Everything is batched over N robots
in torch so the same code serves evaluation (N=200) and PPO (N=512).  A thin
single-robot Gymnasium-style wrapper (PointEnv) sits at the bottom.
"""
import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DT = 0.01
STEPS_PER_SKILL = 100
N_SKILLS = 3
WALL_X = 0.5


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(override=None):
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Wall:
    """Vertical wall at x=0.5 with a centred gap of width w."""

    def __init__(self, w):
        self.w = w
        self.lo, self.hi = 0.5 - w / 2, 0.5 + w / 2

    def crosses(self, a, b):
        """True where segment a->b crosses x=0.5 outside the gap. a, b: (n,2)."""
        da, db = a[:, 0] - WALL_X, b[:, 0] - WALL_X
        cross = (da * db) < 0
        t = da / (da - db + 1e-12)
        y = a[:, 1] + t * (b[:, 1] - a[:, 1])
        return cross & ((y < self.lo) | (y > self.hi))

    def blocked(self, x):
        """True where a point sits inside the wall's exclusion band (|x-0.5|<1e-3)."""
        return ((x[:, 0] - WALL_X).abs() < 1e-3) & ((x[:, 1] < self.lo) | (x[:, 1] > self.hi))

    def near(self, x, margin=0.02):
        return ((x[:, 0] - WALL_X).abs() < margin) & ((x[:, 1] < self.lo) | (x[:, 1] > self.hi))


class Region:
    """Axis-aligned Gaussian, truncated to free space by rejection."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def sample(self, n, wall, device, gen=None):
        out = torch.empty(0, 2)
        while out.shape[0] < n:
            x = self.mean + self.std * torch.randn(2 * n, 2, generator=gen)
            ok = (x >= 0).all(1) & (x <= 1).all(1) & ~wall.blocked(x)
            # points on the far side of the wall from the mean are unreachable
            ok &= torch.sign(x[:, 0] - WALL_X) == torch.sign(self.mean[0] - WALL_X)
            out = torch.cat([out, x[ok]])
        return out[:n].to(device)

    def inside_2std(self, x):
        z = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return (z ** 2).sum(1) <= 4.0


def regions(w):
    return [
        Region((0.15, 0.5), (0.06, 0.15)),
        Region((0.42, 0.5), (0.03, 0.4 * w)),
        Region((0.58, 0.5), (0.03, 0.4 * w)),
        Region((0.85, 0.5), (0.06, 0.15)),
    ]


def w2(a, b):
    """Exact W2 between two equal-size point clouds (assignment problem)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    c = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r, cidx = linear_sum_assignment(c)
    return float(np.sqrt(c[r, cidx].mean()))


def clip_u(u):
    n = u.norm(dim=1, keepdim=True).clamp_min(1.0)
    return u / n


class BatchEnv:
    """N robots stepping in lockstep.  step(u, eps) -> obs, reward, terminated, truncated, info."""

    def __init__(self, w, sigma_d, n, seed, device, sigma_d_choices=None):
        self.w, self.n, self.device = w, n, device
        self.sigma_d_fixed = sigma_d
        self.sigma_d_choices = sigma_d_choices  # if set, resampled per episode
        self.wall = Wall(w)
        self.rho = regions(w)
        self.gen = torch.Generator().manual_seed(seed)
        self.dev_gen = torch.Generator(device=device).manual_seed(seed)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.dev_gen, device=self.device)

    def _sigma_d(self, m):
        if self.sigma_d_choices is None:
            return torch.full((m,), float(self.sigma_d_fixed), device=self.device)
        idx = torch.randint(len(self.sigma_d_choices), (m,), generator=self.gen)
        return torch.tensor(self.sigma_d_choices, device=self.device)[idx.to(self.device)]

    def obs(self):
        k1h = torch.nn.functional.one_hot(self.k.clamp(max=N_SKILLS - 1), N_SKILLS).float()
        tau = (self.t.float() / STEPS_PER_SKILL).unsqueeze(1)
        return torch.cat([self.x, k1h, tau], 1)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
            self.x = torch.zeros(self.n, 2, device=self.device)
            self.k = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.t = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.effort = torch.zeros(self.n, device=self.device)
            self.length = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.sigma_d = torch.zeros(self.n, device=self.device)
            self.handoff = [torch.full((self.n, 2), float("nan"), device=self.device) for _ in range(2)]
        m = int(mask.sum())
        if m:
            self.x[mask] = self.rho[0].sample(m, self.wall, self.device, self.gen)
            self.k[mask] = 0
            self.t[mask] = 0
            self.effort[mask] = 0
            self.length[mask] = 0
            self.sigma_d[mask] = self._sigma_d(m)
            for h in self.handoff:
                h[mask] = float("nan")
        return self.obs()

    def step(self, u, eps):
        """u: (n,2) control, eps: (n,) bridge noise level."""
        u = clip_u(u)
        eps = eps.reshape(-1)
        noise = (eps * DT).sqrt().unsqueeze(1) * self._randn(self.n, 2)
        proc = (0.2 * self.sigma_d * DT ** 0.5).unsqueeze(1) * self._randn(self.n, 2)
        x_new = (self.x + u * DT + noise + proc).clamp(0.0, 1.0)
        collided = self.wall.crosses(self.x, x_new)
        self.x = x_new
        self.t += 1
        self.length += 1
        u2 = (u ** 2).sum(1)
        self.effort += u2 * DT
        reward = -0.01 * u2 - 0.001 + torch.where(collided, -1.0, 0.0)

        handoff = (self.t >= STEPS_PER_SKILL) & ~collided
        success = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if handoff.any():
            k_next = self.k + 1
            for j in range(2):
                sel = handoff & (k_next == j + 1)
                self.handoff[j][sel] = self.x[sel]
            done_k = handoff & (k_next == N_SKILLS)
            success = done_k & self.rho[3].inside_2std(self.x)
            reward = reward + success.float()
            mid = handoff & ~done_k
            kick = self.sigma_d.unsqueeze(1) * self._randn(self.n, 2)
            self.x = torch.where(mid.unsqueeze(1), (self.x + kick).clamp(0, 1), self.x)
            self.k = torch.where(handoff, k_next, self.k)
            self.t = torch.where(handoff, torch.zeros_like(self.t), self.t)
        terminated = collided | (self.k >= N_SKILLS)
        info = {"collision": collided, "success": success, "effort": self.effort.clone(),
                "length": self.length.clone()}
        return self.obs(), reward, terminated, torch.zeros_like(terminated), info


class PointEnv:
    """Single-robot Gymnasium-style view: reset() -> obs, step(action) -> 5-tuple.
    action = (ux, uy, eps)."""

    def __init__(self, w=0.2, sigma_d=0.0, seed=0, device="cpu"):
        self.env = BatchEnv(w, sigma_d, 1, seed, torch.device(device))

    def reset(self, seed=None):
        return self.env.reset()[0].cpu().numpy(), {}

    def step(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.env.device).reshape(3)
        obs, r, term, trunc, info = self.env.step(a[:2].unsqueeze(0), a[2:3])
        info = {k: v[0].item() for k, v in info.items()}
        return obs[0].cpu().numpy(), r.item(), bool(term[0]), bool(trunc[0]), info
