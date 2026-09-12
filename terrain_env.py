"""Terrain environment: point robot on a random roughness field with slip,
roughness-scaled process noise and random pushes.  No wall.

Fixed route ρ₀..ρ₃ (means on y = 0.5, std 0.05); layouts differ only in the
roughness field, which is resampled until every route segment crosses a
high-roughness region (r >= 0.7).  Batched over n robots with a per-robot
layout id so training can mix layouts.
"""
import math
import numpy as np
import torch
from env import DT, STEPS_PER_SKILL, N_SKILLS, clip_u, w2  # noqa: F401

GRID = 64
ROUTE = torch.tensor([[0.12, 0.5], [0.37, 0.5], [0.63, 0.5], [0.88, 0.5]])
WP_STD = 0.05
N_RAYS, RAY_R = 16, 0.08
ROUGH = {"mild": (0.3, 0.3, 0.05), "rough": (0.6, 0.8, 0.10)}  # (s, theta, sigma_p)
OBS_DIM = 2 + N_SKILLS + N_RAYS


def make_field(layout_id, n_wave=6, min_peak=0.7, clearing=0.06):
    """Sum of 6 random sinusoids on a 64x64 grid, min-max normalised to [0,1],
    with smooth Gaussian clearings (std `clearing`) carved at the four waypoint
    means so the targets are holdable.  Resampled until each route segment
    crosses r >= min_peak."""
    rng = np.random.RandomState(1000 + layout_id)
    g = (np.arange(GRID) + 0.5) / GRID
    X, Y = np.meshgrid(g, g, indexing="xy")  # X varies along columns
    while True:
        f = np.zeros((GRID, GRID))
        for _ in range(n_wave):
            freq, ang, ph = rng.uniform(1.5, 4.0), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
            f += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * freq * (X * np.cos(ang) + Y * np.sin(ang)) + ph)
        f = (f - f.min()) / (f.max() - f.min())
        for m in ROUTE.numpy():
            f *= 1 - np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (2 * clearing ** 2))
        f = f / f.max()
        ft = torch.tensor(f, dtype=torch.float32)[None]
        ok = True
        for j in range(N_SKILLS):
            s = torch.linspace(0, 1, 50)[:, None]
            pts = ROUTE[j] * (1 - s) + ROUTE[j + 1] * s
            ok &= bool(roughness(ft, pts, torch.zeros(50, dtype=torch.long)).max() >= min_peak)
        if ok:
            return ft[0]


def load_fields(ids, device):
    return torch.stack([make_field(i) for i in ids]).to(device)


def roughness(fields, x, lid):
    """Bilinear lookup.  fields: (L,GRID,GRID) indexed [row=y, col=x]; x: (n,2); lid: (n,)."""
    p = (x.clamp(0, 1) * GRID - 0.5).clamp(0, GRID - 1)
    i0 = p.floor().long().clamp(max=GRID - 2)
    fr = p - i0.float()
    cx, cy = i0[:, 0], i0[:, 1]
    f00 = fields[lid, cy, cx]; f10 = fields[lid, cy, cx + 1]
    f01 = fields[lid, cy + 1, cx]; f11 = fields[lid, cy + 1, cx + 1]
    fx, fy = fr[:, 0], fr[:, 1]
    return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy) + f01 * (1 - fx) * fy + f11 * fx * fy)


_ANG = torch.arange(N_RAYS) * 2 * math.pi / N_RAYS
_RAYS = torch.stack([_ANG.cos(), _ANG.sin()], 1) * RAY_R  # (16,2)


def rays(fields, x, lid):
    """Local roughness on a circle of radius 0.08 around x -> (n,16)."""
    n = x.shape[0]
    pts = (x[:, None, :] + _RAYS.to(x.device)[None]).reshape(-1, 2)
    return roughness(fields, pts, lid.repeat_interleave(N_RAYS)).reshape(n, N_RAYS)


def slip(fields, x, lid, u, s, theta):
    """Actual velocity = (1 - s r) R(theta r) u."""
    r = roughness(fields, x, lid)
    a = theta * r
    c, sn = a.cos(), a.sin()
    v = torch.stack([c * u[:, 0] - sn * u[:, 1], sn * u[:, 0] + c * u[:, 1]], 1)
    return (1 - s * r).unsqueeze(1) * v, r


def waypoint_sample(k, n, device, gen=None):
    return ROUTE[k].to(device) + WP_STD * torch.randn(n, 2, generator=gen, device=device)


def goal_sample(k, n, device, gen=None):
    """rho_k sample truncated to 1.5 std (targets for the MPC / demos): aim well inside the 2-std success set."""
    out = torch.empty(0, 2, device=device)
    while out.shape[0] < n:
        x = waypoint_sample(k, 2 * n, device, gen)
        z = ((x - ROUTE[k].to(device)) / WP_STD) ** 2
        out = torch.cat([out, x[z.sum(1) <= 2.25]])
    return out[:n]


def inside_goal(x):
    return (((x - ROUTE[3].to(x.device)) / WP_STD) ** 2).sum(1) <= 4.0


_CUM = torch.cat([torch.zeros(1), (ROUTE[1:] - ROUTE[:-1]).norm(dim=1).cumsum(0)])


def corridor_coords(x):
    """(progress along the route polyline, lateral distance) for x: (n,2)."""
    best_d, best_p = None, None
    for j in range(N_SKILLS):
        a, b = ROUTE[j].to(x.device), ROUTE[j + 1].to(x.device)
        ab = b - a
        s = (((x - a) * ab).sum(1) / (ab ** 2).sum()).clamp(0, 1)
        d = (x - (a + s[:, None] * ab)).norm(dim=1)
        p = _CUM[j].item() + s * ab.norm()
        if best_d is None:
            best_d, best_p = d, p
        else:
            better = d < best_d
            best_d, best_p = torch.where(better, d, best_d), torch.where(better, p, best_p)
    return best_p, best_d


def segment_dist(x, k):
    R = ROUTE.to(x.device)
    a, b = R[k], R[k + 1]
    ab = b - a
    s = (((x - a) * ab).sum(1) / (ab ** 2).sum()).clamp(0, 1)
    return (x - (a + s[:, None] * ab)).norm(dim=1)


class TerrainEnv:
    """Gymnasium-style batched env.  step(u) -> obs, reward, terminated, truncated, info.
    obs = x ⊕ one-hot k ⊕ 16 rays."""

    def __init__(self, fields, lids, rough, sigma_k, p_push, seed, device, disturb=True):
        self.fields, self.lids, self.device = fields, lids.to(device), device
        self.n = lids.shape[0]
        self.s, self.theta, self.sigma_p = ROUGH[rough] if isinstance(rough, str) else rough
        self.sigma_k, self.p_push, self.disturb = sigma_k, p_push, disturb
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.REC_TOL = 0.05

    def _randn(self, *s):
        return torch.randn(*s, generator=self.gen, device=self.device)

    def obs(self):
        k1h = torch.nn.functional.one_hot(self.k.clamp(max=N_SKILLS - 1), N_SKILLS).float()
        return torch.cat([self.x, k1h, rays(self.fields, self.x, self.lids)], 1)

    def reset(self, mask=None):
        n, d = self.n, self.device
        if mask is None:
            mask = torch.ones(n, dtype=torch.bool, device=d)
            self.x = torch.zeros(n, 2, device=d); self.k = torch.zeros(n, dtype=torch.long, device=d)
            self.t = torch.zeros(n, dtype=torch.long, device=d); self.effort = torch.zeros(n, device=d)
            self.dev_sum = torch.zeros(n, device=d)
            self.handoff = [torch.zeros(n, 2, device=d) for _ in range(2)]
            self.push_active = torch.zeros(n, dtype=torch.bool, device=d)
            self.push_t = torch.zeros(n, dtype=torch.long, device=d)
            self.pre_prog = torch.zeros(n, device=d); self.pre_lat = torch.zeros(n, device=d)
            self.push_log = []  # (robot, push step, recovery steps or -1)
        m = int(mask.sum())
        if m:
            self.x[mask] = waypoint_sample(0, m, d, self.gen)
            self.k[mask] = 0; self.t[mask] = 0; self.effort[mask] = 0; self.dev_sum[mask] = 0
            self.push_active[mask] = False
        return self.obs()

    def step(self, u):
        u = clip_u(u)
        v, r = slip(self.fields, self.x, self.lids, u, self.s, self.theta)
        x_new = self.x + v * DT + (self.sigma_p ** 2 * r * DT).sqrt().unsqueeze(1) * self._randn(self.n, 2)
        pushed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if self.disturb and self.sigma_k > 0:
            pushed = torch.rand(self.n, generator=self.gen, device=self.device) < self.p_push
            prog0, lat0 = corridor_coords(x_new)
            kick = self.sigma_k * self._randn(self.n, 2)
            x_new = torch.where(pushed[:, None], x_new + kick, x_new)
            prog1, lat1 = corridor_coords(x_new)
            # only pushes that move the robot laterally (off its corridor lane) by > tol are tracked;
            # a push that lands while another is still open supersedes it (the open one is dropped)
            eff = pushed & ((lat1 - lat0).abs() > self.REC_TOL)
            self.push_active |= eff
            self.push_t = torch.where(eff, self.t, self.push_t)
            self.pre_prog = torch.where(eff, prog0, self.pre_prog); self.pre_lat = torch.where(eff, lat0, self.pre_lat)
        self.x = x_new.clamp(0, 1)
        self.t += 1
        u2 = (u ** 2).sum(1); self.effort += u2 * DT
        self.dev_sum += segment_dist(self.x, self.k.clamp(max=N_SKILLS - 1))
        # recovery: back in lane (lateral distance within tol of its pre-push value)
        if self.push_active.any():
            prog, lat = corridor_coords(self.x)
            rec = self.push_active & ((lat - self.pre_lat).abs() <= self.REC_TOL)
            for i in rec.nonzero().flatten().tolist():
                self.push_log.append((i, int(self.push_t[i]), int(self.t[i] - self.push_t[i])))
            self.push_active &= ~rec
        reward = -0.01 * u2 - 0.001
        handoff = self.t >= STEPS_PER_SKILL
        success = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if handoff.any():
            k_next = self.k + 1
            for j in range(2):
                sel = handoff & (k_next == j + 1)
                self.handoff[j][sel] = self.x[sel]
            done_k = handoff & (k_next == N_SKILLS)
            success = done_k & inside_goal(self.x)
            reward = reward + success.float()
            self.k = torch.where(handoff, k_next, self.k)
            self.t = torch.where(handoff, torch.zeros_like(self.t), self.t)
        terminated = self.k >= N_SKILLS
        if terminated.any():
            for i in (terminated & self.push_active).nonzero().flatten().tolist():
                self.push_log.append((i, int(self.push_t[i]), -1))
            self.push_active &= ~terminated
        info = {"success": success, "pushed": pushed, "effort": self.effort.clone(),
                "corridor_dev": self.dev_sum / (STEPS_PER_SKILL * N_SKILLS)}
        return self.obs(), reward, terminated, torch.zeros_like(terminated), info
