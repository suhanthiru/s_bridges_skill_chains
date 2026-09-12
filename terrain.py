"""Terrain classes, spatial layouts, obstacles and skill marginals.

Terrain is a 64x64 grid of classes laid out as spatial patches (nearest-seed
regions, not iid per cell).  Each class carries a friction coefficient, an
anisotropic body-frame slip covariance and a push-disturbance rate.  The
continuous property fields derived from the class map are looked up
bilinearly so Sigma(x) is smooth enough to integrate along a bridge.
"""
import math
import numpy as np
import torch
from se2 import rot

GRID = 64
# name: (friction, slip std long / lat / heading (body frame), push rate per step)
CLASSES = {
    "compacted": (1.00, (0.020, 0.020, 0.020), 0.002),
    "rutted":    (0.80, (0.040, 0.140, 0.100), 0.010),
    "gravel":    (0.70, (0.100, 0.100, 0.080), 0.020),
    "grass":     (0.85, (0.060, 0.060, 0.120), 0.008),
    "wet":       (0.55, (0.160, 0.160, 0.140), 0.030),
}
NAMES = list(CLASSES)
WET = NAMES.index("wet")
RAIN_FRICTION_DROP = 0.40


def class_map(layout_id, n_seed=14, flip_rate=0.0, boundary_jitter=0.0, seed_off=0):
    """Spatial patches by nearest-seed assignment.  flip_rate / boundary_jitter
    corrupt the map for the Phase 4 noisy-map condition."""
    rng = np.random.RandomState(9000 + 131 * layout_id + seed_off)
    g = (np.arange(GRID) + 0.5) / GRID
    X, Y = np.meshgrid(g, g, indexing="xy")
    if boundary_jitter > 0:
        X = X + rng.randn(GRID, GRID) * boundary_jitter
        Y = Y + rng.randn(GRID, GRID) * boundary_jitter
    sx, sy = rng.rand(n_seed), rng.rand(n_seed)
    cls = rng.randint(0, len(NAMES), n_seed)
    d = (X[..., None] - sx) ** 2 + (Y[..., None] - sy) ** 2
    m = cls[d.argmin(-1)]
    if flip_rate > 0:
        f = rng.rand(GRID, GRID) < flip_rate
        m = np.where(f, rng.randint(0, len(NAMES), (GRID, GRID)), m)
    return m.astype(np.int64)


def property_fields(cmap, rain=False):
    """(3+1+1, GRID, GRID): slip stds (3), friction, push rate."""
    fric = np.array([CLASSES[n][0] for n in NAMES])
    slip = np.array([CLASSES[n][1] for n in NAMES])
    push = np.array([CLASSES[n][2] for n in NAMES])
    if rain:
        fric = fric.copy(); fric[WET] *= (1 - RAIN_FRICTION_DROP)
        slip = slip.copy(); slip[WET] *= 1.5
    return np.concatenate([slip[cmap].transpose(2, 0, 1), fric[cmap][None], push[cmap][None]], 0).astype(np.float32)


def lookup(fields, xy):
    """Bilinear lookup of the property fields.  fields (F,G,G); xy (n,2) -> (n,F)."""
    p = (xy.clamp(0, 1) * GRID - 0.5).clamp(0, GRID - 1)
    i0 = p.floor().long().clamp(max=GRID - 2)
    fr = p - i0.float()
    cx, cy = i0[:, 0], i0[:, 1]
    fx, fy = fr[:, 0:1], fr[:, 1:2]
    f00, f10 = fields[:, cy, cx].T, fields[:, cy, cx + 1].T
    f01, f11 = fields[:, cy + 1, cx].T, fields[:, cy + 1, cx + 1].T
    return f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy) + f01 * (1 - fx) * fy + f11 * fx * fy


def slip_cov_body(fields, xy, scale=1.0):
    """Body-frame slip covariance Sigma_body(x) -> (n,3,3) diagonal."""
    p = lookup(fields, xy)
    s = (p[:, :3] * scale) ** 2
    return torch.diag_embed(s)


PROBES = torch.tensor([[0.06, 0.0], [-0.06, 0.0], [0.0, 0.06], [0.0, -0.06]])


def terrain_feats(fields, g, body_frame):
    """Local terrain features at 4 probes: (n, 4*4).  Body-frame probes for SE(2)
    (left-invariant); world-frame probes for the flat baseline."""
    n = g.shape[0]
    off = PROBES.to(g.device).expand(n, 4, 2)
    if body_frame:
        off = torch.einsum("nij,nkj->nki", rot(g[:, 2]), off)
    pts = (g[:, None, :2] + off).reshape(-1, 2)
    p = lookup(fields, pts)[:, [0, 1, 3, 4]]
    return p.reshape(n, 16)


N_TFEAT = 16


# ------------------------------------------------------------- geometry
class Layout:
    """Wall at x=0.5 with one or two gaps, plus an optional circular pile."""

    def __init__(self, name):
        self.name = name
        if name == "L1":
            self.gaps = [(0.5, 0.22)]
            self.pile = None
        else:
            self.gaps = [(0.30, 0.16), (0.70, 0.16)]
            self.pile = (0.62, 0.70, 0.075)

    def in_gap(self, y):
        ok = torch.zeros_like(y, dtype=torch.bool)
        for c, w in self.gaps:
            ok |= (y > c - w / 2) & (y < c + w / 2)
        return ok

    def collides(self, a, b):
        """Segment a->b (n,2 positions) hits the wall, the pile, or leaves the box."""
        da, db = a[:, 0] - 0.5, b[:, 0] - 0.5
        cross = (da * db) < 0
        t = da / (da - db + 1e-12)
        y = a[:, 1] + t * (b[:, 1] - a[:, 1])
        hit = cross & ~self.in_gap(y)
        if self.pile is not None:
            px, py, pr = self.pile
            d = ((b[:, 0] - px) ** 2 + (b[:, 1] - py) ** 2).sqrt()
            hit |= d < pr
        hit |= (b < 0).any(1) | (b > 1).any(1)
        return hit


def marginals(layout, route="lower", gap_lat=0.040):
    """Four SE(2) marginals (mean, tangent covariance) along the route."""
    if layout.name == "L1":
        ys = [0.5, 0.5, 0.5, 0.5]
    else:
        y = 0.30 if route == "lower" else 0.70
        ys = [0.5, y, y, 0.5]
    xs = [0.12, 0.40, 0.60, 0.88]
    th = [0.0, 0.0, 0.0, 0.0]
    if layout.name == "L2":
        th = [math.atan2(ys[1] - ys[0], xs[1] - xs[0]), 0.0, 0.0,
              math.atan2(ys[3] - ys[2], xs[3] - xs[2])]
    wide = torch.diag(torch.tensor([0.050, 0.080, 0.250]) ** 2)
    narrow = torch.diag(torch.tensor([0.025, gap_lat, 0.150]) ** 2)
    means = [torch.tensor([x, y, t]) for x, y, t in zip(xs, ys, th)]
    covs = [wide, narrow, narrow, wide]
    return means, covs
