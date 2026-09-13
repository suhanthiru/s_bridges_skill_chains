"""Phase 1b: does the Lie group matter?

Synthetic, terrain-free routes with configurable curvature, heading-noise scale h
and body-frame slip anisotropy a (lateral/longitudinal).  Reference = `slip`
(best of Phase 1) with a constant body-frame covariance diag(s^2, (a s)^2, (h/2)^2),
iteration-0 bridges (IPF did not help in Phase 1).  Both manifolds train on the
same marginal samples; physics and metrics are SE(2) for both.

Checks: covariance fidelity (analytic vs 10k Monte-Carlo, no training), equivariance
of the trained skills under a rigid motion of the scene (+ theta wrap), handoff
containment and success over the grid, solver cost.
"""
import math
import time
import numpy as np
import pandas as pd
import torch

import se2 as S
import terrain as TR
import task as TK
import solver as SV
from sde import Reference

H_SCALES = (0.05, 0.2, 0.5, 1.0)
ANISO = (1, 3, 10)
ROUTES = ("straight", "turn90", "scurve")
S_LONG = 0.02
POS_STD = 0.04


def route_means(kind):
    if kind == "straight":
        m = [(0.15, 0.5, 0.0), (0.40, 0.5, 0.0), (0.65, 0.5, 0.0), (0.90, 0.5, 0.0)]
    elif kind == "turn90":
        m = [(0.20, 0.20, 0.0), (0.50, 0.20, 0.0), (0.72, 0.40, math.pi / 2), (0.72, 0.75, math.pi / 2)]
    else:  # S-curve, 180 degrees of heading change over skills 2-3
        m = [(0.20, 0.30, 0.0), (0.55, 0.30, 0.0), (0.75, 0.50, math.pi / 2), (0.55, 0.70, math.pi)]
    return [torch.tensor(x) for x in m]


def marginals_1b(kind, h, wrap_shift=0.0):
    means = route_means(kind)
    if wrap_shift:                       # theta wrap test: put rho_2 at pi +- 0.1
        means[2] = torch.tensor([means[2][0], means[2][1], math.pi + wrap_shift])
        means[3] = torch.tensor([means[3][0], means[3][1], math.pi + wrap_shift])
    cov = torch.diag(torch.tensor([POS_STD, POS_STD, h]) ** 2)
    return means, [cov.clone() for _ in range(4)]


class UniformTask(TK.Task):
    """Terrain-free SE(2) task with constant body-frame noise and no obstacles."""

    def __init__(self, means, covs, body_std, n, seed, device):
        self.layout, self.dist, self.n, self.device = TR.Layout("none"), "uniform", n, device
        self.obs_fields = torch.zeros(5, TR.GRID, TR.GRID, device=device)
        self.means, self.covs = [m.to(device) for m in means], [c.to(device) for c in covs]
        self.body_std = body_std.to(device)
        self.push = None
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def dynamics(self, g, u, step):
        n = g.shape[0]
        u = TK.clip_u(u)
        xi = u * TK.DT + math.sqrt(TK.DT) * self.body_std * torch.randn(n, 3, generator=self.gen, device=self.device)
        g_new = S.compose(g, S.exp_se2(xi))
        hit = self.layout.collides(g[:, :2], g_new[:, :2])
        return torch.where(hit.unsqueeze(1), g, g_new), hit


def body_std(h, a):
    return torch.tensor([S_LONG, a * S_LONG, 0.5 * h])


# ------------------------------------------------ 1. covariance fidelity
def adj_inv(xi):
    """Ad(exp(-xi)) for small xi = (rho, phi): first-order left-invariant transport."""
    n = xi.shape[0]
    A = torch.eye(3).expand(n, 3, 3).clone()
    c, s = torch.cos(-xi[:, 2]), torch.sin(-xi[:, 2])
    A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, -s, s, c
    A[:, 0, 2], A[:, 1, 2] = -xi[:, 1], xi[:, 0]
    return A


@torch.no_grad()
def covariance_fidelity(kind, h, a, n_mc=10000, seed=0):
    """One skill (the curved one: skill 2) along its geodesic with body-frame noise.
    Analytic prediction under each model vs Monte-Carlo."""
    gen = torch.Generator().manual_seed(seed)
    means, covs = marginals_1b(kind, h)
    g0m, g1m = means[1], means[2]
    bs = body_std(h, a)
    L = torch.linalg.cholesky(covs[1])
    g = S.compose(g0m.expand(n_mc, 3), S.exp_se2(torch.randn(n_mc, 3, generator=gen) @ L.T))
    v = S.between(g0m[None], g1m[None])[0]
    T = TK.T_SKILL
    # SE(2) analytic: P_{t+1} = Ad(exp(-v dt)) P_t Ad^T + Sigma_body dt  (body frame)
    P = covs[1].clone(); Adv = adj_inv((v * TK.DT)[None])[0]; Sb = torch.diag(bs ** 2) * TK.DT
    # flat analytic: world-frame Gaussian, body noise rotated at the mean heading, linear in t
    theta_bar = 0.5 * (g0m[2] + S.wrap(g1m[2:3] - g0m[2:3]).item() * 0.5 + g0m[2])  # mean heading along the arc
    F = S.rot3(torch.tensor([theta_bar]))[0]
    Sf = F @ covs[1] @ F.T
    for _ in range(T):
        xi = (v * TK.DT).expand(n_mc, 3) + math.sqrt(TK.DT) * bs * torch.randn(n_mc, 3, generator=gen)
        g = S.compose(g, S.exp_se2(xi))
        P = Adv @ P @ Adv.T + Sb
        Sf = Sf + F @ (torch.diag(bs ** 2) * TK.DT) @ F.T
    gbar = S.compose(g0m[None], S.exp_se2(v[None]))
    # SE(2) coverage / KL in the tangent space at the predicted mean
    z = S.between(gbar.expand(n_mc, 3), g)
    md2 = torch.einsum("ni,ij,nj->n", z, torch.linalg.inv(P), z)
    cov_se2 = float((md2 <= 4).float().mean())
    kl_se2 = kl_gauss(z, P)
    # flat coverage / KL in world coordinates (theta wrapped) at the same mean
    d = torch.cat([g[:, :2] - gbar[:, :2], S.wrap(g[:, 2:3] - gbar[:, 2:3])], 1)
    md2f = torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(Sf), d)
    cov_flat = float((md2f <= 4).float().mean())
    kl_flat = kl_gauss(d, Sf)
    return dict(route=kind, h=h, aniso=a, nominal_2sigma=float(0.7385),
                cover_se2=cov_se2, cover_flat=cov_flat, kl_se2=kl_se2, kl_flat=kl_flat)


def kl_gauss(z, P):
    """KL( N(fit to samples z) || N(0, P) )."""
    m = z.mean(0); C = torch.cov(z.T) + 1e-9 * torch.eye(3)
    Pi = torch.linalg.inv(P)
    return float(0.5 * (torch.trace(Pi @ C) + m @ Pi @ m - 3 + torch.logdet(P) - torch.logdet(C)))


# ------------------------------------------------------- 2-4. trained grid
def train_cell(kind, h, a, mf, seed, device, steps0=1000, K=0, wrap_shift=0.0, log=lambda m: None):
    means, covs = marginals_1b(kind, h, wrap_shift)
    tk = UniformTask(means, covs, body_std(h, a), 64, 1000 + seed, device)
    ref = Reference("slip", kappa=2.0, body_cov=body_std(h, a) ** 2)
    t0 = time.time(); nets, diag = {}, []
    for k in range(3):
        nets[k], d = SV.train_skill(tk, k, ref, mf, seed * 17 + k, device, K=K, n_pair=6000, steps0=steps0,
                                    steps_ipf=500, batch=512, lr=1e-3, n_sim=50, log=log, with_bwd=K > 0)
        diag += d
    return nets, time.time() - t0, diag, tk


@torch.no_grad()
def equivariance(nets, mf, tk, device, it=0, n=300, seed=0):
    """Deterministic drift flow from n starts, in the original and a rigidly moved scene."""
    gen = torch.Generator(device=device).manual_seed(seed)
    a = S.exp_se2(torch.tensor([[0.25, -0.15, 0.9]], device=device)).expand(n, 3).contiguous()
    ctl = SV.BridgeController(nets, mf, tk, it, device, with_D=False)
    g = tk.sample(0, n); gr = S.compose(a, g)
    means_r = [S.compose(a, m[None])[0] for m in tk.means]
    tk_r = UniformTask(means_r, tk.covs, tk.body_std, n, 5, device)
    ctl_r = SV.BridgeController(nets, mf, tk_r, it, device, with_D=False)
    for k in range(3):
        for t in range(TK.T_SKILL):
            tau = torch.full((n,), t / TK.T_SKILL, device=device)
            u, _ = ctl(g, k, tau, 0); ur, _ = ctl_r(gr, k, tau, 0)
            g = S.compose(g, S.exp_se2(TK.clip_u(u) * TK.DT)); gr = S.compose(gr, S.exp_se2(TK.clip_u(ur) * TK.DT))
    return float(S.between(S.compose(a, g), gr).norm(dim=1).mean())


def eval_cell(nets, mf, tk_train, kind, h, a, seed, device, n=500, wrap_shift=0.0, it=0):
    means, covs = marginals_1b(kind, h, wrap_shift)
    tk = UniformTask(means, covs, body_std(h, a), n, 50_000 + seed, device)
    ctl = SV.BridgeController(nets, mf, tk, it, device, with_D=False)
    out = tk.rollout(ctl, n=n)
    m = tk.metrics(out)
    m["equiv_dev"] = equivariance(nets, mf, tk, device, it)
    return m


def job(args):
    kind, h, a, mf_name, seed, wrap_shift = args
    torch.set_num_threads(2)
    device = torch.device("cpu"); mf = S.MANIFOLDS[mf_name]
    torch.manual_seed(seed * 7 + int(h * 100) + a)
    nets, el, _, tk = train_cell(kind, h, a, mf, seed, device, wrap_shift=wrap_shift)
    m = eval_cell(nets, mf, tk, kind, h, a, seed, device, wrap_shift=wrap_shift)
    print(f"[1b s{seed} {kind} h={h} a={a} {mf_name}{' wrap' if wrap_shift else ''}] succ {m['success']:.2f} "
          f"hd1 {m['handoff1_md']:.2f} equiv {m['equiv_dev']:.1e} ({el:.0f}s)", flush=True)
    return dict(phase="1b", seed=seed, route=kind, h=h, aniso=a, manifold=mf_name, wrap_shift=wrap_shift,
                train_s=el, **m)


def solver_cost_job(args):
    kind, mf_name, seed = args
    torch.set_num_threads(2)
    device = torch.device("cpu"); mf = S.MANIFOLDS[mf_name]
    nets, el, diag, tk = train_cell(kind, 0.5, 3, mf, seed, device, steps0=1000, K=4)
    d = pd.DataFrame(diag)
    # convergence: first iteration whose pooled transport cost is within 2% of the final one
    c = d.groupby("iter").transport_cost.mean()
    conv = int(next((i for i in c.index if abs(c[i] - c.iloc[-1]) <= 0.02 * abs(c.iloc[-1])), c.index[-1]))
    return dict(phase="1b_solver", seed=seed, route=kind, manifold=mf_name, wall_s=el, iters_to_converge=conv,
                cost_iter0=float(c.iloc[0]), cost_final=float(c.iloc[-1]))


def run(seed, workers=8, quick=False, manifolds=("flat", "se2")):
    import multiprocessing as mp
    hs, an, rt = (H_SCALES[:2], ANISO[:2], ROUTES[:2]) if quick else (H_SCALES, ANISO, ROUTES)
    jobs = [(r, h, a, mf, seed, 0.0) for r in rt for h in hs for a in an for mf in manifolds]
    jobs += [("scurve", 0.2, 3, mf, seed, ws) for mf in manifolds for ws in (0.1, -0.1)]   # theta wrap
    with mp.get_context("spawn").Pool(workers) as p:
        rows = p.map(job, jobs, chunksize=1)
        cost = p.map(solver_cost_job, [(r, mf, seed) for r in rt for mf in manifolds], chunksize=1)
    return rows, cost


def run_covfid():
    rows = [covariance_fidelity(r, h, a) for r in ROUTES for h in H_SCALES for a in ANISO]
    return pd.DataFrame(rows)
