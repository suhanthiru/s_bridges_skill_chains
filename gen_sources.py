"""Data generators for the bridge-as-generator suite.

Every generator runs on the same *reference kinematics* the bridge lives in:
g <- g . exp(u dt + sqrt(dt) Sigma(x)^1/2 z), no friction, no base noise, with
Sigma(x) the slip-reference covariance (or an isotropic match) and z a shared
stream of standard-normal draws fixed by the seed.  Only the controller differs:

  BRIDGE-*     the learned drift of a bridge with the given reference
  PD-noise     kp=10 + feed-forward tracking of the nearest demo path
  PD-iso       the same tracker under isotropic noise of matched total variance
  DART         the demonstrator (Nominal, kp=6) executed with action noise
               eta = Sigma^1/2 z / sqrt(dt) and no displacement noise; clean u recorded
  MPC-rollout  a sampling MPC that knows the kinematics, rolled out under the noise
  MPC-relabel  BRIDGE-slip's states with every action replaced by the MPC's

Datasets are (G (n,301,3), U (n,300,3)) chained trajectories in Phase 2's format.
"""
import math
import numpy as np
import torch
from scipy.spatial import cKDTree

import se2 as S
import terrain as TR
import task as TK
import solver as SV
import bridge_fast as BF
import phase2 as P2
from sde import Reference

DT, T_SKILL, N_SKILL = TK.DT, TK.T_SKILL, TK.N_SKILL
KP_PD = 10.0


# ------------------------------------------------------------ environment
class GenTask(TK.Task):
    """Task with the suite's environment axes: slip anisotropy (lateral std x aniso),
    terrain correlation length (Voronoi seed count), push-rate multiplier, marginal
    heading std, and a class-flip rate for the map the generator is allowed to see."""

    def __init__(self, layout, disturbance, n, seed, device, slip_scale=0.7, aniso=1.0, n_seed=14,
                 push_mult=1.0, heading_std=None, map_flip=0.0, layout_id=0, route="lower", push_mag=0.25):
        self.layout, self.dist, self.n, self.device = layout, disturbance, n, device
        self.cmap = TR.class_map(layout_id, n_seed=n_seed)
        tf = TR.property_fields(self.cmap); rf = TR.property_fields(self.cmap, rain=True)
        for f in (tf, rf):
            f[1] *= aniso; f[4] *= push_mult
        self.true_fields, self.rain_fields = torch.tensor(tf, device=device), torch.tensor(rf, device=device)
        if map_flip > 0:
            of = TR.property_fields(TR.class_map(layout_id, n_seed=n_seed, flip_rate=map_flip)); of[1] *= aniso; of[4] *= push_mult
            self.obs_fields = torch.tensor(of, device=device)
        else:
            self.obs_fields = self.true_fields
        self.means, self.covs = TR.marginals(layout, route)
        self.means = [m.to(device) for m in self.means]; self.covs = [c.to(device).clone() for c in self.covs]
        if heading_std is not None:
            for c in self.covs:
                c[2, 2] = heading_std ** 2
        self.slip_scale = slip_scale
        self.push = TK.PushField(layout_id, push_mag, device) if disturbance == "push" else None
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.rain_t = int(torch.randint(60, 240, (1,), generator=self.gen, device=device)) if disturbance == "rain" else 10 ** 9


def make_ref(kind, tk, slip_scale=None):
    return Reference(kind, sigma=0.05, kappa=2.0, slip_scale=slip_scale if slip_scale is not None else tk.slip_scale,
                     fields=tk.obs_fields)


class NoiseStream:
    """Shared standard-normal draws z[t] for n episodes, fixed by the seed."""

    def __init__(self, n, seed, device, T=N_SKILL * T_SKILL):
        g = torch.Generator(device=device).manual_seed(90_000 + seed)
        self.z = torch.randn(n, T, 3, generator=g, device=device)

    def __call__(self, t):
        return self.z[:, t]


def noise_step(ref, mf, g, z, iso_var=None):
    """sqrt(dt) Sigma(g)^1/2 z in the manifold's tangent coordinates."""
    if iso_var is not None:
        return math.sqrt(DT) * math.sqrt(iso_var) * z
    mode = BF.mode_of(ref, mf)
    Sg = BF.cov_at(ref, g, mf, mode)
    if mode == "scalar":
        return math.sqrt(DT) * Sg.sqrt()[:, None] * z
    if mode == "diag":
        return math.sqrt(DT) * Sg.sqrt() * z
    from sde import psd_sqrt
    return math.sqrt(DT) * torch.einsum("nij,nj->ni", psd_sqrt(Sg), z)


def step_kin(mf, g, u, xi_noise):
    """Reference kinematics on the manifold: body command u (clipped) plus tangent noise."""
    u = TK.clip_u(u)
    if mf.name == "flat":
        F = mf.frame(g)
        xi = torch.einsum("nij,nj->ni", F, u) * DT + xi_noise          # world-frame tangent
        return mf.retract(g, xi), u
    return mf.retract(g, u * DT + xi_noise), u


# ------------------------------------------------------------ controllers
class PDTracker:
    """kp=10 + feed-forward on the nearest demo (by start pose) - TRACK's controller on SE(2)."""

    def __init__(self, demo_G, demo_U):
        self.G, self.U, self.idx = demo_G, demo_U, None

    def __call__(self, g, t):
        if t == 0 or self.idx is None:
            d = S.se2_dist2(g, self.G[:, 0]); self.idx = d.argmin(1)
        ref = self.G[self.idx, t + 1]
        return self.U[self.idx, t] + KP_PD * S.between(g, ref)


class MPC:
    """Sampling MPC (K candidates, horizon H) on the reference kinematics toward the current
    skill's next marginal mean; MPPI-weighted, warm-started.  Knows the kinematics exactly."""

    def __init__(self, tk, K=200, H=30, noise=0.5, lam=0.01, seed=0):
        self.tk, self.K, self.H, self.noise, self.lam = tk, K, H, noise, lam
        self.gen = torch.Generator(device=tk.device).manual_seed(seed)
        self.plan = None

    @torch.no_grad()
    def act(self, g, k):
        n, d = g.shape[0], g.device
        goal = self.tk.means[k + 1].expand(n, 3)
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 3, device=d)
        base = self.plan.clone(); base[:, -1] = TK.clip_u(2.0 * S.between(g, goal))
        U = base[:, None] + self.noise * torch.randn(n, self.K, self.H, 3, generator=self.gen, device=d) * torch.tensor([1, 1, 3.0], device=d)
        U[:, 0] = base
        U = TK.clip_u(U.reshape(-1, 3)).reshape(n, self.K, self.H, 3)
        gs = g[:, None].expand(n, self.K, 3).reshape(-1, 3); gg = goal[:, None].expand(n, self.K, 3).reshape(-1, 3)
        cost = torch.zeros(n * self.K, device=d)
        for h in range(self.H):
            u = U[:, :, h].reshape(-1, 3)
            gs = S.compose(gs, S.exp_se2(u * DT))
            xi = S.between(gs, gg); d2 = (xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2
            cost += 0.3 * d2 + 0.0005 * (u ** 2).sum(1) * DT
        xi = S.between(gs, gg); cost += 10.0 * ((xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2)
        cost = cost.reshape(n, self.K)
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (w[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]


# -------------------------------------------------------------- rollouts
@torch.no_grad()
def rollout(tk, mf, act_fn, n, z, ref, iso_var=None, action_noise_ref=None, states=None, world_heading=None):
    """Chain three skills on the reference kinematics.  act_fn(g, k, t) -> body command.
    action_noise_ref: DART mode (noise on the action, none on the displacement).
    states: relabel mode - impose the given states, only record act_fn's commands."""
    g = tk.sample(0, n) if states is None else states[:, 0].clone()
    G, U = [g.clone()], []
    for k in range(N_SKILL):
        for t in range(T_SKILL):
            s = k * T_SKILL + t
            u = act_fn(g, k, t)
            U.append(TK.clip_u(u))
            if states is not None:
                g = states[:, s + 1]; G.append(g.clone()); continue
            if action_noise_ref is not None:
                eta = noise_step(action_noise_ref, mf, g, z(s)) / DT
                g, _ = step_kin(mf, g, u + eta, torch.zeros_like(g))
            elif world_heading is not None:
                # "world-frame" noise model: body covariance rotated at the skill's mean heading,
                # applied in the world frame regardless of the robot's actual heading
                gbar = torch.tensor([0.0, 0.0, world_heading[k]], device=g.device).expand(g.shape[0], 3)
                xi_w = noise_step(ref, S.Flat, gbar, z(s))
                xi_b = torch.einsum("nij,nj->ni", S.rot3(g[:, 2]).transpose(1, 2), xi_w)
                g, _ = step_kin(mf, g, u, xi_b)
            else:
                g, _ = step_kin(mf, g, u, noise_step(ref, mf, g, z(s), iso_var))
            G.append(g.clone())
    return torch.stack(G, 1), torch.stack(U, 1)


def bridge_act(nets, mf, tk):
    ctl = SV.BridgeController(nets, mf, tk, 0, tk.device, with_D=False)
    return lambda g, k, t: ctl(g, k, torch.full((g.shape[0],), t / T_SKILL, device=g.device), 0)[0]


def pd_act(demo_G, demo_U):
    pd = PDTracker(demo_G, demo_U)
    return lambda g, k, t: pd(g, k * T_SKILL + t)


def nominal_act(tk):
    nom = P2.Nominal(tk)
    return lambda g, k, t: nom(g, k, torch.full((g.shape[0],), t / T_SKILL, device=g.device), 0)[0]


def mpc_act(tk, seed):
    m = MPC(tk, seed=seed)
    return lambda g, k, t: m.act(g, k)


def iso_variance(ref, mf, G):
    """Mean total variance tr(Sigma(x))/3 over the states of a reference rollout set."""
    mode = BF.mode_of(ref, mf)
    x = G.reshape(-1, 3)
    Sg = BF.cov_at(ref, x, mf, mode)
    tr = 3 * Sg if mode == "scalar" else (Sg.sum(1) if mode == "diag" else torch.diagonal(Sg, dim1=1, dim2=2).sum(1))
    return float(tr.mean() / 3)


# -------------------------------------------------------------- datasets
def noised_copies(G, U, m, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(G.shape[0], (m,), generator=g, device=device)
    noise = torch.randn(m, G.shape[1], 3, generator=g, device=device) * torch.tensor([0.03, 0.03, 0.1], device=device)
    Gn = torch.cat([G[idx, :, :2] + noise[:, :, :2], S.wrap(G[idx, :, 2:3] + noise[:, :, 2:3])], 2)
    return Gn, U[idx]


def coverage(G, demo_G, grid=32, nth=8, thr=0.05):
    """Off-path coverage: distinct (x, y, heading-bin) cells visited by states farther than
    `thr` (SE(2) tangent metric) from any demo state; and the mean displacement."""
    X = G.reshape(-1, 3).cpu().numpy(); Dm = demo_G.reshape(-1, 3).cpu().numpy()
    emb = lambda a: np.stack([a[:, 0], a[:, 1], S.HEADING_W * np.cos(a[:, 2]), S.HEADING_W * np.sin(a[:, 2])], 1)
    d, _ = cKDTree(emb(Dm)).query(emb(X))
    off = X[d > thr]
    cells = set(zip((off[:, 0] * grid).astype(int).clip(0, grid - 1), (off[:, 1] * grid).astype(int).clip(0, grid - 1),
                    (((off[:, 2] + math.pi) / (2 * math.pi)) * nth).astype(int).clip(0, nth - 1)))
    return dict(cov_cells=len(cells), off_frac=float((d > thr).mean()), mean_disp=float(d.mean()))
