"""Seam experiment: is the handoff marginal load-bearing?

Environment: the planar terrain env (terrain_env.py) exactly as in the terrain
run - mild/rough, pushes sigma_k, held-out layouts 20-29, the same MPC demos.
State is (x, y); "heading axis" below means the direction of travel (x) and
"corridor normal" means lateral (y); the tangent-space Mahalanobis is the R^2 one.

Every condition is a *reference generator* wrapped in the SAME PD tracker
(kp = 10 + feed-forward, TRACK's frozen config).  Nothing runs open-loop.

  TRACK-oracle        nearest MPC demo (terrain-aware path)
  TRACK-naive         straight line through the marginal MEANS ("handoff point")
  BRIDGE-tracked      bridge drift flow from the entry state, Brownian reference
  BRIDGE-slip-tracked bridge drift flow, unicycle (OU pull) + terrain-slip reference
  TSM                 behaviour-cloned skills fine-tuned so terminal states land in
                      the next skill's measured initiation set (terminal-state matching)

Bridges here are trained on samples of the marginals themselves (not on demo
endpoints as in the terrain run), because Phase B rescales the marginals and
no demos exist for the rescaled ones; Phase A uses the same recipe at w = 1.
"""
import json
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import env as E
import terrain_env as T
import demos as D
import conditions as C
from run_terrain import TRAIN_L, TEST_L, DATA, P_PUSH, fields_all

RES, MODELS, FIGS = "results_seam", os.path.join("results_seam", "models"), "figures"
SETTINGS = ("mild", "rough")
SIGMA_K = (0.0, 0.06, 0.10, 0.15)
SEEDS = (0, 1, 2, 3, 4)
N_EP = 200
KP, KD = 10.0, 0.0                       # TRACK's frozen PD
STD = T.WP_STD                            # 0.05, isotropic nominal marginals
KAPPA = 2.0                               # OU pull of the unicycle reference
GL_X, GL_W = np.polynomial.legendre.leggauss(8)
CONDS_A = ("TRACK-oracle", "TRACK-naive", "BRIDGE-tracked", "BRIDGE-slip-tracked", "TSM")
WIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0)
CFG = dict(bridge_steps=3000, bridge_batch=1024, bridge_lr=3e-4, n_pair=20000,
           bc_steps=3000, cls_steps=1500, ft_iters=150, ft_lambda=1.0, probe_roll=50, probe_grid=13)
QUICK = dict(CFG, bridge_steps=200, bc_steps=200, cls_steps=200, ft_iters=10, probe_roll=5, probe_grid=5, n_pair=2000)


def route(device):
    return T.ROUTE.to(device)


def marg_std(k, w):
    """Only rho_1 and rho_2 are rescaled by w."""
    return STD * (w if k in (1, 2) else 1.0)


def sample_marg(k, n, device, gen, w=1.0, mean=None, cov=None):
    m = route(device)[k] if mean is None else mean
    if cov is None:
        return m + marg_std(k, w) * torch.randn(n, 2, generator=gen, device=device)
    L = torch.linalg.cholesky(cov)
    return m + torch.randn(n, 2, generator=gen, device=device) @ L.T


def mahal(x, mean, cov):
    d = x - mean
    Si = torch.linalg.inv(cov)
    return torch.einsum("ni,ij,nj->n", d, Si, d).clamp_min(0).sqrt()


def md_iso(x, k, w=1.0):
    return (x - route(x.device)[k]).norm(dim=1) / marg_std(k, w)


# ------------------------------------------------------------ PD tracker
class PDRef:
    """TRACK's PD (kp=10, kd=0, + feed-forward) on a per-skill reference produced by
    gen(k, x_entry, lids) -> (X (n,101,2), U (n,100,2)), regenerated at every skill start."""

    def __init__(self, gen, device):
        self.gen, self.device, self.X = gen, device, None

    @torch.no_grad()
    def act(self, env, obs):
        n = env.n
        if self.X is None:
            self.X = torch.zeros(n, T.STEPS_PER_SKILL + 1, 2, device=self.device)
            self.U = torch.zeros(n, T.STEPS_PER_SKILL, 2, device=self.device)
            self.e_prev = torch.zeros(n, 2, device=self.device)
        u = torch.zeros(n, 2, device=self.device)
        for k in range(3):
            sel = env.k == k
            if not sel.any():
                continue
            start = sel & (env.t == 0)
            if start.any():
                X, U = self.gen(k, env.x[start], env.lids[start])
                self.X[start], self.U[start], self.e_prev[start] = X, U, 0.0
            t = env.t[sel].clamp(max=T.STEPS_PER_SKILL - 1)
            idx = sel.nonzero().flatten()
            e = self.X[idx, t + 1] - env.x[sel]
            u[sel] = self.U[idx, t] + KP * e + KD * (e - self.e_prev[sel]) / E.DT
            self.e_prev[sel] = e
        return u


def gen_naive(device):
    R = route(device)
    tt = torch.arange(T.STEPS_PER_SKILL + 1, device=device).float() / T.STEPS_PER_SKILL

    def gen(k, x, lids):
        n = x.shape[0]
        X = (R[k] + tt[:, None] * (R[k + 1] - R[k])).expand(n, -1, -1)
        U = (R[k + 1] - R[k]).expand(n, T.STEPS_PER_SKILL, 2)          # nominal velocity over 1 s
        return X, U
    return gen


def gen_oracle(rough, device):
    S = [torch.tensor(D.load_demos(DATA, TRAIN_L, rough, k)[0], device=device) for k in range(3)]
    A = [torch.tensor(D.load_demos(DATA, TRAIN_L, rough, k)[1], device=device) for k in range(3)]

    def gen(k, x, lids):
        idx = torch.cdist(x, S[k][:, 0]).argmin(1)
        return S[k][idx], A[k][idx]
    return gen


def gen_ode(step_fn, F, device):
    """Roll a state-space velocity field (bridge drift) or a policy through the mean
    dynamics from the entry state; that path is the PD reference, the commands the feed-forward."""

    def gen(k, x, lids):
        n = x.shape[0]
        X = torch.zeros(n, T.STEPS_PER_SKILL + 1, 2, device=device); U = torch.zeros(n, T.STEPS_PER_SKILL, 2, device=device)
        X[:, 0] = x
        for t in range(T.STEPS_PER_SKILL):
            tau = torch.full((n,), t / T.STEPS_PER_SKILL, device=device)
            u, dx = step_fn(k, X[:, t], tau, lids)
            U[:, t] = u; X[:, t + 1] = (X[:, t] + dx * E.DT).clamp(0, 1)
        return X, U
    return gen


def bridge_step(nets, use_obs, F):
    def f(k, x, tau, lids):
        u = nets[k](x, tau, T.rays(F, x, lids) if use_obs else None)
        return u, u                        # drift is a state-space velocity: dx = u
    return f


def policy_step(pols, F, rough):
    s, th, _ = T.ROUGH[rough]

    def f(k, x, tau, lids):
        u = E.clip_u(pols[k](x, tau, T.rays(F, x, lids)))
        v, _ = T.slip(F, x, lids, u, s, th)  # known mean dynamics
        return u, v
    return f


# --------------------------------------------------------- 2-D bridges
def train_bridge2d(F, layouts, k, ref, rough, w, seed, device, cfg, log=print):
    """Iteration-0 bridge matching in R^2 with an OU-deviation reference:
    d(xi) = -kappa xi dt + sigma(x) dW,  sigma^2(x) = sigma_p^2 * (0.5 | r(x)).
    brownian: kappa = 0, constant sigma^2 = sigma_p^2 / 2 (env noise at mean roughness).
    slip:     kappa = KAPPA, sigma^2(x) = sigma_p^2 r(x) frozen along the interpolant."""
    use_obs = ref == "slip"
    kappa = KAPPA if ref == "slip" else 0.0
    sp2 = T.ROUGH[rough][2] ** 2
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    net = C.TerrainDrift(use_obs).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["bridge_lr"])
    n = cfg["n_pair"]
    x0 = sample_marg(k, n, device, gen, w); x1 = sample_marg(k + 1, n, device, gen, w)
    lid = torch.tensor(layouts, device=device)[torch.randint(len(layouts), (n,), generator=gen, device=device)]
    vnom = x1 - x0
    X = torch.tensor(GL_X, device=device, dtype=torch.float32); W = torch.tensor(GL_W, device=device, dtype=torch.float32)

    def sig2(x, l):
        return sp2 * (0.5 * torch.ones(x.shape[0], device=device) if ref == "brownian" else T.roughness(F, x, l))

    def Q(a, b, x0b, vb, lb):
        half, mid = (b - a) / 2, (b + a) / 2
        nodes = mid[:, None] + half[:, None] * X
        wq = half[:, None] * W * torch.exp(-2 * kappa * (b[:, None] - nodes))
        if ref == "brownian":
            return wq.sum(1) * sp2 * 0.5
        m = x0b.shape[0]
        xbar = (x0b[:, None] + nodes[:, :, None] * vb[:, None]).reshape(-1, 2)
        s2 = sig2(xbar, lb.repeat_interleave(8)).reshape(m, 8)
        return (wq * s2).sum(1)

    Q10 = Q(torch.zeros(n, device=device), torch.ones(n, device=device), x0, vnom, lid)
    B = cfg["bridge_batch"]
    for s in range(cfg["bridge_steps"]):
        i = torch.randint(n, (B,), generator=gen, device=device)
        a, v, l, q10 = x0[i], vnom[i], lid[i], Q10[i]
        tau = torch.rand(B, generator=gen, device=device).clamp(1e-3, 1 - 1e-3)
        qt0 = Q(torch.zeros(B, device=device), tau, a, v, l)
        phi = torch.exp(-kappa * (1 - tau))
        P = (qt0 - qt0 ** 2 * phi ** 2 / q10).clamp_min(0)
        xi = P.sqrt()[:, None] * torch.randn(B, 2, generator=gen, device=device)
        g = a + tau[:, None] * v + xi
        rem = (1 - tau).clamp_min(0.02)
        q1t = Q(1 - rem, torch.ones(B, device=device), a, v, l)
        phi2 = torch.exp(-kappa * rem)
        K = sig2(g, l) * phi2 ** 2 / q1t.clamp_min(1e-12)
        target = v - kappa * xi - K[:, None] * xi
        pred = net(g, tau, T.rays(F, g, l) if use_obs else None)
        loss = ((pred - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  bridge {ref} {rough} w={w} k={k + 1} s={seed}: loss {loss.item():.4f}")
    return net


def bridge_path(ref, rough, w, seed):
    return os.path.join(MODELS, f"bridge_{ref}_{rough}_w{w}_s{seed}.pt")


def get_bridges(F, ref, rough, w, seed, device, cfg, log=print):
    p = bridge_path(ref, rough, w, seed)
    use_obs = ref == "slip"
    nets = [C.TerrainDrift(use_obs).to(device) for _ in range(3)]
    if os.path.exists(p):
        for n_, sd in zip(nets, torch.load(p, map_location=device)):
            n_.load_state_dict(sd)
    else:
        nets = [train_bridge2d(F, TRAIN_L, k, ref, rough, w, seed * 10 + k, device, cfg, log) for k in range(3)]
        torch.save([n_.state_dict() for n_ in nets], p)
    for n_ in nets:
        n_.eval()
    return nets


# ------------------------------------------------------------------ TSM
class InitCls(nn.Module):
    """Smooth initiation-set classifier c(x, rays) -> logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + T.N_RAYS, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x, o):
        return self.net(torch.cat([x, o], 1)).squeeze(1)


def train_tsm(F, rough, seed, device, cfg, log=print):
    """Terminal-state matching.
    1. BC_k on demo (state, action) pairs, MSE.
    2. Initiation set of skill k+1: probe grid of entry states (probe_grid^2 over +-4 sigma
       around rho_k, per training layout); BC_{k+1} rolled probe_roll times with process
       noise; label = success (end inside rho_{k+1} 2 sigma) >= 0.8.  A classifier c_{k+1}
       is fitted to the labels so the set is smooth and layout-aware.
    3. Skill k fine-tuned: loss = BC + lambda * softplus(-c_{k+1}(x_T)) with x_T from a
       differentiable rollout through the known mean slip dynamics (skill 3 uses the goal
       ellipse directly).  This is the density-free "reach the next initiation set" form."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    s_, th_, sp = T.ROUGH[rough]
    pols, data = [], []
    for k in range(3):
        S, A, G, Lid = D.load_demos(DATA, TRAIN_L, rough, k)
        S, A, Lid = torch.tensor(S, device=device), torch.tensor(A, device=device), torch.tensor(Lid, device=device)
        data.append((S, A, Lid))
        pol = C.TerrainDrift(True).to(device); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
        N = S.shape[0]
        for _ in range(cfg["bc_steps"]):
            i = torch.randint(N, (1024,), generator=gen, device=device)
            t = torch.randint(T.STEPS_PER_SKILL, (1024,), generator=gen, device=device)
            x, a, l = S[i, t], A[i, t], Lid[i]
            loss = ((pol(x, t.float() / T.STEPS_PER_SKILL, T.rays(F, x, l)) - a) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        pols.append(pol)
        log(f"  bc {rough} k={k + 1}: loss {loss.item():.4f}")

    # initiation sets of skills 2 and 3
    R = route(device); cls = {}
    G_ = cfg["probe_grid"]; ax = torch.linspace(-4 * STD, 4 * STD, G_, device=device)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    for k in (1, 2):
        pts = (R[k] + grid).repeat(len(TRAIN_L), 1)
        lids = torch.tensor(TRAIN_L, device=device).repeat_interleave(grid.shape[0])
        m = pts.shape[0]; rr = cfg["probe_roll"]
        envp = T.TerrainEnv(F, lids.repeat(rr), rough, 0.0, 0.0, 777 + k, device, disturb=False)
        envp.reset(); envp.x = pts.repeat(rr, 1); envp.k[:] = k; envp.t[:] = 0
        with torch.no_grad():
            for t in range(T.STEPS_PER_SKILL):
                tau = torch.full((m * rr,), t / T.STEPS_PER_SKILL, device=device)
                envp.step(pols[k](envp.x, tau, T.rays(F, envp.x, envp.lids)))
        ok = (md_iso(envp.x, k + 1) <= 2.0).float().reshape(rr, m).mean(0)
        lab = (ok >= 0.8).float()
        c = InitCls().to(device); opt = torch.optim.Adam(c.parameters(), lr=1e-3)
        o = T.rays(F, pts, lids)
        for _ in range(cfg["cls_steps"]):
            loss = nn.functional.binary_cross_entropy_with_logits(c(pts, o), lab)
            opt.zero_grad(); loss.backward(); opt.step()
        cls[k] = c
        log(f"  init set skill {k + 1}: {lab.mean().item():.2f} of probes inside, cls loss {loss.item():.3f}")

    # terminal fine-tuning of skills 1 and 2 (indices 0, 1); skill 3 toward the goal ellipse
    for k in range(3):
        pol, (S, A, Lid) = pols[k], data[k]
        opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
        for _ in range(cfg["ft_iters"]):
            i = torch.randint(S.shape[0], (256,), generator=gen, device=device)
            t = torch.randint(T.STEPS_PER_SKILL, (256,), generator=gen, device=device)
            x_bc, a_bc, l = S[i, t], A[i, t], Lid[i]
            bc = ((pol(x_bc, t.float() / T.STEPS_PER_SKILL, T.rays(F, x_bc, l)) - a_bc) ** 2).sum(1).mean()
            x = S[i, 0]
            for tt in range(T.STEPS_PER_SKILL):
                tau = torch.full((256,), tt / T.STEPS_PER_SKILL, device=device)
                u = E.clip_u(pol(x, tau, T.rays(F, x, l)))
                v, _ = T.slip(F, x, l, u, s_, th_)
                x = (x + v * E.DT).clamp(0, 1)
            if k < 2:
                term = nn.functional.softplus(-cls[k + 1](x, T.rays(F, x, l))).mean()
            else:
                term = torch.relu(md_iso(x, 3) - 2.0).mean()
            loss = bc + cfg["ft_lambda"] * term
            opt.zero_grad(); loss.backward(); opt.step()
        log(f"  tsm ft k={k + 1}: bc {bc.item():.4f} term {term.item():.4f}")
    return pols, cls


def tsm_path(rough, seed):
    return os.path.join(MODELS, f"tsm_{rough}_s{seed}.pt")


def get_tsm(F, rough, seed, device, cfg, log=print):
    p = tsm_path(rough, seed)
    if os.path.exists(p):
        z = torch.load(p, map_location=device)
        pols = [C.TerrainDrift(True).to(device) for _ in range(3)]
        for n_, sd in zip(pols, z["pols"]):
            n_.load_state_dict(sd)
        cls = {k: InitCls().to(device) for k in (1, 2)}
        for k in (1, 2):
            cls[k].load_state_dict(z["cls"][k])
    else:
        pols, cls = train_tsm(F, rough, seed, device, cfg, log)
        torch.save(dict(pols=[p_.state_dict() for p_ in pols], cls={k: cls[k].state_dict() for k in cls}), p)
    for n_ in pols + list(cls.values()):
        n_.eval()
    return pols, cls


# ------------------------------------------------------------ evaluation
@torch.no_grad()
def run_cell(gen, rough, sk, seed, F, device, w=1.0, n=N_EP, start_k=0, entry=None, cls=None):
    """One cell: n episodes on held-out layouts.  Returns per-episode rows."""
    lids = torch.tensor(TEST_L, device=device).repeat(n // len(TEST_L) + 1)[:n]
    envr = T.TerrainEnv(F, lids, rough, sk, P_PUSH, 20_000 + seed, device)
    obs = envr.reset()
    if entry is not None:
        envr.x = entry.clone(); envr.k[:] = start_k; envr.t[:] = 0
    ctl = PDRef(gen, device)
    succ = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range((3 - start_k) * T.STEPS_PER_SKILL):
        obs, r, term, _, info = envr.step(ctl.act(envr, obs)); succ |= info["success"]
    h1, h2, fin = envr.handoff[0], envr.handoff[1], envr.x
    R = route(device)
    rows = []
    for i in range(n):
        rows.append(dict(episode=i, layout=int(lids[i]), success=int(succ[i]), effort=float(info["effort"][i]),
                         md1=float(md_iso(h1[i:i + 1], 1, w)) if start_k == 0 else float("nan"),
                         md2=float(md_iso(h2[i:i + 1], 2, w)), md3=float(md_iso(fin[i:i + 1], 3)),
                         d1=float((h1[i] - R[1]).norm()) if start_k == 0 else float("nan"),
                         d2=float((h2[i] - R[2]).norm())))
    return rows


def make_gen(cond, F, rough, seed, device, cfg, w=1.0, log=print):
    if cond == "TRACK-naive":
        return gen_naive(device)
    if cond == "TRACK-oracle":
        return gen_oracle(rough, device)
    if cond == "BRIDGE-tracked":
        return gen_ode(bridge_step(get_bridges(F, "brownian", rough, w, seed, device, cfg, log), False, F), F, device)
    if cond == "BRIDGE-slip-tracked":
        return gen_ode(bridge_step(get_bridges(F, "slip", rough, w, seed, device, cfg, log), True, F), F, device)
    if cond == "TSM":
        pols, _ = get_tsm(F, rough, seed, device, cfg, log)
        return gen_ode(policy_step(pols, F, rough), F, device)
    raise ValueError(cond)


def summarise(ep, keys):
    ep = pd.DataFrame(ep)
    g = ep.groupby(keys)
    out = g.agg(success=("success", "mean"), effort=("effort", "mean"), md1=("md1", "median"), md2=("md2", "median"),
                md3=("md3", "median"), in1=("md1", lambda v: float((v <= 2).mean())),
                in2=("md2", lambda v: float((v <= 2).mean())), n=("success", "size")).reset_index()
    return out


# --------------------------------------------------------------- phases
def phase_A(seed, quick, device):
    cfg = QUICK if quick else CFG
    F = fields_all(device); ep = []
    for cond in CONDS_A:
        for rough in SETTINGS:
            t0 = time.time()
            gen = make_gen(cond, F, rough, seed, device, cfg)
            for sk in SIGMA_K:
                for r in run_cell(gen, rough, sk, seed, F, device):
                    ep.append(dict(phase="A", condition=cond, rough=rough, sigma_k=sk, seed=seed, w=1.0, **r))
            print(f"[A s{seed} {cond} {rough}] {time.time() - t0:.0f}s  succ@0.10 "
                  f"{np.mean([e['success'] for e in ep if e['condition'] == cond and e['rough'] == rough and e['sigma_k'] == 0.10]):.2f}", flush=True)
    return ep, summarise(ep, ["phase", "condition", "rough", "sigma_k", "seed", "w"])


def phase_B(seed, quick, device):
    cfg = QUICK if quick else CFG
    F = fields_all(device); ep = []
    ref = "slip"
    for w in WIDTHS:
        for rough in SETTINGS:
            t0 = time.time()
            gen = gen_ode(bridge_step(get_bridges(F, ref, rough, w, seed, device, cfg), True, F), F, device)
            for sk in (0.0, 0.06, 0.10):
                for r in run_cell(gen, rough, sk, seed, F, device, w=w):
                    ep.append(dict(phase="B", condition="BRIDGE-slip-tracked", rough=rough, sigma_k=sk, seed=seed, w=w, **r))
            print(f"[B s{seed} w={w} {rough}] {time.time() - t0:.0f}s", flush=True)
    return ep, summarise(ep, ["phase", "condition", "rough", "sigma_k", "seed", "w"])


def shift_specs():
    specs = [("none", 0.0, None, 1.0)]
    for ax in ("normal", "heading"):
        for d in (0.5, 1.0, 2.0, 3.0):
            specs.append((f"shift_{ax}", d, None, 1.0))
    specs += [("rot45", 0.0, 45.0, 1.0), ("rot90", 0.0, 90.0, 1.0), ("shrink", 0.0, None, 0.25), ("grow", 0.0, None, 4.0)]
    return specs


def shifted_entry(kind, delta, rot, scale, n, device, gen):
    """rho_1' for Phase D.  Nominal rho_1 is isotropic, so the rotations are exact no-ops;
    they are run anyway and reported as such."""
    R = route(device)
    mean = R[1].clone()
    if kind == "shift_normal":
        mean[1] += delta * STD
    elif kind == "shift_heading":
        mean[0] += delta * STD
    cov = (STD ** 2) * torch.eye(2, device=device) * scale
    if rot is not None:
        a = math.radians(rot); Rm = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], device=device)
        cov = Rm @ cov @ Rm.T
    return sample_marg(1, n, device, gen, mean=mean, cov=cov), mean, cov


def phase_D(seed, quick, device):
    cfg = QUICK if quick else CFG
    F = fields_all(device); ep = []
    rough, sk = "rough", 0.06
    gens = {c: make_gen(c, F, rough, seed, device, cfg) for c in ("BRIDGE-tracked", "BRIDGE-slip-tracked", "TSM")}
    _, cls = get_tsm(F, rough, seed, device, cfg)
    for kind, delta, rot, scale in shift_specs():
        g = torch.Generator(device=device).manual_seed(30_000 + seed)
        entry, mean, cov = shifted_entry(kind, delta, rot, scale, N_EP, device, g)
        lids = torch.tensor(TEST_L, device=device).repeat(N_EP // len(TEST_L) + 1)[:N_EP]
        with torch.no_grad():
            frac_init = float((cls[1](entry, T.rays(F, entry, lids)) > 0).float().mean())
        md_nom = float(md_iso(entry, 1).mean())
        for cond, gen in gens.items():
            for r in run_cell(gen, rough, sk, seed, F, device, start_k=1, entry=entry):
                ep.append(dict(phase="D", condition=cond, rough=rough, sigma_k=sk, seed=seed, w=1.0, shift=kind,
                               delta=delta, rot=rot if rot is not None else 0.0, scale=scale,
                               entry_md_nominal=md_nom, frac_in_tsm_init=frac_init, **r))
        print(f"[D s{seed} {kind} d={delta} rot={rot} sc={scale}] " +
              "  ".join(f"{c}: {np.mean([e['success'] for e in ep if e['condition'] == c and e['shift'] == kind and e['delta'] == delta and e['rot'] == (rot or 0.0) and e['scale'] == scale]):.2f}" for c in gens),
              flush=True)
    return ep, summarise(ep, ["phase", "condition", "rough", "sigma_k", "seed", "w", "shift", "delta", "rot", "scale",
                              "entry_md_nominal", "frac_in_tsm_init"])


def run_phase(phase, seed, quick=False):
    device = E.get_device(os.environ.get("SB_DEVICE") or None)
    os.makedirs(MODELS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    t0 = time.time()
    fn = {"A": phase_A, "B": phase_B, "D": phase_D}[phase]
    ep, summ = fn(seed, quick, device)
    tag = f"{phase}{'q' if quick else ''}"
    pd.DataFrame(ep).to_parquet(os.path.join(RES, f"phase{tag}_ep_seed{seed}.parquet"), index=False)
    summ.to_parquet(os.path.join(RES, f"phase{tag}_seed{seed}.parquet"), index=False)
    el = round(time.time() - t0, 1)
    tp = os.path.join(RES, "timing.json"); old = json.load(open(tp)) if os.path.exists(tp) else {}
    old[f"phase{tag}_seed{seed}"] = el; json.dump(old, open(tp, "w"), indent=1)
    print(f"=== seam phase {phase} seed {seed} took {el}s ===", flush=True)


def merge():
    import glob
    cells = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RES, "phase[ABD]_seed*.parquet")))]
    eps = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RES, "phase[ABD]_ep_seed*.parquet")))]
    pd.concat(cells, ignore_index=True).to_parquet("results_seam.parquet", index=False)
    pd.concat(eps, ignore_index=True).to_parquet("results_seam_episodes.parquet", index=False)
    print(f"merged {len(cells)} cell shards and {len(eps)} episode shards")
