"""Phase 2 (terrain robustness) + trajectory logging for Phase 3.

Methods: nominal (PD on the geodesic through the marginal means), ppo, diffusion,
bridge_iter0, bridge_ipf (K=5).  Bridges use the `slip` reference on the manifold
chosen by Phase 1b (MF below).  Grid: method x disturbance {none, slip, rain, push}
x layout {L1, L2}, plus the slip-magnitude sweep {0.5, 1, 2, 4} x nominal on L1.

Demo set (shared by diffusion, and by Phase 5): 500 rollouts per skill per layout
of the nominal PD controller under no disturbance, on the same terrain map.
PPO: goal-conditioned Gaussian policy on x ⊕ one-hot k ⊕ terrain probes, trained
under the `slip` disturbance, 1M steps (compute cap; stated in the findings).
"""
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import se2 as S
import terrain as TR
import task as TK
import solver as SV
import rl as R
from bridge import sinusoidal

RES, MODELS, DATA = "results", os.path.join("results", "models_se2"), "data_se2"
MF_NAME = os.environ.get("SB_MF", "se2")
LAYOUTS = ("L1", "L2")
DISTS = ("none", "slip", "rain", "push")
METHODS = ("nominal", "ppo", "diffusion", "bridge_iter0", "bridge_ipf")
SLIP_SWEEP = (0.5, 1.0, 2.0, 4.0)
N_EVAL = 500
K = 5
CFG = dict(ppo_steps=1_000_000, ppo_envs=512, diff_steps=15000, n_demo=500,
           bridge=dict(n_pair=8000, steps0=1500, steps_ipf=600, batch=512, lr=1e-3, n_sim=50))
QUICK = dict(ppo_steps=30_000, ppo_envs=128, diff_steps=300, n_demo=60,
             bridge=dict(n_pair=1500, steps0=120, steps_ipf=60, batch=512, lr=1e-3, n_sim=20))
CHUNK, EXEC = 8, 4


def obs_of(tk, g, k, tau):
    """x, y, cos, sin, one-hot k, tau, world-frame terrain probes -> (n, 24)."""
    if not torch.is_tensor(k):
        k = torch.full((g.shape[0],), int(k), dtype=torch.long, device=g.device)
    k1h = torch.nn.functional.one_hot(k.clamp(max=2), 3).float()
    return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), k1h, tau.unsqueeze(1),
                      TR.terrain_feats(tk.obs_fields, g, False)], 1)


OBS_DIM = 4 + 3 + 1 + TR.N_TFEAT


# --------------------------------------------------------------- nominal
class Nominal:
    def __init__(self, tk, kp=6.0):
        self.tk, self.kp = tk, kp

    def __call__(self, g, k, tau, step):
        a, b = self.tk.means[k].unsqueeze(0).expand_as(g), self.tk.means[k + 1].unsqueeze(0).expand_as(g)
        ref = S.SE2.interp(a, b, (tau + 1.0 / TK.T_SKILL).clamp(max=1.0))
        return self.kp * S.between(g, ref), None


# ----------------------------------------------------------------- demos
def demo_path(layout):
    return os.path.join(DATA, f"demos_{layout}.npz")


@torch.no_grad()
def make_demos(layout, device, n):
    tk = TK.Task(TR.Layout(layout), "none", n, 4242, device, layout_id=0)
    out = tk.rollout(Nominal(tk), n=n, record=True)
    G = out["traj"]                                    # (n, 301, 3)
    # recover the commands from consecutive poses (clip-consistent): u = log(g_t^-1 g_{t+1}) / dt
    U = S.between(G[:, :-1].reshape(-1, 3), G[:, 1:].reshape(-1, 3)).reshape(n, 300, 3) / TK.DT
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(demo_path(layout), G=G.cpu().numpy(), U=U.cpu().numpy(), success=out["success"].cpu().numpy())
    return float(out["success"].float().mean())


def load_demos(layout, device):
    z = np.load(demo_path(layout))
    return torch.tensor(z["G"], device=device), torch.tensor(z["U"], device=device)


# ------------------------------------------------------------------- PPO
class PPOEnv:
    """Per-step stepping with auto-reset for rl.ppo_train on the SE(2) task."""

    def __init__(self, tk, n):
        self.tk, self.n, self.device = tk, n, tk.device
        self.env = self                                # rl.ppo_train reads env.env.n

    def reset(self, mask=None):
        d = self.device
        if mask is None:
            self.g = torch.zeros(self.n, 3, device=d); self.k = torch.zeros(self.n, dtype=torch.long, device=d)
            self.t = torch.zeros(self.n, dtype=torch.long, device=d); self.alive = torch.ones(self.n, dtype=torch.bool, device=d)
            mask = torch.ones(self.n, dtype=torch.bool, device=d)
        m = int(mask.sum())
        if m:
            self.g[mask] = self.tk.sample(0, m); self.k[mask] = 0; self.t[mask] = 0; self.alive[mask] = True
        return obs_of(self.tk, self.g, self.k, self.t.float() / TK.T_SKILL)

    def step(self, u, eps):
        tau = self.t.float() / TK.T_SKILL
        g_new, hit = self.tk.dynamics(self.g, u, self.k * TK.T_SKILL + self.t)
        self.g = g_new; self.t += 1
        r = -0.01 * (u ** 2).sum(1) - 0.001 - hit.float()
        done = hit.clone(); success = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        ho = self.t >= TK.T_SKILL
        if ho.any():
            self.k = torch.where(ho, self.k + 1, self.k); self.t = torch.where(ho, torch.zeros_like(self.t), self.t)
            fin = ho & (self.k >= 3)
            success = fin & (S.mahalanobis(self.g, self.tk.means[3], self.tk.covs[3]) <= 2.0)
            r = r + success.float(); done |= fin
        obs = obs_of(self.tk, self.g, self.k, self.t.float() / TK.T_SKILL)
        return obs, r, done, torch.zeros_like(done), {"success": success}


class PPOPolicy(nn.Module):
    d_act = 3

    def __init__(self):
        super().__init__()
        self.mu = R.mlp(OBS_DIM, 3); self.log_std = nn.Parameter(torch.full((3,), math.log(0.5)))

    def dist(self, obs):
        mu = self.mu(obs); return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @staticmethod
    def to_env(a):
        return torch.tanh(a) * torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX], device=a.device), None


class PPOCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.v = R.mlp(OBS_DIM, 1)

    def forward(self, obs):
        return self.v(obs).squeeze(1)


class PPOCtl:
    def __init__(self, pol, tk):
        self.pol, self.tk = pol, tk

    def __call__(self, g, k, tau, step):
        return PPOPolicy.to_env(self.pol.mu(obs_of(self.tk, g, k, tau)))[0], None


# ------------------------------------------------------------- diffusion
class DiffPolicy(nn.Module):
    def __init__(self, n_train=50, hidden=256):
        super().__init__()
        self.n_train = n_train
        self.net = nn.Sequential(nn.Linear(CHUNK * 3 + OBS_DIM + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, CHUNK * 3))
        self.register_buffer("abar", torch.cumprod(1 - torch.linspace(1e-4, 0.02, n_train), 0))
        self.scale = torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX])

    def forward(self, a, obs, t):
        return self.net(torch.cat([a, obs, sinusoidal(t.float() / self.n_train, 16)], 1))

    @torch.no_grad()
    def sample(self, obs, n_ddim=10):
        n = obs.shape[0]; a = torch.randn(n, CHUNK * 3, device=obs.device)
        ts = torch.linspace(self.n_train - 1, 0, n_ddim + 1).long()[:-1]
        for i, t in enumerate(ts):
            ab = self.abar[t]; ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
            eps = self(a, obs, torch.full((n,), int(t), device=obs.device))
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        return a.reshape(n, CHUNK, 3) * self.scale.to(obs.device)


def train_diffusion(tk, G, U, seed, device, steps, log=print):
    torch.manual_seed(seed)
    pol = DiffPolicy().to(device); opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    n = G.shape[0]; sc = pol.scale.to(device)
    for s in range(steps):
        i = torch.randint(n, (1024,), device=device); t0 = torch.randint(300 - CHUNK + 1, (1024,), device=device)
        g = G[i, t0]; k = t0 // TK.T_SKILL; tau = (t0 % TK.T_SKILL).float() / TK.T_SKILL
        obs = obs_of(tk, g, k, tau)
        idx = t0[:, None] + torch.arange(CHUNK, device=device)[None]
        a0 = (U[i[:, None], idx] / sc).clamp(-1, 1).reshape(1024, -1)
        t = torch.randint(pol.n_train, (1024,), device=device); ab = pol.abar[t][:, None]
        noise = torch.randn_like(a0)
        loss = ((pol(ab.sqrt() * a0 + (1 - ab).sqrt() * noise, obs, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  diffusion loss {loss.item():.4f}")
    return pol


class DiffCtl:
    def __init__(self, pol, tk):
        self.pol, self.tk, self.buf, self.ptr = pol, tk, None, EXEC

    def __call__(self, g, k, tau, step):
        if self.buf is None or self.ptr >= EXEC or step % TK.T_SKILL == 0:
            self.buf, self.ptr = self.pol.sample(obs_of(self.tk, g, k, tau)), 0
        u = self.buf[:, self.ptr]; self.ptr += 1
        return u, None


# ---------------------------------------------------------------- driver
def mpath(name, layout, seed):
    return os.path.join(MODELS, f"{name}_{layout}_s{seed}.pt")


def get_models(layout, seed, device, cfg, log=print):
    os.makedirs(MODELS, exist_ok=True)
    mf = S.MANIFOLDS[MF_NAME]
    tk = TK.Task(TR.Layout(layout), "slip", cfg["ppo_envs"], 1000 + seed, device, layout_id=0)
    if not os.path.exists(demo_path(layout)):
        log(f"  demos {layout}: nominal success {make_demos(layout, device, cfg['n_demo']):.2f}")
    G, U = load_demos(layout, device)
    out = {}
    p = mpath("bridge", layout, seed)
    if os.path.exists(p):
        out["bridge"] = torch.load(p, map_location=device)
    else:
        torch.manual_seed(seed)
        tkc = TK.Task(TR.Layout(layout), "none", 64, 1000 + seed, torch.device("cpu"), layout_id=0)
        nets, _, _ = SV.train_all(tkc, "slip", mf, seed, torch.device("cpu"), K=K, log=lambda m: None, **cfg["bridge"])
        out["bridge"] = nets; torch.save(nets, p)
    p = mpath("ppo", layout, seed)
    if os.path.exists(p):
        pol = PPOPolicy().to(device); pol.load_state_dict(torch.load(p, map_location=device)); out["ppo"] = pol
    else:
        torch.manual_seed(seed); pol, crit = PPOPolicy().to(device), PPOCritic().to(device)
        t0 = time.time()
        R.ppo_train(PPOEnv(tk, cfg["ppo_envs"]), pol, crit, cfg["ppo_steps"], seed, device, log=lambda m: None)
        log(f"  ppo {layout} s{seed}: {time.time() - t0:.0f}s"); torch.save(pol.state_dict(), p); out["ppo"] = pol
    p = mpath("diff", layout, seed)
    if os.path.exists(p):
        pol = DiffPolicy().to(device); pol.load_state_dict(torch.load(p, map_location=device)); out["diffusion"] = pol
    else:
        pol = train_diffusion(tk, G, U, seed, device, cfg["diff_steps"], log); torch.save(pol.state_dict(), p); out["diffusion"] = pol
    for v in out.values():
        if isinstance(v, nn.Module):
            v.eval()
    return out


def controller(method, models, tk, device):
    if method == "nominal":
        return Nominal(tk)
    if method == "ppo":
        return PPOCtl(models["ppo"], tk)
    if method == "diffusion":
        return DiffCtl(models["diffusion"], tk)
    it = 0 if method == "bridge_iter0" else K
    return SV.BridgeController(models["bridge"], S.MANIFOLDS[MF_NAME], tk, it, device, with_D=True)


@torch.no_grad()
def eval_cell(method, models, layout, dist, seed, device, slip_scale=0.7, record=False):
    tk = TK.Task(TR.Layout(layout), dist, N_EVAL, 50_000 + seed, device, layout_id=0, slip_scale=slip_scale)
    out = tk.rollout(controller(method, models, tk, device), n=N_EVAL, record=record)
    m = tk.metrics(out)
    if "D" in out:
        m["D_mean"] = float(out["D"][:, :, 0].mean())
    return m, out


def run(seed, quick=False):
    cfg = QUICK if quick else CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for layout in LAYOUTS:
        t0 = time.time(); models = get_models(layout, seed, device, cfg)
        print(f"[p2 s{seed} {layout}] models ready {time.time() - t0:.0f}s", flush=True)
        for method in METHODS:
            for dist in DISTS:
                rec = method.startswith("bridge")
                m, out = eval_cell(method, models, layout, dist, seed, device, record=rec)
                rows.append(dict(phase=2, seed=seed, layout=layout, method=method, disturbance=dist, slip_scale=0.7, **m))
                if rec:
                    np.savez_compressed(os.path.join(RES, f"phase2_traj_{method}_{layout}_{dist}_s{seed}.npz"),
                                        traj=out["traj"].cpu().numpy(), D=out["D"].cpu().numpy(),
                                        success=out["success"].cpu().numpy(), alive=out["alive"].cpu().numpy())
            print(f"[p2 s{seed} {layout} {method}] " + " ".join(f"{d}:{r['success']:.2f}" for d, r in
                  zip(DISTS, rows[-4:])), flush=True)
        if layout == "L1":
            for sc in SLIP_SWEEP:
                for method in METHODS:
                    m, _ = eval_cell(method, models, layout, "slip", seed, device, slip_scale=0.7 * sc)
                    rows.append(dict(phase=2, seed=seed, layout=layout, method=method, disturbance=f"slip_x{sc}",
                                     slip_scale=0.7 * sc, **m))
    return rows
