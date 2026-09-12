"""Phase 4: handoff marginals and bimodality on L2 with the slip reference and a
noisy terrain map (class boundaries jittered by 0.02, 10% class flips; the
dynamics use the true map, the bridges see the corrupted one).

4a  width sweep: rho_2 covariance scaled by w in {0.5, 1, 2, 4}, skills 2 and 3
    retrained per w, evaluated at slip magnitude {0.5x, 1x, 2x}.
4b  bimodal rho_2 (one mode behind each gap).  Multi-marginal bridge: skill 2
    trained rho_1 -> mixture, skill 3 trained mixture -> rho_3.  Two-bridge
    alternative: a unimodal bridge per gap plus a runtime choice of the gap with
    the lower expected slip along the straight route (from the observed map).
    Gap assignment = nearer mode at the second handoff.  Ten terrain maps so the
    assignment can be correlated with the slip difference between the gaps.
"""
import os
import time
import numpy as np
import pandas as pd
import torch

import se2 as S
import terrain as TR
import task as TK
import solver as SV
from sde import Reference

RES, MODELS = "results", os.path.join("results", "models_se2")
MF_NAME = os.environ.get("SB_MF", "se2")
WIDTHS = (0.5, 1.0, 2.0, 4.0)
SLIP_LEVELS = (0.5, 1.0, 2.0)
MAPS = tuple(range(10))
N_EVAL = 500
BCFG = dict(n_pair=6000, steps0=1200, steps_ipf=500, batch=512, lr=1e-3, n_sim=50)


class MixTask(TK.Task):
    """L2 task whose rho_2 is a 50/50 mixture of the lower- and upper-gap marginals."""

    def __init__(self, n, seed, device, layout_id, map_noise=True, **kw):
        super().__init__(TR.Layout("L2"), "slip", n, seed, device, route="lower", map_noise=map_noise,
                         layout_id=layout_id, **kw)
        mu, cu = TR.marginals(self.layout, "upper")
        self.mean_up, self.cov_up = mu[2].to(device), cu[2].to(device)
        self.mean_lo, self.cov_lo = self.means[2], self.covs[2]

    def sample(self, k, n):
        if k != 2:
            return super().sample(k, n)
        pick = torch.rand(n, generator=self.gen, device=self.device) < 0.5
        lo = S.sample_marginal(self.mean_lo, self.cov_lo, n, self.device, self.gen)
        up = S.sample_marginal(self.mean_up, self.cov_up, n, self.device, self.gen)
        return torch.where(pick.unsqueeze(1), up, lo)


def train_skills(tk, ks, seed, device, log=lambda m: None):
    ref = Reference("slip", kappa=2.0, fields=tk.obs_fields)
    mf = S.MANIFOLDS[MF_NAME]
    out = {}
    for k in ks:
        out[k], _ = SV.train_skill(tk, k, ref, mf, seed * 17 + k, device, K=0, log=log, with_bwd=False, **BCFG)
    return out


def expected_slip(tk, mean_a, mean_b, n_pt=16):
    """Mean slip variance (sum of the three body stds^2) along the straight route on the OBSERVED map."""
    s = torch.linspace(0, 1, n_pt, device=tk.device)[:, None]
    pts = mean_a[None, :2] * (1 - s) + mean_b[None, :2] * s
    p = TR.lookup(tk.obs_fields, pts)
    return float((p[:, :3] ** 2).sum(1).mean())


# ------------------------------------------------------------------- 4a
def job_4a(args):
    w, seed = args
    torch.set_num_threads(2); device = torch.device("cpu"); torch.manual_seed(seed)
    tk = TK.Task(TR.Layout("L2"), "slip", 64, 1000 + seed, device, route="lower", map_noise=True, layout_id=0)
    tk.covs[2] = tk.covs[2] * w
    p = os.path.join(MODELS, f"p4a_w{w}_s{seed}.pt")
    if os.path.exists(p):
        nets = torch.load(p, map_location=device)
    else:
        nets = train_skills(tk, (0, 1, 2), seed, device); torch.save(nets, p)
    rows = []
    for lv in SLIP_LEVELS:
        te = TK.Task(TR.Layout("L2"), "slip", N_EVAL, 50_000 + seed, device, route="lower", map_noise=True,
                     layout_id=0, slip_scale=0.7 * lv)
        te.covs[2] = te.covs[2] * w
        ctl = SV.BridgeController(nets, S.MANIFOLDS[MF_NAME], te, 0, device, with_D=False)
        out = te.rollout(ctl, n=N_EVAL); m = te.metrics(out)
        m["handoff2_md_nominal"] = float(S.mahalanobis(out["handoff"][1], te.means[2], te.covs[2] / w).median())
        rows.append(dict(phase="4a", seed=seed, w=w, slip_level=lv, **m))
    print(f"[4a s{seed} w={w}] " + " ".join(f"x{lv}:{r['success']:.2f}" for lv, r in zip(SLIP_LEVELS, rows)), flush=True)
    return rows


# ------------------------------------------------------------------- 4b
@torch.no_grad()
def eval_4b(kind, nets, tk, seed, device, layout_id):
    n = N_EVAL
    te = MixTask(n, 50_000 + seed, device, layout_id)
    mf = S.MANIFOLDS[MF_NAME]
    if kind == "multi":
        ctl = SV.BridgeController(nets, mf, te, 0, device, with_D=False)
        ctl_fn = ctl
    else:
        lo, up = nets["lower"], nets["upper"]
        e_lo = expected_slip(te, te.means[1], te.mean_lo); e_up = expected_slip(te, te.means[1], te.mean_up)
        chosen = up if e_up < e_lo else lo
        te_r = TK.Task(TR.Layout("L2"), "slip", n, 50_000 + seed, device, route="upper" if e_up < e_lo else "lower",
                       map_noise=True, layout_id=layout_id)
        te_r.gen = te.gen
        ctl_fn = SV.BridgeController(chosen, mf, te_r, 0, device, with_D=False)
    out = te.rollout(ctl_fn, n=n)
    h2 = out["handoff"][1]
    d_lo = S.mahalanobis(h2, te.mean_lo, te.cov_lo); d_up = S.mahalanobis(h2, te.mean_up, te.cov_up)
    upper = (d_up < d_lo).float()
    inside = torch.minimum(d_lo, d_up) <= 2.0
    return dict(success=float(out["success"].float().mean()), collision=float(1 - out["alive"].float().mean()),
                p_upper=float(upper.mean()), handoff2_inside=float(inside.float().mean()),
                w2_goal=te.metrics(out)["w2_goal"], energy=float(out["energy"].mean()))


def job_4b(args):
    layout_id, seed = args
    torch.set_num_threads(2); device = torch.device("cpu"); torch.manual_seed(seed * 31 + layout_id)
    tk = MixTask(64, 1000 + seed, device, layout_id)
    p = os.path.join(MODELS, f"p4b_map{layout_id}_s{seed}.pt")
    if os.path.exists(p):
        z = torch.load(p, map_location=device); multi, two = z["multi"], z["two"]
    else:
        multi = train_skills(tk, (0, 1, 2), seed, device)
        two = {}
        for route in ("lower", "upper"):
            tr = TK.Task(TR.Layout("L2"), "slip", 64, 1000 + seed, device, route=route, map_noise=True, layout_id=layout_id)
            nets = train_skills(tr, (1, 2), seed, device); nets[0] = multi[0]
            two[route] = nets
        torch.save(dict(multi=multi, two=two), p)
    e_lo = expected_slip(tk, tk.means[1], tk.mean_lo); e_up = expected_slip(tk, tk.means[1], tk.mean_up)
    rows = []
    for kind, nets in (("multi", multi), ("two_bridge", two)):
        m = eval_4b(kind, nets, tk, seed, device, layout_id)
        rows.append(dict(phase="4b", seed=seed, map=layout_id, kind=kind, slip_lower=e_lo, slip_upper=e_up,
                         slip_diff=e_lo - e_up, **m))
    print(f"[4b s{seed} map{layout_id}] multi succ {rows[0]['success']:.2f} p_up {rows[0]['p_upper']:.2f} | "
          f"two succ {rows[1]['success']:.2f} p_up {rows[1]['p_upper']:.2f} | slip lo-up {e_lo - e_up:+.4f}", flush=True)
    return rows


def run(seed, workers=8, quick=False):
    import multiprocessing as mp
    os.makedirs(MODELS, exist_ok=True)
    ws, maps = (WIDTHS[:2], MAPS[:2]) if quick else (WIDTHS, MAPS)
    with mp.get_context("spawn").Pool(workers) as p:
        a = p.map(job_4a, [(w, seed) for w in ws], chunksize=1)
        b = p.map(job_4b, [(m, seed) for m in maps], chunksize=1)
    rows = [r for o in a + b for r in o]
    return rows
