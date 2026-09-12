"""Phase 5: bridge-generated demonstrations as augmentation for the diffusion policy.

Datasets (per layout, per seed), each of 500 or 1000 chained trajectories:
  a  demos                      500 nominal-PD demos (Phase 2 demo set)
  b  demos + noised copies      500 demos + 500 copies with Gaussian state noise
                                (pos std 0.03, heading std 0.1), actions kept
  c  demos + bridge samples     500 demos + 500 rollouts of the Phase-2 slip bridge
                                (iteration 0) under its reference SDE, actions =
                                the drift commands it issued
  d  bridge samples only        1000 bridge rollouts
The diffusion policy is Phase 2's; evaluation is Phase 2's disturbance grid.
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
import bridge_fast as BF
import phase2 as P2
from sde import Reference

RES, MODELS = "results", os.path.join("results", "models_se2")
DATASETS = ("a_demos", "b_demos_noised", "c_demos_bridge", "d_bridge_only")
DIFF_STEPS = 10000


@torch.no_grad()
def bridge_samples(nets, layout, n, seed, device):
    """Roll the iteration-0 slip bridge through the chain with its reference noise."""
    tk = TK.Task(TR.Layout(layout), "none", n, 7000 + seed, device, layout_id=0)
    mf = S.MANIFOLDS[P2.MF_NAME]
    ref = Reference("slip", kappa=2.0, fields=tk.obs_fields)
    mode = BF.mode_of(ref, mf)
    gen = torch.Generator(device=device).manual_seed(7000 + seed)
    ctl = SV.BridgeController(nets, mf, tk, 0, device, with_D=False)
    g = tk.sample(0, n); G, U = [g.clone()], []
    for k in range(3):
        for t in range(TK.T_SKILL):
            tau = torch.full((n,), t / TK.T_SKILL, device=device)
            u, _ = ctl(g, k, tau, 0); u = TK.clip_u(u)
            Sg = BF.cov_at(ref, g, mf, mode); z = torch.randn(n, 3, generator=gen, device=device)
            z = (Sg.sqrt() * z) if mode == "diag" else (Sg.sqrt()[:, None] * z if mode == "scalar" else z)
            xi = u * TK.DT + np.sqrt(TK.DT) * z
            g = mf.retract(g, xi) if mf.name == "se2" else S.compose(g, S.exp_se2(torch.einsum("nij,nj->ni", mf.frame(g).transpose(1, 2), xi)))
            G.append(g.clone()); U.append(u)
    return torch.stack(G, 1), torch.stack(U, 1)


def build(name, layout, seed, device):
    G, U = P2.load_demos(layout, device)
    n = G.shape[0]
    if name == "a_demos":
        return G, U
    if name == "b_demos_noised":
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(G.shape, generator=gen, device=device) * torch.tensor([0.03, 0.03, 0.1], device=device)
        Gn = torch.cat([G[:, :, :2] + noise[:, :, :2], S.wrap(G[:, :, 2:3] + noise[:, :, 2:3])], 2)
        return torch.cat([G, Gn]), torch.cat([U, U])
    nets = torch.load(P2.mpath("bridge", layout, seed), map_location=device)
    if name == "c_demos_bridge":
        Gb, Ub = bridge_samples(nets, layout, n, seed, device)
        return torch.cat([G, Gb]), torch.cat([U, Ub])
    Gb, Ub = bridge_samples(nets, layout, 2 * n, seed, device)
    return Gb, Ub


def run(seed, quick=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 300 if quick else DIFF_STEPS
    P2.DEMO_TAG = "_quick" if quick else ""
    rows = []
    for layout in P2.LAYOUTS:
        tk = TK.Task(TR.Layout(layout), "slip", 64, 1000 + seed, device, layout_id=0)
        for name in DATASETS:
            t0 = time.time()
            G, U = build(name, layout, seed, device)
            pol = P2.train_diffusion(tk, G, U, seed, device, steps, log=lambda m: None)
            torch.save(pol.state_dict(), os.path.join(MODELS, f"p5_{name}_{layout}_s{seed}.pt"))
            for dist in P2.DISTS:
                m, _ = P2.eval_cell("diffusion", {"diffusion": pol}, layout, dist, seed, device)
                rows.append(dict(phase=5, seed=seed, layout=layout, dataset=name, n_traj=int(G.shape[0]),
                                 disturbance=dist, **m))
            print(f"[p5 s{seed} {layout} {name} n={G.shape[0]}] " + " ".join(f"{d}:{r['success']:.2f}" for d, r in zip(P2.DISTS, rows[-4:]))
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    return rows
