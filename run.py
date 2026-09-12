"""Experiment driver.

    python run.py --phase tests            # the gate
    python run.py --phase 1 --seed 0       # Phase 1 grid for one seed
    python run.py --merge                  # collect per-seed shards into results.parquet

Every phase appends rows with a column for each grid axis, so `results.parquet`
is a single long table.  Phase 1 trains 4 references x 2 manifolds per seed and
checkpoints IPF iteration 0 and K from the same run, then evaluates each under
every disturbance mode.

This machine's GPU has ~0.16 ms kernel-launch overhead, and the SE(2)/bridge
machinery is hundreds of small ops, so the workload is launch-bound and CPU is
faster (measured: 9 ms vs 26 ms per training step).  Runs therefore go on CPU,
parallel across configurations.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import numpy as np
import pandas as pd
import torch

import se2 as S
import terrain as TR
import task as TK
import solver as SV

RES = "results"
REFERENCES = ("brownian", "killed", "unicycle", "slip")
MANIFOLDS = ("flat", "se2")
PHASE1_DIST = ("none", "slip", "rain")
K_IPF = 5

CFG = dict(K=K_IPF, n_pair=8000, steps0=1500, steps_ipf=600, batch=512, lr=1e-3, n_sim=50, sigma=0.05)
QUICK = dict(CFG, K=1, n_pair=1500, steps0=120, steps_ipf=60, n_sim=20)
N_EVAL = 500


class NominalController:
    """PD tracking of the geodesic between consecutive marginal means.
    Context line for Phase 1; the full baseline set belongs to Phase 2."""

    def __init__(self, tk, kp=6.0):
        self.tk, self.kp = tk, kp

    def __call__(self, g, k, tau, step):
        a, b = self.tk.means[k].unsqueeze(0).expand_as(g), self.tk.means[k + 1].unsqueeze(0).expand_as(g)
        ref = S.SE2.interp(a, b, (tau + 1.0 / TK.T_SKILL).clamp(max=1.0))
        return self.kp * S.between(g, ref), None


def eval_controller(ctl, layout, dist, seed, n, device, **kw):
    tk = TK.Task(layout, dist, n, 50_000 + seed, device, **kw)
    if hasattr(ctl, "tk"):
        ctl.tk = tk
    out = tk.rollout(ctl, n=n)
    m = tk.metrics(out)
    if "D" in out:
        D = out["D"]
        m["D_mean"], m["D_p90"] = float(D[:, :, 0].mean()), float(D[:, :, 0].quantile(0.9))
    return m, out


def phase1_job(args):
    seed, ref_kind, mf_name, quick = args
    torch.set_num_threads(2)
    cfg = QUICK if quick else CFG
    device = torch.device("cpu")
    mf = S.MANIFOLDS[mf_name]
    layout = TR.Layout("L1")
    TK_train = TK.Task(layout, "none", 64, 1000 + seed, device, layout_id=0)
    torch.manual_seed(seed * 101 + hash(ref_kind + mf_name) % 1000)
    t0 = time.time()
    nets, train_s, ipf_diag = SV.train_all(TK_train, ref_kind, mf, seed, device,
                                 sigma=cfg["sigma"], K=cfg["K"], log=lambda m: None,
                                 n_pair=cfg["n_pair"], steps0=cfg["steps0"], steps_ipf=cfg["steps_ipf"],
                                 batch=cfg["batch"], lr=cfg["lr"], n_sim=cfg["n_sim"])
    rows = []
    for it in (0, cfg["K"]):
        for dist in PHASE1_DIST:
            tk = TK.Task(layout, dist, N_EVAL, 50_000 + seed, device, layout_id=0)
            ctl = SV.BridgeController(nets, mf, tk, it, device)
            out = tk.rollout(ctl, n=N_EVAL)
            m = tk.metrics(out)
            D = out["D"]
            m.update(D_mean=float(D[:, :, 0].mean()), D_p90=float(D[:, :, 0].quantile(0.9)))
            rows.append(dict(phase=1, seed=seed, reference=ref_kind, manifold=mf_name, ipf=it,
                             condition=f"bridge_iter{it}", disturbance=dist, layout="L1",
                             train_s=train_s, **m))
    ipf_rows = [dict(phase=1, seed=seed, reference=ref_kind, manifold=mf_name, **d) for d in ipf_diag]
    print(f"[p1 s{seed} {ref_kind}/{mf_name}] {time.time() - t0:.0f}s  "
          f"it0/slip {rows[1]['success']:.2f} -> it{cfg['K']}/slip {rows[4]['success']:.2f}", flush=True)
    return rows, ipf_rows


def phase1(seed, quick, workers):
    jobs = [(seed, r, m, quick) for r in REFERENCES for m in MANIFOLDS]
    if workers <= 1:
        out = [phase1_job(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as p:
            out = p.map(phase1_job, jobs, chunksize=1)
    rows = [r for o in out for r in o[0]]
    ipf_rows = [r for o in out for r in o[1]]
    # nominal-tracking context line (not a bridge; no reference/manifold)
    device = torch.device("cpu")
    for dist in PHASE1_DIST:
        tk = TK.Task(TR.Layout("L1"), dist, N_EVAL, 50_000 + seed, device, layout_id=0)
        out_n = tk.rollout(NominalController(tk), n=N_EVAL)
        rows.append(dict(phase=1, seed=seed, reference="-", manifold="-", ipf=-1, condition="nominal",
                         disturbance=dist, layout="L1", train_s=0.0, **tk.metrics(out_n)))
    return rows, ipf_rows


def write(rows, phase, seed):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, f"phase{phase}_seed{seed}.parquet")
    pd.DataFrame(rows).to_parquet(p, index=False)
    print(f"wrote {len(rows)} rows -> {p}")


def merge():
    import glob
    fs = sorted(glob.glob(os.path.join(RES, "phase*_seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df.to_parquet("results.parquet", index=False)
    print(f"merged {len(fs)} shards, {len(df)} rows -> results.parquet")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.merge:
        merge(); return
    if a.phase == "tests":
        os.system(f"{os.sys.executable} tests.py"); return
    t0 = time.time()
    if a.phase == "1":
        rows, ipf_rows = phase1(a.seed, a.quick, a.workers)
        write(rows, 1, a.seed)
        pd.DataFrame(ipf_rows).to_parquet(os.path.join(RES, f"phase1_ipf_seed{a.seed}.parquet"), index=False)
    else:
        raise SystemExit(f"phase {a.phase} not implemented yet")
    el = round(time.time() - t0, 1)
    tp = os.path.join(RES, "timing.json")
    old = json.load(open(tp)) if os.path.exists(tp) else {}
    old[f"phase{a.phase}_seed{a.seed}"] = el
    json.dump(old, open(tp, "w"), indent=1)
    print(f"=== phase {a.phase} seed {a.seed} took {el}s ===")


if __name__ == "__main__":
    main()
