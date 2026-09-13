"""Phase 3: forward-backward drift disagreement D(x,t) as a failure predictor and
as a replan trigger.

Offline part (from the Phase 2 bridge trajectories): for horizon H, positives are
the steps of failing episodes at least H steps before the failure time (collision
step, or the end of the episode for an arrival miss); negatives are all steps of
successful episodes.  AUC of D is compared with two cheap baselines computed from
the same trajectories: SE(2) distance to the nominal geodesic at the same tau, and
the one-step residual of the unicycle model (constant body twist along the
geodesic).

Closed-loop part: raw D grows like 1/(tau(1-tau)) towards both ends of a skill, so
a single threshold on it fires late in every skill regardless of state; the
trigger therefore uses the tau-normalised D~ = D tau(1-tau) (the deviation from
the bridge's support itself).  Its threshold is chosen on validation seeds (0, 1)
as the 90th percentile of the per-episode maximum of D~ (over tau in [0.1, 0.9])
among SUCCESSFUL episodes, i.e. a 10% false-replan budget by construction; on test
seeds (2-4) a replan fires when D~ exceeds it during tau in [0.1, 0.9] (at most
once per skill).  Replan = re-solve
the current skill's bridge from the current state as the new initial marginal;
for an iteration-0 bridge with independent coupling that bridge is exactly the
learned drift restarted at tau = 0 from the current pose, so the skill clock is
reset and the skill re-runs from where the robot is (the episode grows by up to
100 steps per replan).  False-replan rate = replans fired in episodes that
succeeded without the trigger under the same RNG seed.
"""
import glob
import os
import numpy as np
import pandas as pd
import torch

import se2 as S
import terrain as TR
import task as TK
import solver as SV
import phase2 as P2

RES = "results"
HORIZONS = (5, 10, 20)
VAL_SEEDS, TEST_SEEDS = (0, 1), (2, 3, 4)


def auc(score, label):
    """Rank-based AUC (Mann-Whitney)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    if y.all() or (~y).all():
        return np.nan
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    n1, n0 = float(y.sum()), float((~y).sum())      # floats: n1*n0 overflows a 32-bit int on Windows
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def baseline_signals(traj, layout, device):
    """Distance to the nominal geodesic and the unicycle one-step residual, per step."""
    means, _ = TR.marginals(TR.Layout(layout))
    means = [m.to(device) for m in means]
    G = torch.tensor(traj, device=device)                      # (n, 301, 3)
    n = G.shape[0]
    dist, resid = torch.zeros(n, 300, device=device), torch.zeros(n, 300, device=device)
    for k in range(3):
        a, b = means[k].expand(n, 3), means[k + 1].expand(n, 3)
        vnom = S.between(a, b)
        for t in range(TK.T_SKILL):
            s = k * TK.T_SKILL + t
            tau = torch.full((n,), t / TK.T_SKILL, device=device)
            ref = S.SE2.interp(a, b, tau)
            xi = S.between(ref, G[:, s])
            dist[:, s] = (xi[:, :2] ** 2).sum(1).add((S.HEADING_W * xi[:, 2]) ** 2).sqrt()
            step = S.between(G[:, s], G[:, s + 1])
            r = step - vnom * TK.DT
            resid[:, s] = (r[:, :2] ** 2).sum(1).add((S.HEADING_W * r[:, 2]) ** 2).sqrt()
    return dist.cpu().numpy(), resid.cpu().numpy()


def offline(device):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "phase2_traj_*.npz"))):
        name = os.path.basename(f)[len("phase2_traj_"):-4]
        method, layout, dist_name, seed = name.rsplit("_", 3)
        seed = int(seed[1:])
        z = np.load(f)
        D, succ, alive, traj = z["D"], z["success"].astype(bool), z["alive"].astype(bool), z["traj"]
        n = D.shape[0]
        # failure time: last alive step for collisions (pose stops changing), else 300
        moved = np.abs(np.diff(traj[:, :, :2], axis=1)).sum(2) > 0
        t_fail = np.where(alive, 300, np.argmin(moved, axis=1))
        t_fail = np.where(~alive & (moved.all(1)), 300, t_fail)
        dist, resid = baseline_signals(traj, layout, device)
        sig = {"D": D[:, :, 0], "D_norm": D[:, :, 1], "dist_nominal": dist, "unicycle_resid": resid}
        for H in HORIZONS:
            tt = np.arange(300)[None]
            pos = (~succ)[:, None] & (tt <= (t_fail[:, None] - H))
            neg = succ[:, None] & np.ones((1, 300), bool)
            m = pos | neg
            for k, v in sig.items():
                rows.append(dict(phase=3, method=method, layout=layout, disturbance=dist_name, seed=seed, horizon=H,
                                 signal=k, auc=auc(v[m], pos[m]), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                                 fail_rate=float((~succ).mean())))
    return pd.DataFrame(rows)


def choose_threshold(df_val_trajs, budget=0.10):
    """Episode-level rule on validation seeds: the (1-budget) quantile of the per-episode
    maximum of normalised D~ within the trigger window among successful episodes, so the
    false-replan rate on validation data is `budget` by construction.  Also returns the
    fraction of failing validation episodes that would have fired (the trigger's recall)."""
    mx_ok, mx_fail = [], []
    tau = (np.arange(300) % 100) / 100.0
    win = (tau >= 0.1) & (tau <= 0.9)
    for f in df_val_trajs:
        z = np.load(f); D, succ = z["D"][:, :, 1], z["success"].astype(bool)
        mx = D[:, win].max(1)
        mx_ok.append(mx[succ]); mx_fail.append(mx[~succ])
    ok, fail = np.concatenate(mx_ok), np.concatenate(mx_fail)
    thr = float(np.quantile(ok, 1 - budget)) if len(ok) else float("inf")
    recall = float((fail > thr).mean()) if len(fail) else float("nan")
    return thr, recall


@torch.no_grad()
def rollout_trigger(tk, ctl, thr, n, max_replan=1):
    """Task.rollout with the replan trigger; also returns per-episode replan counts."""
    d = tk.device
    g = tk.sample(0, n)
    alive = torch.ones(n, dtype=torch.bool, device=d)
    k = torch.zeros(n, dtype=torch.long, device=d); t = torch.zeros(n, dtype=torch.long, device=d)
    replans = torch.zeros(n, dtype=torch.long, device=d); rp_skill = torch.zeros(n, dtype=torch.long, device=d)
    done = torch.zeros(n, dtype=torch.bool, device=d); success = torch.zeros(n, dtype=torch.bool, device=d)
    energy = torch.zeros(n, device=d); step = 0
    while not done.all() and step < 3 * TK.T_SKILL + max_replan * 3 * TK.T_SKILL:
        tau = t.float() / TK.T_SKILL
        u = torch.zeros(n, 3, device=d); Dv = torch.zeros(n, device=d)
        for kk in range(3):
            sel = (k == kk) & ~done
            if sel.any():
                uu, ex = ctl(g[sel], kk, tau[sel], 0)
                u[sel] = uu; Dv[sel] = ex[:, 1]          # normalised D~
        # trigger: D above threshold in the middle of a skill, at most max_replan per skill
        fire = (~done) & (Dv > thr) & (tau >= 0.1) & (tau <= 0.9) & (rp_skill < max_replan)
        t = torch.where(fire, torch.zeros_like(t), t); replans += fire.long(); rp_skill += fire.long()
        tau = t.float() / TK.T_SKILL
        g_new, hit = tk.dynamics(g, u, step)
        g = torch.where((alive & ~done).unsqueeze(1), g_new, g)
        energy += (u ** 2).sum(1) * TK.DT * (~done).float()
        alive &= ~hit; done |= hit
        t = t + 1
        ho = (t >= TK.T_SKILL) & ~done
        if ho.any():
            fin = ho & (k == 2)
            success |= fin & (S.mahalanobis(g, tk.means[3], tk.covs[3]) <= 2.0)
            done |= fin
            k = torch.where(ho & ~fin, k + 1, k); t = torch.where(ho, torch.zeros_like(t), t)
            rp_skill = torch.where(ho, torch.zeros_like(rp_skill), rp_skill)
        step += 1
    return dict(success=success, alive=alive, replans=replans, energy=energy)


def closed_loop(thr, device, quick=False):
    rows = []
    seeds = TEST_SEEDS if not quick else (96,)
    for layout in P2.LAYOUTS:
        for seed in seeds:
            p = P2.mpath("bridge", layout, seed)
            if not os.path.exists(p):
                continue
            nets = torch.load(p, map_location=device)
            for it_name, it in (("bridge_iter0", 0), ("bridge_ipf", P2.K)):
                for dist in P2.DISTS:
                    res = {}
                    for use in (False, True):
                        tk = TK.Task(TR.Layout(layout), dist, P2.N_EVAL, 50_000 + seed, device, layout_id=0)
                        ctl = SV.BridgeController(nets, S.MANIFOLDS[P2.MF_NAME], tk, it, device, with_D=True)
                        res[use] = rollout_trigger(tk, ctl, thr if use else float("inf"), P2.N_EVAL)
                    s0, s1 = res[False]["success"], res[True]["success"]
                    fired = res[True]["replans"] > 0
                    rows.append(dict(phase=3, layout=layout, seed=seed, method=it_name, disturbance=dist, threshold=thr,
                                     success_no_trigger=float(s0.float().mean()), success_trigger=float(s1.float().mean()),
                                     replan_rate=float(fired.float().mean()),
                                     false_replan_rate=float((fired & s0).float().mean() / max(float(s0.float().mean()), 1e-9)),
                                     mean_replans=float(res[True]["replans"].float().mean())))
                    print(f"[p3 {layout} s{seed} {it_name} {dist}] {rows[-1]['success_no_trigger']:.2f} -> "
                          f"{rows[-1]['success_trigger']:.2f}  replan {rows[-1]['replan_rate']:.2f} false {rows[-1]['false_replan_rate']:.2f}", flush=True)
    return pd.DataFrame(rows)


def run(quick=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    off = offline(device)
    off.to_parquet(os.path.join(RES, "phase3_offline.parquet"), index=False)
    allf = glob.glob(os.path.join(RES, "phase2_traj_*.npz"))
    val = [f for f in allf if int(f[:-4].rsplit("_s", 1)[1]) in VAL_SEEDS] or allf
    thr, rec = choose_threshold(val)
    print(f"[p3] threshold on normalised D~ (10% false-replan budget on validation): {thr:.4f}; "
          f"validation recall on failing episodes {rec:.2f}", flush=True)
    cl = closed_loop(thr, device, quick)
    cl.to_parquet(os.path.join(RES, "phase3_closedloop.parquet"), index=False)
    return off, cl
