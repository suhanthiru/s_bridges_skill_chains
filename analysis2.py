"""Tables and figures for phases 1b-5 of the SE(2) suite.  `python analysis2.py --phase 1b|2|3|4|5`."""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from analysis import ci95, fmt, save, agg, RES, FIND

os.makedirs(FIND, exist_ok=True)


def shards(pattern):
    fs = sorted(glob.glob(os.path.join(RES, pattern)))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def paired(a, b):
    j = a.index.intersection(b.index); d = (a[j] - b[j]).values
    m, h = ci95(d); p = float(stats.ttest_rel(a[j], b[j]).pvalue) if len(j) > 1 else np.nan
    return m, h, p


# ------------------------------------------------------------------ 1b
def phase1b():
    df = shards("phase1b_seed*.parquet"); cf = pd.read_parquet(os.path.join(RES, "phase1b_covfid.parquet"))
    so = shards("phase1b_solver_seed*.parquet")
    main = df[df.wrap_shift == 0]
    # fig: success and handoff gap (se2 - flat) over the grid
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
    for j, route in enumerate(("straight", "turn90", "scurve")):
        for a_, mk in zip((1, 3, 10), ("o", "s", "^")):
            d = main[(main.route == route) & (main.aniso == a_)]
            for mf, ls, c in (("flat", "--", "C1"), ("se2", "-", "C3")):
                g = d[d.manifold == mf].groupby("h")
                axes[0, j].errorbar(g.success.mean().index, g.success.mean(), g.success.apply(lambda v: ci95(v)[1]),
                                    ls=ls, marker=mk, color=c, ms=4, capsize=2, label=f"{mf}, a={a_}")
                axes[1, j].errorbar(g.handoff1_md.median().index, g.handoff1_md.median(), ls=ls, marker=mk, color=c, ms=4, label=f"{mf}, a={a_}")
        axes[0, j].set_title(route); axes[1, j].set_xlabel("heading-noise scale h"); axes[0, j].set_xscale("log"); axes[1, j].set_xscale("log")
    axes[0, 0].set_ylabel("success"); axes[1, 0].set_ylabel("handoff-1 Mahalanobis (median)"); axes[0, 2].legend(fontsize=6, ncol=2)
    fig.suptitle("Phase 1b: flat (dashed) vs SE(2) (solid); marker = anisotropy", fontsize=10); save(fig, "phase1b_grid")
    # covariance fidelity figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, lab in zip(axes, ("cover", "kl"), ("fraction of MC inside predicted 2σ set", "KL(MC fit || predicted)")):
        for mf, c in (("flat", "C1"), ("se2", "C3")):
            for route, mk in zip(("straight", "turn90", "scurve"), ("o", "s", "^")):
                d = cf[cf.route == route]
                ax.scatter(d.aniso * (1 + 0.15 * d.h), d[f"{col}_{mf}"], color=c, marker=mk, s=25, label=f"{mf} {route}")
        ax.set_xscale("log"); ax.set_xlabel("anisotropy (x jittered by h)"); ax.set_ylabel(lab)
        if col == "cover":
            ax.axhline(0.7385, color="k", ls=":", lw=1, label="nominal 2σ (χ²₃)")
        else:
            ax.set_yscale("log")
    axes[1].legend(fontsize=6, ncol=2); fig.suptitle("Phase 1b: covariance fidelity, analytic vs 10k Monte-Carlo", fontsize=10)
    save(fig, "phase1b_covariance_fidelity")
    # tables
    t = ["| route | h | a | success flat | success SE(2) | Δ (paired) | p | handoff-1 md flat | SE(2) | equiv dev flat | SE(2) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    gaps = []
    for route in ("straight", "turn90", "scurve"):
        for h in sorted(main.h.unique()):
            for a_ in sorted(main.aniso.unique()):
                d = main[(main.route == route) & (main.h == h) & (main.aniso == a_)]
                f, s = d[d.manifold == "flat"].set_index("seed"), d[d.manifold == "se2"].set_index("seed")
                m, hw, p = paired(s.success, f.success); gaps.append(dict(route=route, h=h, aniso=a_, gap=m, p=p))
                t.append(f"| {route} | {h} | {a_} | {fmt(*ci95(f.success))} | {fmt(*ci95(s.success))} | {m:+.3f} | {p:.3f} | "
                         f"{f.handoff1_md.median():.2f} | {s.handoff1_md.median():.2f} | {f.equiv_dev.mean():.2e} | {s.equiv_dev.mean():.2e} |")
    w = df[df.wrap_shift != 0]
    wt = ["| manifold | rho_2 heading | success | handoff-2 md |", "|---|---|---|---|"]
    for mf in ("flat", "se2"):
        for ws in sorted(w.wrap_shift.unique()):
            d = w[(w.manifold == mf) & (w.wrap_shift == ws)]
            wt.append(f"| {mf} | π{ws:+.1f} | {fmt(*ci95(d.success))} | {d.handoff2_md.median():.2f} |")
    ct = ["| route | h | a | cover flat | cover SE(2) | KL flat | KL SE(2) |", "|---|---|---|---|---|---|---|"]
    for _, r in cf.iterrows():
        ct.append(f"| {r.route} | {r.h} | {r.aniso} | {r.cover_flat:.3f} | {r.cover_se2:.3f} | {r.kl_flat:.3f} | {r.kl_se2:.3f} |")
    st = ["| route | manifold | wall-clock s (K=4) | iters to converge (2%) | transport cost it0 → final |", "|---|---|---|---|---|"]
    for route in ("straight", "turn90", "scurve"):
        for mf in ("flat", "se2"):
            d = so[(so.route == route) & (so.manifold == mf)]
            if len(d):
                st.append(f"| {route} | {mf} | {fmt(*ci95(d.wall_s), 0)} | {d.iters_to_converge.mean():.1f} | {d.cost_iter0.mean():.4f} → {d.cost_final.mean():.4f} |")
    out = "## Grid\n\n" + "\n".join(t) + "\n\n## θ wrap\n\n" + "\n".join(wt) + "\n\n## Covariance fidelity\n\n" + "\n".join(ct) + "\n\n## Solver cost\n\n" + "\n".join(st) + "\n"
    open(os.path.join(FIND, "_phase1b_tables.md"), "w", encoding="utf-8").write(out); print(out)
    return pd.DataFrame(gaps)


# ------------------------------------------------------------------- 2
def phase2():
    df = shards("phase2_seed[0-9].parquet")
    main = df[~df.disturbance.str.startswith("slip_x")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    meth = ("nominal", "ppo", "diffusion", "bridge_iter0", "bridge_ipf"); x = np.arange(4); w = 0.16
    for ax, layout in zip(axes, ("L1", "L2")):
        for j, m in enumerate(meth):
            s = agg(main[(main.layout == layout) & (main.method == m)], ("disturbance",), "success").set_index("disturbance").reindex(["none", "slip", "rain", "push"])
            ax.bar(x + (j - 2) * w, s["mean"], w, yerr=s["ci"], capsize=2, label=m, color=f"C{j}")
        ax.set_xticks(x); ax.set_xticklabels(["none", "slip", "rain", "push"]); ax.set_title(layout); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success"); axes[1].legend(fontsize=7)
    fig.suptitle("Phase 2: terrain robustness (mean ± 95% CI over seeds)", fontsize=10); save(fig, "phase2_success")
    sw = df[df.disturbance.str.startswith("slip_x")]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for j, m in enumerate(meth):
        s = agg(sw[sw.method == m], ("slip_scale",), "success").sort_values("slip_scale")
        ax.errorbar(s.slip_scale / 0.7, s["mean"], s["ci"], marker="o", capsize=2, label=m, color=f"C{j}")
    ax.set_xscale("log"); ax.set_xlabel("slip magnitude (x nominal)"); ax.set_ylabel("success (L1, slip)"); ax.legend(fontsize=7)
    save(fig, "phase2_slip_sweep")
    t = ["| layout | method | none | slip | rain | push | slip×0.5 | ×1 | ×2 | ×4 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for layout in ("L1", "L2"):
        for m in meth:
            d = main[(main.layout == layout) & (main.method == m)]
            cells = [fmt(*ci95(d[d.disturbance == x_].success)) for x_ in ("none", "slip", "rain", "push")]
            if layout == "L1":
                ds = sw[sw.method == m]
                cells += [fmt(*ci95(ds[ds.disturbance == f"slip_x{sc}"].success)) for sc in (0.5, 1.0, 2.0, 4.0)]
            else:
                cells += ["-"] * 4
            t.append(f"| {layout} | {m} | " + " | ".join(cells) + " |")
    pt = ["| layout | disturbance | bridge_ipf − diffusion | 95% CI | p | bridge_iter0 − diffusion | 95% CI | p |", "|---|---|---|---|---|---|---|---|"]
    for layout in ("L1", "L2"):
        for x_ in ("none", "slip", "rain", "push"):
            d = main[(main.layout == layout) & (main.disturbance == x_)]
            dif = d[d.method == "diffusion"].set_index("seed").success
            row = f"| {layout} | {x_} |"
            for m in ("bridge_ipf", "bridge_iter0"):
                mm, hw, p = paired(d[d.method == m].set_index("seed").success, dif)
                row += f" {mm:+.3f} | [{mm - hw:+.3f}, {mm + hw:+.3f}] | {p:.3f} |"
            pt.append(row)
    out = "## Main grid\n\n" + "\n".join(t) + "\n\n## Bridge − diffusion (paired over seeds)\n\n" + "\n".join(pt) + "\n"
    open(os.path.join(FIND, "_phase2_tables.md"), "w", encoding="utf-8").write(out); print(out)


# ------------------------------------------------------------------- 3
def phase3():
    off = pd.read_parquet(os.path.join(RES, "phase3_offline.parquet")); cl = pd.read_parquet(os.path.join(RES, "phase3_closedloop.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, m in zip(axes, ("bridge_iter0", "bridge_ipf")):
        d = off[(off.method == m) & (off.disturbance != "none")]
        for j, sig in enumerate(("D", "D_norm", "dist_nominal", "unicycle_resid")):
            s = agg(d[d.signal == sig], ("horizon",), "auc").sort_values("horizon")
            ax.errorbar(s.horizon, s["mean"], s["ci"], marker="o", capsize=2, label=sig, color=f"C{j}")
        ax.axhline(0.5, color="k", ls=":", lw=1); ax.set_title(m); ax.set_xlabel("horizon (steps before failure)")
    axes[0].set_ylabel("AUC (failure prediction)"); axes[1].legend(fontsize=7)
    fig.suptitle("Phase 3: D(x,t) vs baselines as a failure predictor (disturbed conditions, pooled)", fontsize=10)
    save(fig, "phase3_auc")
    t = ["| method | disturbance | horizon | AUC D | AUC D·τ(1−τ) | AUC dist-to-nominal | AUC unicycle residual |", "|---|---|---|---|---|---|---|"]
    for m in ("bridge_iter0", "bridge_ipf"):
        for x_ in ("none", "slip", "rain", "push"):
            for H in (5, 10, 20):
                d = off[(off.method == m) & (off.disturbance == x_) & (off.horizon == H)]
                t.append(f"| {m} | {x_} | {H} | " + " | ".join(fmt(*ci95(d[d.signal == s].auc)) for s in ("D", "D_norm", "dist_nominal", "unicycle_resid")) + " |")
    ct = ["| method | layout | disturbance | success no trigger | with trigger | Δ (paired) | replan rate | false-replan rate |", "|---|---|---|---|---|---|---|---|"]
    for m in ("bridge_iter0", "bridge_ipf"):
        for layout in ("L1", "L2"):
            for x_ in ("none", "slip", "rain", "push"):
                d = cl[(cl.method == m) & (cl.layout == layout) & (cl.disturbance == x_)]
                if d.empty:
                    continue
                dl, hw = ci95((d.success_trigger - d.success_no_trigger).values)
                ct.append(f"| {m} | {layout} | {x_} | {fmt(*ci95(d.success_no_trigger))} | {fmt(*ci95(d.success_trigger))} | {fmt(dl, hw)} | {d.replan_rate.mean():.2f} | {d.false_replan_rate.mean():.2f} |")
    out = f"## Offline AUC\n\n" + "\n".join(t) + f"\n\n## Closed loop (threshold on raw D = {cl.threshold.iloc[0]:.3f})\n\n" + "\n".join(ct) + "\n"
    open(os.path.join(FIND, "_phase3_tables.md"), "w", encoding="utf-8").write(out); print(out)


# ------------------------------------------------------------------- 4
def phase4():
    df = shards("phase4_seed[0-9].parquet")
    a = df[df.phase == "4a"]; b = df[df.phase == "4b"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8)); rng = np.random.RandomState(0); am = []
    for j, lv in enumerate(sorted(a.slip_level.unique())):
        d = a[a.slip_level == lv]; g = d.groupby("w").success
        ax.errorbar(g.mean().index, g.mean(), g.apply(lambda v: ci95(v)[1]), marker="o", capsize=2, label=f"slip x{lv}", color=f"C{j}")
        piv = d.pivot_table(index="seed", columns="w", values="success")
        bs = [piv.iloc[rng.randint(len(piv), size=len(piv))].mean().idxmax() for _ in range(1000)]
        am.append(f"| x{lv} | {g.mean().idxmax()} [{np.percentile(bs, 2.5)}, {np.percentile(bs, 97.5)}] | {g.mean().max():.3f} |")
    ax.set_xscale("log"); ax.set_xlabel("rho_2 width scale w"); ax.set_ylabel("success (L2, noisy map)"); ax.legend(fontsize=7)
    save(fig, "phase4a_width")
    t4a = ["| w | " + " | ".join(f"slip x{lv}: success" for lv in sorted(a.slip_level.unique())) + " | handoff-2 md (nominal cov, x1) |", "|---|" + "---|" * (len(a.slip_level.unique()) + 1)]
    for w in sorted(a.w.unique()):
        d = a[a.w == w]
        t4a.append(f"| {w} | " + " | ".join(fmt(*ci95(d[d.slip_level == lv].success)) for lv in sorted(a.slip_level.unique())) + f" | {d[d.slip_level == 1.0].handoff2_md_nominal.median():.2f} |")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for kind, c in (("multi", "C3"), ("two_bridge", "C0")):
        d = b[b.kind == kind].groupby("map").agg(sd=("slip_diff", "mean"), pu=("p_upper", "mean"))
        ax.scatter(d.sd, d.pu, color=c, label=kind)
    ax.set_xlabel("expected slip: lower − upper route (observed map)"); ax.set_ylabel("P(upper gap chosen)"); ax.legend(fontsize=8)
    save(fig, "phase4b_gap_choice")
    m = b[b.kind == "multi"].groupby("map").agg(sd=("slip_diff", "mean"), pu=("p_upper", "mean"))
    r = stats.pearsonr(m.sd, m.pu) if len(m) > 2 else (np.nan, np.nan)
    t4b = ["| kind | success | collision | P(upper) | handoff-2 inside either mode | W2 goal |", "|---|---|---|---|---|---|"]
    for kind in ("multi", "two_bridge"):
        d = b[b.kind == kind]
        t4b.append(f"| {kind} | {fmt(*ci95(d.groupby('seed').success.mean()))} | {d.collision.mean():.3f} | {d.p_upper.mean():.2f} | {d.handoff2_inside.mean():.2f} | {d.w2_goal.mean():.3f} |")
    out = "## 4a width sweep\n\n" + "\n".join(t4a) + "\n\n| slip level | argmax_w [bootstrap 95% CI] | success |\n|---|---|---|\n" + "\n".join(am) + \
          f"\n\n## 4b bimodal\n\n" + "\n".join(t4b) + f"\n\nPearson r(P(upper), slip_lower − slip_upper) over {len(m)} maps, multi-marginal bridge: r = {r[0]:.3f}, p = {r[1]:.3f}\n"
    open(os.path.join(FIND, "_phase4_tables.md"), "w", encoding="utf-8").write(out); print(out)


# ------------------------------------------------------------------- 5
def phase5():
    df = shards("phase5_seed[0-9].parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True); x = np.arange(4); w = 0.2
    ds = ("a_demos", "b_demos_noised", "c_demos_bridge", "d_bridge_only")
    for ax, layout in zip(axes, ("L1", "L2")):
        for j, name in enumerate(ds):
            s = agg(df[(df.layout == layout) & (df.dataset == name)], ("disturbance",), "success").set_index("disturbance").reindex(["none", "slip", "rain", "push"])
            ax.bar(x + (j - 1.5) * w, s["mean"], w, yerr=s["ci"], capsize=2, label=name, color=f"C{j}")
        ax.set_xticks(x); ax.set_xticklabels(["none", "slip", "rain", "push"]); ax.set_title(layout)
    axes[0].set_ylabel("diffusion-policy success"); axes[1].legend(fontsize=7)
    fig.suptitle("Phase 5: training data for the diffusion policy", fontsize=10); save(fig, "phase5_augmentation")
    t = ["| layout | dataset | n | none | slip | rain | push |", "|---|---|---|---|---|---|---|"]
    for layout in ("L1", "L2"):
        for name in ds:
            d = df[(df.layout == layout) & (df.dataset == name)]
            t.append(f"| {layout} | {name} | {int(d.n_traj.iloc[0]) if len(d) else 0} | " + " | ".join(fmt(*ci95(d[d.disturbance == x_].success)) for x_ in ("none", "slip", "rain", "push")) + " |")
    pt = ["| layout | disturbance | c − b | 95% CI | p |", "|---|---|---|---|---|"]
    for layout in ("L1", "L2"):
        for x_ in ("slip", "rain", "push"):
            d = df[(df.layout == layout) & (df.disturbance == x_)]
            m, hw, p = paired(d[d.dataset == "c_demos_bridge"].set_index("seed").success, d[d.dataset == "b_demos_noised"].set_index("seed").success)
            pt.append(f"| {layout} | {x_} | {m:+.3f} | [{m - hw:+.3f}, {m + hw:+.3f}] | {p:.3f} |")
    out = "## Success by dataset\n\n" + "\n".join(t) + "\n\n## (c) − (b), paired over seeds\n\n" + "\n".join(pt) + "\n"
    open(os.path.join(FIND, "_phase5_tables.md"), "w", encoding="utf-8").write(out); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="1b"); a = ap.parse_args()
    {"1b": phase1b, "2": phase2, "3": phase3, "4": phase4, "5": phase5}[a.phase]()
