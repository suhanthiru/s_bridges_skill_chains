"""Tables and figures for the generator suite.  `python gen_plots.py --phase g0|g1|...`."""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from analysis import ci95, fmt, save

RES, FIND = "results_generator", "findings"
DISTS = ("none", "slip", "rain", "push")
COL = {"DEMO": "0.4", "NOISED": "0.7", "BRIDGE-slip": "C3", "BRIDGE-brownian": "C1", "BRIDGE-unicycle": "C6",
       "PD-noise": "C0", "PD-iso": "C9", "DART": "C2", "MPC-rollout": "k", "MPC-relabel": "C4"}


def load(phase):
    fs = sorted(glob.glob(os.path.join(RES, f"phase{phase}_seed[0-9].parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def paired(a, b):
    j = a.index.intersection(b.index); d = (a[j] - b[j]).values; m, h = ci95(d)
    p = float(stats.ttest_rel(a[j], b[j]).pvalue) if len(j) > 1 else np.nan
    return m, h, p


def cell_table(df, sources, layouts=("L1", "L2"), dists=DISTS, col="success"):
    t = ["| source | layout | " + " | ".join(dists) + " |", "|---|---|" + "---|" * len(dists)]
    for s in sources:
        for L in layouts:
            d = df[(df.source == s) & (df.layout == L)]
            if d.empty:
                continue
            t.append(f"| {s} | {L} | " + " | ".join(fmt(*ci95(d[d.disturbance == x][col].values)) for x in dists) + " |")
    return "\n".join(t)


def pair_table(df, a, b, layouts=("L1", "L2"), dists=("slip", "rain", "push"), extra=None):
    t = [f"| layout | disturbance | {a} − {b} | 95% CI | p |", "|---|---|---|---|---|"]
    for L in layouts:
        for x in dists:
            d = df[(df.layout == L) & (df.disturbance == x)]
            if extra is not None:
                d = d[extra(d)]
            A, B = d[d.source == a].set_index("seed").success, d[d.source == b].set_index("seed").success
            if A.empty or B.empty:
                continue
            m, h, p = paired(A, B)
            t.append(f"| {L} | {x} | {m:+.3f} | [{m - h:+.3f}, {m + h:+.3f}] | {p:.3f} |")
    return "\n".join(t)


def bars(df, sources, name, title, layouts=("L1", "L2"), dists=DISTS):
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.5 * len(layouts), 3.8), sharey=True, squeeze=False)
    x = np.arange(len(dists)); w = 0.8 / len(sources)
    for ax, L in zip(axes[0], layouts):
        for j, s in enumerate(sources):
            d = df[(df.source == s) & (df.layout == L)]
            m = [ci95(d[d.disturbance == k].success.values) for k in dists]
            ax.bar(x + (j - len(sources) / 2 + 0.5) * w, [v[0] for v in m], w, yerr=[v[1] for v in m], capsize=2, label=s, color=COL.get(s, f"C{j}"))
        ax.set_xticks(x); ax.set_xticklabels(dists); ax.set_title(L); ax.set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("success"); axes[0, -1].legend(fontsize=6)
    fig.suptitle(title, fontsize=10); save(fig, name)


def g0():
    df = load("g0"); srcs = ["DEMO", "NOISED", "BRIDGE-slip", "PD-noise", "DART", "MPC-rollout"]
    bars(df, srcs, "gen_0_success", "g0 kill test: diffusion policy success by training-data source (mean ± 95% CI over seeds)")
    out = "## Success\n\n" + cell_table(df, srcs) + "\n\n## BRIDGE-slip − PD-noise (paired)\n\n" + pair_table(df, "BRIDGE-slip", "PD-noise") + \
          "\n\n## BRIDGE-slip − DART (paired)\n\n" + pair_table(df, "BRIDGE-slip", "DART") + "\n\n## MPC-rollout − BRIDGE-slip (paired)\n\n" + pair_table(df, "MPC-rollout", "BRIDGE-slip") + "\n"
    open(os.path.join(FIND, "_gen0_tables.md"), "w", encoding="utf-8").write(out); print(out)


def g1():
    df = load("g1"); srcs = ["BRIDGE-slip", "BRIDGE-brownian", "BRIDGE-unicycle", "PD-noise", "PD-iso", "MPC-relabel"]
    bars(df, srcs, "gen_1_isolation", "g1: what in the bridge does the work (L1)", layouts=("L1",))
    cov = df[df.disturbance == "slip"].groupby("source").agg(cov_cells=("cov_cells", "mean"), off_frac=("off_frac", "mean"),
                                                             mean_disp=("mean_disp", "mean"), success=("success", "mean")).reindex(srcs)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for s, r in cov.iterrows():
        ax.scatter(r.cov_cells, r.success, color=COL.get(s, "C5"), s=40, label=s)
    ax.set_xlabel("off-path coverage (distinct cells)"); ax.set_ylabel("success under slip"); ax.legend(fontsize=6)
    r_ = stats.pearsonr(cov.cov_cells, cov.success) if len(cov) > 2 else (np.nan, np.nan)
    ax.set_title(f"coverage vs success, r = {r_[0]:.2f}"); save(fig, "gen_1_coverage")
    ct = ["| source | off-path cells | off-path fraction | mean displacement from demos | success (slip) |", "|---|---|---|---|---|"]
    for s, r in cov.iterrows():
        ct.append(f"| {s} | {r.cov_cells:.0f} | {r.off_frac:.3f} | {r.mean_disp:.4f} | {r.success:.3f} |")
    out = "## Success (L1)\n\n" + cell_table(df, srcs, layouts=("L1",)) + "\n\n## Isolations (paired, L1)\n\n" + \
          pair_table(df, "BRIDGE-slip", "BRIDGE-unicycle", layouts=("L1",)) + "\n\n" + pair_table(df, "PD-noise", "PD-iso", layouts=("L1",)) + "\n\n" + \
          pair_table(df, "MPC-relabel", "BRIDGE-slip", layouts=("L1",)) + "\n\n" + pair_table(df, "BRIDGE-slip", "BRIDGE-brownian", layouts=("L1",)) + \
          f"\n\n## Coverage\n\n" + "\n".join(ct) + f"\n\nPearson r(coverage cells, success under slip) over sources: {r_[0]:.3f} (p = {r_[1]:.3f})\n"
    open(os.path.join(FIND, "_gen1_tables.md"), "w", encoding="utf-8").write(out); print(out)


def g2():
    df = load("g2"); out = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, ph, xcol, lab in zip(axes, ("g2a", "g2b", "g2c"), ("gen_mult", "n_demo", "slip_scale_eval"),
                                 ("generated size (x N_demo)", "N_demo", "eval slip magnitude (x nominal)")):
        d = df[(df.phase == ph) & (df.disturbance == "slip")]
        t = [f"| source | " + " | ".join(str(v) for v in sorted(d[xcol].unique())) + " |", "|---|" + "---|" * d[xcol].nunique()]
        for s in sorted(d.source.unique()):
            g = d[d.source == s].groupby(xcol).success
            xs = g.mean().index / (0.7 if xcol == "slip_scale_eval" else 1)
            ax.errorbar(xs, g.mean(), g.apply(lambda v: ci95(v)[1]), marker="o", capsize=2, label=s, color=COL.get(s, "C5"))
            t.append(f"| {s} | " + " | ".join(fmt(*ci95(d[(d.source == s) & (d[xcol] == v)].success.values)) for v in sorted(d[xcol].unique())) + " |")
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_title(ph); ax.legend(fontsize=7)
        out.append(f"## {ph} (success under slip, L1)\n\n" + "\n".join(t))
    axes[0].set_ylabel("success under slip"); save(fig, "gen_2_scale")
    open(os.path.join(FIND, "_gen2_tables.md"), "w", encoding="utf-8").write("\n\n".join(out) + "\n"); print("\n\n".join(out))


def g3():
    df = load("g3"); d = df[df.disturbance == "slip"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True); kinds = ("diffusion", "flow", "bc", "bc_gmm"); x = np.arange(4)
    t = ["| layout | policy | BRIDGE-slip | PD-noise | Δ | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    for ax, L in zip(axes, ("L1", "L2")):
        for j, s in enumerate(("BRIDGE-slip", "PD-noise")):
            m = [ci95(d[(d.layout == L) & (d.source == s) & (d.policy == k)].success.values) for k in kinds]
            ax.bar(x + (j - 0.5) * 0.4, [v[0] for v in m], 0.4, yerr=[v[1] for v in m], capsize=2, label=s, color=COL[s])
        ax.set_xticks(x); ax.set_xticklabels(kinds); ax.set_title(f"{L}, slip")
        for k in kinds:
            dd = d[(d.layout == L) & (d.policy == k)]
            A, B = dd[dd.source == "BRIDGE-slip"].set_index("seed").success, dd[dd.source == "PD-noise"].set_index("seed").success
            if A.empty:
                continue
            m_, h, p = paired(A, B)
            t.append(f"| {L} | {k} | {fmt(*ci95(A.values))} | {fmt(*ci95(B.values))} | {m_:+.3f} | [{m_ - h:+.3f}, {m_ + h:+.3f}] | {p:.3f} |")
    axes[0].set_ylabel("success under slip"); axes[1].legend(fontsize=7); save(fig, "gen_3_policy_class")
    out = "## Success under slip by policy class\n\n" + "\n".join(t) + "\n"
    open(os.path.join(FIND, "_gen3_tables.md"), "w", encoding="utf-8").write(out); print(out)


def g4():
    df = load("g4"); t = ["| h | a | BRIDGE se2 | BRIDGE flat | Δ (se2−flat) | p | PD tangent | PD world | Δ (tangent−world) | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h in sorted(df.heading_std.unique()):
        for a in sorted(df.aniso.unique()):
            d = df[(df.heading_std == h) & (df.aniso == a)]
            g = lambda s: d[d.source == s].set_index("seed").success
            m1, h1, p1 = paired(g("BRIDGE-slip-se2"), g("BRIDGE-slip-flat")); m2, h2, p2 = paired(g("PD-noise-tangent"), g("PD-noise-world"))
            t.append(f"| {h} | {a} | {fmt(*ci95(g('BRIDGE-slip-se2').values))} | {fmt(*ci95(g('BRIDGE-slip-flat').values))} | {m1:+.3f} | {p1:.3f} | "
                     f"{fmt(*ci95(g('PD-noise-tangent').values))} | {fmt(*ci95(g('PD-noise-world').values))} | {m2:+.3f} | {p2:.3f} |")
    fig, ax = plt.subplots(figsize=(7, 3.8)); cells = [(h, a) for h in sorted(df.heading_std.unique()) for a in sorted(df.aniso.unique())]
    x = np.arange(len(cells)); w = 0.2
    for j, s in enumerate(("BRIDGE-slip-se2", "BRIDGE-slip-flat", "PD-noise-tangent", "PD-noise-world")):
        m = [ci95(df[(df.source == s) & (df.heading_std == h) & (df.aniso == a)].success.values) for h, a in cells]
        ax.bar(x + (j - 1.5) * w, [v[0] for v in m], w, yerr=[v[1] for v in m], capsize=2, label=s, color=f"C{j}")
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}\na={a}" for h, a in cells], fontsize=8); ax.set_ylabel("success (S-curve, body noise)"); ax.legend(fontsize=7)
    save(fig, "gen_4_manifold")
    out = "## S-curve route: success by generator manifold\n\n" + "\n".join(t) + "\n"
    open(os.path.join(FIND, "_gen4_tables.md"), "w", encoding="utf-8").write(out); print(out)


def g5():
    lg = pd.read_parquet(os.path.join(RES, "phaseg5_search.parquet"))
    srch = lg[~lg.search.str.startswith("validate")]
    t = ["| search | eval | f | " + " | ".join(("slip_mag", "aniso", "corr_len", "push_rate", "heading", "n_demo", "demo_noise")) + " |", "|---|---|---|" + "---|" * 7]
    for _, r in pd.concat([srch.sort_values("f").tail(5), srch.sort_values("f").head(5)]).iterrows():
        t.append(f"| {r.search} | {int(r['eval'])} | {r.f:+.3f} | {r.slip_mag:.2f} | {r.aniso:.2f} | {r.corr_len:.2f} | {r.push_rate:.3f} | {r.heading:.2f} | {int(r.n_demo)} | {r.demo_noise:.2f} |")
    v = lg[lg.search.str.startswith("validate")]
    vt = ["| validation | f (10 seeds, 200 ep) | slip_mag | aniso | corr_len | push_rate | heading | n_demo | demo_noise |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in v.iterrows():
        vt.append(f"| {r.search} | {r.f:+.3f} | {r.slip_mag:.2f} | {r.aniso:.2f} | {r.corr_len:.2f} | {r.push_rate:.3f} | {r.heading:.2f} | {int(r.n_demo)} | {r.demo_noise:.2f} |")
    fac = lg[lg.search == "factorial"]; sens = {}
    axes_ = ("slip_mag", "aniso", "corr_len", "push_rate", "heading", "n_demo", "demo_noise")
    for a in axes_:
        sens[a] = float(fac.groupby(pd.qcut(fac[a], 3, duplicates="drop")).f.mean().max() - fac.groupby(pd.qcut(fac[a], 3, duplicates="drop")).f.mean().min()) if fac[a].nunique() > 1 else 0.0
    top2 = sorted(sens, key=sens.get, reverse=True)[:2]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sc = ax.scatter(np.log10(fac[top2[0]]) if fac[top2[0]].min() > 0 else fac[top2[0]], fac[top2[1]], c=fac.f, cmap="coolwarm", vmin=-0.3, vmax=0.3, s=60, edgecolor="k")
    ax.set_xlabel(f"log10 {top2[0]}"); ax.set_ylabel(top2[1]); fig.colorbar(sc, label="f = success(BRIDGE data) − success(PD-noise data)")
    ax.set_title("g5 factorial: f over the two most sensitive axes"); save(fig, "gen_5_niche_map")
    st = ["| axis | range of f across its 3 levels (factorial) |", "|---|---|"] + [f"| {a} | {sens[a]:.3f} |" for a in sorted(sens, key=sens.get, reverse=True)]
    out = "## Search (best and worst 5 of the CMA-ES evaluations)\n\n" + "\n".join(t) + "\n\n## Validation\n\n" + "\n".join(vt) + \
          f"\n\n## Factorial sensitivity (27 runs)\n\n" + "\n".join(st) + f"\n\nfactorial f: mean {fac.f.mean():+.3f}, min {fac.f.min():+.3f}, max {fac.f.max():+.3f}; search f range [{srch.f.min():+.3f}, {srch.f.max():+.3f}]\n"
    open(os.path.join(FIND, "_gen5_tables.md"), "w", encoding="utf-8").write(out); print(out)


def g6():
    df = load("g6"); a, b = df[df.phase == "g6a"], df[df.phase == "g6b"]
    t = ["| transfer | BRIDGE-slip | PD-noise | DART |", "|---|---|---|---|"]
    for lab in sorted(a.transfer.unique()):
        d = a[a.transfer == lab]; t.append(f"| {lab} | " + " | ".join(fmt(*ci95(d[d.source == s].success.values)) for s in ("BRIDGE-slip", "PD-noise", "DART")) + " |")
    bt = ["| class-flip rate in the generator's map | BRIDGE-slip | PD-noise | DART |", "|---|---|---|---|"]
    ref = load("g0"); ref = ref[(ref.layout == "L1") & (ref.disturbance == "slip")]
    bt.append("| 0 (g0) | " + " | ".join(fmt(*ci95(ref[ref.source == s].success.values)) for s in ("BRIDGE-slip", "PD-noise", "DART")) + " |")
    for fl in sorted(b.map_flip.unique()):
        d = b[b.map_flip == fl]; bt.append(f"| {fl} | " + " | ".join(fmt(*ci95(d[d.source == s].success.values)) for s in ("BRIDGE-slip", "PD-noise", "DART")) + " |")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for s in ("BRIDGE-slip", "PD-noise", "DART"):
        xs = [0.0] + sorted(b.map_flip.unique()); ys = [ci95(ref[ref.source == s].success.values)] + [ci95(b[(b.map_flip == fl) & (b.source == s)].success.values) for fl in sorted(b.map_flip.unique())]
        ax.errorbar(xs, [y[0] for y in ys], [y[1] for y in ys], marker="o", capsize=2, label=s, color=COL[s])
    ax.set_xlabel("class-flip rate of the map the generator used"); ax.set_ylabel("success under slip (true map)"); ax.legend(fontsize=7)
    save(fig, "gen_6_model_error")
    out = "## Transfer across slip magnitude (success under slip at the target)\n\n" + "\n".join(t) + "\n\n## Wrong terrain model in the generator\n\n" + "\n".join(bt) + "\n"
    open(os.path.join(FIND, "_gen6_tables.md"), "w", encoding="utf-8").write(out); print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="g0"); a = ap.parse_args()
    os.makedirs(FIND, exist_ok=True)
    {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g6": g6}[a.phase]()
