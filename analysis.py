"""Tables and figures from results.parquet.  `python analysis.py --phase 1`."""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES, FIGS, FIND = "results", "figures", "findings"
REFS = ("brownian", "killed", "unicycle", "slip")
MFS = ("flat", "se2")
DISTS = ("none", "slip", "rain")


def ci95(x):
    """Mean and half-width of the 95% CI over seeds (t, n-1 df)."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(m, h, p=3):
    return f"{m:.{p}f} ± {h:.{p}f}" if np.isfinite(h) else f"{m:.{p}f}"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def agg(df, keys, col):
    rows = []
    for k, g in df.groupby(list(keys)):
        m, h = ci95(g[col].values)
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), mean=m, ci=h, n=len(g)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ phase 1
def phase1(df, ipf):
    K = int(df[df.ipf > 0].ipf.max())
    br = df[df.condition.str.startswith("bridge")]
    nom = df[df.condition == "nominal"]

    # ---- figure 1: success by reference, manifold and IPF checkpoint
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    w = 0.2
    for ax, dist in zip(axes, DISTS):
        d = br[br.disturbance == dist]
        x = np.arange(len(REFS))
        for j, (mf, it) in enumerate([(m, i) for m in MFS for i in (0, K)]):
            s = agg(d[(d.manifold == mf) & (d.ipf == it)], ("reference",), "success").set_index("reference").reindex(REFS)
            ax.bar(x + (j - 1.5) * w, s["mean"], w, yerr=s["ci"], capsize=2,
                   label=f"{mf}, IPF {it}", color=f"C{j}", alpha=0.55 if it == 0 else 1.0)
        m, h = ci95(nom[nom.disturbance == dist].success.values)
        ax.axhline(m, color="k", ls="--", lw=1.2, label="nominal PD")
        ax.set_xticks(x); ax.set_xticklabels(REFS, fontsize=8, rotation=15)
        ax.set_title(f"disturbance: {dist}"); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("success rate"); axes[2].legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 1: reference process x manifold x IPF (mean ± 95% CI over seeds)", fontsize=10)
    save(fig, "phase1_success")

    # ---- figure 2: paired IPF effect (iteration K minus iteration 0, per seed)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    deltas = []
    for ax, dist in zip(axes, DISTS):
        d = br[br.disturbance == dist]
        labels, ms, hs = [], [], []
        for mf in MFS:
            for ref in REFS:
                a = d[(d.manifold == mf) & (d.reference == ref) & (d.ipf == 0)].set_index("seed").success
                b = d[(d.manifold == mf) & (d.reference == ref) & (d.ipf == K)].set_index("seed").success
                j = a.index.intersection(b.index)
                dl = (b[j] - a[j]).values
                m, h = ci95(dl)
                p = float(stats.ttest_rel(b[j], a[j]).pvalue) if len(j) > 1 else np.nan
                deltas.append(dict(disturbance=dist, manifold=mf, reference=ref, delta=m, ci=h, p=p, n=len(j)))
                labels.append(f"{ref}\n{mf}"); ms.append(m); hs.append(h)
        xs = np.arange(len(labels))
        ax.bar(xs, ms, 0.65, yerr=hs, capsize=2,
               color=["C3" if m < 0 else "C2" for m in ms])
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(f"disturbance: {dist}")
    axes[0].set_ylabel(f"success(IPF {K}) - success(IPF 0)")
    fig.suptitle("Phase 1: paired effect of IPF refinement (positive = IPF helps)", fontsize=10)
    save(fig, "phase1_ipf_effect")

    # ---- figure 3: IPF convergence diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, lab in zip(axes, ("w2_endpoint", "transport_cost"),
                            ("W2(simulated endpoints, rho_k)", "transport cost E|log(x0^-1 x1)|^2")):
        for j, ref in enumerate(REFS):
            for mf, ls in zip(MFS, ("--", "-")):
                s = agg(ipf[(ipf.reference == ref) & (ipf.manifold == mf)], ("iter",), col).sort_values("iter")
                ax.errorbar(s["iter"], s["mean"], s["ci"], color=f"C{j}", ls=ls, marker="o", ms=3,
                            label=f"{ref}/{mf}", lw=1.2)
        ax.set_xlabel("IPF iteration"); ax.set_ylabel(lab); ax.set_yscale("log")
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Phase 1: does IPF converge? (pooled over the three skills)", fontsize=10)
    save(fig, "phase1_ipf_convergence")

    # ---- tables
    lines = ["| reference | manifold | IPF | " + " | ".join(f"succ ({d})" for d in DISTS)
             + " | " + " | ".join(f"W2 ({d})" for d in DISTS) + " |",
             "|---|---|---|" + "---|" * (2 * len(DISTS))]
    for mf in MFS:
        for ref in REFS:
            for it in (0, K):
                d = br[(br.manifold == mf) & (br.reference == ref) & (br.ipf == it)]
                sc = [fmt(*ci95(d[d.disturbance == x].success.values)) for x in DISTS]
                w2 = [fmt(*ci95(d[d.disturbance == x].w2_goal.values)) for x in DISTS]
                lines.append(f"| {ref} | {mf} | {it} | " + " | ".join(sc) + " | " + " | ".join(w2) + " |")
    d = nom
    sc = [fmt(*ci95(d[d.disturbance == x].success.values)) for x in DISTS]
    lines.append("| nominal PD | - | - | " + " | ".join(sc) + " | " + " | ".join(["-"] * len(DISTS)) + " |")
    table = "\n".join(lines)

    dl = pd.DataFrame(deltas)
    dlines = ["| disturbance | reference | manifold | delta success (IPF K - IPF 0) | paired p |", "|---|---|---|---|---|"]
    for _, r in dl.iterrows():
        dlines.append(f"| {r.disturbance} | {r.reference} | {r.manifold} | {fmt(r.delta, r.ci)} | {r.p:.3f} |")
    os.makedirs(FIND, exist_ok=True)
    open(os.path.join(FIND, "_phase1_tables.md"), "w", encoding="utf-8").write(
        "## Main table\n\n" + table + "\n\n## Paired IPF effect\n\n" + "\n".join(dlines) + "\n")
    print(table)
    print()
    print(dl.to_string(index=False))
    return dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="1")
    a = ap.parse_args()
    df = pd.read_parquet("results.parquet")
    if a.phase == "1":
        import glob
        ipf = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(RES, "phase1_ipf_seed*.parquet"))],
                        ignore_index=True)
        phase1(df[df.phase == 1], ipf)


if __name__ == "__main__":
    main()
