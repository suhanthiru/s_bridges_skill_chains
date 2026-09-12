"""Seam-suite figures and Phase C (containment) analysis.  `python run.py --phase C`."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FIGS, FIND = "figures", "findings"
STYLE = {"TRACK-oracle": dict(color="k", ls="--"), "TRACK-naive": dict(color="C1", ls="-"),
         "BRIDGE-tracked": dict(color="C3", ls="-"), "BRIDGE-slip-tracked": dict(color="C3", ls="-", marker="o"),
         "TSM": dict(color="C0", ls="-")}


def ci95(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(m, h, p=3):
    return f"{m:.{p}f} ± {h:.{p}f}" if np.isfinite(h) else f"{m:.{p}f}"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


def load():
    return pd.read_parquet("results_seam.parquet"), pd.read_parquet("results_seam_episodes.parquet")


def paired(a, b):
    """Paired over seeds: mean diff, CI half-width, p."""
    j = a.index.intersection(b.index)
    d = (a[j] - b[j]).values
    m, h = ci95(d)
    p = float(stats.ttest_rel(a[j], b[j]).pvalue) if len(j) > 1 else np.nan
    return m, h, p


# --------------------------------------------------------------- Phase A
def phase_A(cells):
    A = cells[cells.phase == "A"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, rough in zip(axes, ("mild", "rough")):
        for cond, st in STYLE.items():
            d = A[(A.rough == rough) & (A.condition == cond)]
            if d.empty:
                continue
            g = d.groupby("sigma_k").success
            m = g.mean(); h = g.apply(lambda v: ci95(v)[1])
            ax.errorbar(m.index, m.values, h.values, label=cond, capsize=2, lw=2, **st)
        ax.set_title(f"{rough} terrain"); ax.set_xlabel("push magnitude sigma_k"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("success rate"); axes[1].legend(fontsize=7)
    fig.suptitle("Phase A: point vs cloud, same PD tracker (mean ± 95% CI over seeds)", fontsize=10)
    save(fig, "seam_A_success")
    lines = ["| condition | terrain | " + " | ".join(f"σ_k={s}" for s in (0.0, 0.06, 0.10, 0.15)) + " |", "|---|---|---|---|---|---|"]
    for cond in STYLE:
        for rough in ("mild", "rough"):
            d = A[(A.rough == rough) & (A.condition == cond)]
            if d.empty:
                continue
            lines.append(f"| {cond} | {rough} | " + " | ".join(fmt(*ci95(d[d.sigma_k == s].success)) for s in (0.0, 0.06, 0.10, 0.15)) + " |")
    tests = ["| terrain | σ_k | comparison | Δ success | 95% CI | p |", "|---|---|---|---|---|---|"]
    for rough in ("mild", "rough"):
        for s in (0.06, 0.10, 0.15):
            base = A[(A.rough == rough) & (A.sigma_k == s) & (A.condition == "TRACK-naive")].set_index("seed").success
            for c in ("BRIDGE-tracked", "BRIDGE-slip-tracked", "TSM"):
                b = A[(A.rough == rough) & (A.sigma_k == s) & (A.condition == c)].set_index("seed").success
                if b.empty:
                    continue
                m, h, p = paired(b, base)
                tests.append(f"| {rough} | {s} | {c} − TRACK-naive | {m:+.3f} | [{m - h:+.3f}, {m + h:+.3f}] | {p:.3f} |")
            orc = A[(A.rough == rough) & (A.sigma_k == s) & (A.condition == "TRACK-oracle")].set_index("seed").success
            b = A[(A.rough == rough) & (A.sigma_k == s) & (A.condition == "BRIDGE-tracked")].set_index("seed").success
            if not b.empty:
                m, h, p = paired(b, orc)
                tests.append(f"| {rough} | {s} | BRIDGE-tracked − TRACK-oracle | {m:+.3f} | [{m - h:+.3f}, {m + h:+.3f}] | {p:.3f} |")
    return "\n".join(lines), "\n".join(tests)


# --------------------------------------------------------------- Phase B
def phase_B(cells):
    B = cells[cells.phase == "B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    argmax = ["| terrain | σ_k | argmax_w (bootstrap 95% CI) | success at argmax |", "|---|---|---|---|"]
    rng = np.random.RandomState(0)
    for ax, rough in zip(axes, ("mild", "rough")):
        for j, s in enumerate((0.0, 0.06, 0.10)):
            d = B[(B.rough == rough) & (B.sigma_k == s)]
            g = d.groupby("w").success
            m, h = g.mean(), g.apply(lambda v: ci95(v)[1])
            ax.errorbar(m.index, m.values, h.values, label=f"sigma_k={s}", marker="o", capsize=2, color=f"C{j}")
            # bootstrap argmax over seeds
            piv = d.pivot_table(index="seed", columns="w", values="success")
            if len(piv) >= 2:
                bs = [piv.iloc[rng.randint(len(piv), size=len(piv))].mean().idxmax() for _ in range(1000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                argmax.append(f"| {rough} | {s} | {m.idxmax()} [{lo}, {hi}] | {m.max():.3f} |")
        ax.set_xscale("log"); ax.set_xlabel("handoff width scale w"); ax.set_title(f"{rough} terrain")
    axes[0].set_ylabel("success rate"); axes[1].legend(fontsize=8)
    fig.suptitle("Phase B: handoff-width sweep, BRIDGE-slip-tracked", fontsize=10)
    save(fig, "seam_B_width")
    lines = ["| terrain | σ_k | " + " | ".join(f"w={w}" for w in (0.25, 0.5, 1.0, 2.0, 4.0)) + " |", "|---|---|" + "---|" * 5]
    for rough in ("mild", "rough"):
        for s in (0.0, 0.06, 0.10):
            d = B[(B.rough == rough) & (B.sigma_k == s)]
            lines.append(f"| {rough} | {s} | " + " | ".join(fmt(*ci95(d[d.w == w].success)) for w in (0.25, 0.5, 1.0, 2.0, 4.0)) + " |")
    return "\n".join(lines), "\n".join(argmax)


# --------------------------------------------------------------- Phase C
def phase_C(eps=None):
    if eps is None:
        _, eps = load()
    e = eps[(eps.phase.isin(["A", "B"])) & (eps.w == 1.0)].copy()
    e["skill2_ok"] = (e.md2 <= 2).astype(int)       # skill 2 delivered inside rho_2
    e["skill3_ok"] = e.success                       # skill 3 delivered inside rho_3 (= success)
    rows = ["| condition | terrain | skill | P(ok \\| entered inside 2σ) | P(ok \\| entered outside) | gap | n_in / n_out |",
            "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    bins = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6])
    for r_, rough in enumerate(("mild", "rough")):
        for c_, (skill, ent, out) in enumerate(((2, "md1", "skill2_ok"), (3, "md2", "skill3_ok"))):
            ax = axes[r_, c_]
            for cond, st in STYLE.items():
                d = e[(e.rough == rough) & (e.condition == cond) & np.isfinite(e[ent])]
                if d.empty:
                    continue
                inside, outside = d[d[ent] <= 2], d[d[ent] > 2]
                pin, pout = inside[out].mean(), outside[out].mean() if len(outside) else np.nan
                rows.append(f"| {cond} | {rough} | {skill} | {pin:.3f} | {pout:.3f} | {pin - pout:+.3f} | {len(inside)} / {len(outside)} |")
                cut = pd.cut(d[ent], bins)
                g = d.groupby(cut, observed=True)[out].agg(["mean", "size"])
                g = g[g["size"] >= 20]
                x = [iv.mid for iv in g.index]
                ax.plot(x, g["mean"], marker="o", label=cond, **st)
            ax.axvline(2, color="0.5", ls=":"); ax.set_title(f"{rough}: skill {skill}, entry distance to rho_{skill - 1}")
            ax.set_xlabel("Mahalanobis entry distance (binned)")
        axes[r_, 0].set_ylabel("P(skill delivers inside next cloud)")
    axes[0, 1].legend(fontsize=7)
    fig.suptitle("Phase C: seam containment - success vs how far outside the precondition cloud the skill started", fontsize=10)
    save(fig, "seam_C_containment")
    table = "\n".join(rows)
    os.makedirs(FIND, exist_ok=True)
    open(os.path.join(FIND, "_seam_C_table.md"), "w", encoding="utf-8").write(table + "\n")
    print(table)
    return table


# --------------------------------------------------------------- Phase D
def phase_D(cells):
    Dd = cells[cells.phase == "D"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    tab = ["| condition | axis | " + " | ".join(f"δ={d}" for d in (0.0, 0.5, 1.0, 2.0, 3.0)) + " | first δ < 0.5 |", "|---|---|" + "---|" * 6]
    for ax, axis in zip(axes, ("normal", "heading")):
        for cond in ("BRIDGE-tracked", "BRIDGE-slip-tracked", "TSM"):
            d = Dd[(Dd.condition == cond) & ((Dd.shift == f"shift_{axis}") | (Dd.shift == "none"))]
            g = d.groupby("delta").success
            m, h = g.mean(), g.apply(lambda v: ci95(v)[1])
            ax.errorbar(m.index, m.values, h.values, marker="o", capsize=2, label=cond, **STYLE[cond])
            below = [dl for dl in m.index if m[dl] < 0.5]
            tab.append(f"| {cond} | {axis} | " + " | ".join(fmt(*ci95(d[d.delta == dl].success)) for dl in (0.0, 0.5, 1.0, 2.0, 3.0))
                       + f" | {below[0] if below else 'none'} |")
        ax.axhline(0.5, color="0.5", ls=":"); ax.set_xlabel(f"mean shift δ (in σ) along the corridor {axis}")
        ax.set_title(f"shift along {axis}")
    axes[0].set_ylabel("success of skills 2+3"); axes[1].legend(fontsize=7)
    fig.suptitle("Phase D: breaking the seam - skill 2 started from a shifted rho_1' (rough, sigma_k = 0.06)", fontsize=10)
    save(fig, "seam_D_shift")
    other = ["| condition | none | rot45 | rot90 | shrink 0.25x | grow 4x |", "|---|---|---|---|---|---|"]
    for cond in ("BRIDGE-tracked", "BRIDGE-slip-tracked", "TSM"):
        d = Dd[Dd.condition == cond]
        other.append(f"| {cond} | " + " | ".join(fmt(*ci95(d[d.shift == k].success)) for k in ("none", "rot45", "rot90", "shrink", "grow")) + " |")
    init = Dd.groupby(["shift", "delta"]).frac_in_tsm_init.mean().round(2).to_dict()
    return "\n".join(tab), "\n".join(other), init


def main():
    cells, eps = load()
    out = {}
    if (cells.phase == "A").any():
        out["A"] = phase_A(cells)
    if (cells.phase == "B").any():
        out["B"] = phase_B(cells)
    if (cells.phase == "D").any():
        out["D"] = phase_D(cells)
    out["C"] = phase_C(eps)
    for k, v in out.items():
        print(f"\n===== {k} =====")
        for t in (v if isinstance(v, tuple) else (v,)):
            print(t if isinstance(t, str) else t)


if __name__ == "__main__":
    main()
