"""Figures from results/*.csv and results/*.npz only.  PNG + PDF into figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

RES, FIGS = "results", "figures"
STYLE = {"B": dict(color="k", ls="--", lw=1.5, label="B oracle"),
         "C": dict(color="C3", ls="-", lw=2.5, label="C learned eps (bridge)"),
         "D": dict(color="C0", ls="-", lw=2.5, label="D learned eps (no IPF)"),
         "Dprime": dict(color="C2", ls="-", lw=1.0, label="D' flow matching"),
         "E": dict(color="C7", ls=":", lw=1.5, label="E plain PPO")}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_wall(ax, w):
    ax.add_patch(Rectangle((0.495, 0), 0.01, 0.5 - w / 2, color="k"))
    ax.add_patch(Rectangle((0.495, 0.5 + w / 2), 0.01, 0.5 - w / 2, color="k"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")


def regions(w):
    return [((0.15, 0.5), (0.06, 0.15)), ((0.42, 0.5), (0.03, 0.4 * w)),
            ((0.58, 0.5), (0.03, 0.4 * w)), ((0.85, 0.5), (0.06, 0.15))]


def fig1_success(summ):
    widths = sorted(summ.w.unique())
    fig, axes = plt.subplots(1, len(widths), figsize=(4.2 * len(widths), 3.6), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), widths):
        s = summ[summ.w == w]
        for cond in sorted(s.condition.unique()):
            d = s[s.condition == cond].sort_values("sigma_d")
            if cond.startswith("A"):
                ax.plot(d.sigma_d, d.success_mean, color="0.6", lw=0.8, label="A fixed eps" if cond == "A0.001" else None)
                ax.annotate(cond[1:], (d.sigma_d.iloc[-1], d.success_mean.iloc[-1]), fontsize=6, color="0.4")
            elif cond in STYLE:
                ax.plot(d.sigma_d, d.success_mean, **STYLE[cond])
                ax.fill_between(d.sigma_d, d.success_mean - d.success_std.fillna(0),
                                d.success_mean + d.success_std.fillna(0), color=STYLE[cond]["color"], alpha=0.12)
        ax.set_title(f"gap w = {w:.2f}"); ax.set_xlabel("disturbance sigma_d"); ax.set_ylim(-0.02, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("success rate"); np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower left")
    save(fig, "fig1_success_vs_disturbance")


def fig2_heatmap(w=0.10, cond="C"):
    p = os.path.join(RES, "heatmaps.npz")
    if not os.path.exists(p):
        return
    hm = np.load(p)
    keys = [k for k in hm.files if k.startswith(f"{cond}_w{w:.2f}_")]
    if not keys:
        return
    H = np.mean([hm[k] for k in keys], 0)  # (3, 50, 50) mean over seeds
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for k, ax in enumerate(axes):
        im = ax.imshow(np.exp(H[k]), origin="lower", extent=(0, 1, 0, 1), cmap="viridis",
                       norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1e-1))
        draw_wall(ax, w)
        m, s = regions(w)[k]; m2, s2 = regions(w)[k + 1]
        ax.add_patch(Ellipse(m, 4 * s[0], 4 * s[1], fill=False, ec="w", ls="--"))
        ax.add_patch(Ellipse(m2, 4 * s2[0], 4 * s2[1], fill=False, ec="w"))
        ax.set_title(f"skill {k + 1}: mean eps(x)")
    fig.colorbar(im, ax=axes, label="eps", fraction=0.02)
    fig.suptitle(f"condition {cond}, w={w:.2f} (one policy per w, trained with sigma_d randomised; "
                 f"{len(keys)} seeds averaged)", fontsize=9)
    save(fig, f"fig2_eps_heatmap_{cond}")


def fig3_pareto(summ):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for cond in sorted(summ.condition.unique()):
        d = summ[summ.condition == cond]
        if cond.startswith("A"):
            ax.scatter(d.effort_mean, d.success_mean, s=12, color="0.6", label="A fixed eps" if cond == "A0.001" else None)
        elif cond in STYLE:
            ax.scatter(d.effort_mean, d.success_mean, s=28, color=STYLE[cond]["color"], label=STYLE[cond]["label"],
                       marker="s" if cond == "B" else "o", edgecolor="k", lw=0.3)
    ax.set_xlabel("mean control effort  (integral of |u|^2 dt)"); ax.set_ylabel("success rate")
    ax.set_title("all conditions x all 12 cells (seed means)"); ax.legend(fontsize=7)
    save(fig, "fig3_success_vs_effort")


def fig4_bridge_sanity(w=0.10, seed=0):
    p = os.path.join(RES, "verify_samples_bridge.npz")
    if not os.path.exists(p):
        return
    S = np.load(p); ver = pd.read_csv(os.path.join(RES, "verify.csv")); ipf = pd.read_csv(os.path.join(RES, "ipf.csv"))
    eps_list = sorted(ver.eps.unique())
    fig = plt.figure(figsize=(3.3 * len(eps_list) + 4, 9.5))
    gs = fig.add_gridspec(3, len(eps_list) + 1, width_ratios=[1] * len(eps_list) + [1.3])
    for k in range(3):
        for j, eps in enumerate(eps_list):
            ax = fig.add_subplot(gs[k, j])
            key = f"bridge_w{w:.2f}_s{seed}_k{k + 1}_eps{eps}"
            if key + "_end" not in S.files:
                continue
            end, alive, tgt = S[key + "_end"], S[key + "_alive"], S[key + "_target"]
            ax.scatter(tgt[:, 0], tgt[:, 1], s=3, color="0.7", label="rho_k samples")
            ax.scatter(end[alive, 0], end[alive, 1], s=3, color="C3", label="bridge endpoints")
            ax.scatter(end[~alive, 0], end[~alive, 1], s=3, color="C0", alpha=0.4, label="killed")
            m, s = regions(w)[k + 1]
            ax.add_patch(Ellipse(m, 4 * s[0], 4 * s[1], fill=False, ec="k"))
            draw_wall(ax, w)
            r = ver[(ver.kind == "bridge") & (ver.w == w) & (ver.seed == seed) & (ver.skill == k + 1) & (ver.eps == eps)]
            ax.set_title(f"skill {k + 1}, eps={eps}: W2={r.w2.mean():.3f}, killed={r.killed.mean():.2f}", fontsize=8)
            if k == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper left")
    ax = fig.add_subplot(gs[:, -1])
    d = ipf[(ipf.kind == "bridge") & (ipf.w == w)]
    for k in range(3):
        g = d[d.skill == k + 1].groupby("iter")
        ax.errorbar(g.w2.mean().index, g.w2.mean(), g.w2.std().fillna(0), label=f"skill {k + 1} W2(endpoints, rho_k)", marker="o")
        ax.plot(g.sinkhorn.mean().index, g.sinkhorn.mean(), ls="--", marker="x", color=f"C{k}", label=f"skill {k + 1} Sinkhorn cost")
    ax.set_xlabel("IPF iteration"); ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_title(f"IPF convergence, w={w:.2f}")
    fig.suptitle(f"Bridge sanity check, w={w:.2f}, seed {seed}", fontsize=10)
    save(fig, "fig4_bridge_sanity")


def fig5_handoff(w=0.10, sd=0.05):
    h = pd.read_csv(os.path.join(RES, "handoff.csv"))
    h = h[(h.w == w) & (h.sigma_d == sd) & (h.condition != "E")]
    per_seed = h.groupby(["condition", "handoff", "seed"]).w2.mean().reset_index()
    g = per_seed.groupby(["condition", "handoff"]).w2.agg(["mean", "std"]).reset_index()
    conds = sorted(g.condition.unique(), key=lambda c: (not c.startswith("A"), c))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(conds)); bw = 0.38
    for j in (1, 2):
        d = g[g.handoff == j].set_index("condition").reindex(conds)
        ax.bar(x + (j - 1.5) * bw, d["mean"], bw, yerr=d["std"].fillna(0), capsize=2, label=f"handoff {j} (to rho_{j})")
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=30, fontsize=8)
    ax.set_ylabel("W2(handoff states, rho_k)"); ax.set_title(f"handoff error, w={w:.2f}, sigma_d={sd}"); ax.legend(fontsize=8)
    save(fig, "fig5_handoff_error")


def main():
    os.makedirs(FIGS, exist_ok=True)
    summ = pd.read_csv(os.path.join(RES, "summary.csv"))
    fig1_success(summ); fig2_heatmap(); fig2_heatmap(cond="D"); fig3_pareto(summ); fig4_bridge_sanity(); fig5_handoff()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
