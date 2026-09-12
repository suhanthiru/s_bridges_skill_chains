"""Terrain experiment figures from results_terrain/*.csv|npz.  PNG + PDF into figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.environ.get("SB_RES", "results_terrain")
FIGS = os.environ.get("SB_FIGS", "figures")
STYLE = {"ORACLE": dict(color="k", ls="--", lw=1.5), "TRACK": dict(color="C1", ls="-", lw=2),
         "BRIDGE-plain": dict(color="C3", ls="-", lw=2), "BRIDGE-obs": dict(color="C3", ls="-", lw=3, marker="o"),
         "DIFF": dict(color="C0", ls="-", lw=2), "PPO": dict(color="C7", ls=":", lw=1.5)}
ROUTE = np.array([[0.12, 0.5], [0.37, 0.5], [0.63, 0.5], [0.88, 0.5]])


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight"); plt.close(fig)


def lines(summ, col, ylabel, name, err="std"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, rough in zip(axes, ("mild", "rough")):
        s = summ[summ.rough == rough]
        for cond in [c for c in STYLE if c in set(s.condition)]:
            d = s[s.condition == cond].sort_values("sigma_k")
            ax.plot(d.sigma_k, d[col + "_mean"], label=cond, **STYLE[cond])
            sd = d[col + "_" + err].fillna(0)
            ax.fill_between(d.sigma_k, d[col + "_mean"] - sd, d[col + "_mean"] + sd, color=STYLE[cond]["color"], alpha=0.12)
        ax.set_title(f"{rough} terrain"); ax.set_xlabel("push magnitude sigma_k")
    axes[0].set_ylabel(ylabel); axes[1].legend(fontsize=7)
    save(fig, name)


def fig2_recovery(pushes):
    """Median recovery time with IQR bands (over pushes pooled across seeds) and never-recovered fraction."""
    p = pushes.assign(rec=pushes.recovery.where(pushes.recovery >= 0))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for j, rough in enumerate(("mild", "rough")):
        s = p[(p.rough == rough) & (p.sigma_k > 0)]
        for cond in [c for c in STYLE if c in set(s.condition)]:
            g = s[s.condition == cond].groupby("sigma_k")
            med, q1, q3 = g.rec.median(), g.rec.quantile(0.25), g.rec.quantile(0.75)
            axes[0, j].plot(med.index, med, label=cond, **STYLE[cond])
            axes[0, j].fill_between(med.index, q1, q3, color=STYLE[cond]["color"], alpha=0.12)
            nev = g.apply(lambda x: (x.recovery < 0).mean(), include_groups=False)
            axes[1, j].plot(nev.index, nev, label=cond, **STYLE[cond])
        axes[0, j].set_title(f"{rough} terrain"); axes[1, j].set_xlabel("push magnitude sigma_k")
    axes[0, 0].set_ylabel("recovery time (steps), median + IQR"); axes[1, 0].set_ylabel("fraction never recovered")
    axes[0, 1].legend(fontsize=7)
    save(fig, "fig2t_recovery_vs_push")


def fig3_rollouts():
    p = os.path.join(RES, "rollouts.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    conds = [c for c in ("BRIDGE-obs", "TRACK", "DIFF", "PPO") if f"{c}_traj" in z.files]
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4))
    for ax, cond in zip(np.atleast_1d(axes), conds):
        ax.imshow(z["field"], origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=0, vmax=1)
        tr, pu = z[f"{cond}_traj"], z[f"{cond}_push"]
        for i in range(tr.shape[0]):
            ax.plot(tr[i, :, 0], tr[i, :, 1], color="C3", lw=0.7, alpha=0.7)
            idx = np.where(pu[i])[0] + 1
            ax.scatter(tr[i, idx, 0], tr[i, idx, 1], s=8, color="C0", zorder=3)
        for m in ROUTE:
            ax.add_patch(plt.Circle(m, 0.1, fill=False, ec="k", lw=0.8))
        ax.set_title(cond); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.suptitle("20 rollouts, rough terrain, sigma_k=0.10, layout 25 (blue dots = pushes; grey = roughness)", fontsize=9)
    save(fig, "fig3t_example_rollouts")


def fig4_dev_vs_success(summ):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for cond in [c for c in STYLE if c in set(summ.condition)]:
        d = summ[summ.condition == cond]
        ax.scatter(d.corridor_dev_mean, d.success_mean, label=cond, color=STYLE[cond]["color"], s=28, edgecolor="k", lw=0.3,
                   marker="s" if cond == "ORACLE" else "o")
    ax.set_xlabel("corridor deviation (mean distance from skill segment)"); ax.set_ylabel("success rate")
    ax.set_title("all conditions x 10 cells"); ax.legend(fontsize=7)
    save(fig, "fig4t_deviation_vs_success")


def fig5_plain_vs_obs(summ):
    s = summ[summ.condition.isin(["BRIDGE-plain", "BRIDGE-obs"])]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cells = [(r, k) for r in ("mild", "rough") for k in sorted(s.sigma_k.unique())]
    x = np.arange(len(cells)); bw = 0.38
    for j, cond in enumerate(("BRIDGE-plain", "BRIDGE-obs")):
        d = s[s.condition == cond].set_index(["rough", "sigma_k"]).reindex(cells)
        ax.bar(x + (j - 0.5) * bw, d.success_mean, bw, yerr=d.success_std.fillna(0), capsize=2, label=cond,
               color="C3" if j else "C6")
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n{k}" for r, k in cells], fontsize=7)
    ax.set_ylabel("success rate"); ax.legend(fontsize=8); ax.set_title("what the 16-ray roughness observation buys")
    save(fig, "fig5t_plain_vs_obs")


def main():
    os.makedirs(FIGS, exist_ok=True)
    summ = pd.read_csv(os.path.join(RES, "summary.csv"))
    lines(summ, "success", "success rate", "fig1t_success_vs_push")
    pp = os.path.join(RES, "pushes.csv")
    if os.path.exists(pp) and os.path.getsize(pp) > 10:
        fig2_recovery(pd.read_csv(pp))
    fig3_rollouts(); fig4_dev_vs_success(summ); fig5_plain_vs_obs(summ)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
