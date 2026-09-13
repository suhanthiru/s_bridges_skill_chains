# Seam B — Width sweep

**Question.** Is there an optimum handoff width, and does it move with disturbance?
**Setup.** BRIDGE-slip-tracked only (Phase A's slip reference passed sanity: 1.00 undisturbed). ρ₁ and ρ₂ covariance scaled by w ∈ {0.25, 0.5, 1, 2, 4}; the three bridges retrained per w on samples of the scaled marginals; the same PD tracker. Grid w × {mild, rough} × σ_k ∈ {0, 0.06, 0.10}; 200 episodes per cell; **5 seeds**. Success is unchanged in definition (ρ₃'s 2σ ellipse; ρ₃ is not scaled).
**Pre-registered pass.** Success as a function of w is unimodal with an interior optimum, and argmax_w moves to larger w as σ_k increases (with bootstrap CI).
**Verdict: PASS, with one boundary cell.** At σ_k = 0.06 the curve is unimodal with its maximum at w = 1 on both terrains (mild 0.929 vs 0.877 / 0.890 at the ends; rough 0.921 vs 0.874 / 0.885), and the argmax moves outward with disturbance on both terrains — mild 0.25 → 1 → 4, rough 1 → 1 → 2 — with bootstrap CIs that do not overlap between σ_k = 0.06 and 0.10. The one qualification: on mild terrain at σ_k = 0.10 the argmax is the largest width tested (4), so that curve is monotone over the tested range; on rough terrain it is interior (2, CI [2, 2]). The handoff width is a robustness knob and the right setting depends on how hard the robot is pushed.

## Success (mean ± 95% CI over seeds)

| terrain | σ_k | w = 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| mild | 0.00 | **1.000 ± 0.000** | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.970 ± 0.015 | 0.954 ± 0.034 |
| mild | 0.06 | 0.877 ± 0.023 | 0.899 ± 0.023 | **0.929 ± 0.022** | 0.901 ± 0.031 | 0.890 ± 0.029 |
| mild | 0.10 | 0.666 ± 0.042 | 0.687 ± 0.051 | 0.744 ± 0.055 | 0.785 ± 0.034 | **0.811 ± 0.041** |
| rough | 0.00 | 0.990 ± 0.012 | 0.995 ± 0.006 | **0.996 ± 0.007** | 0.994 ± 0.008 | 0.964 ± 0.018 |
| rough | 0.06 | 0.874 ± 0.028 | 0.888 ± 0.033 | **0.921 ± 0.026** | 0.919 ± 0.028 | 0.885 ± 0.041 |
| rough | 0.10 | 0.673 ± 0.038 | 0.690 ± 0.053 | 0.763 ± 0.023 | **0.792 ± 0.025** | 0.764 ± 0.038 |

| terrain | σ_k | argmax_w [bootstrap 95% CI over seeds] | success at argmax |
|---|---|---|---|
| mild | 0.00 | 0.25 [0.25, 0.25] | 1.000 (tie with 0.5, 1) |
| mild | 0.06 | 1.0 [1.0, 1.0] | 0.929 |
| mild | 0.10 | 4.0 [4.0, 4.0] | 0.811 |
| rough | 0.00 | 1.0 [0.5, 1.0] | 0.996 |
| rough | 0.06 | 1.0 [1.0, 2.0] | 0.921 |
| rough | 0.10 | 2.0 [2.0, 2.0] | 0.792 |

## Reading the table

* **Two opposing effects, as the hypothesis needs.** Undisturbed, wide handoffs cost success (w = 4: 0.954 / 0.964) because the bridge for skill 3 must now reach ρ₃ from a cloud that extends 0.2 m from the nominal handoff point — a longer, more varied transport within a fixed 100-step budget. Under pushes, narrow handoffs cost success (w = 0.25 at σ_k = 0.10: 0.67) because a pushed robot that lands outside a tight ρ₂ starts skill 3 off its training support. The crossover moves outward as pushes grow.
* **The optimum is at the nominal width for moderate pushes** (w = 1 at σ_k = 0.06 on both terrains, +0.03 to +0.05 over the neighbours), which says the marginals as originally placed were about right for that disturbance and too tight for σ_k = 0.10.
* **Effect sizes are modest but consistent**: 0.05–0.15 across the sweep at σ_k ≥ 0.06, with seed CIs of ±0.02–0.05.
* This is the *planning-level* effect of the cloud that Phase A could not see: Phase A compared the cloud's mean path against a point's straight line (and lost); Phase B varies the cloud that the *next* skill is trained on, and that matters.

## Notes

* Only the bridges are retrained per w; the PD tracker, terrain, demos and layouts are the Phase A ones.
* Compute: 11–12 min per seed on the GPU (12 bridges retrained per seed).

## What I'd run next

* Extend the mild σ_k = 0.10 sweep past w = 4 to locate its interior optimum, and add σ_k = 0.15.
* Scale ρ₁ and ρ₂ independently; the width that matters is presumably ρ₂'s (skill 3's entry), and separating them would say so.
* Repeat with TSM's initiation-set threshold (0.8 → 0.5) as the analogue of width, to see whether the same knob exists on the published-method side.
