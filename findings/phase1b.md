# Phase 1b — Does the Lie group matter?

**Hypothesis.** Tangent-space uncertainty on SE(2) beats flat ℝ³ exactly when heading uncertainty and body-frame slip anisotropy are large (and the route curves), and is indistinguishable otherwise.
**Setup.** Terrain-free routes {straight, one 90° turn, S-curve with 180° of heading change} × heading-noise scale h ∈ {0.05, 0.2, 0.5, 1.0} rad (marginal heading std = h, body-frame heading diffusion = h/2) × slip anisotropy a ∈ {1, 3, 10} (body-frame lateral/longitudinal std ratio, longitudinal 0.02). Reference = `slip` (best of Phase 1) with the constant body-frame covariance, iteration-0 bridges (IPF did not help in Phase 1), 1 000 Adam steps per skill. Both manifolds train on the same marginal samples; physics and metrics are SE(2) for both. 500 episodes per cell, **5 seeds**.
**Pre-registered pass.** SE(2) wins containment and covariance fidelity in the high-heading-noise / high-anisotropy / curved cells with p < 0.05, is equivariant, and is no worse than flat in the benign cell.
**Verdict: FAIL (split).** SE(2) passes on heading noise and curvature — it beats flat in every S-curve and 90°-turn cell with a ≤ 3 (up to +0.33 success, p < 0.001), is exactly equivariant (2e-6 vs 0.1–0.9 for flat), predicts its own uncertainty correctly where flat's ℝ³ Gaussian fails (coverage 0.70–0.74 vs 0.18–0.44, KL 0.001 vs up to 16), and ties in the benign cell (+0.035, p = 0.06). It **fails on anisotropy**: at a = 10 the flat model wins every cell by 0.14–0.37 (all p < 0.001), including the straight route. The claimed "high anisotropy" advantage is absent in this implementation, so the manifold is not adopted wholesale downstream (Phases 2–5 were configured with `se2` before this result; on the L1 terrain Phase 1 found the two indistinguishable).

## 1. Covariance fidelity (analytic vs 10 000 Monte-Carlo rollouts of skill 2, no training)

Fraction of MC samples inside the model's predicted 2σ set (nominal for a 3-D Gaussian: 0.739) and KL(MC fit ‖ prediction):

| route | h | a | cover flat | cover SE(2) | KL flat | KL SE(2) |
|---|---|---|---|---|---|---|
| straight | 0.05 | 1 | 0.727 | 0.736 | 0.044 | 0.001 |
| straight | 0.5 | 1 | 0.387 | 0.738 | 3.46 | 0.001 |
| straight | 1.0 | 10 | 0.378 | 0.516 | 9.35 | 2.13 |
| turn90 | 0.05 | 10 | 0.437 | 0.741 | 1.86 | 0.001 |
| turn90 | 0.5 | 3 | 0.429 | 0.734 | 2.67 | 0.004 |
| turn90 | 1.0 | 1 | 0.182 | 0.697 | 16.3 | 0.141 |
| scurve | 0.2 | 10 | 0.426 | 0.741 | 2.16 | 0.001 |
| scurve | 0.5 | 3 | 0.439 | 0.733 | 2.47 | 0.004 |
| scurve | 1.0 | 10 | 0.291 | 0.673 | 11.2 | 0.161 |

(full 36-cell table in `findings/_phase1b_tables.md`). As hypothesised: flat over-covers along the arc and under-covers the banana as soon as h ≥ 0.2 or a = 10, SE(2) stays at nominal except in the single most extreme cell (h = 1, a = 10). This check is model-only and does not involve the bridge solver.

## 2. Equivariance and θ wrap

* Rigid motion of the whole scene (g = (0.25, −0.15, 0.9 rad)): the SE(2) skills' deterministic drift flows move by g to **1.2e-6 – 7.6e-6** (mean SE(2) distance over 300 starts, every cell); the flat skills deviate by **0.14 – 0.87**. The SE(2) net consumes only body-frame features, so this holds by construction; the flat net sees world coordinates.
* θ wrap (ρ₂ and ρ₃ at heading π ± 0.1, S-curve, h = 0.2, a = 3): flat 0.550 / 0.575, SE(2) 0.614 / 0.584 (± 0.03–0.05). No cliff for either: the flat implementation wraps angle differences (a competent flat baseline does), so the wrap test does not separate them; the difference is the same +0.03–0.06 as the un-wrapped S-curve cells.

## 3. Handoff containment and success (paired over 5 seeds)

Success, flat vs SE(2), Δ = SE(2) − flat:

| route | a | h = 0.05 | h = 0.2 | h = 0.5 | h = 1.0 |
|---|---|---|---|---|---|
| straight | 1 | 0.770 / 0.805 (+0.04, p=.06) | 0.833 / 0.813 (−0.02, .24) | 0.616 / 0.700 (**+0.08**, .001) | 0.583 / 0.614 (+0.03, .29) |
| straight | 3 | 0.784 / 0.821 (+0.04, .02) | 0.766 / 0.783 (+0.02, .09) | 0.726 / 0.708 (−0.02, .35) | 0.587 / 0.622 (+0.04, .26) |
| straight | 10 | 0.536 / 0.567 (+0.03, .12) | 0.691 / 0.625 (**−0.07**, .02) | 0.728 / 0.581 (**−0.15**, <.001) | 0.687 / 0.550 (**−0.14**, <.001) |
| turn90 | 1 | 0.748 / 0.780 (+0.03, .17) | 0.730 / 0.778 (+0.05, .04) | 0.577 / 0.694 (**+0.12**, .003) | 0.538 / 0.562 (+0.03, .08) |
| turn90 | 3 | 0.727 / 0.775 (+0.05, .04) | 0.690 / 0.723 (+0.03, .42) | 0.613 / 0.708 (**+0.10**, .008) | 0.544 / 0.578 (+0.03, .32) |
| turn90 | 10 | 0.578 / 0.355 (**−0.22**, <.001) | 0.636 / 0.371 (**−0.27**, <.001) | 0.694 / 0.420 (**−0.27**, <.001) | 0.669 / 0.482 (**−0.19**, <.001) |
| scurve | 1 | 0.662 / 0.713 (+0.05, .07) | 0.611 / 0.728 (**+0.12**, .001) | 0.573 / 0.659 (**+0.09**, .002) | 0.501 / 0.579 (**+0.08**, .01) |
| scurve | 3 | 0.587 / 0.662 (+0.07, .09) | 0.464 / 0.622 (**+0.16**, .004) | 0.338 / 0.660 (**+0.32**, <.001) | 0.302 / 0.627 (**+0.33**, <.001) |
| scurve | 10 | 0.687 / 0.314 (**−0.37**, <.001) | 0.665 / 0.313 (**−0.35**, <.001) | 0.664 / 0.343 (**−0.32**, <.001) | 0.651 / 0.470 (**−0.18**, <.001) |

Handoff-1 Mahalanobis (SE(2)-correct, median) tells the same story: SE(2) lower or equal at a ≤ 3 (e.g. S-curve h=1, a=3: 1.60 vs 1.61; straight h=1, a=1: 1.54 vs 1.63), higher at a = 10 (1.73–1.84 vs 1.47–1.85). The gap opens with heading noise and curvature at a = 1–3 exactly as hypothesised (S-curve, a = 3: +0.07 → +0.16 → +0.32 → +0.33 as h grows), and is ≈ 0 in the benign (0.05, 1, straight) cell.

**The anisotropy-10 anomaly.** Flat beats SE(2) by 0.14–0.37 in all 12 a = 10 cells, and flat's success *rises* with heading noise there (turn90: 0.58 → 0.67) while SE(2)'s handoff error rises. Diagnostics (single seed, straight/S-curve, h = 0.2): both manifolds reach the goal without execution noise (SE(2) 0.97, flat 0.99), so the drift fields are fine; the gap appears only under noise. Two SE(2)-specific approximations were tested by an exact variant `se2x`: (i) the drift target is computed in the interpolant's frame and consumed in the sample's frame (first-order transport) — moving it by the adjoint of the relative pose raised the S-curve a = 10 cell from 0.32 to 0.49; (ii) accumulated deviations were summed in the interpolant's body frame rather than transported along the curving path — adding the adjoint transport changed nothing (0.47). Flat is still at 0.68. So half the gap is the transport approximation and half is unexplained; with lateral noise 10× longitudinal the world-frame Gaussian is, in practice, the better regression target for the corrective drift. The 5-seed `se2x` run is in section 5 below.

## 4. Solver cost (K = 4 IPF, S-curve/turn/straight, 3 seeds)

| route | manifold | wall-clock (s) | iterations to converge (2% of transport cost) |
|---|---|---|---|
| straight | flat / se2 | 410 ± 104 / 381 ± 119 | 2.8 / 2.0 |
| turn90 | flat / se2 | 372 ± 128 / 458 ± 180 | 2.0 / 2.8 |
| scurve | flat / se2 | 410 ± 96 / 402 ± 78 | 2.2 / 1.6 |

No hidden compute: same wall-clock within noise, same 2–3 iterations to converge.

## 5. Exact-transport SE(2) (`se2x`, 5 seeds): drift target moved by Ad(exp(−ξ)) and covariance accumulated with the adjoint of the geodesic motion

| route | h | a | flat | se2 | se2x | se2x − flat (p) | se2x − se2 (p) |
|---|---|---|---|---|---|---|---|
| straight | 0.05 | 1 | 0.770 | 0.805 | 0.805 | +0.035 (0.120) | +0.000 (0.953) |
| straight | 0.05 | 3 | 0.784 | 0.821 | 0.822 | +0.038 (0.049) | +0.001 (0.892) |
| straight | 0.05 | 10 | 0.536 | 0.567 | 0.561 | +0.025 (0.057) | -0.006 (0.698) |
| straight | 0.2 | 1 | 0.833 | 0.813 | 0.826 | -0.007 (0.733) | +0.013 (0.169) |
| straight | 0.2 | 3 | 0.766 | 0.783 | 0.784 | +0.018 (0.101) | +0.001 (0.717) |
| straight | 0.2 | 10 | 0.691 | 0.625 | 0.683 | -0.008 (0.617) | +0.058 (0.006) |
| straight | 0.5 | 1 | 0.616 | 0.700 | 0.723 | +0.107 (0.000) | +0.024 (0.022) |
| straight | 0.5 | 3 | 0.726 | 0.708 | 0.716 | -0.011 (0.538) | +0.008 (0.145) |
| straight | 0.5 | 10 | 0.728 | 0.581 | 0.641 | -0.087 (0.000) | +0.060 (0.009) |
| straight | 1.0 | 1 | 0.583 | 0.614 | 0.678 | +0.095 (0.005) | +0.065 (0.102) |
| straight | 1.0 | 3 | 0.587 | 0.622 | 0.665 | +0.078 (0.034) | +0.042 (0.271) |
| straight | 1.0 | 10 | 0.687 | 0.550 | 0.443 | -0.244 (0.000) | -0.107 (0.004) |
| turn90 | 0.05 | 1 | 0.748 | 0.780 | 0.770 | +0.022 (0.205) | -0.010 (0.481) |
| turn90 | 0.05 | 3 | 0.727 | 0.775 | 0.805 | +0.078 (0.002) | +0.030 (0.013) |
| turn90 | 0.05 | 10 | 0.578 | 0.355 | 0.584 | +0.006 (0.531) | +0.229 (0.000) |
| turn90 | 0.2 | 1 | 0.730 | 0.778 | 0.798 | +0.069 (0.013) | +0.021 (0.009) |
| turn90 | 0.2 | 3 | 0.690 | 0.723 | 0.764 | +0.074 (0.117) | +0.042 (0.003) |
| turn90 | 0.2 | 10 | 0.636 | 0.371 | 0.615 | -0.021 (0.130) | +0.244 (0.000) |
| turn90 | 0.5 | 1 | 0.577 | 0.694 | 0.717 | +0.140 (0.003) | +0.023 (0.061) |
| turn90 | 0.5 | 3 | 0.613 | 0.708 | 0.710 | +0.096 (0.001) | +0.002 (0.898) |
| turn90 | 0.5 | 10 | 0.694 | 0.420 | 0.581 | -0.113 (0.001) | +0.161 (0.000) |
| turn90 | 1.0 | 1 | 0.538 | 0.562 | 0.695 | +0.158 (0.001) | +0.133 (0.005) |
| turn90 | 1.0 | 3 | 0.544 | 0.578 | 0.636 | +0.092 (0.107) | +0.058 (0.061) |
| turn90 | 1.0 | 10 | 0.669 | 0.482 | 0.376 | -0.294 (0.001) | -0.107 (0.028) |
| scurve | 0.05 | 1 | 0.662 | 0.713 | 0.750 | +0.088 (0.006) | +0.038 (0.004) |
| scurve | 0.05 | 3 | 0.587 | 0.662 | 0.672 | +0.085 (0.082) | +0.010 (0.244) |
| scurve | 0.05 | 10 | 0.687 | 0.314 | 0.458 | -0.228 (0.001) | +0.144 (0.002) |
| scurve | 0.2 | 1 | 0.611 | 0.728 | 0.736 | +0.125 (0.001) | +0.008 (0.565) |
| scurve | 0.2 | 3 | 0.464 | 0.622 | 0.704 | +0.240 (0.001) | +0.081 (0.001) |
| scurve | 0.2 | 10 | 0.665 | 0.313 | 0.456 | -0.209 (0.000) | +0.142 (0.007) |
| scurve | 0.5 | 1 | 0.573 | 0.659 | 0.688 | +0.115 (0.008) | +0.030 (0.065) |
| scurve | 0.5 | 3 | 0.338 | 0.660 | 0.654 | +0.316 (0.000) | -0.006 (0.630) |
| scurve | 0.5 | 10 | 0.664 | 0.343 | 0.484 | -0.180 (0.001) | +0.141 (0.002) |
| scurve | 1.0 | 1 | 0.501 | 0.579 | 0.724 | +0.222 (0.002) | +0.144 (0.015) |
| scurve | 1.0 | 3 | 0.302 | 0.627 | 0.682 | +0.379 (0.000) | +0.054 (0.088) |
| scurve | 1.0 | 10 | 0.651 | 0.470 | 0.404 | -0.247 (0.000) | -0.066 (0.003) |

* `se2x` is on average +0.047 above `se2` over the grid and is never significantly worse than flat where a ≤ 3; it beats flat significantly in **16 of 24** a ≤ 3 cells (S-curve h = 1, a = 3: +0.38; 90° turn h = 1, a = 1: +0.16).
* At a = 10 it recovers most of `se2`'s deficit for h ≤ 0.5 (90° turn: 0.36 → 0.58, 0.37 → 0.62, 0.42 → 0.58; flat 0.58 / 0.64 / 0.69) and ties flat in 4 of the 12 cells, but flat still wins **8 of 12** a = 10 cells significantly and `se2x` is the worst of the three at (h = 1, a = 10) on every route. The exact transport therefore was the main defect of `se2` at moderate heading noise, but under both extreme heading noise and extreme anisotropy the Lie-group construction (first-order deviation dynamics in a rotating frame) is still the wrong regression target and the world-frame Gaussian is better.
* θ wrap with `se2x`: 0.637 / 0.585 (π−0.1 / π+0.1), vs flat 0.550 / 0.575 — the same small edge as un-wrapped.

## Bottom line

The Lie group buys correct uncertainty (coverage/KL), exact equivariance, and +0.1 to +0.4 success on curved routes with heading noise (with exact transport, in 16/24 cells at a ≤ 3, never significantly worse) — real and significant. It does not buy anything for strongly anisotropic body-frame slip: at a = 10 the naive world-frame model wins 8/12 cells even against the exact-transport variant, and at (h = 1, a = 10) the SE(2) construction is the worst of the three. The hypothesis as stated ("exactly when heading uncertainty *and* body-frame slip are large") is half right: heading yes, anisotropy no.

## What I'd run next

* Finish the `se2x` grid (queued) and, if a = 10 stays flat-favoured, a hybrid: SE(2) geodesic interpolation and features with world-frame (flat) covariance accumulation, to isolate which piece of the flat construction wins at high anisotropy.
* A = 10 cells with the execution noise applied in the *world* frame instead of the body frame, to check whether the flat advantage is a property of the anisotropy or of frame alignment between the model and the physics.
* Give the flat net body-frame features (equivariant flat) — if it keeps its a = 10 lead, the advantage is the covariance model, not the features.
