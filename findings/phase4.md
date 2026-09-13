# Phase 4 — Handoff marginals and bimodality

**Hypothesis.** The shared marginal is the right place to put robustness, not the noise.
**Setup.** Layout L2 (wall with two gaps, pile beside the upper route), `slip` reference on SE(2), iteration-0 bridges, **noisy terrain map** for the bridges (class boundaries jittered by 0.02, 10% class flips; dynamics use the true map), `slip` disturbance. 500 episodes per cell, **5 seeds**.
**Verdict: FAIL on both halves.**
* 4a — success is **monotone** in the handoff width, not unimodal: wider ρ₂ helps at every disturbance level and the argmax sits at the largest width tested (w = 4) for 1× and 2× slip. The width is a knob, but not a tuned one — "as wide as the geometry allows" is the answer, and the optimum did not move to an interior value.
* 4b — the multi-marginal bridge with a bimodal ρ₂ is **worse** than two unimodal bridges plus a runtime gap choice (0.29 vs 0.39 success, collision 0.61 vs 0.45), and its gap assignment does **not** correlate with the lower-slip route (Pearson r = −0.31, p = 0.39 over 10 maps; the sign is wrong). It splits mass 60/40 between the gaps regardless of the map. The bridge does not pick the lower-slip gap without being told to.

## 4a — width sweep (ρ₂ covariance × w, skills 2 and 3 retrained per w)

| w | slip × 0.5 | slip × 1 | slip × 2 | handoff-2 Mahalanobis w.r.t. the *nominal* ρ₂ |
|---|---|---|---|---|
| 0.5 | 0.331 ± 0.101 | 0.138 ± 0.034 | 0.028 ± 0.008 | 1.79 |
| 1.0 | 0.374 ± 0.112 | 0.162 ± 0.030 | 0.033 ± 0.014 | 1.97 |
| 2.0 | 0.426 ± 0.044 | 0.189 ± 0.035 | 0.040 ± 0.010 | 2.28 |
| 4.0 | 0.426 ± 0.071 | 0.202 ± 0.029 | 0.049 ± 0.008 | 3.30 |

| slip level | argmax_w [bootstrap 95% CI over seeds] | success at argmax |
|---|---|---|
| × 0.5 | 2.0 [1.0, 4.0] | 0.426 |
| × 1 | 4.0 [2.0, 4.0] | 0.202 |
| × 2 | 4.0 [4.0, 4.0] | 0.049 |

Reading: a wider ρ₂ loosens the handoff (the delivered state is 3.3 nominal σ from the nominal mean at w = 4) and still raises success, because skill 3 is retrained from the wider cloud and tolerates it. The pre-gap skill is the bottleneck on L2 — collisions dominate (see 4b) — and a wider target lets skill 2 aim for a larger region behind the gap. The argmax does drift upward with disturbance (2 → 4 → 4), consistent in direction with the hypothesis, but since it is pinned at the boundary the "interior optimum that moves" claim is not supported. Absolute success is low on L2 with a corrupted map (0.03–0.43): the diagonal route through a 0.16-wide gap with an independent coupling collides in 45–60% of episodes.

## 4b — bimodal ρ₂ (one mode behind each gap), 10 terrain maps × 5 seeds

| method | success | collision | P(upper gap) | handoff-2 inside either mode | W₂ to goal |
|---|---|---|---|---|---|
| multi-marginal bridge (ρ₁ → mixture, mixture → ρ₃) | 0.293 ± 0.024 | 0.611 | 0.60 | 0.25 | 0.283 |
| two unimodal bridges + choose the lower-slip gap | 0.390 ± 0.015 | 0.447 | 0.45 | 0.41 | 0.237 |

Gap assignment vs terrain: Pearson r(P(upper), E[slip] on the lower route − E[slip] on the upper route) = **−0.305 (p = 0.39)** for the multi-marginal bridge. The two-bridge chooser picks the upper gap on 45% of maps by construction (it compares the same expected-slip numbers), which is what an informed rule looks like; the bridge's 0.60 is the mixture weight plus the pile's asymmetry, not terrain.

Why the multi-marginal bridge is worse: with an independent coupling the drift at ρ₁ is the average of the pulls toward the two modes, so trajectories start toward the wall between the gaps and only commit late; that costs collisions (0.61 vs 0.45) and lands only 25% of the survivors inside either mode. A runtime discrete choice followed by a unimodal bridge commits from the first step.

## Notes

* Bridges see the corrupted map (their Σ(x) quadrature and terrain features use it); the dynamics use the true map. This is the map-noise condition the prompt asked for; the `slip` reference's advantage from Phase 1 (+0.55 on L1 with the true map) does not carry to L2 with a noisy map — L2's failure mode is collision at the gap, which the reference covariance does not address.
* Marginal geometry: ρ₁ centred before the wall at (0.42, 0.5), modes behind the gaps at (0.58, 0.38) and (0.58, 0.62), gaps of width 0.16 at y = 0.38 / 0.62, pile at (0.70, 0.66, r = 0.05) beside the upper route.
* Compute: 7.5 min per seed on CPU (8 workers).

## What I'd run next

* 4a with widths beyond 4 and with ρ₁ widened too, to find where "wider is better" stops (presumably when skill 3 can no longer reach ρ₃ from the tails).
* 4b with the killed reference (reject wall-crossing pairs) so the multi-marginal coupling cannot average across the wall; that is the version of the experiment where a bimodal bridge could plausibly pick a gap.
* Replace the informed chooser with the *bridge's own* transport cost per gap (solve two bridges, compare E‖log(x₀⁻¹x₁)‖²) — a bridge-native gap selection that does not need the terrain rule.
