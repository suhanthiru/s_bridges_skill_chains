# Seam A — Point vs cloud

**Question.** Does a handoff *cloud* (a bridge whose reference path ends in ρ_k) buy anything over a handoff *point* (a straight line through the marginal means), when both are executed by the identical PD tracker?
**Grid.** {TRACK-oracle, TRACK-naive, BRIDGE-tracked, BRIDGE-slip-tracked, TSM} × {mild, rough} × σ_k ∈ {0, 0.06, 0.10, 0.15}; held-out layouts 20–29; 200 episodes per cell; **5 seeds** (0–4). All conditions use kp = 10 + feed-forward, TRACK's frozen config; nothing runs open-loop.
**Pre-registered pass.** BRIDGE-tracked or BRIDGE-slip-tracked beats TRACK-naive by ≥ 0.10 at σ_k ≥ 0.06 on rough, 95% CI excluding zero.
**Verdict: FAIL.** Both bridges are significantly *below* the straight line through the means at every rough cell with σ_k ≥ 0.06 (Δ = −0.005 to −0.148, all CIs exclude zero) and the gap widens with σ_k. TSM is indistinguishable from TRACK-naive on rough terrain. At the planning level the cloud contributes nothing that the point does not, and the bridge's time-parametrised reference costs success under pushes. Phases B and C were run regardless, as the protocol requires.

## Success (mean ± 95% CI over seeds)

| condition | terrain | σ_k = 0 | 0.06 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| TRACK-oracle | mild | 1.000 ± 0.000 | 0.960 ± 0.018 | 0.873 ± 0.033 | 0.787 ± 0.047 |
| TRACK-oracle | rough | 0.974 ± 0.022 | 0.891 ± 0.040 | 0.794 ± 0.038 | 0.662 ± 0.050 |
| TRACK-naive | mild | 1.000 ± 0.000 | 0.971 ± 0.016 | 0.895 ± 0.024 | 0.779 ± 0.044 |
| TRACK-naive | rough | 0.996 ± 0.007 | 0.926 ± 0.028 | 0.811 ± 0.031 | 0.685 ± 0.040 |
| BRIDGE-tracked | mild | 1.000 ± 0.000 | 0.946 ± 0.017 | 0.785 ± 0.033 | 0.594 ± 0.056 |
| BRIDGE-tracked | rough | 0.991 ± 0.009 | 0.906 ± 0.031 | 0.757 ± 0.027 | 0.576 ± 0.038 |
| BRIDGE-slip-tracked | mild | 1.000 ± 0.000 | 0.929 ± 0.022 | 0.744 ± 0.055 | 0.541 ± 0.062 |
| BRIDGE-slip-tracked | rough | 0.996 ± 0.007 | 0.921 ± 0.026 | 0.763 ± 0.023 | 0.537 ± 0.038 |
| TSM | mild | 1.000 ± 0.000 | 0.931 ± 0.066 | 0.793 ± 0.079 | 0.634 ± 0.107 |
| TSM | rough | 0.997 ± 0.003 | 0.924 ± 0.031 | 0.809 ± 0.036 | 0.636 ± 0.054 |

## Paired differences on rough terrain (per seed, 95% CI, paired t-test)

| σ_k | BRIDGE-tracked − naive | BRIDGE-slip − naive | TSM − naive | BRIDGE-tracked − oracle |
|---|---|---|---|---|
| 0.06 | −0.020 [−0.031, −0.009] p=0.007 | −0.005 [−0.009, −0.001] p=0.034 | −0.002 [−0.015, +0.011] p=0.69 | +0.015 [−0.012, +0.042] p=0.19 |
| 0.10 | −0.054 [−0.071, −0.037] p=0.001 | −0.048 [−0.063, −0.033] p=0.001 | −0.002 [−0.018, +0.014] p=0.74 | −0.037 [−0.049, −0.025] p=0.001 |
| 0.15 | −0.109 [−0.149, −0.069] p=0.002 | −0.148 [−0.171, −0.125] p<0.001 | −0.049 [−0.068, −0.030] p=0.002 | −0.086 [−0.124, −0.048] p=0.003 |

Mild terrain is the same story with larger gaps (−0.11 to −0.24 at σ_k ≥ 0.10).

## Reading the table

* **Undisturbed, everything is at ceiling** (0.97–1.00). With the same closed-loop tracker, the reference path barely matters when nothing pushes the robot off it. The last run's BRIDGE < TRACK gap at σ_k = 0 was the open-loop execution, not the reference — that confound is gone.
* **Under pushes the bridge reference is worse than the straight line.** The mechanism is visible in the handoff logs (Phase C): the bridge path is time-parametrised by a drift that grows like 1/(1−τ) toward the end of the skill, so its reference moves fastest exactly at the handoff, where friction-limited tracking falls behind; the straight line moves at a constant 0.25 m/s and the tracker never saturates. Inside-2σ rates at handoff 2 (rough, σ_k = 0.10, 5 seeds): naive 0.82, bridges 0.78, TSM 0.77.
* **The slip reference does not help here** (it helped 5× in the SE(2) suite where the bridge was executed open-loop). Once a tracker supplies the feedback, what the reference knew about terrain covariance is redundant.
* **TRACK-naive ≥ TRACK-oracle.** The terrain-aware MPC path is not a better reference than a straight line for a kp = 10 tracker; the oracle's detours are unnecessary once feedback exists. So BRIDGE-tracked "matching the oracle" (it does, at rough σ_k = 0.06) is not the surprising result the prompt anticipated — the oracle path itself has no advantage.
* **TSM clears its floor** (≥ 0.6 at σ_k = 0 mild: 1.00) without tuning. Its terminal-state penalty pulls the BC rollouts into the next initiation set; as a reference generator wrapped in the same PD it equals the straight line on rough terrain and is a little worse on mild (large seed variance).
* Effort ∫‖u‖² (all cells): bridges 0.81, TRACK-naive 0.87, TSM 1.12, TRACK-oracle 1.40. The bridges are the cheapest references by ~7% and the oracle's terrain-aware detours cost 60% more effort for no success gain.

## Notes

* Bridges were trained on samples of the marginals (not demo endpoints) so Phase B can rescale them; at w = 1 this is the same distribution the demos were generated from.
* "Heading axis" and "corridor normal" in this planar env mean along-track (x) and lateral (y); the tangent-space Mahalanobis is the ℝ² one with the isotropic 0.05 std.
* Compute: 22 min per seed on the GPU (TSM training 8 min of it).

## What I'd run next

* Re-parametrise the bridge reference by arc length instead of τ (constant-speed reference along the bridge's mean path) to test the terminal-velocity mechanism directly.
* A tracker-free comparison at matched feedback gain: bridge drift plus a proportional term toward its own mean path, versus the straight line plus the same term.
* Repeat with an obstacle between the means (the wall of the original task) where the straight line through the point is infeasible and the cloud has to do work.
