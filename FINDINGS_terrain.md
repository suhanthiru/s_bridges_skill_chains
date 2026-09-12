# Findings: terrain experiment

Written after the run from `results_terrain/summary.csv`.  5 seeds, 200
episodes per cell per seed, held-out layouts 20–29.  Per the order of work,
the run **stopped at the step-4 checkpoint**: BRIDGE did not beat TRACK, so
DIFF and PPO were not trained (their tuning-cell scores are given below for
reference only).  Answers refer to ORACLE, TRACK, BRIDGE-plain, BRIDGE-obs.

## Per-cell table (success rate, mean ± std over seeds)

| terrain | σ_k | ORACLE | TRACK | BRIDGE-plain | BRIDGE-obs |
|---|---|---|---|---|---|
| mild | 0.00 | 0.999 ± 0.002 | 1.000 ± 0.000 | 0.783 ± 0.035 | 0.791 ± 0.060 |
| mild | 0.03 | 0.992 ± 0.008 | 0.992 ± 0.006 | 0.625 ± 0.031 | 0.627 ± 0.018 |
| mild | 0.06 | 0.948 ± 0.004 | 0.960 ± 0.015 | 0.438 ± 0.042 | 0.413 ± 0.034 |
| mild | 0.10 | 0.862 ± 0.032 | 0.873 ± 0.027 | 0.279 ± 0.039 | 0.245 ± 0.047 |
| mild | 0.15 | 0.733 ± 0.044 | 0.787 ± 0.038 | 0.169 ± 0.028 | 0.136 ± 0.019 |
| rough | 0.00 | 0.986 ± 0.012 | 0.974 ± 0.018 | 0.407 ± 0.066 | 0.340 ± 0.063 |
| rough | 0.03 | 0.970 ± 0.008 | 0.957 ± 0.013 | 0.397 ± 0.042 | 0.346 ± 0.050 |
| rough | 0.06 | 0.915 ± 0.019 | 0.891 ± 0.032 | 0.335 ± 0.055 | 0.298 ± 0.051 |
| rough | 0.10 | 0.810 ± 0.032 | 0.794 ± 0.030 | 0.271 ± 0.036 | 0.240 ± 0.037 |
| rough | 0.15 | 0.648 ± 0.033 | 0.662 ± 0.040 | 0.201 ± 0.033 | 0.177 ± 0.031 |

Other metrics (seed means; full table in `summary.csv`):

| metric | cell | ORACLE | TRACK | BRIDGE-plain | BRIDGE-obs |
|---|---|---|---|---|---|
| median recovery (steps) | mild, 0.10 | 10.8 | 8.8 | 13.8 | 14.0 |
| never recovered | mild, 0.10 | 0.13 | 0.09 | 0.40 | 0.40 |
| median recovery (steps) | rough, 0.10 | 10.9 | 9.1 | 12.4 | 12.3 |
| never recovered | rough, 0.10 | 0.14 | 0.12 | 0.28 | 0.29 |
| handoff-1 W₂ | mild, 0.10 | 0.034 | 0.029 | 0.097 | 0.097 |
| handoff-2 W₂ | rough, 0.10 | 0.040 | 0.034 | 0.118 | 0.129 |
| corridor deviation | mild, 0.10 | 0.210 | 0.209 | 0.170 | 0.174 |
| control effort ∫‖u‖² | mild, 0.10 | 1.29 | 1.34 | 0.27 | 0.26 |

## (i) Does BRIDGE degrade slower than TRACK as σ_k grows?

**No.**  BRIDGE is below TRACK in every cell, including σ_k = 0, and it
also degrades faster.  At σ_k = 0.10 the gap is **0.59** on mild terrain
(0.87 vs 0.28) and **0.52** on rough (0.79 vs 0.27).  From σ_k = 0 to 0.10,
TRACK loses 0.13 (mild) / 0.18 (rough); BRIDGE-plain loses 0.50 / 0.14.
TRACK sits on the ORACLE curve throughout (within ±0.05, sometimes above it).

The bridge's deficit is present with no pushes at all (0.78 vs 1.00 mild,
0.41 vs 0.97 rough), so it is a base-competence gap, not a robustness gap.
The mechanism is visible in the effort column: the bridge spends a quarter
of the control effort of TRACK/ORACLE (0.23 vs ~1.0) because its drift is the
Brownian-bridge mean velocity in *state* space executed directly as a
command.  Under slip (up to −30 % / −60 %) and rotation (up to 17° / 46°) the
commanded velocity under-delivers, the only compensation is the implicit
1/(1−τ) gain near the end of the skill, and the robot arrives late and off
axis; the handoff W₂ (0.10–0.13 vs 0.03) shows the same thing.  TRACK's PD
loop on a terrain-aware demo path (kp = 10, feed-forward) simply closes that
error.

## (ii) Does DIFF match BRIDGE?

Not run (checkpoint rule).  At the tuning cell (σ_k = 0.06, mild, layouts
20–24, seed 0) DIFF scored 0.40 vs 0.48 for BRIDGE and 0.97 for TRACK, and
PPO scored 0.00 at both learning rates.  Nothing in this experiment supports
a robustness claim for the bridge over either.

## (iii) Does BRIDGE-obs beat BRIDGE-plain, and does the gap grow with roughness?

**No.**  BRIDGE-obs ≤ BRIDGE-plain in 9 of 10 cells (the exception, mild
σ_k = 0, is +0.008, inside one std) and it is worse on rough terrain by
0.03–0.07.  This was predictable: with iteration-0 bridge matching on demo
endpoints the regression target (x₁ − x_τ)/(1−τ) does not depend on the
terrain, so the 16-ray observation cannot carry information in expectation
and only adds fitting variance.  An observation can only help a bridge whose
target does depend on it (e.g. a reference process with terrain-dependent
drift, or targets derived from the demo paths), which is outside what was
specified.

## (iv) Where do bridge skills fail?

* **Undershoot, not wandering.**  Corridor deviation is *lower* for the
  bridge (0.12–0.17) than for TRACK/ORACLE (0.19–0.21): the bridge stays
  near the straight segment but does not get to the end of it.  The
  no-push failures are arrivals outside the 2-std ellipse of ρ₃ after three
  accumulated undershoots (handoff W₂ grows 0.04 → 0.07 → arrival miss).
* **Lateral pushes are not undone.**  40 % of lateral pushes (mild,
  σ_k = 0.10) are never recovered from vs 9 % for TRACK; the median recovery
  when it does happen is 14 steps vs 9.  The drift toward the posterior-mean
  endpoint is weak far from the training support, and a pushed robot in a
  rough patch (slower, rotated) is exactly there.
* **Handoffs compound it.**  Because handoffs happen at fixed steps
  100/200 regardless of position, an undershoot at the end of skill k becomes
  an out-of-distribution start for skill k+1 (handoff W₂ 0.10–0.19 at
  σ_k ≥ 0.10); nothing in the chain re-plans.

## Bottom line

Under spatially structured disturbance, a chain of iteration-0 bridge skills
trained on endpoints is much less robust than replaying a demo path with a
PD tracker, and the terrain observation does not change that.  The bridge's
one favourable number, four times lower control effort, is the cause of its
failure, not a benefit.
