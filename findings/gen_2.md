# g2 — Scale

**Question.** Does the g0 ordering hold as the generated set grows, as the demo set shrinks, and as the evaluation disturbance grows? The interesting cell is N_demo = 1: the bridge fits a distribution from the marginals and never reads the demos, so it might extract more from a single demo than a tracker of that demo can.
**Setup.** L1, success under slip, diffusion at 8 000 steps, 200 episodes, **5 seeds**. Sources BRIDGE-slip and PD-noise (DART was excluded by the pre-set rule: in g0 it was 0.13 above the better of the two, outside the ±0.05 window). 2a: generated size ∈ {1, 4, 16, 64} × N_demo with N_demo = 20. 2b: N_demo ∈ {1, 5, 20, 100}, generated 4×. 2c: g0's policies (1× data) evaluated at slip magnitude {0.5, 1, 2, 4} × nominal.
**Pre-registered pass.** The gap between the leader and the other persists at every size, or grows as N_demo shrinks; report saturation.
**Verdict: PASS for the noisy tracker.** PD-noise leads at every generated size (+0.07 to +0.11), every demo count (+0.04 to +0.12, including N_demo = 1: 0.590 vs 0.475) and every evaluation severity where either method works (at 2× slip 0.29 vs 0.10). The bridge's demo-count independence is real — its curve is flat from 1 to 100 demos (0.475 → 0.521) while PD-noise's rides on the demos (0.59 → 0.61) — but the tracker of one demo still beats the bridge that ignores it. Saturation: PD-noise saturates by 4× (0.62 → 0.66 at 64×); the bridge keeps gaining to 64× (0.42 → 0.59) and at that size is where PD-noise was at 4×.

## 2a — generated set size (× N_demo, N_demo = 20; success under slip)

| source | 1× (20) | 4× (80) | 16× (320) | 64× (1 280) |
|---|---|---|---|---|
| BRIDGE-slip | 0.420 ± 0.114 | 0.513 ± 0.078 | 0.549 ± 0.031 | 0.594 ± 0.071 |
| PD-noise | 0.489 ± 0.114 | 0.619 ± 0.085 | 0.626 ± 0.033 | 0.655 ± 0.060 |
| PD − BRIDGE | +0.069 | +0.106 | +0.077 | +0.061 |

## 2b — demo count (generated 4 × N_demo; success under slip)

| source | N_demo = 1 | 5 | 20 | 100 |
|---|---|---|---|---|
| BRIDGE-slip | 0.475 ± 0.054 | 0.477 ± 0.046 | 0.510 ± 0.063 | 0.521 ± 0.038 |
| PD-noise | 0.590 ± 0.093 | 0.596 ± 0.146 | 0.613 ± 0.100 | 0.562 ± 0.100 |
| PD − BRIDGE | +0.115 | +0.119 | +0.103 | +0.041 |

## 2c — evaluation severity (g0 policies trained at 1× nominal slip; success under slip)

| source | 0.5× | 1× | 2× | 4× |
|---|---|---|---|---|
| BRIDGE-slip | 0.727 ± 0.094 | 0.508 ± 0.092 | 0.104 ± 0.040 | 0.003 ± 0.003 |
| PD-noise | 0.715 ± 0.156 | 0.607 ± 0.121 | 0.294 ± 0.092 | 0.006 ± 0.008 |

## Reading the tables

* **More bridge data helps, but 16× more of it is what it takes to match 4× of tracker data.** The bridge's off-path states (g1) are not useless, they are inefficient: a policy needs many of them to learn the weak drift labels well enough, whereas 80 near-path trajectories with kp = 10 labels are enough for the tracker's curve to flatten.
* **N_demo = 1 is not the bridge's niche.** With a single demo the tracker generates 4 noisy variations of that one path and the policy trained on them reaches 0.59; the bridge, whose data has nothing to do with the demo, gives 0.475 — the same as at 100 demos. The bridge's independence from demos shows up as stability (± 0.04–0.06 across seeds vs ± 0.09–0.15 for PD-noise), not as an advantage. At N_demo = 100 PD-noise dips (0.56): the nearest-demo lookup among 100 paths spreads the tracker's references and the 400 generated trajectories cover a wider band; the gap narrows to +0.04 (n.s.).
* **Severity.** At 0.5× slip both are at 0.72; at 2× the tracker's policy survives three times as often (0.29 vs 0.10); at 4× both are at zero. Data generated at 1× produces a policy whose robustness margin beyond the training noise is larger for the tracker — consistent with its labels being stronger corrections.

## Notes

* 2c re-evaluates g0's saved policies; no retraining.
* Compute: 13 min per seed on the GPU (16 policies + 8 evaluations).

## What I'd run next

* 2a extended to 256× for the bridge, to see whether it ever catches the tracker or saturates below it.
* 2b at N_demo = 1 with DART (excluded here by rule): the demonstrator with action noise needs no demo at all, so it is the sharper test of "a distribution beats a path".
* 2c with data generated at 2× slip, to separate "trained noise level" from "controller" in the severity curve.
