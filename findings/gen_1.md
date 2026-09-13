# g1 — What in the bridge is doing the work

**Question.** With the bridge already beaten by a noisy tracker (g0), which of its components carry any weight: the reference covariance, its action labels, or the state coverage it produces?
**Setup.** L1, N_demo = 20, 80 generated trajectories, diffusion at 8 000 steps, 200 episodes per cell, **5 seeds**; BRIDGE-slip and PD-noise are g0's policies re-evaluated (identical data and seed). Sources: BRIDGE-{slip, brownian, unicycle}, PD-noise, PD-iso (isotropic noise with the same mean total variance as the slip covariance along the bridge's states: σ² = 0.0086), MPC-relabel (BRIDGE-slip's states, every action replaced by the MPC oracle's).
**Pre-registered pass.** At least one isolation attributes ≥ 0.05 of the g0 gap to a bridge-specific component (the drift's action labels, or coverage PD-noise does not produce). Fail: the gap is fully explained by the noise model.
**Verdict: PASS on the letter, with the sign reversed.** There is a large bridge-specific component — the terrain-covariance-aware drift is worth +0.34 to +0.35 under slip relative to the same bridge with a Brownian or unicycle reference — and it is not a noise-model effect, because for the tracker the noise shape does not matter (PD-iso ≥ PD-noise). But that component is what keeps the bridge from being far worse; it does not produce a lead. The bridge's other distinctive property, wide off-path coverage, is *negatively* correlated with downstream success across sources (r = −0.82). The g0 gap is explained by the controller's labels off the path, not by the noise model.

## Success (L1, mean ± 95% CI over seeds)

| source | none | slip | rain | push |
|---|---|---|---|---|
| BRIDGE-slip | 0.993 ± 0.010 | 0.509 ± 0.084 | 0.191 ± 0.017 | 0.210 ± 0.016 |
| BRIDGE-brownian | 0.953 ± 0.032 | 0.167 ± 0.022 | 0.078 ± 0.014 | 0.036 ± 0.023 |
| BRIDGE-unicycle | 0.949 ± 0.025 | 0.158 ± 0.062 | 0.055 ± 0.024 | 0.040 ± 0.019 |
| PD-noise | 0.992 ± 0.007 | 0.619 ± 0.121 | 0.296 ± 0.034 | 0.505 ± 0.070 |
| PD-iso | 0.995 ± 0.006 | **0.708 ± 0.046** | **0.327 ± 0.023** | **0.536 ± 0.037** |
| MPC-relabel | 0.002 ± 0.003 | 0.198 ± 0.076 | 0.226 ± 0.052 | 0.069 ± 0.036 |

## The three isolations (paired over seeds, L1)

| isolation | comparison | slip | rain | push |
|---|---|---|---|---|
| reference covariance, in the bridge | BRIDGE-slip − BRIDGE-unicycle | **+0.351** [+0.260, +0.442] p<0.001 | +0.136 [+0.096, +0.176] p=0.001 | +0.170 [+0.157, +0.183] p<0.001 |
| | BRIDGE-slip − BRIDGE-brownian | **+0.342** [+0.273, +0.411] p<0.001 | +0.113 [+0.086, +0.140] p<0.001 | +0.174 [+0.158, +0.190] p<0.001 |
| noise model, in the tracker | PD-noise − PD-iso | −0.089 [−0.228, +0.050] p=0.15 | −0.031 [−0.076, +0.014] p=0.13 | −0.031 [−0.092, +0.030] p=0.23 |
| action labels | MPC-relabel − BRIDGE-slip | −0.311 [−0.455, −0.167] p=0.004 | +0.035 [−0.015, +0.085] p=0.12 | −0.141 [−0.170, −0.112] p<0.001 |

## Coverage (generated part, 80 trajectories; grid 32 × 32 × 8 headings, off-path = farther than 0.05 from any of the 500 demos' states)

| source | off-path cells | off-path fraction | mean displacement | success (slip) |
|---|---|---|---|---|
| BRIDGE-slip | 416 | 0.289 | 0.038 | 0.509 |
| BRIDGE-brownian | 393 | 0.321 | 0.041 | 0.167 |
| BRIDGE-unicycle | 375 | 0.299 | 0.039 | 0.158 |
| PD-noise | 44 | 0.004 | 0.007 | 0.619 |
| PD-iso | 11 | 0.001 | 0.006 | 0.708 |
| MPC-relabel | 422 | 0.311 | 0.040 | 0.198 |

Pearson r(off-path cells, success under slip) over the six sources: **−0.820 (p = 0.046)**.

## Reading the table

* **The reference covariance is a bridge-specific mechanism, and a big one.** The three bridges see the same noise draws and produce the same coverage (375–416 cells); only the drift target differs. With Σ(x) in the h-transform the drift steers harder where accumulated slip is larger, and the diffusion policy learns that; with a constant σ the labels are the plain transport velocity and the policy is barely better than the raw demos (0.16 vs 0.17). This is the Phase 1 result (5× as an executed controller) re-appearing as a data-generation result.
* **For the tracker the noise shape is irrelevant.** PD-iso, with isotropic noise of the same total variance, is at least as good as PD-noise everywhere; its labels are kp = 10 corrections whatever the perturbation looked like. So the covariance information reaches the policy only through labels that depend on it — the bridge's — and never through the states.
* **Coverage hurts.** The four sources that wander (bridges, MPC-relabel: ~30% of states off the demo manifold) all trail the two that hug the path (0.1–0.4% off). At this budget and policy class, 80 trajectories of which 25 are far off-path teach a worse policy than 80 near-path trajectories with strong corrective labels: the off-path states are where the bridge's labels are weakest (the drift's gain is 1/(1−τ), the tracker's is 10), so the policy learns weak corrections exactly where it needs strong ones.
* **MPC-relabel is not the upper bound it was meant to be, for a reason worth recording.** Relabelling every step of a bridge trajectory with the MPC's command gives (state, action) pairs that are individually oracle-good but *sequentially* inconsistent: the next state was produced by the bridge, not by the MPC's command, so an 8-step chunk of relabelled actions is not a trajectory the MPC would ever execute, and its warm-started plan drifts. A chunked policy trained on it reaches the goal 0.2% of the time undisturbed. The action-label isolation is therefore inconclusive as designed; the correct version relabels only the first action of each chunk (or uses a single-step policy), listed below.
* g0's remark that the bridge "covers 35% of states off-path vs 0.3% for PD-noise" (from the smoke run) is 29% vs 0.4% with the full sets.

## Notes

* Undisturbed, BRIDGE-brownian/unicycle are at 0.95 while BRIDGE-slip and the trackers are at 0.99: the constant-σ bridges' data already produce a slightly worse policy before any disturbance.
* Compute: 4.8 min per seed on the GPU (two sources reuse g0's policies).

## What I'd run next

* MPC-relabel with first-action-only relabelling (or a 1-step BC policy) so the "states vs labels" question gets a real answer.
* A bridge with kp-style gain: add a proportional term toward the bridge's own mean path to its labels, keeping its states, to test whether label strength alone closes the gap to PD-noise.
* PD-noise with the bridge's coverage forced (start the tracker from the bridge's off-path states) — the converse of relabelling.
