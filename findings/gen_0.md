# g0 — The kill test

**Question.** Does the Phase 5 result (bridge rollouts beat noised demos as diffusion-policy training data) need the bridge, or does any controller rolled out under the same noise do the same?
**Setup.** N_demo = 20 (first 20 of the Phase 5 clean demos), generated set 4 × 20 = 80 trajectories, diffusion policy at 8 000 steps for every source, L1 and L2, disturbances {none, slip, rain, push}, 200 episodes per cell, **5 seeds**. Every generator runs on the reference kinematics with the *same* standard-normal noise draws for a given seed (`tests_gen.py`), so PD-noise differs from BRIDGE-slip only in the controller.
**Pre-registered pass.** BRIDGE-slip − PD-noise ≥ +0.05 in at least three of the six disturbed cells, CI clear of zero.
**Verdict: FAIL — the bridge is unnecessary, and it is the weakest of the noisy controllers.** BRIDGE-slip is *below* PD-noise in all six disturbed cells (four significantly: L1 slip −0.11, rain −0.11, push −0.32; L2 push −0.07), below DART in all six (L1 slip −0.24, push −0.28; L2 slip −0.13), and below the MPC upper bound by 0.05–0.30. DART — the standard published augmentation — is the best source on L1 (0.745 under slip, tied with MPC-rollout 0.686 ± 0.03) and tied for best on L2. Everything downstream of this file is therefore about noisy-controller augmentation; the bridge is one such controller and a poor one.

## Success by training-data source (mean ± 95% CI over seeds)

| source | layout | none | slip | rain | push |
|---|---|---|---|---|---|
| DEMO | L1 | 0.894 ± 0.122 | 0.174 ± 0.064 | 0.081 ± 0.015 | 0.068 ± 0.019 |
| NOISED | L1 | 0.739 ± 0.210 | 0.100 ± 0.018 | 0.054 ± 0.024 | 0.028 ± 0.019 |
| BRIDGE-slip | L1 | 0.993 ± 0.012 | 0.505 ± 0.035 | 0.183 ± 0.049 | 0.196 ± 0.028 |
| PD-noise | L1 | 0.988 ± 0.014 | 0.618 ± 0.122 | 0.295 ± 0.035 | 0.513 ± 0.066 |
| DART | L1 | 0.994 ± 0.007 | **0.745 ± 0.047** | 0.331 ± 0.061 | 0.476 ± 0.057 |
| MPC-rollout | L1 | 0.990 ± 0.010 | 0.686 ± 0.031 | **0.350 ± 0.074** | 0.493 ± 0.038 |
| DEMO | L2 | 0.329 ± 0.117 | 0.059 ± 0.031 | 0.029 ± 0.016 | 0.026 ± 0.023 |
| NOISED | L2 | 0.357 ± 0.140 | 0.054 ± 0.043 | 0.033 ± 0.009 | 0.013 ± 0.010 |
| BRIDGE-slip | L2 | 0.944 ± 0.064 | 0.318 ± 0.038 | 0.112 ± 0.028 | 0.076 ± 0.031 |
| PD-noise | L2 | 0.697 ± 0.234 | 0.333 ± 0.148 | 0.151 ± 0.060 | 0.148 ± 0.041 |
| DART | L2 | 0.910 ± 0.137 | **0.444 ± 0.117** | 0.172 ± 0.058 | 0.152 ± 0.078 |
| MPC-rollout | L2 | 0.881 ± 0.107 | 0.440 ± 0.051 | **0.193 ± 0.063** | 0.125 ± 0.040 |

## Paired differences over seeds

| layout | disturbance | BRIDGE-slip − PD-noise | BRIDGE-slip − DART | MPC-rollout − BRIDGE-slip |
|---|---|---|---|---|
| L1 | slip | −0.113 [−0.221, −0.005] p=0.044 | −0.240 [−0.295, −0.185] p<0.001 | +0.181 [+0.158, +0.204] p<0.001 |
| L1 | rain | −0.112 [−0.166, −0.058] p=0.005 | −0.148 [−0.197, −0.099] p=0.001 | +0.167 [+0.091, +0.243] p=0.004 |
| L1 | push | −0.317 [−0.394, −0.240] p<0.001 | −0.280 [−0.356, −0.204] p=0.001 | +0.297 [+0.280, +0.314] p<0.001 |
| L2 | slip | −0.015 [−0.151, +0.121] p=0.77 | −0.126 [−0.228, −0.024] p=0.026 | +0.122 [+0.046, +0.198] p=0.011 |
| L2 | rain | −0.039 [−0.107, +0.029] p=0.19 | −0.060 [−0.132, +0.012] p=0.081 | +0.081 [+0.004, +0.158] p=0.043 |
| L2 | push | −0.072 [−0.122, −0.022] p=0.016 | −0.076 [−0.160, +0.008] p=0.066 | +0.049 [+0.000, +0.098] p=0.049 |

## Reading the table

* **Phase 5 replicates, and its control was missing.** Bridge rollouts still beat noised demos by a wide margin (L1 slip 0.505 vs 0.100; Phase 5 had 0.489 vs 0.175 with 500 demos). But with the same noise draws, a kp = 10 tracker of the demo path produces better training data than the bridge, the demonstrator with DART action noise produces better data still, and the MPC oracle sits with DART. The property that mattered in Phase 5 was "a corrective controller rolled out under realistic noise", not the bridge.
* **Why the bridge is the worst noisy controller.** Its states cover far more of the off-path space (g1 quantifies this: 29% of its states are off the demo manifold vs 0.4% for PD-noise), but its action labels there are the transport drift — an open-loop-ish velocity toward the goal marginal with the weak implicit 1/(1−τ) gain — and the downstream policy learns that weak correction. The trackers' labels off-path are strong corrections (kp = 10, or the MPC's re-plan), which is what a pushed robot needs. Push is where the gap is largest (L1: 0.196 vs 0.513 vs 0.476).
* **DART ≈ MPC on L1** means the demonstrator's own PD (kp = 6) with action noise is already near the ceiling of what this policy class extracts from 80 noisy trajectories; the upper bound is not far above the standard method.
* **Undisturbed, every noisy source is at 0.94–0.99** and DEMO/NOISED at 0.33–0.89: with 20 demos the plain demo set is not even enough to reproduce the demonstrator on L2, and Gaussian state-noising makes it worse (as in Phase 5).
* **Seed variance is large for PD-noise and DART on L2** (± 0.12–0.23): the 20-demo nearest-path lookup makes those sources sensitive to which 20 demos the seed picked; the bridge, which ignores the demos, is the most seed-stable source (± 0.04–0.06). That is the one property it has that the others do not, and g2b tests whether it matters when N_demo is small.

## Notes

* Datasets = 20 demos + 80 generated trajectories (DEMO: 20 demos only; NOISED: 20 + 80 noised copies), so every non-DEMO row has the same 100 trajectories of 300 steps.
* Compute: 10.4 min per seed on the GPU (6 sources × 2 layouts; MPC-rollout generation is ~1 min of it, bridge nets are Phase 2's).

## What I'd run next

* PD-noise and DART with the bridge's *states* (relabel the bridge rollouts with the tracker's command) — g1 does the MPC version; the tracker version is the cheapest way to confirm the "labels, not states" reading.
* The same six sources at N_demo = 1 (g2b) — the bridge's seed-stability advantage should show there if anywhere.
* DART with a kp = 10 demonstrator: if DART's lead over PD-noise is the demonstrator's smoother commands rather than the action-noise mechanism, this closes it.
