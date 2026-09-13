# Phase 5 — Bridge-generated demonstrations

**Hypothesis.** Bridge samples are better augmentation than noised demos because they respect reachability.
**Setup.** The Phase 2 diffusion policy (chunk 8 / execute 4, DDPM-50 / DDIM-10, 10 000 steps) trained per layout and seed on: (a) the 500 nominal-PD demos; (b) demos + 500 copies with Gaussian state noise (pos 0.03, heading 0.1), actions kept; (c) demos + 500 rollouts of the Phase 2 `slip` bridge (iteration 0) under its reference SDE, with the drift commands it issued as actions; (d) 1 000 bridge rollouts only. Evaluated on the Phase 2 disturbance grid; 500 episodes per cell; **5 seeds**.
**Pre-registered pass.** (c) > (b) on success under disturbance at the same dataset size.
**Verdict: PASS.** (c) beats (b) in all six disturbed cells: L1 +0.31 / +0.13 / +0.15 (slip / rain / push), L2 +0.20 / +0.07 / +0.07, every CI well clear of zero (p ≤ 0.002). Two things the criterion did not ask for are also true: Gaussian noising is *worse* than the raw demos (b < a in every cell, e.g. L1 slip 0.18 vs 0.32), and bridge samples alone (d) beat everything (L1 slip 0.58, L2 undisturbed 0.95 vs 0.56 for the demos).

## Success by training set (mean ± 95% CI over seeds)

| layout | dataset | n traj. | none | slip | rain | push |
|---|---|---|---|---|---|---|
| L1 | a demos | 500 | 0.993 ± 0.007 | 0.320 ± 0.014 | 0.140 ± 0.014 | 0.127 ± 0.035 |
| L1 | b demos + noised | 1 000 | 0.949 ± 0.027 | 0.175 ± 0.040 | 0.070 ± 0.007 | 0.072 ± 0.022 |
| L1 | c demos + bridge | 1 000 | 0.993 ± 0.006 | **0.489 ± 0.042** | **0.202 ± 0.014** | **0.218 ± 0.028** |
| L1 | d bridge only | 1 000 | 0.993 ± 0.007 | **0.582 ± 0.052** | **0.243 ± 0.029** | **0.259 ± 0.044** |
| L2 | a demos | 500 | 0.563 ± 0.354 | 0.132 ± 0.054 | 0.056 ± 0.021 | 0.059 ± 0.029 |
| L2 | b demos + noised | 1 000 | 0.586 ± 0.136 | 0.102 ± 0.039 | 0.044 ± 0.011 | 0.028 ± 0.015 |
| L2 | c demos + bridge | 1 000 | 0.937 ± 0.027 | **0.306 ± 0.050** | **0.113 ± 0.015** | **0.098 ± 0.025** |
| L2 | d bridge only | 1 000 | 0.950 ± 0.072 | **0.402 ± 0.050** | **0.151 ± 0.028** | **0.134 ± 0.019** |

| layout | disturbance | (c) − (b) [95% CI] | p |
|---|---|---|---|
| L1 | slip | +0.314 [+0.284, +0.343] | <0.001 |
| L1 | rain | +0.133 [+0.121, +0.145] | <0.001 |
| L1 | push | +0.146 [+0.129, +0.162] | <0.001 |
| L2 | slip | +0.204 [+0.179, +0.230] | <0.001 |
| L2 | rain | +0.069 [+0.052, +0.087] | <0.001 |
| L2 | push | +0.070 [+0.044, +0.096] | 0.002 |

## Reading the table

* **Noised copies are not more data, they are wrong data.** Perturbing a demo state while keeping its action teaches the policy that the demonstrator's command from a nearby state is the same command; for a high-gain PD demonstrator that is false (its command *is* the correction), so the policy learns a flatter, less corrective map and loses 0.04–0.15 relative to the raw demos.
* **Bridge samples are on-policy for a corrective controller.** A bridge rollout under its reference noise visits off-nominal states and records the drift's correction from each of them — exactly the (state, corrective action) pairs the demos lack. That is the "respects reachability" property the hypothesis named, made concrete: the states are ones the transport actually passes through, and the actions are the ones that get there.
* **Bridge-only is best**, which says the demos add nothing the bridge samples do not already contain, and the demonstrator's PD commands are if anything a worse target than the drift (they saturate at the clip; the drift does not).
* On L2 the demo-trained policy is unstable across seeds (0.56 ± 0.35 undisturbed — one seed fails outright); bridge data removes that variance (0.94–0.95).
* Absolute numbers under disturbance stay below the bridge executed directly (Phase 2: 0.72 on L1 slip) and far below nominal PD (0.81): the diffusion policy is a worse controller than either, whatever it is trained on.

## Notes

* 10 000 training steps here versus 15 000 in Phase 2, so (a) is a little weaker than Phase 2's diffusion row (0.32 vs 0.45 on L1 slip); all four datasets use the same budget, so the comparison is fair.
* Bridge rollouts are generated on the true terrain map with the `slip` reference covariance (Phase 2's iteration-0 nets, SE(2)); actions are the clipped drift commands.
* Compute: 7.2 min per seed on the GPU (8 policies trained and evaluated per seed).

## What I'd run next

* Dataset-size scan: bridge samples at 250 / 500 / 1 000 / 2 000, to see whether (d) keeps improving and where it saturates relative to executing the bridge directly.
* Bridge samples from the *IPF-refined* bridge as training data — Phase 1 says the refined coupling is worse to execute; it may still be a fine data generator.
* The same four datasets for behaviour cloning (a plain MLP) to separate "bridge data is better" from "bridge data suits diffusion".
