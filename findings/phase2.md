# Phase 2 — Terrain robustness

**Hypothesis.** Bridge skills degrade more gracefully than trackers and single-shot policies under structured disturbance.
**Grid.** method ∈ {nominal, ppo, diffusion, bridge_iter0, bridge_ipf (K = 5)} × disturbance ∈ {none, slip, rain, push} × layout ∈ {L1, L2}, plus the slip-magnitude sweep {0.5, 1, 2, 4}× nominal on L1. Bridges: `slip` reference on SE(2) (chosen before Phase 1b; on L1 terrain Phase 1 found flat ≈ SE(2)). Same demo set (500 nominal-PD rollouts per layout, clean commands as labels) for diffusion and Phase 5; same seeds; 500 episodes per cell; **5 seeds**.
**Pre-registered pass.** bridge success − diffusion success is positive and grows with disturbance severity.
**Verdict: half PASS.** The difference is positive and significant under every disturbance on both layouts (L1 slip +0.27, L2 slip +0.33, rain +0.11/+0.12, push +0.10/+0.07 for bridge_iter0; all p ≤ 0.015) and zero without disturbance on L1 — the bridge is more robust than the diffusion policy trained on the same demos. It does **not** keep growing with severity: over the slip sweep the gap is +0.39 → +0.27 → +0.04 → +0.00 at 0.5× → 4×, because both methods collapse past 2× nominal slip. Nominal PD tracking beats every learned method under every disturbance (L1 slip 0.81 vs 0.72), so "more graceful than trackers" fails outright.

## Main grid (success, mean ± 95% CI over seeds)

| layout | method | none | slip | rain | push | slip × 0.5 | × 1 | × 2 | × 4 |
|---|---|---|---|---|---|---|---|---|---|
| L1 | nominal | 0.996 ± 0.006 | **0.814 ± 0.027** | **0.441 ± 0.026** | **0.689 ± 0.021** | 0.930 ± 0.013 | 0.814 ± 0.027 | **0.540 ± 0.037** | **0.031 ± 0.010** |
| L1 | ppo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 | diffusion | 0.993 ± 0.006 | 0.446 ± 0.016 | 0.203 ± 0.015 | 0.240 ± 0.036 | 0.552 ± 0.060 | 0.446 ± 0.027 | 0.133 ± 0.032 | 0.002 ± 0.002 |
| L1 | bridge_iter0 | 0.993 ± 0.006 | 0.719 ± 0.042 | 0.312 ± 0.027 | 0.342 ± 0.025 | **0.942 ± 0.017** | 0.719 ± 0.042 | 0.168 ± 0.022 | 0.004 ± 0.002 |
| L1 | bridge_ipf | 0.991 ± 0.006 | 0.658 ± 0.076 | 0.275 ± 0.037 | 0.296 ± 0.036 | 0.901 ± 0.040 | 0.658 ± 0.076 | 0.139 ± 0.034 | 0.001 ± 0.001 |
| L2 | nominal | 0.996 ± 0.006 | **0.630 ± 0.025** | **0.318 ± 0.030** | **0.180 ± 0.020** | – | – | – | – |
| L2 | ppo | 0.000 | 0.000 | 0.000 | 0.000 | – | – | – | – |
| L2 | diffusion | 0.816 ± 0.265 | 0.208 ± 0.092 | 0.088 ± 0.035 | 0.085 ± 0.047 | – | – | – | – |
| L2 | bridge_iter0 | 0.980 ± 0.016 | 0.535 ± 0.054 | 0.208 ± 0.041 | 0.152 ± 0.010 | – | – | – | – |
| L2 | bridge_ipf | 0.982 ± 0.012 | 0.493 ± 0.060 | 0.186 ± 0.032 | 0.132 ± 0.024 | – | – | – | – |

## Bridge − diffusion, paired over seeds

| layout | disturbance | bridge_iter0 − diffusion [95% CI] | p | bridge_ipf − diffusion [95% CI] | p |
|---|---|---|---|---|---|
| L1 | none | +0.000 [−0.001, +0.002] | 0.37 | −0.002 [−0.006, +0.002] | 0.34 |
| L1 | slip | **+0.273 [+0.227, +0.319]** | <0.001 | +0.213 [+0.133, +0.293] | 0.002 |
| L1 | rain | **+0.109 [+0.093, +0.125]** | <0.001 | +0.072 [+0.036, +0.108] | 0.005 |
| L1 | push | **+0.102 [+0.076, +0.128]** | <0.001 | +0.056 [+0.021, +0.091] | 0.011 |
| L2 | none | +0.164 [−0.094, +0.422] | 0.15 | +0.166 [−0.097, +0.429] | 0.16 |
| L2 | slip | **+0.327 [+0.243, +0.411]** | <0.001 | +0.284 [+0.180, +0.388] | 0.002 |
| L2 | rain | **+0.120 [+0.069, +0.171]** | 0.003 | +0.098 [+0.050, +0.145] | 0.005 |
| L2 | push | **+0.066 [+0.021, +0.112]** | 0.015 | +0.046 [+0.006, +0.087] | 0.034 |

Slip sweep, bridge_iter0 − diffusion: +0.39 (0.5×), +0.27 (1×), +0.035 (2×), +0.002 (4×). Nominal − bridge_iter0: −0.01, +0.10, +0.37, +0.03.

## Reading the table

* **Bridge vs diffusion.** Same demos, same seeds: the bridge chain is 1.5–2.5× more successful under every disturbance. The diffusion policy imitates the demonstrator's commands and has no notion of a target set; the bridge's drift is a feedback field toward ρ_k that keeps pulling after a push. Undisturbed they tie (0.993 vs 0.993 on L1); on L2 the diffusion policy has one bad seed (0.816 ± 0.265), the bridges do not.
* **Not more graceful than the tracker.** Nominal PD (kp = 6 on the geodesic through the means) is above the bridges everywhere except at 0.5× slip (0.930 vs 0.942, within CI) and degrades more slowly (at 2× slip: 0.54 vs 0.17). This repeats Phase 1 and both earlier runs.
* **Severity.** The bridge's advantage over diffusion is largest at mild disturbance and vanishes at 2–4× nominal slip, where the process noise per step exceeds what any of the learned controllers can correct (nominal PD is at 0.03 at 4×).
* **PPO** is exactly 0 in every cell: 1M steps (compute cap) of a Gaussian policy with the sparse ±1 terminal reward and a step penalty converges to standing still, as in the first experiment; it stays in the table as the floor.
* **IPF** costs 0.03–0.06 under disturbance on both layouts (bridge_ipf < bridge_iter0 in 6/6 disturbed cells), the same sign and size as Phase 1's paired effect.
* **Layouts.** L2's diagonal route through a 0.16-wide gap is much harder for everything (nominal 0.63 vs 0.81 under slip); push is worst on L2 (0.13–0.18) because the unmodelled field pushes into the wall.

## Notes

* Diffusion baseline: chunk 8, execute 4, DDPM-50 train / DDIM-10 test, **β up to 0.2** so that ᾱ_T ≈ 0.007. The first pass used the 1000-step DDPM β_max = 0.02 (ᾱ_T = 0.6) and demo actions recovered from noisy poses; that version scored 0.02 undisturbed. Both fixes were validated before the seeds were re-run (4k-step probe: 0.96 undisturbed), all five seeds use the corrected version, and the earlier terrain-run DIFF (Experiment 2) shares the schedule defect.
* Trajectories, D(x,t) and outcomes of every bridge episode are saved (`results/phase2_traj_*.npz`) for Phase 3.
* Compute: 27–48 min per seed (bridge training on a CPU shared with Phase 1b/4; PPO 35 s on the GPU; diffusion ~80 s), plus 5 min per seed for the diffusion re-runs.

## What I'd run next

* Bridge drift + PD correction toward its own mean path (the seam suite's BRIDGE-tracked), evaluated on this SE(2) task, to see whether the bridge's robustness advantage over diffusion survives once both are wrapped in a tracker.
* PPO with a shaped reward (distance-to-ρ_k) so the floor is a real policy rather than 0.
* Diffusion with the bridge's target-set information (condition on ρ_k's mean/cov) — the cheapest way to test whether the bridge's edge is the target set or the transport.
