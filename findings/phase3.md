# Phase 3 — Bidirectionality as a recovery trigger

**Hypothesis.** The forward–backward drift disagreement D(x,t) = ‖b_fwd(x,t) + b_bwd(x,t)‖ spikes before failures and can be used as a replan signal.
**Data.** Every Phase 2 bridge episode (bridge_iter0 and bridge_ipf × {none, slip, rain, push} × {L1, L2} × 5 seeds × 500 episodes) with D logged per step. Offline: AUC of D as a failure predictor at horizons {5, 10, 20} (positives = steps of failing episodes ≥ H steps before the failure time; negatives = all steps of successful episodes), against distance-to-nominal-path and the one-step unicycle residual computed from the same trajectories. Closed loop: threshold on the τ-normalised D̃ = D·τ(1−τ) chosen on validation seeds 0–1 as the 90th percentile of the per-episode maximum over successful episodes (a 10% false-replan budget by construction; recall on failing validation episodes 0.40), tested on seeds 2–4 with at most one replan per skill; replan = restart the skill from the current state (for an iteration-0 bridge with independent coupling the re-solved bridge from a point *is* the learned drift restarted at τ = 0).
**Pre-registered pass.** AUC(D) > AUC(both baselines) at ≥ 10-step horizon, and the trigger raises success under push/rain without raising it under none.
**Verdict: FAIL on both parts.** D is at chance as a failure predictor under every disturbance (AUC 0.47–0.52), the distance-to-nominal baseline is better in every disturbed cell (0.52–0.59), and the closed-loop trigger moves success by −0.013 to +0.018 with CIs through zero everywhere. It is not "replanning constantly" — it barely replans at all under slip and rain (1–9%) — but where it does fire (push, 38–69%), 71–85% of the firings are in episodes that would have succeeded anyway.

## Offline: AUC (mean ± 95% CI over seeds; 0.5 = chance)

| method | disturbance | horizon | D | D·τ(1−τ) | dist-to-nominal | unicycle residual |
|---|---|---|---|---|---|---|
| bridge_iter0 | none | 10 | 0.631 ± 0.279 | 0.567 ± 0.211 | 0.649 ± 0.233 | 0.519 ± 0.020 |
| bridge_iter0 | slip | 5 | 0.516 ± 0.019 | 0.519 ± 0.016 | **0.566 ± 0.015** | 0.485 ± 0.004 |
| bridge_iter0 | slip | 10 | 0.510 ± 0.019 | 0.518 ± 0.017 | **0.561 ± 0.015** | 0.483 ± 0.004 |
| bridge_iter0 | slip | 20 | 0.500 ± 0.018 | 0.510 ± 0.017 | **0.553 ± 0.013** | 0.479 ± 0.003 |
| bridge_iter0 | rain | 10 | 0.507 ± 0.012 | 0.517 ± 0.010 | **0.533 ± 0.011** | 0.482 ± 0.004 |
| bridge_iter0 | push | 10 | 0.470 ± 0.026 | 0.482 ± 0.021 | **0.525 ± 0.015** | 0.467 ± 0.011 |
| bridge_ipf | none | 10 | 0.795 ± 0.145 | 0.708 ± 0.123 | 0.827 ± 0.135 | 0.539 ± 0.032 |
| bridge_ipf | slip | 10 | 0.517 ± 0.014 | 0.522 ± 0.013 | **0.585 ± 0.013** | 0.484 ± 0.003 |
| bridge_ipf | rain | 10 | 0.513 ± 0.010 | 0.521 ± 0.009 | **0.548 ± 0.011** | 0.482 ± 0.004 |
| bridge_ipf | push | 10 | 0.471 ± 0.021 | 0.481 ± 0.018 | **0.542 ± 0.010** | 0.462 ± 0.011 |

(all 24 cells × 4 signals in `findings/_phase3_tables.md`). Under `none` the AUCs look higher (0.63–0.83) but rest on < 1% failures per seed, hence the ± 0.2 CIs. The unicycle residual is slightly *below* chance everywhere (successful episodes are noisier per step than failing ones, which spend time stuck at the wall).

## Closed loop (test seeds 2–4, 500 episodes each)

| method | layout | disturbance | success, no trigger | with trigger | Δ (paired) | replan rate | false-replan rate |
|---|---|---|---|---|---|---|---|
| bridge_iter0 | L1 | none | 0.997 | 0.997 | +0.000 ± 0.000 | 0.00 | – |
| bridge_iter0 | L1 | slip | 0.721 | 0.724 | +0.003 ± 0.013 | 0.02 | 0.01 |
| bridge_iter0 | L1 | rain | 0.307 | 0.318 | +0.011 ± 0.025 | 0.09 | 0.05 |
| bridge_iter0 | L1 | push | 0.331 | 0.349 | +0.018 ± 0.067 | 0.69 | 0.85 |
| bridge_iter0 | L2 | slip | 0.543 | 0.545 | +0.002 ± 0.009 | 0.01 | 0.00 |
| bridge_iter0 | L2 | rain | 0.200 | 0.201 | +0.001 ± 0.008 | 0.03 | 0.02 |
| bridge_iter0 | L2 | push | 0.153 | 0.144 | −0.009 ± 0.039 | 0.38 | 0.71 |
| bridge_ipf | L1 | push | 0.291 | 0.291 | −0.001 ± 0.056 | 0.65 | 0.85 |
| bridge_ipf | L2 | push | 0.132 | 0.119 | −0.013 ± 0.037 | 0.39 | 0.73 |

(bridge_ipf's slip/rain rows are the same picture: Δ ≤ 0.007, replan rate ≤ 0.07.)

## Why it fails

* **D measures the wrong thing for these failures.** By construction D̃ ≈ ‖ξ‖/(τ(1−τ)) is the deviation from the bridge's *support*, and the support of an independent-coupling bridge between two broad Gaussians is wide: a robot can be far from where it should be for its start point and still be on the support of some other pair. The failures in this task are arrival misses and gap collisions driven by accumulated slip, which do not look like "off-support" to either net.
* **The push field is the only disturbance that produces large D̃**, because it drags the robot sideways; hence the trigger fires there — and mostly on episodes that were going to succeed, because the drift's own 1/(1−τ) gain already absorbs the push.
* **Restarting the skill clock does not help even when the trigger is right**: the skill gets 100 more steps, but the same drift from the same off-path state produces the same outcome; the replan is not a different plan.
* The nearest cheap alternative, distance to the nominal geodesic, is only marginally informative itself (0.52–0.59): failure here is largely not predictable from the state 10 steps ahead.

## Notes

* Threshold rule: an episode-level false-replan budget rather than Youden's J on pooled steps; the latter, tried in the smoke run, fires in ~97% of episodes because failure is common and D̃ is heavy-tailed. The budget rule is the pre-declared one for the reported numbers.
* An integer overflow (n₁·n₀ > 2³¹ with numpy's 32-bit default on Windows) corrupted the first AUC pass; fixed (`phase3.auc` uses floats) and recomputed from the same saved trajectories.
* Compute: 11 min on the GPU (offline part reads 80 trajectory files; closed loop = 32 cells × 2 runs × 500 episodes).

## What I'd run next

* A trigger on the *goal-conditional* residual instead: distance between the current state and the bridge's own predicted endpoint distribution (E[x₁ | x_t] from the forward net) — that is what "will I arrive" actually measures.
* Replan as a genuinely different plan: on trigger, re-solve with the killed reference or switch to the tracker, rather than restarting the same drift.
* D on the IPF-refined bridge with a *narrow* coupling (e.g. after killing wall-crossing pairs), where the support is tight enough for off-support to mean something.
