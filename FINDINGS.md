# Findings

Written after the full run (5 seeds, 200 episodes per cell, `results/summary.csv`).
Numbers are success rates, mean over seeds; std over seeds is 0.02–0.10.

## (i) Does learned ε beat the best fixed ε as disturbance grows?

**No.**  Condition C (learned ε on the IPF-refined bridge) never beats the
best fixed ε in any of the 12 cells, and the gap widens with disturbance:

| cell (w, σ_d) | best fixed ε | C | D |
|---|---|---|---|
| 0.10, 0.00 | 0.657 (ε=0.003) | 0.608 | 0.780 |
| 0.10, 0.05 | 0.437 (ε=0.003) | 0.398 | 0.485 |
| 0.10, 0.10 | 0.242 (ε=0.003) | 0.187 | 0.209 |
| 0.20, 0.00 | 0.695 (ε=0.001) | 0.651 | 0.827 |
| 0.20, 0.05 | 0.564 (ε=0.003) | 0.508 | 0.644 |
| 0.20, 0.10 | 0.408 (ε=0.01)  | 0.305 | 0.379 |
| 0.40, 0.00 | 0.754 (ε=0.001) | 0.705 | 0.819 |
| 0.40, 0.05 | 0.669 (ε=0.01)  | 0.604 | 0.685 |
| 0.40, 0.10 | 0.590 (ε=0.01)  | 0.434 | 0.491 |

At σ_d = 0 C is within one std of the best fixed ε; at σ_d = 0.10 it is
clearly below it (and below the oracle B).  The best fixed ε does shift
upward with disturbance for the wider gaps (0.001 → 0.01), which is the
effect the ε-policy was meant to exploit, but the policy did not learn it.

Condition D (learned ε on the iteration-0 bridge) beats every fixed-ε
setting for σ_d ≤ 0.05 and is roughly level with the best fixed ε at
σ_d = 0.10.  That advantage is **not** due to the learned ε — see (iii).

## (ii) Does the ε heatmap dip at the gap?

**No.**  In both C and D the learned mean ε sits at the floor of the allowed
range (≈ 1e-3) almost everywhere, for all three skills
(`figures/fig2_eps_heatmap_C.png`, `_D.png`).  The only structure is a mild
rise (to ≈ 3e-3) at the far right of skill 3, i.e. inside the goal region
where the episode is about to end and noise costs nothing.  There is no dip
at the gap because there is no rise anywhere else.

The reason is visible in the fixed-ε sweep: in this task extra bridge noise
has no upside.  Lower ε always means fewer collisions and a tighter endpoint,
and the drift at low ε already transports the mass; the −0.01‖u‖² effort
term is small compared to the ±1 outcome, so PPO correctly drives ε to its
minimum.  The oracle (ε = 0.03 outside the gap skill) is worse than fixed
ε = 0.003 in every cell for the same reason.  The premise that a high ε is
useful away from the gap does not hold for this reward.

## (iii) Does the flow-matching baseline (D) match condition C?

**No — D is better than C**, by 0.02–0.18 in every cell.  D is the
bridge-matching net at IPF iteration 0 (independent coupling, no refinement)
with its own PPO ε-policy; the only difference from C is the five IPF
refinement iterations.

An isolating check (`results/check_iter0_vs_ipf_fixed_eps.csv`, fixed
ε = 0.001, no policy) shows the same ordering without any RL involved:

| w, σ_d | IPF-refined nets | iteration-0 nets |
|---|---|---|
| 0.10, 0.00 | 0.643 | 0.775 |
| 0.20, 0.00 | 0.695 | 0.914 |
| 0.40, 0.00 | 0.754 | 0.908 |
| 0.20, 0.05 | 0.531 | 0.711 |

So D's advantage is entirely the un-refined drift; the learned ε on top of it
adds nothing (D at 0.827 ≤ iteration-0 nets at fixed ε = 0.001, 0.914, for
w = 0.20, σ_d = 0).  In this implementation the IPF refinement **hurts**.
The per-iteration diagnostics (`results/ipf.csv`, fig. 4 right panel) agree:
the endpoint W₂ and Sinkhorn cost do not decrease over the five iterations,
they fluctuate at the iteration-0 level, and the coupling cost E‖x₁−x₀‖²
decreases only slightly.  The likely mechanism is survivor bias from the
killed reference: each half-iteration fits a drift on trajectories that
survived the wall, so the backward-simulated start points no longer cover
ρ_{k−1} and the forward net is refit on a biased start distribution.  This
was not diagnosed further.

## Other observations

* **Plain PPO (E) fails completely**: 0.0 success in every cell.  With the
  sparse ±1 terminal reward and a per-step penalty it converges to standing
  still (effort ≈ 0.002, collision ≈ 0.001).  It serves as the sanity floor
  and nothing more; no reward shaping was added because the prompt fixed the
  reward.
* **Disturbance dominates at σ_d = 0.10**: a kick of std 0.10 at the pre-gap
  handoff against a gap half-width of 0.05 (w = 0.10) kills most episodes no
  matter what ε does; all bridge conditions collapse to 0.19–0.29 there.
* **Bridge sanity** (`results/verify.csv`): each skill alone reaches its
  target with W₂ = 0.02–0.04 for ε ∈ {0.003, 0.01, 0.03}; killed fractions
  behave as the geometry predicts (skill 2 at w = 0.10: 0.30 / 0.50 / 0.74).
  At w = 0.40 skill 2 kills ≈ 22 % of trajectories even at ε = 0.003 — these
  start with |y − 0.5| > 0.2 and the mean-field drift does not steer them into
  the gap before crossing.
* **Handoff error** (fig. 5) is 0.02–0.04 for every condition except fixed
  ε = 0.1 (0.12); C and D are not better than the good fixed settings.
* **Effort** (fig. 3): C and D use slightly *less* effort than fixed
  ε = 0.003 (0.15 vs 0.16) because ε at the floor needs fewer corrections;
  the oracle uses the most (0.26).  Learned ε does not move the Pareto front
  beyond what fixed ε = 0.001–0.003 already achieves.

## Bottom line

The ε lever, as posed, has no upside in this task: the ε-policy learns to
switch noise off.  The one clear effect in the data is that IPF refinement
with a killed reference degraded the drift; the un-refined bridge-matching
net is the best skill representation tested here.
