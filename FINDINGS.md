# Findings — Schrödinger-bridge skills on terrain

This file rolls up all four experiments in this repo. Experiments 1 and 2 are the earlier runs (learned ε; open-loop bridges vs a tracker), kept below unchanged. Experiments 3 (SE(2) terrain suite, six phases) and 4 (seam suite, `FINDINGS_seam.md`) are the two suites run last; each phase has its own file under `findings/` with tables, seeds and a verdict.

## Experiment 3 — SE(2) terrain suite

Chained SE(2) skills on a terrain map with slip, rain and pushes; reference SDEs `brownian` / `killed` / `unicycle` / `slip`; manifolds `flat` / `se2` (/ `se2x`); baselines nominal PD, PPO, diffusion, bridge at IPF 0 and 5. Gate: the IPF logic reproduces the closed-form Gaussian Schrödinger bridge to 3.5e-7 (`tests.py`), so no result below is a projection bug. 10 seeds in Phase 1, 5 elsewhere.

| phase | hypothesis | pre-registered pass | headline number | verdict |
|---|---|---|---|---|
| 1 | IPF only hurts because the reference is uninformative | slip/unicycle + IPF beats its own iter-0 and beats brownian iter-0 | `slip` iter-0 0.70 vs brownian 0.15 under slip; IPF −0.03 to −0.05 on `slip` (p ≤ 0.025), neutral elsewhere; frozen-coupling control −0.003 | **fail** — reference matters (5×), IPF never helps |
| 1b | the Lie group matters when heading noise and anisotropy are large | SE(2) wins containment + covariance fidelity in those cells, equivariant, ties in the benign cell | wins 16/24 cells at anisotropy ≤ 3 (up to +0.38), covariance coverage 0.74 vs 0.18–0.44, equivariance 2e-6 vs 0.1–0.9; **loses 8/12 cells at anisotropy 10** | **fail (split)** — heading yes, anisotropy no |
| 2 | bridges degrade more gracefully than trackers and single-shot policies | bridge − diffusion positive and growing with severity | bridge − diffusion +0.27 (L1 slip), +0.33 (L2 slip), positive in all 6 disturbed cells; gap shrinks past 2× slip; nominal PD > bridge everywhere | **half pass** — beats diffusion, not the tracker, not growing |
| 3 | drift disagreement D predicts failure and triggers useful replans | AUC(D) > both baselines at ≥ 10 steps; trigger helps under push/rain only | AUC(D) 0.47–0.52 (chance), distance-to-nominal 0.52–0.59; trigger Δ ≤ 0.02, false-replan 0.71–0.85 under push | **fail** |
| 4 | the marginal is the right place to put robustness | 4a interior optimum in width that moves with disturbance; 4b bimodal bridge picks the low-slip gap and ≥ two-bridge | 4a monotone in w (argmax at the boundary); 4b multi 0.29 vs two-bridge 0.39, gap choice r = −0.31 (p = 0.39) | **fail** on both |
| 5 | bridge samples are better augmentation than noised demos | (c) > (b) at equal size under disturbance | +0.07 to +0.31 in all 6 cells (p ≤ 0.002); noised demos hurt; bridge-only best (L1 slip 0.58 vs 0.32) | **pass** |

**Phase 1.** What the reference knows about the terrain covariance is the whole story: `slip` (OU pull + Σ(x) from the map) lifts success under slip disturbance from 0.15 to 0.70 and halves the terminal W₂; `unicycle` (the same pull, constant σ) is indistinguishable from Brownian. IPF converges in two iterations and lowers the transport cost by 4–5% exactly as designed, leaves the marginal fit at the sampling floor as it must, and that lower-cost coupling is worth −0.03 to −0.05 success under disturbance; re-fitting for the same budget on the frozen coupling changes nothing, so it is the coupling, not the optimiser. The prompt's stop rule (halt if Phase 1 fails for every reference) was overridden by the instruction to complete the suite; later phases use iteration 0.

**Phase 1b.** On terrain-free curved routes the SE(2) construction predicts its own uncertainty correctly where an ℝ³ Gaussian does not (2σ coverage 0.70–0.74 vs 0.18–0.44, KL ≤ 0.2 vs up to 16), is exactly left-equivariant, and beats flat by 0.1–0.4 success wherever heading noise and curvature dominate — with the exact-transport variant `se2x` never significantly worse than flat at anisotropy ≤ 3. At lateral/longitudinal anisotropy 10 the world-frame model is better in 8 of 12 cells even against `se2x`; the transport approximation explained half of `se2`'s deficit there and the rest is not understood. Same wall-clock and iteration count for both.

**Phase 2.** Same demos, same seeds: the bridge chain is 1.5–2.5× more successful than the diffusion policy under every disturbance on both layouts and ties it undisturbed; the advantage peaks at nominal slip and vanishes at 2–4× because everything collapses. Nominal PD tracking beats every learned method under every disturbance and degrades more slowly; PPO is 0 everywhere (sparse reward, 1M steps); IPF costs 0.03–0.06.

**Phase 3.** D(x,t) = ‖b_fwd + b_bwd‖ is the deviation from the bridge's support, and the failures here (arrival misses, gap collisions from accumulated slip) do not look like leaving a broad support. It predicts failure at chance under every disturbance, loses to distance-to-nominal-path, and as a replan trigger with a 10% false-replan budget it barely fires under slip/rain and mostly misfires under push; restarting the skill from the current state is not a different plan.

**Phase 4.** On L2 with a corrupted map, wider ρ₂ is monotonically better (argmax at the widest tested); the bimodal bridge averages its pull across the wall, collides more (0.61 vs 0.45) and splits 60/40 between the gaps regardless of terrain, while two unimodal bridges plus an informed gap choice are better (0.39 vs 0.29).

**Phase 5.** Rolling the bridge under its reference noise produces (off-nominal state, corrective action) pairs that the demos lack; adding them to the diffusion policy's data raises disturbed success by 0.07–0.31, Gaussian noising of demos lowers it, and bridge samples alone are the best training set. This is the one clean win for the bridge in the suite, and it is as a data generator, not as a controller.

**What holds up across the suite.** A terrain-covariance-aware reference is worth 5×; the SE(2) construction is worth +0.1–0.4 on curved routes with heading noise; bridge rollouts are the best training data tested. **What does not:** IPF (never helps, small consistent cost), the bridge as an executed controller against a PD tracker (loses in every experiment in this repo), D as a trigger, the multi-marginal bridge, and the tangent-space model under extreme anisotropy.

### What I'd run next

* Bridge drift wrapped in the tracker on the SE(2) task (the seam suite did this on the planar env and the bridge lost to a straight line there too), to settle whether the reference's 5× survives feedback.
* IPF under a terrain-aware cost instead of transport cost.
* A hybrid manifold: SE(2) interpolation and features with world-frame covariance accumulation, to isolate what the flat construction gets right at high anisotropy.

## Experiment 4 — seam suite

See `FINDINGS_seam.md`. Summary: with the same PD tracker in the loop, the cloud's mean path is worse than a straight line through the marginal means (A, fail); the handoff width is a real robustness knob with an interior optimum that moves outward with pushes (B, pass); bridge skills entered outside their cloud deliver half as often (C, pass) but a forced 3σ break of the seam costs at most 0.024 (D, vacuous), so C is selection rather than causation; E skipped by rule. The handoff marginal is not load-bearing.

---

# Experiment 1 — learned ε (original findings)

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
adds nothing.  In that implementation the IPF refinement **hurt**; Experiment 3's
Phase 1 reproduces the sign with a verified solver and a frozen-coupling control
(the killed reference's survivor bias, suspected then, was not the mechanism —
the refined coupling itself is).

## Other observations (Experiment 1)

* **Plain PPO (E) fails completely**: 0.0 success in every cell.  With the
  sparse ±1 terminal reward and a per-step penalty it converges to standing
  still.
* **Disturbance dominates at σ_d = 0.10**: a kick of std 0.10 at the pre-gap
  handoff against a gap half-width of 0.05 (w = 0.10) kills most episodes no
  matter what ε does.
* **Bridge sanity** (`results/verify.csv`): each skill alone reaches its
  target with W₂ = 0.02–0.04 for ε ∈ {0.003, 0.01, 0.03}.
* **Effort** (fig. 3): learned ε does not move the Pareto front beyond what
  fixed ε = 0.001–0.003 already achieves.

The ε lever, as posed, has no upside in that task: the ε-policy learns to
switch noise off.

# Experiment 2 — terrain robustness, open-loop bridges vs tracker

See `FINDINGS_terrain.md`: open-loop bridge skills (Brownian reference) lost to PD tracking of a demo path everywhere including σ_k = 0; that comparison was confounded by open-loop execution and is superseded by the seam suite. Its diffusion baseline used a 50-step DDPM schedule with β_max = 0.02 (ᾱ_T = 0.6), which Experiment 3 found to be a defect; its DIFF number (0.40 at the tuning cell) should be read as a floor.
