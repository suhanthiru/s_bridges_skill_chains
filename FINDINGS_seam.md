# Seam experiment — is the handoff marginal load-bearing?

Planar terrain env of Experiment 2 (mild/rough, pushes σ_k, held-out layouts 20–29), same MPC demos, 5 seeds, 200 episodes per cell. Every condition is a reference generator wrapped in the identical PD tracker (kp = 10 + feed-forward), so nothing is executed open-loop and the earlier BRIDGE-vs-TRACK confound is gone. Per-phase files: `findings/seam_A.md` … `seam_D.md`; figures `figures/seam_*`; data `results_seam.parquet`, `results_seam_episodes.parquet`.

| phase | question | pre-registered pass | result | verdict |
|---|---|---|---|---|
| A | cloud vs point on the path | bridge ≥ naive + 0.10 at σ_k ≥ 0.06 rough | bridges −0.005 to −0.15 *below* the straight line; TSM ≈ naive; naive ≥ oracle | **fail** |
| B | width sweep | unimodal, interior optimum that moves outward with σ_k | optimum w = 1 at σ_k = 0.06 both terrains; argmax 0.25→1→4 (mild), 1→1→2 (rough) | **pass** (mild σ_k = 0.10 at the boundary) |
| C | seam containment | gap ≥ 0.20 for some skill, curve falls past 2σ | bridges +0.33 to +0.65, cliff at 2–3.5σ; trackers +0.06 to +0.13, flat | **pass** (bridges only) |
| D | broken seam | bridge degrades slower with δ than TSM | nobody degrades: ≤ 0.024 over 0–3σ; TSM flattest | **fail** (vacuous) |
| E | time-budget split | only if A passes | not run | skipped |

**A.** With the same closed-loop tracker, all five conditions are at 0.97–1.00 undisturbed; under pushes both bridge references are significantly worse than a straight line through the marginal means (rough, σ_k = 0.10: 0.76 vs 0.81; σ_k = 0.15: 0.54–0.58 vs 0.69), the slip reference does not help, and the terrain-aware oracle path is no better than the straight line. The bridge's time-parametrised reference (drift ∝ 1/(1−τ), fastest at the handoff) is the likely mechanism. TSM clears its 0.6 floor without tuning (1.00) and equals the straight line on rough terrain. The bridges are the cheapest references (7 % less effort than naive, 40 % less than the oracle).

**B.** Widening ρ₁, ρ₂ trades an undisturbed cost (w = 4: −0.04) against a robustness gain (w = 0.25 at σ_k = 0.10: −0.15), with the crossover moving outward as pushes grow. This is the one place the cloud does something a point cannot: it is the training distribution of the *next* skill, and its width is a robustness knob.

**C.** Observationally, a bridge skill entered outside its precondition cloud delivers half as often as one entered inside (0.28–0.55 vs 0.87–0.94), with a cliff between 2 and 3.5σ; the trackers do not care where they start. **D** then shows this is mostly selection, not causation: forcing skill 2 to start from a ρ₁′ translated by up to 3σ, rotated, shrunk or grown changes success by at most 0.024 for the bridges and nothing for TSM. Being outside the cloud at a handoff is a symptom of a heavily pushed episode that keeps being pushed, not the cause of the failure that follows.

**Engineering.** TSM = behaviour cloning + a differentiable-rollout terminal penalty into the next skill's measured initiation set (classifier on a 13×13 probe grid × 20 layouts × 50 rollouts, label = success ≥ 0.8); it reaches 1.00 undisturbed on the first attempt, so the published-method comparison is present. Bridges are trained on marginal samples (not demo endpoints) so B can rescale them. "Heading axis" is along-track and "corridor normal" lateral in this planar env; rotations of the isotropic nominal cloud are exact no-ops and are reported as such.

**Is the handoff marginal load-bearing? No** — with feedback in the loop the point does at least as well as the cloud on the path (A), the seam survives a 3σ break (D), and the one real effect (B: its width is the next skill's training distribution) is a tuning knob worth about 0.05–0.15 success, not a guarantee.
