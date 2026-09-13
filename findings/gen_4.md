# g4 — Manifold through the data

**Question.** Phase 1b found the SE(2) bridge predicts its own uncertainty correctly where a flat ℝ³ model does not, and beats flat as a controller on curved routes with heading noise. Does the correct uncertainty produce better *training data*?
**Setup.** Phase 1b's S-curve route (180° of heading change) with body-frame slip anisotropy a ∈ {1, 3} and heading-noise scale h ∈ {0.2, 0.5}; no terrain. 20 demonstrator rollouts (Nominal, kp = 6) as demos, 80 generated trajectories, diffusion at 8 000 steps, evaluation on the same route under its body-frame noise, 200 episodes, **5 seeds**. Generators: BRIDGE-slip trained and rolled out on SE(2) vs on flat ℝ³; PD-noise with the noise applied in the tangent (body) frame vs in a world frame fixed at each skill's mean heading.
**Pre-registered pass.** SE(2)-generated data beats flat-generated data downstream by ≥ 0.05 in the high-heading-noise cells, for both generator types.
**Verdict: FAIL.** For the bridge, SE(2)-generated data is *worse* than flat-generated data at h = 0.2 (−0.14, p = 0.02 at a = 1; −0.06, p = 0.005 at a = 3) and not different at h = 0.5 (−0.03 and +0.06, both n.s.). For the tracker, the noise frame makes no difference in any cell (|Δ| ≤ 0.03, p ≥ 0.32). The manifold's correctness, real for the bridge's own uncertainty (1b), does not reach the downstream policy through the data.

## Success on the S-curve (mean ± 95% CI over seeds)

| h | a | BRIDGE se2 | BRIDGE flat | Δ se2 − flat (p) | PD tangent | PD world | Δ tangent − world (p) |
|---|---|---|---|---|---|---|---|
| 0.2 | 1 | 0.324 ± 0.073 | **0.463 ± 0.129** | −0.139 (0.021) | 0.281 ± 0.102 | 0.298 ± 0.076 | −0.017 (0.33) |
| 0.2 | 3 | 0.208 ± 0.077 | **0.266 ± 0.074** | −0.058 (0.005) | 0.210 ± 0.102 | 0.217 ± 0.065 | −0.007 (0.69) |
| 0.5 | 1 | 0.445 ± 0.087 | 0.478 ± 0.099 | −0.033 (0.40) | 0.503 ± 0.060 | 0.535 ± 0.061 | −0.032 (0.32) |
| 0.5 | 3 | 0.352 ± 0.116 | 0.288 ± 0.063 | +0.064 (0.10) | 0.358 ± 0.065 | 0.336 ± 0.060 | +0.022 (0.56) |

## Reading the table

* **The generator's geometry is invisible to the policy.** Whatever manifold the generator used, the recorded data is (world pose, body command) pairs and the policy sees `obs_of` features and body-twist chunks. The SE(2) bridge's advantage in 1b came from *executing* an equivariant, correctly-scaled drift; once that drift is only used to produce states and labels for a world-coordinate policy, the difference is at most the distribution of states visited, and here the flat bridge's data (which wanders differently) trained the better policy at low heading noise.
* **Noise frame does not matter for the tracker** for the same reason it did not matter in g1 (PD-iso ≥ PD-noise): the labels are kp = 10 corrections regardless of how the perturbation was shaped.
* **Everything is low on this route** (0.2–0.5). The S-curve with body noise and a 100-step budget per skill is hard for a chunk policy trained on 100 trajectories; the bridge and the tracker are within noise of each other here (unlike L1/L2), which says the tracker's g0 advantage depends on having good demo paths to track — the 20 demonstrator rollouts on the S-curve are less useful references than the Phase 5 demos on L1.
* This is consistent with 1b's own caveat: SE(2) beat flat at a ≤ 3 as a controller, but that was ~+0.1 at h = 0.2 on the S-curve; the data-generation route loses that and more.

## Notes

* The flat bridge records body commands by rotating its world-frame drift into the body frame at each step (`gen_sources.step_kin`), so both bridges produce the same kind of labels.
* The world-frame noise model uses each skill's mean geodesic heading (`phase4` in `gen_phases.py`); with a = 1 it differs from the tangent model only through heading-position coupling.
* Compute: 14.8 min per seed on the GPU (24 bridge nets trained on CPU per seed, 16 policies).

## What I'd run next

* Evaluate the *generator itself* (bridge executed) on the same S-curve cells alongside the policies, to confirm the 1b controller advantage is present in this exact setting while the data advantage is absent.
* An SE(2)-native downstream policy (invariant features, body-frame chunk) trained on the two bridge datasets: if the manifold's benefit needs an equivariant consumer, this is where it would show.
* Higher heading noise (h = 1.0), where 1b's covariance-fidelity gap was largest.
