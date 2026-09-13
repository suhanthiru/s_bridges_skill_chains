# Schrödinger-bridge skill chaining with a learned noise level

A 2D point robot must cross a wall through a gap by chaining three
ε-conditioned Schrödinger-bridge skills (ρ₀→ρ₁→ρ₂→ρ₃).  The question is
whether a PPO policy that picks the bridge noise level ε per state beats the
best fixed ε as handoff disturbances grow, and whether the refined (IPF)
coupling matters for that.

Everything is plain PyTorch + NumPy; no frameworks, no ROS, no gymnasium
dependency (the env follows the Gymnasium `reset()/step()` contract).

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py            # everything: nets, PPO, evaluation, figures
```

Outputs: `results/` (raw CSVs, trained nets, timing), `figures/` (PNG+PDF),
and `FINDINGS.md` (written by hand after reading `results/summary.csv`).

Useful variants:

```bash
python run_all.py --quick                      # ~4 min smoke test, tiny budgets
python run_all.py --stage eval --stage plots   # re-evaluate and re-plot from saved nets
python run_all.py --tune                       # hyper-parameter check at the tuning cell
python plots.py                                # figures only, from results/*.csv|npz
```

Each figure is produced by one function in `plots.py`
(`fig1_success`, `fig2_heatmap`, `fig3_pareto`, `fig4_bridge_sanity`,
`fig5_handoff`).  `python plots.py` regenerates all of them.

Measured wall-clock on an RTX 3080 Ti shared with other processes
(`results/timing.json`): bridges 24 min, iteration-0 nets 5.5 min, PPO C/D/E
6.5 + 6.5 + 3 min, evaluation 9 min, plots 4 s — about 55 min total.  Seeds
(0–4) and the frozen config are logged in `results/seeds.json`.

## Files

| file | contents |
|---|---|
| `env.py` | wall geometry, truncated-Gaussian regions, exact W₂, batched env, single-robot wrapper |
| `bridge.py` | drift MLP, Brownian-bridge loss, DSBM/IPF training, SDE execution, Sinkhorn, verification |
| `rl.py` | ε-policy, critic, bridge-driven env, vectorised PPO, ε heatmap |
| `baselines.py` | flow-matching net (D′), plain-PPO policy/env (E) |
| `run_all.py` | stages, multiprocessing, evaluation, CSV writing |
| `plots.py` | the five figures |

## Experiment

* Unit square, dt = 0.01, single integrator, ‖u‖ ≤ 1, wall at x = 0.5 with a
  centred gap of width w ∈ {0.10, 0.20, 0.40}.  Disturbance: a kick
  N(0, σ_d²I) at each skill handoff plus process noise N(0, (0.2σ_d)² dt) each
  step, σ_d ∈ {0, 0.02, 0.05, 0.10}.  12 cells.
* Skills: 3 drift nets f_θ(x, τ, ε) (4×256 SiLU, sinusoidal features of τ and
  log ε) trained by Diffusion Schrödinger Bridge Matching with 5 IPF
  iterations.  One bridge step = one env step (dτ = dt = 0.01, 100 steps per
  skill).  Success = final x inside the 2-std ellipse of ρ₃, no collision.
* Conditions: A fixed ε ∈ {0.001, 0.003, 0.01, 0.03, 0.1}; B oracle
  (ε = 0.003 at the gap skill, 0.03 elsewhere); C learned ε on the IPF-refined
  bridge; D learned ε on the iteration-0 bridge (no refinement); E plain PPO.
* 5 training seeds, 200 evaluation episodes per cell per seed.

## Design choices worth knowing

**ε range.**  The prompt's ε ∈ [0.01, 1] was rescaled to **[1e-3, 1e-1]**.
With the bridge pinned at both ends the mid-skill position spread is
√(ε τ(1−τ)); at ε = 0.8 that is ≈ 0.45 in a unit box, so every fixed-ε
setting above ≈ 0.1 would simply hit the wall.  At ε = 0.03 the midpoint std
is ≈ 0.09, comparable to the gap widths, which is where the tradeoff lives.

**Killed reference process.**  The reference is Brownian motion killed on the
wall.  In code: (1) endpoint pairs whose straight line crosses the wall are
rejected, (2) trajectories that hit the wall while simulating the coupling
during IPF are dropped, (3) the drift loss adds λ·relu(drift component toward
the wall)² for bridge samples within 0.02 of the wall (λ = 1, tuned at the
tuning cell; λ = 10 was not better).

**IPF convergence.**  `results/ipf.csv` logs per iteration the mean coupling
cost E‖x₁−x₀‖², the Sinkhorn (entropic OT, reg 0.01) cost between simulated
endpoints and ρ_k, the W₂ endpoint error and the killed fraction at ε = 0.01.

**Condition D.**  D is the bridge-matching net at IPF iteration 0
(stochastic Brownian-bridge interpolant, independent coupling, no refinement)
with its own PPO ε-policy trained exactly as in C.  The only difference
between C and D is the refined coupling.  A deterministic-interpolant flow
matching net (D′, `baselines.train_flow`) is available with `--with-dprime`;
it is the weaker baseline because it never saw noise in training and is not
part of the main run.

**PPO scope.**  One ε-policy per gap width w per seed, trained with σ_d drawn
uniformly from the sweep at each episode, then evaluated at every σ_d.  This
is why the heatmap in fig. 2 is shared across σ_d: the policy was trained
across σ_d, not because ε is σ_d-independent in principle.  2M env steps per
run, 512 parallel envs, rollout 32, 4 epochs, clip 0.2, lr 3e-4.  At
evaluation the policy's mean log ε is used (no sampling).

**Time-varying ε is slightly off-distribution for the drift.**  The drift is
trained with one ε per trajectory; the PPO policy can change ε every step.
Fixed-ε verification passed and the PPO runs were stable (and the learned ε
turned out nearly constant), so the fallback of training the drift with a
piecewise-constant ε schedule was not needed.

**Plain PPO (E).**  Gaussian policy over u with inputs (x, t/300), no bridge
noise, same reward.  To keep the disturbance schedule identical, kicks are
applied at steps 100 and 200.

**Metrics.**  W₂ is computed exactly by assignment
(`scipy.optimize.linear_sum_assignment`) on ≤ 500 points, both for bridge
verification (endpoints vs ρ_k samples, survivors only, no disturbance) and
for handoff error (states at each handoff vs ρ_k).  Effort is Σ‖u‖² dt.

**Extra check.**  `results/check_iter0_vs_ipf_fixed_eps.csv` evaluates the
IPF-refined and the iteration-0 nets at fixed ε = 0.001 with no policy (12
cells × 5 seeds, σ_d ∈ {0, 0.05}); it was run after the main results to
isolate why D beats C.  Regenerate with the snippet at the bottom of this
file.

**Tuning.**  Only at w = 0.20, σ_d = 0.02, seed 0 (`results/tuning.csv`):
penalty λ ∈ {1, 10} × PPO lr ∈ {1e-4, 3e-4} with a quarter PPO budget.
Differences were within evaluation noise; λ = 1, lr = 3e-4 was frozen into
`CONFIG` in `run_all.py` and never touched again.

## Regenerating the extra check

```python
import pandas as pd, run_all as RA
d = RA.dev(); cache = {}; rows = []
for kind in ("bridge", "flow0"):
    for w in RA.WIDTHS:
        for sd in (0.0, 0.05):
            for s in RA.SEEDS:
                cache[("bridge", w, s)] = [RA.B.load_net(RA.net_path(kind, w, s, k), d) for k in range(3)]
                r, _ = RA.run_episodes("A0.001", w, sd, s, RA.CONFIG, d, cache)
                rows.append(dict(kind=kind, w=w, sigma_d=sd, seed=s, success=sum(x["success"] for x in r) / len(r)))
pd.DataFrame(rows).to_csv("results/check_iter0_vs_ipf_fixed_eps.csv", index=False)
```

---

# Experiment 2: are bridge skills more robust to terrain disturbance?

Second experiment, same repo, new files: `terrain_env.py`, `demos.py`,
`conditions.py`, `run_terrain.py`, `plots_terrain.py`; results in
`results_terrain/`, figures `figures/fig*t_*`, findings in
`FINDINGS_terrain.md`.  The bridges reuse `bridge.py`'s sinusoidal features
and the iteration-0 (independent-coupling) bridge-matching loss; PPO reuses
`rl.ppo_train`.

```bash
python run_terrain.py                       # layouts, demos, training, eval, figures
python run_terrain.py --quick               # smoke test (~20 min, dominated by the oracle MPC)
python run_terrain.py --stage eval --stage plots --conds ORACLE,TRACK,BRIDGE-plain,BRIDGE-obs
python run_terrain.py --tune                # tuning at (sigma_k=0.06, mild), layouts 20-24
python plots_terrain.py                     # figures from results_terrain/*.csv|npz
```

**What was run.**  Per the order of work the run stopped at the step-4
checkpoint (ORACLE, TRACK, BRIDGE-plain, BRIDGE-obs evaluated on all 10
cells × 5 seeds).  DIFF and PPO are implemented in `conditions.py` and were
tuned, but not trained or evaluated: `python run_terrain.py --stage
train_diff --stage train_ppo --stage eval --stage plots` runs them.
Figures 1, 2, 4, 5 are produced from the four conditions; figure 3 (example
rollouts) shows the conditions available.

**Environment.**  Unit square, dt = 0.01, ‖u‖ ≤ 1, no wall.  Roughness
field r(x) ∈ [0,1] on a 64×64 grid: sum of 6 random sinusoids, min-max
normalised, with smooth Gaussian clearings (std 0.06) carved at the four
waypoint means; fields are resampled until every route segment crosses
r ≥ 0.7.  The clearings are a deviation from a pure sinusoid field: without
them the oracle could not hold a goal inside a rough patch (process noise per
step exceeds the slip-limited correction) and the required ≥ 95 % no-push
success was unreachable (74 % on rough terrain).  Slip
v = (1 − s·r) R(θ·r) u, process noise √(σ_p² r dt), pushes N(0, σ_k² I) with
probability 0.02 per step.  Route means (0.12, 0.37, 0.63, 0.88) × 0.5, std
0.05, fixed across layouts so demo paths are transferable.  Layouts 0–19
train, 20–24 tuning eval, 20–29 test; the layout ids reproduce the fields
(`results_terrain/layouts.npz` is a cache).

**Oracle / demos.**  MPPI-style sampling MPC (K = 200, H = 30, noise 0.5,
λ = 0.002, running + terminal goal cost) that knows the true field.  Targets
are ρ_k samples truncated to 1.5 std so the target sits inside the 2-std
success set.  Verified: 100 % chain success without pushes on all 30
layouts in both settings (`layout_check.csv`).  500 demos per skill per
training layout per setting (`data/`, 39 MB, float16, committed).

**Conditions.**  BRIDGE-plain/obs: iteration-0 bridge matching on the demo
(start, end) pairs of all 20 training layouts, reference noise
ε = σ_p²·0.5 (env process noise at mean roughness); obs variant appends the
16-ray roughness at x_τ.  Executed as drift only; the env's process noise
plays the role of the bridge noise.  Handoffs at steps 100/200 regardless of
position.  TRACK: nearest training demo by start state (across layouts) as
the nominal path, PD on position error plus the demo's feed-forward action.
DIFF: MLP denoiser over 8×2 action chunks conditioned on x ⊕ one-hot k ⊕
rays, DDPM-50 train / DDIM-10 test, executes 4 of 8.  PPO: Gaussian policy
on the same observation, 2M steps, trained at σ_k = 0.10.

**Metrics.**  Recovery is measured in corridor coordinates (progress along
the route polyline, lateral distance).  Only pushes that move the robot
laterally by > 0.05 are tracked (an along-track push would otherwise count
as instantly recovered); recovered = lateral distance back within 0.05 of its
pre-push value; a push arriving while one is open supersedes it; open pushes
at episode end are "never".  Corridor deviation = mean distance from the
current skill's straight segment.  Handoff W₂ by assignment vs ρ_k samples.

**Tuning** (`results_terrain/tuning.csv`, cell σ_k = 0.06 mild, layouts
20–24, seed 0): TRACK kp ∈ {2, 5, 10} × kd ∈ {0, 0.05} → kp = 10, kd = 0
(0.97; kp = 10 is the grid edge); BRIDGE lr ∈ {3e-4, 1e-3} → 3e-4 (0.485 vs
0.48); DIFF lr ∈ {1e-4, 3e-4} → 1e-4 (0.40); PPO lr ∈ {1e-4, 3e-4} → 3e-4
(0.00 at both).  Frozen in `CONFIG`.

**Wall-clock** (RTX 3080 Ti, shared): layouts 25 s, demos 9.3 min, tuning
6.4 min, bridge training 15.8 min (30 nets), eval of four conditions 17.4 min
(ORACLE 10.9 min of that).  `results_terrain/timing.json`.

---

# Experiment 3: SE(2) terrain suite (reference process, Lie group, robustness, seam trigger, marginals, augmentation)

Files: `se2.py` (SE(2) ops + manifold interface), `terrain.py` (terrain classes, layouts L1/L2, marginals),
`sde.py` (reference SDEs and their Gaussian bridges, closed-form Gaussian SB), `bridge_fast.py`
(shape-specialised training path, tested against `sde.py`), `task.py` (chained-skill SE(2) task),
`solver.py` (drift nets, DSBM/IPF, drift-disagreement D), `phase1b.py`–`phase5.py`, `analysis.py`,
`analysis2.py`, `run.py`, `tests.py`. Results in `results/` (per-seed parquet shards, merged into
`results.parquet` by `python run.py --merge`), figures `figures/phase*`, findings `findings/phase*.md`.

```bash
python run.py --phase tests                 # gate: linear-Gaussian covariance steering to 1e-3, SE(2), terrain, W2
python run.py --phase 1 --seed S            # reference x manifold x disturbance x IPF (L1)
python run.py --phase 1frozen --seed S      # optimiser-churn control for the IPF effect
python run.py --phase 1b --seed S           # Lie group: curved routes x heading noise x anisotropy (flat, se2)
python run.py --phase 1bx --seed S          #   ... the exact-transport SE(2) variant (se2x)
python run.py --phase 2 --seed S            # nominal / ppo / diffusion / bridge_iter0 / bridge_ipf x disturbance x layout
python run.py --phase 3                     # D(x,t) as failure predictor + replan trigger (uses phase-2 trajectories)
python run.py --phase 4 --seed S            # L2, noisy map: width sweep (4a) and bimodal rho_2 (4b)
python run.py --phase 5 --seed S            # diffusion trained on demos / noised / +bridge samples / bridge only
python run.py --merge && python analysis.py --phase 1 && python analysis2.py --phase 1b   # etc.
make all                                    # tests, phase 1 (10 seeds), merge
```

Design notes that matter for reading the findings:

* **Reference SDEs.** All four are OU deviations from the interpolant, `d xi = -kappa xi dt + L(x) dW`,
  so the bridge is a closed-form Doob h-transform and the position-dependent slip covariance enters
  by quadrature along the interpolant (frozen-coefficient approximation). `brownian`/`killed`: kappa=0,
  constant sigma; `unicycle`: kappa=2 pull toward the next marginal; `slip`: the same pull with
  Sigma(x) from the terrain map. The linear-Gaussian case has the known entropic-OT closed form,
  which the IPF logic reproduces to 3.5e-7 (`tests.py`).
* **Manifolds.** `flat` models (x, y, theta) in R^3 with world-frame noise; `se2` uses geodesic
  interpolation and left-invariant (body-frame) noise with invariant network features, so its bridge
  construction is exactly equivariant (tested). The physics is SE(2) and every metric is SE(2)-correct
  for both. `se2` carries deviations by first-order transport; `se2x` (Phase 1b) transports the drift
  target and the accumulated covariance by the SE(2) adjoint.
* **D(x,t)** = ||b_fwd + b_bwd||; near the bridge's support this equals ||xi||/(tau(1-tau)), the
  normalised deviation from the support. Phase 3 uses the tau-normalised version for the trigger.
* **Calibration** (declared before the grid): FRIC_SCALE=0.5 and slip magnitude 0.7 so that the
  nominal PD controller is at about 0.99/0.80/0.43 on none/slip/rain.
* **Baselines.** PPO: 1M steps (compute cap), 512 envs, trained under `slip`. Diffusion: chunk 8,
  execute 4, DDPM-50 / DDIM-10 with beta up to 0.2 (a 50-step schedule must reach abar_T ~ 0; the
  earlier terrain-run DIFF used beta max 0.02, abar_T = 0.6, and was weak for that reason). Demos: 500
  nominal-PD rollouts per layout with the demonstrator's clean commands as labels.
* **Compute.** The GPU on this machine has ~0.16 ms kernel-launch overhead; the bridge machinery is
  hundreds of small ops per step, so bridge training runs on CPU (8 configs x 2 threads). PPO and
  diffusion run on the GPU. Phase 1 used 10 seeds; the rest 5 (stated in each findings file).

# Experiment 4: seam suite (is the handoff marginal load-bearing?)

Files: `seam.py`, `seam_plots.py`, `tests_seam.py`; results `results_seam/`, `results_seam.parquet`
(cells) and `results_seam_episodes.parquet` (episodes); figures `figures/seam_*`; findings
`findings/seam_*.md`; roll-up `FINDINGS_seam.md`.

```bash
python run.py --phase seamtests             # PD following, Mahalanobis, shifted marginals
python run.py --phase A --seed S            # point vs cloud, 5 conditions x terrain x sigma_k
python run.py --phase B --seed S            # handoff-width sweep (BRIDGE-slip-tracked)
python run.py --phase C                     # seam containment, from the A/B episode logs
python run.py --phase D --seed S            # broken seam: skill 2 started from shifted rho_1'
python run.py --merge                       # also merges the seam shards
```

Every condition is a reference generator wrapped in the same PD tracker (kp=10 + feed-forward):
TRACK-oracle (nearest MPC demo), TRACK-naive (straight line through the marginal means),
BRIDGE-tracked (Brownian-reference iteration-0 bridge drift flow from the entry state),
BRIDGE-slip-tracked (unicycle + terrain-slip reference), TSM (behaviour-cloned skills fine-tuned by a
differentiable rollout so terminal states land in the next skill's measured initiation set; the set is
a classifier fitted to a 13x13 probe grid x 20 layouts x 50 rollouts, label = success >= 0.8).
The environment is the planar terrain env of Experiment 2, unchanged: "heading axis" = along-track,
"corridor normal" = lateral, Mahalanobis in R^2. Bridges are trained on marginal samples (not demo
endpoints) so Phase B can rescale the marginals. Phase E (time-budget split) runs only if A passes.

---

# Experiment 5: bridge-as-generator suite (is the bridge necessary as a data generator?)

Files: `gen_sources.py` (generators, shared noise stream, SE(2) sampling-MPC oracle, coverage),
`gen_policies.py` (diffusion / flow-matching / BC / BC-GMM behind one interface), `gen_phases.py`
(phases g0–g6, CMA-ES niche search), `gen_plots.py`, `tests_gen.py`. Results `results_generator/`,
merged `results_generator.parquet`, search log `results_generator/phaseg5_search.parquet`, figures
`figures/gen_*`, findings `findings/gen_*.md`, roll-up `FINDINGS_generator.md`.

```bash
python run.py --phase gentests                  # shared noise, DART covariance, MPC-relabel, schedule
python run.py --phase g0 --seed S               # kill test: DEMO, NOISED, BRIDGE-slip, PD-noise, DART, MPC-rollout
python run.py --phase g1 --seed S               # isolations: references, action labels, coverage (L1)
python run.py --phase g2 --seed S               # scale: generated size, N_demo, eval severity
python run.py --phase g3 --seed S               # policy class: diffusion, flow, bc, bc_gmm
python run.py --phase g4 --seed S               # manifold through the data (S-curve route)
python run.py --phase g5 --seed 0               # CMA-ES niche search + L27 factorial + validation
python run.py --phase g6 --seed S               # transfer and wrong-terrain-model generators
python run.py --suite gen --phase 0 --seed S    # same as --phase g0 (the prompt's numbering)
python run.py --merge                           # also merges the generator shards
```

Design notes:

* **Every generator runs on the reference kinematics** g ← g·exp(u dt + √dt Σ(x)^½ z): no friction,
  no base noise, Σ(x) the slip-reference covariance (isotropic match for PD-iso), z a standard-normal
  stream fixed by the seed and shared by all sources. Only the controller differs, so PD-noise is a
  control for BRIDGE-slip in the strict sense (`tests_gen.py` checks the shared draws).
* **Datasets** = the first N_demo Phase-5 demos (clean demonstrator commands) + 4·N_demo generated
  trajectories (DEMO: demos only). The bridge itself never sees the demos — it is trained on the
  marginals — which is what the N_demo sweep (g2b) probes.
* **Oracle MPC** (`gen_sources.MPC`): K = 200 sampled body-twist sequences, horizon 30, MPPI-weighted,
  warm-started, on the same kinematics, toward the current skill's next marginal mean. MPC-relabel
  imposes BRIDGE-slip's states and records the MPC's commands.
* **DART**: the demonstrator (Nominal, kp = 6) executed with action noise η = Σ^½ z/√dt and no
  displacement noise, clean commands recorded; the displacement covariance matches Σ dt (tested).
* **Environment axes** (g5/g6): slip magnitude (× nominal 0.7), lateral-slip anisotropy (lateral std
  × a), terrain correlation length (Voronoi seed count), push-rate multiplier, marginal heading std,
  N_demo, demonstrator action noise; class-flip rate for the map the generator sees.
* **Training budget**: 8 000 steps for every policy in g0–g4/g6; 2 500 in the g5 search evaluator
  (3 seeds, 100 episodes) with 600-step bridges; validation of the top/bottom-3 θ with 10 seeds and
  200 episodes at the full budget. CMA-ES via the `cma` package (σ₀ = 0.25 on [0,1]^7), 60 evaluations
  maximising and 60 minimising f, plus a 3^(7−4) resolution-III fractional factorial (27 runs).
