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
