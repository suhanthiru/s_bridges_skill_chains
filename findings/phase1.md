# Phase 1 — Reference process

**Hypothesis.** IPF only hurts because the reference is uninformative.
**Grid.** reference ∈ {brownian, killed, unicycle, slip} × state ∈ {flat, SE(2)} × disturbance ∈ {none, slip, rain} × IPF ∈ {0, 5}, layout L1, 500 episodes per cell, **10 seeds** (0–9). Nominal PD tracking of the geodesic through the marginal means is shown as a context line (the full baseline set is Phase 2).
**Verdict: FAIL on the pre-registered criterion, for every reference — but not for the reason the criterion anticipated.** The informative reference works (`slip` beats `brownian` iteration-0 by +0.55 success under slip disturbance, 5×, CIs nowhere near touching). IPF does not: it is neutral for the three uninformative references and significantly *negative* for `slip` (−0.03 to −0.05). The solver is not the cause: the linear-Gaussian gate reproduces the closed-form Schrödinger bridge to 3.5e-7, IPF lowers the transport cost by 4–5% exactly as it should, and a frozen-coupling control shows the degradation comes from the refined coupling rather than optimizer churn. IPF optimises a quantity (transport cost at fixed marginals) that this task does not reward.

## Gate: unit tests (all pass; `python tests.py`)

| test | result |
|---|---|
| IMF fixed point at the closed-form Gaussian SB coupling | max abs err 3.5e-7 (< 1e-3) |
| IMF convergence from ρ₀⊗ρ₁ (12 iterations) | 3.9e-7 |
| marginal covariances Σ_t vs closed form | 2.0e-7 |
| ε→0 recovers the Bures coupling | 5.0e-10 |
| SE(2) exp/log round-trip, composition, geodesic endpoints, left-equivariance | ≤ 2e-6 |
| terrain lookup, patch structure, anisotropy, rain | exact / 0.94 / 12.2× / 0.600 |
| W2 on a known Gaussian pair | exact translation, 1.7% on equal-cov pair |
| reference bridge reduces to the Brownian bridge at κ=0 | drift 2.5e-6, covariance 1% |
| fast path vs reference implementation (all 4 references × 2 manifolds) | ≤ 4e-6 |
| bridge construction equivariance: SE(2) / flat | 1e-6 / 0.1 (flat is not, as intended) |

The IPF test propagates moments rather than samples, so the 1e-3 tolerance measures the projection logic, not fitting error. That was the point: any degradation seen below is not "IPF implemented wrong".

## Main table (success, mean ± 95% CI over 10 seeds)

| reference | manifold | IPF | none | slip | rain | W₂(none) | W₂(slip) | W₂(rain) |
|---|---|---|---|---|---|---|---|---|
| brownian | flat | 0 | 0.962 ± 0.019 | 0.149 ± 0.014 | 0.071 ± 0.007 | 0.094 | 0.211 | 0.249 |
| brownian | flat | 5 | 0.955 ± 0.011 | 0.143 ± 0.010 | 0.070 ± 0.008 | 0.093 | 0.222 | 0.256 |
| killed | flat | 0 | 0.970 ± 0.005 | 0.151 ± 0.009 | 0.073 ± 0.007 | 0.094 | 0.210 | 0.247 |
| killed | flat | 5 | 0.954 ± 0.011 | 0.151 ± 0.009 | 0.071 ± 0.006 | 0.093 | 0.219 | 0.253 |
| unicycle | flat | 0 | 0.962 ± 0.021 | 0.152 ± 0.012 | 0.073 ± 0.008 | 0.095 | 0.205 | 0.243 |
| unicycle | flat | 5 | 0.955 ± 0.011 | 0.149 ± 0.012 | 0.071 ± 0.007 | 0.093 | 0.216 | 0.252 |
| **slip** | flat | 0 | **0.992 ± 0.003** | **0.701 ± 0.020** | **0.293 ± 0.011** | 0.116 | **0.113** | **0.153** |
| slip | flat | 5 | 0.991 ± 0.003 | 0.673 ± 0.020 | 0.274 ± 0.010 | 0.114 | 0.115 | 0.156 |
| brownian | se2 | 0 | 0.962 ± 0.020 | 0.143 ± 0.014 | 0.070 ± 0.008 | 0.094 | 0.213 | 0.251 |
| brownian | se2 | 5 | 0.949 ± 0.010 | 0.144 ± 0.010 | 0.070 ± 0.006 | 0.093 | 0.222 | 0.256 |
| killed | se2 | 0 | 0.969 ± 0.009 | 0.149 ± 0.012 | 0.072 ± 0.008 | 0.094 | 0.211 | 0.248 |
| killed | se2 | 5 | 0.952 ± 0.013 | 0.148 ± 0.008 | 0.072 ± 0.006 | 0.094 | 0.217 | 0.253 |
| unicycle | se2 | 0 | 0.963 ± 0.020 | 0.147 ± 0.012 | 0.074 ± 0.010 | 0.095 | 0.205 | 0.244 |
| unicycle | se2 | 5 | 0.950 ± 0.009 | 0.147 ± 0.011 | 0.070 ± 0.009 | 0.093 | 0.216 | 0.252 |
| **slip** | se2 | 0 | **0.992 ± 0.002** | **0.704 ± 0.021** | **0.300 ± 0.016** | 0.118 | **0.113** | **0.150** |
| slip | se2 | 5 | 0.991 ± 0.003 | 0.663 ± 0.021 | 0.271 ± 0.017 | 0.115 | 0.114 | 0.156 |
| nominal PD | – | – | 0.994 ± 0.002 | 0.810 ± 0.012 | 0.446 ± 0.013 | – | – | – |

W₂ is the SE(2) 2-Wasserstein between the terminal states and ρ₃ (assignment on 400 points, heading weighted 0.3 m/rad); the sampling floor for ρ₃ is 0.034.

## Paired IPF effect (success at IPF 5 minus IPF 0, per seed)

| disturbance | reference | flat Δ (p) | SE(2) Δ (p) |
|---|---|---|---|
| none | brownian | −0.008 (0.46) | −0.014 (0.21) |
| none | killed | −0.016 (0.008) | −0.017 (0.003) |
| none | unicycle | −0.008 (0.49) | −0.012 (0.25) |
| none | slip | −0.001 (0.02) | −0.001 (0.07) |
| slip | brownian | −0.006 (0.20) | +0.001 (0.88) |
| slip | killed | +0.000 (0.95) | −0.001 (0.89) |
| slip | unicycle | −0.003 (0.36) | +0.000 (0.97) |
| **slip** | **slip** | **−0.028 (0.025)** | **−0.041 (0.003)** |
| rain | brownian | −0.001 (0.73) | −0.000 (0.94) |
| rain | killed | −0.002 (0.59) | −0.000 (0.95) |
| rain | unicycle | −0.002 (0.45) | −0.004 (0.37) |
| **rain** | **slip** | **−0.019 (0.006)** | **−0.029 (0.0007)** |

## Why IPF does not help: convergence diagnostics (pooled over skills and seeds)

| reference | transport cost it0 → it5 (rel.) | endpoint W₂ it0 → it5 | sampling floor |
|---|---|---|---|
| brownian | 0.0820 → 0.0781 (−4.7 ± 2.4%) | 0.0258 → 0.0259 | 0.019–0.034 |
| killed | 0.0810 → 0.0777 (−4.1 ± 1.6%) | 0.0256 → 0.0257 | |
| unicycle | 0.0820 → 0.0783 (−4.5 ± 2.7%) | 0.0256 → 0.0258 | |
| slip | 0.0979 → 0.0930 (−5.0 ± 1.8%) | 0.0302 → 0.0305 | |

IPF converges by iteration 2, lowers E‖log(x₀⁻¹x₁)‖² by 4–5% in every seed, and leaves the terminal marginal fit unchanged — which is exactly its contract (Markovian and reciprocal projections both preserve the marginals). Iteration 0 already sits at the finite-sample W₂ floor, so there is no marginal-fit headroom to gain. What changes is the coupling: a lower-cost coupling means straighter, less corrective transport, and under terrain disturbance that is worth a few points of success. The `killed` reference barely kills anything here (kept fraction 0.99), so the survivor-bias explanation from the earlier run does not apply.

**Optimizer-churn control** (frozen coupling: the identical extra 6 000 steps re-fitted on the original independent coupling, SE(2), seeds 0–4):

| reference | disturbance | IPF Δ (real coupling) | frozen-coupling Δ |
|---|---|---|---|
| brownian | slip | +0.010 ± 0.020 | +0.004 ± 0.011 |
| brownian | rain | +0.005 ± 0.008 | +0.004 ± 0.009 |
| slip | slip | −0.053 ± 0.053 | −0.003 ± 0.057 |
| slip | rain | −0.031 ± 0.027 | −0.002 ± 0.026 |

Extra optimisation on its own changes nothing; the refined coupling is what costs success.

## Other observations

* **Only the terrain-dependent covariance matters.** `unicycle` (OU pull, constant σ) is indistinguishable from `brownian`; `slip` (same pull, Σ(x) from the terrain) is 5× better under disturbance. The drift target Σ(x)Φ Q(1,τ)⁻¹(−Φξ) steers harder where the accumulated slip is larger; that, not the drift toward the goal, is the useful information.
* **flat vs SE(2) is a wash on L1** (straight route, heading std 0.15–0.25, isotropic slip). Every paired difference is inside its CI. This is the benign cell where Phase 1b predicts no difference.
* **Nominal PD beats every bridge** (0.81 vs 0.70 under slip; 0.45 vs 0.30 under rain), as in the two earlier runs. Bridges here execute a feed-forward drift with only the implicit 1/(1−τ) gain; a kp = 6 tracker closes error faster.
* **Calibration, declared before the grid:** class friction contrast compressed by FRIC_SCALE = 0.5 and slip magnitude 0.7 so that the nominal PD controller lands at ≈ 0.99 / 0.80 / 0.43 on none / slip / rain; at the original magnitude every reference except `slip` sat at a 0.02–0.05 floor and nothing could be resolved.
* **Compute.** Training + evaluation on CPU (8 configurations in parallel, 2 threads each); this GPU has ~0.16 ms kernel-launch overhead and the SE(2)/bridge machinery is hundreds of tiny ops, so CPU was measured ~2× faster. 10 seeds took 96 min; the frozen control 27 min.

## Consequence for the rest of the suite

The prompt's stop rule (stop if Phase 1 fails for every reference) was overridden by the instruction to complete both suites. Phases 1b–5 use the `slip` reference at **iteration 0**; Phase 2 keeps `bridge_ipf` in the table as specified so the IPF cost shows up in the method comparison.

## What I'd run next

* IPF with a task-aware cost: refine the coupling under a cost that includes the terrain (e.g. path-integrated slip variance) instead of plain transport cost, and see whether the sign of the IPF effect flips.
* A reference with terrain-dependent *drift* as well as covariance (the unicycle pulled along a terrain-aware path), since the covariance alone bought 5× and the OU pull bought nothing.
* Close the gap to nominal PD by wrapping the bridge drift in the same tracker (that is what the seam suite does).
