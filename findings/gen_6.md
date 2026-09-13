# g6 — Transfer and model error

**Question.** How does each data source degrade when the generator's terrain differs from the deployment terrain (transfer), and when the generator's terrain *model* is wrong (class-flip corruption of the map it uses)?
**Setup.** L1, diffusion at 8 000 steps, success under slip at the target, 200 episodes, **5 seeds**. Transfer: data generated at 0.5× nominal slip ("mild") evaluated at 2× ("rough") and vice versa. Model error: data generated with the reference / injected noise drawn from a map with 10 / 30 / 50 % of cells re-labelled to a random class (bridges retrained on the corrupted map), evaluated on the true map at 1× slip. Sources BRIDGE-slip, PD-noise, DART; g0's rows are the 0 % reference.
**Pre-registered pass.** BRIDGE-slip degrades no faster than PD-noise under model error.
**Verdict: FAIL — the bridge is the most fragile of the three, and it is a deployment liability.** A 10 % class-flip costs the bridge 0.16 of its 0.51 (−32 %), the tracker 0.08 of 0.62 (−13 %), DART 0.02 of 0.75 (−3 %); at 50 % the losses are −44 %, −25 %, −20 %. In transfer the bridge's mild-terrain data is useless on rough terrain (0.011 vs 0.13 / 0.15) and its rough-terrain data is the weakest on mild terrain (0.80 vs 0.87 / 0.92). The reason is g1's: the bridge's entire advantage over an uninformed bridge is that its labels depend on Σ(x); when Σ(x) is wrong, so are the labels. The tracker's labels never depended on the map (only its injected states did), and DART's noise is on the action side, so both shed the error.

## Transfer across slip magnitude (success under slip at the target)

| generated at → evaluated at | BRIDGE-slip | PD-noise | DART |
|---|---|---|---|
| mild (0.5×) → rough (2×) | 0.011 ± 0.008 | 0.130 ± 0.012 | **0.145 ± 0.034** |
| rough (2×) → mild (0.5×) | 0.804 ± 0.091 | 0.865 ± 0.058 | **0.918 ± 0.059** |

For reference, g2c: policies trained at 1× and evaluated at 2× reach 0.10 (bridge) / 0.29 (tracker); at 0.5×, 0.73 / 0.72.

## Wrong terrain model in the generator (evaluated on the true map, 1× slip)

| class-flip rate | BRIDGE-slip | PD-noise | DART |
|---|---|---|---|
| 0 (g0) | 0.505 ± 0.035 | 0.618 ± 0.122 | 0.745 ± 0.047 |
| 0.1 | 0.344 ± 0.075 (−32 %) | 0.539 ± 0.133 (−13 %) | 0.726 ± 0.064 (−3 %) |
| 0.3 | 0.316 ± 0.076 (−37 %) | 0.484 ± 0.086 (−22 %) | 0.638 ± 0.038 (−14 %) |
| 0.5 | 0.282 ± 0.087 (−44 %) | 0.461 ± 0.084 (−25 %) | 0.598 ± 0.049 (−20 %) |

## Reading the tables

* **Model error enters the bridge twice.** Its states are perturbed with the wrong covariance *and* its drift labels are computed from the wrong covariance (the h-transform term Σ(x)Φ Q⁻¹); g1 showed the second is where its value lies (BRIDGE-slip − BRIDGE-brownian = +0.34), so corrupting Σ(x) removes most of it — at 10 % corruption the bridge is already only +0.18 above a Brownian bridge's 0.17. For the tracker only the states are perturbed wrongly, and its labels are still kp = 10 corrections toward the demo path. DART's action-side noise with a wrong covariance is just a slightly wrong exploration distribution.
* **Mild → rough is not a bridge-specific failure but the bridge is worst at it.** Data generated under 0.5× slip contains few large deviations for anyone; at 2× evaluation all three policies are poor (0.01–0.15), the bridge's essentially zero. Rough → mild is fine for all (0.80–0.92): training on larger perturbations than deployment costs little.
* **DART is the most robust source on every axis in this suite** — to model error, to transfer, and (g0) in absolute terms — which is the expected property of an augmentation that perturbs the demonstrator's actions rather than modelling the environment.

## Notes

* The corrupted map is what the bridge is *trained* on and what both generators draw their injected noise from; the dynamics and the evaluation use the true map. Boundary jitter was not added (class flips only), per the prompt.
* Compute: 13 min per seed on the GPU (bridges retrained for each slip level and flip rate on CPU).

## What I'd run next

* Model error applied to the *evaluation* map instead (deploy with a wrong map, generate with the right one) — the mirror case, to separate "wrong labels" from "wrong states".
* The bridge with a Brownian reference but the slip-corrupted *states* (Σ used for noise only): if it lands near PD-noise's curve, the fragility is entirely in the labels.
* A robust-reference bridge: Σ(x) inflated toward its mean (shrinkage) as a function of map confidence, to see whether the fragility can be bought back at the cost of g1's covariance advantage.
