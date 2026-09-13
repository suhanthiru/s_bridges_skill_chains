# g3 — Does the downstream policy class matter?

**Question.** Is the g0 ordering (PD-noise data > BRIDGE-slip data) a property of the diffusion policy, or of the data?
**Setup.** BRIDGE-slip vs PD-noise as training data for four chunk policies with identical architecture width, optimiser and 8 000-step budget: diffusion (DDPM-50 / DDIM-10), conditional flow matching (10 Euler steps), plain BC-MLP (MSE on the 8×3 chunk), BC with a 5-component Gaussian-mixture head (NLL; executes the highest-weight component's mean). L1 and L2, slip disturbance (and none), 200 episodes, **5 seeds**; the diffusion rows are g0's policies.
**Pre-registered pass.** The sign of the BRIDGE − PD gap is the same across all four classes. Fail: the bridge only helps diffusion.
**Verdict: FAIL on the letter, PASS on the substance.** For diffusion, flow matching and plain BC the sign is the same — the tracker's data is better, and by more for the simpler policies (L1: −0.11, −0.21, −0.21; L2: 0.00, −0.09, −0.22). The mixture-head BC flips the sign (+0.48 on L1, +0.27 on L2), but only because it fails to fit the tracker's data at all: its undisturbed success on PD-noise data is 0.45 / 0.44 against 0.99–1.00 for every other (class, source) pair, with seed-to-seed swings of ± 0.27. That is a fitting pathology of a 5-component GMM on 100 near-path trajectories with nearly deterministic labels, not evidence for the bridge; on the three classes that fit, the result is about the data. The plain BC-MLP is also the best downstream policy in this regime (0.88 under slip on L1 vs 0.76 flow, 0.62 diffusion).

## Success under slip (mean ± 95% CI over seeds)

| layout | policy | BRIDGE-slip | PD-noise | Δ (BRIDGE − PD) | 95% CI | p |
|---|---|---|---|---|---|---|
| L1 | diffusion | 0.513 ± 0.085 | 0.622 ± 0.071 | −0.109 | [−0.177, −0.041] | 0.011 |
| L1 | flow | 0.555 ± 0.069 | 0.762 ± 0.030 | −0.207 | [−0.261, −0.153] | <0.001 |
| L1 | bc | 0.666 ± 0.077 | **0.879 ± 0.029** | −0.213 | [−0.302, −0.124] | 0.003 |
| L1 | bc_gmm | 0.718 ± 0.066 | 0.235 ± 0.268 | +0.483 | [+0.206, +0.760] | 0.008 |
| L2 | diffusion | 0.309 ± 0.056 | 0.309 ± 0.152 | +0.000 | [−0.130, +0.130] | 1.00 |
| L2 | flow | 0.351 ± 0.065 | 0.445 ± 0.083 | −0.094 | [−0.216, +0.028] | 0.10 |
| L2 | bc | 0.474 ± 0.035 | **0.692 ± 0.044** | −0.218 | [−0.240, −0.196] | <0.001 |
| L2 | bc_gmm | 0.480 ± 0.066 | 0.213 ± 0.158 | +0.267 | [+0.103, +0.431] | 0.011 |

## Success undisturbed

| layout | policy | BRIDGE-slip | PD-noise |
|---|---|---|---|
| L1 | diffusion / flow / bc | 0.993 / 0.998 / 0.994 | 0.993 / 0.989 / 0.999 |
| L1 | bc_gmm | 0.996 | **0.452** |
| L2 | diffusion / flow / bc | 0.944 / 0.931 / 0.959 | 0.722 / 0.957 / 0.999 |
| L2 | bc_gmm | 0.972 | **0.435** |

## Reading the tables

* **The data ordering is class-independent where the class fits.** Three different objectives (denoising, flow regression, chunk regression) trained on the same two datasets rank them the same way, and the simpler the policy the larger the tracker's lead — a chunk regressor on kp = 10 labels is close to the tracker itself (0.88 vs the nominal PD's 0.81 in Phase 2).
* **The bridge's data is easier to fit with a mixture.** BRIDGE-slip's 29% off-path states give the GMM head enough spread to place its components; the tracker's data (0.4% off-path, labels a near-deterministic function of state) makes the NLL objective collapse components and the argmax-component controller is then wrong in 2–3 seeds out of 5. This is a genuine difference between the datasets — the bridge's is more diverse — but it manifests as robustness of an ill-suited estimator, not as a better policy: the GMM trained on bridge data (0.72) is still below plain BC trained on tracker data (0.88).
* **L2 diffusion ties** (0.31 vs 0.31) because PD-noise's diffusion policy is seed-unstable on L2 (± 0.15, one seed near zero, also visible in g0); flow and BC on the same data do not have that problem (0.96–1.00 undisturbed), so it is the diffusion fit, not the data.

## Notes

* All four classes share `phase2.obs_of` inputs, the 8-step chunk / 4-step execution, batch 1 024, Adam 3e-4 and 8 000 steps. Flow matching integrates 10 Euler steps from Gaussian noise; the GMM head clamps log-σ to [−5, 2].
* Compute: 8.6 min per seed on the GPU (12 new policies; diffusion re-used).

## What I'd run next

* Plain BC as the downstream policy for g0's six sources (it is the strongest class here and cheapest); if DART > MPC > PD > BRIDGE holds for it too, the suite's conclusion is class-independent for the best class.
* GMM head with a variance floor or fewer components on the tracker data, to confirm the flip is estimator fragility.
* Single-step (non-chunked) BC, which also repairs the MPC-relabel isolation of g1.
