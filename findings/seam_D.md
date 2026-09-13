# Seam D — Break the seam on purpose

**Question.** How tolerant is each method of an unmet precondition? Skills 2 and 3 are trained on the nominal ρ₁, ρ₂; at test time skill 2 is started from a reshaped ρ₁′.
**Setup.** Rough terrain, σ_k = 0.06, held-out layouts 20–29, 200 episodes per cell, **5 seeds**. Shifts: mean translated by δ ∈ {0, 0.5, 1, 2, 3}σ along the corridor normal (lateral) and along the heading axis (along-track) separately; covariance rotated by 45° and 90°; covariance shrunk to 0.25× and grown to 4×. Conditions: BRIDGE-tracked, BRIDGE-slip-tracked, TSM (its initiation set is measured, so "shifting it" means the same shifted entries; the fraction of entries inside the set is reported).
**Pre-registered pass.** BRIDGE degrades more slowly with δ than TSM. Fail: both cliff at the same δ, or TSM is flatter.
**Verdict: FAIL (vacuous — nobody cliffs).** Over the whole shift range, success stays at 0.88–0.94 for every condition; the largest effect is −0.024 for BRIDGE-tracked at a 3σ lateral shift, and along-track shifts *raise* bridge success (+0.03) because they shorten the remaining path. No condition drops below 0.5 at any δ. TSM is flattest (+0.01 over the sweep), so if anything the fail clause "TSM is flatter" holds. With the same PD tracker in the loop, an entry 3σ off the nominal cloud is simply corrected; the seam is not load-bearing at this shift range.

## Success of skills 2 + 3 from a shifted ρ₁′ (mean ± 95% CI over seeds)

| condition | axis | δ = 0 | 0.5σ | 1σ | 2σ | 3σ | first δ with success < 0.5 |
|---|---|---|---|---|---|---|---|
| BRIDGE-tracked | normal (lateral) | 0.906 ± 0.012 | 0.906 ± 0.015 | 0.904 ± 0.015 | 0.893 ± 0.013 | 0.882 ± 0.009 | none |
| BRIDGE-slip-tracked | normal | 0.920 ± 0.016 | 0.916 ± 0.012 | 0.916 ± 0.015 | 0.913 ± 0.018 | 0.902 ± 0.009 | none |
| TSM | normal | 0.932 ± 0.026 | 0.935 ± 0.020 | 0.936 ± 0.019 | 0.937 ± 0.019 | 0.942 ± 0.016 | none |
| BRIDGE-tracked | heading (along-track) | 0.906 ± 0.012 | 0.924 ± 0.009 | 0.926 ± 0.012 | 0.936 ± 0.015 | 0.938 ± 0.014 | none |
| BRIDGE-slip-tracked | heading | 0.920 ± 0.016 | 0.929 ± 0.016 | 0.932 ± 0.020 | 0.933 ± 0.022 | 0.919 ± 0.017 | none |
| TSM | heading | 0.932 ± 0.026 | 0.936 ± 0.021 | 0.938 ± 0.019 | 0.944 ± 0.016 | 0.945 ± 0.012 | none |

| condition | nominal | rotate 45° | rotate 90° | shrink 0.25× | grow 4× |
|---|---|---|---|---|---|
| BRIDGE-tracked | 0.906 ± 0.012 | 0.906 | 0.906 | 0.918 ± 0.010 | 0.890 ± 0.009 |
| BRIDGE-slip-tracked | 0.920 ± 0.016 | 0.920 | 0.920 | 0.929 ± 0.013 | 0.896 ± 0.021 |
| TSM | 0.932 ± 0.026 | 0.932 | 0.932 | 0.932 ± 0.026 | 0.925 ± 0.026 |

Fraction of shifted entries inside TSM's measured initiation set: 0.99 nominal; 1.00 for every lateral shift; 0.98 / 0.97 / 0.89 / 0.74 for along-track 0.5 / 1 / 2 / 3σ; 0.84 for grow 4×.

## Reading the table

* **Rotations are exact no-ops** — the nominal ρ₁ is isotropic (σ = 0.05 in both axes), so a rotated covariance is the same covariance; the rows are identical by construction and are reported to close the item.
* **Lateral 3σ = 0.15 m off the corridor** costs the bridges 0.02 and TSM nothing. In Phase C the same nominal distance was associated with a 0.3–0.6 drop; the difference is that Phase C's "outside" entries were the delivered states of episodes being pushed hard, which went on being pushed, while here the entry is a clean sample and the episode is at σ_k = 0.06. The precondition gap in Phase C is therefore mostly a selection effect, not the causal cost of starting outside the cloud.
* **Grow 4× (entry std 0.10) costs 0.02–0.03**, the largest reshaping effect: a few entries now start 0.2–0.3 m off, beyond what a 100-step skill recovers.
* **TSM is the most tolerant** and even improves with lateral shift; its initiation set covers the whole ±3σ lateral band (fraction inside 1.00), so nothing here is outside its precondition either. Along-track shifts of 3σ take 26 % of entries outside the set with no visible cost.
* None of this contradicts Phase A's ordering (TSM ≈ naive > bridges): it is the same ordering at every δ.

## Notes

* Phase D starts the episode at skill 2 with the shifted entry and runs skills 2 and 3 (200 steps); success is the usual ρ₃ 2σ test.
* Compute: 2.2 min per seed on the GPU.

## What I'd run next

* Shifts of 4–8σ and shifts combined with σ_k = 0.15, to find where the cliff actually is for each method (Phase C's binned curves put the bridge's at ≈ 3.5σ under continued pushes).
* Break the seam *inside* the skill instead of at its start (a single large push at τ = 0.5) — the seam guarantee is about the handoff, but the failure mode Phase C saw is mid-skill.
* An anisotropic nominal ρ₁ so the rotation items have content.
