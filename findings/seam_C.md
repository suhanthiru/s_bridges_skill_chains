# Seam C — Seam containment

**Question.** Is the precondition cloud load-bearing: does a skill entered outside its ρ_{k−1} 2σ ellipse succeed less often than one entered inside?
**Data.** All Phase A and Phase B (w = 1) episodes: for skill 2, P(delivered inside ρ₂ | entered inside / outside ρ₁'s 2σ); for skill 3, P(success | entered inside / outside ρ₂'s 2σ); plus the same probability binned by Mahalanobis entry distance. Entry distances arise naturally from the pushes, so this is observational on the same episodes (Phase D manipulates them directly). 5 seeds, 4 000–7 000 episodes per condition and terrain.
**Pre-registered pass.** For at least one skill the inside/outside gap is ≥ 0.20 and the binned curve falls monotonically past ≈ 2σ.
**Verdict: PASS — for the bridge skills specifically.** Both bridge conditions show gaps of +0.33 to +0.65 on every skill and terrain, with binned curves that are flat inside the cloud and fall steeply beyond 2σ (mild, skill 3: 0.93 at 1.75σ → 0.70 at 2.25σ → 0.25 at 2.75σ → 0.05 at 3.5σ). The trackers show gaps of only +0.06 to +0.13 with nearly flat curves, and TSM +0.14 to +0.19 with a drop that starts later (≈ 2.5σ, the edge of its measured initiation set). The composition guarantee is not vacuous in this environment: a bridge skill's precondition is exactly where its competence ends. Whether that is a virtue is another matter — Phase A showed the trackers, which do not need the precondition, succeed more overall.

## Inside vs outside the precondition cloud

| condition | terrain | skill | P(ok \| inside 2σ) | P(ok \| outside) | gap | n inside / outside |
|---|---|---|---|---|---|---|
| TRACK-oracle | mild | 2 | 0.911 | 0.826 | +0.085 | 3 540 / 460 |
| TRACK-naive | mild | 2 | 0.926 | 0.818 | +0.108 | 3 594 / 406 |
| BRIDGE-tracked | mild | 2 | 0.920 | 0.440 | **+0.479** | 3 446 / 554 |
| BRIDGE-slip-tracked | mild | 2 | 0.935 | 0.432 | **+0.503** | 6 087 / 913 |
| TSM | mild | 2 | 0.658 | 0.517 | +0.141 | 2 480 / 1 520 |
| TRACK-oracle | mild | 3 | 0.912 | 0.846 | +0.066 | 3 605 / 395 |
| TRACK-naive | mild | 3 | 0.916 | 0.859 | +0.057 | 3 659 / 341 |
| BRIDGE-tracked | mild | 3 | 0.911 | 0.370 | **+0.541** | 3 413 / 587 |
| BRIDGE-slip-tracked | mild | 3 | 0.926 | 0.278 | **+0.648** | 6 085 / 915 |
| TSM | mild | 3 | 0.914 | 0.726 | +0.187 | 2 417 / 1 583 |
| TRACK-oracle | rough | 2 | 0.846 | 0.732 | +0.115 | 3 299 / 701 |
| TRACK-naive | rough | 2 | 0.878 | 0.748 | +0.130 | 3 405 / 595 |
| BRIDGE-tracked | rough | 2 | 0.880 | 0.547 | **+0.332** | 3 357 / 643 |
| BRIDGE-slip-tracked | rough | 2 | 0.903 | 0.544 | **+0.360** | 6 084 / 916 |
| TSM | rough | 2 | 0.827 | 0.654 | +0.173 | 3 272 / 728 |
| TRACK-oracle | rough | 3 | 0.847 | 0.751 | +0.096 | 3 305 / 695 |
| TRACK-naive | rough | 3 | 0.871 | 0.757 | +0.114 | 3 433 / 567 |
| BRIDGE-tracked | rough | 3 | 0.865 | 0.532 | **+0.333** | 3 305 / 695 |
| BRIDGE-slip-tracked | rough | 3 | 0.896 | 0.522 | **+0.374** | 5 994 / 1 006 |
| TSM | rough | 3 | 0.879 | 0.696 | +0.183 | 3 181 / 819 |

(figure: `figures/seam_C_containment.png`, success vs binned entry distance, four panels)

## Reading the table

* **Inside the cloud, all five conditions are equal** (0.85–0.94). The precondition is not what separates them.
* **Outside the cloud the bridge halves.** A bridge reference regenerated from an entry state 3σ off the corridor is a drift-flow path from a region the net rarely saw; the flow still points toward the goal mean, but the delivered handoff lands outside ρ_k more often than not, and the *next* skill then starts outside its cloud too — the cascade Phase D measures directly. The mild-terrain curve for skill 3 is the cleanest: flat at 0.9 to 2σ, then 0.70 / 0.25 / 0.05 at 2.25 / 2.75 / 3.5σ.
* **The trackers are indifferent to where they start** because their reference does not depend on the entry state (TRACK-naive) or depends on it only through the nearest-demo lookup (TRACK-oracle); kp = 10 pulls a 3σ-off robot back onto the line within a few steps.
* **TSM's competence boundary is its initiation set, not the Gaussian.** Its curve stays flat to ≈ 2.5σ and then falls (mild skill 3: 0.51 at 3.5σ, 0.04 at 5σ) — the classifier was fitted on a ±4σ probe grid with an 80 % success threshold, so its edge is wider than ρ_{k−1}'s 2σ. On mild terrain TSM's skill-2 delivery is low even from inside (0.66) because its fine-tuned BC policy over-shoots ρ₂ laterally on mild ground; that is a TSM training artefact, not a seam effect.
* **Rough terrain softens every curve** (bridge gaps 0.33–0.37 instead of 0.48–0.65) because the delivered state is noisy for everyone; the ordering is unchanged.

## Caveat from Phase D

This analysis is observational: an entry outside ρ_{k−1} is the delivered state of an episode that has just been pushed hard and goes on being pushed. Phase D forces clean entries from a ρ₁′ shifted by up to 3σ and finds the bridges lose at most 0.024 — so most of the gap above is selection (being outside is a symptom of a bad episode), and the causal cost of an unmet precondition at these shift sizes is small.

## What it means for the question

The handoff marginal is load-bearing as a *precondition*: for a bridge skill, "entered inside ρ_{k−1}" predicts delivery, and the guarantee that skill k delivers into skill k+1's precondition is what keeps the chain alive. But Phase A says a tracker that ignores the precondition entirely does better overall, so in this environment the precondition is describing the bridge's limitation rather than buying robustness.

## What I'd run next

* The same analysis with entry distance measured in the *next* skill's own initiation set (probe it as for TSM) for every condition, so all five are judged by their own competence boundary rather than by the Gaussian.
* Phase D-style forced entries at fixed distances for the trackers, to check the flat curves are not a selection effect of pushes landing near the corridor.
* Bridges with a wider training cloud (Phase B's w = 2–4) in this analysis: does widening the cloud move the cliff outward or lower the plateau?
