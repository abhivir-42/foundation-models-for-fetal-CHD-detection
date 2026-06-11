# Headline results

All numbers below use the canonical estimand: the per-fold metric under the
subject-disjoint five-fold split, reported as the mean over folds (with SD where
shown). Numbers labelled otherwise are explicitly noted. The full set with
confidence intervals appears in the dissertation (`report/report.pdf`).

## Discrimination

| Task | Backbone + head | AUROC (5-fold) | Reproduced by |
|------|-----------------|----------------|---------------|
| Auditable-four | FetalCLIP + transformer (F2a) | **0.911 +/- 0.022** | `figures/make_roc_discrimination.py` (needs labels) |
| All-nine | FetalCLIP + transformer (F2a) | **0.891 +/- 0.019** | `scripts/reproduce.py`, `predictions/fetalclip/` |
| Auditable-four | DINOv2 ViT-B/14 + linear probe (E2) | 0.818 +/- 0.022 | `figures/make_roc_discrimination.py` (needs labels) |
| All-nine | DINOv2 ViT-B/14 + linear probe (E2) | 0.797 | `scripts/reproduce.py`, `predictions/dinov2_vitb/` |

The backbone-swap discrimination gap (FetalCLIP transformer minus DINOv2 linear
probe, auditable-four) is +0.093.

## Sensitivity at fixed specificity

- **Auditable-four sensitivity at specificity 0.95** (FetalCLIP transformer):
  **0.593** (5-fold mean). Above the 50% FASP screening floor; below the UK
  audit per-condition range.
- Per-condition sensitivity at specificity 0.95 (auditable-four conditions):
  HLHS 0.813, AVSD 0.548, TOF 0.539, TGA 0.527 — all clear the 50% floor.

## Scale-matched transformer comparison (auditable-four)

Holding the transformer head fixed and varying only the backbone resolves the
size confound: at matched ViT-L capacity, the domain backbone still leads.

| Backbone (transformer head) | AUROC (5-fold) |
|-----------------------------|----------------|
| FetalCLIP (ViT-L, domain) | **0.911** |
| DINOv2 ViT-L/14 (general, size-matched) | **0.837** |
| DINOv2 ViT-B/14 (general, smaller) | **0.825** |

Reproduced by `figures/make_roc_discrimination.py`. Enlarging the general
backbone from ViT-B to ViT-L recovers only about one seventh of the gap, so the
advantage is attributable to domain pretraining rather than model size.

## Clinical utility: the decision-curve dual finding

Decision-curve net benefit at a 0.20 decision threshold, auditable-four task:

| Backbone + head | Net benefit @ p_t=0.20 | Folds clearing treat-none |
|-----------------|------------------------|---------------------------|
| DINOv2 ViT-B/14 + linear probe (E2) | **+0.0149** | **5/5** |
| FetalCLIP + transformer (F2a) | **-0.0372** | **0/5** |

Reproduced by `figures/make_dca_per_fold.py` and `figures/make_decision_curve.py`
(both need the labels). The domain transformer wins discrimination but the
general-purpose linear probe wins the clinical decision: the simplest head is the
calibration lever. The +0.0149 figure is a five-fold-mean property; the linear
probe also clears treat-none 5/5 on the larger and the domain backbones
(+0.0250 DINOv2-large, +0.0290 FetalCLIP).

## Predictive values at screening prevalence

Transporting operating-point sensitivity and specificity to a 1% disease prior:

- Positive predictive value approximately **0.11**.
- Negative predictive value approximately **0.996**.

These quantify the cost of low prevalence: a confident negative is highly
trustworthy, while a positive flag is mostly a request for confirmatory review.

## Reproduction note

`scripts/reproduce.py` recomputes the all-nine AUROC and sensitivity-at-specificity
numbers directly from the committed prediction CSVs, with no raw data. The
auditable-four and per-condition numbers use the same committed predictions but
additionally require the restricted iFIND condition labels for task slicing (see
`DATA.md`).
