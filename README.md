# Foundation Models for Fetal Congenital Heart Disease Detection

A controlled evaluation of whether frozen embeddings from a general-purpose
vision foundation model (DINOv2) versus a domain-pretrained one (FetalCLIP) can
detect fetal congenital heart disease (CHD) from second-trimester ultrasound, and
whether the resulting classifiers are clinically useful at screening prevalence.
Backbones are frozen; only lightweight classification heads are trained, so the
comparison isolates what each pretrained representation already encodes about the
fetal heart. The study covers binary detection (disease versus healthy across nine
conditions), an auditable four-condition subset (TGA, AVSD, TOF, HLHS), per-condition
discrimination, calibration, and decision-curve net benefit at realistic prevalence.

## Quickstart

The headline numbers and the figures derived from model predictions can be
reproduced without the raw imaging data, because the per-fold predictions are
committed to this repository.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/reproduce.py        # recompute the all-nine headline numbers
```

See [REPRODUCE.md](REPRODUCE.md) for the full pipeline (embedding extraction,
head training, evaluation, figure regeneration) and for which steps need the
restricted iFIND corpus.

## Repository map

| Path | Contents |
|------|----------|
| `src/` | Embedding extraction, classification-head training, and evaluation modules. Filesystem locations are resolved from environment variables in `src/paths.py`. |
| `figures/` | Scripts that regenerate the report figures from the committed predictions (and, where noted, the restricted labels). |
| `predictions/` | Per-fold test predictions for the two report heads on each backbone, as CSVs. These drive `scripts/reproduce.py` and the figures without the raw data. |
| `scripts/` | `reproduce.py`: recompute headline numbers from the committed predictions. |
| `report/` | The dissertation PDF. |
| `requirements.txt`, `.python-version` | Pinned environment. |
| `DATA.md`, `RESULTS.md`, `REPRODUCE.md` | Data governance, headline results, and reproduction guide. |

## For collaborators and future students

The pipeline is built around three independent extension seams:

1. **Swap the backbone.** Extraction is per-backbone (`src/extract_embeddings.py`
   for DINOv2, `src/extract_raw_fetalclip.py` for FetalCLIP). Adding a new frozen
   encoder means writing one extractor that emits the same per-subject embedding
   format; the heads and evaluation are backbone-agnostic.
2. **Swap the head.** Heads are defined in `src/exp05_temporal_modelling.py` and
   selected by name (for example the `F2a_hidden64` transformer or the
   `E2_mean_pool_baseline` linear probe). New heads slot into the same training
   and k-fold harness (`src/exp07_kfold_fetalclip.py`, `src/exp08_kfold_dinov2.py`).
3. **Swap the task.** The label resolution and slicing logic
   (`src/eval/per_condition.py`) define the binary, auditable-four, and
   per-condition tasks; a new target (a different condition grouping, or a held-out
   anomaly) is a new slicing function over the same predictions.





## License and citation

Code is released under the MIT License (see `LICENSE`); the iFIND imaging data is
**not** included and is governed separately (see `DATA.md`). Please cite the
dissertation and this repository as described in `CITATION.cff`.

## Acknowledgements

This work was carried out as an MEng final-year dissertation in the Department of
Computing, Imperial College London, supervised by Professor Bernhard Kainz, with
guidance from Matt Baugh, and uses data from the iFIND research programme. Full
acknowledgements are in the dissertation.
