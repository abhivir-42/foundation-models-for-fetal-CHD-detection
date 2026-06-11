# Reproduction guide

This guide covers two levels of reproduction:

- **From the committed predictions** (no raw data, no GPU): recompute the
  headline numbers and regenerate the prediction-derived figures.
- **From scratch** (requires the restricted iFIND corpus and a GPU): extract
  embeddings, train the heads, and evaluate end to end.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Python: see `.python-version` (3.12.3).
- PyTorch is pinned to a CUDA 11.8 build (`torch==2.7.1+cu118`). On a CPU-only
  machine, install the CPU build of torch instead; the prediction-level
  reproduction in Section 3 does not need a GPU.

## 2. Paths and the data root

All filesystem locations are read from environment variables in `src/paths.py`.
Nothing is hardcoded to a particular machine.

| Variable | Meaning | Default |
|----------|---------|---------|
| `IFIND_DATA` | Root of the iFIND corpus and `subject_level_labels.csv`. Restricted (see `DATA.md`). | `<repo>/data` |
| `EMBEDDINGS_DIR` | Where cached embedding `.pkl` files are written/read. | `<repo>/embeddings` |
| `RESULTS_DIR` | Where experiment outputs are written/read. | `<repo>/results` |
| `PREDICTIONS_DIR` | Committed per-fold prediction CSVs. | `<repo>/predictions` |
| `FETALCLIP_DIR` | FetalCLIP weights and config (`fetalclip/`, `fetalclip-repo/`). | `<repo>/checkpoints` |

Set the ones you need before running, for example:

```bash
export IFIND_DATA=/path/to/ifind/corpus
export RESULTS_DIR=/path/to/results
```

## 3. Reproduce from the committed predictions (no raw data)

The per-fold test predictions for both report heads on all three backbones are
committed under `predictions/`. Each CSV has columns
`subject_id, y_true, y_prob_transformer, y_prob_linear`, where `y_true` is the
all-nine binary label (disease vs healthy) and `subject_id` is an integer index.

```bash
python scripts/reproduce.py
```

This recomputes, per backbone and head, the all-nine AUROC (per fold and the
5-fold mean) and the sensitivity at fixed specificity 0.95. The numbers match the
values in `RESULTS.md` (for example FetalCLIP transformer all-nine AUROC
0.8906 +/- 0.0185).

To regenerate the prediction-based figures, first materialise the per-fold
`results.json` files (the format the figure scripts read) from the committed
CSVs, then run a figure:

```bash
export RESULTS_DIR=$(pwd)/results
export FIGURES_OUT=$(pwd)/figures/out
python scripts/reproduce.py --emit-results-json
python figures/make_error_analysis.py     # all-nine, predictions-only
```

The figures that depend only on the all-nine predictions (for example the
score-distribution error analysis) regenerate this way without any raw data.
The auditable-four, per-condition, and decision-curve figures additionally need
the restricted condition labels (Section 5), because their task slicings use the
individual condition columns; point `IFIND_DATA` at the label file to run those.

## 4. Reproduce from scratch (requires iFIND + GPU)

With `IFIND_DATA` pointing at the corpus and the FetalCLIP weights in place:

```bash
# (a) extract frozen embeddings
python src/extract_embeddings.py            # DINOv2
python src/extract_raw_fetalclip.py         # FetalCLIP

# (b) train the heads under subject-disjoint five-fold cross-validation
python src/exp07_kfold_fetalclip.py --execute   # FetalCLIP, F2a transformer head
python src/exp08_kfold_dinov2.py --execute --configs E2_mean_pool_baseline F2a_hidden64

# (c) evaluate (per-condition discrimination, calibration, decision curves)
python -m src.eval.per_condition --results $RESULTS_DIR/exp07_kfold_fetalclip/fold_0/results.json
```

Outputs land under `RESULTS_DIR`. The figure scripts then read from there.

## 5. Protocol details

- **Seed.** All training uses a fixed random seed of `42`.
- **Five-fold split.** Folds are subject-disjoint: fold `k` holds the subjects
  whose `subject_id % 5 == k`; the validation set is a seeded 10% of the
  remaining subjects, and the rest are training. All subjects rotate through the
  test position exactly once. This is the canonical estimand: the per-fold metric
  averaged over the five folds (mean +/- SD).
- **Single-split cross-check.** A stratified 70/15/15 split
  (`src/phase6_train_test_split.py`, `random_state=42`, stratified on the binary
  label) is used as a conservative secondary check. It is not a never-tuned
  holdout, so it is reported only as a cross-check, never as the headline.
- **GPU non-determinism.** cuDNN runs in non-deterministic mode for speed, so
  training from scratch reproduces the reported numbers up to small
  floating-point variation. The committed predictions are exact, so the
  prediction-level reproduction in Section 3 is bit-stable.
