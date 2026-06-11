#!/usr/bin/env python3
"""
Experiment 07: 5-fold replication of Tier 2 lead config (F2a) on FetalCLIP.

The single biggest remaining methodological caveat for the dissertation is
that the AUROC = 0.888 result is from a single train / val / test split.
This experiment partitions all 4128 subjects into 5 subject-disjoint folds
by subject_id mod 5, retrains F2a_hidden64 on each fold, and reports
AUROC ± std across folds.

Splits per fold k:
  test_k  = subjects with subject_id mod 5 == k
  val_k   = 10% of (subjects with subject_id mod 5 != k), seed=42
  train_k = rest of (subjects with subject_id mod 5 != k)

Reuses exp05_temporal_modelling for the actual training; monkey-patches
t1.load_data, t1.RAW_EMBEDDINGS_PATH, t1.OUTPUT_PATH, t1.get_experiment_configs.

Usage:
  python src/exp07_kfold_fetalclip.py --execute            # full 5 folds
  python src/exp07_kfold_fetalclip.py --execute --folds 0  # just fold 0
  python src/exp07_kfold_fetalclip.py --dry-run            # generate splits + print, no training
  python src/exp07_kfold_fetalclip.py --check-splits       # CPU-only sanity check; no training
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import exp05_temporal_modelling as t1
from exp06_fetalclip_modelling import (
    FETALCLIP_EMBEDDINGS_PATH, sanity_check_clips_per_subject,
    tag_configs_with_backbone,
)


from paths import RESULTS_DIR

OUTPUT_ROOT = RESULTS_DIR / "exp07_kfold_fetalclip"
LEAD_CONFIG = "F2a_hidden64"
N_FOLDS = 5
VAL_FRAC = 0.10
SEED = 42


def make_kfold_splits(all_subjects: list[int], n_folds: int = N_FOLDS,
                      val_frac: float = VAL_FRAC, seed: int = SEED) -> list[dict]:
    """Subject-disjoint k-fold splits: test = mod==k; val = 10% of rest."""
    rng = np.random.default_rng(seed)
    folds = []
    for k in range(n_folds):
        test = sorted([s for s in all_subjects if int(s) % n_folds == k])
        rest = sorted([s for s in all_subjects if int(s) % n_folds != k])
        rest_arr = np.array(rest)
        rng.shuffle(rest_arr)
        n_val = int(round(len(rest_arr) * val_frac))
        val = sorted(rest_arr[:n_val].tolist())
        train = sorted(rest_arr[n_val:].tolist())
        folds.append({"k": k, "train": train, "val": val, "test": test})
    return folds


def _resolve_labels_path(labels_csv: Path | None = None) -> Path:
    if labels_csv is not None:
        return Path(labels_csv)
    import os as _os
    env = _os.environ.get('SCALING_LABELS_CSV') or _os.environ.get('LABELS_CSV')
    if env:
        return Path(env)
    return t1.DATA_PATH / "subject_level_labels.csv"


def patched_load_data_factory(fold: dict, embeddings_path: Path,
                              labels_csv: Path | None = None):
    """Returns a load_data replacement that uses fold's splits."""
    labels_path = _resolve_labels_path(labels_csv)
    def _load_data(_=None):
        with open(embeddings_path, "rb") as f:
            raw = pickle.load(f)
        t1._infer_embedding_dim(raw)
        labels_df = pd.read_csv(labels_path)
        cond_cols = [c for c in labels_df.columns if c != "subject"]
        labels_df["unhealthy"] = (labels_df[cond_cols].sum(axis=1) > 0).astype(int)
        return raw, labels_df, fold["train"], fold["val"], fold["test"]
    return _load_data


def run_kfold(folds_to_run: list[int] | None, dry_run: bool,
              configs: list[str] | None = None,
              embeddings_path: Path | None = None,
              output_root: Path | None = None,
              labels_csv: Path | None = None):
    embeddings_path = Path(embeddings_path) if embeddings_path else FETALCLIP_EMBEDDINGS_PATH
    output_root = Path(output_root) if output_root else OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    labels_path = _resolve_labels_path(labels_csv)

    with open(embeddings_path, "rb") as f:
        emb = pickle.load(f)
    all_subjects_in_emb = sorted(int(s) for s in emb.keys())
    del emb

    labels_df = pd.read_csv(labels_path)
    cond_cols = [c for c in labels_df.columns if c != "subject"]
    labels_df["unhealthy"] = (labels_df[cond_cols].sum(axis=1) > 0).astype(int)
    all_with_labels = set(labels_df["subject"].astype(int).tolist())
    universe = sorted(set(all_subjects_in_emb) & all_with_labels)
    print(f"Universe: {len(universe)} subjects (in embeddings AND labels).")

    folds = make_kfold_splits(universe)
    unhealthy_by_subject = dict(zip(
        labels_df["subject"].astype(int).tolist(),
        labels_df["unhealthy"].astype(int).tolist(),
    ))
    summary = []
    for fold in folds:
        n_pos_test = int(sum(unhealthy_by_subject.get(int(s), 0) for s in fold["test"]))
        summary.append({
            "k": int(fold["k"]),
            "n_train": int(len(fold["train"])),
            "n_val": int(len(fold["val"])),
            "n_test": int(len(fold["test"])),
            "n_test_pos": n_pos_test,
            "test_prevalence": float(n_pos_test / max(len(fold["test"]), 1)),
        })
    print("\nFold sizes (subject counts; test prevalence in last col):")
    print(pd.DataFrame(summary).to_string(index=False))

    splits_dump = {
        "n_folds": N_FOLDS, "val_frac": VAL_FRAC, "seed": SEED,
        "summary": summary,
        "folds": [{"k": int(f["k"]),
                   "train": [int(s) for s in f["train"]],
                   "val":   [int(s) for s in f["val"]],
                   "test":  [int(s) for s in f["test"]]}
                  for f in folds],
    }
    splits_path = output_root / "kfold_splits.json"
    splits_path.write_text(json.dumps(splits_dump))
    print(f"\nSplits dumped to {splits_path}")

    if dry_run:
        print("\n[dry-run] no training. Re-run with --execute to train.")
        return

    targets = folds_to_run if folds_to_run is not None else list(range(N_FOLDS))
    print(f"\nTraining on folds: {targets}")

    # Skip the FetalCLIP-vs-DINOv2 clips-per-subject sanity check when running
    # on a scaled custom pkl (the comparison only makes sense at the baseline
    # 10f/20c configuration). Re-enable via env var if needed.
    import os as _os
    if embeddings_path == FETALCLIP_EMBEDDINGS_PATH and not _os.environ.get('SKIP_SANITY_CHECK'):
        from paths import DINOV2_EMBEDDINGS
        sanity_check_clips_per_subject(
            pickle.load(open(FETALCLIP_EMBEDDINGS_PATH, "rb")),
            dinov2_path=str(DINOV2_EMBEDDINGS),
        )

    selected_configs = configs if configs is not None else [LEAD_CONFIG]
    print(f"Configs to train per fold: {selected_configs}")

    original_load_data = t1.load_data
    original_get_configs = t1.get_experiment_configs
    t1.RAW_EMBEDDINGS_PATH = embeddings_path

    def _get_configs_subset():
        cfgs = original_get_configs()
        cfgs = OrderedDict((k, v) for k, v in cfgs.items() if k in selected_configs)
        return tag_configs_with_backbone(cfgs, "none")

    t1.get_experiment_configs = _get_configs_subset

    for k in targets:
        fold = folds[k]
        fold_dir = output_root / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        t1.OUTPUT_PATH = fold_dir
        t1.load_data = patched_load_data_factory(fold, embeddings_path, labels_csv=labels_path)
        print(f"\n=== Fold {k} ===")
        print(f"  train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])}")
        print(f"  output: {fold_dir}")
        t1.run_all_experiments(
            quick=False,
            selected=selected_configs,
            embeddings_path=str(embeddings_path),
            resume=True,  # skip configs already in fold's results.json
        )

    t1.load_data = original_load_data
    t1.get_experiment_configs = original_get_configs


def main():
    import os as _os
    parser = argparse.ArgumentParser(description="Exp07: 5-fold replication of F2a on FetalCLIP")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run training. Without this flag, prints the plan + splits.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate splits + print summary; no training.")
    parser.add_argument("--check-splits", action="store_true",
                        help="CPU-only: dump splits + summary; no embedding load.")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Subset of fold indices to run (default: all 5).")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of TIER2_SUBSET to train per fold "
                             "(default: F2a_hidden64 only).")
    parser.add_argument("--embeddings-path", type=Path, default=None,
                        help="Override FETALCLIP_EMBEDDINGS_PATH (e.g. for a custom/scaled embeddings pickle). "
                             "Defaults to env var SCALING_EMBEDDINGS_PATH if set.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Override the per-fold output root (e.g. for an alternative output root). "
                             "Defaults to env var SCALING_OUTPUT_DIR if set.")
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="Override subject_level_labels.csv path (e.g. for a "
                             "subject_level_labels_coa_cleaned.csv). Defaults to env var "
                             "SCALING_LABELS_CSV or LABELS_CSV if set.")
    args = parser.parse_args()

    if args.check_splits:
        labels_path = _resolve_labels_path(args.labels_csv)
        labels_df = pd.read_csv(labels_path)
        universe = sorted(labels_df["subject"].astype(int).tolist())
        print(f"Using labels from: {labels_path}")
        print(f"Universe (subjects in labels): {len(universe)}")
        folds = make_kfold_splits(universe)
        for f in folds:
            print(f"fold {f['k']}: train={len(f['train'])}, val={len(f['val'])}, test={len(f['test'])}")
        return

    embeddings_path = args.embeddings_path
    if embeddings_path is None and _os.environ.get('SCALING_EMBEDDINGS_PATH'):
        embeddings_path = Path(_os.environ['SCALING_EMBEDDINGS_PATH'])
    output_root = args.output_root
    if output_root is None and _os.environ.get('SCALING_OUTPUT_DIR'):
        output_root = Path(_os.environ['SCALING_OUTPUT_DIR'])

    run_kfold(folds_to_run=args.folds, dry_run=(args.dry_run or not args.execute),
              configs=args.configs,
              embeddings_path=embeddings_path,
              output_root=output_root,
              labels_csv=args.labels_csv)


if __name__ == "__main__":
    main()
