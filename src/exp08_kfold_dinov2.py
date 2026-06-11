#!/usr/bin/env python3
"""5-fold replication on DINOv2 (Tier 1 baseline) --- k-fold mirror of exp07.

Same subject-disjoint mod-5 partition as exp07_kfold_fetalclip.
Backbone is frozen DINOv2 ViT-L/14 (the Tier 1 baseline).
This validates the Tier 1 calibration / DCA finding (E2 best-calibrated)
across folds, and lets us check whether the architecture-flip story
(E2 wins on DINOv2; F2a wins on FetalCLIP) holds across data partitions.

Reuses exp07's split generator, monkey-patch pattern, and CLI flags.

Usage:
  python src/exp08_kfold_dinov2.py --execute --configs E2_mean_pool_baseline F2a_hidden64
  python src/exp08_kfold_dinov2.py --execute --configs B2_cross_attn  # one-arch run
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
from exp06_fetalclip_modelling import tag_configs_with_backbone
from exp07_kfold_fetalclip import (
    make_kfold_splits, patched_load_data_factory,
    N_FOLDS, VAL_FRAC, SEED,
)


from paths import DINOV2_EMBEDDINGS, RESULTS_DIR

DINOV2_EMBEDDINGS_PATH = DINOV2_EMBEDDINGS
OUTPUT_ROOT = RESULTS_DIR / "exp08_kfold_dinov2"
TIER1_DEFAULT_CONFIGS = ["E2_mean_pool_baseline", "F2a_hidden64", "B2_cross_attn"]


def run_kfold_dinov2(folds_to_run: list[int] | None, dry_run: bool,
                     configs: list[str] | None = None,
                     embeddings_path: Path | None = None,
                     output_root: Path | None = None,
                     labels_csv: Path | None = None):
    from exp07_kfold_fetalclip import _resolve_labels_path
    embeddings_path = Path(embeddings_path) if embeddings_path else DINOV2_EMBEDDINGS_PATH
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
    print(f"Universe (DINOv2 embeddings + labels): {len(universe)} subjects.")

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
    print("\nFold sizes (DINOv2):")
    print(pd.DataFrame(summary).to_string(index=False))

    splits_dump = {
        "n_folds": N_FOLDS, "val_frac": VAL_FRAC, "seed": SEED,
        "backbone": "DINOv2",
        "summary": summary,
        "folds": [{"k": int(f["k"]),
                   "train": [int(s) for s in f["train"]],
                   "val":   [int(s) for s in f["val"]],
                   "test":  [int(s) for s in f["test"]]}
                  for f in folds],
    }
    splits_path = output_root / "kfold_splits.json"
    splits_path.write_text(json.dumps(splits_dump))
    print(f"Splits dumped to {splits_path}")

    if dry_run:
        print("[dry-run] no training. Re-run with --execute to train.")
        return

    targets = folds_to_run if folds_to_run is not None else list(range(N_FOLDS))
    selected_configs = configs if configs is not None else TIER1_DEFAULT_CONFIGS
    print(f"\nTraining configs {selected_configs} on folds {targets}")

    original_load_data = t1.load_data
    original_get_configs = t1.get_experiment_configs
    t1.RAW_EMBEDDINGS_PATH = embeddings_path

    def _get_configs_subset():
        cfgs = original_get_configs()
        cfgs = OrderedDict((k, v) for k, v in cfgs.items() if k in selected_configs)
        return tag_configs_with_backbone(cfgs, "none")

    # Reset the FetalCLIP backbone tag back to DINOv2 to avoid lying in metadata.
    # tag_configs_with_backbone hardcodes "FetalCLIP"; override here.
    def _get_configs_subset_dinov2():
        cfgs = _get_configs_subset()
        for name, cfg in cfgs.items():
            cfg["embedding_backbone"] = "DINOv2"
        return cfgs

    t1.get_experiment_configs = _get_configs_subset_dinov2

    for k in targets:
        fold = folds[k]
        fold_dir = output_root / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        t1.OUTPUT_PATH = fold_dir
        t1.load_data = patched_load_data_factory(fold, embeddings_path, labels_csv=labels_path)
        print(f"\n=== Fold {k} (DINOv2) ===")
        print(f"  train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])}")
        print(f"  output: {fold_dir}")
        t1.run_all_experiments(
            quick=False,
            selected=selected_configs,
            embeddings_path=str(embeddings_path),
            resume=True,
        )

    t1.load_data = original_load_data
    t1.get_experiment_configs = original_get_configs


def main():
    import os as _os
    parser = argparse.ArgumentParser(description="Exp08: 5-fold replication on DINOv2")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None,
                        help="Override DINOV2_EMBEDDINGS_PATH (e.g. for a custom/scaled embeddings pickle). "
                             "Defaults to env var SCALING_EMBEDDINGS_PATH if set.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Override the per-fold output root (e.g. for an alternative output root). "
                             "Defaults to env var SCALING_OUTPUT_DIR if set.")
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="Override subject_level_labels.csv path (e.g. for a "
                             "subject_level_labels_coa_cleaned.csv). Defaults to env var "
                             "SCALING_LABELS_CSV or LABELS_CSV if set.")
    args = parser.parse_args()

    embeddings_path = args.embeddings_path
    if embeddings_path is None and _os.environ.get('SCALING_EMBEDDINGS_PATH'):
        embeddings_path = Path(_os.environ['SCALING_EMBEDDINGS_PATH'])
    output_root = args.output_root
    if output_root is None and _os.environ.get('SCALING_OUTPUT_DIR'):
        output_root = Path(_os.environ['SCALING_OUTPUT_DIR'])

    run_kfold_dinov2(folds_to_run=args.folds, dry_run=(args.dry_run or not args.execute),
                     configs=args.configs,
                     embeddings_path=embeddings_path,
                     output_root=output_root,
                     labels_csv=args.labels_csv)


if __name__ == "__main__":
    main()
