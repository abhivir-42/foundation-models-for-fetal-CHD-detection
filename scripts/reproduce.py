#!/usr/bin/env python3
"""Recompute the all-nine headline numbers from the committed predictions.

This script needs no raw imaging data and no GPU. It reads the per-fold
prediction CSVs under predictions/ (subject_id, y_true, y_prob_transformer,
y_prob_linear) and reports, for each backbone and head:

  - all-nine binary AUROC per fold and the 5-fold mean +/- SD,
  - sensitivity at fixed specificity 0.95 (5-fold mean).

The all-nine task uses the binary y_true shipped in the CSVs (disease vs
healthy across the nine conditions). The auditable-four and per-condition
analyses require the restricted iFIND condition labels (see DATA.md) and are
therefore reproduced only when IFIND_DATA points at the label file; this
script restricts itself to what is reproducible from the committed data alone.

Usage:
  python scripts/reproduce.py
  python scripts/reproduce.py --emit-results-json   # also rebuild the per-fold
                                                    # results.json the figures read
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = Path(os.environ.get("PREDICTIONS_DIR", str(REPO_ROOT / "predictions")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(REPO_ROOT / "results")))

BACKBONES = {
    "FetalCLIP (ViT-L, domain)": "fetalclip",
    "DINOv2 ViT-B/14 (general)": "dinov2_vitb",
    "DINOv2 ViT-L/14 (general)": "dinov2_vitl",
}
# Map each backbone CSV directory to the experiment directory the figures expect.
EXP_DIR = {
    "fetalclip": "exp07_kfold_fetalclip",
    "dinov2_vitb": "exp08_kfold_dinov2",
    "dinov2_vitl": "exp08b_kfold_dinov2large",
}
HEADS = {
    "transformer (F2a)": "y_prob_transformer",
    "linear probe (E2)": "y_prob_linear",
}
# results.json head names the figures key on.
HEAD_KEY = {
    "y_prob_transformer": "F2a_hidden64",
    "y_prob_linear": "E2_mean_pool_baseline",
}
N_FOLDS = 5


def load_fold(backbone_dir: str, fold: int):
    path = PRED_DIR / backbone_dir / f"fold_{fold}.csv"
    rows = list(csv.DictReader(path.open()))
    subject_ids = [int(r["subject_id"]) for r in rows]
    y_true = np.array([int(r["y_true"]) for r in rows])
    probs = {col: np.array([float(r[col]) for r in rows]) for col in
             ("y_prob_transformer", "y_prob_linear")}
    return subject_ids, y_true, probs


def emit_results_json():
    """Rebuild the per-fold results.json files the figure scripts read, from the
    committed prediction CSVs, into RESULTS_DIR. Only the two report heads and
    the all-nine test predictions are reconstructed."""
    for bdir, exp in EXP_DIR.items():
        for f in range(N_FOLDS):
            subject_ids, y_true, probs = load_fold(bdir, f)
            blob = {}
            for col, head in HEAD_KEY.items():
                blob[head] = {
                    "test_predictions": {
                        "y_true": [float(v) for v in y_true.tolist()],
                        "y_prob": [float(v) for v in probs[col].tolist()],
                        "subject_ids": subject_ids,
                    }
                }
            out = RESULTS_DIR / exp / f"fold_{f}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(json.dumps(blob))
    print(f"Wrote per-fold results.json under {RESULTS_DIR} for: "
          f"{', '.join(EXP_DIR.values())}")


def sens_at_spec(y_true, y_prob, target_spec=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    ok = spec >= target_spec
    return float(tpr[ok].max()) if ok.any() else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-results-json", action="store_true",
                        help="Also rebuild the per-fold results.json files the "
                             "figure scripts read, into RESULTS_DIR.")
    args = parser.parse_args()

    print(f"Reading committed predictions from: {PRED_DIR}\n")
    print(f"{'backbone':28s} {'head':20s} {'all-9 AUROC (5-fold)':24s} {'sens@spec0.95':14s}")
    print("-" * 88)
    for bname, bdir in BACKBONES.items():
        for hname, hcol in HEADS.items():
            aurocs, senses = [], []
            for f in range(N_FOLDS):
                _, y_true, probs = load_fold(bdir, f)
                yp = probs[hcol]
                aurocs.append(roc_auc_score(y_true, yp))
                senses.append(sens_at_spec(y_true, yp, 0.95))
            mean, sd = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
            s95 = float(np.nanmean(senses))
            print(f"{bname:28s} {hname:20s} {mean:.4f} +/- {sd:.4f}        {s95:.3f}")
    print(
        "\nNote: these are all-nine binary numbers (disease vs healthy). The "
        "auditable-four headline (AUROC 0.911) and per-condition results require "
        "the restricted iFIND condition labels; see RESULTS.md and DATA.md."
    )

    if args.emit_results_json:
        print()
        emit_results_json()


if __name__ == "__main__":
    main()
