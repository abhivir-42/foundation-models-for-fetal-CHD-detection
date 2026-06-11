#!/usr/bin/env python3
"""Figure roc_discrimination (Ch4 §4.1 headline).

Mean-ROC on the auditable-four task (TGA/AVSD/TOF/HLHS positives vs healthy-only,
slicing-i negatives) for the TRANSFORMER head on each of the three backbones, with
the HEAD HELD FIXED so only the backbone varies (the controlled, scale-matched
comparison). This puts the dimension-matched general-purpose backbone on the same
figure as the domain one, so the gain reads as domain pretraining, not model size:
  - FetalCLIP transformer        (ViT-L, domain)            AUROC 0.911
  - DINOv2 ViT-L/14 transformer  (general, size-matched)    AUROC 0.837
  - DINOv2 ViT-B/14 transformer  (general, smaller)         AUROC 0.825
FetalCLIP leads the size-matched ViT-L general backbone by 0.074 and the smaller
ViT-B by 0.086; enlarging the general backbone B->L recovers only 0.012.

Curves are the vertical (fold-averaged) ROC over a common FPR grid: the AUC of a
fold-averaged ROC equals the per-fold mean AUROC, so the drawn curve matches the
headline AUROC. Operating dots at spec 0.95 / 0.99 are on the FetalCLIP transformer
curve (5-fold mean sens@spec).
"""
from __future__ import annotations


import os as _os_fig
from pathlib import Path as _Path_fig
REPO_ROOT = _Path_fig(__file__).resolve().parent.parent
_RESULTS = _Path_fig(_os_fig.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results')))
_PREDS   = _Path_fig(_os_fig.environ.get('PREDICTIONS_DIR', str(REPO_ROOT / 'predictions')))
_IFIND   = _Path_fig(_os_fig.environ.get('IFIND_DATA', str(REPO_ROOT / 'data')))
_FIGOUT  = _Path_fig(_os_fig.environ.get('FIGURES_OUT', str(REPO_ROOT / 'figures' / 'out')))
_FIGOUT.mkdir(parents=True, exist_ok=True)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# REPO_ROOT provided by scrub header
sys.path.insert(0, str(REPO_ROOT))
from src.eval.per_condition import (  # noqa: E402
    CONDITIONS,
    ConfigPredictions,
    slicing_auditable_four,
    sensitivity_at_specificity,
)

LABELS_PATH = str(_IFIND / "subject_level_labels.csv")
EXP07 = str(_RESULTS / "exp07_kfold_fetalclip")        # FetalCLIP
EXP08 = str(_RESULTS / "exp08_kfold_dinov2")           # DINOv2 ViT-B/14
EXP08B = str(_RESULTS / "exp08b_kfold_dinov2large")    # DINOv2 ViT-L/14
N_FOLDS = 5
OUT = _FIGOUT

# Prompt house-style palette
BLUE = "#0072B2"     # FetalCLIP / domain
ORANGE = "#D55E00"   # DINOv2 ViT-L/14 / general, size-matched
AMBER = "#E69F00"    # DINOv2 ViT-B/14 / general, smaller
GREY = "#999999"     # reference / chance

# Transformer head fixed; only the backbone varies (the controlled comparison).
# (legend label, exp dir, head, colour, linestyle, zorder)
CONFIGS = [
    ("FetalCLIP (domain, ViT-L)", EXP07, "F2a_hidden64", BLUE, "-", 5),
    ("DINOv2 ViT-L/14 (general)", EXP08B, "F2a_hidden64", ORANGE, "-", 4),
    ("DINOv2 ViT-B/14 (general)", EXP08, "F2a_hidden64", AMBER, "-", 3),
]


def load_preds(d: str, head: str, fold: int) -> ConfigPredictions:
    tp = json.loads(Path(f"{d}/fold_{fold}/results.json").read_text())[head][
        "test_predictions"
    ]
    return ConfigPredictions(
        name=head,
        y_prob=np.asarray(tp["y_prob"], dtype=float),
        subject_ids=np.asarray(tp["subject_ids"]),
        n_params=None,
    )


def fold_arrays(d: str, head: str, labels: pd.DataFrame):
    yts, yps = [], []
    for f in range(N_FOLDS):
        yt, yp, _ = slicing_auditable_four(load_preds(d, head, f), labels)
        yts.append(np.asarray(yt, int))
        yps.append(np.asarray(yp, float))
    return yts, yps


def mean_roc(yts, yps, grid):
    """Vertical (fold-averaged) ROC: mean TPR over folds at each FPR on grid.
    AUC of this mean curve == per-fold-mean AUROC (trapezoid on a dense grid)."""
    tprs = []
    aucs = []
    for yt, yp in zip(yts, yps):
        fpr, tpr, _ = roc_curve(yt, yp)
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(yt, yp))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return mean_tpr, float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH).set_index("subject")
    labels["unhealthy"] = (labels[list(CONDITIONS)].sum(axis=1) > 0).astype(int)

    grid = np.linspace(0.0, 1.0, 2001)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.figsize": (5.4, 3.6),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots()

    # chance diagonal
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=GREY, zorder=1)

    verify = {}
    fc_tpr_curve = None
    for label, d, head, col, ls, z in CONFIGS:
        yts, yps = fold_arrays(d, head, labels)
        mtpr, au_mean, au_sd = mean_roc(yts, yps, grid)
        ax.plot(
            grid,
            mtpr,
            color=col,
            ls=ls,
            lw=2,
            zorder=z,
            label=f"{label} (AUROC {au_mean:.3f})",
        )
        # per-fold-mean sens@spec for operating dots
        s95 = np.nanmean(
            [sensitivity_at_specificity(yt, yp, 0.95)[0] for yt, yp in zip(yts, yps)]
        )
        s99 = np.nanmean(
            [sensitivity_at_specificity(yt, yp, 0.99)[0] for yt, yp in zip(yts, yps)]
        )
        verify[label] = dict(auroc=au_mean, auroc_sd=au_sd, sens95=s95, sens99=s99)
        if d == EXP07:  # operating dots on the FetalCLIP (headline) curve only
            fc_tpr_curve = (mtpr, s95, s99, col)

    # operating points on the FetalCLIP transformer curve (the headline config)
    mtpr, s95, s99, col = fc_tpr_curve
    # FPR = 1 - spec; dot sits on the mean curve at that FPR
    for spec, sens in ((0.95, s95), (0.99, s99)):
        fpr = 1.0 - spec
        y_on_curve = float(np.interp(fpr, grid, mtpr))
        ax.plot(fpr, y_on_curve, "o", color=col, ms=6, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)

    # labels for the two operating points (placed to avoid the curves)
    f95 = 1.0 - 0.95
    f99 = 1.0 - 0.99
    y95 = float(np.interp(f95, grid, mtpr))
    y99 = float(np.interp(f99, grid, mtpr))
    ax.annotate(
        f"specificity 0.95, sensitivity {s95:.2f}",
        xy=(f95, y95),
        xytext=(f95 + 0.16, y95 + 0.10),
        fontsize=8,
        color="#333333",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#777777"),
        ha="left",
        va="center",
    )
    ax.annotate(
        f"specificity 0.99, sensitivity {s99:.2f}",
        xy=(f99, y99),
        xytext=(f99 + 0.16, y99 + 0.07),
        fontsize=8,
        color="#333333",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#777777"),
        ha="left",
        va="center",
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_aspect("equal", adjustable="box")
    # legend sits in the empty triangle below the chance diagonal (high-FPR
    # region) so it never touches the ROC curves or the operating-point labels
    ax.legend(loc="upper left", frameon=False, fontsize=8.5,
              bbox_to_anchor=(0.44, 0.27))
    fig.tight_layout()

    fig.savefig(OUT / "roc_discrimination.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUT / "roc_discrimination.png", dpi=150, bbox_inches="tight", pad_inches=0.06
    )
    plt.close(fig)

    print("VERIFY:")
    for k, v in verify.items():
        print(
            f"  {k:32s} AUROC={v['auroc']:.4f}±{v['auroc_sd']:.4f}  "
            f"sens@95={v['sens95']:.4f}  sens@99={v['sens99']:.4f}"
        )
    print(f"Wrote {OUT/'roc_discrimination.pdf'}")
    print(f"Wrote {OUT/'roc_discrimination.png'}")


if __name__ == "__main__":
    main()
