#!/usr/bin/env python3
"""Figure decision_curve (Ch5 §5.3 / Discussion §6.3): the 5-fold-mean decision curve.

Net benefit vs threshold probability p_t on the auditable-four lumped binary task
(TGA/AVSD/TOF/HLHS positives vs healthy-only negatives, slicing-i), per-fold mean
across the five subject-disjoint folds (subject_id % 5, seed 42), for:
  - the DINOv2 mean-pooling linear probe (E2 on DINOv2 ViT-B/14, exp08)   [general]
  - the FetalCLIP transformer head      (F2a hidden=64 on FetalCLIP, exp07) [domain]
with refer-all (treat-all) and refer-none (NB=0) references.

The threshold probability is an ASSUMPTION (Vickers & Elkin 2006 is general-medicine,
not fetal), so the full curve is shown across the whole clinically-plausible band
p_t in [0.005, 0.5] rather than committing to a single threshold. The linear probe
stays above refer-none across the low-threshold (miss-averse) range a screening
service would use; the transformer head, the stronger ranker, does not.

Net benefit per Vickers & Elkin (2006): NB(p_t) = TP/n - FP/n * p_t/(1 - p_t).
Recomputed from disk on the canonical 5-fold test predictions (NO training).
House style mirrors make_dca_per_fold.py.
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

# REPO_ROOT provided by scrub header
sys.path.insert(0, str(REPO_ROOT))
from src.eval.per_condition import (  # noqa: E402
    CONDITIONS,
    ConfigPredictions,
    slicing_auditable_four,
)
from src.eval.dca import net_benefit, treat_all_net_benefit  # noqa: E402

LABELS_PATH = str(_IFIND / "subject_level_labels.csv")
EXP07 = str(_RESULTS / "exp07_kfold_fetalclip")  # FetalCLIP
EXP08 = str(_RESULTS / "exp08_kfold_dinov2")     # DINOv2 ViT-B/14
N_FOLDS = 5
OUT = _FIGOUT

BLUE = "#0072B2"    # FetalCLIP / domain
ORANGE = "#D55E00"  # DINOv2 / general
GREY = "#999999"

CONFIGS = [
    ("DINOv2 linear probe (general)", EXP08, "E2_mean_pool_baseline", ORANGE, 5),
    ("FetalCLIP transformer head (domain)", EXP07, "F2a_hidden64", BLUE, 4),
]


def load_preds(d, head, fold):
    tp = json.loads(Path(f"{d}/fold_{fold}/results.json").read_text())[head]["test_predictions"]
    return ConfigPredictions(name=head, y_prob=np.asarray(tp["y_prob"], float),
                             subject_ids=np.asarray(tp["subject_ids"]), n_params=None)


def fold_arrays(d, head, fold, labels):
    yt, yp, _ = slicing_auditable_four(load_preds(d, head, fold), labels)
    return np.asarray(yt, int), np.asarray(yp, float)


def main():
    labels = pd.read_csv(LABELS_PATH).set_index("subject")
    labels["unhealthy"] = (labels[list(CONDITIONS)].sum(axis=1) > 0).astype(int)

    pt = np.linspace(0.005, 0.50, 100)  # full clinically-plausible band, not just [0,0.3]

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "figure.figsize": (6.6, 4.0), "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots()

    # refer-none (NB = 0)
    ax.axhline(0.0, color="#404040", lw=1.0, ls=":", zorder=2)
    ax.text(0.495, 0.004, "refer none", color="#404040", fontsize=8, ha="right", va="bottom")

    # refer-all = 5-fold mean of treat-all NB
    nb_all = np.zeros_like(pt)
    for f in range(N_FOLDS):
        yt0, _ = fold_arrays(CONFIGS[0][1], CONFIGS[0][2], f, labels)
        nb_all += np.array([treat_all_net_benefit(yt0, p) for p in pt])
    nb_all /= N_FOLDS
    ax.plot(pt, nb_all, ls="--", lw=1.1, color=GREY, zorder=1)
    ax.text(0.075, nb_all[np.argmin(np.abs(pt - 0.075))] - 0.004, "refer all",
            color=GREY, fontsize=8, ha="left", va="top")

    nb20 = {}
    for label, d, head, col, z in CONFIGS:
        nb = np.zeros_like(pt)
        for f in range(N_FOLDS):
            yt, yp = fold_arrays(d, head, f, labels)
            nb += np.array([net_benefit(yt, yp, p) for p in pt])
        nb /= N_FOLDS
        ax.plot(pt, nb, color=col, lw=2.0, zorder=z, label=label)
        nb20[label] = nb[np.argmin(np.abs(pt - 0.20))]

    # shade the low-threshold (miss-averse) band a screening service would use
    ax.axvspan(0.005, 0.20, color="#000000", alpha=0.035, zorder=0)
    ax.text(0.105, -0.123, "clinically-plausible referral band", color="#666666",
            fontsize=7.5, ha="center", va="bottom", style="italic")

    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(-0.13, 0.11)
    ax.set_xlabel(r"Threshold probability $p_t$  (a higher $p_t$ = a missed case judged less costly)")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "decision_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "decision_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote decision_curve.pdf/.png")
    for k, v in nb20.items():
        print(f"  NB@p_t=0.20  {k}: {v:+.4f}")


if __name__ == "__main__":
    main()
