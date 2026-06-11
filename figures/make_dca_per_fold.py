#!/usr/bin/env python3
"""Figure dca_per_fold (Ch5 clinical-utility, the per-fold decision curves).

Five small decision-curve panels, one per subject-disjoint fold (subject_id % 5,
seed 42), on the auditable-four lumped binary task (TGA/AVSD/TOF/HLHS positives
vs healthy-only negatives, slicing-i). Each panel plots net benefit vs threshold
probability p_t in [0.005, 0.5] (the full clinically-plausible band) for:
  - the DINOv2 mean-pooling linear probe (E2 on DINOv2 ViT-B/14, exp08)
  - the FetalCLIP transformer head (F2a hidden=64 on FetalCLIP, exp07)
with refer-all (treat-all, declining) and refer-none (NB=0, flat) references.

The point: the DINOv2 mean-pooling linear probe sits above refer-none in ALL
FIVE folds (the dual finding is not an averaging artefact).

Curves recomputed from disk via src/eval/dca.py on the canonical 5-fold test
predictions (NO training). Net benefit per Vickers & Elkin (2006):
    NB(p_t) = TP/n - FP/n * p_t/(1 - p_t)
House style mirrors make_roc_discrimination.py (DejaVu Serif, prompt palette).
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
from src.eval.dca import (  # noqa: E402
    net_benefit,
    treat_all_net_benefit,
)

LABELS_PATH = str(_IFIND / "subject_level_labels.csv")
EXP07 = str(_RESULTS / "exp07_kfold_fetalclip")  # FetalCLIP
EXP08 = str(_RESULTS / "exp08_kfold_dinov2")     # DINOv2 ViT-B/14
N_FOLDS = 5
OUT = _FIGOUT

# Prompt house-style palette (matches make_roc_discrimination.py)
BLUE = "#0072B2"    # FetalCLIP / domain
ORANGE = "#D55E00"  # DINOv2 / general
GREY = "#999999"    # reference / neutral

# (legend label, exp dir, head, colour, zorder)
CONFIGS = [
    ("DINOv2 mean-pooling linear probe", EXP08, "E2_mean_pool_baseline", ORANGE, 5),
    ("FetalCLIP transformer head", EXP07, "F2a_hidden64", BLUE, 4),
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


def fold_arrays(d: str, head: str, fold: int, labels: pd.DataFrame):
    yt, yp, _ = slicing_auditable_four(load_preds(d, head, fold), labels)
    return np.asarray(yt, int), np.asarray(yp, float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_PATH).set_index("subject")
    labels["unhealthy"] = (labels[list(CONDITIONS)].sum(axis=1) > 0).astype(int)

    pt = np.linspace(0.005, 0.50, 100)  # full clinically-plausible band; skip p_t=0 (undefined)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, N_FOLDS, figsize=(9.6, 3.3), sharey=True)

    verify = {}  # (head, fold) -> NB@{0.10,0.15,0.20} for printout
    for f in range(N_FOLDS):
        ax = axes[f]
        # task labels identical across configs (same subjects) -> take from first
        yt0, _ = fold_arrays(CONFIGS[0][1], CONFIGS[0][2], f, labels)
        n = len(yt0)
        prev = float(yt0.mean())

        # reference strategies
        nb_none = np.zeros_like(pt)
        nb_all = np.array([treat_all_net_benefit(yt0, p) for p in pt])
        ax.plot(pt, nb_none, ls=":", lw=1.0, color="#404040", zorder=2,
                label="refer-none (NB = 0)")
        ax.plot(pt, nb_all, ls="--", lw=1.1, color=GREY, zorder=1,
                label="refer-all")

        for label, d, head, col, z in CONFIGS:
            yt, yp = fold_arrays(d, head, f, labels)
            nb = np.array([net_benefit(yt, yp, p) for p in pt])
            ax.plot(pt, nb, color=col, lw=1.8, zorder=z, label=label)
            verify[(head, f)] = {
                p: float(net_benefit(yt, yp, p)) for p in (0.10, 0.15, 0.20)
            }

        ax.set_title(f"Fold {f}\n$n$ = {n}, prev. {prev:.3f}", fontsize=8.5)
        ax.set_xlim(0.0, 0.50)
        ax.set_ylim(-0.15, 0.11)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.set_xticklabels(["0", "0.25", "0.5"])
        ax.set_xlabel("threshold probability $p_t$", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("net benefit")

    # one shared legend, off the data, centred below the panels (single row)
    handles, lbls = axes[0].get_legend_handles_labels()
    # plotted order: refer-none(0), refer-all(1), DINOv2(2), FetalCLIP(3)
    order = [2, 3, 0, 1]  # models first, then references
    handles = [handles[i] for i in order]
    lbls = [lbls[i] for i in order]
    fig.legend(handles, lbls, loc="lower center", frameon=False, fontsize=8.5,
               ncol=4, bbox_to_anchor=(0.5, -0.02), handlelength=1.8,
               columnspacing=1.8)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "dca_per_fold.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "dca_per_fold.png", dpi=150, bbox_inches="tight",
                pad_inches=0.06)
    plt.close(fig)

    # verification printout: NB and beats-treat-none per fold
    print("VERIFY (auditable-4, per fold) NB at p_t = 0.10/0.15/0.20:")
    for head_name, d, head in [("DINOv2 linear (E2)", EXP08, "E2_mean_pool_baseline"),
                                ("FetalCLIP transformer (F2a)", EXP07, "F2a_hidden64")]:
        means20 = []
        for f in range(N_FOLDS):
            v = verify[(head, f)]
            beats = {p: (v[p] > 0) for p in (0.10, 0.15, 0.20)}
            means20.append(v[0.20])
            print(f"  {head_name:30s} fold {f}: "
                  f"NB10={v[0.10]:+.4f} NB15={v[0.15]:+.4f} NB20={v[0.20]:+.4f}  "
                  f">none@10/15/20={beats[0.10]}/{beats[0.15]}/{beats[0.20]}")
        n_pos20 = sum(verify[(head, f)][0.20] > 0 for f in range(N_FOLDS))
        print(f"  -> {head_name}: NB@0.20 5-fold mean = {np.mean(means20):+.4f}, "
              f"beats refer-none on {n_pos20}/5 folds at p_t=0.20\n")
    print(f"Wrote {OUT/'dca_per_fold.pdf'}")
    print(f"Wrote {OUT/'dca_per_fold.png'}")


if __name__ == "__main__":
    main()
