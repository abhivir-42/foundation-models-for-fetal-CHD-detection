#!/usr/bin/env python3
"""Figure embedding_umap (embedding-space view; UMAP / t-SNE projection).

Subject-level UMAP of frozen DINOv2 ViT-B/14 mean-pooled embeddings (768-dim,
n=4128 subjects: 3488 structurally-normal, 640 any-CHD). Left panel: normal cloud
(grey) vs any-CHD (domain blue), alpha-blended so the rare minority is visible.
Right inset: two single-condition highlights (HLHS, TGA) on the same projection.

Honest reading (matches the experiment README): the foundation embedding already
imposes strong 2-D structure, but that structure does NOT track the disease label
(silhouette +0.018, essentially zero) — the normal cloud dominates and any-CHD is
mixed throughout. The minority region is the hard part; shaping the space is the
problem. No claim of visual separation is made.

Source coords + labels: umap_subject_coords.csv under RESULTS_DIR/umap_per_disease.
Silhouette: metrics.txt under RESULTS_DIR/umap_per_disease.
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

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(str(_RESULTS / "umap_per_disease"))
COORDS = EXP / "umap_subject_coords.csv"
METRICS = EXP / "metrics.txt"
OUT = Path(str(_FIGOUT))

# Prompt house-style Okabe-Ito palette
BLUE = "#0072B2"    # FetalCLIP/domain -> here the affected (any-CHD) minority
ORANGE = "#D55E00"  # DINOv2/general
GREEN = "#009E73"   # third series
GREY = "#999999"    # reference / neutral -> normal (healthy) cloud


def read_silhouette() -> float:
    for line in METRICS.read_text().splitlines():
        if line.startswith("silhouette_unhealthy_vs_healthy_umap:"):
            return float(line.split(":")[1])
    raise RuntimeError("silhouette line not found")


def main() -> None:
    df = pd.read_csv(COORDS)
    sil = read_silhouette()

    normal = df["unhealthy"] == 0
    chd = df["unhealthy"] == 1
    n_normal = int(normal.sum())
    n_chd = int(chd.sum())
    n_hlhs = int((df["hlhs"] == 1).sum())
    n_tga = int((df["tga"] == 1).sum())

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })

    fig = plt.figure(figsize=(9.2, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.95, 1.0], wspace=0.22, hspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    # ---- main panel: normal cloud vs any-CHD ----
    ax.scatter(df.loc[normal, "umap_0"], df.loc[normal, "umap_1"],
               s=7, c=GREY, alpha=0.30, linewidths=0,
               label=f"Structurally normal (n={n_normal})", zorder=1, rasterized=True)
    ax.scatter(df.loc[chd, "umap_0"], df.loc[chd, "umap_1"],
               s=12, c=BLUE, alpha=0.65, linewidths=0,
               label=f"Any congenital heart disease (n={n_chd})", zorder=3,
               rasterized=True)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # legend OFF the data: data lives in x>=-3.4; put legend in the empty
    # upper-left wedge above the lower-left blob, frameon=False.
    ax.legend(frameon=False, loc="upper left", fontsize=8.4,
              bbox_to_anchor=(-0.01, 1.005), handletextpad=0.4,
              borderaxespad=0.0, labelspacing=0.5, markerscale=1.6)

    # silhouette annotation (separability number) in empty lower-right wedge
    ax.text(0.985, 0.03,
            f"Normal vs CHD silhouette = {sil:+.3f}\n"
            "(no 2-D disease separation:\nthe minority region is the hard part)",
            transform=ax.transAxes, fontsize=7.8, color="#333333",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6,
                      alpha=0.85))

    # ---- inset panels: single-condition highlights on the SAME projection ----
    for axi, cond, label, n_pos in [
        (ax_h, "hlhs", "HLHS", n_hlhs),
        (ax_t, "tga", "TGA", n_tga),
    ]:
        pos = df[cond] == 1
        axi.scatter(df.loc[~pos, "umap_0"], df.loc[~pos, "umap_1"],
                    s=3, c=GREY, alpha=0.22, linewidths=0, zorder=1,
                    rasterized=True)
        axi.scatter(df.loc[pos, "umap_0"], df.loc[pos, "umap_1"],
                    s=11, c=BLUE, alpha=0.85, linewidths=0, zorder=3,
                    rasterized=True)
        axi.set_xticks([])
        axi.set_yticks([])
        axi.grid(False)
        axi.set_title(f"{label} (n={n_pos})", fontsize=8.6, pad=2.5)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "embedding_umap.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "embedding_umap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"silhouette={sil:+.4f}")
    print(f"n_total={len(df)} n_normal={n_normal} n_chd={n_chd} "
          f"hlhs={n_hlhs} tga={n_tga}")
    print("saved:", OUT / "embedding_umap.pdf")


if __name__ == "__main__":
    main()
