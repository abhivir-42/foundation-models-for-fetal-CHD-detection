#!/usr/bin/env python3
"""Figures: FetalCLIP subject-level UMAP (standalone) + DINOv2-vs-FetalCLIP compare.

KEY QUESTION: does the domain-pretrained FetalCLIP backbone organise the latent
space by disease better than general-purpose DINOv2? Answer (honest): no. The
normal-vs-CHD 2-D silhouette is FetalCLIP -0.011 vs DINOv2 +0.018 -- both are
essentially zero; the CHD signal is high-dimensional, not 2-D-visible.

House style: DejaVu Serif, font 10, top/right spines off, grid alpha 0.25;
healthy = grey, any-CHD = #0072B2; legend frameon=False off the data; UMAP-1/-2
axes, no ticks, no title.
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
COORDS_D = EXP / "umap_subject_coords.csv"
METRICS_D = EXP / "metrics.txt"
COORDS_F = EXP / "umap_subject_coords_fetalclip.csv"
METRICS_F = EXP / "metrics_fetalclip.txt"
OUT = Path(str(_FIGOUT))

BLUE = "#0072B2"   # any-CHD minority
GREY = "#999999"   # structurally-normal cloud


def read_silhouette(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("silhouette_unhealthy_vs_healthy_umap:"):
            return float(line.split(":")[1])
    raise RuntimeError(f"silhouette line not found in {path}")


HOUSE = {
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
}


def scatter_panel(ax, df, n_normal, n_chd, *, legend, sil_text=None,
                  legend_loc="upper left", legend_anchor=(-0.01, 1.005)):
    normal = df["unhealthy"] == 0
    chd = df["unhealthy"] == 1
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
    if legend:
        ax.legend(frameon=False, loc=legend_loc, fontsize=8.4,
                  bbox_to_anchor=legend_anchor, handletextpad=0.4,
                  borderaxespad=0.0, labelspacing=0.5, markerscale=1.6)
    if sil_text is not None:
        ax.text(0.985, 0.03, sil_text, transform=ax.transAxes, fontsize=7.8,
                color="#333333", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc",
                          lw=0.6, alpha=0.85))


def main() -> None:
    df_d = pd.read_csv(COORDS_D)
    df_f = pd.read_csv(COORDS_F)
    sil_d = read_silhouette(METRICS_D)
    sil_f = read_silhouette(METRICS_F)

    n_normal = int((df_f["unhealthy"] == 0).sum())
    n_chd = int((df_f["unhealthy"] == 1).sum())
    assert int((df_d["unhealthy"] == 0).sum()) == n_normal
    assert int((df_d["unhealthy"] == 1).sum()) == n_chd

    plt.rcParams.update(HOUSE)

    # ---- standalone FetalCLIP figure ----
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    scatter_panel(
        ax, df_f, n_normal, n_chd, legend=True,
        sil_text=(f"Normal vs CHD silhouette = {sil_f:+.3f}\n"
                  "(no 2-D disease separation:\nthe minority region is the hard part)"),
    )
    fig.savefig(OUT / "embedding_umap_fetalclip.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "embedding_umap_fetalclip.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- side-by-side comparison: DINOv2 (left) vs FetalCLIP (right) ----
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.5))
    scatter_panel(axl, df_d, n_normal, n_chd, legend=False,
                  sil_text=f"silhouette = {sil_d:+.3f}")
    scatter_panel(axr, df_f, n_normal, n_chd, legend=False,
                  sil_text=f"silhouette = {sil_f:+.3f}")
    axl.set_title("DINOv2 (general-purpose)", fontsize=10, pad=6)
    axr.set_title("FetalCLIP (domain-pretrained)", fontsize=10, pad=6)

    # shared legend, frameon=False, above both panels off the data
    handles, labels = axl.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4, columnspacing=1.6, markerscale=1.6)
    fig.subplots_adjust(wspace=0.12, bottom=0.13)

    fig.savefig(OUT / "embedding_umap_compare.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(OUT / "embedding_umap_compare.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"DINOv2  silhouette = {sil_d:+.4f}")
    print(f"FetalCLIP silhouette = {sil_f:+.4f}")
    print(f"n_total={len(df_f)} n_normal={n_normal} n_chd={n_chd}")
    print("saved:", OUT / "embedding_umap_fetalclip.pdf")
    print("saved:", OUT / "embedding_umap_compare.pdf")


if __name__ == "__main__":
    main()
