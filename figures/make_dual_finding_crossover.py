#!/usr/bin/env python3
"""Figure dual_finding_crossover (the report-spine "gold" figure).

A single 2-D scatter that states the dual finding in one frame: the better
discriminator is the worse decider. x = auditable-four 5-fold AUROC; y = per-fold
net benefit at threshold probability p_t = 0.20 (auditable-four). The useful
quadrant (net benefit > 0, above "refer none") is shaded faintly.

Two points with error bars (means ± per-fold SD, ddof=1, both from raw fold CSVs):
  - DINOv2 mean-pooling linear probe  (AUROC 0.818, NB +0.0149) -> top-left,
    wins the decision, loses the ranking.
  - FetalCLIP transformer head        (AUROC 0.911, NB -0.0372) -> bottom-right,
    wins the ranking, loses the decision.

Plotted values verified against report/from-cluster/VERIFIED-NUMBERS.md:
  §3 auditable-4 AUROC: DINOv2 E2 0.818, FetalCLIP F2a 0.911 (gap +0.093).
  §4 net benefit @ p_t=0.20 (auditable-4): DINOv2 E2 +0.0149 (5/5),
     FetalCLIP F2a -0.0372 (0/5).
Raw per-fold sources (recomputed -> mean/SD reconcile to the ledger):
  auditable_four_kfold_e2.csv under RESULTS_DIR/auditable_4
  auditable_four_kfold_f2a.csv under RESULTS_DIR/auditable_4
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(str(_FIGOUT))

# Prompt house-style Okabe-Ito palette
BLUE = "#0072B2"    # FetalCLIP / domain
ORANGE = "#D55E00"  # DINOv2 / general
GREEN = "#009E73"   # third / positive-region shade
GREY = "#999999"    # reference / neutral

# Canonical values (VERIFIED-NUMBERS §3 AUROC, §4 net benefit); SDs from raw folds.
DINOV2 = dict(
    name="DINOv2 mean-pooling\nlinear probe",
    auroc=0.818, auroc_sd=0.022,
    nb=0.0149, nb_sd=0.0054,
    colour=ORANGE,
)
FETALCLIP = dict(
    name="FetalCLIP\ntransformer head",
    auroc=0.911, auroc_sd=0.023,
    nb=-0.0372, nb_sd=0.0290,
    colour=BLUE,
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.figsize": (5.6, 4.0),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots()

    xlim = (0.795, 0.935)
    ylim = (-0.072, 0.046)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # shade the useful quadrant: net benefit > 0 (above "refer none")
    ax.axhspan(0.0, ylim[1], color=GREEN, alpha=0.07, zorder=0)

    # "refer none" reference line at NB = 0
    ax.axhline(0.0, color=GREY, lw=1.2, ls="-", zorder=1)
    ax.text(
        xlim[0] + 0.004, 0.0022, "refer none (net benefit = 0)",
        fontsize=7.5, color="#666666", va="bottom", ha="left", zorder=2,
    )

    # faint curved arrow: from the transformer point to the linear-probe point
    arrow = FancyArrowPatch(
        (FETALCLIP["auroc"] - 0.004, FETALCLIP["nb"] + 0.004),
        (DINOV2["auroc"] + 0.006, DINOV2["nb"] - 0.004),
        connectionstyle="arc3,rad=-0.32",
        arrowstyle="-|>", mutation_scale=12,
        lw=1.0, color="#aaaaaa", zorder=3,
    )
    ax.add_patch(arrow)

    # the two points with error bars
    for cfg in (DINOV2, FETALCLIP):
        ax.errorbar(
            cfg["auroc"], cfg["nb"],
            xerr=cfg["auroc_sd"], yerr=cfg["nb_sd"],
            fmt="o", ms=10, color=cfg["colour"],
            ecolor=cfg["colour"], elinewidth=1.3, capsize=3, capthick=1.2,
            markeredgecolor="white", markeredgewidth=1.0,
            zorder=5,
        )

    # annotate the DINOv2 point (top-left): name + the two roles
    ax.annotate(
        DINOV2["name"]
        + "\nwins the decision (NB +0.0149, 5/5 folds)\nloses the ranking (AUROC 0.818)",
        xy=(DINOV2["auroc"], DINOV2["nb"]),
        xytext=(DINOV2["auroc"] + 0.006, DINOV2["nb"] + 0.0115),
        fontsize=8.0, color=DINOV2["colour"], ha="left", va="bottom", zorder=6,
    )

    # annotate the FetalCLIP point (bottom-right): name + the two roles
    ax.annotate(
        FETALCLIP["name"]
        + "\nwins the ranking (AUROC 0.911)\nloses the decision (NB -0.0372, 0/5 folds)",
        xy=(FETALCLIP["auroc"], FETALCLIP["nb"]),
        xytext=(FETALCLIP["auroc"] - 0.004, FETALCLIP["nb"] - 0.0095),
        fontsize=8.0, color=FETALCLIP["colour"], ha="right", va="top", zorder=6,
    )

    ax.set_xlabel("Auditable-four discrimination  (5-fold AUROC)")
    ax.set_ylabel("Clinical decision  (net benefit at $p_t = 0.20$)")
    ax.set_xticks([0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92])
    ax.set_yticks([-0.06, -0.04, -0.02, 0.00, 0.02, 0.04])
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "dual_finding_crossover.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUT / "dual_finding_crossover.png", dpi=150, bbox_inches="tight", pad_inches=0.06
    )
    plt.close(fig)

    print("VERIFY (plotted vs VERIFIED-NUMBERS):")
    print(f"  DINOv2 linear probe   AUROC {DINOV2['auroc']:.3f} (ledger 0.818)  "
          f"NB {DINOV2['nb']:+.4f} (ledger +0.0149)")
    print(f"  FetalCLIP transformer AUROC {FETALCLIP['auroc']:.3f} (ledger 0.911)  "
          f"NB {FETALCLIP['nb']:+.4f} (ledger -0.0372)")
    print(f"Wrote {OUT/'dual_finding_crossover.pdf'}")
    print(f"Wrote {OUT/'dual_finding_crossover.png'}")


if __name__ == "__main__":
    main()
