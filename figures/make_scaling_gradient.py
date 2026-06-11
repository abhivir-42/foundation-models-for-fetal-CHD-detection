#!/usr/bin/env python
"""scaling_gradient (Ch4, §4.3): sampling-budget gradient, FetalCLIP head fixed."""

import os as _os_fig
from pathlib import Path as _Path_fig
REPO_ROOT = _Path_fig(__file__).resolve().parent.parent
_RESULTS = _Path_fig(_os_fig.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results')))
_PREDS   = _Path_fig(_os_fig.environ.get('PREDICTIONS_DIR', str(REPO_ROOT / 'predictions')))
_IFIND   = _Path_fig(_os_fig.environ.get('IFIND_DATA', str(REPO_ROOT / 'data')))
_FIGOUT  = _Path_fig(_os_fig.environ.get('FIGURES_OUT', str(REPO_ROOT / 'figures' / 'out')))
_FIGOUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.figsize": (5.4, 3.6),
})

CSV = str(_RESULTS / "scaling_curve/scaling_curve_perfold.csv")
df = pd.read_csv(CSV)
a4 = df[df.task == "auditable4"].sort_values("total_tokens").reset_index(drop=True)

tok = a4.total_tokens.values
auroc = a4.auroc_mean.values
auroc_sd = a4.auroc_sd.values
sens = a4.sens_at_spec95_mean.values
sens_sd = a4.sens_at_spec95_sd.values
nf = a4.num_frames.values
nc = a4.max_clips.values

C_AUROC = "#0072B2"   # blue (domain head)
C_SENS = "#009E73"    # green (third series)
C_REF = "#999999"     # grey reference

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.grid(False)
ax2.spines["top"].set_visible(False)

# AUROC on left axis
l1 = ax1.errorbar(tok, auroc, yerr=auroc_sd, color=C_AUROC, lw=2,
                  marker="o", ms=5, capsize=3, capthick=1,
                  label="Auditable-four AUROC")
# sens@spec95 on right axis
l2 = ax2.errorbar(tok, sens, yerr=sens_sd, color=C_SENS, lw=2,
                  marker="s", ms=5, capsize=3, capthick=1,
                  label="Sensitivity at 95% specificity")

ax1.set_xscale("log")
ax1.set_xticks(tok)
# label frames x clips under the token count
ax1.set_xticklabels([f"{t}\n({f}x{c})" for t, f, c in zip(tok, nf, nc)])
ax1.set_xlabel("Per-subject token budget  (frames x clips)")
ax1.set_ylabel("Auditable-four AUROC", color=C_AUROC)
ax2.set_ylabel("Sensitivity at 95% specificity", color=C_SENS)
ax1.tick_params(axis="y", labelcolor=C_AUROC)
ax2.tick_params(axis="y", labelcolor=C_SENS)

# axis ranges chosen so both gradients read clearly
ax1.set_ylim(0.86, 0.97)
ax2.set_ylim(0.50, 0.82)

# mark 200-token baseline and 3600 asymptote
for x in (tok[0], tok[-1]):
    ax1.axvline(x, color=C_REF, lw=0.8, ls=":", zorder=0)
ax1.text(tok[0]*1.08, 0.966, "baseline", color=C_REF, fontsize=8, va="top", ha="left")
ax1.text(tok[-1]*0.92, 0.966, "asymptote (~20x)", color=C_REF, fontsize=8, va="top", ha="right")

# endpoint value labels (offsets nudged so text clears the markers/error bars)
ax1.annotate(f"{auroc[0]:.3f}", (tok[0], auroc[0]), textcoords="offset points",
             xytext=(10, 6), fontsize=8, color=C_AUROC)
ax1.annotate(f"{auroc[-1]:.3f}", (tok[-1], auroc[-1]), textcoords="offset points",
             xytext=(-9, 9), fontsize=8, color=C_AUROC, ha="right")
ax2.annotate(f"{sens[0]:.3f}", (tok[0], sens[0]), textcoords="offset points",
             xytext=(10, -14), fontsize=8, color=C_SENS)
ax2.annotate(f"{sens[-1]:.3f}", (tok[-1], sens[-1]), textcoords="offset points",
             xytext=(-9, -16), fontsize=8, color=C_SENS, ha="right")

# combined legend, frame off, lower-centre (clear of rising data top-left->right)
handles = [l1, l2]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, frameon=False, loc="lower center",
           fontsize=9, bbox_to_anchor=(0.52, 0.0))

fig.tight_layout()
for ext in ("pdf", "png"):
    dpi = 150 if ext == "png" else None
    fig.savefig(fstr(_FIGOUT / "scaling_gradient.{ext}"),
                dpi=dpi, bbox_inches="tight")
print("saved scaling_gradient.pdf/.png")
print("AUROC:", list(np.round(auroc, 4)))
print("sens@95:", list(np.round(sens, 4)))
print("tokens:", list(tok))
