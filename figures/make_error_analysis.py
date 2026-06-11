
import os as _os_fig
from pathlib import Path as _Path_fig
REPO_ROOT = _Path_fig(__file__).resolve().parent.parent
_RESULTS = _Path_fig(_os_fig.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results')))
_PREDS   = _Path_fig(_os_fig.environ.get('PREDICTIONS_DIR', str(REPO_ROOT / 'predictions')))
_IFIND   = _Path_fig(_os_fig.environ.get('IFIND_DATA', str(REPO_ROOT / 'data')))
_FIGOUT  = _Path_fig(_os_fig.environ.get('FIGURES_OUT', str(REPO_ROOT / 'figures' / 'out')))
_FIGOUT.mkdir(parents=True, exist_ok=True)

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.figsize": (6.6, 4.0),
})

HEALTHY = "#56B4E9"   # negative class (light blue)
CHD = "#D55E00"       # positive class (vermillion)
THRESH = "#333333"

# Headline FetalCLIP transformer head (F2a_hidden64), all-nine binary task (CHD vs healthy),
# per-fold test predictions pooled across the five subject-disjoint folds.
FOLDS = sorted(glob.glob(str(_RESULTS / "exp07_kfold_fetalclip/fold_*/results.json")))
CFG = "F2a_hidden64"

y_true, y_prob = [], []
for f in FOLDS:
    tp = json.load(open(f))[CFG]["test_predictions"]
    y_true.extend(int(round(v)) for v in tp["y_true"])
    y_prob.extend(float(v) for v in tp["y_prob"])
y_true = np.array(y_true)
y_prob = np.array(y_prob)

neg = y_prob[y_true == 0]
pos = y_prob[y_true == 1]

# operating threshold at specificity 0.95 = 95th percentile of healthy scores
t95 = np.quantile(neg, 0.95)

# pooled AUROC sanity check (rank statistic) vs VERIFIED-NUMBERS all-9 = 0.891
order = np.argsort(y_prob)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(y_prob) + 1)
auroc = (ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print(f"pooled n={len(y_true)} pos={len(pos)} neg={len(neg)} AUROC={auroc:.3f} t@spec95={t95:.3f}")

fig, ax = plt.subplots()
# no median bars: they read as a second threshold line and confuse the figure
parts = ax.violinplot([neg, pos], positions=[1, 2], widths=0.8,
                      showmeans=False, showextrema=False, showmedians=False)
for body, col in zip(parts["bodies"], [HEALTHY, CHD]):
    body.set_facecolor(col)
    body.set_alpha(0.55)
    body.set_edgecolor(col)
    body.set_linewidth(1.0)

# jittered raw points (subsample healthy for legibility)
rng_neg_idx = np.linspace(0, len(neg) - 1, min(len(neg), 600)).astype(int)
ax.scatter(1 + (np.arange(len(rng_neg_idx)) % 7 - 3) * 0.018, neg[rng_neg_idx],
           s=3, color=HEALTHY, alpha=0.35, zorder=2)
ax.scatter(2 + (np.arange(len(pos)) % 7 - 3) * 0.018, pos,
           s=4, color=CHD, alpha=0.45, zorder=2)

ax.axhline(t95, ls="--", lw=1.5, color=THRESH, zorder=3)
ax.text(2.47, t95 + 0.014,
        f"operating threshold = {t95:.2f}\n(set so 95% of healthy fall below it,\ni.e. 0.95 specificity)",
        fontsize=7.5, ha="right", va="bottom", color=THRESH)
# annotate error regions (dark text for legibility on either violin)
ax.annotate("false positives", xy=(1.0, min(0.97, t95 + 0.12)), fontsize=8,
            ha="center", va="bottom", color=THRESH)
ax.annotate("missed cases", xy=(1.60, max(0.03, t95 - 0.18)), fontsize=8,
            ha="center", va="center", color=THRESH)

ax.set_xticks([1, 2])
ax.set_xticklabels([f"Healthy\n(n={len(neg)})", f"Congenital heart\ndisease (n={len(pos)})"])
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("Predicted probability of disease")
ax.set_title("FetalCLIP transformer head, pooled five-fold test predictions", fontsize=10)

fig.tight_layout()
fig.savefig("error_analysis.pdf", bbox_inches="tight")
fig.savefig("error_analysis.png", dpi=200, bbox_inches="tight")
print("wrote error_analysis.pdf / .png")
