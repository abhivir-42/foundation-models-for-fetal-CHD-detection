
import os as _os_fig
from pathlib import Path as _Path_fig
REPO_ROOT = _Path_fig(__file__).resolve().parent.parent
_RESULTS = _Path_fig(_os_fig.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results')))
_PREDS   = _Path_fig(_os_fig.environ.get('PREDICTIONS_DIR', str(REPO_ROOT / 'predictions')))
_IFIND   = _Path_fig(_os_fig.environ.get('IFIND_DATA', str(REPO_ROOT / 'data')))
_FIGOUT  = _Path_fig(_os_fig.environ.get('FIGURES_OUT', str(REPO_ROOT / 'figures' / 'out')))
_FIGOUT.mkdir(parents=True, exist_ok=True)

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

MODEL = "#0072B2"   # FetalCLIP domain backbone (blue)
AUDIT = "#999999"   # UK 20-week programme reference (grey)
FLOOR = "#D55E00"   # FASP minimum-acceptable floor (orange reference line)
VV = "#009E73"      # Van Velzen 2018 unselected pooled level (green reference line)

# --- model: FetalCLIP transformer head, 5-fold sens@specificity=0.95 (200 tokens) ---
# per-lesion mean/SD from scaling_curve_percondition.csv (config nf10xnc20);
# lumped auditable-four mean/SD from scaling_curve_perfold.csv (auditable4 row).
# All reconcile with VERIFIED-NUMBERS sec.6.
lesions = ["HLHS", "TGA", "TOF", "AVSD", "Auditable-four\n(lumped)"]
model_mean = np.array([0.8127, 0.5269, 0.5386, 0.5484, 0.5935])
model_sd   = np.array([0.1299, 0.1462, 0.1750, 0.1128, 0.0410])

# --- UK 20-week-audit antenatal detection rate (Aldridge 2023). No published
#     per-group rate for the lumped auditable-four -> no audit bar for that group. ---
audit = np.array([0.927, 0.849, 0.754, 0.692, np.nan])

# --- reference levels ---
FASP_FLOOR = 0.50    # FASP 2023 minimum acceptable standard
VANVELZEN = 0.451    # Van Velzen 2018 pooled unselected antenatal CHD detection

x = np.arange(len(lesions))
w = 0.38

fig, ax = plt.subplots()

# model bars (blue) with +/-1 SD error bar
ax.bar(x - w / 2, model_mean, w, color=MODEL, zorder=3,
       label="Model: FetalCLIP transformer head, sens at specificity 0.95 (5-fold)")
ax.errorbar(x - w / 2, model_mean, yerr=model_sd, fmt="none",
            ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)

# audit bars (grey); skip the lumped group (no published per-group rate)
audit_x = x[:-1] + w / 2
ax.bar(audit_x, audit[:-1], w, color=AUDIT, zorder=3,
       label="UK 20-week audit antenatal detection (Aldridge 2023)")

# reference lines
ax.axhline(FASP_FLOOR, ls="--", lw=1.2, color=FLOOR, zorder=2,
           label="FASP minimum-acceptable floor (0.50)")
ax.axhline(VANVELZEN, ls=":", lw=1.2, color=VV, zorder=2,
           label="Van Velzen 2018 unselected pooled (0.45)")

ax.set_xticks(x)
ax.set_xticklabels(lesions)
ax.set_ylabel("Per-lesion detection sensitivity")
ax.set_ylim(0, 1.0)
ax.grid(axis="x", visible=False)

# honesty note: NOT like-for-like (model read at spec 0.95; programme near spec 0.99)
# placed in the upper-right whitespace above the AVSD / auditable-four groups, where
# the tallest element is the AVSD audit bar (0.69); clears every bar, error-bar cap
# and reference line.
ax.text(0.995, 0.985,
        "Not like-for-like: model read at specificity 0.95;\n"
        "the UK programme operates near specificity 0.99.",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6))

# legend off the data: below the axes
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=1, fontsize=7.6, handlelength=1.6, labelspacing=0.4)

fig.tight_layout()
fig.savefig(str(_FIGOUT / "sonographer_baseline.pdf"),
            bbox_inches="tight")
fig.savefig(str(_FIGOUT / "sonographer_baseline.png"),
            dpi=150, bbox_inches="tight")

print("model mean +/- sd:")
for l, m, s in zip(lesions, model_mean, model_sd):
    print(f"  {l.splitlines()[0]:<18s} {m:.3f} +/- {s:.3f}")
print("audit:", dict(zip(lesions[:-1], audit[:-1])))
print("FASP floor", FASP_FLOOR, "Van Velzen", VANVELZEN)
