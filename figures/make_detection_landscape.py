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
    "figure.figsize": (6.6, 3.2),
})

# Okabe-Ito, consistent with make_sonographer_baseline.py
LMIC = "#56B4E9"   # low-resource selected series (light blue)
VV = "#009E73"     # unselected pooled (green) - matches sibling figure
REGION = "#0072B2" # regional/programme spread (blue)
AUDIT = "#999999"  # mature national programme per-condition (grey)
FLOOR = "#D55E00"  # FASP minimum-acceptable floor (orange)

# --- the detection-rate landscape (antenatal CHD detection, fraction). MODEL-FREE:
#     the model's per-lesion bars live on the Ch5 sonographer figure; this Ch2 figure
#     establishes only the clinical landscape so the "where detection is lowest" niche
#     is planted before the model is introduced. All anchors are the report's own cites. ---
# rows, top (highest) -> bottom (lowest)
rows = [
    # label, low, high, point (np.nan if none), colour, cite
    ("Mature national programme,\nper-condition (UK 20-week audit)", 0.70, 0.93, np.nan, AUDIT),
    ("Reported range across\nregions and programmes",                 0.33, 0.82, np.nan, REGION),
    ("Unselected populations,\npooled",                               np.nan, np.nan, 0.451, VV),
    ("Low-resource selected\nseries",                                 0.15, 0.35, np.nan, LMIC),
]

FASP_FLOOR = 0.50

fig, ax = plt.subplots()
y = np.arange(len(rows))[::-1]  # first row at top

for yi, (label, lo, hi, pt, col) in zip(y, rows):
    if not np.isnan(lo):
        ax.plot([lo, hi], [yi, yi], color=col, lw=7, solid_capstyle="round", zorder=3)
        ax.plot([lo, hi], [yi, yi], color=col, lw=7, alpha=0.0)  # spacing safety
        ax.scatter([lo, hi], [yi, yi], color=col, s=18, zorder=4)
    if not np.isnan(pt):
        ax.scatter([pt], [yi], color=col, s=70, zorder=4, edgecolor="white", linewidth=0.8)

# 50% FASP floor reference line
ax.axvline(FASP_FLOOR, ls="--", lw=1.3, color=FLOOR, zorder=2)
ax.text(FASP_FLOOR + 0.008, len(rows) - 0.62, "FASP floor 0.50",
        color=FLOOR, fontsize=8.5, rotation=90, va="top", ha="left")

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("Antenatal detection of congenital heart disease (fraction)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

legend_handles = [
    Line2D([0], [0], color=AUDIT, lw=7, label="Reported range"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VV, markersize=8,
           markeredgecolor="white", label="Pooled point estimate"),
    Line2D([0], [0], color=FLOOR, lw=1.3, ls="--", label="Programme minimum standard"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig("detection_landscape.pdf", bbox_inches="tight")
fig.savefig("detection_landscape.png", dpi=200, bbox_inches="tight")
print("wrote detection_landscape.pdf / .png")
