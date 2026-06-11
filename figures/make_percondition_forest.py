
import os as _os_fig
from pathlib import Path as _Path_fig
REPO_ROOT = _Path_fig(__file__).resolve().parent.parent
_RESULTS = _Path_fig(_os_fig.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results')))
_PREDS   = _Path_fig(_os_fig.environ.get('PREDICTIONS_DIR', str(REPO_ROOT / 'predictions')))
_IFIND   = _Path_fig(_os_fig.environ.get('IFIND_DATA', str(REPO_ROOT / 'data')))
_FIGOUT  = _Path_fig(_os_fig.environ.get('FIGURES_OUT', str(REPO_ROOT / 'figures' / 'out')))
_FIGOUT.mkdir(parents=True, exist_ok=True)

import pandas as pd
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
    "figure.figsize": (5.4, 3.6),
})

DOMAIN = "#0072B2"   # auditable-four (FetalCLIP domain colour)
GREY = "#999999"     # other conditions / reference

CSV = str(_RESULTS / "multilabel_5fold/binary_per_fold.csv")
df = pd.read_csv(CSV)
df = df[df.arm == "binary"]

g = df.groupby("condition").agg(mean=("auroc", "mean"), sd=("auroc", "std"),
                                npos_sum=("n_pos", "sum")).reset_index()
stats = {r.condition: (r.mean, r.sd) for r in g.itertuples()}

# Canonical headline CoA uses the cleaned subset (n=59); per-fold SD from the
# uncleaned per-fold CSV is used for the error bar (best available estimate).
COA_CLEAN_MEAN = 0.879

# Display names, auditable-four membership, exploratory flag (n_pos < ~30)
META = {
    "hlhs":       ("Hypoplastic left heart syndrome (HLHS)", True,  False),
    "a_stenosis": ("Aortic stenosis",                        False, True),
    "avsd":       ("Atrioventricular septal defect (AVSD)",  True,  False),
    "p_atresia":  ("Pulmonary atresia",                      False, False),
    "tetralogy":  ("Tetralogy of Fallot (TOF)",              True,  False),
    "tga":        ("Transposition of great arteries (TGA)",  True,  False),
    "p_stenosis": ("Pulmonary stenosis",                     False, True),
    "coa":        ("Coarctation of the aorta (CoA)",         False, False),
    "raa":        ("Right aortic arch (RAA)",                False, False),
}

rows = []
for cond, (name, aud4, expl) in META.items():
    m, sd = stats[cond]
    if cond == "coa":
        m = COA_CLEAN_MEAN  # cleaned-subset headline
    label = name + ("*" if expl else "")
    rows.append((cond, label, m, sd, aud4))

# sort by AUROC descending; plot so the largest sits at the top
rows.sort(key=lambda r: r[2], reverse=True)
labels = [r[1] for r in rows]
means = np.array([r[2] for r in rows])
sds = np.array([r[3] for r in rows])
colors = [DOMAIN if r[4] else GREY for r in rows]

y = np.arange(len(rows))[::-1]  # top = first (highest AUROC)

fig, ax = plt.subplots()

for yi, m, sd, c in zip(y, means, sds, colors):
    ax.errorbar(m, yi, xerr=sd, fmt="o", color=c, ecolor=c,
                elinewidth=1.5, capsize=3, markersize=5, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("5-fold AUROC (condition vs healthy)")
ax.set_xlim(0.76, 1.0)

ax.grid(axis="y", visible=False)

legend_handles = [
    Line2D([0], [0], marker="o", color=DOMAIN, lw=0, markersize=6,
           label="Auditable-four"),
    Line2D([0], [0], marker="o", color=GREY, lw=0, markersize=6,
           label="Other condition"),
    Line2D([0], [0], marker="", color="none", label=r"$\ast$ exploratory ($n_{pos}\!\leq\!28$)"),
]
# horizontal legend above the axes so it never overlaps any data row/marker
ax.legend(handles=legend_handles, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8, handletextpad=0.4,
          columnspacing=1.4)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
# bbox_inches="tight" so the above-axes legend is not clipped at the figure edge
fig.savefig(str(_FIGOUT / "percondition_forest.pdf"),
            bbox_inches="tight")
fig.savefig(str(_FIGOUT / "percondition_forest.png"),
            dpi=150, bbox_inches="tight")

print("means(sorted):")
for r in rows:
    print(f"  {r[1]:<40s} {r[2]:.3f} +/- {r[3]:.3f}  aud4={r[4]}")
