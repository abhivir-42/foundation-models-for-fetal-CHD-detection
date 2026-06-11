#!/usr/bin/env python
"""ppv_prevalence figure (Ch5 §5): screening asymmetry.

PPV and NPV Bayes-transported across population prevalence for the auditable-four
operating point (spec=0.95) of the FetalCLIP F2a transformer. Class-conditional
sens/spec are prevalence-invariant, so PPV(pi)/NPV(pi) are exact transports of the
on-disk 5-fold-mean operating point.

Source of constants: the PPV/NPV table under RESULTS_DIR (ppv_npv_table.csv).
(F2a/FetalCLIP, auditable-4, spec=0.95): sens=0.5935, realized spec=0.9507.
Cross-checked vs report/from-cluster/VERIFIED-NUMBERS.md (PPV 0.583 cohort,
0.108 @1%, NPV 0.996).
"""

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

# ---- house style -----------------------------------------------------------
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
BLUE = "#0072B2"    # FetalCLIP / domain
ORANGE = "#D55E00"  # DINOv2 / general
GREEN = "#009E73"
GREY = "#999999"

# ---- operating point (auditable-4, spec=0.95) ------------------------------
SENS_F = 0.5935   # FetalCLIP F2a 5-fold mean sens@spec0.95
SPEC_F = 0.9507   # realized spec (drives transport)
SENS_E = 0.3546   # DINOv2 E2 (optional comparison)
SPEC_E = 0.9507


def ppv(sens, spec, pi):
    return sens * pi / (sens * pi + (1 - spec) * (1 - pi))


def npv(sens, spec, pi):
    return spec * (1 - pi) / (spec * (1 - pi) + (1 - sens) * pi)


# prevalence axis (log): ~0.1% to cohort 0.104
pi = np.logspace(np.log10(0.001), np.log10(0.104), 400)

ppv_f = ppv(SENS_F, SPEC_F, pi)
npv_f = npv(SENS_F, SPEC_F, pi)
ppv_e = ppv(SENS_E, SPEC_E, pi)

# ---- the three priors ------------------------------------------------------
COHORT = 0.104       # enriched cohort, PPV 0.583
POP1 = 0.01          # all-CHD deployment prior, PPV 0.108
AUD4 = 0.0015        # auditable-four true prior (subset of CHD), PPV ~0.02

fig, ax = plt.subplots()

# PPV curves
ax.plot(pi, ppv_f, color=BLUE, lw=2, zorder=4,
        label="PPV  FetalCLIP transformer")
ax.plot(pi, ppv_e, color=ORANGE, lw=2, ls="--", zorder=3,
        label="PPV  DINOv2 linear probe")
# NPV (FetalCLIP) on same axis
ax.plot(pi, npv_f, color=GREEN, lw=2, zorder=4,
        label="NPV  FetalCLIP transformer")

# ---- prior guides ----------------------------------------------------------
prior_specs = [
    (AUD4, "auditable-four prior\n~0.15%  (PPV 0.02)", ppv(SENS_F, SPEC_F, AUD4)),
    (POP1, "all-CHD deployment\n~1%  (PPV 0.11)", ppv(SENS_F, SPEC_F, POP1)),
    (COHORT, "enriched cohort\n10.4%  (PPV 0.58)\nnot deployment", ppv(SENS_F, SPEC_F, COHORT)),
]
for x, _lab, _y in prior_specs:
    ax.axvline(x, color=GREY, lw=0.9, ls=":", zorder=1)

# markers on the FetalCLIP PPV curve at each prior
for x, _lab, y in prior_specs:
    ax.plot([x], [y], "o", ms=5, color=BLUE, zorder=5,
            markeredgecolor="white", markeredgewidth=0.6)

# annotate the three priors (point text away from data)
ax.annotate("auditable-four\n~0.15% prior\nPPV 0.02", xy=(AUD4, ppv(SENS_F, SPEC_F, AUD4)),
            xytext=(AUD4 * 1.05, 0.30), fontsize=7.3, color="#444444", ha="left",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.annotate("all-CHD deployment\n~1% prior\nPPV 0.11", xy=(POP1, ppv(SENS_F, SPEC_F, POP1)),
            xytext=(POP1 * 0.9, 0.50), fontsize=7.3, color="#444444", ha="left",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.annotate("enriched cohort\n10.4%, PPV 0.58\nnot deployment",
            xy=(COHORT, ppv(SENS_F, SPEC_F, COHORT)),
            xytext=(0.022, 0.70), fontsize=7.3, color="#444444", ha="left",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))

# NPV stays high annotation
ax.annotate("NPV stays $\\geq$0.996 across the screening range",
            xy=(0.0035, npv(SENS_F, SPEC_F, 0.0035)), xytext=(0.0012, 0.90),
            fontsize=7.6, color=GREEN, ha="left")

ax.set_xscale("log")
ax.set_xlim(0.001, 0.13)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Disease prevalence in the screened population (log scale)")
ax.set_ylabel("Predictive value")

# clean log ticks
ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.104]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{100*t:g}%" for t in ticks])
ax.minorticks_off()

ax.legend(loc="center left", frameon=False, fontsize=8, handlelength=1.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    dpi = 150 if ext == "png" else None
    fig.savefig(
        fstr(_FIGOUT / "ppv_prevalence.{ext}"),
        dpi=dpi, bbox_inches="tight",
    )

# ---- echo verification numbers --------------------------------------------
print("VERIFY:")
print(f"  PPV @cohort 0.104 = {ppv(SENS_F, SPEC_F, COHORT):.4f}  (expect 0.583)")
print(f"  PPV @pop 0.01     = {ppv(SENS_F, SPEC_F, POP1):.4f}  (expect 0.108)")
print(f"  PPV @aud4 0.0015  = {ppv(SENS_F, SPEC_F, AUD4):.4f}  (expect ~0.02)")
print(f"  NPV @cohort 0.104 = {npv(SENS_F, SPEC_F, COHORT):.4f}")
print(f"  NPV @pop 0.01     = {npv(SENS_F, SPEC_F, POP1):.4f}  (expect 0.996)")
print(f"  NPV @aud4 0.0015  = {npv(SENS_F, SPEC_F, AUD4):.4f}")
