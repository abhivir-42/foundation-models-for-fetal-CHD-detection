#!/usr/bin/env python3
"""Forest plot of sens@spec=0.95 per condition across the six Tier 1 configs."""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = Path(os.environ.get("RESULTS_DIR", str(REPO_ROOT / "results")))
DAY2 = _RESULTS / "exp05_temporal_unfiltered" / "per_condition"
OUT_DIR = _RESULTS / "forest_plot"

LEAD_CONFIG = "E2_mean_pool_baseline"
CONFIG_ORDER = [
    LEAD_CONFIG, "B2_cross_attn", "F1b_learnable_pos",
    "C1_temporal_cnn", "F2b_hidden128", "F2a_hidden64",
]
OKABE_ITO = {
    "E2_mean_pool_baseline": "#D55E00",
    "B2_cross_attn":         "#0072B2",
    "F1b_learnable_pos":     "#009E73",
    "C1_temporal_cnn":       "#CC79A7",
    "F2b_hidden128":         "#56B4E9",
    "F2a_hidden64":          "#E69F00",
}
DISAGREE = {
    "tga": 94.53, "avsd": 100.0, "tetralogy": 96.52, "coa": 53.47,
    "p_atresia": 91.89, "raa": 93.58, "hlhs": 98.80,
    "a_stenosis": 100.0, "p_stenosis": 100.0,
}
N_POS_FULL = {
    "tga": 128, "avsd": 86, "tetralogy": 115, "coa": 101, "p_atresia": 74,
    "raa": 109, "hlhs": 83, "a_stenosis": 27, "p_stenosis": 28,
}


def _caption(split: str) -> str:
    return (
        f"sens@spec=0.95, 1000-resample per-subject bootstrap 95% CI ({split}, "
        "slicing i: healthy negatives only). E2 lead config in vermilion squares. "
        "8/9 conditions n_pos<20 -> CI widths 0.30-0.50; selection rule yields no winner."
    )


def render_forest(metrics_csv: Path, output_png: Path, *, split_label: str = "test",
                  configs: list[str] | None = None,
                  lead_config: str = LEAD_CONFIG,
                  caption: str | None = None) -> None:
    df = pd.read_csv(metrics_csv)
    configs = configs if configs is not None else CONFIG_ORDER
    row_offsets = np.linspace(-0.30, 0.30, len(configs))
    cond_order = (
        df[["condition", "n_pos"]].drop_duplicates()
          .sort_values("n_pos", ascending=False)["condition"].tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    for ci, cond in enumerate(cond_order):
        if ci % 2 == 0:
            ax.axhspan(ci - 0.5, ci + 0.5, color="black", alpha=0.04, lw=0)
        for cfg, off in zip(configs, row_offsets):
            sub = df[(df.condition == cond) & (df.config == cfg)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            x, lo, hi = row.sens_at_spec95, row.sens_at_spec95_lo, row.sens_at_spec95_hi
            is_lead = cfg == lead_config
            ax.errorbar(
                x, ci + off, xerr=[[max(x - lo, 0)], [max(hi - x, 0)]],
                fmt="s" if is_lead else "o",
                ms=6.5 if is_lead else 5,
                color=OKABE_ITO[cfg],
                elinewidth=2.0 if is_lead else 1.2,
                capsize=3, label=cfg if ci == 0 else None,
            )
    yticklabels = [
        f"{c:>11s}  (n={int(df[df.condition == c].n_pos.iloc[0]):>2d})"
        for c in cond_order
    ]
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels(yticklabels, family="monospace")
    ax.set_ylim(-0.6, len(cond_order) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.set_xlabel(f"Sensitivity at spec=0.95 ({split_label})")
    ax.grid(axis="x", which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(0.02, -0.02, caption if caption is not None else _caption(split_label),
             fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_disagreement_panel(rates_dict: dict[str, float], output_png: Path) -> None:
    cond_order = sorted(rates_dict, key=lambda c: -N_POS_FULL.get(c, 0))
    rates = np.array([rates_dict[c] for c in cond_order])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(cond_order)), rates, color=plt.cm.viridis(rates / 100))
    for i, c in enumerate(cond_order):
        ax.text(rates[i] + 1, i, f"{rates[i]:.0f}%  (n={N_POS_FULL.get(c, '?')})",
                va="center", fontsize=8)
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels(cond_order, family="monospace")
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Token-4 vs 9-col disagreement rate (%)")
    fig.text(
        0.02, -0.04,
        "Full cohort (n=4128). coa is the only condition with substantial agreement; "
        "three conditions have ZERO agreement. Forest-plot ranking is conditional on "
        "the 9-col label set.",
        fontsize=8, wrap=True,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_forest(DAY2 / "metrics_slicing_i.csv", OUT_DIR / "forest_test.png")
    render_disagreement_panel(DISAGREE, OUT_DIR / "disagreement_panel.png")


if __name__ == "__main__":
    main()
