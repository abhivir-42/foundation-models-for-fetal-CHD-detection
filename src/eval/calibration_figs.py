#!/usr/bin/env python3
"""Reliability diagrams for Tier 1 calibration figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.per_condition import reliability_curve


def render_reliability(y_true: np.ndarray, y_prob: np.ndarray,
                       config_name: str, output_path: Path,
                       n_bins: int = 10) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_h, ax_r = axes

    healthy = y_prob[y_true == 0]
    unhealthy = y_prob[y_true == 1]
    bins = np.linspace(0.0, 1.0, 31)
    ax_h.hist([healthy, unhealthy], bins=bins, stacked=True,
              label=['healthy', 'unhealthy'],
              color=['#4c72b0', '#dd8452'], edgecolor='black', linewidth=0.3)
    ax_h.set_xlabel('sigmoid output')
    ax_h.set_ylabel('count')
    ax_h.set_title('score histogram')
    ax_h.legend(loc='best', fontsize=8)
    ax_h.set_xlim(0, 1)

    ax_r.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='ideal')
    confs = np.asarray(curve['mean_confs'])
    accs = np.asarray(curve['accs'])
    counts = np.asarray(curve['counts'])
    ax_r.plot(confs, accs, 'o-', color='C0', label='reliability')
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.set_xlabel('mean predicted')
    ax_r.set_ylabel('observed positive rate')
    ax_r.set_title('reliability (equal-mass)')
    ax_r.legend(loc='best', fontsize=8)

    if len(counts) > 0:
        ax_b = ax_r.twinx()
        width = 1.0 / max(n_bins, 1) * 0.7
        ax_b.bar(confs, counts / counts.sum(), width=width,
                 alpha=0.25, color='grey')
        ax_b.set_ylabel('bin mass', color='grey', fontsize=8)
        ax_b.tick_params(axis='y', labelcolor='grey', labelsize=7)
        ax_b.set_ylim(0, max(0.5, (counts / counts.sum()).max() * 3))

    fig.suptitle(config_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
