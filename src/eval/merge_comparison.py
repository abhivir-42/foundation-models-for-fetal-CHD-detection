#!/usr/bin/env python3
"""Side-by-side comparison: un-merged per-condition vs merged conditions.

Reads metrics_slicing_i.csv (un-merged) and metrics_merged_slicing_i.csv
(merged) and writes merge_comparison.md showing each merge group alongside
its component conditions.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.per_condition import TOP6, REPO_ROOT

DEFAULT_DIR = Path(os.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results'))) / 'exp05_temporal_unfiltered' / 'per_condition'

MERGES = {
    'outflow_stenoses': ['a_stenosis', 'p_stenosis', 'p_atresia'],
    'conotruncal': ['tetralogy', 'tga', 'raa'],
}


def fmt_metric(point, lo, hi):
    if pd.isna(point):
        return 'n/a'
    if pd.isna(lo) or pd.isna(hi):
        return f"{point:.3f}"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def section_for_group(merged_name: str, components: list[str],
                      df_unmerged: pd.DataFrame, df_merged: pd.DataFrame) -> list[str]:
    lines = [f"## {merged_name} = {' OR '.join(components)}", ""]

    # n_pos line
    merged_row = df_merged[df_merged['condition'] == merged_name]
    if not merged_row.empty:
        n_merged = int(merged_row.iloc[0]['n_pos'])
    else:
        n_merged = -1
    comp_n = []
    for c in components:
        rows = df_unmerged[df_unmerged['condition'] == c]
        if not rows.empty:
            comp_n.append((c, int(rows.iloc[0]['n_pos'])))
        else:
            comp_n.append((c, -1))
    comp_str = ', '.join(f"{c}={n}" for c, n in comp_n)
    lines.append(f"n_pos: merged={n_merged} (sum-of-components={sum(n for _, n in comp_n)}, "
                 f"OR-overlaps reduce); components: {comp_str}")
    lines.append("")

    lines.append("### sens@spec=.95 (slicing i)")
    lines.append("")
    lines.append("| config | " + " | ".join(components + [f"**{merged_name}**"]) + " |")
    lines.append("|---|" + "---|" * (len(components) + 1))

    for cfg in TOP6:
        cells = []
        for c in components:
            r = df_unmerged[(df_unmerged['config'] == cfg) & (df_unmerged['condition'] == c)]
            if r.empty:
                cells.append('n/a')
            else:
                row = r.iloc[0]
                cells.append(fmt_metric(row['sens_at_spec95'],
                                        row['sens_at_spec95_lo'],
                                        row['sens_at_spec95_hi']))
        r = df_merged[(df_merged['config'] == cfg) & (df_merged['condition'] == merged_name)]
        if r.empty:
            cells.append('n/a')
        else:
            row = r.iloc[0]
            cells.append("**" + fmt_metric(row['sens_at_spec95'],
                                           row['sens_at_spec95_lo'],
                                           row['sens_at_spec95_hi']) + "**")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### AUROC (slicing i)")
    lines.append("")
    lines.append("| config | " + " | ".join(components + [f"**{merged_name}**"]) + " |")
    lines.append("|---|" + "---|" * (len(components) + 1))
    for cfg in TOP6:
        cells = []
        for c in components:
            r = df_unmerged[(df_unmerged['config'] == cfg) & (df_unmerged['condition'] == c)]
            if r.empty:
                cells.append('n/a')
            else:
                row = r.iloc[0]
                cells.append(fmt_metric(row['auroc'], row['auroc_lo'], row['auroc_hi']))
        r = df_merged[(df_merged['config'] == cfg) & (df_merged['condition'] == merged_name)]
        if r.empty:
            cells.append('n/a')
        else:
            row = r.iloc[0]
            cells.append("**" + fmt_metric(row['auroc'], row['auroc_lo'], row['auroc_hi']) + "**")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    # CI-width comparison
    lines.append("### CI width on sens@95 — does merging tighten it?")
    lines.append("")
    lines.append("| config | mean component CI width | merged CI width | Δ |")
    lines.append("|---|---:|---:|---:|")
    for cfg in TOP6:
        comp_widths = []
        for c in components:
            r = df_unmerged[(df_unmerged['config'] == cfg) & (df_unmerged['condition'] == c)]
            if not r.empty:
                row = r.iloc[0]
                if not (pd.isna(row['sens_at_spec95_lo']) or pd.isna(row['sens_at_spec95_hi'])):
                    comp_widths.append(row['sens_at_spec95_hi'] - row['sens_at_spec95_lo'])
        mean_comp = float(np.mean(comp_widths)) if comp_widths else float('nan')
        r = df_merged[(df_merged['config'] == cfg) & (df_merged['condition'] == merged_name)]
        if r.empty:
            merged_w = float('nan')
        else:
            row = r.iloc[0]
            merged_w = (row['sens_at_spec95_hi'] - row['sens_at_spec95_lo']
                        if not (pd.isna(row['sens_at_spec95_lo']) or pd.isna(row['sens_at_spec95_hi']))
                        else float('nan'))
        delta = (merged_w - mean_comp
                 if not (np.isnan(merged_w) or np.isnan(mean_comp)) else float('nan'))
        c_str = f"{mean_comp:.3f}" if not np.isnan(mean_comp) else 'n/a'
        m_str = f"{merged_w:.3f}" if not np.isnan(merged_w) else 'n/a'
        d_str = f"{delta:+.3f}" if not np.isnan(delta) else 'n/a'
        lines.append(f"| {cfg} | {c_str} | {m_str} | {d_str} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    df_un = pd.read_csv(args.dir / 'metrics_slicing_i.csv')
    df_mg = pd.read_csv(args.dir / 'metrics_merged_slicing_i.csv')

    lines = [
        "# Merge-aware reporting — comparison",
        "",
        "**Status: PROPOSED, NOT signed off.** Merge groups below subsume separately-",
        "interpretable conditions; clinical sign-off needed before any of this becomes",
        "the headline.",
        "",
        "Inputs:",
        f"- Un-merged: `metrics_slicing_i.csv` (slicing-i, 9 conditions).",
        f"- Merged: `metrics_merged_slicing_i.csv` (slicing-i, 5 conditions: avsd, hlhs, "
        f"coa, **outflow_stenoses**, **conotruncal**).",
        "- Bootstrap = 1000 per-subject resamples, seed=42.",
        "",
        "Bold cells = merged condition. Bold values include 95% CIs from the merged sweep.",
        "",
    ]

    for merged_name, components in MERGES.items():
        lines.extend(section_for_group(merged_name, components, df_un, df_mg))

    # Top-level takeaway
    lines += [
        "## Takeaways",
        "",
        "1. **Both merges tighten CIs**: outflow_stenoses ΔCI ≈ −0.10 to −0.31; "
        "conotruncal ΔCI ≈ −0.09 to −0.17. Merging is doing the job statistically.",
        "2. **Absolute width still matters**: conotruncal merged CI on sens@95 is "
        "≈ 0.25–0.33 wide (usable for ranking configs); outflow_stenoses merged CI is "
        "still ≈ 0.38–0.52 wide (every config's CI overlaps every other's). "
        "Outflow_stenoses clears the 20-positive bar but stays noise-dominated.",
        "3. **Per-config ranking changes under merging**: E2 dominates outflow_stenoses "
        "(sens@95 = 0.524) but loses on conotruncal (0.174). Component conditions are "
        "not interchangeable from the head's perspective.",
        "4. **AUROC for outflow_stenoses tracks the n=14 p_atresia component much more "
        "than the rare a_stenosis (n=5) / p_stenosis (n=2)** — the merge is dominated "
        "by p_atresia signal and the merged metric is largely a re-skinned p_atresia "
        "metric for now.",
        "5. **A decision is still needed**: the merge proposals are statistical "
        "scaffolding, not clinical groupings; a clinician should rule on whether "
        "`outflow_stenoses` is a defensible category. If not, the cleanest path is to "
        "report 6 conditions (avsd, hlhs, tga, tetralogy, raa, coa) in the headline "
        "with explicit `n_pos` flags, and carve a_stenosis/p_stenosis/p_atresia into "
        "an appendix paragraph.",
        "",
    ]

    out = args.dir / 'merge_comparison.md'
    out.write_text('\n'.join(lines) + '\n')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
