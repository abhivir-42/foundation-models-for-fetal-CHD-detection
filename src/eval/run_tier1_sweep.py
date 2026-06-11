#!/usr/bin/env python3
"""Tier 1 metric sweep.

Per-condition x top-6 x slicing-(i)+(ii) with 1000-resample per-subject block
bootstrap CIs (seed=42), aggregate sensitivity-checks (any-disease AUROC,
prevalence-weighted sens@spec=.95), and 4-clause selection-rule scoring with
per-clause output.

Outputs (under --out, default = RESULTS_DIR/exp05_temporal_unfiltered/per_condition/):
  per_condition_counts.csv
  metrics_slicing_i.csv     metrics_slicing_ii.csv
  aggregate.csv
  selection_rule.md         selection_rule.json
  summary.md

Usage:
  python -m src.eval.run_tier1_sweep \
      --results $RESULTS_DIR/exp05_temporal/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.per_condition import (
    CONDITIONS, TOP6, SPEC_TARGETS, MIN_POSITIVES,
    DEFAULT_RESULTS, DEFAULT_TEST_SPLIT, DEFAULT_N_BOOT, DEFAULT_SEED,
    REPO_ROOT,
    _coerce_id,
    load_results, load_test_labels, per_condition_counts,
    slicing_i, slicing_ii,
    auroc, aupr, sensitivity_at_specificity, ppv_npv_at_threshold,
    expected_calibration_error, adaptive_ece, brier_murphy,
    bootstrap_ci,
)

HEADLINE_SPEC = 0.95
DEFAULT_OUT = Path(os.environ.get('RESULTS_DIR', str(REPO_ROOT / 'results'))) / 'exp05_temporal_unfiltered' / 'per_condition'


def load_n_params(results_path: Path) -> dict[str, int | None]:
    """n_params lives in the sibling comparison_table.csv (results.json's config
    block dropped it). 'N/A' for the LogReg baseline becomes 0 (effectively
    fewest params for tiebreaker)."""
    table = results_path.parent / 'comparison_table.csv'
    if not table.exists():
        return {}
    df = pd.read_csv(table)
    out: dict[str, int | None] = {}
    for _, row in df.iterrows():
        v = row.get('n_params')
        if pd.isna(v) or v == 'N/A':
            out[row['experiment']] = 0
        else:
            try:
                out[row['experiment']] = int(v)
            except (TypeError, ValueError):
                out[row['experiment']] = None
    return out


def _sens_at(spec: float):
    def fn(y, p):
        s, _ = sensitivity_at_specificity(y, p, spec)
        return s
    return fn


def _row_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 n_boot: int, seed: int) -> dict:
    nan = float('nan')
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos < 1 or n_neg < 1:
        return {
            'n_pos': n_pos, 'n_neg': n_neg,
            'auroc': nan, 'auroc_lo': nan, 'auroc_hi': nan,
            'aupr': nan, 'aupr_lo': nan, 'aupr_hi': nan,
            'sens_at_spec90': nan, 'thr_at_spec90': nan,
            'sens_at_spec95': nan, 'sens_at_spec95_lo': nan, 'sens_at_spec95_hi': nan,
            'thr_at_spec95': nan,
            'sens_at_spec99': nan, 'thr_at_spec99': nan,
            'ppv_at_spec95': nan, 'npv_at_spec95': nan,
            'ece': nan,
            'brier': nan, 'brier_rel': nan, 'brier_res': nan,
            'brier_unc': nan, 'brier_residual': nan,
        }
    a_pt, a_lo, a_hi = bootstrap_ci(y_true, y_prob, auroc, n_boot=n_boot, seed=seed)
    p_pt, p_lo, p_hi = bootstrap_ci(y_true, y_prob, aupr, n_boot=n_boot, seed=seed)
    s95_pt, s95_lo, s95_hi = bootstrap_ci(y_true, y_prob, _sens_at(0.95),
                                          n_boot=n_boot, seed=seed)
    sens90, thr90 = sensitivity_at_specificity(y_true, y_prob, 0.90)
    _, thr95 = sensitivity_at_specificity(y_true, y_prob, 0.95)
    sens99, thr99 = sensitivity_at_specificity(y_true, y_prob, 0.99)
    if not np.isnan(thr95):
        ppv95, npv95 = ppv_npv_at_threshold(y_true, y_prob, thr95)
    else:
        ppv95, npv95 = nan, nan
    # `ece` column reports adaptive (equal-mass) ECE; the column name stays
    # `ece` so downstream consumers keep working. Note equal-width vs
    # adaptive ECE values are NOT directly comparable.
    ece = adaptive_ece(y_true, y_prob, n_bins=10)
    bm = brier_murphy(y_true, y_prob, n_bins=10)
    return {
        'n_pos': n_pos, 'n_neg': n_neg,
        'auroc': a_pt, 'auroc_lo': a_lo, 'auroc_hi': a_hi,
        'aupr': p_pt, 'aupr_lo': p_lo, 'aupr_hi': p_hi,
        'sens_at_spec90': sens90, 'thr_at_spec90': thr90,
        'sens_at_spec95': s95_pt, 'sens_at_spec95_lo': s95_lo, 'sens_at_spec95_hi': s95_hi,
        'thr_at_spec95': thr95,
        'sens_at_spec99': sens99, 'thr_at_spec99': thr99,
        'ppv_at_spec95': ppv95, 'npv_at_spec95': npv95,
        'ece': ece,
        'brier': bm['brier'], 'brier_rel': bm['rel'],
        'brier_res': bm['res'], 'brier_unc': bm['unc'],
        'brier_residual': bm['residual'],
    }


def per_condition_sweep(preds_dict, labels, configs, conditions, n_boot, seed):
    rows_i, rows_ii = [], []
    for cfg in configs:
        if cfg not in preds_dict:
            print(f"  WARN: {cfg} missing from results.json")
            continue
        p = preds_dict[cfg]
        for cond in conditions:
            yi, pi, _ = slicing_i(p, labels, cond)
            r = _row_metrics(yi, pi, n_boot, seed)
            r.update({'config': cfg, 'condition': cond, 'slicing': 'i'})
            rows_i.append(r)
            yii, pii, _ = slicing_ii(p, labels, cond)
            r2 = _row_metrics(yii, pii, n_boot, seed)
            r2.update({'config': cfg, 'condition': cond, 'slicing': 'ii'})
            rows_ii.append(r2)
            print(f"    {cfg:22s} / {cond:11s}  (i) AUROC={r['auroc']:.3f} "
                  f"sens@95={r['sens_at_spec95']:.3f} CI[{r['sens_at_spec95_lo']:.3f},"
                  f"{r['sens_at_spec95_hi']:.3f}]  ECE={r['ece']:.3f}")
    cols_first = ['config', 'condition', 'slicing', 'n_pos', 'n_neg']
    df_i = pd.DataFrame(rows_i)
    df_ii = pd.DataFrame(rows_ii)
    df_i = df_i[cols_first + [c for c in df_i.columns if c not in cols_first]]
    df_ii = df_ii[cols_first + [c for c in df_ii.columns if c not in cols_first]]
    return df_i, df_ii


def aggregate_sweep(preds_dict, labels, configs, conditions, n_boot, seed):
    rows = []
    for cfg in configs:
        if cfg not in preds_dict:
            continue
        p = preds_dict[cfg]
        keep, yt = [], []
        for i, raw_sid in enumerate(p.subject_ids):
            sid = _coerce_id(raw_sid)
            if sid in labels.index:
                keep.append(i)
                yt.append(int(labels.loc[sid, 'unhealthy']))
        keep = np.asarray(keep, dtype=int)
        yt = np.asarray(yt, dtype=int)
        pp = p.y_prob[keep]
        a_pt, a_lo, a_hi = bootstrap_ci(yt, pp, auroc, n_boot=n_boot, seed=seed)
        sens95_full, _ = sensitivity_at_specificity(yt, pp, HEADLINE_SPEC)
        # Prevalence-weighted sens@95 across conditions (slicing i)
        per_sens, per_n = [], []
        for cond in conditions:
            yi, pi, _ = slicing_i(p, labels, cond)
            n_pos = int(yi.sum())
            s, _ = sensitivity_at_specificity(yi, pi, HEADLINE_SPEC)
            per_sens.append(s)
            per_n.append(n_pos)
        per_sens = np.asarray(per_sens, dtype=float)
        per_n = np.asarray(per_n, dtype=float)
        valid = ~np.isnan(per_sens) & (per_n > 0)
        pw_sens = (float((per_sens[valid] * per_n[valid]).sum() / per_n[valid].sum())
                   if valid.any() else float('nan'))
        rows.append({
            'config': cfg,
            'any_disease_auroc': a_pt,
            'any_disease_auroc_lo': a_lo,
            'any_disease_auroc_hi': a_hi,
            'any_disease_sens_at_spec95': sens95_full,
            'prevalence_weighted_sens_at_spec95': pw_sens,
            'n_params': p.n_params,
        })
    return pd.DataFrame(rows)


def coord_bootstrap_mean_sens95(preds_dict, labels, configs, conditions,
                                n_boot, seed):
    """Coordinated subject-level bootstrap of mean-across-conditions sens@95
    (slicing i). All configs share the resample sequence."""
    sid_arr = np.asarray(labels.index.tolist())
    n_test = len(sid_arr)
    cond_mask = {c: (labels[c] == 1).values for c in conditions}
    healthy_mask = (labels['unhealthy'] == 0).values

    pred_idx = {}
    for cfg in configs:
        if cfg not in preds_dict:
            continue
        p = preds_dict[cfg]
        m = {_coerce_id(s): i for i, s in enumerate(p.subject_ids)}
        pred_idx[cfg] = (p, m)

    # Point estimate per config
    point = {}
    for cfg, (p, m) in pred_idx.items():
        per = []
        for cond in conditions:
            yi, pi, _ = slicing_i(p, labels, cond)
            s, _ = sensitivity_at_specificity(yi, pi, HEADLINE_SPEC)
            per.append(s)
        valid = [v for v in per if not np.isnan(v)]
        point[cfg] = float(np.mean(valid)) if valid else float('nan')

    rng = np.random.default_rng(seed)
    boot = {cfg: [] for cfg in pred_idx}
    for _ in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        resampled_sids = sid_arr[idx]
        # Per-resample, per-condition computed positives + healthy negatives
        # (resampled with replacement from the full test split).
        for cfg, (p, m) in pred_idx.items():
            per = []
            for cond in conditions:
                pos_pos = idx[cond_mask[cond][idx]]
                neg_pos = idx[healthy_mask[idx]]
                pos_pred = [m[s] for s in sid_arr[pos_pos] if s in m]
                neg_pred = [m[s] for s in sid_arr[neg_pos] if s in m]
                if not pos_pred or not neg_pred:
                    continue
                yt = np.concatenate([np.ones(len(pos_pred), dtype=int),
                                     np.zeros(len(neg_pred), dtype=int)])
                yp = np.concatenate([p.y_prob[pos_pred], p.y_prob[neg_pred]])
                s, _ = sensitivity_at_specificity(yt, yp, HEADLINE_SPEC)
                if not np.isnan(s):
                    per.append(s)
            if per:
                boot[cfg].append(float(np.mean(per)))

    out = {}
    for cfg in pred_idx:
        if boot[cfg]:
            lo = float(np.quantile(boot[cfg], 0.025))
            hi = float(np.quantile(boot[cfg], 0.975))
        else:
            lo = hi = float('nan')
        out[cfg] = {'point': point[cfg], 'lo': lo, 'hi': hi}
    return out


def four_clause_selection(df_i, mean_ci, configs, n_params_map, conditions):
    sens95 = {cfg: {} for cfg in configs}
    ece = {cfg: {} for cfg in configs}
    brier = {cfg: {} for cfg in configs}
    rel = {cfg: {} for cfg in configs}
    res = {cfg: {} for cfg in configs}
    for _, row in df_i.iterrows():
        sens95[row['config']][row['condition']] = row['sens_at_spec95']
        ece[row['config']][row['condition']] = row['ece']
        brier[row['config']][row['condition']] = row.get('brier', float('nan'))
        rel[row['config']][row['condition']] = row.get('brier_rel', float('nan'))
        res[row['config']][row['condition']] = row.get('brier_res', float('nan'))
    mean_ece = {cfg: float(np.nanmean(list(ece[cfg].values())))
                for cfg in configs if ece[cfg]}
    mean_brier = {cfg: float(np.nanmean(list(brier[cfg].values())))
                  for cfg in configs if brier[cfg]}
    mean_rel = {cfg: float(np.nanmean(list(rel[cfg].values())))
                for cfg in configs if rel[cfg]}
    mean_res = {cfg: float(np.nanmean(list(res[cfg].values())))
                for cfg in configs if res[cfg]}

    sortable = [c for c in configs if c in mean_ci and not np.isnan(mean_ci[c]['point'])]
    sortable.sort(key=lambda c: mean_ci[c]['point'], reverse=True)
    if not sortable:
        return {'mean_ci_table': [], 'clause2_table': [], 'clause3_table': [],
                'tiebreaker_table': [], 'survivors_final': [], 'winner': None,
                'best_by_mean': None}
    best = sortable[0]
    best_lo = mean_ci[best]['lo']

    table1, surv1 = [], []
    for cfg in configs:
        ci = mean_ci.get(cfg, {'point': float('nan'), 'lo': float('nan'),
                               'hi': float('nan')})
        passes = (not np.isnan(ci['hi'])) and ci['hi'] >= best_lo
        table1.append({'config': cfg, **ci, 'pass': bool(passes)})
        if passes:
            surv1.append(cfg)

    per_cond_max = {}
    for cond in conditions:
        vs = [sens95[c].get(cond, float('nan')) for c in configs]
        vs = [v for v in vs if not np.isnan(v)]
        per_cond_max[cond] = max(vs) if vs else float('nan')

    table2, surv2 = [], []
    for cfg in surv1:
        worst, worst_cond = 0.0, None
        for cond in conditions:
            v = sens95[cfg].get(cond, float('nan'))
            if np.isnan(v) or np.isnan(per_cond_max[cond]):
                continue
            gap = per_cond_max[cond] - v
            if gap > worst:
                worst, worst_cond = gap, cond
        passes = worst <= 0.05
        table2.append({'config': cfg, 'worst_gap': worst,
                       'worst_cond': worst_cond, 'pass': bool(passes)})
        if passes:
            surv2.append(cfg)

    if surv2:
        best_ece = min(mean_ece.get(c, float('inf')) for c in surv2)
    else:
        finite = [mean_ece[c] for c in configs if c in mean_ece]
        best_ece = min(finite) if finite else float('nan')
    table3, surv3 = [], []
    cfgs_for_clause3 = surv2 if surv2 else [c for c in configs if c in mean_ece]
    for cfg in cfgs_for_clause3:
        e = mean_ece.get(cfg, float('inf'))
        passes = (e - best_ece) <= 0.05
        table3.append({'config': cfg, 'mean_ece': e,
                       'mean_brier': mean_brier.get(cfg, float('nan')),
                       'mean_rel': mean_rel.get(cfg, float('nan')),
                       'mean_res': mean_res.get(cfg, float('nan')),
                       'pass': bool(passes)})
        if surv2 and passes:
            surv3.append(cfg)

    tiebreak = sorted([(c, n_params_map.get(c)) for c in surv3],
                      key=lambda x: (x[1] if x[1] is not None else 1e18))
    winner = tiebreak[0][0] if tiebreak else None

    return {
        'best_by_mean': best,
        'mean_ci_table': table1,
        'clause2_table': table2,
        'clause3_table': table3,
        'tiebreaker_table': [{'config': c, 'n_params': n} for c, n in tiebreak],
        'survivors_final': surv3,
        'winner': winner,
    }


def write_selection_md(rule, out_path: Path):
    lines = [
        "# Tier 1 selection rule — 4-clause output",
        "",
        "Bootstrap = 1000 per-subject resamples, seed=42, slicing (i) headline.",
        "Generated by `src/eval/run_tier1_sweep.py`.",
        "",
        "## Clause 1 — Best-or-tied on mean sens@spec=.95 across 9 conditions (CI overlap with best)",
        "",
        "| config | mean sens@95 | 95% CI | pass |",
        "|---|---:|---|:---:|",
    ]
    for r in rule['mean_ci_table']:
        ci = (f"[{r['lo']:.3f}, {r['hi']:.3f}]"
              if not np.isnan(r['lo']) else "n/a")
        pt = f"{r['point']:.3f}" if not np.isnan(r['point']) else "n/a"
        lines.append(f"| {r['config']} | {pt} | {ci} | {'✓' if r['pass'] else '✗'} |")
    lines += ["",
              f"Best by point estimate: **{rule.get('best_by_mean') or 'n/a'}**",
              ""]

    lines += [
        "## Clause 2 — No >5pp loss to per-condition best on any individual condition",
        "",
        "| config | worst per-cond gap | worst condition | pass |",
        "|---|---:|---|:---:|",
    ]
    for r in rule['clause2_table']:
        lines.append(f"| {r['config']} | {r['worst_gap']:.3f} | "
                     f"{r['worst_cond'] or 'n/a'} | {'✓' if r['pass'] else '✗'} |")
    lines.append("")

    lines += [
        "## Clause 3 — Mean adaptive-ECE-10 ≤ best + 0.05  (Brier readout)",
        "",
        "| config | mean adaptive ECE | mean Brier | REL | RES | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    def _f(v): return f"{v:.3f}" if not np.isnan(v) else "n/a"
    for r in rule['clause3_table']:
        mb = r.get('mean_brier', float('nan'))
        mr = r.get('mean_rel', float('nan'))
        ms = r.get('mean_res', float('nan'))
        lines.append(f"| {r['config']} | {_f(r['mean_ece'])} | "
                     f"{_f(mb)} | {_f(mr)} | {_f(ms)} | "
                     f"{'✓' if r['pass'] else '✗'} |")
    lines.append("")

    lines += ["## Clause 4 — Tiebreaker: fewer params", ""]
    if rule.get('tiebreaker_table'):
        lines += ["| config | n_params |", "|---|---:|"]
        for r in rule['tiebreaker_table']:
            np_ = r['n_params']
            lines.append(f"| {r['config']} | "
                         f"{int(np_) if np_ is not None else 'unknown'} |")
        lines.append("")
        lines.append(f"## Winner: **{rule.get('winner') or 'n/a'}**")
    else:
        lines.append("No survivors after clauses 1–3; selection rule yields no winner.")
        lines.append("Fallback: highest mean sens@95 — see Clause 1 table.")
    out_path.write_text('\n'.join(lines) + '\n')


def write_summary_md(df_i, df_ii, agg_df, rule, mean_ci, counts_df,
                     n_boot, seed, results_path: Path, out_path: Path):
    lines = [
        "# Tier 1 metric sweep — summary",
        "",
        f"Bootstrap = {n_boot} per-subject resamples, seed={seed}.",
        f"Predictions: `{results_path}`.",
        f"Top-6 configs: {', '.join(TOP6)}.",
        "",
        "## Per-condition counts",
        "",
        "| condition | n_pos | n_healthy_neg | flag |",
        "|---|---:|---:|:---:|",
    ]
    for _, r in counts_df.iterrows():
        flag = '⚠ <20' if r['flag_low_n'] else ''
        lines.append(f"| {r['condition']} | {int(r['n_positive'])} | "
                     f"{int(r['n_healthy_neg'])} | {flag} |")
    lines.append("")

    lines += [
        "## Slicing-(i) sens@spec=.95 — pivot",
        "",
        "Rows = condition, cols = config. Values = point estimate.",
        "",
    ]
    pt = df_i.pivot_table(index='condition', columns='config',
                          values='sens_at_spec95')
    pt = pt.reindex(CONDITIONS)
    cols = [c for c in TOP6 if c in pt.columns]
    pt = pt[cols]
    lines.append("| condition | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for cond in CONDITIONS:
        if cond not in pt.index:
            continue
        cells = []
        for c in cols:
            v = pt.loc[cond, c]
            cells.append(f"{v:.3f}" if not np.isnan(v) else 'n/a')
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Slicing-(i) AUROC — pivot",
        "",
        "| condition | " + " | ".join(cols) + " |",
        "|---|" + "|".join(["---:"] * len(cols)) + "|",
    ]
    pt2 = df_i.pivot_table(index='condition', columns='config',
                           values='auroc').reindex(CONDITIONS)[cols]
    for cond in CONDITIONS:
        if cond not in pt2.index:
            continue
        cells = []
        for c in cols:
            v = pt2.loc[cond, c]
            cells.append(f"{v:.3f}" if not np.isnan(v) else 'n/a')
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Aggregate sensitivity-checks (reported, NOT used to pick winner)",
        "",
        "| config | any-disease AUROC (95% CI) | any-disease sens@95 | prev-weighted sens@95 |",
        "|---|---|---:|---:|",
    ]
    for _, row in agg_df.iterrows():
        ci = (f"{row['any_disease_auroc']:.3f} "
              f"[{row['any_disease_auroc_lo']:.3f}, "
              f"{row['any_disease_auroc_hi']:.3f}]")
        lines.append(
            f"| {row['config']} | {ci} | "
            f"{row['any_disease_sens_at_spec95']:.3f} | "
            f"{row['prevalence_weighted_sens_at_spec95']:.3f} |"
        )
    lines.append("")

    lines += ["## Mean sens@95 across 9 conditions (coordinated bootstrap)", ""]
    lines.append("| config | point | 95% CI |")
    lines.append("|---|---:|---|")
    for cfg in TOP6:
        if cfg not in mean_ci:
            continue
        ci = mean_ci[cfg]
        lo_hi = (f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"
                 if not np.isnan(ci['lo']) else 'n/a')
        lines.append(f"| {cfg} | {ci['point']:.3f} | {lo_hi} |")
    lines.append("")

    lines += ["## Selection-rule output", ""]
    if rule.get('winner'):
        lines.append(f"4-clause winner: **{rule['winner']}**.")
    else:
        lines.append("No config passed all 4 clauses — fallback: highest mean sens@95.")
    lines.append("")
    lines.append("See `selection_rule.md` for per-clause detail.")
    lines.append("")

    lines += [
        "## Caveat",
        "",
        f"8 of 9 conditions have <{MIN_POSITIVES} test positives. Bootstrap CIs on "
        "sens@95 are wide enough that several configs are statistically tied on the "
        "headline metric. Supervisor decision needed on:",
        "",
        "- Merging structurally similar conditions (see `merge_comparison.md`).",
        "- Carving rare conditions (a_stenosis n=5, p_stenosis n=2) into appendix.",
        "- Whether to use the 9-col CSV alone, or report an "
        "agreement-with-token-4 sensitivity slice "
        "(see the label cross-tabulation notes).",
        "",
    ]
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--test-split', type=Path, default=DEFAULT_TEST_SPLIT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--n-boot', type=int, default=DEFAULT_N_BOOT)
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--configs', type=str, default=','.join(TOP6),
                    help='Comma-separated config names (default: top-6)')
    ap.add_argument('--conditions', type=str, default=','.join(CONDITIONS),
                    help='Comma-separated conditions to evaluate (default: 9-col)')
    ap.add_argument('--merge-spec', type=str, default=None,
                    help='Comma-separated merge groups, "name=col1+col2+...". '
                         'Adds a merged column to labels = OR over source columns.')
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"results.json not found: {args.results}")
    if not args.test_split.exists():
        raise SystemExit(f"test_split.csv not found: {args.test_split}")

    args.out.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    print(f"=== Tier 1 metric sweep ===")
    print(f"Results : {args.results}")
    print(f"Labels  : {args.test_split}")
    print(f"Output  : {args.out}")
    print(f"Configs : {configs}")
    print(f"Cond    : {conditions}")
    print(f"Boots   : {args.n_boot}  Seed: {args.seed}")

    labels = load_test_labels(args.test_split)
    if args.merge_spec:
        for entry in args.merge_spec.split(','):
            entry = entry.strip()
            if not entry or '=' not in entry:
                continue
            name, src = entry.split('=', 1)
            sources = [s.strip() for s in src.split('+') if s.strip()]
            missing = [s for s in sources if s not in labels.columns]
            if missing:
                raise SystemExit(f"merge spec {name}: missing source cols {missing}")
            labels[name.strip()] = (labels[sources].sum(axis=1) > 0).astype(int)
            print(f"  merged {name.strip()} = {' OR '.join(sources)}  "
                  f"n_pos={int(labels[name.strip()].sum())}")
    preds_dict = load_results(args.results)
    table_n = load_n_params(args.results)
    n_params_map = {}
    for cfg in configs:
        if cfg not in preds_dict:
            continue
        n = preds_dict[cfg].n_params
        if n is None:
            n = table_n.get(cfg)
        n_params_map[cfg] = n
    print(f"  n_params loaded for {len(n_params_map)} configs from "
          f"{'comparison_table.csv' if table_n else 'results.json only'}")

    counts = per_condition_counts(labels)
    counts.to_csv(args.out / 'per_condition_counts.csv', index=False)

    print(f"\n--- Per-condition sweep ({len(configs)} configs x "
          f"{len(conditions)} cond x 2 slicings) ---")
    t0 = time.time()
    df_i, df_ii = per_condition_sweep(preds_dict, labels, configs, conditions,
                                      args.n_boot, args.seed)
    df_i.to_csv(args.out / 'metrics_slicing_i.csv', index=False)
    df_ii.to_csv(args.out / 'metrics_slicing_ii.csv', index=False)
    print(f"  per-condition sweep: {time.time() - t0:.1f}s")

    print(f"\n--- Aggregate sweep ---")
    t0 = time.time()
    agg = aggregate_sweep(preds_dict, labels, configs, conditions,
                          args.n_boot, args.seed)
    agg.to_csv(args.out / 'aggregate.csv', index=False)
    print(f"  aggregate: {time.time() - t0:.1f}s")
    print(agg.to_string(index=False))

    print(f"\n--- Coordinated bootstrap on mean sens@95 ---")
    t0 = time.time()
    mean_ci = coord_bootstrap_mean_sens95(preds_dict, labels, configs,
                                          conditions, args.n_boot, args.seed)
    print(f"  coord bootstrap: {time.time() - t0:.1f}s")
    for cfg in configs:
        if cfg in mean_ci:
            c = mean_ci[cfg]
            print(f"  {cfg}: point={c['point']:.3f}  CI=[{c['lo']:.3f}, {c['hi']:.3f}]")

    print(f"\n--- 4-clause selection rule ---")
    rule = four_clause_selection(df_i, mean_ci, configs, n_params_map, conditions)
    print(f"  best by mean: {rule.get('best_by_mean')}")
    print(f"  survivors after clauses 1-3: {rule['survivors_final']}")
    print(f"  winner: {rule.get('winner')}")

    (args.out / 'selection_rule.json').write_text(
        json.dumps({'rule': rule, 'mean_ci': mean_ci}, indent=2, default=str))
    write_selection_md(rule, args.out / 'selection_rule.md')
    write_summary_md(df_i, df_ii, agg, rule, mean_ci, counts,
                     args.n_boot, args.seed, args.results,
                     args.out / 'summary.md')
    print(f"\nDone. Outputs in {args.out}")


if __name__ == '__main__':
    main()
