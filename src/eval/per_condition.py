#!/usr/bin/env python3
"""
Per-condition evaluation pipeline (Tier 1).

Loads saved subject-level sigmoid predictions from exp05's results.json,
joins to per-subject condition labels, and produces per-condition metrics
under two negative-set slicings:

  (i)  HEADLINE: positives  = subjects with condition X (any co-morbidity)
                 negatives  = healthy subjects only
  (ii) APPENDIX: positives  = subjects with condition X
                 negatives  = healthy + non-X-condition subjects
                 (binary head's natural test; expected near-random)

Predictions in results.json are SUBJECT-LEVEL (post-aggregation in exp05),
so the bootstrap unit is naturally subject and there is no per-clip
double-counting concern.

Selection rule (4-clause, replaces single ranking metric):
  1. Best-or-tied (within bootstrap CI overlap) on mean sens-at-spec=95
     across the 9 per-condition tables.
  2. Does not lose by > 5 pp to next-best on any individual condition.
  3. ECE <= next-best ECE + 0.05.
  4. Tiebreaker: simpler (fewer params).

Sensitivity-check metrics (reported but NOT used to pick winner):
aggregate any-disease AUROC; prevalence-weighted sens-at-spec=95.

Day-1 deliverable: load + validate test-set membership + per-condition
counts (flag conditions with < 20 positives for merging or appendix
carve-out). A later stage fills in the full metric sweep using the primitives
defined below.

Usage:
  python -m src.eval.per_condition                 # default top-6 inspection
  python -m src.eval.per_condition --config B2_cross_attn
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ['avsd', 'hlhs', 'tga', 'tetralogy', 'raa', 'coa',
              'p_atresia', 'a_stenosis', 'p_stenosis']

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / 'experiments' / 'exp05_temporal_unfiltered' / 'results.json'
DEFAULT_TEST_SPLIT = REPO_ROOT / 'data_exploration' / 'results' / 'remote_results' / 'test_split.csv'

TOP6 = ['B2_cross_attn', 'F2a_hidden64', 'F2b_hidden128',
        'C1_temporal_cnn', 'F1b_learnable_pos', 'E2_mean_pool_baseline']

MIN_POSITIVES = 20  # flag condition for merging/appendix if test positives < this
SPEC_TARGETS = (0.90, 0.95, 0.99)
DEFAULT_SEED = 42
DEFAULT_N_BOOT = 1000


@dataclass(frozen=True)
class ConfigPredictions:
    name: str
    y_prob: np.ndarray         # P(any disease), one per subject
    subject_ids: np.ndarray    # aligned with y_prob
    n_params: int | None       # for selection-rule tiebreaker


def load_results(results_path: Path) -> dict[str, ConfigPredictions]:
    with open(results_path) as f:
        all_results = json.load(f)
    out: dict[str, ConfigPredictions] = {}
    for name, blob in all_results.items():
        preds = blob.get('test_predictions')
        if not preds or 'y_prob' not in preds:
            continue
        sids = preds.get('subject_ids')
        if sids is None:
            raise ValueError(f"{name}: subject_ids missing in test_predictions")
        n_params = blob.get('config', {}).get('n_params')
        if isinstance(n_params, str):
            n_params = None
        out[name] = ConfigPredictions(
            name=name,
            y_prob=np.asarray(preds['y_prob'], dtype=float),
            subject_ids=np.asarray(sids),
            n_params=n_params,
        )
    return out


def load_test_labels(test_split_path: Path) -> pd.DataFrame:
    df = pd.read_csv(test_split_path)
    if 'subject' not in df.columns:
        raise ValueError(f"{test_split_path}: missing 'subject' column")
    df = df.set_index('subject')
    missing = [c for c in CONDITIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Test split missing condition columns: {missing}")
    if 'unhealthy' not in df.columns:
        raise ValueError("Test split missing 'unhealthy' column")
    return df


def validate_membership(preds: ConfigPredictions, labels: pd.DataFrame) -> dict:
    label_ids = set(labels.index.tolist())
    pred_ids = []
    for sid in preds.subject_ids.tolist():
        try:
            pred_ids.append(int(sid))
        except (TypeError, ValueError):
            pred_ids.append(sid)
    pred_set = set(pred_ids)
    return {
        'n_pred': len(pred_set),
        'n_labels': len(label_ids),
        'preds_not_in_labels': sorted(pred_set - label_ids),
        'labels_not_in_preds': sorted(label_ids - pred_set),
    }


def per_condition_counts(labels: pd.DataFrame) -> pd.DataFrame:
    healthy_n = int((labels['unhealthy'] == 0).sum())
    rows = []
    for cond in CONDITIONS:
        n_pos = int((labels[cond] == 1).sum())
        rows.append({
            'condition': cond,
            'n_positive': n_pos,
            'n_healthy_neg': healthy_n,
            'n_total_slicing_i': n_pos + healthy_n,
            'flag_low_n': n_pos < MIN_POSITIVES,
        })
    return pd.DataFrame(rows)


def _coerce_id(sid):
    try:
        return int(sid)
    except (TypeError, ValueError):
        return sid


def slicing_i(preds: ConfigPredictions, labels: pd.DataFrame,
              condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Headline slicing: positives = condition X subjects (any co-morbidity);
    negatives = healthy subjects only. Returns (y_true, y_prob, subject_ids)."""
    keep_idx, y_true, kept_ids = [], [], []
    for i, raw_sid in enumerate(preds.subject_ids):
        sid = _coerce_id(raw_sid)
        if sid not in labels.index:
            continue
        row = labels.loc[sid]
        if int(row[condition]) == 1:
            keep_idx.append(i); y_true.append(1); kept_ids.append(sid)
        elif int(row['unhealthy']) == 0:
            keep_idx.append(i); y_true.append(0); kept_ids.append(sid)
    keep_idx = np.asarray(keep_idx, dtype=int)
    return (np.asarray(y_true, dtype=int),
            preds.y_prob[keep_idx],
            np.asarray(kept_ids))


def slicing_ii(preds: ConfigPredictions, labels: pd.DataFrame,
               condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appendix slicing: positives = condition X subjects; negatives = everyone
    else (healthy + other-condition subjects). Expected near-random for the
    binary head; demonstrates the head is not condition-specific by design."""
    keep_idx, y_true, kept_ids = [], [], []
    for i, raw_sid in enumerate(preds.subject_ids):
        sid = _coerce_id(raw_sid)
        if sid not in labels.index:
            continue
        keep_idx.append(i)
        y_true.append(int(labels.loc[sid, condition] == 1))
        kept_ids.append(sid)
    keep_idx = np.asarray(keep_idx, dtype=int)
    return (np.asarray(y_true, dtype=int),
            preds.y_prob[keep_idx],
            np.asarray(kept_ids))


AUDITABLE_FOUR = ('tga', 'avsd', 'tetralogy', 'hlhs')


def slicing_auditable_four(preds: ConfigPredictions, labels: pd.DataFrame,
                           include_nonauditable_negatives: bool = False
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auditable-4 lumped binary slicing (UK FASP subgroup).

    Positives = subjects with ANY of {TGA, AVSD, Tetralogy, HLHS}
    (regardless of co-morbidity).

    Negatives default to slicing-i style: healthy only. With
    include_nonauditable_negatives=True, subjects unhealthy with no
    auditable-4 condition are also negatives (alternative slicing
    sensitivity check).

    Returns (y_true, y_prob, subject_ids).
    """
    keep_idx, y_true, kept_ids = [], [], []
    for i, raw_sid in enumerate(preds.subject_ids):
        sid = _coerce_id(raw_sid)
        if sid not in labels.index:
            continue
        row = labels.loc[sid]
        has_auditable = any(int(row[c]) == 1 for c in AUDITABLE_FOUR)
        if has_auditable:
            keep_idx.append(i); y_true.append(1); kept_ids.append(sid)
        elif int(row['unhealthy']) == 0:
            keep_idx.append(i); y_true.append(0); kept_ids.append(sid)
        elif include_nonauditable_negatives:
            keep_idx.append(i); y_true.append(0); kept_ids.append(sid)
        # else: non-auditable CHD subject; excluded from both sides
    keep_idx = np.asarray(keep_idx, dtype=int)
    return (np.asarray(y_true, dtype=int),
            preds.y_prob[keep_idx],
            np.asarray(kept_ids))


# --- Metric primitives -----------------------------------------------------

def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def aupr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(average_precision_score(y_true, y_prob))


def sensitivity_at_specificity(y_true: np.ndarray, y_prob: np.ndarray,
                               target_spec: float) -> tuple[float, float]:
    """Return (sensitivity, threshold) at the lowest threshold achieving the
    target specificity, picking the threshold with highest sensitivity among
    those meeting the target. (NaN, NaN) if target unreachable or single class."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y_true)) < 2:
        return float('nan'), float('nan')
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    valid = spec >= target_spec
    if not np.any(valid):
        return float('nan'), float('nan')
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmax(tpr[valid_idx])]
    return float(tpr[best]), float(thresh[best])


def ppv_npv_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float) -> tuple[float, float]:
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    npv = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    return ppv, npv


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece, n = 0.0, len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def _equal_mass_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int):
    n = len(y_prob)
    if n == 0:
        return
    order = np.argsort(y_prob)
    yt, yp = y_true[order], y_prob[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        yield ((hi - lo) / n, float(yt[lo:hi].mean()),
               float(yp[lo:hi].mean()), int(hi - lo))


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray,
                 n_bins: int = 10) -> float:
    if len(y_prob) == 0:
        return float('nan')
    return float(sum(w * abs(acc - conf)
                     for w, acc, conf, _ in _equal_mass_bins(y_true, y_prob, n_bins)))


def brier_murphy(y_true: np.ndarray, y_prob: np.ndarray,
                 n_bins: int = 10) -> dict:
    if len(y_prob) == 0:
        nan = float('nan')
        return {'brier': nan, 'rel': nan, 'res': nan, 'unc': nan,
                'residual': nan}
    brier = float(np.mean((y_prob - y_true) ** 2))
    mean_y = float(y_true.mean())
    unc = mean_y * (1 - mean_y)
    rel = res = 0.0
    for w, acc, conf, _ in _equal_mass_bins(y_true, y_prob, n_bins):
        rel += w * (acc - conf) ** 2
        res += w * (acc - mean_y) ** 2
    return {'brier': brier, 'rel': float(rel), 'res': float(res),
            'unc': float(unc),
            'residual': float(brier - (rel - res + unc))}


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray,
                      n_bins: int = 10) -> dict:
    if len(y_prob) == 0:
        return {'bin_centres': [], 'accs': [], 'counts': [], 'mean_confs': []}
    centres, accs, counts, mean_confs = [], [], [], []
    for _, acc, conf, count in _equal_mass_bins(y_true, y_prob, n_bins):
        centres.append(conf)
        mean_confs.append(conf)
        accs.append(acc)
        counts.append(count)
    return {'bin_centres': centres, 'accs': accs, 'counts': counts,
            'mean_confs': mean_confs}


def _bca_bounds(samples: list[float], point: float, y_true: np.ndarray,
                y_prob: np.ndarray, metric_fn, alpha: float
                ) -> tuple[float, float]:
    """BCa quantile transformation (Efron 1987).

    Returns (lo, hi). Falls back to percentile + RuntimeWarning when
    bias-correction or acceleration degenerates at the boundary.
    """
    import warnings
    from scipy.stats import norm as _norm
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    prop_below = float(np.mean(arr < point))
    if prop_below <= 0.0 or prop_below >= 1.0:
        warnings.warn("BCa: z0 degenerate (point at boundary of bootstrap "
                      "distribution); falling back to percentile.",
                      RuntimeWarning)
        return (float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)))
    z0 = _norm.ppf(prop_below)
    n = len(y_true)
    jack = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            v = metric_fn(y_true[mask], y_prob[mask])
        except Exception:
            continue
        if v is not None and not np.isnan(v):
            jack.append(v)
    if len(jack) < 2:
        warnings.warn("BCa: jackknife produced < 2 valid estimates; falling "
                      "back to percentile.", RuntimeWarning)
        return (float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)))
    jack = np.asarray(jack, dtype=float)
    jbar = float(jack.mean())
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * float(np.sum((jbar - jack) ** 2)) ** 1.5
    if den <= 0.0 or not np.isfinite(den):
        a = 0.0
    else:
        a = num / den
    z_a = _norm.ppf(alpha)
    z_1ma = _norm.ppf(1 - alpha)
    def _adj(z_q):
        denom = 1.0 - a * (z0 + z_q)
        if denom <= 0.0 or not np.isfinite(denom):
            return None
        return _norm.cdf(z0 + (z0 + z_q) / denom)
    alpha_lo = _adj(z_a)
    alpha_hi = _adj(z_1ma)
    if alpha_lo is None or alpha_hi is None:
        warnings.warn("BCa: acceleration adjustment denominator non-positive; "
                      "falling back to percentile.", RuntimeWarning)
        return (float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)))
    alpha_lo = float(np.clip(alpha_lo, 1e-6, 1 - 1e-6))
    alpha_hi = float(np.clip(alpha_hi, 1e-6, 1 - 1e-6))
    return (float(np.quantile(arr, alpha_lo)),
            float(np.quantile(arr, alpha_hi)))


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn,
                 n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                 ci: float = 0.95,
                 method: str = 'percentile') -> tuple[float, float, float]:
    """Per-subject block bootstrap (predictions are already subject-level).
    Returns (point, lo, hi).

    method='percentile' (default) gives standard percentile intervals; kept
    as default for backwards compatibility. method='bca' returns
    bias-corrected and accelerated (Efron 1987) intervals; degenerate cases
    (point at boundary, jackknife singletons, negative acceleration
    denominator) fall back to percentile + emit RuntimeWarning."""
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_prob)
    n = len(y_true)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            v = metric_fn(y_true[idx], y_prob[idx])
        except Exception:
            continue
        if not np.isnan(v):
            samples.append(v)
    if not samples:
        return float(point), float('nan'), float('nan')
    alpha = (1 - ci) / 2
    if method == 'bca':
        lo, hi = _bca_bounds(samples, float(point), y_true, y_prob,
                             metric_fn, alpha)
        return float(point), lo, hi
    return (float(point),
            float(np.quantile(samples, alpha)),
            float(np.quantile(samples, 1 - alpha)))


# --- Day 1 main: load + validate + counts ----------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS,
                        help=f'Path to exp05 results.json (default: {DEFAULT_RESULTS})')
    parser.add_argument('--test-split', type=Path, default=DEFAULT_TEST_SPLIT,
                        help=f'Path to test_split.csv (default: {DEFAULT_TEST_SPLIT})')
    parser.add_argument('--config', type=str, default=None,
                        help='Single config to inspect; default = all top-6')
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"results.json not found: {args.results}\n"
            "Generate it by running the relevant experiment, or point --results "
            "at an existing per-fold results.json (see RESULTS_DIR in src/paths.py)."
        )

    labels = load_test_labels(args.test_split)
    preds = load_results(args.results)

    print("=== per_condition.py | day-1 inspection ===")
    print(f"Loaded {len(preds)} configs from {args.results.name}")
    print(f"Loaded {len(labels)} test subjects from {args.test_split.name}")

    print("\n--- Per-condition counts (test split) ---")
    counts = per_condition_counts(labels)
    print(counts.to_string(index=False))
    flagged = counts[counts['flag_low_n']]
    if len(flagged):
        print(f"\nFLAG: {len(flagged)} conditions have < {MIN_POSITIVES} positives:")
        for _, row in flagged.iterrows():
            print(f"  - {row['condition']}: n_pos={row['n_positive']}")
        print("Decision needed: merge variants (e.g. p_stenosis family) "
              "OR carve into 'rare conditions' appendix paragraph.")
    else:
        print(f"\nAll {len(CONDITIONS)} conditions have >= {MIN_POSITIVES} positives.")

    configs_to_inspect = [args.config] if args.config else TOP6
    print("\n--- Test-set membership validation ---")
    for cfg_name in configs_to_inspect:
        if cfg_name not in preds:
            print(f"  {cfg_name}: NOT FOUND in results.json")
            continue
        v = validate_membership(preds[cfg_name], labels)
        status = "OK" if not (v['preds_not_in_labels'] or v['labels_not_in_preds']) else "MISMATCH"
        print(f"  {cfg_name}: pred={v['n_pred']} labels={v['n_labels']}  {status}")
        if v['preds_not_in_labels']:
            print(f"    preds_not_in_labels ({len(v['preds_not_in_labels'])}): "
                  f"{v['preds_not_in_labels'][:5]}...")
        if v['labels_not_in_preds']:
            print(f"    labels_not_in_preds ({len(v['labels_not_in_preds'])}): "
                  f"{v['labels_not_in_preds'][:5]}...")

    print("\n--- Slicing-(i) sanity check on B2_cross_attn / hlhs ---")
    if 'B2_cross_attn' in preds:
        y_true, y_prob, sids = slicing_i(preds['B2_cross_attn'], labels, 'hlhs')
        print(f"  n={len(y_true)}  n_pos={int(y_true.sum())}  n_neg={int((y_true==0).sum())}")
        print(f"  AUROC = {auroc(y_true, y_prob):.4f}")
        print(f"  AUPR  = {aupr(y_true, y_prob):.4f}")
        for spec in SPEC_TARGETS:
            sens, thr = sensitivity_at_specificity(y_true, y_prob, spec)
            print(f"  sens@spec={spec:.2f}: sens={sens:.3f}  thr={thr:.3f}")


if __name__ == '__main__':
    main()
