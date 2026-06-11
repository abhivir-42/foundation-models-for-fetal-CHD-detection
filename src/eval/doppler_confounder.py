#!/usr/bin/env python3
"""Doppler-as-confounder statistical analysis.

Quantifies how much of the binary-head signal correlates with per-subject
Doppler-burden. Framed neutrally on clinical advice: a learned
proxy for sonographer suspicion may be useful signal rather than a
confound to be removed.

Primary target  : Tier 2 FetalCLIP (saw Doppler at extraction).
Sanity check    : Tier 1 DINOv2 E2 (Doppler-filtered at extraction -- should
                  show ~zero confound effect).

Bootstrap unit  : subject. CI method: BCa via per_condition.bootstrap_ci.

Outputs land under RESULTS_DIR.
"""
from __future__ import annotations


import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .per_condition import (CONDITIONS, bootstrap_ci, load_results,
                            load_test_labels)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOPPLER_CSV = REPO_ROOT / 'results' / 'doppler_filter' / 'doppler_results_full.csv'
PER_SUBJECT_FEATURES_CSV = REPO_ROOT / 'results' / 'doppler_filter' / 'per_subject_doppler_features.csv'
TEST_SPLIT = REPO_ROOT / 'data_exploration' / 'results' / 'remote_results' / 'test_split.csv'

TIER2_RESULTS = Path(str(_Path_scrub(RESULTS_DIR) / 'exp06_fetalclip/results.json'))
TIER1_TEMPSCALED_RESULTS = (REPO_ROOT / 'experiments' / 'exp05_temporal_unfiltered'
                            / 'tempscaling' / 'results_temperature_scaled.json')

AUDITABLE_4 = ('tga', 'avsd', 'tetralogy', 'hlhs')

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42


# -- Step 1: per-subject Doppler features -----------------------------------

def subject_id_from_filename(fn: str) -> int:
    """First underscore-separated token is the subject ID (iFIND convention)."""
    return int(str(fn).split('_')[0])


def build_per_subject_doppler_features(doppler_csv: Path = DOPPLER_CSV
                                       ) -> pd.DataFrame:
    """Aggregate per-video Doppler annotations to per-subject features.

    Columns returned (indexed by subject):
        n_videos, n_doppler_videos, frac_doppler_videos,
        mean_max_doppler_pct, max_doppler_pct, sum_doppler_pct.
    """
    df = pd.read_csv(doppler_csv)
    df['subject'] = df['filename'].apply(subject_id_from_filename)
    has = df['has_doppler'].astype(bool)
    g = df.assign(_has=has.astype(int)).groupby('subject')
    feats = pd.DataFrame({
        'n_videos': g.size(),
        'n_doppler_videos': g['_has'].sum(),
        'frac_doppler_videos': g['_has'].mean(),
        'mean_max_doppler_pct': g['max_doppler_pct'].mean(),
        'max_doppler_pct': g['max_doppler_pct'].max(),
        'sum_doppler_pct': g['max_doppler_pct'].sum(),
    })
    return feats


# -- Step 2: condition x Doppler cross-tab ----------------------------------

def condition_doppler_table(labels: pd.DataFrame, doppler_feats: pd.DataFrame
                            ) -> pd.DataFrame:
    """For each condition group, mean per-subject frac_doppler_videos.

    Index rows: healthy, any_unhealthy, auditable_4_positive, each of CONDITIONS.
    """
    joined = labels.join(doppler_feats, how='inner')
    rows = []

    def _row(label: str, mask: pd.Series) -> dict:
        sub = joined.loc[mask]
        return {
            'group': label,
            'n_subjects': int(len(sub)),
            'mean_frac_doppler': float(sub['frac_doppler_videos'].mean()),
            'median_frac_doppler': float(sub['frac_doppler_videos'].median()),
            'mean_max_doppler_pct': float(sub['mean_max_doppler_pct'].mean()),
            'mean_n_videos': float(sub['n_videos'].mean()),
        }

    healthy_mask = (joined['unhealthy'] == 0)
    unhealthy_mask = (joined['unhealthy'] == 1)
    auditable_mask = joined[list(AUDITABLE_4)].sum(axis=1) > 0

    rows.append(_row('healthy', healthy_mask))
    rows.append(_row('any_unhealthy', unhealthy_mask))
    rows.append(_row('auditable_4_positive', auditable_mask))
    for c in CONDITIONS:
        rows.append(_row(c, joined[c] == 1))

    return pd.DataFrame(rows)


# -- Step 3 helpers: quartile-stratified AUROC, propensity match ------------

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def quartile_stratified_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                              dop: np.ndarray, n_boot: int = DEFAULT_N_BOOT,
                              seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Stratify into 4 quartiles by `dop`, compute AUROC + BCa CI per quartile."""
    qs = np.quantile(dop, [0.25, 0.5, 0.75])
    bins = np.digitize(dop, qs, right=False)  # 0..3
    rows = []
    for q in range(4):
        m = (bins == q)
        if m.sum() == 0:
            rows.append({'quartile': f'Q{q+1}', 'n': 0, 'n_pos': 0,
                         'frac_doppler_lo': float('nan'),
                         'frac_doppler_hi': float('nan'),
                         'auroc': float('nan'),
                         'ci_lo': float('nan'), 'ci_hi': float('nan')})
            continue
        yt, yp = y_true[m], y_prob[m]
        try:
            point, lo, hi = bootstrap_ci(yt, yp, _safe_auroc,
                                         n_boot=n_boot, seed=seed,
                                         method='bca')
        except Exception:
            point, lo, hi = _safe_auroc(yt, yp), float('nan'), float('nan')
        rows.append({
            'quartile': f'Q{q+1}',
            'n': int(m.sum()),
            'n_pos': int(yt.sum()),
            'frac_doppler_lo': float(dop[m].min()),
            'frac_doppler_hi': float(dop[m].max()),
            'auroc': point,
            'ci_lo': lo,
            'ci_hi': hi,
        })
    return pd.DataFrame(rows)


def delta_q4_minus_q1_bootstrap(y_true: np.ndarray, y_prob: np.ndarray,
                                dop: np.ndarray, n_boot: int = DEFAULT_N_BOOT,
                                seed: int = DEFAULT_SEED
                                ) -> tuple[float, float, float]:
    """Bootstrap CI for AUROC(Q4) - AUROC(Q1).

    Each bootstrap draws subjects with replacement, recomputes quartile cuts on
    the resampled `dop` (preserving the relative-rank logic), and recomputes
    Q1/Q4 AUROC on the resample. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _point(yt, yp, d):
        qs = np.quantile(d, [0.25, 0.75])
        q1 = d <= qs[0]
        q4 = d >= qs[1]
        a1 = _safe_auroc(yt[q1], yp[q1])
        a4 = _safe_auroc(yt[q4], yp[q4])
        if np.isnan(a1) or np.isnan(a4):
            return float('nan')
        return a4 - a1

    point = _point(y_true, y_prob, dop)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = _point(y_true[idx], y_prob[idx], dop[idx])
        if not np.isnan(v):
            samples.append(v)
    if not samples:
        return float(point), float('nan'), float('nan')
    return float(point), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def propensity_match_within_pair(labels: pd.DataFrame, scores: pd.Series,
                                 dop: pd.Series, tol: float = 0.05,
                                 seed: int = DEFAULT_SEED
                                 ) -> pd.DataFrame:
    """Nearest-neighbour 1-1 match on frac_doppler_videos.

    For each pathological subject (unhealthy==1), find a healthy subject with
    |dop_p - dop_h| <= tol and minimal absolute difference (unused-once).
    Returns paired DataFrame: subject_p, subject_h, dop_p, dop_h, score_p,
    score_h, score_diff (=score_p - score_h).
    """
    rng = np.random.default_rng(seed)
    joined = labels.join(scores.to_frame('score')).join(dop.to_frame('dop'))
    pos = joined[joined['unhealthy'] == 1].copy()
    neg = joined[joined['unhealthy'] == 0].copy()
    # Shuffle positives so the greedy order is randomised.
    pos = pos.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    used_neg: set = set()
    pairs = []
    for sid_p, row_p in pos.iterrows():
        d_p = row_p['dop']
        cand = neg.loc[~neg.index.isin(used_neg)]
        if cand.empty:
            break
        diffs = (cand['dop'] - d_p).abs()
        best = diffs.idxmin()
        if diffs.loc[best] > tol:
            continue
        used_neg.add(best)
        row_h = neg.loc[best]
        pairs.append({
            'subject_pos': sid_p, 'subject_neg': best,
            'dop_pos': float(d_p), 'dop_neg': float(row_h['dop']),
            'score_pos': float(row_p['score']),
            'score_neg': float(row_h['score']),
            'score_diff': float(row_p['score'] - row_h['score']),
        })
    return pd.DataFrame(pairs)


# -- Step 4: Spearman correlation within healthy ----------------------------

def spearman_within_healthy_bootstrap(scores: np.ndarray, dop: np.ndarray,
                                      n_boot: int = DEFAULT_N_BOOT,
                                      seed: int = DEFAULT_SEED
                                      ) -> dict:
    """Spearman rho between model score and Doppler-burden (within-healthy).

    Returns dict: rho_point, ci_lo, ci_hi, p_value_point.
    """
    point = spearmanr(scores, dop)
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            r = spearmanr(scores[idx], dop[idx]).correlation
        except Exception:
            continue
        if r is not None and not np.isnan(r):
            samples.append(float(r))
    if not samples:
        return {'rho': float(point.correlation), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'p_value_point': float(point.pvalue)}
    return {'rho': float(point.correlation),
            'ci_lo': float(np.quantile(samples, 0.025)),
            'ci_hi': float(np.quantile(samples, 0.975)),
            'p_value_point': float(point.pvalue),
            'n': int(n)}


# -- Step 5: with/without Doppler subset AUROC ------------------------------

def subset_auroc(y_true: np.ndarray, y_prob: np.ndarray, dop: np.ndarray,
                 mask: np.ndarray, n_boot: int = DEFAULT_N_BOOT,
                 seed: int = DEFAULT_SEED) -> dict:
    yt, yp = y_true[mask], y_prob[mask]
    if len(np.unique(yt)) < 2:
        return {'n': int(mask.sum()), 'n_pos': int(yt.sum()),
                'auroc': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan')}
    try:
        point, lo, hi = bootstrap_ci(yt, yp, _safe_auroc, n_boot=n_boot,
                                     seed=seed, method='bca')
    except Exception:
        point, lo, hi = _safe_auroc(yt, yp), float('nan'), float('nan')
    return {'n': int(mask.sum()), 'n_pos': int(yt.sum()),
            'auroc': point, 'ci_lo': lo, 'ci_hi': hi}


# -- Driver -----------------------------------------------------------------

def _load_config_preds(results_path: Path, config_name: str
                        ) -> pd.DataFrame:
    """Returns DataFrame indexed by subject with columns y_true, y_prob."""
    preds = load_results(results_path)
    if config_name not in preds:
        raise KeyError(f"{config_name} not in {results_path}; available: "
                       f"{sorted(preds)}")
    cp = preds[config_name]
    sid = [int(s) for s in cp.subject_ids.tolist()]
    blob = json.load(open(results_path))[config_name]['test_predictions']
    y_true = np.asarray(blob['y_true'], dtype=int)
    df = pd.DataFrame({'y_true': y_true, 'y_prob': cp.y_prob}, index=sid)
    df.index.name = 'subject'
    return df


def run_full_pipeline(out_dir: Path, n_boot: int = DEFAULT_N_BOOT,
                      tier2_config: str = 'F2a_hidden64',
                      tier1_config: str = 'E2_mean_pool_baseline') -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = out_dir / 'outputs'
    outputs.mkdir(exist_ok=True)

    # Step 1 -- per-subject Doppler features (or load cached)
    if PER_SUBJECT_FEATURES_CSV.exists():
        feats = pd.read_csv(PER_SUBJECT_FEATURES_CSV).set_index('subject')
    else:
        feats = build_per_subject_doppler_features(DOPPLER_CSV)
        PER_SUBJECT_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(PER_SUBJECT_FEATURES_CSV)

    sanity_mean = float(feats['frac_doppler_videos'].mean())

    # Step 2 -- condition x Doppler cross-tab
    labels = load_test_labels(TEST_SPLIT)
    cross = condition_doppler_table(labels, feats)
    cross.to_csv(outputs / 'condition_vs_doppler_table.csv', index=False)

    healthy_frac = float(cross.loc[cross['group'] == 'healthy',
                                   'mean_frac_doppler'].iloc[0])
    unhealthy_frac = float(cross.loc[cross['group'] == 'any_unhealthy',
                                     'mean_frac_doppler'].iloc[0])
    delta_step2 = unhealthy_frac - healthy_frac
    short_circuit = abs(delta_step2) < 0.05

    summary = {
        'n_subjects_with_doppler_features': int(len(feats)),
        'sanity_mean_frac_doppler_videos': sanity_mean,
        'sanity_target_22_59_pct': 0.2259,
        'healthy_mean_frac_doppler': healthy_frac,
        'unhealthy_mean_frac_doppler': unhealthy_frac,
        'delta_unhealthy_minus_healthy': delta_step2,
        'short_circuit_at_step2': bool(short_circuit),
        'tier2_config': tier2_config,
        'tier1_config': tier1_config,
    }

    # Step 3-5 only if not short-circuiting
    if not short_circuit:
        for tier_name, results_path, config_name in (
                ('tier2', TIER2_RESULTS, tier2_config),
                ('tier1', TIER1_TEMPSCALED_RESULTS, tier1_config)):
            try:
                preds = _load_config_preds(results_path, config_name)
            except (KeyError, FileNotFoundError) as e:
                summary[f'{tier_name}_error'] = str(e)
                continue
            joined = preds.join(feats[['frac_doppler_videos']], how='inner')
            joined = joined.join(labels[['unhealthy']], how='inner')
            y_true = joined['y_true'].to_numpy().astype(int)
            y_prob = joined['y_prob'].to_numpy().astype(float)
            dop = joined['frac_doppler_videos'].to_numpy().astype(float)

            # 3a) quartile-stratified AUROC
            quart = quartile_stratified_auroc(y_true, y_prob, dop,
                                              n_boot=n_boot)
            quart.to_csv(outputs / f'quartile_auroc_{tier_name}.csv',
                         index=False)

            # 3b) Q4 - Q1 delta (with bootstrap CI on the difference)
            dpt, dlo, dhi = delta_q4_minus_q1_bootstrap(y_true, y_prob, dop,
                                                       n_boot=n_boot)

            # 3c) propensity match
            scores_s = joined['y_prob']
            dop_s = joined['frac_doppler_videos']
            pairs = propensity_match_within_pair(
                labels.join(feats, how='inner').loc[joined.index],
                scores_s, dop_s, tol=0.05)
            pairs.to_csv(outputs / f'propensity_pairs_{tier_name}.csv',
                         index=False)
            if not pairs.empty:
                # Bootstrap CI on the mean score_diff across pairs.
                diffs = pairs['score_diff'].to_numpy()
                rng = np.random.default_rng(DEFAULT_SEED)
                bs = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(diffs), size=len(diffs))
                    bs.append(float(np.mean(diffs[idx])))
                pm = {'n_pairs': int(len(diffs)),
                      'mean_score_diff_pos_minus_neg': float(diffs.mean()),
                      'ci_lo': float(np.quantile(bs, 0.025)),
                      'ci_hi': float(np.quantile(bs, 0.975))}
            else:
                pm = {'n_pairs': 0,
                      'mean_score_diff_pos_minus_neg': float('nan'),
                      'ci_lo': float('nan'), 'ci_hi': float('nan')}

            # Step 4 -- Spearman within healthy
            hm = joined['unhealthy'] == 0
            spr = spearman_within_healthy_bootstrap(
                y_prob[hm.to_numpy()], dop[hm.to_numpy()], n_boot=n_boot)

            # Step 5 -- with/without Doppler subset AUROCs
            full = subset_auroc(y_true, y_prob, dop,
                                np.ones_like(y_true, dtype=bool),
                                n_boot=n_boot)
            lowmask = dop <= 0.1
            highmask = dop >= 0.3
            low = subset_auroc(y_true, y_prob, dop, lowmask, n_boot=n_boot)
            high = subset_auroc(y_true, y_prob, dop, highmask, n_boot=n_boot)

            summary[tier_name] = {
                'quartile_table_csv': str(outputs / f'quartile_auroc_{tier_name}.csv'),
                'q1_auroc': float(quart.loc[0, 'auroc']),
                'q4_auroc': float(quart.loc[3, 'auroc']),
                'delta_q4_minus_q1': dpt,
                'delta_q4_minus_q1_ci': [dlo, dhi],
                'propensity_match': pm,
                'spearman_within_healthy': spr,
                'full_auroc': full,
                'low_doppler_subset_auroc': low,
                'high_doppler_subset_auroc': high,
                'n_subjects_joined': int(len(joined)),
            }
    else:
        summary['note'] = ('healthy vs unhealthy frac_doppler differ by < 0.05; '
                           'short-circuited.')

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path,
                        default=REPO_ROOT / 'experiments' / 'doppler_confounder')
    parser.add_argument('--n-boot', type=int, default=DEFAULT_N_BOOT)
    args = parser.parse_args()
    s = run_full_pipeline(args.out_dir, n_boot=args.n_boot)
    print(json.dumps(s, indent=2, default=float))
