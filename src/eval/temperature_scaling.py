#!/usr/bin/env python3
"""Temperature-scaling killer experiment for the 5 deep configs.

Temperature-scaling calibration analysis.

Fits a single scalar T per config on val sigmoid-recovered logits via NLL
minimisation; applies to test, recomputes per-condition mean sens@spec=0.95
+ paired bootstrap (E2 - cfg) on shared resamples, both pre- and post-T.
"""
from __future__ import annotations


import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

from src.eval.per_condition import (
    CONDITIONS, ConfigPredictions, _coerce_id,
    load_results, load_test_labels, sensitivity_at_specificity,
)

DEEP_CONFIGS = ['B2_cross_attn', 'F2a_hidden64', 'F2b_hidden128',
                'C1_temporal_cnn', 'F1b_learnable_pos']
REFERENCE = 'E2_mean_pool_baseline'
HEADLINE_SPEC = 0.95
EPS = 1e-6
N_BOOT = 1000
SEED = 42


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(test_probs: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-probs_to_logits(test_probs) / T))


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> tuple[float, float, float]:
    def nll(T: float) -> float:
        z = val_logits / T
        log_sig = -np.logaddexp(0.0, -z)
        log_one_minus_sig = -np.logaddexp(0.0, z)
        return float(-(val_y * log_sig + (1.0 - val_y) * log_one_minus_sig).mean())
    res = minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded',
                          options={'xatol': 1e-4})
    return float(res.x), nll(1.0), nll(res.x)


def regenerate_val_probs(exp_name: str, ckpt_path: Path, val_loader,
                         device: torch.device) -> tuple[np.ndarray, np.ndarray, list]:
    from src.exp05_temporal_modelling import (
        evaluate_video_model, get_experiment_configs,
    )
    configs = get_experiment_configs()
    cfg = configs[exp_name]
    model = cfg['model_fn']().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    _, y_t, y_p, sids = evaluate_video_model(model, val_loader, device)
    return y_t.astype(int), y_p.astype(float), sids


def mean_sens95_for(preds: ConfigPredictions, labels: pd.DataFrame,
                    sid_arr: np.ndarray, idx: np.ndarray,
                    cond_mask: dict, healthy_mask: np.ndarray,
                    sid_to_pred: dict) -> float:
    per = []
    for cond in CONDITIONS:
        pos_pos = idx[cond_mask[cond][idx]]
        neg_pos = idx[healthy_mask[idx]]
        pos_pred = [sid_to_pred[s] for s in sid_arr[pos_pos] if s in sid_to_pred]
        neg_pred = [sid_to_pred[s] for s in sid_arr[neg_pos] if s in sid_to_pred]
        if not pos_pred or not neg_pred:
            continue
        yt = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
        yp = np.concatenate([preds.y_prob[pos_pred], preds.y_prob[neg_pred]])
        s, _ = sensitivity_at_specificity(yt, yp, HEADLINE_SPEC)
        if not np.isnan(s):
            per.append(s)
    return float(np.mean(per)) if per else float('nan')


def paired_bootstrap_e2_minus(reference_preds: ConfigPredictions,
                              other_preds: dict[str, ConfigPredictions],
                              labels: pd.DataFrame,
                              n_boot: int = N_BOOT, seed: int = SEED):
    sid_arr = np.asarray(labels.index.tolist())
    n_test = len(sid_arr)
    cond_mask = {c: (labels[c] == 1).values for c in CONDITIONS}
    healthy_mask = (labels['unhealthy'] == 0).values

    def sid_map(p: ConfigPredictions) -> dict:
        return {_coerce_id(s): i for i, s in enumerate(p.subject_ids)}
    ref_map = sid_map(reference_preds)
    other_maps = {cfg: sid_map(p) for cfg, p in other_preds.items()}

    full = np.arange(n_test)
    point = {cfg: mean_sens95_for(p, labels, sid_arr, full, cond_mask, healthy_mask,
                                  other_maps[cfg])
             for cfg, p in other_preds.items()}
    point[REFERENCE] = mean_sens95_for(reference_preds, labels, sid_arr, full,
                                       cond_mask, healthy_mask, ref_map)

    rng = np.random.default_rng(seed)
    deltas = {cfg: [] for cfg in other_preds}
    for _ in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        e2 = mean_sens95_for(reference_preds, labels, sid_arr, idx,
                             cond_mask, healthy_mask, ref_map)
        if np.isnan(e2):
            continue
        for cfg, p in other_preds.items():
            v = mean_sens95_for(p, labels, sid_arr, idx, cond_mask, healthy_mask,
                                other_maps[cfg])
            if not np.isnan(v):
                deltas[cfg].append(e2 - v)

    rows = []
    for cfg in other_preds:
        d = np.array(deltas[cfg]) if deltas[cfg] else np.array([np.nan])
        lo = float(np.quantile(d, 0.025)) if not np.all(np.isnan(d)) else float('nan')
        hi = float(np.quantile(d, 0.975)) if not np.all(np.isnan(d)) else float('nan')
        p_le = float(np.mean(d <= 0)) if not np.all(np.isnan(d)) else float('nan')
        rows.append({
            'config': cfg,
            'point_e2': point[REFERENCE],
            'point_cfg': point[cfg],
            'point_delta': point[REFERENCE] - point[cfg],
            'ci_lo': lo, 'ci_hi': hi,
            'p_le': p_le,
            'sig': (lo > 0 or hi < 0) if not np.isnan(lo) else False,
        })
    return rows, point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results', type=Path,
                   default=Path(str(_Path_scrub(RESULTS_DIR) / 'exp05_temporal/results.json')))
    p.add_argument('--checkpoints', type=Path,
                   default=Path(str(_Path_scrub(RESULTS_DIR) / 'exp05_temporal/checkpoints')))
    p.add_argument('--embeddings', type=Path,
                   default=Path(str(_Path_scrub(EMBEDDINGS_DIR) / 'raw_video_embeddings_clean.pkl')))
    p.add_argument('--test-split', type=Path,
                   default=Path(str(_Path_scrub(RESULTS_DIR) / 'results/remote_results/test_split.csv')))
    p.add_argument('--out', type=Path,
                   default=Path(str(_Path_scrub(RESULTS_DIR) / 'exp05_temporal_unfiltered/tempscaling')))
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=64)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[tempscaling] device={device}", flush=True)

    from src.exp05_temporal_modelling import (
        VideoDataset, collate_video_batch, load_data,
    )
    from torch.utils.data import DataLoader

    t0 = time.time()
    raw_emb, labels_df, _, val_subj, _ = load_data(args.embeddings)
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    val_ds = VideoDataset(val_subj, raw_emb, labels_dict)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_video_batch, num_workers=0)
    print(f"[tempscaling] val dataset built ({len(val_ds)} videos, {len(val_subj)} subjects) "
          f"in {time.time()-t0:.1f}s", flush=True)

    with open(args.results) as f:
        all_results = json.load(f)
    test_preds_all = load_results(args.results)
    if REFERENCE not in test_preds_all:
        sys.exit(f"reference config {REFERENCE} missing from {args.results}")
    test_labels = load_test_labels(args.test_split)
    print(f"[tempscaling] test labels loaded ({len(test_labels)} subjects)", flush=True)

    pre_other = {cfg: test_preds_all[cfg] for cfg in DEEP_CONFIGS if cfg in test_preds_all}
    pre_rows, pre_points = paired_bootstrap_e2_minus(test_preds_all[REFERENCE],
                                                     pre_other, test_labels)
    print(f"[tempscaling] pre-T paired bootstrap done in {time.time()-t0:.1f}s",
          flush=True)

    temperatures = {}
    post_other = {}
    scaled_results = json.loads(json.dumps(all_results))
    for cfg in DEEP_CONFIGS:
        if cfg not in test_preds_all:
            print(f"[tempscaling] skip {cfg} (no test preds)", flush=True)
            continue
        ckpt = args.checkpoints / f"{cfg}.pt"
        if not ckpt.exists():
            print(f"[tempscaling] skip {cfg} (no checkpoint at {ckpt})", flush=True)
            continue
        t1 = time.time()
        val_y, val_p, _ = regenerate_val_probs(cfg, ckpt, val_loader, device)
        val_logits = probs_to_logits(val_p)
        T, nll_pre, nll_post = fit_temperature(val_logits, val_y)
        temperatures[cfg] = {
            'T': T,
            'val_nll_pre': nll_pre,
            'val_nll_post': nll_post,
            'val_n': int(len(val_y)),
            'wall_seconds': float(time.time() - t1),
        }
        print(f"[tempscaling] {cfg}: T={T:.3f}  val_nll {nll_pre:.4f} -> {nll_post:.4f}  "
              f"wall={time.time()-t1:.1f}s", flush=True)
        test_pred_block = scaled_results[cfg]['test_predictions']
        test_p = np.asarray(test_pred_block['y_prob'])
        test_p_T = apply_temperature(test_p, T)
        test_pred_block['y_prob'] = test_p_T.tolist()
        post_other[cfg] = ConfigPredictions(
            name=cfg,
            y_prob=test_p_T,
            subject_ids=test_preds_all[cfg].subject_ids,
            n_params=test_preds_all[cfg].n_params,
        )

    post_rows, post_points = paired_bootstrap_e2_minus(test_preds_all[REFERENCE],
                                                       post_other, test_labels)
    print(f"[tempscaling] post-T paired bootstrap done in {time.time()-t0:.1f}s",
          flush=True)

    (args.out / 'temperatures.json').write_text(json.dumps(temperatures, indent=2))
    (args.out / 'results_temperature_scaled.json').write_text(
        json.dumps(scaled_results, indent=None, separators=(',', ':')))

    delta_md = ['# Temperature-scaling delta table',
                f'\nGenerated {time.strftime("%Y-%m-%d %H:%M")} on {device}. '
                f'n_boot={N_BOOT}, seed={SEED}, spec={HEADLINE_SPEC}, '
                f'val n={next(iter(temperatures.values()))["val_n"] if temperatures else 0}.\n',
                '| config | T | mean sens@95 pre | mean sens@95 post | Δ post-pre | (E2-cfg) pre CI | (E2-cfg) post CI |',
                '|---|---:|---:|---:|---:|---|---|']
    for cfg in DEEP_CONFIGS:
        if cfg not in temperatures:
            continue
        T = temperatures[cfg]['T']
        pre_pt = pre_points[cfg]
        post_pt = post_points[cfg]
        pre_ci = next(r for r in pre_rows if r['config'] == cfg)
        post_ci = next(r for r in post_rows if r['config'] == cfg)
        delta_md.append(
            f"| {cfg} | {T:.3f} | {pre_pt:.4f} | {post_pt:.4f} | "
            f"{post_pt - pre_pt:+.4f} | "
            f"[{pre_ci['ci_lo']:+.3f}, {pre_ci['ci_hi']:+.3f}] (p={pre_ci['p_le']:.3f}) | "
            f"[{post_ci['ci_lo']:+.3f}, {post_ci['ci_hi']:+.3f}] (p={post_ci['p_le']:.3f}) |"
        )
    delta_md.append('')
    delta_md.append(f'E2 reference (unscaled) mean sens@95 = {pre_points[REFERENCE]:.4f}.')
    (args.out / 'delta_table.md').write_text('\n'.join(delta_md))

    post_means = [post_points[c] for c in DEEP_CONFIGS if c in post_points]
    n_close = sum(1 for v in post_means if v >= 0.30)
    n_shrink = sum(1 for v in post_means if 0.22 <= v < 0.30)
    n_flat = sum(1 for v in post_means if v < 0.22)
    if n_close >= 3:
        verdict = 'CLOSES — calibration is the mechanism'
    elif n_shrink >= 3:
        verdict = 'SHRINKS — calibration necessary, not sufficient'
    elif n_flat >= 3:
        verdict = 'UNCHANGED — reframe to capacity / aggregation'
    else:
        verdict = 'MIXED — see delta_table.md'

    summary = [
        '# Temperature-scaling summary',
        f'\nGenerated {time.strftime("%Y-%m-%d %H:%M")} on {device}.',
        f'\n**Verdict (pre-registered, plan §9): {verdict}**',
        '',
        f'E2 reference: {pre_points[REFERENCE]:.4f}',
        '',
        '| config | T | mean sens@95 pre | mean sens@95 post |',
        '|---|---:|---:|---:|',
    ]
    for cfg in DEEP_CONFIGS:
        if cfg in temperatures:
            summary.append(f"| {cfg} | {temperatures[cfg]['T']:.3f} | "
                           f"{pre_points[cfg]:.4f} | {post_points[cfg]:.4f} |")
    summary += [
        '',
        f'Closes ≥0.30: {n_close}/5 — Shrinks [0.22, 0.30): {n_shrink}/5 — Flat <0.22: {n_flat}/5.',
        '',
        'Caveat: T is fit on val (also used for early-stopping checkpoint selection) '
        'per Guo 2017 §4.1; documented one-line in writeup.',
    ]
    (args.out / 'summary.md').write_text('\n'.join(summary))

    print(f"[tempscaling] DONE in {time.time()-t0:.1f}s. Verdict: {verdict}", flush=True)
    print(f"[tempscaling] outputs: {args.out}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
