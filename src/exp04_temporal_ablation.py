#!/usr/bin/env python3
"""
Experiment 04: Temporal Sampling Ablation
Tests how the number of frames sampled per video affects classification.

Strategy: Raw embeddings extracted at 30 frames/video are subsampled to
{5, 10, 15, 20, 25, 30} frames. Each count is evaluated with multiple
aggregation methods and Logistic Regression (balanced).

This answers: does sampling more frames per video (i.e., finer temporal
resolution across the cardiac cycle) improve anomaly detection?

Usage:
  python exp04_temporal_ablation.py
"""


import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
from pathlib import Path
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(str(IFIND_DATA))
RAW_EMBEDDINGS_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "raw_video_embeddings_30f.pkl"))
METADATA_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "raw_embedding_30f_metadata.json"))
RESULTS_PATH = Path(str(RESULTS_DIR))
OUTPUT_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "exp04_temporal_ablation"))

SEED = 42
EMBEDDING_DIM = 768

# Frame counts to test (subsampled from 30-frame extraction)
FRAME_COUNTS = [5, 10, 15, 20, 25, 30]

# Aggregation methods to compare at each frame count
AGGREGATION_METHODS = ['mean', 'median', 'max_frame_mean_video']


def set_seed(seed):
    np.random.seed(seed)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load raw 30-frame embeddings and labels."""
    print("Loading raw 30-frame embeddings...")
    with open(RAW_EMBEDDINGS_PATH, 'rb') as f:
        raw_embeddings = pickle.load(f)

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)

    train_subjects = pd.read_csv(RESULTS_PATH / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(RESULTS_PATH / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(RESULTS_PATH / "test_subjects.csv")['subject'].tolist()

    return raw_embeddings, labels_df, train_subjects, val_subjects, test_subjects


def get_valid_subjects(raw_embeddings, labels_df, subjects):
    """Filter to subjects present in both embeddings and labels."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    return [s for s in subjects if s in raw_embeddings and s in labels_dict]


# ============================================================
# FRAME SUBSAMPLING
# ============================================================

def subsample_video(video_embedding, n_frames):
    """Subsample a single video's embeddings to n_frames.

    video_embedding: np.array of shape (N, 768) where N <= 30
    n_frames: target number of frames

    Returns: np.array of shape (min(n_frames, N), 768)
    """
    actual_frames = video_embedding.shape[0]
    if actual_frames <= n_frames:
        return video_embedding
    indices = np.linspace(0, actual_frames - 1, n_frames, dtype=int)
    return video_embedding[indices]


def subsample_subject(video_embeddings, n_frames):
    """Subsample all videos for a subject to n_frames each."""
    return [subsample_video(v, n_frames) for v in video_embeddings]


# ============================================================
# AGGREGATION METHODS
# ============================================================

def aggregate_mean(video_embeddings):
    """Mean over frames per video, then mean over videos."""
    video_means = np.stack([v.mean(axis=0) for v in video_embeddings])
    return video_means.mean(axis=0)


def aggregate_median(video_embeddings):
    """Median over frames per video, then median over videos."""
    video_medians = np.stack([np.median(v, axis=0) for v in video_embeddings])
    return np.median(video_medians, axis=0)


def aggregate_max_frame_mean_video(video_embeddings):
    """Max over frames per video, then mean over videos."""
    video_maxs = np.stack([v.max(axis=0) for v in video_embeddings])
    return video_maxs.mean(axis=0)


AGG_FUNCTIONS = {
    'mean': aggregate_mean,
    'median': aggregate_median,
    'max_frame_mean_video': aggregate_max_frame_mean_video,
}


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, y_proba):
    """Compute all evaluation metrics."""
    metrics = {
        'auroc': float(roc_auc_score(y_true, y_proba)),
        'aupr': float(average_precision_score(y_true, y_proba)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float((y_true == y_pred).mean()),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
    }
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    return metrics


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_single_config(n_frames, agg_name, raw_embeddings, labels_df,
                      train_subjects, val_subjects, test_subjects):
    """Run one (frame_count, aggregation) configuration."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    agg_fn = AGG_FUNCTIONS[agg_name]

    def aggregate_split(subjects):
        X, y = [], []
        for sid in subjects:
            if sid in raw_embeddings and sid in labels_dict:
                subsampled = subsample_subject(raw_embeddings[sid], n_frames)
                X.append(agg_fn(subsampled))
                y.append(labels_dict[sid])
        return np.array(X), np.array(y)

    X_train, y_train = aggregate_split(train_subjects)
    X_val, y_val = aggregate_split(val_subjects)
    X_test, y_test = aggregate_split(test_subjects)

    # Standardise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=SEED, solver='lbfgs'
    )
    clf.fit(X_train_s, y_train)

    result = {
        'n_frames': n_frames,
        'aggregation': agg_name,
    }

    for split_name, X, y in [('train', X_train_s, y_train),
                              ('val', X_val_s, y_val),
                              ('test', X_test_s, y_test)]:
        proba = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        result[split_name] = compute_metrics(y, pred, proba)

    return result


# ============================================================
# TEMPORAL COVERAGE ANALYSIS
# ============================================================

def compute_temporal_stats(raw_embeddings):
    """Compute statistics about temporal coverage at each frame count."""
    # Collect actual frame counts per video from 30-frame extraction
    frames_per_video = []
    for vid_list in raw_embeddings.values():
        for v in vid_list:
            frames_per_video.append(v.shape[0])

    frames_per_video = np.array(frames_per_video)

    stats = {
        'total_videos': len(frames_per_video),
        'mean_frames_per_video': float(np.mean(frames_per_video)),
        'std_frames_per_video': float(np.std(frames_per_video)),
        'min_frames_per_video': int(np.min(frames_per_video)),
        'max_frames_per_video': int(np.max(frames_per_video)),
    }

    # For each target frame count, how many videos actually get subsampled?
    for n in FRAME_COUNTS:
        n_subsampled = int(np.sum(frames_per_video > n))
        n_unchanged = int(np.sum(frames_per_video <= n))
        stats[f'videos_subsampled_at_{n}'] = n_subsampled
        stats[f'videos_unchanged_at_{n}'] = n_unchanged

    return stats


# ============================================================
# PLOTTING
# ============================================================

def generate_plots(all_results, temporal_stats, output_path):
    """Generate temporal ablation comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment 04: Temporal Sampling Ablation', fontsize=14, y=1.02)

    # Organise results by aggregation method
    by_agg = {}
    for r in all_results:
        agg = r['aggregation']
        if agg not in by_agg:
            by_agg[agg] = {'frames': [], 'auroc': [], 'aupr': [], 'f1': [],
                           'recall': [], 'precision': [],
                           'train_auroc': [], 'val_auroc': []}
        by_agg[agg]['frames'].append(r['n_frames'])
        by_agg[agg]['auroc'].append(r['test']['auroc'])
        by_agg[agg]['aupr'].append(r['test']['aupr'])
        by_agg[agg]['f1'].append(r['test']['f1'])
        by_agg[agg]['recall'].append(r['test']['recall'])
        by_agg[agg]['precision'].append(r['test']['precision'])
        by_agg[agg]['train_auroc'].append(r['train']['auroc'])
        by_agg[agg]['val_auroc'].append(r['val']['auroc'])

    colors = {'mean': '#3498db', 'median': '#2ecc71', 'max_frame_mean_video': '#e74c3c'}
    markers = {'mean': 'o', 'median': 's', 'max_frame_mean_video': '^'}

    # Panel 1: Test AUROC vs frame count
    ax = axes[0, 0]
    for agg, data in by_agg.items():
        ax.plot(data['frames'], data['auroc'], marker=markers[agg],
                color=colors[agg], label=agg, linewidth=2, markersize=8)
    ax.set_xlabel('Frames per Video')
    ax.set_ylabel('Test AUROC')
    ax.set_title('Test AUROC vs Frame Count')
    ax.legend(fontsize=9)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)

    # Panel 2: Test AUPR vs frame count
    ax = axes[0, 1]
    for agg, data in by_agg.items():
        ax.plot(data['frames'], data['aupr'], marker=markers[agg],
                color=colors[agg], label=agg, linewidth=2, markersize=8)
    ax.set_xlabel('Frames per Video')
    ax.set_ylabel('Test AUPR')
    ax.set_title('Test AUPR vs Frame Count')
    ax.legend(fontsize=9)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)

    # Panel 3: Test F1 vs frame count
    ax = axes[0, 2]
    for agg, data in by_agg.items():
        ax.plot(data['frames'], data['f1'], marker=markers[agg],
                color=colors[agg], label=agg, linewidth=2, markersize=8)
    ax.set_xlabel('Frames per Video')
    ax.set_ylabel('Test F1')
    ax.set_title('Test F1 vs Frame Count')
    ax.legend(fontsize=9)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)

    # Panel 4: Train vs Test AUROC (overfitting check)
    ax = axes[1, 0]
    for agg, data in by_agg.items():
        ax.plot(data['frames'], data['train_auroc'], marker=markers[agg],
                color=colors[agg], linestyle='--', alpha=0.5, label=f'{agg} (train)')
        ax.plot(data['frames'], data['auroc'], marker=markers[agg],
                color=colors[agg], label=f'{agg} (test)')
    ax.set_xlabel('Frames per Video')
    ax.set_ylabel('AUROC')
    ax.set_title('Train vs Test AUROC (Overfitting)')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)

    # Panel 5: Precision vs Recall at different frame counts (mean agg only)
    ax = axes[1, 1]
    if 'mean' in by_agg:
        data = by_agg['mean']
        scatter = ax.scatter(data['recall'], data['precision'],
                            c=data['frames'], cmap='viridis', s=100, zorder=3)
        for i, n in enumerate(data['frames']):
            ax.annotate(f'{n}f', (data['recall'][i], data['precision'][i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frames per Video')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off (Mean Agg)')
    ax.grid(True, alpha=0.3)

    # Panel 6: Cardiac cycle coverage estimate
    ax = axes[1, 2]
    # Typical fetal heart rate: 120-160 BPM
    # Typical video: 25 fps, ~3 seconds
    # Temporal spacing = video_duration / (n_frames - 1)
    # Cycles covered = video_duration / cycle_period
    assumed_fps = 25
    assumed_video_frames = 75  # ~3 sec at 25 fps
    video_duration = assumed_video_frames / assumed_fps

    for hr in [120, 140, 160]:
        cycle_period = 60.0 / hr
        # With N uniformly sampled frames, temporal resolution = video_duration / (N-1)
        resolutions = [video_duration / max(n - 1, 1) for n in FRAME_COUNTS]
        # Nyquist: need at least 2 samples per cycle to capture it
        cycles_sampled = [video_duration / cycle_period for _ in FRAME_COUNTS]
        samples_per_cycle = [n / (video_duration / cycle_period) for n in FRAME_COUNTS]
        ax.plot(FRAME_COUNTS, samples_per_cycle,
                marker='o', label=f'{hr} BPM', linewidth=2)

    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Nyquist (2 samples/cycle)')
    ax.set_xlabel('Frames per Video')
    ax.set_ylabel('Samples per Cardiac Cycle')
    ax.set_title(f'Temporal Resolution\n(assuming {assumed_fps} fps, {video_duration:.0f}s video)')
    ax.legend(fontsize=9)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'temporal_ablation.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Summary line plot (clean version for report)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ['auroc', 'aupr', 'f1']
    metric_colors = {'auroc': '#3498db', 'aupr': '#e74c3c', 'f1': '#2ecc71'}

    if 'mean' in by_agg:
        data = by_agg['mean']
        for metric in metrics_to_plot:
            ax.plot(data['frames'], data[metric], marker='o',
                    color=metric_colors[metric], label=metric.upper(),
                    linewidth=2.5, markersize=8)

    ax.set_xlabel('Frames per Video', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Effect of Temporal Sampling on Classification (Mean Pooling)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(FRAME_COUNTS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'temporal_ablation_summary.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("  Plots saved")


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(all_results, temporal_stats, output_path):
    """Generate markdown report."""
    lines = []
    lines.append("# Experiment 04: Temporal Sampling Ablation")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%d %B %Y')}")
    lines.append(f"**Configurations tested**: {len(all_results)}")
    lines.append(f"**Frame counts**: {FRAME_COUNTS}")
    lines.append(f"**Aggregation methods**: {AGGREGATION_METHODS}")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("Previous experiments (exp01-03) used **10 frames per video**.")
    lines.append("At a typical fetal heart rate of 120-160 BPM (cycle period 0.375-0.5s),")
    lines.append("10 frames may under-sample the cardiac cycle, missing diagnostically")
    lines.append("relevant phases (e.g., valve opening/closing, chamber filling).")
    lines.append("")
    lines.append("This experiment extracts 30 frames per video and subsamples to")
    lines.append("{5, 10, 15, 20, 25, 30} frames to measure the effect of temporal")
    lines.append("resolution on classification performance.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Temporal stats
    lines.append("## Extraction Statistics")
    lines.append("")
    lines.append(f"- Total videos: {temporal_stats['total_videos']}")
    lines.append(f"- Frames per video: {temporal_stats['mean_frames_per_video']:.1f} "
                 f"+/- {temporal_stats['std_frames_per_video']:.1f} "
                 f"(range: {temporal_stats['min_frames_per_video']}-"
                 f"{temporal_stats['max_frames_per_video']})")
    lines.append("")

    # Results by aggregation method
    for agg_name in AGGREGATION_METHODS:
        agg_results = [r for r in all_results if r['aggregation'] == agg_name]
        agg_results.sort(key=lambda x: x['n_frames'])

        lines.append(f"## Results: {agg_name} aggregation")
        lines.append("")
        lines.append("| Frames | Test AUROC | Test AUPR | Test F1 | Test Recall | Test Precision | Train AUROC |")
        lines.append("|--------|-----------|----------|---------|------------|---------------|------------|")

        for r in agg_results:
            t = r['test']
            tr = r['train']
            lines.append(
                f"| {r['n_frames']} | **{t['auroc']:.4f}** | {t['aupr']:.4f} | "
                f"{t['f1']:.4f} | {t['recall']:.4f} | {t['precision']:.4f} | "
                f"{tr['auroc']:.4f} |"
            )
        lines.append("")

    # Best config
    best = max(all_results, key=lambda x: x['test']['auroc'])
    baseline_10 = [r for r in all_results
                   if r['n_frames'] == 10 and r['aggregation'] == 'mean']

    lines.append("## Key Findings")
    lines.append("")
    lines.append(f"1. **Best configuration**: {best['n_frames']} frames, "
                 f"{best['aggregation']} (AUROC={best['test']['auroc']:.4f})")
    if baseline_10:
        b = baseline_10[0]
        improvement = best['test']['auroc'] - b['test']['auroc']
        lines.append(f"2. **Baseline (10 frames, mean)**: AUROC={b['test']['auroc']:.4f}")
        lines.append(f"3. **Improvement over baseline**: {improvement:+.4f} AUROC")
    lines.append("")

    # Check monotonic trend
    mean_results = sorted([r for r in all_results if r['aggregation'] == 'mean'],
                          key=lambda x: x['n_frames'])
    aurocs = [r['test']['auroc'] for r in mean_results]
    is_monotonic = all(a <= b for a, b in zip(aurocs, aurocs[1:]))
    if is_monotonic:
        lines.append("4. **Trend**: AUROC increases monotonically with frame count (mean agg)")
    else:
        lines.append("4. **Trend**: AUROC does NOT increase monotonically with frame count")
        peak_idx = np.argmax(aurocs)
        lines.append(f"   Peak at {mean_results[peak_idx]['n_frames']} frames "
                     f"(AUROC={aurocs[peak_idx]:.4f})")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.json` | Full metrics for all configurations |")
    lines.append("| `comparison_table.csv` | Results in tabular format |")
    lines.append("| `temporal_ablation.png` | 6-panel comparison figure |")
    lines.append("| `temporal_ablation_summary.png` | Clean summary plot for report |")
    lines.append("| `experiment_report.md` | This report |")
    lines.append("")

    with open(output_path / "experiment_report.md", 'w') as f:
        f.write('\n'.join(lines))
    print("  Report saved")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    raw_embeddings, labels_df, train_subjects, val_subjects, test_subjects = load_data()

    # Filter to valid subjects
    train_subjects = get_valid_subjects(raw_embeddings, labels_df, train_subjects)
    val_subjects = get_valid_subjects(raw_embeddings, labels_df, val_subjects)
    test_subjects = get_valid_subjects(raw_embeddings, labels_df, test_subjects)

    print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    # Temporal stats
    temporal_stats = compute_temporal_stats(raw_embeddings)
    print(f"Total videos: {temporal_stats['total_videos']}")
    print(f"Frames per video: {temporal_stats['mean_frames_per_video']:.1f} "
          f"+/- {temporal_stats['std_frames_per_video']:.1f}")

    # Run all configurations
    all_results = []
    total_start = time.time()

    for n_frames in FRAME_COUNTS:
        for agg_name in AGGREGATION_METHODS:
            config_name = f"{n_frames}f_{agg_name}"
            print(f"\n  {config_name}...", end=' ')
            start = time.time()

            result = run_single_config(
                n_frames, agg_name, raw_embeddings, labels_df,
                train_subjects, val_subjects, test_subjects
            )
            result['config_name'] = config_name
            result['train_time_sec'] = round(time.time() - start, 2)
            all_results.append(result)

            print(f"AUROC={result['test']['auroc']:.4f}  "
                  f"AUPR={result['test']['aupr']:.4f}  "
                  f"F1={result['test']['f1']:.4f}  "
                  f"({result['train_time_sec']}s)")

    total_time = time.time() - total_start
    print(f"\nAll {len(all_results)} configurations complete in {total_time:.1f} seconds")

    # Save results
    with open(OUTPUT_PATH / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV table
    rows = []
    for r in all_results:
        row = {
            'config': r['config_name'],
            'n_frames': r['n_frames'],
            'aggregation': r['aggregation'],
            'train_time_sec': r['train_time_sec'],
        }
        for split in ['train', 'val', 'test']:
            for metric in ['auroc', 'aupr', 'f1', 'precision', 'recall', 'accuracy']:
                row[f'{split}_{metric}'] = r[split].get(metric)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['aggregation', 'n_frames'])
    df.to_csv(OUTPUT_PATH / "comparison_table.csv", index=False)

    # Generate outputs
    generate_plots(all_results, temporal_stats, OUTPUT_PATH)
    generate_report(all_results, temporal_stats, OUTPUT_PATH)

    # Print summary table
    print("\n=== SUMMARY (Mean Aggregation) ===")
    print(f"{'Frames':<8} {'AUROC':<8} {'AUPR':<8} {'F1':<8} {'Recall':<8} {'Precision':<8}")
    print("-" * 50)
    for r in sorted([r for r in all_results if r['aggregation'] == 'mean'],
                    key=lambda x: x['n_frames']):
        t = r['test']
        print(f"{r['n_frames']:<8} {t['auroc']:<8.4f} {t['aupr']:<8.4f} "
              f"{t['f1']:<8.4f} {t['recall']:<8.4f} {t['precision']:<8.4f}")

    print(f"\nAll outputs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
