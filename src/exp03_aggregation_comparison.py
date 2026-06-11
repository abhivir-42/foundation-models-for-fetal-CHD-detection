#!/usr/bin/env python3
"""
Experiment 03: Aggregation Method Comparison
Tests how frame-level DINOv2 embeddings are combined into subject-level
representations before classification.

Aggregation methods tested:
  Simple (no learnable params):
    1. Mean pooling (baseline)
    2. Max pooling
    3. Mean + Std concatenation (1536-dim)
    4. Mean + Max concatenation (1536-dim)

  Learnable:
    5. Attention pooling (learnable query)
    6. Gated attention pooling
    7. Multi-head attention pooling

  Two-level aggregation (frames->video, videos->subject):
    Each method is tested for both levels independently.

Uses the best classifier config from exp02 (or logistic regression as default).

Usage:
  python exp03_aggregation_comparison.py
  python exp03_aggregation_comparison.py --quick
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
import copy
import warnings
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(str(IFIND_DATA))
RAW_EMBEDDINGS_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "raw_video_embeddings.pkl"))
RESULTS_PATH = Path(str(RESULTS_DIR))
OUTPUT_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "exp03_aggregation_comparison"))

SEED = 42
EMBEDDING_DIM = 768


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load raw embeddings and labels."""
    print("Loading raw embeddings...")
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
    """Filter to subjects that exist in both embeddings and labels."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    valid = []
    for sid in subjects:
        if sid in raw_embeddings and sid in labels_dict:
            valid.append(sid)
    return valid


# ============================================================
# SIMPLE AGGREGATION METHODS (no learnable params)
# ============================================================

def aggregate_mean(video_embeddings):
    """Mean over frames per video, then mean over videos."""
    video_means = np.stack([v.mean(axis=0) for v in video_embeddings])
    return video_means.mean(axis=0)


def aggregate_max(video_embeddings):
    """Max over frames per video, then max over videos."""
    video_maxs = np.stack([v.max(axis=0) for v in video_embeddings])
    return video_maxs.max(axis=0)


def aggregate_mean_std(video_embeddings):
    """Concatenate mean and std across all frames (1536-dim)."""
    all_frames = np.vstack(video_embeddings)
    return np.concatenate([all_frames.mean(axis=0), all_frames.std(axis=0)])


def aggregate_mean_max(video_embeddings):
    """Concatenate mean and max across all frames (1536-dim)."""
    all_frames = np.vstack(video_embeddings)
    return np.concatenate([all_frames.mean(axis=0), all_frames.max(axis=0)])


def aggregate_median(video_embeddings):
    """Median over frames per video, then median over videos."""
    video_medians = np.stack([np.median(v, axis=0) for v in video_embeddings])
    return np.median(video_medians, axis=0)


def aggregate_mean_video_max_subject(video_embeddings):
    """Mean over frames, then max over videos."""
    video_means = np.stack([v.mean(axis=0) for v in video_embeddings])
    return video_means.max(axis=0)


def aggregate_max_video_mean_subject(video_embeddings):
    """Max over frames, then mean over videos."""
    video_maxs = np.stack([v.max(axis=0) for v in video_embeddings])
    return video_maxs.mean(axis=0)


SIMPLE_AGGREGATIONS = OrderedDict([
    ('mean', {
        'fn': aggregate_mean,
        'output_dim': EMBEDDING_DIM,
        'description': 'Mean pool frames, mean pool videos (baseline)',
    }),
    ('max', {
        'fn': aggregate_max,
        'output_dim': EMBEDDING_DIM,
        'description': 'Max pool frames, max pool videos',
    }),
    ('median', {
        'fn': aggregate_median,
        'output_dim': EMBEDDING_DIM,
        'description': 'Median pool frames, median pool videos',
    }),
    ('mean_std', {
        'fn': aggregate_mean_std,
        'output_dim': EMBEDDING_DIM * 2,
        'description': 'Concatenate mean and std of all frames (1536-dim)',
    }),
    ('mean_max', {
        'fn': aggregate_mean_max,
        'output_dim': EMBEDDING_DIM * 2,
        'description': 'Concatenate mean and max of all frames (1536-dim)',
    }),
    ('mean_frame_max_video', {
        'fn': aggregate_mean_video_max_subject,
        'output_dim': EMBEDDING_DIM,
        'description': 'Mean pool frames, then max pool across videos',
    }),
    ('max_frame_mean_video', {
        'fn': aggregate_max_video_mean_subject,
        'output_dim': EMBEDDING_DIM,
        'description': 'Max pool frames, then mean pool across videos',
    }),
])


# ============================================================
# LEARNABLE AGGREGATION MODELS
# ============================================================

class AttentionPool(nn.Module):
    """Attention pooling: learn a query vector that scores each embedding."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, embeddings):
        """embeddings: (N, embed_dim) — all frames from all videos for one subject."""
        attn_scores = self.attention(embeddings)  # (N, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (N, 1)
        pooled = (attn_weights * embeddings).sum(dim=0)  # (embed_dim,)
        logit = self.classifier(pooled)  # (1,)
        return logit.squeeze(), attn_weights.squeeze()


class GatedAttentionPool(nn.Module):
    """Gated attention: element-wise gating before attention scoring."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, embeddings):
        v = self.attn_V(embeddings)  # (N, hidden)
        u = self.attn_U(embeddings)  # (N, hidden)
        attn_scores = self.attn_w(v * u)  # (N, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        pooled = (attn_weights * embeddings).sum(dim=0)
        logit = self.classifier(pooled)
        return logit.squeeze(), attn_weights.squeeze()


class MultiHeadAttentionPool(nn.Module):
    """Multi-head attention pooling with a learnable query token."""

    def __init__(self, embed_dim, n_heads=4, n_layers=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, embeddings):
        """embeddings: (N, embed_dim)"""
        # Prepend learnable query token: query is (1, 1, D), embeddings is (N, D)
        emb_seq = embeddings.unsqueeze(0)  # (1, N, D)
        query = self.query.expand(1, -1, -1)  # (1, 1, D)
        seq = torch.cat([query, emb_seq], dim=1)  # (1, N+1, D)
        out = self.transformer(seq)  # (1, N+1, D)
        pooled = out[0, 0, :]  # query token output
        logit = self.classifier(pooled)
        return logit.squeeze(), None


# ============================================================
# TRAINING FOR LEARNABLE AGGREGATION
# ============================================================

class SubjectDataset(Dataset):
    """Dataset that returns variable-length frame sequences per subject."""

    def __init__(self, subjects, raw_embeddings, labels_dict):
        self.subjects = subjects
        self.raw_embeddings = raw_embeddings
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        video_list = self.raw_embeddings[sid]
        # Flatten all frames from all videos into single sequence
        all_frames = np.vstack(video_list)  # (total_frames, 768)
        label = self.labels_dict[sid]
        return torch.FloatTensor(all_frames), torch.FloatTensor([label])


def collate_subjects(batch):
    """Custom collate: pad sequences to max length in batch."""
    embeddings, labels = zip(*batch)
    lengths = [e.shape[0] for e in embeddings]
    max_len = max(lengths)
    dim = embeddings[0].shape[1]

    padded = torch.zeros(len(embeddings), max_len, dim)
    mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool)

    for i, (emb, length) in enumerate(zip(embeddings, lengths)):
        padded[i, :length] = emb
        mask[i, :length] = True

    labels = torch.cat(labels)
    return padded, labels, mask, lengths


def train_learnable_aggregation(model, train_subjects, val_subjects,
                                 raw_embeddings, labels_dict,
                                 lr=1e-3, weight_decay=1e-4, epochs=100,
                                 patience=15, device='cpu'):
    """Train a learnable aggregation model."""
    model = model.to(device)

    # Compute pos_weight for class balancing
    train_labels = [labels_dict[s] for s in train_subjects]
    n_neg = sum(1 for l in train_labels if l == 0)
    n_pos = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 'val_auroc': []}
    best_val_auroc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training — iterate subject by subject (variable length)
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_subjects)

        for sid in train_subjects:
            video_list = raw_embeddings[sid]
            all_frames = torch.FloatTensor(np.vstack(video_list)).to(device)
            label = torch.FloatTensor([labels_dict[sid]]).to(device)

            optimizer.zero_grad()
            logit, _ = model(all_frames)
            loss = criterion(logit.unsqueeze(0), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_subjects)

        # Evaluation
        model.eval()
        train_metrics = eval_learnable(model, train_subjects, raw_embeddings,
                                        labels_dict, device)
        val_metrics = eval_learnable(model, val_subjects, raw_embeddings,
                                      labels_dict, device)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(0.0)  # Not computing val loss separately
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_auroc'].append(val_metrics['auroc'])

        scheduler.step(val_metrics['auroc'])

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_epoch'] = int(np.argmax(history['val_auroc']))
    history['total_epochs'] = len(history['train_loss'])

    return model, history


def eval_learnable(model, subjects, raw_embeddings, labels_dict, device):
    """Evaluate learnable aggregation model."""
    model.eval()
    y_true, y_proba = [], []

    with torch.no_grad():
        for sid in subjects:
            video_list = raw_embeddings[sid]
            all_frames = torch.FloatTensor(np.vstack(video_list)).to(device)
            logit, _ = model(all_frames)
            prob = torch.sigmoid(logit).cpu().item()
            y_true.append(labels_dict[sid])
            y_proba.append(prob)

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)

    return compute_metrics(y_true, y_pred, y_proba)


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

def run_simple_aggregation(agg_name, agg_config, raw_embeddings, labels_df,
                            train_subjects, val_subjects, test_subjects):
    """Run one simple (non-learnable) aggregation experiment."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    agg_fn = agg_config['fn']
    output_dim = agg_config['output_dim']

    # Aggregate all subjects
    def aggregate_split(subjects):
        X, y, ids = [], [], []
        for sid in subjects:
            if sid in raw_embeddings and sid in labels_dict:
                X.append(agg_fn(raw_embeddings[sid]))
                y.append(labels_dict[sid])
                ids.append(sid)
        return np.array(X), np.array(y), ids

    X_train, y_train, _ = aggregate_split(train_subjects)
    X_val, y_val, _ = aggregate_split(val_subjects)
    X_test, y_test, _ = aggregate_split(test_subjects)

    # Standardise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train logistic regression (fair comparison, same classifier for all)
    clf = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=SEED, solver='lbfgs'
    )
    clf.fit(X_train_s, y_train)

    result = {
        'type': 'simple',
        'aggregation': agg_name,
        'description': agg_config['description'],
        'output_dim': output_dim,
        'classifier': 'LogisticRegression(balanced)',
    }

    for split_name, X, y in [('train', X_train_s, y_train),
                              ('val', X_val_s, y_val),
                              ('test', X_test_s, y_test)]:
        proba = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        result[split_name] = compute_metrics(y, pred, proba)
        result[f'{split_name}_proba'] = proba.tolist()

    return result


def run_learnable_aggregation(name, model_class, model_kwargs,
                               raw_embeddings, labels_df,
                               train_subjects, val_subjects, test_subjects,
                               device='cpu'):
    """Run one learnable aggregation experiment."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()

    train_subs = [s for s in train_subjects if s in raw_embeddings and s in labels_dict]
    val_subs = [s for s in val_subjects if s in raw_embeddings and s in labels_dict]
    test_subs = [s for s in test_subjects if s in raw_embeddings and s in labels_dict]

    model = model_class(**model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())

    model, history = train_learnable_aggregation(
        model, train_subs, val_subs,
        raw_embeddings, labels_dict,
        device=device,
    )

    result = {
        'type': 'learnable',
        'aggregation': name,
        'n_parameters': n_params,
        'history': history,
    }

    for split_name, subjects in [('train', train_subs),
                                  ('val', val_subs),
                                  ('test', test_subs)]:
        metrics = eval_learnable(model, subjects, raw_embeddings, labels_dict, device)
        result[split_name] = metrics

    return result, model


# ============================================================
# PLOTTING
# ============================================================

def generate_plots(all_results, output_path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 03: Aggregation Method Comparison', fontsize=14)

    names = [r['aggregation'] for r in all_results.values()]
    test_aurocs = [r['test']['auroc'] for r in all_results.values()]
    test_auprs = [r['test']['aupr'] for r in all_results.values()]
    test_f1s = [r['test']['f1'] for r in all_results.values()]
    test_recalls = [r['test']['recall'] for r in all_results.values()]
    types = [r['type'] for r in all_results.values()]
    colors = ['#3498db' if t == 'simple' else '#e74c3c' for t in types]

    # AUROC comparison
    ax = axes[0, 0]
    y_pos = range(len(names))
    ax.barh(y_pos, test_aurocs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test AUROC')
    ax.set_title('Test AUROC by Aggregation Method')
    ax.axvline(x=test_aurocs[0], color='gray', linestyle='--', alpha=0.5,
               label='Mean baseline')
    ax.legend(fontsize=8)

    # AUPR comparison
    ax = axes[0, 1]
    ax.barh(y_pos, test_auprs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test AUPR')
    ax.set_title('Test AUPR by Aggregation Method')

    # F1 comparison
    ax = axes[1, 0]
    ax.barh(y_pos, test_f1s, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test F1')
    ax.set_title('Test F1 by Aggregation Method')

    # Multi-metric grouped comparison (top methods)
    ax = axes[1, 1]
    metrics = ['auroc', 'aupr', 'f1', 'recall', 'precision']
    x = np.arange(len(metrics))
    width = 0.8 / len(all_results)

    for i, (name, result) in enumerate(all_results.items()):
        values = [result['test'][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=result['aggregation'],
               alpha=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('All Metrics Comparison')
    ax.set_xticks(x + width * len(all_results) / 2)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path / 'aggregation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Training curves for learnable methods
    learnable = {n: r for n, r in all_results.items() if r.get('history')}
    if learnable:
        fig, axes = plt.subplots(1, len(learnable), figsize=(5 * len(learnable), 4))
        if len(learnable) == 1:
            axes = [axes]
        for ax, (name, result) in zip(axes, learnable.items()):
            h = result['history']
            ax.plot(h['train_auroc'], label='Train AUROC')
            ax.plot(h['val_auroc'], label='Val AUROC')
            ax.axvline(x=h['best_epoch'], color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('AUROC')
            ax.set_title(f'{result["aggregation"]}')
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path / 'learnable_training_curves.png', dpi=200,
                    bbox_inches='tight')
        plt.close()

    print("  Plots saved")


def generate_report(all_results, output_path):
    """Generate markdown report."""
    lines = []
    lines.append("# Experiment 03: Aggregation Method Comparison")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%d %B %Y')}")
    lines.append(f"**Methods tested**: {len(all_results)}")
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append("The baseline (exp01, exp02) uses **mean pooling** to aggregate")
    lines.append("frame-level DINOv2 embeddings into a single subject-level vector.")
    lines.append("This experiment tests whether alternative aggregation strategies")
    lines.append("capture more discriminative information from the raw embeddings.")
    lines.append("")
    lines.append("All simple methods use **Logistic Regression** (balanced) as the")
    lines.append("classifier to isolate the effect of aggregation from classifier choice.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| Method | Type | Dim | Test AUROC | Test AUPR | Test F1 | Test Recall | Test Precision |")
    lines.append("|--------|------|-----|-----------|----------|---------|------------|---------------|")

    ranked = sorted(all_results.items(), key=lambda x: x[1]['test']['auroc'], reverse=True)
    for name, result in ranked:
        t = result['test']
        dim = result.get('output_dim', '-')
        rtype = result['type']
        lines.append(
            f"| {result['aggregation']} | {rtype} | {dim} | "
            f"**{t['auroc']:.4f}** | {t['aupr']:.4f} | {t['f1']:.4f} | "
            f"{t['recall']:.4f} | {t['precision']:.4f} |"
        )
    lines.append("")

    # Overfitting analysis
    lines.append("## Overfitting Analysis")
    lines.append("")
    lines.append("| Method | Train AUROC | Test AUROC | Gap |")
    lines.append("|--------|-----------|-----------|------|")
    for name, result in ranked:
        tr = result['train']['auroc']
        te = result['test']['auroc']
        gap = tr - te
        lines.append(f"| {result['aggregation']} | {tr:.4f} | {te:.4f} | {gap:.4f} |")
    lines.append("")

    # Key findings placeholder
    lines.append("## Key Findings")
    lines.append("")
    best_name, best_result = ranked[0]
    baseline_result = all_results.get('simple_mean', list(all_results.values())[0])
    lines.append(f"1. **Best method**: {best_result['aggregation']} "
                 f"(AUROC={best_result['test']['auroc']:.4f})")
    lines.append(f"2. **Baseline (mean)**: AUROC={baseline_result['test']['auroc']:.4f}")
    improvement = best_result['test']['auroc'] - baseline_result['test']['auroc']
    lines.append(f"3. **Improvement over baseline**: {improvement:+.4f} AUROC")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.json` | Full metrics for all methods |")
    lines.append("| `comparison_table.csv` | Results in tabular format |")
    lines.append("| `aggregation_comparison.png` | 4-panel comparison figure |")
    lines.append("| `learnable_training_curves.png` | Training curves for learnable methods |")
    lines.append("| `experiment_report.md` | This report |")
    lines.append("")

    with open(output_path / "experiment_report.md", 'w') as f:
        f.write('\n'.join(lines))
    print("  Report saved")


# ============================================================
# MAIN
# ============================================================

def run_all(quick=False):
    set_seed(SEED)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    raw_embeddings, labels_df, train_subjects, val_subjects, test_subjects = load_data()

    # Filter to subjects present in raw embeddings
    train_subjects = get_valid_subjects(raw_embeddings, labels_df, train_subjects)
    val_subjects = get_valid_subjects(raw_embeddings, labels_df, val_subjects)
    test_subjects = get_valid_subjects(raw_embeddings, labels_df, test_subjects)

    print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    # Data stats
    total_videos = sum(len(raw_embeddings[s]) for s in train_subjects + val_subjects + test_subjects
                       if s in raw_embeddings)
    total_frames = sum(v.shape[0] for s in raw_embeddings for v in raw_embeddings[s])
    print(f"Total videos: {total_videos}, Total frames: {total_frames}")

    all_results = OrderedDict()
    total_start = time.time()

    # --- Simple aggregation methods ---
    print("\n=== Simple Aggregation Methods ===")
    for agg_name, agg_config in SIMPLE_AGGREGATIONS.items():
        print(f"\n  {agg_name}: {agg_config['description']}")
        start = time.time()
        result = run_simple_aggregation(
            agg_name, agg_config, raw_embeddings, labels_df,
            train_subjects, val_subjects, test_subjects
        )
        result['train_time_sec'] = round(time.time() - start, 2)
        all_results[f'simple_{agg_name}'] = result
        print(f"    Test AUROC={result['test']['auroc']:.4f}  "
              f"AUPR={result['test']['aupr']:.4f}  "
              f"F1={result['test']['f1']:.4f}")

    # --- Learnable aggregation methods ---
    print("\n=== Learnable Aggregation Methods ===")

    learnable_configs = OrderedDict([
        ('attention', {
            'class': AttentionPool,
            'kwargs': {'embed_dim': EMBEDDING_DIM, 'hidden_dim': 128},
            'description': 'Attention pooling with learnable query',
        }),
        ('gated_attention', {
            'class': GatedAttentionPool,
            'kwargs': {'embed_dim': EMBEDDING_DIM, 'hidden_dim': 128},
            'description': 'Gated attention pooling',
        }),
        ('multihead_attention', {
            'class': MultiHeadAttentionPool,
            'kwargs': {'embed_dim': EMBEDDING_DIM, 'n_heads': 4, 'n_layers': 1},
            'description': 'Multi-head attention with learnable query token',
        }),
    ])

    if quick:
        # Only run attention pooling in quick mode
        learnable_configs = OrderedDict(
            [(k, v) for k, v in learnable_configs.items() if k == 'attention']
        )

    for name, config in learnable_configs.items():
        print(f"\n  {name}: {config['description']}")
        start = time.time()
        try:
            result, model = run_learnable_aggregation(
                name, config['class'], config['kwargs'],
                raw_embeddings, labels_df,
                train_subjects, val_subjects, test_subjects,
                device=device,
            )
            result['train_time_sec'] = round(time.time() - start, 2)
            result['description'] = config['description']
            all_results[f'learnable_{name}'] = result
            print(f"    Test AUROC={result['test']['auroc']:.4f}  "
                  f"F1={result['test']['f1']:.4f}  "
                  f"({result['train_time_sec']}s, {result['history']['total_epochs']} epochs)")
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    total_time = time.time() - total_start
    print(f"\nAll experiments complete in {total_time:.1f} seconds")

    # --- Save and generate outputs ---
    # Serialise results (strip non-serialisable items)
    serialisable = {}
    for k, v in all_results.items():
        sv = {key: val for key, val in v.items() if key != 'history' or val is None}
        if v.get('history'):
            sv['history'] = {
                'total_epochs': v['history']['total_epochs'],
                'best_epoch': v['history']['best_epoch'],
            }
        serialisable[k] = sv

    with open(OUTPUT_PATH / "results.json", 'w') as f:
        json.dump(serialisable, f, indent=2, default=str)

    # CSV table
    rows = []
    for name, result in all_results.items():
        row = {
            'method': name,
            'aggregation': result['aggregation'],
            'type': result['type'],
            'description': result.get('description', ''),
            'output_dim': result.get('output_dim', '-'),
            'n_parameters': result.get('n_parameters', '-'),
            'train_time_sec': result.get('train_time_sec', 0),
        }
        for split in ['train', 'val', 'test']:
            for metric in ['auroc', 'aupr', 'f1', 'precision', 'recall', 'accuracy']:
                row[f'{split}_{metric}'] = result[split].get(metric)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('test_auroc', ascending=False)
    df.to_csv(OUTPUT_PATH / "comparison_table.csv", index=False)

    generate_plots(all_results, OUTPUT_PATH)
    generate_report(all_results, OUTPUT_PATH)

    print(f"\nAll outputs saved to {OUTPUT_PATH}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Exp03: Aggregation Comparison')
    parser.add_argument('--quick', action='store_true',
                        help='Run fewer configurations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--embeddings-path', type=str, default=None,
                        help='Override raw embeddings path')
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_PATH = Path(args.output_dir)
    if args.embeddings_path:
        RAW_EMBEDDINGS_PATH = Path(args.embeddings_path)

    run_all(quick=args.quick)
