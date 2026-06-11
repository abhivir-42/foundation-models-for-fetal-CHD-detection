#!/usr/bin/env python3
"""
Experiment 02: Systematic Classifier Comparison
Frozen DINOv2 embeddings + {Logistic Regression, MLP variants}

This experiment systematically evaluates classifier architectures on top of
frozen DINOv2-base subject-level embeddings (768-dim, mean-pooled).

Experiments:
  A. Logistic Regression baseline (reproduce exp01)
  B. MLP depth ablation: 1, 2, 3, 4, 5 hidden layers
  C. MLP width ablation: 128, 256, 512, 768, 1024
  D. Regularisation ablation: dropout rates, weight decay
  E. Activation function comparison: ReLU, GELU, LeakyReLU
  F. Class weighting strategies: balanced, focal-like, none

Outputs:
  - results.json: All metrics for every configuration
  - comparison_table.csv: Side-by-side comparison
  - training_curves.png: Loss/metric curves per model
  - performance_comparison.png: Bar charts comparing all models
  - embedding_analysis.png: t-SNE/UMAP of embedding space
  - experiment_report.md: Full report with tables and analysis

Usage:
  python exp02_classifier_comparison.py
  python exp02_classifier_comparison.py --quick   # Fewer configs for testing
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
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold

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
DEFAULT_EMBEDDINGS_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "subject_embeddings.pkl"))
EMBEDDINGS_PATH = DEFAULT_EMBEDDINGS_PATH
RESULTS_PATH = Path(str(RESULTS_DIR))
OUTPUT_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "exp02_classifier_comparison"))

SEED = 42
EMBEDDING_DIM = 768


# ============================================================
# DATA LOADING
# ============================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load embeddings, labels, and splits."""
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)

    train_subjects = pd.read_csv(RESULTS_PATH / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(RESULTS_PATH / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(RESULTS_PATH / "test_subjects.csv")['subject'].tolist()

    return embeddings, labels_df, train_subjects, val_subjects, test_subjects


def prepare_split(embeddings, labels_df, subjects):
    """Convert embeddings dict to arrays for a given split."""
    X, y, ids = [], [], []
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()

    for sid in subjects:
        if sid in embeddings and sid in labels_dict:
            X.append(embeddings[sid])
            y.append(labels_dict[sid])
            ids.append(sid)

    return np.array(X), np.array(y), ids


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
        'specificity': float(
            (y_true[y_true == 0] == y_pred[y_true == 0]).mean()
        ) if (y_true == 0).any() else 0.0,
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((y_true == 0).sum()),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

    return metrics


def compute_optimal_threshold(y_true, y_proba):
    """Find threshold that maximises F1 score."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# ============================================================
# MLP MODELS (PyTorch)
# ============================================================

class MLPClassifier(nn.Module):
    """Configurable MLP classifier."""

    def __init__(self, input_dim, hidden_dims, dropout=0.0,
                 activation='relu', use_batchnorm=False):
        super().__init__()
        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'activation': activation,
            'use_batchnorm': use_batchnorm,
        }

        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU,
        }[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_mlp(model, X_train, y_train, X_val, y_val,
              lr=1e-3, weight_decay=0.0, epochs=200, patience=20,
              batch_size=64, class_weight=None, device='cpu'):
    """Train MLP with early stopping. Returns training history."""
    model = model.to(device)

    # Prepare data
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss with optional class weighting
    if class_weight == 'balanced':
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
    elif isinstance(class_weight, (int, float)):
        pos_weight = torch.FloatTensor([class_weight]).to(device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_f1': [], 'val_f1': [],
        'lr': [],
    }

    best_val_auroc = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)
        epoch_loss /= len(X_train)

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_logits = model(X_tr)
            val_logits = model(X_v)

            train_loss = criterion(train_logits, y_tr).item()
            val_loss = criterion(val_logits, y_v).item()

            train_proba = torch.sigmoid(train_logits).cpu().numpy()
            val_proba = torch.sigmoid(val_logits).cpu().numpy()

            train_pred = (train_proba >= 0.5).astype(int)
            val_pred = (val_proba >= 0.5).astype(int)

            try:
                train_auroc = roc_auc_score(y_train, train_proba)
            except ValueError:
                train_auroc = 0.5
            try:
                val_auroc = roc_auc_score(y_val, val_proba)
            except ValueError:
                val_auroc = 0.5

            train_f1 = f1_score(y_train, train_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

        scheduler.step(val_auroc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history['best_epoch'] = int(np.argmax(history['val_auroc']))
    history['total_epochs'] = len(history['train_loss'])

    return model, history


def evaluate_mlp(model, X, y, device='cpu'):
    """Evaluate a trained MLP. Returns metrics and probabilities."""
    model.eval()
    X_t = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        logits = model(X_t)
        proba = torch.sigmoid(logits).cpu().numpy()

    pred = (proba >= 0.5).astype(int)
    metrics = compute_metrics(y, pred, proba)

    # Also compute with optimal threshold
    opt_thresh, opt_f1 = compute_optimal_threshold(y, proba)
    opt_pred = (proba >= opt_thresh).astype(int)
    metrics['optimal_threshold'] = float(opt_thresh)
    metrics['optimal_f1'] = float(opt_f1)
    metrics['optimal_recall'] = float(recall_score(y, opt_pred, zero_division=0))
    metrics['optimal_precision'] = float(precision_score(y, opt_pred, zero_division=0))

    return metrics, proba


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

def get_experiment_configs(quick=False):
    """Define all experiment configurations."""
    configs = OrderedDict()

    # --- A. Logistic Regression Baseline ---
    configs['A1_logreg_balanced'] = {
        'type': 'logreg',
        'params': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000},
        'group': 'A_baseline',
        'description': 'Logistic Regression with balanced class weights (reproduce exp01)',
    }
    configs['A2_logreg_unweighted'] = {
        'type': 'logreg',
        'params': {'C': 1.0, 'class_weight': None, 'max_iter': 1000},
        'group': 'A_baseline',
        'description': 'Logistic Regression without class weighting',
    }

    # --- B. MLP Depth Ablation (fixed width=512) ---
    for n_layers in [1, 2, 3, 4, 5]:
        if quick and n_layers > 3:
            continue
        dims = [512] * n_layers
        name = f'B{n_layers}_depth{n_layers}_w512'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': dims,
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'B_depth_ablation',
            'description': f'MLP with {n_layers} hidden layer(s), width=512',
        }

    # --- C. MLP Width Ablation (fixed depth=2) ---
    for width in [128, 256, 512, 768, 1024]:
        if quick and width in [128, 1024]:
            continue
        name = f'C_depth2_w{width}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [width, width],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'C_width_ablation',
            'description': f'MLP 2-layer, width={width}',
        }

    # --- D. Regularisation Ablation (best depth/width from B/C, use 2x512) ---
    for dropout in [0.0, 0.1, 0.2, 0.3, 0.5]:
        if quick and dropout in [0.1, 0.2]:
            continue
        name = f'D1_dropout{str(dropout).replace(".", "")}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [512, 512],
            'dropout': dropout,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'D_regularisation',
            'description': f'MLP 2x512, dropout={dropout}',
        }

    for wd in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
        if quick and wd in [1e-5, 1e-2]:
            continue
        wd_str = f'{wd:.0e}'.replace('+', '').replace('-0', '-')
        name = f'D2_wd_{wd_str}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [512, 512],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': wd,
            'class_weight': 'balanced',
            'group': 'D_regularisation',
            'description': f'MLP 2x512, weight_decay={wd}',
        }

    # --- E. Activation Function Comparison ---
    for act in ['relu', 'gelu', 'leaky_relu']:
        name = f'E_{act}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [512, 512],
            'dropout': 0.3,
            'activation': act,
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'E_activation',
            'description': f'MLP 2x512 with {act} activation',
        }

    # --- F. Class Weighting Strategies ---
    configs['F1_no_weighting'] = {
        'type': 'mlp',
        'hidden_dims': [512, 512],
        'dropout': 0.3,
        'activation': 'relu',
        'use_batchnorm': True,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'class_weight': None,
        'group': 'F_class_weight',
        'description': 'MLP 2x512, no class weighting',
    }
    configs['F2_balanced'] = {
        'type': 'mlp',
        'hidden_dims': [512, 512],
        'dropout': 0.3,
        'activation': 'relu',
        'use_batchnorm': True,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'class_weight': 'balanced',
        'group': 'F_class_weight',
        'description': 'MLP 2x512, balanced class weighting',
    }
    for w in [2.0, 5.0, 10.0]:
        if quick and w == 2.0:
            continue
        name = f'F3_posweight_{int(w)}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [512, 512],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': w,
            'group': 'F_class_weight',
            'description': f'MLP 2x512, positive class weight={w}x',
        }

    # --- G. Bottleneck / Tapered Architectures ---
    tapered_archs = {
        'G1_768_256': [768, 256],
        'G2_768_512_256': [768, 512, 256],
        'G3_768_512_256_128': [768, 512, 256, 128],
        'G4_512_256_128': [512, 256, 128],
    }
    if quick:
        tapered_archs = {'G1_768_256': [768, 256], 'G2_768_512_256': [768, 512, 256]}

    for name, dims in tapered_archs.items():
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': dims,
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'G_tapered',
            'description': f'Tapered MLP {" -> ".join(map(str, dims))}',
        }

    # --- H. Learning Rate Ablation ---
    for lr_val in [1e-4, 5e-4, 1e-3, 5e-3]:
        if quick and lr_val == 5e-4:
            continue
        lr_str = f'{lr_val:.0e}'.replace('+', '').replace('-0', '-')
        name = f'H_lr_{lr_str}'
        configs[name] = {
            'type': 'mlp',
            'hidden_dims': [512, 512],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batchnorm': True,
            'lr': lr_val,
            'weight_decay': 1e-4,
            'class_weight': 'balanced',
            'group': 'H_learning_rate',
            'description': f'MLP 2x512, lr={lr_val}',
        }

    return configs


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(X_trainval, y_trainval, config, n_folds=5, device='cpu'):
    """Run stratified k-fold CV for more reliable estimates on small data."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        X_tr, X_v = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_v = y_trainval[train_idx], y_trainval[val_idx]

        # Standardise within fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_v = scaler.transform(X_v)

        if config['type'] == 'logreg':
            clf = LogisticRegression(
                random_state=SEED, solver='lbfgs', **config['params']
            )
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_v)[:, 1]
            pred = clf.predict(X_v)
        else:
            model = MLPClassifier(
                input_dim=EMBEDDING_DIM,
                hidden_dims=config['hidden_dims'],
                dropout=config.get('dropout', 0.0),
                activation=config.get('activation', 'relu'),
                use_batchnorm=config.get('use_batchnorm', False),
            )
            model, _ = train_mlp(
                model, X_tr, y_tr, X_v, y_v,
                lr=config.get('lr', 1e-3),
                weight_decay=config.get('weight_decay', 0.0),
                class_weight=config.get('class_weight'),
                device=device,
            )
            m, proba = evaluate_mlp(model, X_v, y_v, device=device)
            pred = (proba >= 0.5).astype(int)

        fold_metrics.append(compute_metrics(y_v, pred, proba))

    # Aggregate
    metric_keys = ['auroc', 'aupr', 'f1', 'precision', 'recall', 'accuracy', 'specificity']
    cv_results = {}
    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics]
        cv_results[f'cv_{key}_mean'] = float(np.mean(values))
        cv_results[f'cv_{key}_std'] = float(np.std(values))
        cv_results[f'cv_{key}_values'] = [float(v) for v in values]

    return cv_results


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_single_experiment(config, X_train, y_train, X_val, y_val,
                          X_test, y_test, scaler, device='cpu'):
    """Run a single experiment configuration. Returns results dict."""
    result = {
        'config': {k: v for k, v in config.items()
                   if k not in ('type',)},
        'type': config['type'],
    }

    start_time = time.time()

    if config['type'] == 'logreg':
        clf = LogisticRegression(random_state=SEED, solver='lbfgs', **config['params'])
        clf.fit(X_train, y_train)

        for split_name, X, y in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test)]:
            proba = clf.predict_proba(X)[:, 1]
            pred = clf.predict(X)
            result[split_name] = compute_metrics(y, pred, proba)
            result[f'{split_name}_proba'] = proba.tolist()

        result['n_parameters'] = int(clf.coef_.size + clf.intercept_.size)
        result['history'] = None

    else:
        model = MLPClassifier(
            input_dim=EMBEDDING_DIM,
            hidden_dims=config['hidden_dims'],
            dropout=config.get('dropout', 0.0),
            activation=config.get('activation', 'relu'),
            use_batchnorm=config.get('use_batchnorm', False),
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        result['n_parameters'] = int(n_params)

        model, history = train_mlp(
            model, X_train, y_train, X_val, y_val,
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 0.0),
            class_weight=config.get('class_weight'),
            device=device,
        )

        for split_name, X, y in [('train', X_train, y_train),
                                  ('val', X_val, y_val),
                                  ('test', X_test, y_test)]:
            metrics, proba = evaluate_mlp(model, X, y, device=device)
            result[split_name] = metrics
            result[f'{split_name}_proba'] = proba.tolist()

        result['history'] = history

    result['train_time_sec'] = round(time.time() - start_time, 2)

    return result


def run_all_experiments(quick=False):
    """Run the full experiment suite."""
    set_seed(SEED)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    embeddings, labels_df, train_subjects, val_subjects, test_subjects = load_data()
    X_train, y_train, train_ids = prepare_split(embeddings, labels_df, train_subjects)
    X_val, y_val, val_ids = prepare_split(embeddings, labels_df, val_subjects)
    X_test, y_test, test_ids = prepare_split(embeddings, labels_df, test_subjects)

    print(f"Train: {len(X_train)} ({y_train.sum()} pos, {(y_train==0).sum()} neg)")
    print(f"Val:   {len(X_val)} ({y_val.sum()} pos, {(y_val==0).sum()} neg)")
    print(f"Test:  {len(X_test)} ({y_test.sum()} pos, {(y_test==0).sum()} neg)")

    # Standardise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Get configs
    configs = get_experiment_configs(quick=quick)
    print(f"\nRunning {len(configs)} experiment configurations...")

    all_results = OrderedDict()
    total_start = time.time()

    for i, (name, config) in enumerate(configs.items()):
        print(f"\n[{i+1}/{len(configs)}] {name}: {config['description']}")
        result = run_single_experiment(
            config, X_train_s, y_train, X_val_s, y_val,
            X_test_s, y_test, scaler, device=device
        )
        all_results[name] = result
        print(f"  Test AUROC={result['test']['auroc']:.4f}  "
              f"AUPR={result['test']['aupr']:.4f}  "
              f"F1={result['test']['f1']:.4f}  "
              f"Recall={result['test']['recall']:.4f}  "
              f"({result['train_time_sec']}s)")

    total_time = time.time() - total_start
    print(f"\nAll experiments complete in {total_time/60:.1f} minutes")

    # --- Cross-validation on the top 5 models ---
    print("\n" + "="*60)
    print("Running 5-fold cross-validation on top models...")
    print("="*60)

    # Rank by test AUROC
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['test']['auroc'], reverse=True)
    top_names = [name for name, _ in ranked[:5]]

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    for name in top_names:
        config = configs[name]
        print(f"\n  CV: {name}")
        cv_results = run_cross_validation(X_trainval, y_trainval, config,
                                          n_folds=5, device=device)
        all_results[name]['cv'] = cv_results
        print(f"    CV AUROC: {cv_results['cv_auroc_mean']:.4f} "
              f"+/- {cv_results['cv_auroc_std']:.4f}")

    # --- Save results ---
    print("\nSaving results...")

    # Save full results (convert numpy types for JSON)
    results_serializable = json.loads(
        json.dumps(all_results, default=lambda o: o.tolist()
                   if hasattr(o, 'tolist') else str(o))
    )
    with open(OUTPUT_PATH / "results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # --- Generate all outputs ---
    generate_comparison_table(all_results, configs, OUTPUT_PATH)
    generate_plots(all_results, configs, OUTPUT_PATH)
    generate_training_curves(all_results, OUTPUT_PATH)
    generate_embedding_analysis(X_train_s, y_train, X_test_s, y_test, OUTPUT_PATH)
    generate_report(all_results, configs, X_train, y_train, X_val, y_val,
                    X_test, y_test, total_time, OUTPUT_PATH)

    print(f"\nAll outputs saved to {OUTPUT_PATH}")
    return all_results


# ============================================================
# OUTPUT GENERATION
# ============================================================

def generate_comparison_table(all_results, configs, output_path):
    """Generate CSV comparison table."""
    rows = []
    for name, result in all_results.items():
        row = {
            'experiment': name,
            'group': configs[name]['group'],
            'description': configs[name]['description'],
            'type': result['type'],
            'n_parameters': result['n_parameters'],
            'train_time_sec': result['train_time_sec'],
        }
        for split in ['train', 'val', 'test']:
            for metric in ['auroc', 'aupr', 'f1', 'precision', 'recall',
                           'accuracy', 'specificity']:
                row[f'{split}_{metric}'] = result[split].get(metric, None)

        if 'cv' in result:
            for metric in ['auroc', 'aupr', 'f1']:
                row[f'cv_{metric}_mean'] = result['cv'].get(f'cv_{metric}_mean')
                row[f'cv_{metric}_std'] = result['cv'].get(f'cv_{metric}_std')

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('test_auroc', ascending=False)
    df.to_csv(output_path / "comparison_table.csv", index=False)

    # Also save a pretty-printed version
    summary_cols = ['experiment', 'description', 'n_parameters',
                    'test_auroc', 'test_aupr', 'test_f1',
                    'test_recall', 'test_precision', 'train_auroc']
    summary = df[summary_cols].copy()
    summary.to_csv(output_path / "comparison_summary.csv", index=False)

    print(f"  Comparison table saved ({len(rows)} experiments)")


def generate_plots(all_results, configs, output_path):
    """Generate performance comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment 02: Classifier Comparison Results', fontsize=14, y=1.02)

    # Collect data by group
    groups = OrderedDict()
    for name, result in all_results.items():
        g = configs[name]['group']
        if g not in groups:
            groups[g] = []
        groups[g].append((name, result))

    group_colors = {
        'A_baseline': '#e74c3c',
        'B_depth_ablation': '#3498db',
        'C_width_ablation': '#2ecc71',
        'D_regularisation': '#9b59b6',
        'E_activation': '#f39c12',
        'F_class_weight': '#1abc9c',
        'G_tapered': '#e67e22',
        'H_learning_rate': '#34495e',
    }

    # --- Plot 1: Test AUROC by group ---
    ax = axes[0, 0]
    all_names, all_aurocs, all_colors = [], [], []
    for g, items in groups.items():
        for name, result in items:
            short_name = name.split('_', 1)[-1] if '_' in name else name
            all_names.append(short_name)
            all_aurocs.append(result['test']['auroc'])
            all_colors.append(group_colors.get(g, '#95a5a6'))

    y_pos = range(len(all_names))
    ax.barh(y_pos, all_aurocs, color=all_colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_names, fontsize=6)
    ax.set_xlabel('Test AUROC')
    ax.set_title('Test AUROC by Configuration')
    ax.axvline(x=0.845, color='red', linestyle='--', alpha=0.5, label='Exp01 baseline')
    ax.legend(fontsize=8)

    # --- Plot 2: AUROC vs Number of Parameters ---
    ax = axes[0, 1]
    for g, items in groups.items():
        params = [r['n_parameters'] for _, r in items]
        aurocs = [r['test']['auroc'] for _, r in items]
        ax.scatter(params, aurocs, color=group_colors.get(g, '#95a5a6'),
                   label=g.split('_', 1)[-1], s=50, alpha=0.7)
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test AUROC')
    ax.set_title('AUROC vs Model Complexity')
    ax.set_xscale('log')
    ax.legend(fontsize=7)

    # --- Plot 3: Train vs Test AUROC (overfitting check) ---
    ax = axes[0, 2]
    for g, items in groups.items():
        train_aurocs = [r['train']['auroc'] for _, r in items]
        test_aurocs = [r['test']['auroc'] for _, r in items]
        ax.scatter(train_aurocs, test_aurocs, color=group_colors.get(g, '#95a5a6'),
                   label=g.split('_', 1)[-1], s=50, alpha=0.7)
    ax.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Train AUROC')
    ax.set_ylabel('Test AUROC')
    ax.set_title('Overfitting Analysis')
    ax.legend(fontsize=7)

    # --- Plot 4: Depth Ablation ---
    ax = axes[1, 0]
    depth_items = groups.get('B_depth_ablation', [])
    if depth_items:
        depths = list(range(1, len(depth_items) + 1))
        aurocs = [r['test']['auroc'] for _, r in depth_items]
        auprs = [r['test']['aupr'] for _, r in depth_items]
        f1s = [r['test']['f1'] for _, r in depth_items]
        ax.plot(depths, aurocs, 'o-', label='AUROC')
        ax.plot(depths, auprs, 's-', label='AUPR')
        ax.plot(depths, f1s, '^-', label='F1')
        ax.set_xlabel('Number of Hidden Layers')
        ax.set_ylabel('Score')
        ax.set_title('Depth Ablation (width=512)')
        ax.legend()
        ax.set_xticks(depths)

    # --- Plot 5: Width Ablation ---
    ax = axes[1, 1]
    width_items = groups.get('C_width_ablation', [])
    if width_items:
        widths = [c['hidden_dims'][0] for c in
                  [configs[n] for n, _ in width_items]]
        aurocs = [r['test']['auroc'] for _, r in width_items]
        auprs = [r['test']['aupr'] for _, r in width_items]
        f1s = [r['test']['f1'] for _, r in width_items]
        ax.plot(widths, aurocs, 'o-', label='AUROC')
        ax.plot(widths, auprs, 's-', label='AUPR')
        ax.plot(widths, f1s, '^-', label='F1')
        ax.set_xlabel('Hidden Layer Width')
        ax.set_ylabel('Score')
        ax.set_title('Width Ablation (depth=2)')
        ax.legend()

    # --- Plot 6: Recall vs Precision trade-off ---
    ax = axes[1, 2]
    for g, items in groups.items():
        recalls = [r['test']['recall'] for _, r in items]
        precisions = [r['test']['precision'] for _, r in items]
        ax.scatter(recalls, precisions, color=group_colors.get(g, '#95a5a6'),
                   label=g.split('_', 1)[-1], s=50, alpha=0.7)
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Precision-Recall Trade-off')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path / 'performance_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Performance comparison plots saved")


def generate_training_curves(all_results, output_path):
    """Generate training curves for MLP models."""
    mlp_results = {n: r for n, r in all_results.items()
                   if r.get('history') is not None}

    if not mlp_results:
        return

    # Select representative models (one per group)
    representative = {}
    for name, result in mlp_results.items():
        group = result['config']['group']
        if group not in representative:
            representative[group] = (name, result)

    n_models = min(len(representative), 8)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 3 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (group, (name, result)) in enumerate(list(representative.items())[:n_models]):
        h = result['history']

        # Loss curves
        ax = axes[idx, 0]
        ax.plot(h['train_loss'], label='Train Loss', alpha=0.7)
        ax.plot(h['val_loss'], label='Val Loss', alpha=0.7)
        ax.axvline(x=h['best_epoch'], color='green', linestyle='--',
                   alpha=0.5, label=f'Best epoch ({h["best_epoch"]})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}: Loss')
        ax.legend(fontsize=7)

        # AUROC curves
        ax = axes[idx, 1]
        ax.plot(h['train_auroc'], label='Train AUROC', alpha=0.7)
        ax.plot(h['val_auroc'], label='Val AUROC', alpha=0.7)
        ax.axvline(x=h['best_epoch'], color='green', linestyle='--',
                   alpha=0.5, label=f'Best epoch ({h["best_epoch"]})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.set_title(f'{name}: AUROC')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  Training curves saved")


def generate_embedding_analysis(X_train, y_train, X_test, y_test, output_path):
    """Generate t-SNE and PCA visualisations of the embedding space."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("  Skipping embedding analysis (sklearn.manifold not available)")
        return

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    split_labels = (['Train'] * len(X_train)) + (['Test'] * len(X_test))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('DINOv2 Embedding Space Analysis', fontsize=14)

    # PCA
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_all)
    explained_var = pca.explained_variance_ratio_

    ax = axes[0]
    for label, color, marker in [(0, '#3498db', 'o'), (1, '#e74c3c', '^')]:
        mask = y_all == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker,
                   alpha=0.5, s=30, label='Healthy' if label == 0 else 'Unhealthy')
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    ax.set_title('PCA of DINOv2 Embeddings')
    ax.legend()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    X_tsne = tsne.fit_transform(X_all)

    ax = axes[1]
    for label, color, marker in [(0, '#3498db', 'o'), (1, '#e74c3c', '^')]:
        mask = y_all == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, marker=marker,
                   alpha=0.5, s=30, label='Healthy' if label == 0 else 'Unhealthy')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE of DINOv2 Embeddings')
    ax.legend()

    # t-SNE coloured by split
    ax = axes[2]
    split_arr = np.array(split_labels)
    for split, color in [('Train', '#95a5a6'), ('Test', '#2ecc71')]:
        mask = split_arr == split
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color,
                   alpha=0.4, s=30, label=split)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE by Train/Test Split')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'embedding_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save PCA explained variance for report
    pca_full = PCA(n_components=min(50, X_all.shape[1]), random_state=SEED)
    pca_full.fit(X_all)
    np.save(output_path / 'pca_explained_variance.npy',
            pca_full.explained_variance_ratio_)

    print("  Embedding analysis plots saved")


def generate_report(all_results, configs, X_train, y_train, X_val, y_val,
                    X_test, y_test, total_time, output_path):
    """Generate comprehensive markdown report."""

    # Rank by test AUROC
    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['test']['auroc'], reverse=True)

    # Build tables
    lines = []
    lines.append("# Experiment 02: Systematic Classifier Comparison")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%d %B %Y')}")
    lines.append(f"**Total runtime**: {total_time/60:.1f} minutes")
    lines.append(f"**Total configurations tested**: {len(all_results)}")
    lines.append(f"**Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Dataset summary
    lines.append("## Dataset")
    lines.append("")
    lines.append("| Split | Total | Healthy | Unhealthy | Unhealthy % |")
    lines.append("|-------|-------|---------|-----------|-------------|")
    for name, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val),
                        ('Test', X_test, y_test)]:
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        pct = 100 * n_pos / len(y)
        lines.append(f"| {name} | {len(y)} | {n_neg} | {n_pos} | {pct:.1f}% |")
    lines.append("")
    lines.append(f"**Embedding dimension**: {EMBEDDING_DIM}")
    lines.append(f"**Embedding model**: DINOv2-base (ViT-B/14), frozen")
    lines.append(f"**Aggregation**: Mean pooling (frames -> video -> subject)")
    lines.append(f"**Frames per video**: 10, **Max videos per subject**: 20")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top results
    lines.append("## Top 10 Results (by Test AUROC)")
    lines.append("")
    lines.append("| Rank | Experiment | Description | Test AUROC | Test AUPR | Test F1 | Test Recall | Test Precision | Train AUROC | #Params |")
    lines.append("|------|-----------|-------------|-----------|----------|---------|------------|---------------|------------|---------|")
    for rank, (name, result) in enumerate(ranked[:10], 1):
        t = result['test']
        tr = result['train']
        lines.append(
            f"| {rank} | {name} | {configs[name]['description']} | "
            f"**{t['auroc']:.4f}** | {t['aupr']:.4f} | {t['f1']:.4f} | "
            f"{t['recall']:.4f} | {t['precision']:.4f} | "
            f"{tr['auroc']:.4f} | {result['n_parameters']:,} |"
        )
    lines.append("")

    # Cross-validation results
    cv_models = [(n, r) for n, r in ranked if 'cv' in r]
    if cv_models:
        lines.append("## Cross-Validation Results (Top 5 Models)")
        lines.append("")
        lines.append("5-fold stratified cross-validation on combined train+val data for more reliable estimates.")
        lines.append("")
        lines.append("| Experiment | CV AUROC | CV AUPR | CV F1 | Test AUROC |")
        lines.append("|-----------|---------|---------|-------|-----------|")
        for name, result in cv_models:
            cv = result['cv']
            lines.append(
                f"| {name} | "
                f"{cv['cv_auroc_mean']:.4f} +/- {cv['cv_auroc_std']:.4f} | "
                f"{cv['cv_aupr_mean']:.4f} +/- {cv['cv_aupr_std']:.4f} | "
                f"{cv['cv_f1_mean']:.4f} +/- {cv['cv_f1_std']:.4f} | "
                f"{result['test']['auroc']:.4f} |"
            )
        lines.append("")

    # Group-level analysis
    lines.append("---")
    lines.append("")
    lines.append("## Analysis by Experiment Group")
    lines.append("")

    groups = OrderedDict()
    for name, result in all_results.items():
        g = configs[name]['group']
        if g not in groups:
            groups[g] = []
        groups[g].append((name, result))

    group_titles = {
        'A_baseline': 'A. Logistic Regression Baselines',
        'B_depth_ablation': 'B. MLP Depth Ablation',
        'C_width_ablation': 'C. MLP Width Ablation',
        'D_regularisation': 'D. Regularisation Ablation',
        'E_activation': 'E. Activation Function Comparison',
        'F_class_weight': 'F. Class Weighting Strategies',
        'G_tapered': 'G. Tapered/Bottleneck Architectures',
        'H_learning_rate': 'H. Learning Rate Ablation',
    }

    for g, items in groups.items():
        lines.append(f"### {group_titles.get(g, g)}")
        lines.append("")
        lines.append("| Experiment | Test AUROC | Test AUPR | Test F1 | Test Recall | Test Precision | #Params |")
        lines.append("|-----------|-----------|----------|---------|------------|---------------|---------|")
        for name, result in items:
            t = result['test']
            lines.append(
                f"| {name} | {t['auroc']:.4f} | {t['aupr']:.4f} | "
                f"{t['f1']:.4f} | {t['recall']:.4f} | {t['precision']:.4f} | "
                f"{result['n_parameters']:,} |"
            )
        lines.append("")

        # Group-specific observations
        best_in_group = max(items, key=lambda x: x[1]['test']['auroc'])
        worst_in_group = min(items, key=lambda x: x[1]['test']['auroc'])
        lines.append(f"**Best**: {best_in_group[0]} (AUROC={best_in_group[1]['test']['auroc']:.4f})")
        lines.append(f"**Worst**: {worst_in_group[0]} (AUROC={worst_in_group[1]['test']['auroc']:.4f})")
        lines.append(f"**Range**: {best_in_group[1]['test']['auroc'] - worst_in_group[1]['test']['auroc']:.4f}")
        lines.append("")

    # Overfitting analysis
    lines.append("---")
    lines.append("")
    lines.append("## Overfitting Analysis")
    lines.append("")
    lines.append("| Experiment | Train AUROC | Test AUROC | Gap | Overfit? |")
    lines.append("|-----------|-----------|-----------|------|---------|")
    for name, result in ranked[:15]:
        tr_auroc = result['train']['auroc']
        te_auroc = result['test']['auroc']
        gap = tr_auroc - te_auroc
        overfit = "Yes" if gap > 0.15 else ("Mild" if gap > 0.05 else "No")
        lines.append(f"| {name} | {tr_auroc:.4f} | {te_auroc:.4f} | {gap:.4f} | {overfit} |")
    lines.append("")

    # Test set confusion matrices for top 3
    lines.append("---")
    lines.append("")
    lines.append("## Confusion Matrices (Top 3 Models)")
    lines.append("")
    for rank, (name, result) in enumerate(ranked[:3], 1):
        t = result['test']
        lines.append(f"### #{rank}: {name}")
        lines.append("")
        lines.append("```")
        lines.append("                  Predicted")
        lines.append("                  Healthy  Unhealthy")
        lines.append(f"Actual Healthy    {t.get('tn', '?'):>5}    {t.get('fp', '?'):>5}")
        lines.append(f"       Unhealthy  {t.get('fn', '?'):>5}    {t.get('tp', '?'):>5}")
        lines.append("```")
        lines.append("")

    # Limitations
    lines.append("---")
    lines.append("")
    lines.append("## Limitations and Caveats")
    lines.append("")
    lines.append("1. **Small test set**: Only 50 subjects in the test set "
                 f"({int(y_test.sum())} unhealthy). Metrics may be unstable; "
                 "a single misclassified subject changes recall by ~12.5%.")
    lines.append("2. **Small validation set**: Only 50 subjects for early stopping "
                 "and hyperparameter selection. May lead to suboptimal model selection.")
    lines.append("3. **600-subject sample**: Results are on a stratified subsample, "
                 "not the full 4,128-subject dataset. Performance may change with more data.")
    lines.append("4. **Pre-aggregated embeddings**: All experiments use mean-pooled "
                 "subject-level embeddings. Information loss during aggregation is not "
                 "evaluated here (see Phase 2: temporal modelling).")
    lines.append("5. **Single random seed**: While seed is fixed for reproducibility, "
                 "results may vary with different random initialisations.")
    lines.append("")

    # Key findings
    lines.append("---")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("*(To be filled after running the experiment)*")
    lines.append("")
    lines.append("1. **Does MLP improve over logistic regression?** Compare A1 vs best MLP")
    lines.append("2. **Effect of depth**: Is deeper better? (Section B)")
    lines.append("3. **Effect of width**: Diminishing returns? (Section C)")
    lines.append("4. **Regularisation**: Does dropout/weight decay help? (Section D)")
    lines.append("5. **Class weighting**: Which strategy is best for this imbalanced dataset? (Section F)")
    lines.append("6. **Overfitting**: Do more complex models overfit more? (Overfitting Analysis)")
    lines.append("7. **Clinical relevance**: Which model best balances recall and precision?")
    lines.append("")

    # Files
    lines.append("---")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.json` | Full metrics for every configuration |")
    lines.append("| `comparison_table.csv` | All results in CSV format |")
    lines.append("| `comparison_summary.csv` | Summary table (key metrics only) |")
    lines.append("| `performance_comparison.png` | 6-panel comparison figure |")
    lines.append("| `training_curves.png` | Loss and AUROC training curves |")
    lines.append("| `embedding_analysis.png` | t-SNE and PCA visualisations |")
    lines.append("| `pca_explained_variance.npy` | PCA explained variance ratios |")
    lines.append("| `experiment_report.md` | This report |")
    lines.append("")

    # Reproducibility
    lines.append("---")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# In the project environment, from the repository root:")
    lines.append("python src/exp02_classifier_comparison.py")
    lines.append("```")
    lines.append("")

    report = '\n'.join(lines)
    with open(output_path / "experiment_report.md", 'w') as f:
        f.write(report)

    print("  Experiment report saved")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Exp02: Systematic Classifier Comparison'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run fewer configurations for testing')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--embeddings-path', type=str, default=None,
                        help='Override embeddings pickle path')
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_PATH = Path(args.output_dir)
    if args.embeddings_path:
        EMBEDDINGS_PATH = Path(args.embeddings_path)

    run_all_experiments(quick=args.quick)
