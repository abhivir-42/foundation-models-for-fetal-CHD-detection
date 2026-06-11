#!/usr/bin/env python3
"""
Baseline Experiment: Frozen DINOv2 + Logistic Regression
Binary classification: Healthy vs Unhealthy

Usage:
  python train_baseline.py                    # Full training
  python train_baseline.py --quick            # Quick test with available embeddings
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
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = Path(str(IFIND_DATA))
EMBEDDINGS_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "subject_embeddings.pkl"))
RESULTS_PATH = Path(str(RESULTS_DIR))
OUTPUT_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "exp01_frozen_baseline"))


def load_data(embeddings_path, results_path):
    """Load embeddings and splits."""
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels
    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)
    
    # Load splits
    train_subjects = pd.read_csv(results_path / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(results_path / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(results_path / "test_subjects.csv")['subject'].tolist()
    
    return embeddings, labels_df, train_subjects, val_subjects, test_subjects


def prepare_data(embeddings, labels_df, subjects):
    """Convert embeddings dict to X, y arrays."""
    X, y, valid_subjects = [], [], []
    
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    
    for subject_id in subjects:
        if subject_id in embeddings and subject_id in labels_dict:
            X.append(embeddings[subject_id])
            y.append(labels_dict[subject_id])
            valid_subjects.append(subject_id)
    
    return np.array(X), np.array(y), valid_subjects


def evaluate(y_true, y_pred, y_proba):
    """Calculate evaluation metrics."""
    return {
        'auroc': float(roc_auc_score(y_true, y_proba)),
        'aupr': float(average_precision_score(y_true, y_proba)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float((y_true == y_pred).mean()),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((y_true == 0).sum())
    }


def plot_results(y_test, y_test_proba, y_test_pred, output_path):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy', 'Unhealthy'],
                yticklabels=['Healthy', 'Unhealthy'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auroc = roc_auc_score(y_test, y_test_proba)
    axes[1].plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    aupr = average_precision_score(y_test, y_test_proba)
    axes[2].plot(recall, precision, label=f'AUPR = {aupr:.3f}')
    # Baseline (random classifier)
    baseline = y_test.sum() / len(y_test)
    axes[2].axhline(y=baseline, color='k', linestyle='--', label=f'Baseline = {baseline:.3f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path / 'evaluation_plots.png'}")


def train_and_evaluate(quick_mode=False):
    """Train baseline and evaluate."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    embeddings, labels_df, train_subjects, val_subjects, test_subjects = load_data(
        EMBEDDINGS_PATH, RESULTS_PATH
    )
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    X_train, y_train, train_ids = prepare_data(embeddings, labels_df, train_subjects)
    X_val, y_val, val_ids = prepare_data(embeddings, labels_df, val_subjects)
    X_test, y_test, test_ids = prepare_data(embeddings, labels_df, test_subjects)
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} unhealthy, {(y_train==0).sum()} healthy)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} unhealthy, {(y_val==0).sum()} healthy)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} unhealthy, {(y_test==0).sum()} healthy)")
    
    if len(X_train) < 10:
        print("\nNot enough training samples. Extract more embeddings first.")
        print("Run the extraction script (see src/extract_embeddings.py and REPRODUCE.md).")
        return None
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression with class weighting
    print("\nTraining Logistic Regression (class_weight='balanced')...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    results = {}
    for name, X, y in [
        ('train', X_train_scaled, y_train),
        ('val', X_val_scaled, y_val),
        ('test', X_test_scaled, y_test)
    ]:
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]
        results[name] = evaluate(y, y_pred, y_proba)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS: Frozen DINOv2 + Logistic Regression")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for metric, value in results[split].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Test set detailed report
    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("\n" + "="*60)
    print("Test Set Classification Report:")
    print("="*60)
    print(classification_report(y_test, y_test_pred, target_names=['Healthy', 'Unhealthy']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test):")
    print(f"                  Predicted")
    print(f"                  Healthy  Unhealthy")
    print(f"Actual Healthy    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Unhealthy  {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Save results
    with open(OUTPUT_PATH / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    with open(OUTPUT_PATH / "model.pkl", 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    
    # Generate plots
    plot_results(y_test, y_test_proba, y_test_pred, OUTPUT_PATH)
    
    # Save test predictions for analysis
    test_predictions = pd.DataFrame({
        'subject': test_ids,
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    })
    test_predictions.to_csv(OUTPUT_PATH / "test_predictions.csv", index=False)
    
    print(f"\n✅ Results saved to {OUTPUT_PATH}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train baseline classifier')
    parser.add_argument('--quick', action='store_true', help='Quick mode with available embeddings')
    args = parser.parse_args()
    
    results = train_and_evaluate(quick_mode=args.quick)
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Test AUROC: {results['test']['auroc']:.4f}")
        print(f"Test AUPR:  {results['test']['aupr']:.4f}")
        print(f"Test F1:    {results['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
