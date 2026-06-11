#!/usr/bin/env python3
"""
Phase 6: Train/Test Split Design
Subject-level split with stratification for class imbalance
"""

import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(RESULTS_DIR))

# Configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def main():
    # Load labels
    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    
    # Create binary label (healthy = 0, unhealthy = 1)
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)
    
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Total subjects: {len(labels_df)}")
    print(f"Healthy: {(labels_df['unhealthy'] == 0).sum()}")
    print(f"Unhealthy: {(labels_df['unhealthy'] == 1).sum()}")
    print(f"\nConditions: {condition_cols}")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        labels_df,
        test_size=TEST_RATIO,
        stratify=labels_df['unhealthy'],
        random_state=RANDOM_SEED
    )
    
    # Second split: train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['unhealthy'],
        random_state=RANDOM_SEED
    )
    
    # Verify split
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        healthy = (df['unhealthy'] == 0).sum()
        unhealthy = (df['unhealthy'] == 1).sum()
        print(f"\n{name}: {len(df)} subjects ({healthy} healthy, {unhealthy} unhealthy)")
        print(f"  Healthy %: {healthy/len(df)*100:.1f}%")
    
    # Check for rare condition coverage
    print("\n" + "="*60)
    print("PER-CONDITION COVERAGE")
    print("="*60)
    for col in condition_cols:
        train_count = int(train_df[col].sum())
        val_count = int(val_df[col].sum())
        test_count = int(test_df[col].sum())
        total = train_count + val_count + test_count
        print(f"{col}: train={train_count}, val={val_count}, test={test_count} (total={total})")
        if test_count == 0:
            print(f"  ⚠️ WARNING: {col} has no samples in test set!")
    
    # Save splits
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    train_df[['subject']].to_csv(OUTPUT_PATH / "train_subjects.csv", index=False)
    val_df[['subject']].to_csv(OUTPUT_PATH / "val_subjects.csv", index=False)
    test_df[['subject']].to_csv(OUTPUT_PATH / "test_subjects.csv", index=False)
    
    # Full split info with labels
    train_df.to_csv(OUTPUT_PATH / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_PATH / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_PATH / "test_split.csv", index=False)
    
    summary = {
        'total_subjects': int(len(labels_df)),
        'train_subjects': int(len(train_df)),
        'val_subjects': int(len(val_df)),
        'test_subjects': int(len(test_df)),
        'train_healthy': int((train_df['unhealthy'] == 0).sum()),
        'train_unhealthy': int((train_df['unhealthy'] == 1).sum()),
        'val_healthy': int((val_df['unhealthy'] == 0).sum()),
        'val_unhealthy': int((val_df['unhealthy'] == 1).sum()),
        'test_healthy': int((test_df['unhealthy'] == 0).sum()),
        'test_unhealthy': int((test_df['unhealthy'] == 1).sum()),
        'ratios': {'train': TRAIN_RATIO, 'val': VAL_RATIO, 'test': TEST_RATIO},
        'random_seed': RANDOM_SEED
    }
    
    with open(OUTPUT_PATH / "phase6_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*60)
    print("FILES SAVED")
    print("="*60)
    print(f"  {OUTPUT_PATH}/train_subjects.csv")
    print(f"  {OUTPUT_PATH}/val_subjects.csv")
    print(f"  {OUTPUT_PATH}/test_subjects.csv")
    print(f"  {OUTPUT_PATH}/phase6_summary.json")

if __name__ == "__main__":
    main()
