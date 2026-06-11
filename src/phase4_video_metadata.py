#!/usr/bin/env python3
"""
Phase 4: Video Metadata Extraction
Sample-based approach due to dataset size (255K videos)
"""

import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import random

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(RESULTS_DIR))
SAMPLE_SIZE = 1000  # Sample 1000 videos for efficiency

def get_video_metadata(video_path):
    """Extract metadata from a single video."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'filename': video_path.name,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_sec': round(duration, 2)
        }
    except Exception as e:
        return {'filename': video_path.name, 'error': str(e)}

def main():
    random.seed(42)  # Reproducibility
    
    # Collect all video paths
    video_paths = []
    for folder in ['fetal_cardiac_dataset', 'fetal_cardiac_dataset_from_missing_extras', 
                   'fetal_cardiac_dataset_from_xnat']:
        folder_path = DATA_PATH / folder
        if folder_path.exists():
            video_paths.extend(list(folder_path.glob("*.mp4")))
    
    print(f"Total videos found: {len(video_paths)}")
    
    # Random sample
    sample = random.sample(video_paths, min(SAMPLE_SIZE, len(video_paths)))
    
    # Extract metadata
    metadata = []
    for path in tqdm(sample, desc="Extracting metadata"):
        meta = get_video_metadata(path)
        if meta and 'error' not in meta:
            metadata.append(meta)
    
    df = pd.DataFrame(metadata)
    
    # Statistics
    print("\n" + "="*60)
    print("VIDEO METADATA STATISTICS")
    print("="*60)
    print(f"Sample size: {len(df)}")
    print(f"\nDuration (seconds):")
    print(df['duration_sec'].describe())
    print(f"\nFrame count:")
    print(df['frame_count'].describe())
    print(f"\nFPS:")
    print(df['fps'].value_counts())
    print(f"\nResolution:")
    print(f"  Unique widths: {sorted(df['width'].unique())}")
    print(f"  Unique heights: {sorted(df['height'].unique())}")
    
    # Save results
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH / "video_metadata_sample.csv", index=False)
    
    summary = {
        'sample_size': len(df),
        'total_videos': len(video_paths),
        'duration': {
            'min': float(df['duration_sec'].min()),
            'max': float(df['duration_sec'].max()),
            'mean': float(df['duration_sec'].mean()),
            'median': float(df['duration_sec'].median()),
            'std': float(df['duration_sec'].std())
        },
        'frame_count': {
            'min': int(df['frame_count'].min()),
            'max': int(df['frame_count'].max()),
            'mean': float(df['frame_count'].mean()),
            'median': float(df['frame_count'].median())
        },
        'fps': dict(df['fps'].value_counts()),
        'resolution': {
            'widths': sorted([int(x) for x in df['width'].unique()]),
            'heights': sorted([int(x) for x in df['height'].unique()])
        }
    }
    
    with open(OUTPUT_PATH / "phase4_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
