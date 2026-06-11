#!/usr/bin/env python3
"""
Phase 5: Doppler Overlay Detection
Using a chroma-difference heuristic: (B - G > 50) for >0.5% of pixels
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
SAMPLE_SIZE = 500  # Sample 500 videos
FRAMES_PER_VIDEO = 5  # Check 5 frames per video

def has_doppler(frame):
    """
    Detect Doppler overlay using a chroma-difference heuristic.
    If >0.5% of pixels have (B - G > 50), frame has Doppler.
    """
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False, 0.0
    
    B = frame[:, :, 0].astype(np.int16)  # OpenCV uses BGR
    G = frame[:, :, 1].astype(np.int16)
    
    doppler_pixels = (B - G) > 50
    percentage = doppler_pixels.sum() / doppler_pixels.size * 100
    
    return percentage > 0.5, percentage

def check_video_for_doppler(video_path, num_frames=5):
    """Check multiple frames of a video for Doppler overlay."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        doppler_frames = 0
        max_percentage = 0.0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                is_doppler, pct = has_doppler(frame)
                if is_doppler:
                    doppler_frames += 1
                max_percentage = max(max_percentage, pct)
        
        cap.release()
        
        return {
            'filename': video_path.name,
            'doppler_frames': doppler_frames,
            'total_checked': len(frame_indices),
            'has_doppler': doppler_frames > 0,
            'max_doppler_pct': round(max_percentage, 3)
        }
    except Exception as e:
        return {'filename': video_path.name, 'error': str(e)}

def main():
    random.seed(42)
    
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
    
    # Check for Doppler
    results = []
    for path in tqdm(sample, desc="Checking for Doppler"):
        result = check_video_for_doppler(path, FRAMES_PER_VIDEO)
        if result and 'error' not in result:
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Statistics
    doppler_count = df['has_doppler'].sum()
    doppler_pct = doppler_count / len(df) * 100
    
    print("\n" + "="*60)
    print("DOPPLER DETECTION RESULTS")
    print("="*60)
    print(f"Sample size: {len(df)}")
    print(f"Videos with Doppler: {doppler_count} ({doppler_pct:.1f}%)")
    print(f"Videos without Doppler: {len(df) - doppler_count} ({100 - doppler_pct:.1f}%)")
    print(f"\nMax Doppler percentage distribution:")
    print(df['max_doppler_pct'].describe())
    
    # Decision recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    if doppler_pct < 20:
        rec = 'exclude'
        print(f"Doppler is RARE ({doppler_pct:.1f}%). Consider EXCLUDING Doppler videos.")
    elif doppler_pct > 50:
        rec = 'include'
        print(f"Doppler is COMMON ({doppler_pct:.1f}%). Consider KEEPING all videos.")
    else:
        rec = 'ablation'
        print(f"Doppler is MODERATE ({doppler_pct:.1f}%). Run ablation: with/without Doppler.")
    
    # Save results
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH / "doppler_detection_sample.csv", index=False)
    
    summary = {
        'sample_size': int(len(df)),
        'videos_with_doppler': int(doppler_count),
        'videos_without_doppler': int(len(df) - doppler_count),
        'doppler_percentage': round(float(doppler_pct), 2),
        'recommendation': rec,
        'max_doppler_pct_stats': {
            'min': round(float(df['max_doppler_pct'].min()), 3),
            'max': round(float(df['max_doppler_pct'].max()), 3),
            'mean': round(float(df['max_doppler_pct'].mean()), 3),
            'median': round(float(df['max_doppler_pct'].median()), 3)
        }
    }
    
    with open(OUTPUT_PATH / "phase5_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
