"""
Full-scale Doppler detection for all ~255K fetal cardiac videos.

Detects colour Doppler overlay using a chroma-difference heuristic:
  - For each frame, count pixels where (Blue - Green) > 50
  - If > 0.5% of pixels match, frame has Doppler
  - If ANY checked frame has Doppler, video is flagged

Checks 5 uniformly-sampled frames per video (same as phase5 sample).

Outputs:
  - doppler_results_full.csv: per-video results (filename, has_doppler, max_doppler_pct)
  - doppler_summary.json: aggregate statistics
  - non_doppler_videos.txt: clean video list for downstream extraction
  - doppler_videos.txt: excluded video list

Usage:
  python doppler_filter_full.py [--output-dir DIR] [--resume]
"""


import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
DATA_ROOT = str(IFIND_DATA)
DATASET_DIRS = [
    os.path.join(DATA_ROOT, "fetal_cardiac_dataset"),
    os.path.join(DATA_ROOT, "fetal_cardiac_dataset_from_missing_extras"),
    os.path.join(DATA_ROOT, "fetal_cardiac_dataset_from_xnat"),
]

DEFAULT_OUTPUT = str(_Path_scrub(RESULTS_DIR) / "doppler_filter")

FRAMES_TO_CHECK = 5
DOPPLER_THRESHOLD = 50    # Blue - Green > threshold
PIXEL_PCT_THRESHOLD = 0.5  # percentage of pixels that must exceed


def collect_all_videos():
    """Collect all .mp4 files across all dataset directories."""
    videos = []
    for ddir in DATASET_DIRS:
        if not os.path.isdir(ddir):
            print(f"WARNING: Directory not found: {ddir}")
            continue
        for f in os.listdir(ddir):
            if f.endswith(".mp4"):
                videos.append((ddir, f))
    return videos


def check_doppler_frame(frame):
    """Check if a single frame has Doppler overlay (chroma-difference heuristic)."""
    if frame is None or frame.size == 0:
        return False, 0.0

    # OpenCV loads as BGR
    blue = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)

    doppler_pixels = np.sum((blue - green) > DOPPLER_THRESHOLD)
    total_pixels = frame.shape[0] * frame.shape[1]
    pct = (doppler_pixels / total_pixels) * 100

    return pct > PIXEL_PCT_THRESHOLD, pct


def check_video_doppler(video_path):
    """Check if a video has Doppler overlay by sampling frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, "failed_to_open"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, None, "no_frames"

    # Sample frames uniformly
    if total_frames <= FRAMES_TO_CHECK:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, FRAMES_TO_CHECK, dtype=int)

    doppler_count = 0
    max_pct = 0.0
    frames_checked = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames_checked += 1
        has_doppler, pct = check_doppler_frame(frame)
        if has_doppler:
            doppler_count += 1
        max_pct = max(max_pct, pct)

    cap.release()

    if frames_checked == 0:
        return None, None, "no_readable_frames"

    return doppler_count > 0, max_pct, None


def main():
    parser = argparse.ArgumentParser(description="Full-scale Doppler detection")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_csv = os.path.join(args.output_dir, "doppler_results_full.csv")
    summary_json = os.path.join(args.output_dir, "doppler_summary.json")
    clean_list = os.path.join(args.output_dir, "non_doppler_videos.txt")
    doppler_list = os.path.join(args.output_dir, "doppler_videos.txt")

    # Collect all videos
    print("Collecting video files...")
    all_videos = collect_all_videos()
    print(f"Found {len(all_videos)} videos across {len(DATASET_DIRS)} directories")

    # Load existing results if resuming
    processed = {}
    if args.resume and os.path.exists(results_csv):
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed[row["filename"]] = row
        print(f"Resuming: {len(processed)} videos already processed")

    # Process videos
    results = list(processed.values()) if processed else []
    errors = []
    start_time = time.time()

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "directory", "has_doppler", "max_doppler_pct", "error"
        ])
        writer.writeheader()

        # Write existing results first
        for row in results:
            writer.writerow(row)

        for i, (ddir, fname) in enumerate(tqdm(all_videos, desc="Doppler detection")):
            if fname in processed:
                continue

            video_path = os.path.join(ddir, fname)
            has_doppler, max_pct, error = check_video_doppler(video_path)

            dirname = os.path.basename(ddir)
            row = {
                "filename": fname,
                "directory": dirname,
                "has_doppler": has_doppler if error is None else "",
                "max_doppler_pct": f"{max_pct:.3f}" if max_pct is not None else "",
                "error": error or "",
            }
            writer.writerow(row)
            results.append(row)

            if error:
                errors.append((fname, error))

            # Flush periodically
            if (i + 1) % 1000 == 0:
                f.flush()
                elapsed = time.time() - start_time
                rate = (i + 1 - len(processed)) / elapsed if elapsed > 0 else 0
                remaining = (len(all_videos) - i - 1) / rate if rate > 0 else 0
                print(f"\n  [{i+1}/{len(all_videos)}] "
                      f"{rate:.1f} videos/sec, ~{remaining/60:.0f} min remaining")

    # Compute summary
    doppler_count = sum(1 for r in results if r.get("has_doppler") == "True" or r.get("has_doppler") is True)
    clean_count = sum(1 for r in results if r.get("has_doppler") == "False" or r.get("has_doppler") is False)
    error_count = len(errors)
    total = len(results)

    summary = {
        "total_videos": total,
        "videos_with_doppler": doppler_count,
        "videos_without_doppler": clean_count,
        "videos_with_errors": error_count,
        "doppler_percentage": round(doppler_count / total * 100, 2) if total > 0 else 0,
        "clean_percentage": round(clean_count / total * 100, 2) if total > 0 else 0,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Write clean and doppler video lists (full paths)
    with open(clean_list, "w") as f:
        for r in results:
            if r.get("has_doppler") == "False" or r.get("has_doppler") is False:
                fname = r["filename"]
                # Find the directory for this file
                for ddir in DATASET_DIRS:
                    full_path = os.path.join(ddir, fname)
                    if os.path.exists(full_path):
                        f.write(full_path + "\n")
                        break

    with open(doppler_list, "w") as f:
        for r in results:
            if r.get("has_doppler") == "True" or r.get("has_doppler") is True:
                fname = r["filename"]
                for ddir in DATASET_DIRS:
                    full_path = os.path.join(ddir, fname)
                    if os.path.exists(full_path):
                        f.write(full_path + "\n")
                        break

    # Print summary
    print(f"\n{'='*60}")
    print(f"DOPPLER DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos:      {total}")
    print(f"With Doppler:      {doppler_count} ({summary['doppler_percentage']}%)")
    print(f"Without Doppler:   {clean_count} ({summary['clean_percentage']}%)")
    print(f"Errors:            {error_count}")
    print(f"Elapsed:           {summary['elapsed_seconds']:.0f}s")
    print(f"\nOutputs:")
    print(f"  Results CSV:     {results_csv}")
    print(f"  Summary JSON:    {summary_json}")
    print(f"  Clean list:      {clean_list}")
    print(f"  Doppler list:    {doppler_list}")

    if errors:
        print(f"\nFirst 10 errors:")
        for fname, err in errors[:10]:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    main()
