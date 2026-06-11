#!/usr/bin/env python3
"""
DINOv2 extraction on Doppler-filtered videos only.

Reads non_doppler_videos.txt (output of doppler_filter_full.py) to build
the subject->video mapping, then extracts mean-pooled DINOv2 embeddings.

Identical to extract_embeddings.py except:
  - Video list comes from a filtered txt file, not directory scan
  - Output saved as subject_embeddings_clean.pkl

Usage:
  python extract_embeddings_clean.py [--resume] [--test]
"""

import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(EMBEDDINGS_DIR))
DOPPLER_FILTER_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "doppler_filter/non_doppler_videos.txt"))

FRAMES_PER_VIDEO = 10
MAX_VIDEOS_PER_SUBJECT = 20


class DINOv2Extractor:
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_dim = self.model.config.hidden_size
        print(f"DINOv2 loaded. Embedding dim: {self.embedding_dim}")

    @torch.no_grad()
    def extract_batch_embeddings(self, frames):
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def extract_video_embedding(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        num_frames = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return None
        return self.extract_batch_embeddings(frames)


def get_subject_videos_from_filter(filter_path):
    """Build subject->video mapping from the non-Doppler video list."""
    if not filter_path.exists():
        raise FileNotFoundError(f"Doppler filter list not found: {filter_path}")

    subject_videos = defaultdict(list)
    skipped = 0
    with open(filter_path) as f:
        for line in f:
            video_path = Path(line.strip())
            if not video_path.exists():
                skipped += 1
                continue
            try:
                subject_id = int(video_path.stem.split('_')[0])
                subject_videos[subject_id].append(video_path)
            except (ValueError, IndexError):
                skipped += 1

    print(f"Loaded {sum(len(v) for v in subject_videos.values())} videos "
          f"for {len(subject_videos)} subjects from filter list")
    if skipped:
        print(f"  Skipped {skipped} entries (missing files or unparseable names)")
    return dict(subject_videos)


def extract_subject_embedding(extractor, video_paths, frames_per_video=10, max_videos=None):
    if max_videos and len(video_paths) > max_videos:
        indices = np.random.choice(len(video_paths), max_videos, replace=False)
        video_paths = [video_paths[i] for i in indices]

    all_frame_embeddings = []
    for video_path in video_paths:
        emb = extractor.extract_video_embedding(video_path, frames_per_video)
        if emb is not None:
            all_frame_embeddings.append(emb)

    if not all_frame_embeddings:
        return None, 0

    # Each element is (frames_per_video, 768); stack -> mean over frames -> mean over videos
    video_means = np.stack([e.mean(axis=0) for e in all_frame_embeddings])  # (N, 768)
    return video_means.mean(axis=0), len(all_frame_embeddings)              # (768,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', action='store_true', help='Run on 10 subjects only')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Checkpoint frequency (subjects)')
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    all_subjects = set(labels_df['subject'].tolist())
    print(f"Subjects in labels CSV: {len(all_subjects)}")

    subject_videos = get_subject_videos_from_filter(DOPPLER_FILTER_PATH)

    subjects_with_videos = [s for s in all_subjects if s in subject_videos]
    print(f"Subjects with clean (non-Doppler) videos: {len(subjects_with_videos)}")

    if args.test:
        subjects_with_videos = subjects_with_videos[:10]
        print("TEST MODE: 10 subjects only")

    # Resume
    embeddings = {}
    checkpoint_path = OUTPUT_PATH / "checkpoint_embeddings_clean.pkl"
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Resumed: {len(embeddings)} embeddings already done")

    subjects_to_process = [s for s in subjects_with_videos if s not in embeddings]
    print(f"Subjects to process: {len(subjects_to_process)}")
    if not subjects_to_process:
        print("Nothing to do.")
        return

    extractor = DINOv2Extractor()
    start_time = time.time()
    videos_total = 0

    for i, subject_id in enumerate(tqdm(subjects_to_process, desc="Extracting")):
        try:
            emb, n_videos = extract_subject_embedding(
                extractor, subject_videos[subject_id],
                frames_per_video=FRAMES_PER_VIDEO,
                max_videos=MAX_VIDEOS_PER_SUBJECT,
            )
            if emb is not None:
                embeddings[subject_id] = emb
                videos_total += n_videos

            if (i + 1) % args.batch_size == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(subjects_to_process) - i - 1) / rate / 60
                print(f"\n  Checkpoint: {len(embeddings)} done, "
                      f"~{remaining:.0f} min remaining")
        except Exception as e:
            print(f"\nError on subject {subject_id}: {e}")

    elapsed = time.time() - start_time

    output_file = OUTPUT_PATH / "subject_embeddings_clean.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        'num_subjects': len(embeddings),
        'embedding_dim': 768,
        'frames_per_video': FRAMES_PER_VIDEO,
        'max_videos_per_subject': MAX_VIDEOS_PER_SUBJECT,
        'aggregation': 'mean',
        'doppler_filtered': True,
        'extraction_time_sec': round(elapsed, 2),
        'videos_processed': videos_total,
    }
    with open(OUTPUT_PATH / "embedding_metadata_clean.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Subjects: {len(embeddings)}")
    print(f"Videos:   {videos_total}")
    print(f"Time:     {elapsed/60:.1f} min")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
