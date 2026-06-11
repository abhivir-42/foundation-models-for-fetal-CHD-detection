#!/usr/bin/env python3
"""
Full-Scale Raw Embedding Extraction
Extracts per-video, per-frame DINOv2 embeddings for ALL subjects.
Saves raw (unaggregated) embeddings for aggregation experiments.

Output: raw_video_embeddings_full.pkl
  dict[subject_id] = list of np.array(N_frames, 768)

Includes checkpoint/resume support for long runs.

Usage:
  python extract_raw_full.py
  python extract_raw_full.py --resume
"""

import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

import torch
from transformers import AutoModel, AutoImageProcessor
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pickle
import pandas as pd
import time
from collections import defaultdict

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(EMBEDDINGS_DIR))
RESULTS_PATH = Path(str(RESULTS_DIR))

FRAMES_PER_VIDEO = 10
MAX_VIDEOS_PER_SUBJECT = 20
CHECKPOINT_INTERVAL = 50


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
        print(f"DINOv2 loaded. Embedding dim: {self.model.config.hidden_size}")

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
        return self.extract_batch_embeddings(frames) if frames else None


def get_subject_videos(data_path):
    video_paths = defaultdict(list)
    folders = ['fetal_cardiac_dataset', 'fetal_cardiac_dataset_from_missing_extras',
               'fetal_cardiac_dataset_from_xnat']
    for folder in folders:
        folder_path = data_path / folder
        if folder_path.exists():
            for video_file in folder_path.glob("*.mp4"):
                try:
                    subject_id = int(video_file.stem.split('_')[0])
                    video_paths[subject_id].append(video_file)
                except (ValueError, IndexError):
                    continue
    return dict(video_paths)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Full-scale raw embedding extraction')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    all_subjects = labels_df['subject'].tolist()

    print("Building video path index...")
    subject_videos = get_subject_videos(DATA_PATH)

    subjects_with_videos = [s for s in all_subjects if s in subject_videos]
    print(f"Subjects with labels and videos: {len(subjects_with_videos)}")

    # Load checkpoint if resuming
    checkpoint_path = OUTPUT_PATH / "checkpoint_raw_full.pkl"
    raw_embeddings = {}

    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint...")
        with open(checkpoint_path, 'rb') as f:
            raw_embeddings = pickle.load(f)
        print(f"Loaded {len(raw_embeddings)} existing subjects")

    subjects_to_process = [s for s in subjects_with_videos if s not in raw_embeddings]
    print(f"Subjects to process: {len(subjects_to_process)}")

    if len(subjects_to_process) == 0:
        print("All subjects already processed!")
        return

    extractor = DINOv2Extractor()
    start_time = time.time()
    total_videos = 0
    total_frames = 0

    for i, subject_id in enumerate(tqdm(subjects_to_process, desc="Extracting raw embeddings")):
        try:
            videos = subject_videos[subject_id]
            if MAX_VIDEOS_PER_SUBJECT and len(videos) > MAX_VIDEOS_PER_SUBJECT:
                np.random.seed(42 + subject_id)
                indices = np.random.choice(len(videos), MAX_VIDEOS_PER_SUBJECT, replace=False)
                videos = [videos[j] for j in indices]

            subject_video_embs = []
            for video_path in videos:
                emb = extractor.extract_video_embedding(video_path, FRAMES_PER_VIDEO)
                if emb is not None:
                    subject_video_embs.append(emb)
                    total_frames += emb.shape[0]

            if subject_video_embs:
                raw_embeddings[subject_id] = subject_video_embs
                total_videos += len(subject_video_embs)

            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(raw_embeddings, f)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(subjects_to_process) - i - 1) / rate / 60
                print(f"\nCheckpoint: {len(raw_embeddings)} subjects. "
                      f"Rate: {rate:.2f}/sec. ~{remaining:.0f} min remaining")

        except Exception as e:
            print(f"Error on subject {subject_id}: {e}")
            continue

    elapsed = time.time() - start_time

    # Save final output
    output_file = OUTPUT_PATH / "raw_video_embeddings_full.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(raw_embeddings, f)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        'num_subjects': len(raw_embeddings),
        'total_videos': total_videos,
        'total_frames': total_frames,
        'embedding_dim': 768,
        'frames_per_video': FRAMES_PER_VIDEO,
        'max_videos_per_subject': MAX_VIDEOS_PER_SUBJECT,
        'extraction_time_min': round(elapsed / 60, 1),
    }
    with open(OUTPUT_PATH / "raw_embedding_full_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    videos_per_subject = [len(v) for v in raw_embeddings.values()]
    frames_per_video_actual = [v.shape[0] for vlist in raw_embeddings.values() for v in vlist]

    print(f"\n=== Full Raw Extraction Complete ===")
    print(f"Subjects: {len(raw_embeddings)}")
    print(f"Videos: {total_videos} ({np.mean(videos_per_subject):.1f} +/- {np.std(videos_per_subject):.1f} per subject)")
    print(f"Frames: {total_frames} ({np.mean(frames_per_video_actual):.1f} +/- {np.std(frames_per_video_actual):.1f} per video)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Saved to: {output_file}")
    est_size_mb = total_frames * 768 * 4 / 1e6
    print(f"Estimated file size: ~{est_size_mb:.0f} MB")


if __name__ == "__main__":
    main()
