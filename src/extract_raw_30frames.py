#!/usr/bin/env python3
"""
Raw Embedding Extraction with 30 Frames Per Video
Extracts more frames per video to enable temporal sampling ablation.
Downstream: subsample to 5/10/15/20/25/30 frames and compare.

Output: raw_video_embeddings_30f.pkl
  dict[subject_id] = list of np.array(N_frames, 768)
  where N_frames is up to 30 per video.

Usage:
  python extract_raw_30frames.py

Estimated time: ~3-4 hours on A40 (600 subjects, 30 frames/video)
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
import random

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(EMBEDDINGS_DIR))
RESULTS_PATH = Path(str(RESULTS_DIR))

FRAMES_PER_VIDEO = 30
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
        print(f"DINOv2 loaded. Embedding dim: {self.model.config.hidden_size}")

    @torch.no_grad()
    def extract_batch_embeddings(self, frames, batch_size=15):
        """Extract in sub-batches to avoid OOM with 30 frames."""
        all_embs = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(embs)
        return np.vstack(all_embs)

    def extract_video_embedding(self, video_path, num_frames=30):
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
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)

    print("Building video path index...")
    subject_videos = get_subject_videos(DATA_PATH)

    train_subjects = pd.read_csv(RESULTS_PATH / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(RESULTS_PATH / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(RESULTS_PATH / "test_subjects.csv")['subject'].tolist()

    # Same 600-subject sample as other experiments
    random.seed(42)
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    train_healthy = [s for s in train_subjects if s in subject_videos and labels_dict.get(s, 0) == 0]
    train_unhealthy = [s for s in train_subjects if s in subject_videos and labels_dict.get(s, 0) == 1]
    sample_train_healthy = random.sample(train_healthy, min(350, len(train_healthy)))
    sample_train_unhealthy = random.sample(train_unhealthy, min(150, len(train_unhealthy)))
    sample_val = random.sample([s for s in val_subjects if s in subject_videos], min(50, len(val_subjects)))
    sample_test = random.sample([s for s in test_subjects if s in subject_videos], min(50, len(test_subjects)))
    all_subjects = list(set(sample_train_healthy + sample_train_unhealthy + sample_val + sample_test))

    print(f"\nSample: {len(all_subjects)} subjects")
    print(f"Extracting {FRAMES_PER_VIDEO} frames per video...")

    extractor = DINOv2Extractor()
    raw_embeddings = {}
    start_time = time.time()
    total_videos = 0
    total_frames = 0

    for subject_id in tqdm(all_subjects, desc="Extracting (30 frames/video)"):
        try:
            videos = subject_videos[subject_id]
            if MAX_VIDEOS_PER_SUBJECT and len(videos) > MAX_VIDEOS_PER_SUBJECT:
                random.seed(42 + subject_id)
                videos = random.sample(videos, MAX_VIDEOS_PER_SUBJECT)

            subject_video_embs = []
            for video_path in videos:
                emb = extractor.extract_video_embedding(video_path, FRAMES_PER_VIDEO)
                if emb is not None:
                    subject_video_embs.append(emb)
                    total_frames += emb.shape[0]

            if subject_video_embs:
                raw_embeddings[subject_id] = subject_video_embs
                total_videos += len(subject_video_embs)
        except Exception as e:
            print(f"Error on subject {subject_id}: {e}")
            continue

    elapsed = time.time() - start_time

    output_file = OUTPUT_PATH / "raw_video_embeddings_30f.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(raw_embeddings, f)

    metadata = {
        'num_subjects': len(raw_embeddings),
        'total_videos': total_videos,
        'total_frames': total_frames,
        'embedding_dim': 768,
        'frames_per_video': FRAMES_PER_VIDEO,
        'max_videos_per_subject': MAX_VIDEOS_PER_SUBJECT,
        'extraction_time_min': round(elapsed / 60, 1),
    }
    with open(OUTPUT_PATH / "raw_embedding_30f_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    videos_per_subject = [len(v) for v in raw_embeddings.values()]
    frames_per_video_actual = [v.shape[0] for vlist in raw_embeddings.values() for v in vlist]

    print(f"\n=== 30-Frame Extraction Complete ===")
    print(f"Subjects: {len(raw_embeddings)}")
    print(f"Videos: {total_videos} ({np.mean(videos_per_subject):.1f} +/- {np.std(videos_per_subject):.1f} per subject)")
    print(f"Frames: {total_frames} ({np.mean(frames_per_video_actual):.1f} +/- {np.std(frames_per_video_actual):.1f} per video)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Saved to: {output_file}")
    est_size_mb = total_frames * 768 * 4 / 1e6
    print(f"File size: ~{est_size_mb:.0f} MB")


if __name__ == "__main__":
    main()
