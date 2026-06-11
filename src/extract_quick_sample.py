#!/usr/bin/env python3
"""
Quick Sample Extraction for Faster Baseline Testing
Extracts ~600 stratified subjects instead of all 4,128
Estimated time: ~2 hours instead of ~22 hours
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

# Configuration
DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(EMBEDDINGS_DIR))
RESULTS_PATH = Path(str(RESULTS_DIR))

FRAMES_PER_VIDEO = 10
MAX_VIDEOS_PER_SUBJECT = 20
SAMPLE_SIZE = 600  # Quick sample size


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
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [col for col in labels_df.columns if col != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)
    
    # Get video paths
    print("Building video path index...")
    subject_videos = get_subject_videos(DATA_PATH)
    
    # Load splits
    train_subjects = pd.read_csv(RESULTS_PATH / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(RESULTS_PATH / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(RESULTS_PATH / "test_subjects.csv")['subject'].tolist()
    
    # Create stratified sample
    random.seed(42)
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    
    # From train: get stratified sample
    train_healthy = [s for s in train_subjects if s in subject_videos and labels_dict.get(s, 0) == 0]
    train_unhealthy = [s for s in train_subjects if s in subject_videos and labels_dict.get(s, 0) == 1]
    
    sample_train_healthy = random.sample(train_healthy, min(350, len(train_healthy)))
    sample_train_unhealthy = random.sample(train_unhealthy, min(150, len(train_unhealthy)))
    
    # From val/test: sample some for evaluation
    sample_val = random.sample([s for s in val_subjects if s in subject_videos], min(50, len(val_subjects)))
    sample_test = random.sample([s for s in test_subjects if s in subject_videos], min(50, len(test_subjects)))
    
    all_subjects = list(set(sample_train_healthy + sample_train_unhealthy + sample_val + sample_test))
    
    print(f"\nQuick sample composition:")
    print(f"  Train healthy: {len(sample_train_healthy)}")
    print(f"  Train unhealthy: {len(sample_train_unhealthy)}")
    print(f"  Val sample: {len(sample_val)}")
    print(f"  Test sample: {len(sample_test)}")
    print(f"  Total unique: {len(all_subjects)}")
    
    # Initialize extractor
    extractor = DINOv2Extractor()
    
    # Extract embeddings
    embeddings = {}
    start_time = time.time()
    
    for subject_id in tqdm(all_subjects, desc="Extracting embeddings"):
        try:
            videos = subject_videos[subject_id]
            if MAX_VIDEOS_PER_SUBJECT and len(videos) > MAX_VIDEOS_PER_SUBJECT:
                videos = random.sample(videos, MAX_VIDEOS_PER_SUBJECT)
            
            all_frame_embs = []
            for video_path in videos:
                emb = extractor.extract_video_embedding(video_path, FRAMES_PER_VIDEO)
                if emb is not None:
                    all_frame_embs.append(emb)
            
            if all_frame_embs:
                # Mean over frames, then mean over videos
                video_means = np.stack([e.mean(axis=0) for e in all_frame_embs])
                embeddings[subject_id] = video_means.mean(axis=0)
        except Exception as e:
            print(f"Error on subject {subject_id}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Save
    with open(OUTPUT_PATH / "subject_embeddings.pkl", 'wb') as f:
        pickle.dump(embeddings, f)
    
    metadata = {
        'num_subjects': len(embeddings),
        'embedding_dim': 768,
        'frames_per_video': FRAMES_PER_VIDEO,
        'max_videos_per_subject': MAX_VIDEOS_PER_SUBJECT,
        'aggregation': 'mean',
        'sample_type': 'stratified_quick',
        'extraction_time_min': round(elapsed/60, 1)
    }
    with open(OUTPUT_PATH / "embedding_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Quick Extraction Complete ===")
    print(f"Subjects extracted: {len(embeddings)}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Saved to: {OUTPUT_PATH / 'subject_embeddings.pkl'}")


if __name__ == "__main__":
    main()
