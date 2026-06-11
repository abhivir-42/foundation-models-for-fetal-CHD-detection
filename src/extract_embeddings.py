#!/usr/bin/env python3
"""
DINOv2 Feature Extraction Pipeline
Extracts and caches embeddings for fetal cardiac dataset

Usage:
  # Test mode (10 subjects)
  python extract_embeddings.py --test
  
  # Full extraction
  python extract_embeddings.py
  
  # Resume from checkpoint
  python extract_embeddings.py --resume
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import pickle
import pandas as pd
import argparse
import time
from collections import defaultdict

# Configuration (paths resolved from environment; see src/paths.py + REPRODUCE.md)
from paths import IFIND_DATA, EMBEDDINGS_DIR, RESULTS_DIR
DATA_PATH = IFIND_DATA
OUTPUT_PATH = EMBEDDINGS_DIR
RESULTS_PATH = RESULTS_DIR

# Extraction parameters
FRAMES_PER_VIDEO = 10
MAX_VIDEOS_PER_SUBJECT = 20  # None = use all


class DINOv2Extractor:
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 on {self.device}...")
        
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"DINOv2 loaded. Embedding dim: {self.embedding_dim}")
    
    @torch.no_grad()
    def extract_frame_embedding(self, frame):
        """Extract embedding for a single frame (RGB numpy array)."""
        inputs = self.processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # CLS token is the first token
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    @torch.no_grad()
    def extract_batch_embeddings(self, frames):
        """Extract embeddings for a batch of frames."""
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def extract_video_embedding(self, video_path, num_frames=10):
        """Extract embeddings for sampled frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Uniform sampling
        num_frames = min(num_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Batch extract
        embeddings = self.extract_batch_embeddings(frames)
        return embeddings


def get_subject_videos(data_path):
    """Build mapping from subject ID to video paths."""
    video_paths = defaultdict(list)
    
    folders = [
        'fetal_cardiac_dataset',
        'fetal_cardiac_dataset_from_missing_extras',
        'fetal_cardiac_dataset_from_xnat'
    ]
    
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


def extract_subject_embedding(extractor, video_paths, frames_per_video=10, 
                               max_videos=None, aggregation='mean'):
    """Extract and aggregate embeddings for a subject."""
    if max_videos and len(video_paths) > max_videos:
        # Random sample if too many videos
        indices = np.random.choice(len(video_paths), max_videos, replace=False)
        video_paths = [video_paths[i] for i in indices]
    
    all_frame_embeddings = []
    videos_processed = 0
    
    for video_path in video_paths:
        video_emb = extractor.extract_video_embedding(video_path, frames_per_video)
        if video_emb is not None:
            all_frame_embeddings.append(video_emb)
            videos_processed += 1
    
    if len(all_frame_embeddings) == 0:
        return None, 0
    
    # Stack: list of (frames_per_video, 768) -> (N_videos, frames_per_video, 768)
    all_embeddings = np.stack(all_frame_embeddings)
    
    if aggregation == 'mean':
        # Mean over frames, then mean over videos
        video_means = all_embeddings.mean(axis=1)  # (N_videos, 768)
        subject_mean = video_means.mean(axis=0)     # (768,)
        return subject_mean, videos_processed
    elif aggregation == 'none':
        return all_embeddings, videos_processed
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 embeddings')
    parser.add_argument('--test', action='store_true', help='Test mode (10 subjects only)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--batch-size', type=int, default=100, help='Save checkpoint every N subjects')
    args = parser.parse_args()
    
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    subjects = labels_df['subject'].tolist()
    
    print(f"Total subjects in CSV: {len(subjects)}")
    
    # Get video paths
    print("Building video path index...")
    subject_videos = get_subject_videos(DATA_PATH)
    print(f"Found videos for {len(subject_videos)} subjects")
    
    # Filter to subjects with videos
    subjects_with_videos = [s for s in subjects if s in subject_videos]
    print(f"Subjects with both labels and videos: {len(subjects_with_videos)}")
    
    if args.test:
        # Test mode: only 10 subjects
        subjects_with_videos = subjects_with_videos[:10]
        print(f"TEST MODE: Processing only {len(subjects_with_videos)} subjects")
    
    # Load checkpoint if resuming
    embeddings = {}
    checkpoint_path = OUTPUT_PATH / "checkpoint_embeddings.pkl"
    
    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} existing embeddings")
    
    # Filter out already processed subjects
    subjects_to_process = [s for s in subjects_with_videos if s not in embeddings]
    print(f"Subjects to process: {len(subjects_to_process)}")
    
    if len(subjects_to_process) == 0:
        print("All subjects already processed!")
        return
    
    # Initialize extractor
    extractor = DINOv2Extractor()
    
    # Extract embeddings
    start_time = time.time()
    videos_total = 0
    
    for i, subject_id in enumerate(tqdm(subjects_to_process, desc="Extracting embeddings")):
        try:
            emb, num_videos = extract_subject_embedding(
                extractor,
                subject_videos[subject_id],
                frames_per_video=FRAMES_PER_VIDEO,
                max_videos=MAX_VIDEOS_PER_SUBJECT,
                aggregation='mean'
            )
            
            if emb is not None:
                embeddings[subject_id] = emb
                videos_total += num_videos
            
            # Save checkpoint periodically
            if (i + 1) % args.batch_size == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(subjects_to_process) - i - 1) / rate / 60
                print(f"\nCheckpoint saved. {len(embeddings)} embeddings. ")
                print(f"Rate: {rate:.2f} subjects/sec. Est. remaining: {remaining:.1f} min")
        
        except Exception as e:
            print(f"\nError processing subject {subject_id}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Save final embeddings
    output_file = OUTPUT_PATH / "subject_embeddings.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    # Save metadata
    metadata = {
        'num_subjects': len(embeddings),
        'embedding_dim': 768,
        'frames_per_video': FRAMES_PER_VIDEO,
        'max_videos_per_subject': MAX_VIDEOS_PER_SUBJECT,
        'aggregation': 'mean',
        'extraction_time_sec': round(elapsed, 2),
        'videos_processed': videos_total
    }
    
    with open(OUTPUT_PATH / "embedding_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Total subjects: {len(embeddings)}")
    print(f"Total videos processed: {videos_total}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Rate: {len(embeddings)/elapsed:.2f} subjects/sec")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
