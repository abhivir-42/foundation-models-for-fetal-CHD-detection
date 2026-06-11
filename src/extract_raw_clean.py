#!/usr/bin/env python3
"""
Raw frame-level DINOv2 extraction.

Default behaviour: reads non_doppler_videos.txt (Doppler-filtered) and writes
raw_video_embeddings_clean.pkl.

For scaling runs: pass --include-doppler to bypass the Doppler filter and use
ALL videos in the source tree. Combine with --num-frames /
--max-videos-per-subject / --output-suffix / --gpu-id / --subject-shard to
parameterise per-increment, multi-GPU shard runs.

Output: dict[subject_id] = list of np.array(N_frames, 768)

Usage:
  python extract_raw_clean.py [--resume]
  python extract_raw_clean.py --num-frames 30 --max-videos-per-subject 40 \
      --output-suffix __nf30__nc40__shard0 --gpu-id 0 --subject-shard 0/4 \
      --include-doppler
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
import os
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


def get_subject_videos_unfiltered(data_path):
    """Scan DATA_PATH recursively for all .mp4 files, group by subject_id."""
    subject_videos = defaultdict(list)
    skipped = 0
    for video_path in data_path.rglob("*.mp4"):
        try:
            subject_id = int(video_path.stem.split('_')[0])
            subject_videos[subject_id].append(video_path)
        except (ValueError, IndexError):
            skipped += 1
    print(f"Loaded {sum(len(v) for v in subject_videos.values())} videos "
          f"for {len(subject_videos)} subjects (unfiltered, Doppler IN)")
    if skipped:
        print(f"  Skipped {skipped} entries (unparseable names)")
    return dict(subject_videos)


def parse_shard(shard_str):
    """Parse 'k/N' format. Returns (k, N) with 0 <= k < N."""
    try:
        k_str, n_str = shard_str.split('/')
        k, n = int(k_str), int(n_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"--subject-shard must be 'k/N' (e.g. 0/4); got {shard_str!r}")
    if n < 1 or k < 0 or k >= n:
        raise argparse.ArgumentTypeError(
            f"--subject-shard 'k/N' requires 0 <= k < N; got k={k} N={n}")
    return k, n


def main():
    parser = argparse.ArgumentParser(description='Raw frame-level DINOv2 extraction')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--num-frames', type=int, default=10,
                        help='Frames per video (uniform linspace; default 10)')
    parser.add_argument('--max-videos-per-subject', type=int, default=20,
                        help='Cap on clips per subject (default 20)')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix appended to output filename (e.g. __nf30__nc40)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='CUDA device index (default 0)')
    parser.add_argument('--subject-shard', type=parse_shard, default=(0, 1),
                        help="'k/N' format; process every Nth subject starting "
                             "from offset k (default 0/1 = all)")
    parser.add_argument('--include-doppler', action='store_true',
                        help='Bypass the non_doppler_videos.txt filter and '
                             'embed ALL videos (required for scaling runs)')
    parser.add_argument('--model-name', type=str, default='facebook/dinov2-base',
                        help='HuggingFace DINOv2 checkpoint. Default '
                             'facebook/dinov2-base (768-dim, canonical Tier-1). '
                             'Use facebook/dinov2-large (1024-dim) for the '
                             'ViT-L/14 size-control run; always pair a non-default '
                             'model with a distinct --output-suffix.')
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    shard_k, shard_n = args.subject_shard
    if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(args.gpu_id)
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    all_subjects = set(labels_df['subject'].tolist())
    print(f"Subjects in labels CSV: {len(all_subjects)}")

    if args.include_doppler:
        subject_videos = get_subject_videos_unfiltered(DATA_PATH)
        filter_label = False
    else:
        subject_videos = get_subject_videos_from_filter(DOPPLER_FILTER_PATH)
        filter_label = True
    subjects_with_videos = sorted(s for s in all_subjects if s in subject_videos)
    print(f"Subjects with videos available: {len(subjects_with_videos)}")

    if shard_n > 1:
        subjects_with_videos = [s for i, s in enumerate(subjects_with_videos)
                                if i % shard_n == shard_k]
        print(f"Shard {shard_k}/{shard_n}: {len(subjects_with_videos)} subjects this worker")

    suffix = args.output_suffix
    checkpoint_path = OUTPUT_PATH / f"checkpoint_raw_clean{suffix}.pkl"
    raw_embeddings = {}

    if args.resume and checkpoint_path.exists():
        print(f"Loading checkpoint...")
        with open(checkpoint_path, 'rb') as f:
            raw_embeddings = pickle.load(f)
        print(f"Loaded {len(raw_embeddings)} existing subjects")

    subjects_to_process = [s for s in subjects_with_videos if s not in raw_embeddings]
    print(f"Subjects to process: {len(subjects_to_process)}")

    if not subjects_to_process:
        print("All subjects already processed!")
        return

    extractor = DINOv2Extractor(model_name=args.model_name, device=device)
    start_time = time.time()
    total_videos = 0
    total_frames = 0

    for i, subject_id in enumerate(tqdm(subjects_to_process, desc="Extracting raw embeddings")):
        try:
            videos = subject_videos[subject_id]
            if args.max_videos_per_subject and len(videos) > args.max_videos_per_subject:
                np.random.seed(42 + subject_id)
                indices = np.random.choice(len(videos), args.max_videos_per_subject, replace=False)
                videos = [videos[j] for j in indices]

            subject_video_embs = []
            for video_path in videos:
                emb = extractor.extract_video_embedding(video_path, args.num_frames)
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
    output_file = OUTPUT_PATH / f"raw_video_embeddings_clean{suffix}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(raw_embeddings, f)

    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        'num_subjects': len(raw_embeddings),
        'total_videos': total_videos,
        'total_frames': total_frames,
        'embedding_dim': int(extractor.model.config.hidden_size),
        'model_name': args.model_name,
        'frames_per_video': args.num_frames,
        'max_videos_per_subject': args.max_videos_per_subject,
        'doppler_filtered': filter_label,
        'subject_shard': f"{shard_k}/{shard_n}",
        'gpu_id': args.gpu_id,
        'extraction_time_min': round(elapsed / 60, 1),
    }
    with open(OUTPUT_PATH / f"raw_embedding_clean{suffix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    videos_per_subject = [len(v) for v in raw_embeddings.values()]
    frames_per_video_actual = [v.shape[0] for vlist in raw_embeddings.values() for v in vlist]

    print(f"\n=== Clean Raw Extraction Complete ===")
    print(f"Subjects: {len(raw_embeddings)}")
    print(f"Videos: {total_videos} ({np.mean(videos_per_subject):.1f} +/- {np.std(videos_per_subject):.1f} per subject)")
    print(f"Frames: {total_frames} ({np.mean(frames_per_video_actual):.1f} +/- {np.std(frames_per_video_actual):.1f} per video)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
