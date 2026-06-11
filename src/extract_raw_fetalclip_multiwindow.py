#!/usr/bin/env python3
"""
Multi-window FetalCLIP extraction.

Replaces `extract_raw_fetalclip.py`'s `np.linspace(0, N-1, num_frames)`
linspace sampler with **contiguous-burst** windows at native fps. Each
clip is sliced into multiple W-frame windows of stride S; each window
becomes its own per-window FetalCLIP sequence. Output format mirrors
the linspace pkl so that exp05/exp07 consume it without modification:

  dict[subject_id] -> list of windows -> np.array(W, 768)

The head's transformer attention runs over W tokens *within* a window,
capturing real cardiac motion (vs. structural snapshots in linspace).
Subject-level prediction aggregates per-window probabilities (canonical
`evaluate_video_model` path).

Usage:
  python extract_raw_fetalclip_multiwindow.py \\
      --window-frames 10 --window-stride 10 --max-windows-per-clip 8 \\
      --max-videos-per-subject 20 \\
      --output-suffix __mw_W10__s10__mw8 --gpu-id 0 --subject-shard 0/4 \\
      --include-doppler --resume

Smoke mode (--smoke-subjects N) processes first N subjects only and writes
to ..._smoke.pkl.
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
from PIL import Image
from tqdm import tqdm

import open_clip
from open_clip.factory import _MODEL_CONFIGS

DATA_PATH = Path(str(IFIND_DATA))
OUTPUT_PATH = Path(str(_Path_scrub(EMBEDDINGS_DIR) / "scaling"))
DOPPLER_FILTER_PATH = Path(str(_Path_scrub(RESULTS_DIR) / "doppler_filter/non_doppler_videos.txt"))

CHECKPOINT_DIR = Path(str(_Path_scrub(_CKPT_DIR) / "fetalclip"))
FETALCLIP_REPO = Path(str(_Path_scrub(_CKPT_DIR) / "fetalclip-repo"))
CHECKPOINT_FILE = CHECKPOINT_DIR / "FetalCLIP_weights.pt"
CONFIG_FILE = FETALCLIP_REPO / "FetalCLIP_config.json"

CHECKPOINT_INTERVAL = 50
EMBEDDING_DIM = 768


def register_fetalclip_arch():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"FetalCLIP_config.json not found at {CONFIG_FILE}\n"
            f"Clone the FetalCLIP repo to {FETALCLIP_REPO}."
        )
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    _MODEL_CONFIGS['FetalCLIP'] = cfg
    return cfg


class FetalCLIPExtractor:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = register_fetalclip_arch()
        print(f"Loading FetalCLIP on {self.device}")
        if not CHECKPOINT_FILE.exists():
            raise FileNotFoundError(f"FetalCLIP checkpoint missing: {CHECKPOINT_FILE}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'FetalCLIP', pretrained=str(CHECKPOINT_FILE),
        )
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"FetalCLIP loaded. Embedding dim: {EMBEDDING_DIM}")

    @torch.no_grad()
    def encode_frames(self, frames_rgb):
        """frames_rgb: list of HxWx3 uint8 RGB arrays. Returns (n, 768) ndarray."""
        if not frames_rgb:
            return None
        tensors = [self.preprocess(Image.fromarray(f)) for f in frames_rgb]
        batch = torch.stack(tensors, dim=0).to(self.device)
        feats = self.model.encode_image(batch)
        return feats.cpu().numpy()

    def extract_video_windows(self, video_path, window_frames, window_stride,
                              max_windows_per_clip):
        """Cut video into contiguous W-frame windows at stride S; embed each.

        Returns list of (W, 768) ndarrays — one entry per window — or None if
        the clip is shorter than W frames or unreadable. Each window is a
        contiguous burst at the clip's native fps (no skipping).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        # Read all frames sequentially (5s/30fps clip ≈ 150 frames ≈ 30MB RAM max).
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        total = len(all_frames)
        if total < window_frames:
            return None

        # Window starts: 0, stride, 2*stride, ... s.t. s + W <= total.
        # Cap at max_windows_per_clip.
        starts = list(range(0, total - window_frames + 1, window_stride))[:max_windows_per_clip]
        if not starts:
            return None

        windows = []
        for s in starts:
            burst = all_frames[s:s + window_frames]
            if len(burst) != window_frames:
                continue  # defensive; shouldn't happen
            emb = self.encode_frames(burst)
            if emb is None or emb.shape[0] != window_frames:
                continue
            windows.append(emb)

        return windows if windows else None


def get_subject_videos_from_filter(filter_path):
    if not filter_path.exists():
        raise FileNotFoundError(f"Doppler filter list missing: {filter_path}")
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
          f"for {len(subject_videos)} subjects (filtered)")
    if skipped:
        print(f"  Skipped {skipped} entries")
    return dict(subject_videos)


def get_subject_videos_unfiltered(data_path):
    subject_videos = defaultdict(list)
    skipped = 0
    for video_path in data_path.rglob("*.mp4"):
        try:
            subject_id = int(video_path.stem.split('_')[0])
            subject_videos[subject_id].append(video_path)
        except (ValueError, IndexError):
            skipped += 1
    print(f"Loaded {sum(len(v) for v in subject_videos.values())} videos "
          f"for {len(subject_videos)} subjects (Doppler IN)")
    if skipped:
        print(f"  Skipped {skipped} entries")
    return dict(subject_videos)


def parse_shard(shard_str):
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
    parser = argparse.ArgumentParser(description='Multi-window FetalCLIP extraction')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--smoke-subjects', type=int, default=None,
                        help='Process only first N subjects (smoke test)')
    parser.add_argument('--window-frames', type=int, default=10,
                        help='Frames per contiguous burst (W; default 10)')
    parser.add_argument('--window-stride', type=int, default=10,
                        help='Stride between window starts (default = W, non-overlapping)')
    parser.add_argument('--max-windows-per-clip', type=int, default=8,
                        help='Cap on windows produced per clip (default 8). '
                             'A 5s/30fps clip naturally yields ~15 windows at W=stride=10; '
                             'the cap controls training-time cost vs. spec defaults.')
    parser.add_argument('--max-videos-per-subject', type=int, default=20,
                        help='Cap on clips per subject (default 20)')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Suffix appended to output filename')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--subject-shard', type=parse_shard, default=(0, 1),
                        help="'k/N' format; process every Nth subject from offset k")
    parser.add_argument('--include-doppler', action='store_true',
                        help='Scan DATA_PATH (no Doppler filter)')
    args = parser.parse_args()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    shard_k, shard_n = args.subject_shard
    if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
        torch.cuda.set_device(args.gpu_id)
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    suffix = '_smoke' if args.smoke_subjects else args.output_suffix

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

    if args.smoke_subjects:
        subjects_with_videos = subjects_with_videos[:args.smoke_subjects]
        print(f"SMOKE: first {len(subjects_with_videos)} subjects only")

    if shard_n > 1:
        subjects_with_videos = [s for i, s in enumerate(subjects_with_videos)
                                if i % shard_n == shard_k]
        print(f"Shard {shard_k}/{shard_n}: {len(subjects_with_videos)} subjects this worker")

    checkpoint_path = OUTPUT_PATH / f"checkpoint_raw_fetalclip{suffix}.pkl"
    raw_embeddings = {}

    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            raw_embeddings = pickle.load(f)
        print(f"Resumed: loaded {len(raw_embeddings)} existing subjects")

    subjects_to_process = [s for s in subjects_with_videos if s not in raw_embeddings]
    print(f"Subjects to process: {len(subjects_to_process)}")
    if not subjects_to_process:
        print("All subjects already processed.")
        return

    extractor = FetalCLIPExtractor(device=device)
    start_time = time.time()
    total_windows = 0
    total_frames = 0

    for i, subject_id in enumerate(tqdm(subjects_to_process, desc="multi-window FetalCLIP")):
        try:
            videos = subject_videos[subject_id]
            if args.max_videos_per_subject and len(videos) > args.max_videos_per_subject:
                # Same RNG convention as the linspace extractor for cross-comparability.
                np.random.seed(42 + subject_id)
                indices = np.random.choice(len(videos), args.max_videos_per_subject, replace=False)
                videos = [videos[j] for j in indices]

            subject_windows = []
            for video_path in videos:
                windows = extractor.extract_video_windows(
                    video_path,
                    window_frames=args.window_frames,
                    window_stride=args.window_stride,
                    max_windows_per_clip=args.max_windows_per_clip,
                )
                if windows:
                    subject_windows.extend(windows)
                    total_windows += len(windows)
                    total_frames += sum(w.shape[0] for w in windows)

            if subject_windows:
                raw_embeddings[subject_id] = subject_windows

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(raw_embeddings, f)
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(subjects_to_process) - i - 1) / rate / 60
                print(f"\nCheckpoint: {len(raw_embeddings)} subjects. "
                      f"Rate: {rate:.2f}/s. ~{remaining:.0f} min remaining")
        except Exception as e:
            print(f"Error on subject {subject_id}: {e}")
            continue

    elapsed = time.time() - start_time

    output_file = OUTPUT_PATH / f"raw_video_embeddings_fetalclip{suffix}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(raw_embeddings, f)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    metadata = {
        'backbone': 'FetalCLIP',
        'extraction_mode': 'multi_window_contiguous_bursts',
        'window_frames': args.window_frames,
        'window_stride': args.window_stride,
        'max_windows_per_clip': args.max_windows_per_clip,
        'max_videos_per_subject': args.max_videos_per_subject,
        'huggingface_repo': 'numansaeed/fetalclip-model',
        'arch': 'open_clip ViT-L/14',
        'num_subjects': len(raw_embeddings),
        'total_windows': total_windows,
        'total_frames': total_frames,
        'embedding_dim': EMBEDDING_DIM,
        'doppler_filtered': filter_label,
        'smoke_mode': bool(args.smoke_subjects),
        'subject_shard': f"{shard_k}/{shard_n}",
        'gpu_id': args.gpu_id,
        'extraction_time_min': round(elapsed / 60, 1),
    }
    with open(OUTPUT_PATH / f"raw_embedding_fetalclip{suffix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    windows_per_subject = [len(v) for v in raw_embeddings.values()]
    frames_per_window = [w.shape[0] for vs in raw_embeddings.values() for w in vs]

    print(f"\n=== Multi-window FetalCLIP Extraction Complete ===")
    print(f"Subjects: {len(raw_embeddings)}")
    print(f"Windows: {total_windows} "
          f"({np.mean(windows_per_subject):.1f} ± {np.std(windows_per_subject):.1f}/subj)")
    print(f"Frames per window: {np.mean(frames_per_window):.1f} "
          f"± {np.std(frames_per_window):.1f}  (expect {args.window_frames})")
    if raw_embeddings:
        sample = next(iter(raw_embeddings.values()))[0]
        print(f"Sample window shape: {sample.shape}  "
              f"(expect ({args.window_frames}, {EMBEDDING_DIM}))")
        if sample.shape[1] != EMBEDDING_DIM:
            print(f"  WARNING: actual dim {sample.shape[1]} != expected {EMBEDDING_DIM}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
