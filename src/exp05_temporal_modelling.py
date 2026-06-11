#!/usr/bin/env python3
"""
Experiment 05: Temporal Modelling for Video Classification

Phase 2 of the FYP. Tests whether sequence-aware models can break the
~0.83 AUROC ceiling established by pooling-based Phase 1 experiments.

Key insight from Phase 1: pooling is order-invariant and cannot exploit
temporal structure in cardiac ultrasound videos. DINOv3 (Meta, 2025) showed
that a shallow 4-layer attentive probe on frozen image features matches
dedicated video models — validating that temporal reasoning on frozen
frame embeddings is a viable and efficient strategy.

Architecture groups:

  A. Recurrent models (video-level temporal)
     A1: Bidirectional LSTM
     A2: Bidirectional GRU

  B. Transformer-based models (video-level temporal)
     B1: Temporal Transformer with learned CLS token (DINOv3-inspired)
     B2: Cross-Attention Probe (query attends to frame sequence)
     B3: Temporal Transformer, 4 layers (deeper, closer to DINOv3 spec)

  C. Convolutional temporal models
     C1: Temporal CNN (1D convolutions over frame sequence)

  D. Hierarchical models (video-level temporal + subject-level aggregation)
     D1: BiLSTM + attention aggregation across videos
     D2: Temporal Transformer + attention aggregation across videos

  E. Controls and ablations
     E1: Shuffled frames (destroys temporal order — if performance drops,
         temporal structure matters)
     E2: Mean pooling baseline (reproduces Phase 1 for direct comparison)

  F. Architectural ablations
     F1: Positional encoding ablation (none / sinusoidal / learnable)
     F2: Hidden dimension ablation (64 / 128 / 256 / 512)

Data flow:
  - Raw frame-level DINOv2 embeddings: dict[subject_id] -> list[np.array(N_frames, 768)]
  - Each video has ~10 frames of 768-dim CLS token embeddings
  - Temporal models process per-video frame sequences
  - Subject-level prediction = aggregation of per-video predictions/embeddings

Usage:
  python exp05_temporal_modelling.py                          # all experiments
  python exp05_temporal_modelling.py --quick                  # subset for testing
  python exp05_temporal_modelling.py --experiments A1 B1 E1   # specific experiments
  python exp05_temporal_modelling.py --output-dir /path/to/dir
  python exp05_temporal_modelling.py --embeddings-path /path/to/raw_embeddings.pkl
"""

import argparse
import copy
import json
import math
import pickle
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

from paths import IFIND_DATA, DINOV2_EMBEDDINGS, RESULTS_DIR

DATA_PATH = IFIND_DATA
RAW_EMBEDDINGS_PATH = DINOV2_EMBEDDINGS
RESULTS_PATH = RESULTS_DIR
OUTPUT_PATH = RESULTS_DIR / "exp05_temporal"

SEED = 42
EMBEDDING_DIM = 768
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _infer_embedding_dim(raw_embeddings):
    """Set module EMBEDDING_DIM from loaded embedding tensors so video heads
    size input_proj for DINOv2-base / FetalCLIP (768) or DINOv2-large (1024).
    Defaults are preserved: base/FetalCLIP data infer 768, identical to before."""
    global EMBEDDING_DIM
    for vs in raw_embeddings.values():
        for v in vs:
            if v is not None and getattr(v, 'ndim', 0) >= 2 and v.shape[-1] > 0:
                EMBEDDING_DIM = int(v.shape[-1])
                return EMBEDDING_DIM
    return EMBEDDING_DIM

# Training defaults
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4
DEFAULT_EPOCHS = 80
DEFAULT_PATIENCE = 12
DEFAULT_BATCH_SIZE = 64


def init_head_bias_to_prior(model, pos_rate):
    """
    Initialize the final linear layer's bias to the prior log-odds log(p/(1-p)).

    Rationale: with pos_weight > 1 and a zero bias, the model starts at sigmoid(0) = 0.5
    which sits at a saddle of the weighted BCE loss — gradients are tiny and training
    stalls (observed in smoke test: loss parked at 1.17 for 19 epochs across all temporal
    models). Setting the bias to the prior moves the starting point off the saddle.
    Standard trick from Lin et al., Focal Loss paper.
    """
    pos_rate = float(max(min(pos_rate, 0.999), 1e-3))
    prior_logit = float(np.log(pos_rate / (1 - pos_rate)))
    # All our models end their classifier head in a final nn.Linear(_, 1).
    # Find it and set its bias.
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == 1:
            last_linear = m
    if last_linear is not None and last_linear.bias is not None:
        with torch.no_grad():
            last_linear.bias.fill_(prior_logit)
    return prior_logit


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducible at the seed level, but allow cuDNN to pick fast algorithms.
    # Forcing determinism made Conv1d ~25x slower in the smoke test on t4.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATA LOADING
# ============================================================

def load_data(embeddings_path=None):
    """Load raw frame-level embeddings and subject labels/splits."""
    path = embeddings_path or RAW_EMBEDDINGS_PATH
    print(f"Loading raw embeddings from {path}...")
    with open(path, 'rb') as f:
        raw_embeddings = pickle.load(f)
    _infer_embedding_dim(raw_embeddings)
    print(f"Embedding dim inferred from data: {EMBEDDING_DIM}")

    labels_df = pd.read_csv(DATA_PATH / "subject_level_labels.csv")
    condition_cols = [c for c in labels_df.columns if c != 'subject']
    labels_df['unhealthy'] = (labels_df[condition_cols].sum(axis=1) > 0).astype(int)

    train_subjects = pd.read_csv(RESULTS_PATH / "train_subjects.csv")['subject'].tolist()
    val_subjects = pd.read_csv(RESULTS_PATH / "val_subjects.csv")['subject'].tolist()
    test_subjects = pd.read_csv(RESULTS_PATH / "test_subjects.csv")['subject'].tolist()

    return raw_embeddings, labels_df, train_subjects, val_subjects, test_subjects


def get_valid_subjects(raw_embeddings, labels_df, subjects):
    """Filter to subjects present in both embeddings and labels."""
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()
    return [s for s in subjects if s in raw_embeddings and s in labels_dict]


# ============================================================
# DATASETS
# ============================================================

class VideoDataset(Dataset):
    """
    Returns individual video frame sequences for temporal modelling.

    Each item is one video: (frame_sequence, label, subject_id, video_idx).
    Frame sequence shape: (num_frames, 768).
    """

    def __init__(self, subjects, raw_embeddings, labels_dict, shuffle_frames=False):
        self.items = []
        self.shuffle_frames = shuffle_frames
        for sid in subjects:
            sid = int(sid)
            if sid not in raw_embeddings or sid not in labels_dict:
                continue
            label = int(labels_dict[sid])
            for vidx, video_emb in enumerate(raw_embeddings[sid]):
                if video_emb is None or video_emb.shape[0] == 0:
                    continue
                self.items.append((video_emb, label, sid, vidx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        emb, label, sid, vidx = self.items[idx]
        frames = torch.FloatTensor(emb)
        if self.shuffle_frames:
            perm = torch.randperm(frames.shape[0])
            frames = frames[perm]
        return frames, torch.FloatTensor([label]), sid, vidx


def collate_video_batch(batch):
    """Collate variable-length video sequences with padding."""
    frames_list, labels, sids, vidxs = zip(*batch)
    lengths = torch.LongTensor([f.shape[0] for f in frames_list])
    padded = pad_sequence(frames_list, batch_first=True)  # (B, max_len, 768)
    labels = torch.cat(labels)  # (B,)
    return padded, labels, lengths, list(sids), list(vidxs)


class SubjectDataset(Dataset):
    """
    Returns all videos for a subject as a single item.
    For hierarchical models that process video-level then subject-level.

    Each item: (list_of_video_tensors, label, subject_id).
    """

    def __init__(self, subjects, raw_embeddings, labels_dict):
        self.items = []
        for sid in subjects:
            sid = int(sid)
            if sid not in raw_embeddings or sid not in labels_dict:
                continue
            videos = [torch.FloatTensor(v) for v in raw_embeddings[sid]
                      if v is not None and v.shape[0] > 0]
            if not videos:
                continue
            self.items.append((videos, int(labels_dict[sid]), sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================================
# MODEL A: RECURRENT MODELS
# ============================================================

class BiLSTMClassifier(nn.Module):
    """
    A1: Bidirectional LSTM processes frame sequence, final hidden state
    is passed through a classifier head.
    """

    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        """x: (B, T, 768), lengths: (B,)"""
        x = self.input_norm(x)
        x = self.input_proj(x)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, hidden) — take last layer fwd + bwd
        fwd = h_n[-2]  # last layer forward
        bwd = h_n[-1]  # last layer backward
        hidden = torch.cat([fwd, bwd], dim=1)  # (B, hidden*2)
        hidden = self.dropout(hidden)
        logit = self.classifier(hidden).squeeze(-1)  # (B,)
        return logit

    def get_video_embedding(self, x, lengths=None):
        """Return the temporal embedding (before classifier) for hierarchical models."""
        x = self.input_norm(x)
        x = self.input_proj(x)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(x)
        fwd = h_n[-2]
        bwd = h_n[-1]
        return torch.cat([fwd, bwd], dim=1)


class BiGRUClassifier(nn.Module):
    """A2: Bidirectional GRU — lighter alternative to LSTM."""

    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        x = self.input_norm(x)
        x = self.input_proj(x)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(x)
        fwd = h_n[-2]
        bwd = h_n[-1]
        hidden = torch.cat([fwd, bwd], dim=1)
        hidden = self.dropout(hidden)
        return self.classifier(hidden).squeeze(-1)


# ============================================================
# MODEL B: TRANSFORMER-BASED MODELS
# ============================================================

class TemporalTransformer(nn.Module):
    """
    B1/B3: Self-attention transformer with learnable CLS token.

    Inspired by DINOv3's attentive probe: a shallow transformer on top of
    frozen frame features captures temporal dynamics. The CLS token attends
    to all frames and aggregates temporal information for classification.

    B1: 2 layers (lightweight)
    B3: 4 layers (closer to DINOv3 attentive probe spec)
    """

    def __init__(self, input_dim=768, hidden_dim=256, n_heads=4, n_layers=2,
                 dropout=0.3, pos_encoding='sinusoidal'):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        if pos_encoding == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim)
        elif pos_encoding == 'learnable':
            self.pos_enc = LearnablePositionalEncoding(hidden_dim)
        else:
            self.pos_enc = None

        self.layer_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        """x: (B, T, 768), lengths: (B,)"""
        B, T, _ = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)  # (B, T, hidden)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, hidden)

        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.layer_norm(x)

        # Create padding mask: True = ignore
        if lengths is not None:
            max_len = T + 1  # +1 for CLS
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= (lengths.unsqueeze(1) + 1)
            mask[:, 0] = False  # never mask CLS
        else:
            mask = None

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_out = x[:, 0]  # (B, hidden)
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out).squeeze(-1)

    def get_video_embedding(self, x, lengths=None):
        """Return CLS embedding before classifier."""
        B, T, _ = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.layer_norm(x)
        if lengths is not None:
            max_len = T + 1
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= (lengths.unsqueeze(1) + 1)
            mask[:, 0] = False
        else:
            mask = None
        x = self.transformer(x, src_key_padding_mask=mask)
        return x[:, 0]


class CrossAttentionProbe(nn.Module):
    """
    B2: Cross-attention probe — learnable query attends to frame sequence.

    Instead of prepending a CLS token to the self-attention sequence, this
    model uses cross-attention: a learnable query vector attends to the
    frame embeddings. This is more parameter-efficient and directly models
    the "read-out" operation.
    """

    def __init__(self, input_dim=768, hidden_dim=256, n_heads=4, n_layers=2,
                 dropout=0.3, pos_encoding='sinusoidal'):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        if pos_encoding == 'sinusoidal':
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim)
        elif pos_encoding == 'learnable':
            self.pos_enc = None
            self.pos_emb = LearnablePositionalEncoding(hidden_dim)
        else:
            self.pos_enc = None

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, n_heads, dropout=dropout, batch_first=True,
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            }))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)

        if self.pos_enc is not None:
            x = self.pos_enc(x)
        elif hasattr(self, 'pos_emb'):
            x = self.pos_emb(x)
        x = self.layer_norm(x)

        # Key padding mask for cross-attention
        if lengths is not None:
            key_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            key_mask = None

        q = self.query.expand(B, -1, -1)  # (B, 1, hidden)

        for layer in self.cross_attn_layers:
            # Cross-attention: query attends to frame sequence
            attn_out, _ = layer['cross_attn'](q, x, x, key_padding_mask=key_mask)
            q = layer['norm1'](q + attn_out)
            ffn_out = layer['ffn'](q)
            q = layer['norm2'](q + ffn_out)

        out = q.squeeze(1)  # (B, hidden)
        out = self.dropout(out)
        return self.classifier(out).squeeze(-1)


# ============================================================
# MODEL C: TEMPORAL CNN
# ============================================================

class TemporalCNN(nn.Module):
    """
    C1: 1D convolutional network over frame sequence.

    Captures local temporal patterns (e.g., 3-frame motion patterns) through
    stacked dilated convolutions. Efficient and handles short sequences well.
    """

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Stack of 1D convolutions with increasing dilation
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        """x: (B, T, 768)"""
        x = self.input_norm(x)
        x = self.input_proj(x)  # (B, T, hidden)
        x = x.transpose(1, 2)   # (B, hidden, T) for Conv1d
        x = self.convs(x)       # (B, hidden, T)

        # Global average pooling (respecting lengths)
        if lengths is not None:
            mask = torch.arange(x.size(2), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()  # (B, 1, T)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)

        return self.classifier(x).squeeze(-1)


# ============================================================
# MODEL D: HIERARCHICAL MODELS
# ============================================================

class HierarchicalModel(nn.Module):
    """
    D1/D2: Two-level temporal model.

    Level 1: Temporal encoder processes each video independently -> video embedding.
    Level 2: Attention aggregation across video embeddings -> subject prediction.

    This respects the natural hierarchy: frames form coherent videos,
    videos are independent scans of the same subject.
    """

    def __init__(self, video_encoder, video_embed_dim, dropout=0.3):
        super().__init__()
        self.video_encoder = video_encoder
        self.video_attn = nn.Sequential(
            nn.Linear(video_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(video_embed_dim, video_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(video_embed_dim // 2, 1),
        )

    def forward(self, video_list):
        """
        video_list: list of tensors, each (T_i, 768) — videos for one subject.
        Returns: logit (scalar).
        """
        video_embeddings = []
        for video in video_list:
            video = video.unsqueeze(0).to(next(self.parameters()).device)
            lengths = torch.LongTensor([video.shape[1]]).to(video.device)
            emb = self.video_encoder.get_video_embedding(video, lengths)
            video_embeddings.append(emb.squeeze(0))

        video_stack = torch.stack(video_embeddings)  # (N_videos, embed_dim)

        # Attention aggregation
        attn_scores = self.video_attn(video_stack)  # (N_videos, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        subject_emb = (attn_weights * video_stack).sum(dim=0)  # (embed_dim,)

        logit = self.classifier(subject_emb)
        return logit.squeeze()


# ============================================================
# G. MULTIPLE-INSTANCE LEARNING (MIL) over clips
# bag = subject, instance = clip (clip embedding = frame-mean). The subject label
# is the only supervision; no clip is assumed positive. See the pre-registration notes for
# the MIL probe. G0/G1 isolate the gating term
# the existing additive-attention HierarchicalModel (and the SF clip-attention null)
# lack; G2 lets a single diagnostic clip drive a positive bag and the model ignore
# label-propagated-noisy clips, the principled response to weak subject-level labels.
# ============================================================

class MILAttention(nn.Module):
    """Attention-MIL over clip embeddings (clip = frame-mean).

    gated=False : additive tanh attention (the clean mean-attention anchor, G0).
    gated=True  : Ilse 2018 gated attention, a_i = softmax(w . (tanh(V h_i) (*) sigmoid(U h_i))) (G1).
    """

    def __init__(self, input_dim=768, hidden_dim=64, attn_dim=128, dropout=0.3, gated=True):
        super().__init__()
        self.gated = gated
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim) if gated else None
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _attn(self, h):
        g = torch.tanh(self.attn_V(h))
        if self.gated:
            g = g * torch.sigmoid(self.attn_U(h))
        return torch.softmax(self.attn_w(g), dim=0)  # (N_clips, 1)

    def forward(self, video_list):
        """video_list: list of (T_i, input_dim) clip tensors for one subject. Returns scalar logit."""
        device = next(self.parameters()).device
        clip_means = torch.stack([v.to(device).mean(dim=0) for v in video_list])  # (N, input_dim)
        h = self.proj(clip_means)                       # (N, hidden)
        a = self._attn(h)                               # (N, 1)
        bag = (a * h).sum(dim=0)                        # (hidden,)
        return self.classifier(bag).squeeze()

    def attention_weights(self, video_list):
        """Per-clip attention weights (for the interpretability figure)."""
        device = next(self.parameters()).device
        clip_means = torch.stack([v.to(device).mean(dim=0) for v in video_list])
        return self._attn(self.proj(clip_means)).detach().cpu().squeeze(-1)


class MILInstancePool(nn.Module):
    """Instance-pooling MIL: per-clip logit, bag logit = tau * logsumexp(z_i / tau) - tau * log(N).

    A smooth max over clips: tau->0 approaches max-pooling (one diagnostic clip drives a
    positive bag), tau->inf approaches mean. The normalisation keeps the prior-bias init exact.
    """

    def __init__(self, input_dim=768, hidden_dim=64, dropout=0.3, tau=1.0):
        super().__init__()
        self.tau = float(tau)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.instance_head = nn.Linear(hidden_dim, 1)

    def forward(self, video_list):
        device = next(self.parameters()).device
        clip_means = torch.stack([v.to(device).mean(dim=0) for v in video_list])  # (N, input_dim)
        z = self.instance_head(self.encoder(clip_means)).squeeze(-1)              # (N,) per-clip logits
        n = z.shape[0]
        bag_logit = self.tau * torch.logsumexp(z / self.tau, dim=0) - self.tau * float(np.log(n))
        return bag_logit.squeeze()

    def instance_logits(self, video_list):
        """Per-clip logits (for the interpretability figure)."""
        device = next(self.parameters()).device
        clip_means = torch.stack([v.to(device).mean(dim=0) for v in video_list])
        return self.instance_head(self.encoder(clip_means)).squeeze(-1).detach().cpu()


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def compute_pos_weight(labels_dict, train_subjects):
    """Compute positive class weight for BCEWithLogitsLoss."""
    labels = [labels_dict[s] for s in train_subjects if s in labels_dict]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return n_neg / max(n_pos, 1)


def find_optimal_threshold(y_true, y_prob):
    """Sweep thresholds 0.05-0.95 in steps of 0.01 and return the one maximising F1."""
    best_f1 = -1.0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE — average gap between confidence and accuracy in equal-width prob bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if in_bin.sum() == 0:
            continue
        acc = float(y_true[in_bin].mean())
        conf = float(y_prob[in_bin].mean())
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def recall_at_specificity(y_true, y_prob, target_spec):
    """Largest recall achievable while specificity >= target_spec.

    Sweeps thresholds from high to low; tracks the best (highest) recall
    seen while specificity is still above target.
    """
    if len(set(y_true)) < 2:
        return 0.0
    thresholds = np.unique(y_prob)
    thresholds = np.concatenate([[1.0 + 1e-6], thresholds, [0.0 - 1e-6]])
    best_recall = 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        spec = tn / max(tn + fp, 1)
        if spec >= target_spec:
            recall = tp / max(tp + fn, 1)
            if recall > best_recall:
                best_recall = recall
    return float(best_recall)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Subject-level bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0 or len(set(y_true)) < 2:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        if len(set(y_t)) < 2:
            continue
        try:
            vals.append(float(metric_fn(y_t, y_p)))
        except Exception:
            continue
    if not vals:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
    vals = np.array(vals)
    return {
        'mean': float(vals.mean()),
        'ci_low': float(np.quantile(vals, alpha / 2)),
        'ci_high': float(np.quantile(vals, 1 - alpha / 2)),
        'n_boot': len(vals),
    }


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute all evaluation metrics at a given threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    has_two_classes = len(set(y_true.tolist())) > 1
    metrics = {
        'auroc': roc_auc_score(y_true, y_prob) if has_two_classes else 0.0,
        'aupr': average_precision_score(y_true, y_prob) if has_two_classes else 0.0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'specificity': tn / max(tn + fp, 1),
        'brier': brier_score(y_true, y_prob),
        'ece': expected_calibration_error(y_true, y_prob),
        'recall_at_spec95': recall_at_specificity(y_true, y_prob, 0.95),
        'recall_at_spec90': recall_at_specificity(y_true, y_prob, 0.90),
        'threshold': float(threshold),
        'n_samples': len(y_true),
        'n_positive': int(sum(y_true)),
        'n_negative': int(len(y_true) - sum(y_true)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    return metrics


def evaluate_video_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate a video-level model at the SUBJECT level.

    Aggregates per-video predictions to subject-level by averaging
    predicted probabilities across all videos for each subject.
    """
    model.eval()
    subject_probs = {}
    subject_labels = {}

    with torch.no_grad():
        for padded, labels, lengths, sids, vidxs in dataloader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            logits = model(padded, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()

            for prob, label, sid in zip(probs, labels.numpy(), sids):
                if sid not in subject_probs:
                    subject_probs[sid] = []
                    subject_labels[sid] = label
                subject_probs[sid].append(prob)

    # Aggregate: mean probability across videos
    y_true = []
    y_prob = []
    sids_out = []
    for sid in subject_probs:
        y_true.append(subject_labels[sid])
        y_prob.append(np.mean(subject_probs[sid]))
        sids_out.append(int(sid))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return compute_metrics(y_true, y_prob, threshold), y_true, y_prob, sids_out


def evaluate_hierarchical_model(model, dataset, device, threshold=0.5):
    """Evaluate a hierarchical model (processes all videos per subject at once)."""
    model.eval()
    y_true = []
    y_prob = []
    sids_out = []

    with torch.no_grad():
        for i in range(len(dataset)):
            videos, label, sid = dataset[i]
            logit = model(videos)
            prob = torch.sigmoid(logit).cpu().item()
            y_true.append(label)
            y_prob.append(prob)
            sids_out.append(int(sid))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return compute_metrics(y_true, y_prob, threshold), y_true, y_prob, sids_out


def train_video_model(model, train_loader, val_loader, config, device):
    """
    Train a video-level temporal model.

    Uses per-video training with subject-level validation metrics.
    Early stopping on validation AUROC.
    """
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.get('lr', DEFAULT_LR),
        weight_decay=config.get('wd', DEFAULT_WD),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', DEFAULT_EPOCHS),
    )

    best_val_auroc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', DEFAULT_PATIENCE)
    epochs = config.get('epochs', DEFAULT_EPOCHS)
    history = {'train_loss': [], 'val_auroc': [], 'val_aupr': [], 'grad_norm': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        first_batch_grad_norm = None

        for padded, labels, lengths, sids, vidxs in train_loader:
            padded = padded.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            logits = model(padded, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if first_batch_grad_norm is None:
                first_batch_grad_norm = float(gnorm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        history['grad_norm'].append(first_batch_grad_norm)

        # Validate at subject level
        val_metrics, _, _, _ = evaluate_video_model(model, val_loader, device)
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_aupr'].append(val_metrics['aupr'])

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Log first 3 epochs (to see training start), then every 10
        if epoch < 3 or (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"val_auroc={val_metrics['auroc']:.4f}, val_aupr={val_metrics['aupr']:.4f}, "
                  f"gnorm={first_batch_grad_norm:.3f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val AUROC: {best_val_auroc:.4f})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_val_auroc


def train_hierarchical_model(model, train_dataset, val_dataset, config, device):
    """
    Train a hierarchical model (one subject at a time).
    """
    pos_weight = torch.FloatTensor([config['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.get('lr', DEFAULT_LR),
        weight_decay=config.get('wd', DEFAULT_WD),
    )

    best_val_auroc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', DEFAULT_PATIENCE)
    epochs = config.get('epochs', DEFAULT_EPOCHS)
    history = {'train_loss': [], 'val_auroc': [], 'val_aupr': []}

    indices = list(range(len(train_dataset)))

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0

        for idx in indices:
            videos, label, sid = train_dataset[idx]
            label_t = torch.FloatTensor([label]).to(device)

            logit = model(videos).unsqueeze(0)
            loss = criterion(logit, label_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(indices)
        history['train_loss'].append(avg_loss)

        val_metrics, _, _, _ = evaluate_hierarchical_model(model, val_dataset, device)
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_aupr'].append(val_metrics['aupr'])

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch < 3 or (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"val_auroc={val_metrics['auroc']:.4f}, val_aupr={val_metrics['aupr']:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val AUROC: {best_val_auroc:.4f})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_val_auroc


# ============================================================
# MEAN POOLING BASELINE (for direct comparison)
# ============================================================

def run_mean_pooling_baseline(raw_embeddings, labels_dict, train_subjects, val_subjects,
                              test_subjects, pos_weight):
    """E2: Reproduce Phase 1 mean pooling baseline for direct comparison.

    Uses the grand mean across all frames of all videos for each subject
    (matching exp02_clean / Phase 1), NOT mean-of-video-means, which differs
    when videos have unequal frame counts.
    """
    from sklearn.linear_model import LogisticRegression

    def aggregate(subject_id):
        videos = raw_embeddings[subject_id]
        all_frames = np.concatenate(
            [v for v in videos if v is not None and v.shape[0] > 0], axis=0,
        )
        return all_frames.mean(axis=0)

    def make_arrays(subjects):
        X, y, sids = [], [], []
        for s in subjects:
            s_int = int(s)
            if s_int in raw_embeddings and s_int in labels_dict:
                X.append(aggregate(s_int))
                y.append(labels_dict[s_int])
                sids.append(s_int)
        return np.array(X), np.array(y), sids

    X_train, y_train, _ = make_arrays(train_subjects)
    X_val, y_val, val_sids = make_arrays(val_subjects)
    X_test, y_test, test_sids = make_arrays(test_subjects)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    clf.fit(X_train, y_train)

    results = {}
    probs_by_split = {}
    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        probs = clf.predict_proba(X)[:, 1]
        results[name] = compute_metrics(y, probs)
        probs_by_split[name] = (y, probs)

    val_y, val_p = probs_by_split['val']
    test_y, test_p = probs_by_split['test']
    opt_thr = find_optimal_threshold(val_y, val_p)
    results['val_opt_thresh'] = compute_metrics(val_y, val_p, threshold=opt_thr)
    results['test_opt_thresh'] = compute_metrics(test_y, test_p, threshold=opt_thr)
    results['optimal_threshold'] = opt_thr
    results['test_predictions'] = {
        'y_true': test_y.tolist(),
        'y_prob': test_p.tolist(),
        'subject_ids': test_sids,
    }

    return results


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def get_experiment_configs():
    """Define all temporal modelling experiments."""
    configs = OrderedDict()

    # A. Recurrent models
    configs['A1_bilstm'] = {
        'description': 'Bidirectional LSTM (2 layers, hidden=256)',
        'group': 'A_recurrent',
        'model_fn': lambda: BiLSTMClassifier(hidden_dim=256, num_layers=2),
        'type': 'video',
        'lr': 1e-3, 'wd': 1e-4,
    }
    configs['A2_bigru'] = {
        'description': 'Bidirectional GRU (2 layers, hidden=256)',
        'group': 'A_recurrent',
        'model_fn': lambda: BiGRUClassifier(hidden_dim=256, num_layers=2),
        'type': 'video',
        'lr': 1e-3, 'wd': 1e-4,
    }

    # B. Transformer-based
    configs['B1_transformer_2L'] = {
        'description': 'Temporal Transformer, 2 layers, CLS token (lightweight)',
        'group': 'B_transformer',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=256, n_heads=4, n_layers=2, pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['B2_cross_attn'] = {
        'description': 'Cross-Attention Probe, 2 layers (query attends to frames)',
        'group': 'B_transformer',
        'model_fn': lambda: CrossAttentionProbe(
            hidden_dim=256, n_heads=4, n_layers=2, pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['B3_transformer_4L'] = {
        'description': 'Temporal Transformer, 4 layers (DINOv3-inspired depth)',
        'group': 'B_transformer',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=256, n_heads=4, n_layers=4, pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 3e-4, 'wd': 1e-4,
    }

    # C. Temporal CNN
    configs['C1_temporal_cnn'] = {
        'description': 'Temporal CNN (dilated 1D convolutions)',
        'group': 'C_cnn',
        'model_fn': lambda: TemporalCNN(hidden_dim=256),
        'type': 'video',
        'lr': 1e-3, 'wd': 1e-4,
    }

    # D. Hierarchical models
    configs['D1_bilstm_hierarchical'] = {
        'description': 'BiLSTM per-video + attention aggregation across videos',
        'group': 'D_hierarchical',
        'model_fn': lambda: HierarchicalModel(
            video_encoder=BiLSTMClassifier(hidden_dim=128, num_layers=1),
            video_embed_dim=256,  # 128 * 2 (bidirectional)
        ),
        'type': 'hierarchical',
        'lr': 5e-4, 'wd': 1e-4, 'epochs': 60, 'patience': 10,
    }
    configs['D2_transformer_hierarchical'] = {
        'description': 'Temporal Transformer per-video + attention aggregation',
        'group': 'D_hierarchical',
        'model_fn': lambda: HierarchicalModel(
            video_encoder=TemporalTransformer(
                hidden_dim=128, n_heads=4, n_layers=2, pos_encoding='sinusoidal'),
            video_embed_dim=128,
        ),
        'type': 'hierarchical',
        'lr': 5e-4, 'wd': 1e-4, 'epochs': 60, 'patience': 10,
    }

    # E. Controls
    configs['E1_shuffled_bilstm'] = {
        'description': 'BiLSTM with shuffled frames (control: temporal order destroyed)',
        'group': 'E_control',
        'model_fn': lambda: BiLSTMClassifier(hidden_dim=256, num_layers=2),
        'type': 'video',
        'shuffle_frames': True,
        'lr': 1e-3, 'wd': 1e-4,
    }
    configs['E2_mean_pool_baseline'] = {
        'description': 'Mean pooling + LogReg (Phase 1 baseline on same data)',
        'group': 'E_control',
        'type': 'baseline',
    }

    # F. Ablations
    configs['F1a_no_pos_enc'] = {
        'description': 'Temporal Transformer WITHOUT positional encoding',
        'group': 'F_ablation',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=256, n_heads=4, n_layers=2, pos_encoding='none'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['F1b_learnable_pos'] = {
        'description': 'Temporal Transformer with LEARNABLE positional encoding',
        'group': 'F_ablation',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=256, n_heads=4, n_layers=2, pos_encoding='learnable'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['F2a_hidden64'] = {
        'description': 'Temporal Transformer hidden_dim=64 (minimal)',
        'group': 'F_ablation',
        'model_fn': lambda: TemporalTransformer(
            input_dim=EMBEDDING_DIM, hidden_dim=64, n_heads=4, n_layers=2,
            pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['F2b_hidden128'] = {
        'description': 'Temporal Transformer hidden_dim=128',
        'group': 'F_ablation',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=128, n_heads=4, n_layers=2, pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 5e-4, 'wd': 1e-4,
    }
    configs['F2c_hidden512'] = {
        'description': 'Temporal Transformer hidden_dim=512',
        'group': 'F_ablation',
        'model_fn': lambda: TemporalTransformer(
            hidden_dim=512, n_heads=8, n_layers=2, pos_encoding='sinusoidal'),
        'type': 'video',
        'lr': 3e-4, 'wd': 1e-4,
    }

    # G. Multiple-Instance Learning over clips (pre-registered probe). bag=subject,
    # instance=clip. G0 anchor isolates the gating term G1 adds; G2 instance-pool
    # lets one diagnostic clip drive the bag (the weak-label-noise response).
    configs['G0_mil_meanattn'] = {
        'description': 'MIL mean-attention anchor (frame-mean clips + additive softmax attention, ungated, hidden=64)',
        'group': 'G_mil',
        'model_fn': lambda: MILAttention(input_dim=EMBEDDING_DIM, hidden_dim=64, gated=False),
        'type': 'hierarchical',
        'lr': 5e-4, 'wd': 1e-4, 'epochs': 60, 'patience': 10,
    }
    configs['G1_mil_gated'] = {
        'description': 'Gated-attention MIL over clips (Ilse 2018 tanh*sigmoid gate, hidden=64)',
        'group': 'G_mil',
        'model_fn': lambda: MILAttention(input_dim=EMBEDDING_DIM, hidden_dim=64, gated=True),
        'type': 'hierarchical',
        'lr': 5e-4, 'wd': 1e-4, 'epochs': 60, 'patience': 10,
    }
    configs['G2_mil_instance'] = {
        'description': 'Instance-pooling MIL (per-clip logit + LSE/smooth-max bag pooling, hidden=64, tau=1.0)',
        'group': 'G_mil',
        'model_fn': lambda: MILInstancePool(input_dim=EMBEDDING_DIM, hidden_dim=64, tau=1.0),
        'type': 'hierarchical',
        'lr': 5e-4, 'wd': 1e-4, 'epochs': 60, 'patience': 10,
    }

    return configs


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_all_experiments(quick=False, selected=None, embeddings_path=None, resume=False):
    set_seed(SEED)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_PATH}")

    # Load data
    raw_embeddings, labels_df, train_subjects, val_subjects, test_subjects = load_data(embeddings_path)
    labels_dict = labels_df.set_index('subject')['unhealthy'].to_dict()

    train_subjects = get_valid_subjects(raw_embeddings, labels_df, train_subjects)
    val_subjects = get_valid_subjects(raw_embeddings, labels_df, val_subjects)
    test_subjects = get_valid_subjects(raw_embeddings, labels_df, test_subjects)

    print(f"Train: {len(train_subjects)} subjects, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    n_train_pos = sum(labels_dict[s] for s in train_subjects)
    n_train_neg = len(train_subjects) - n_train_pos
    pos_weight = n_train_neg / max(n_train_pos, 1)
    pos_rate = n_train_pos / max(n_train_pos + n_train_neg, 1)
    prior_logit = float(np.log(pos_rate / (1 - pos_rate)))
    print(f"Train class balance: {n_train_neg} healthy, {n_train_pos} unhealthy "
          f"(pos_weight={pos_weight:.2f}, prior_logit={prior_logit:.3f})")

    # Build datasets
    train_ds = VideoDataset(train_subjects, raw_embeddings, labels_dict)
    val_ds = VideoDataset(val_subjects, raw_embeddings, labels_dict)
    test_ds = VideoDataset(test_subjects, raw_embeddings, labels_dict)

    train_loader = DataLoader(train_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True,
                              collate_fn=collate_video_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False,
                            collate_fn=collate_video_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=False,
                             collate_fn=collate_video_batch, num_workers=0)

    # Hierarchical datasets
    train_subj_ds = SubjectDataset(train_subjects, raw_embeddings, labels_dict)
    val_subj_ds = SubjectDataset(val_subjects, raw_embeddings, labels_dict)
    test_subj_ds = SubjectDataset(test_subjects, raw_embeddings, labels_dict)

    print(f"Video-level: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test videos")
    print(f"Subject-level: {len(train_subj_ds)} train, {len(val_subj_ds)} val, {len(test_subj_ds)} test")

    # Run experiments
    configs = get_experiment_configs()

    if selected:
        configs = OrderedDict((k, v) for k, v in configs.items() if k in selected)
    if quick:
        # Run one from each group
        quick_set = ['A1_bilstm', 'B1_transformer_2L', 'C1_temporal_cnn', 'E2_mean_pool_baseline']
        configs = OrderedDict((k, v) for k, v in configs.items() if k in quick_set)

    all_results = {}
    print(f"\n{'='*70}")
    print(f"Running {len(configs)} experiments")
    print(f"{'='*70}\n")

    def _serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _checkpoint_results():
        """Incremental save so a crash mid-sweep doesn't lose completed runs."""
        try:
            tmp = json.loads(json.dumps(all_results, default=_serializable))
            with open(OUTPUT_PATH / "results.json", 'w') as f:
                json.dump(tmp, f, indent=2)
        except Exception as ce:
            print(f"  [warn] checkpoint save failed: {ce}")

    def _attach_bootstrap_and_save_preds(result_dict, y_true, y_prob, sids):
        """Compute bootstrap CIs from stored test predictions and attach to result."""
        result_dict['bootstrap_test'] = {
            'auroc': bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=1000, seed=SEED),
            'aupr': bootstrap_ci(y_true, y_prob, average_precision_score, n_boot=1000, seed=SEED),
        }
        result_dict['test_predictions'] = {
            'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
            'y_prob': y_prob.tolist() if hasattr(y_prob, 'tolist') else list(y_prob),
            'subject_ids': list(sids) if sids is not None else None,
        }

    def _save_checkpoint(model, exp_name):
        """Save best model state_dict for post-hoc analysis."""
        try:
            ckpt_dir = OUTPUT_PATH / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"{exp_name}.pt")
        except Exception as ce:
            print(f"  [warn] checkpoint save failed for {exp_name}: {ce}")

    # Resume support: skip experiments already in results.json
    existing_results = {}
    if resume:
        results_path = OUTPUT_PATH / "results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
                completed = [k for k, v in existing_results.items() if 'error' not in v]
                print(f"[resume] Found {len(completed)} completed experiments — will skip:")
                for k in completed:
                    print(f"  - {k}")
                all_results.update(existing_results)
            except Exception as re:
                print(f"[resume] failed to load existing results: {re}")

    for exp_name, config in configs.items():
        if resume and exp_name in existing_results and 'error' not in existing_results[exp_name]:
            print(f"\n--- {exp_name}: SKIP (resume — already completed) ---")
            continue
        print(f"\n--- {exp_name}: {config['description']} ---")
        set_seed(SEED)
        start = time.time()

        config['pos_weight'] = pos_weight

        try:
            if config['type'] == 'baseline':
                # Mean pooling baseline
                results = run_mean_pooling_baseline(
                    raw_embeddings, labels_dict, train_subjects,
                    val_subjects, test_subjects, pos_weight,
                )
                elapsed = time.time() - start
                all_results[exp_name] = {
                    'config': {k: v for k, v in config.items()
                               if k not in ('model_fn',)},
                    'type': config['type'],
                    'train': results['train'],
                    'val': results['val'],
                    'test': results['test'],
                    'val_opt_thresh': results['val_opt_thresh'],
                    'test_opt_thresh': results['test_opt_thresh'],
                    'optimal_threshold': results['optimal_threshold'],
                    'elapsed_sec': round(elapsed, 1),
                }
                _attach_bootstrap_and_save_preds(
                    all_results[exp_name],
                    np.asarray(results['test_predictions']['y_true']),
                    np.asarray(results['test_predictions']['y_prob']),
                    results['test_predictions'].get('subject_ids'),
                )

            elif config['type'] == 'video':
                # Build dataloaders (handle shuffled frames control)
                if config.get('shuffle_frames'):
                    shuf_train_ds = VideoDataset(train_subjects, raw_embeddings, labels_dict,
                                                 shuffle_frames=True)
                    shuf_val_ds = VideoDataset(val_subjects, raw_embeddings, labels_dict,
                                               shuffle_frames=True)
                    shuf_test_ds = VideoDataset(test_subjects, raw_embeddings, labels_dict,
                                                shuffle_frames=True)
                    t_loader = DataLoader(shuf_train_ds, batch_size=DEFAULT_BATCH_SIZE,
                                          shuffle=True, collate_fn=collate_video_batch)
                    v_loader = DataLoader(shuf_val_ds, batch_size=DEFAULT_BATCH_SIZE,
                                          shuffle=False, collate_fn=collate_video_batch)
                    te_loader = DataLoader(shuf_test_ds, batch_size=DEFAULT_BATCH_SIZE,
                                           shuffle=False, collate_fn=collate_video_batch)
                else:
                    t_loader, v_loader, te_loader = train_loader, val_loader, test_loader

                model = config['model_fn']().to(DEVICE)
                init_head_bias_to_prior(model, pos_rate)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Parameters: {n_params:,} (head bias init -> {prior_logit:.3f})")

                model, history, best_val = train_video_model(
                    model, t_loader, v_loader, config, DEVICE)

                # Evaluate on all splits at subject level
                train_metrics, _, _, _ = evaluate_video_model(model, t_loader, DEVICE)
                val_metrics, val_y, val_p, _ = evaluate_video_model(model, v_loader, DEVICE)
                test_metrics, y_true, y_prob, test_sids = evaluate_video_model(model, te_loader, DEVICE)

                # Optimal-threshold metrics: pick best F1 on val, apply to test
                opt_thr = find_optimal_threshold(val_y, val_p)
                val_metrics_opt = compute_metrics(val_y, val_p, threshold=opt_thr)
                test_metrics_opt = compute_metrics(y_true, y_prob, threshold=opt_thr)
                elapsed = time.time() - start

                all_results[exp_name] = {
                    'config': {k: v for k, v in config.items()
                               if k not in ('model_fn',)},
                    'type': config['type'],
                    'n_params': n_params,
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics,
                    'val_opt_thresh': val_metrics_opt,
                    'test_opt_thresh': test_metrics_opt,
                    'optimal_threshold': opt_thr,
                    'best_val_auroc': best_val,
                    'epochs_trained': len(history['train_loss']),
                    'history': history,
                    'elapsed_sec': round(elapsed, 1),
                }
                _attach_bootstrap_and_save_preds(
                    all_results[exp_name], y_true, y_prob, test_sids)
                _save_checkpoint(model, exp_name)

            elif config['type'] == 'hierarchical':
                model = config['model_fn']().to(DEVICE)
                init_head_bias_to_prior(model, pos_rate)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Parameters: {n_params:,} (head bias init -> {prior_logit:.3f})")

                model, history, best_val = train_hierarchical_model(
                    model, train_subj_ds, val_subj_ds, config, DEVICE)

                train_metrics, _, _, _ = evaluate_hierarchical_model(model, train_subj_ds, DEVICE)
                val_metrics, val_y, val_p, _ = evaluate_hierarchical_model(model, val_subj_ds, DEVICE)
                test_metrics, y_true, y_prob, test_sids = evaluate_hierarchical_model(model, test_subj_ds, DEVICE)

                opt_thr = find_optimal_threshold(val_y, val_p)
                val_metrics_opt = compute_metrics(val_y, val_p, threshold=opt_thr)
                test_metrics_opt = compute_metrics(y_true, y_prob, threshold=opt_thr)
                elapsed = time.time() - start

                all_results[exp_name] = {
                    'config': {k: v for k, v in config.items()
                               if k not in ('model_fn',)},
                    'type': config['type'],
                    'n_params': n_params,
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics,
                    'val_opt_thresh': val_metrics_opt,
                    'test_opt_thresh': test_metrics_opt,
                    'optimal_threshold': opt_thr,
                    'best_val_auroc': best_val,
                    'epochs_trained': len(history['train_loss']),
                    'history': history,
                    'elapsed_sec': round(elapsed, 1),
                }
                _attach_bootstrap_and_save_preds(
                    all_results[exp_name], y_true, y_prob, test_sids)
                _save_checkpoint(model, exp_name)

            # Print summary
            test_m = all_results[exp_name]['test']
            print(f"  => Test AUROC: {test_m['auroc']:.4f}, AUPR: {test_m['aupr']:.4f}, "
                  f"F1@0.5: {test_m['f1']:.4f}, Recall: {test_m['recall']:.4f} "
                  f"({elapsed:.1f}s)")
            if 'test_opt_thresh' in all_results[exp_name]:
                opt = all_results[exp_name]['test_opt_thresh']
                thr = all_results[exp_name]['optimal_threshold']
                print(f"     Test (opt thr={thr:.2f}): F1={opt['f1']:.4f}, "
                      f"Precision={opt['precision']:.4f}, Recall={opt['recall']:.4f}")
            if 'bootstrap_test' in all_results[exp_name]:
                bs = all_results[exp_name]['bootstrap_test']
                print(f"     Bootstrap 95% CI: AUROC [{bs['auroc']['ci_low']:.4f}-{bs['auroc']['ci_high']:.4f}], "
                      f"AUPR [{bs['aupr']['ci_low']:.4f}-{bs['aupr']['ci_high']:.4f}]")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}

        # Incremental checkpoint after every experiment
        _checkpoint_results()

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Save JSON (strip non-serializable items)
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = json.loads(json.dumps(all_results, default=make_serializable))
    with open(OUTPUT_PATH / "results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)

    # Save comparison table
    rows = []
    for name, res in all_results.items():
        if 'error' in res:
            continue
        row = {
            'experiment': name,
            'description': res['config'].get('description', ''),
            'group': res['config'].get('group', ''),
            'type': res['type'],
            'n_params': res.get('n_params', 'N/A'),
            'train_auroc': res['train']['auroc'],
            'val_auroc': res['val']['auroc'],
            'test_auroc': res['test']['auroc'],
            'test_aupr': res['test']['aupr'],
            'test_f1': res['test']['f1'],
            'test_recall': res['test']['recall'],
            'test_precision': res['test']['precision'],
            'epochs': res.get('epochs_trained', 'N/A'),
            'elapsed_sec': res.get('elapsed_sec', 'N/A'),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('test_auroc', ascending=False)
    df.to_csv(OUTPUT_PATH / "comparison_table.csv", index=False)

    # ============================================================
    # GENERATE REPORT
    # ============================================================

    generate_report(all_results, df, OUTPUT_PATH)
    generate_plots(all_results, OUTPUT_PATH)

    print(f"\n{'='*70}")
    print(f"All experiments complete. Results saved to {OUTPUT_PATH}")
    print(f"{'='*70}")

    # Print final summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Test Results (sorted by AUROC)")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'AUROC':>8} {'AUPR':>8} {'F1':>8} {'Recall':>8} {'Params':>10}")
    print("-" * 82)
    for _, row in df.iterrows():
        params = f"{row['n_params']:,}" if row['n_params'] != 'N/A' else 'N/A'
        print(f"{row['experiment']:<30} {row['test_auroc']:>8.4f} {row['test_aupr']:>8.4f} "
              f"{row['test_f1']:>8.4f} {row['test_recall']:>8.4f} {params:>10}")


def generate_plots(all_results, output_path):
    """Generate comparison plots."""
    # Filter successful experiments
    results = {k: v for k, v in all_results.items() if 'error' not in v}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. AUROC comparison bar chart
    ax = axes[0, 0]
    names = []
    aurocs = []
    colors = []
    color_map = {
        'A_recurrent': '#2196F3', 'B_transformer': '#4CAF50',
        'C_cnn': '#FF9800', 'D_hierarchical': '#9C27B0',
        'E_control': '#F44336', 'F_ablation': '#607D8B',
    }
    for name, res in sorted(results.items(), key=lambda x: x[1]['test']['auroc'], reverse=True):
        names.append(name)
        aurocs.append(res['test']['auroc'])
        group = res['config'].get('group', 'other')
        colors.append(color_map.get(group, '#999999'))

    bars = ax.barh(range(len(names)), aurocs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Test AUROC')
    ax.set_title('Test AUROC by Experiment')
    ax.invert_yaxis()

    # 2. AUROC vs AUPR scatter
    ax = axes[0, 1]
    for name, res in results.items():
        group = res['config'].get('group', 'other')
        color = color_map.get(group, '#999999')
        ax.scatter(res['test']['auroc'], res['test']['aupr'], color=color, s=80, alpha=0.8)
        ax.annotate(name, (res['test']['auroc'], res['test']['aupr']),
                     fontsize=6, ha='center', va='bottom')
    ax.set_xlabel('Test AUROC')
    ax.set_ylabel('Test AUPR')
    ax.set_title('AUROC vs AUPR')

    # 3. Training curves (loss) for video models
    ax = axes[1, 0]
    for name, res in results.items():
        if 'history' in res and 'train_loss' in res['history']:
            ax.plot(res['history']['train_loss'], label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(fontsize=6)

    # 4. Validation AUROC curves
    ax = axes[1, 1]
    for name, res in results.items():
        if 'history' in res and 'val_auroc' in res['history']:
            ax.plot(res['history']['val_auroc'], label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUROC')
    ax.set_title('Validation AUROC Curves')
    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path / "temporal_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_results, df, output_path):
    """Generate markdown experiment report."""
    lines = []
    lines.append("# Experiment 05: Temporal Modelling for Video Classification\n")
    lines.append(f"**Date**: {time.strftime('%d %B %Y')}")
    lines.append(f"**Total configurations tested**: {len(all_results)}")
    lines.append(f"**Device**: {DEVICE.upper()}\n")

    lines.append("---\n")
    lines.append("## Motivation\n")
    lines.append("Phase 1 established that frozen DINOv2 embeddings with pooling-based "
                 "aggregation plateau at ~0.83 AUROC. Pooling is order-invariant and cannot "
                 "exploit temporal structure in cardiac ultrasound. This experiment tests "
                 "whether sequence-aware models (that respect frame ordering) can break "
                 "this ceiling.\n")

    lines.append("## Top Results (by Test AUROC)\n")
    lines.append("| Rank | Experiment | Description | Test AUROC | Test AUPR | Test F1 | Params |")
    lines.append("|------|-----------|-------------|-----------|----------|---------|--------|")
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        params = f"{row['n_params']:,}" if row['n_params'] != 'N/A' else 'N/A'
        lines.append(f"| {i+1} | {row['experiment']} | {row['description'][:50]} | "
                     f"**{row['test_auroc']:.4f}** | {row['test_aupr']:.4f} | "
                     f"{row['test_f1']:.4f} | {params} |")

    # Group analysis
    groups = df.groupby('group')
    lines.append("\n## Analysis by Group\n")
    for group_name, group_df in groups:
        lines.append(f"### {group_name}\n")
        lines.append("| Experiment | Test AUROC | Test AUPR | Test F1 | Recall | Precision | Params |")
        lines.append("|-----------|-----------|----------|---------|--------|-----------|--------|")
        for _, row in group_df.iterrows():
            params = f"{row['n_params']:,}" if row['n_params'] != 'N/A' else 'N/A'
            lines.append(f"| {row['experiment']} | {row['test_auroc']:.4f} | {row['test_aupr']:.4f} | "
                         f"{row['test_f1']:.4f} | {row['test_recall']:.4f} | "
                         f"{row['test_precision']:.4f} | {params} |")
        best = group_df.iloc[0]
        worst = group_df.iloc[-1]
        lines.append(f"\n**Best**: {best['experiment']} (AUROC={best['test_auroc']:.4f})")
        lines.append(f"**Range**: {best['test_auroc'] - worst['test_auroc']:.4f}\n")

    # Key findings
    lines.append("## Key Findings\n")
    lines.append("1. **Does temporal modelling beat pooling?** Compare E2 (mean pooling) vs best temporal model")
    lines.append("2. **Does frame order matter?** Compare A1 (BiLSTM) vs E1 (shuffled BiLSTM)")
    lines.append("3. **Self-attention vs recurrence?** Compare B1 (Transformer) vs A1 (BiLSTM)")
    lines.append("4. **Does hierarchy help?** Compare D1/D2 vs flat video-level models")
    lines.append("5. **Positional encoding effect?** Compare B1 (sinusoidal) vs F1a (none) vs F1b (learnable)")
    lines.append("6. **Capacity scaling?** Compare F2a (64) vs F2b (128) vs B1 (256) vs F2c (512)")

    lines.append("\n## Output Files\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `results.json` | Full metrics for every configuration |")
    lines.append("| `comparison_table.csv` | All results in CSV format |")
    lines.append("| `temporal_comparison.png` | Comparison plots |")
    lines.append("| `experiment_report.md` | This report |")

    with open(output_path / "experiment_report.md", 'w') as f:
        f.write('\n'.join(lines))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Exp05: Temporal Modelling for Video Classification'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run subset of experiments for testing')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Run specific experiments by name (e.g., A1_bilstm B1_transformer_2L)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--embeddings-path', type=str, default=None,
                        help='Override raw embeddings path')
    parser.add_argument('--resume', action='store_true',
                        help='Skip experiments already present in results.json')
    args = parser.parse_args()

    if args.output_dir:
        OUTPUT_PATH = Path(args.output_dir)
    if args.embeddings_path:
        RAW_EMBEDDINGS_PATH = Path(args.embeddings_path)

    run_all_experiments(quick=args.quick, selected=args.experiments,
                        embeddings_path=args.embeddings_path, resume=args.resume)
