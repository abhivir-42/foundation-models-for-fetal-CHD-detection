#!/usr/bin/env python3
"""
Experiment 06: Tier 2 — Temporal Heads on FetalCLIP Embeddings.

Re-trains the Tier 1 winning architectures on FetalCLIP ViT-L/14 embeddings
in place of DINOv2. Same heads, same splits, same hyperparameters — the
only varying axis is the frozen frame backbone. This isolates the
backbone-quality contribution of the FetalCLIP foundation model
(arxiv 2502.14807) on iFIND CHD classification.

Design notes:
  - 5 archs (B2, F2a, F2b, C1, E2_baseline) — Tier 1 winners.
  - Optional L2-per-frame normalisation matches the FetalCLIP paper's
    linear-probe protocol; default 'none' for parity with Tier 1.
  - JSON schema identical to exp05 results.json so per_condition / forest /
    calibration tooling works unchanged.

Usage:
  python exp06_fetalclip_modelling.py --execute            # full Tier 2 run
  python exp06_fetalclip_modelling.py --execute --norm l2_per_frame
  python exp06_fetalclip_modelling.py --execute --experiments B2_cross_attn
  python exp06_fetalclip_modelling.py --dry-run            # print plan only
"""

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Re-use everything from exp05. The wrapper does not redefine models.
import exp05_temporal_modelling as t1


# ============================================================
# CONFIGURATION
# ============================================================

from paths import FETALCLIP_EMBEDDINGS, RESULTS_DIR

FETALCLIP_EMBEDDINGS_PATH = FETALCLIP_EMBEDDINGS
OUTPUT_PATH = RESULTS_DIR / "exp06_fetalclip"

# Tier 1 winners — see plan §1 for justification.
TIER2_SUBSET = [
    "B2_cross_attn",
    "F2a_hidden64",
    "F2b_hidden128",
    "C1_temporal_cnn",
    "E2_mean_pool_baseline",
]


# ============================================================
# FETALCLIP-SPECIFIC PREPROCESSING
# ============================================================

def l2_normalise_per_frame(raw_embeddings):
    """L2-normalise each frame's 768-d vector independently.

    Matches the FetalCLIP linear-probe protocol (open_clip / Radford et al.
    convention: features sit on the unit sphere before the classifier).
    Mutates a copy — the input dict is not touched.
    """
    out = {}
    for sid, videos in raw_embeddings.items():
        new_videos = []
        for v in videos:
            if v is None or v.shape[0] == 0:
                new_videos.append(v)
                continue
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            new_videos.append((v / norms).astype(np.float32))
        out[sid] = new_videos
    return out


def sanity_check_clips_per_subject(fc_embeddings, dinov2_path=None):
    """Plan §8 risk 2: bail if mean clips/subject differs >5% from DINOv2."""
    fc_clips = np.mean([
        len([v for v in vs if v is not None and v.shape[0] > 0])
        for vs in fc_embeddings.values()
    ])
    print(f"FetalCLIP mean clips/subject: {fc_clips:.2f}")
    if dinov2_path is None or not Path(dinov2_path).exists():
        return
    with open(dinov2_path, "rb") as f:
        d2 = pickle.load(f)
    d2_clips = np.mean([
        len([v for v in vs if v is not None and v.shape[0] > 0])
        for vs in d2.values()
    ])
    print(f"DINOv2 mean clips/subject:    {d2_clips:.2f}")
    delta = abs(fc_clips - d2_clips) / max(d2_clips, 1e-6)
    if delta > 0.05:
        raise RuntimeError(
            f"clips/subject differ by {delta:.1%} between backbones — "
            "aggregation step would not be apples-to-apples"
        )


# ============================================================
# CONFIG TAGGING
# ============================================================

def tag_configs_with_backbone(configs, norm_mode):
    """Tag every config dict with FetalCLIP backbone metadata.

    The downstream tooling reads `result['config']` — adding here means the
    extra keys round-trip through results.json without further changes.
    """
    for name, cfg in configs.items():
        cfg["embedding_backbone"] = "FetalCLIP"
        cfg["embedding_norm"] = norm_mode
    return configs


# ============================================================
# RUNNER
# ============================================================

def run_tier2(experiments=None, norm_mode="none", dry_run=False, resume=False):
    """Tier 2 runner. Delegates to exp05 after backbone-specific setup."""
    selected = experiments or TIER2_SUBSET
    print(f"Tier 2 (FetalCLIP) — running {len(selected)} archs:")
    for s in selected:
        print(f"  - {s}")
    print(f"Embedding norm: {norm_mode}")
    print(f"Output dir:     {OUTPUT_PATH}")

    if dry_run:
        print("\n[dry-run] no execution. Re-run with --execute.")
        return

    # Redirect exp05's globals so the existing runner writes to the right place.
    t1.RAW_EMBEDDINGS_PATH = FETALCLIP_EMBEDDINGS_PATH
    t1.OUTPUT_PATH = OUTPUT_PATH

    # If L2-per-frame is requested we need to load + transform up front and
    # then pass a path to a transformed pkl OR monkey-patch t1.load_data.
    # Cleanest is monkey-patch: we keep the disk pkl untouched.
    if norm_mode == "l2_per_frame":
        original_load_data = t1.load_data

        def _load_data_l2(embeddings_path=None):
            raw, labels_df, tr, va, te = original_load_data(embeddings_path)
            print("Applying L2-per-frame normalisation...")
            raw = l2_normalise_per_frame(raw)
            return raw, labels_df, tr, va, te

        t1.load_data = _load_data_l2
    elif norm_mode != "none":
        raise ValueError(f"unknown norm mode: {norm_mode}")

    # Mutate the config dict before exp05 sees it.
    original_get_configs = t1.get_experiment_configs

    def _get_configs_tier2():
        cfgs = original_get_configs()
        cfgs = OrderedDict((k, v) for k, v in cfgs.items() if k in selected)
        return tag_configs_with_backbone(cfgs, norm_mode)

    t1.get_experiment_configs = _get_configs_tier2

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Optional sanity check (cheap; ~1s).
    with open(FETALCLIP_EMBEDDINGS_PATH, "rb") as f:
        fc = pickle.load(f)
    from paths import DINOV2_EMBEDDINGS
    sanity_check_clips_per_subject(
        fc,
        dinov2_path=str(DINOV2_EMBEDDINGS),
    )
    del fc

    t1.run_all_experiments(
        quick=False,
        selected=selected,
        embeddings_path=str(FETALCLIP_EMBEDDINGS_PATH),
        resume=resume,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp06: Tier 2 — FetalCLIP backbone")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually run training. Without this flag, prints the plan and exits."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Alias for absence of --execute (explicit form)."
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Subset of TIER2_SUBSET to run (default: all 5)."
    )
    parser.add_argument(
        "--norm", choices=["none", "l2_per_frame"], default="none",
        help="Frame embedding preprocessing. 'none' = parity with Tier 1; "
             "'l2_per_frame' = FetalCLIP linear-probe convention."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip experiments already in results.json."
    )
    args = parser.parse_args()

    run_tier2(
        experiments=args.experiments,
        norm_mode=args.norm,
        dry_run=(args.dry_run or not args.execute),
        resume=args.resume,
    )
