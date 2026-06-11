"""Central path configuration for the pipeline.

All filesystem locations are resolved from environment variables so the code
runs on any machine without editing source. Set these before running any
extraction, training, or evaluation script (see REPRODUCE.md):

  IFIND_DATA       Root of the iFIND fetal-cardiac corpus and the
                   subject_level_labels.csv label file. The raw imaging data is
                   restricted (see DATA.md) and is required only for embedding
                   extraction and training from scratch.
  EMBEDDINGS_DIR   Where cached per-subject embedding .pkl files are written and
                   read. Defaults to <repo>/embeddings.
  RESULTS_DIR      Where experiment outputs (per-fold results, intermediate
                   CSVs) are written and read. Defaults to <repo>/results.
  PREDICTIONS_DIR  Where the committed per-fold prediction CSVs live. Defaults
                   to <repo>/predictions, which ships with the repository, so
                   figures and headline numbers regenerate without the raw data.

Nothing here points at any particular user's filesystem; defaults are relative
to the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


# Restricted iFIND corpus root (raw videos + subject_level_labels.csv).
IFIND_DATA = _env_path("IFIND_DATA", REPO_ROOT / "data")

# Cached embeddings, experiment outputs, and committed predictions.
EMBEDDINGS_DIR = _env_path("EMBEDDINGS_DIR", REPO_ROOT / "embeddings")
RESULTS_DIR = _env_path("RESULTS_DIR", REPO_ROOT / "results")
PREDICTIONS_DIR = _env_path("PREDICTIONS_DIR", REPO_ROOT / "predictions")

# Convenience handles for the canonical label file and embedding caches.
LABELS_CSV = IFIND_DATA / "subject_level_labels.csv"
DINOV2_EMBEDDINGS = EMBEDDINGS_DIR / "raw_video_embeddings_clean.pkl"
FETALCLIP_EMBEDDINGS = EMBEDDINGS_DIR / "raw_video_embeddings_fetalclip.pkl"
