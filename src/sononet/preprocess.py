"""
Preprocessing for SonoNet inference, matching the canonical example.py exactly.

The canonical pipeline (from github.com/baumgach/SonoNet-weights/example.py):

  1. Read image
  2. Crop with crop_range = [(115, 734), (81, 874)] to strip vendor info / borders
     (this is specific to the iFIND test images — see below for the cardiac case)
  3. Resize to (224, 288) — note: scipy.misc.imresize takes (rows, cols), so
     224 is height and 288 is width
  4. Convert to grayscale via np.mean(image, axis=2) for RGB inputs
  5. Per-image normalisation: 255.0 * (image - mean) / std
     where mean and std are computed PER-IMAGE (not from a dataset), and the
     255.0 is an arbitrary scale factor that the network was trained with

For our fetal cardiac dataset (Doppler-filtered .mp4 clips), we extract frames
with cv2.VideoCapture and apply steps 3-5. The crop step is specific to the
TIFF test images and is NOT applied to our clips. Frames from clinical clips
have already been cropped to the ultrasound cone region by the acquisition
software, so an additional crop would lose useful pixels.
"""
from typing import Optional, Tuple

import numpy as np

# Canonical SonoNet input size (rows, cols).
SONONET_INPUT_SIZE: Tuple[int, int] = (224, 288)

# Canonical crop applied to the iFIND test TIFFs in the SonoNet release.
# Format: [(top, bottom), (left, right)] — same convention as example.py.
CANONICAL_TIFF_CROP = [(115, 734), (81, 874)]


def _resize_bilinear(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear resize matching scipy.misc.imresize defaults (the canonical preprocessing).

    scipy.misc.imresize is gone in modern scipy; we replace it with PIL bilinear,
    which is what scipy <=1.0 wrapped under the hood.
    """
    from PIL import Image

    target_h, target_w = target_hw
    if arr.dtype != np.uint8:
        # PIL Image.fromarray expects uint8 for L mode; we cast and rescale if needed.
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3:
        mode = "RGB"
    else:
        mode = "L"
    img = Image.fromarray(arr, mode=mode)
    # PIL Image.resize takes (W, H) — note the swap from numpy/scipy (H, W).
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(img)


def preprocess_frame_for_sononet(
    frame: np.ndarray,
    crop_range: Optional[list] = None,
    input_size: Tuple[int, int] = SONONET_INPUT_SIZE,
) -> np.ndarray:
    """
    Apply the canonical SonoNet preprocessing to a single frame.

    Args:
        frame: input image as a numpy array. Either (H, W) grayscale or (H, W, 3) RGB.
            Pixel values can be uint8 [0, 255] or float in [0, 1] / [0, 255] —
            this function normalises internally.
        crop_range: optional crop in the format [(top, bottom), (left, right)].
            Pass `CANONICAL_TIFF_CROP` for the iFIND test TIFFs; pass None for
            clinical clip frames (already cropped to the ultrasound cone).
        input_size: (height, width) tuple. Default is canonical (224, 288).

    Returns:
        A float32 numpy array of shape (1, 1, H, W) — NCHW with batch size 1
        and 1 input channel — ready to feed into a SonoNet PyTorch model.
    """
    img = frame
    if crop_range is not None:
        img = img[crop_range[0][0]:crop_range[0][1], crop_range[1][0]:crop_range[1][1], ...]

    img = _resize_bilinear(img, input_size)

    # Grayscale conversion. The canonical uses np.mean(image, axis=2) which
    # is the unweighted average of R, G, B — different from luma but matches
    # what the model was trained on.
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    # Reshape to NCHW float32
    img = img.astype(np.float32).reshape(1, 1, img.shape[0], img.shape[1])

    # Per-image normalisation matching example.py:
    #   image_data = 255.0 * (image_data - mean) / std
    # mean and std computed over THIS image's pixel values.
    mean = img.mean()
    std = img.std()
    if std < 1e-6:
        # Degenerate input (uniform image). Avoid divide-by-zero — return
        # a tensor of zeros which will produce a near-uniform softmax downstream.
        return np.zeros_like(img)
    img = 255.0 * (img - mean) / std

    return img.astype(np.float32)
