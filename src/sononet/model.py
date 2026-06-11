"""
Clean PyTorch port of canonical SonoNet32 (Baumgartner et al. 2017).

This mirrors the canonical Lasagne reference at:
    https://github.com/baumgach/SonoNet-weights/blob/master/models.py

An earlier lab port is a separate implementation that diverges from the
canonical in three ways and is NOT loadable from canonical weights:

  1. Missing `pad=1` on every 3x3 conv (canonical uses "same" padding)
  2. Final pool is `nn.AvgPool2d(2,2)` instead of global average pool
  3. Extra ReLU after the final classification BN — the canonical's final
     conv6_p has `nonlinearity=linear`, so the activation is identity

This file deliberately re-implements the architecture to match the canonical
exactly, so the public Apache-2.0 weights at github.com/baumgach/SonoNet-weights
load and produce numerically equivalent outputs.

Architecture (SonoNet32):
  Block 1: Conv(1->32, 3x3, pad=1) + BN + ReLU
           Conv(32->32, 3x3, pad=1) + BN + ReLU
           MaxPool(2x2)
  Block 2: Conv(32->64, 3x3, pad=1) + BN + ReLU
           Conv(64->64, 3x3, pad=1) + BN + ReLU
           MaxPool(2x2)
  Block 3: Conv(64->128, 3x3, pad=1) + BN + ReLU  x3
           MaxPool(2x2)
  Block 4: Conv(128->256, 3x3, pad=1) + BN + ReLU x3
           MaxPool(2x2)
  Block 5: Conv(256->256, 3x3, pad=1) + BN + ReLU x3
           (no pool)
  Adapt:   Conv(256->128, 1x1) + BN + ReLU
           Conv(128->num_labels, 1x1) + BN  (NO ReLU — linear nonlinearity)
  Output:  AdaptiveAvgPool2d(1)  -> reshape to (B, num_labels)

Total: 15 conv layers (each wrapped in BN), matches the 75 / 5 = 15 blocks
in the canonical .npz checkpoints.

Note on BN epsilon: Lasagne's default is 1e-4 (vs PyTorch's 1e-5). The .npz
stores `inv_std = 1 / sqrt(var + 1e-4)`. We absorb that by setting
`running_var = 1 / inv_std**2` and `eps=0` in nn.BatchNorm2d. The forward
computation then matches Lasagne exactly:
    out = gamma * (x - mean) * inv_std + beta
"""
from typing import List

import torch
import torch.nn as nn


# Canonical 14 SonoNet output classes, in order, per the canonical example.py
# (https://github.com/baumgach/SonoNet-weights/blob/master/example.py).
# This list is THE ground truth ordering for arr_70 (the final classifier head).
LABEL_NAMES: List[str] = [
    "3VV",
    "4CH",
    "Abdominal",
    "Background",
    "Brain (Cb.)",
    "Brain (Tv.)",
    "Femur",
    "Kidneys",
    "Lips",
    "LVOT",
    "Profile",
    "RVOT",
    "Spine (cor.)",
    "Spine (sag.)",
]


def _conv_bn_relu(in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1) -> List[nn.Module]:
    """Conv (no bias) + BN (eps=0) + ReLU. Matches `batch_norm(Conv2DLayer(...))`."""
    return [
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_c, eps=0.0),
        nn.ReLU(inplace=True),
    ]


def _conv_bn_linear(in_c: int, out_c: int, kernel_size: int = 1) -> List[nn.Module]:
    """Conv (no bias) + BN (eps=0) — NO activation. For the final classification head."""
    return [
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_c, eps=0.0),
    ]


class SonoNet32(nn.Module):
    """
    Faithful PyTorch port of canonical SonoNet32.

    Args:
        num_labels: number of output classes. Default 14 to match the canonical
            view classifier checkpoint at SonoNet32.npz.
    """

    def __init__(self, num_labels: int = 14) -> None:
        super().__init__()
        self.num_labels = num_labels

        layers: List[nn.Module] = []

        # Block 1: 32, 32, pool
        layers += _conv_bn_relu(1, 32)
        layers += _conv_bn_relu(32, 32)
        layers += [nn.MaxPool2d(2, ceil_mode=False)]

        # Block 2: 64, 64, pool
        layers += _conv_bn_relu(32, 64)
        layers += _conv_bn_relu(64, 64)
        layers += [nn.MaxPool2d(2, ceil_mode=False)]

        # Block 3: 128, 128, 128, pool
        layers += _conv_bn_relu(64, 128)
        layers += _conv_bn_relu(128, 128)
        layers += _conv_bn_relu(128, 128)
        layers += [nn.MaxPool2d(2, ceil_mode=False)]

        # Block 4: 256, 256, 256, pool
        layers += _conv_bn_relu(128, 256)
        layers += _conv_bn_relu(256, 256)
        layers += _conv_bn_relu(256, 256)
        layers += [nn.MaxPool2d(2, ceil_mode=False)]

        # Block 5: 256, 256, 256 (no pool)
        layers += _conv_bn_relu(256, 256)
        layers += _conv_bn_relu(256, 256)
        layers += _conv_bn_relu(256, 256)

        # Adaption: 256 -> 128 (1x1 + BN + ReLU)
        layers += _conv_bn_relu(256, 128, kernel_size=1, padding=0)

        # Final classification: 128 -> num_labels (1x1 + BN, NO ReLU)
        layers += _conv_bn_linear(128, num_labels, kernel_size=1)

        # Global average pool to (B, num_labels, 1, 1)
        layers += [nn.AdaptiveAvgPool2d(1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (B, num_labels)."""
        out = self.net(x)
        return out.view(x.shape[0], self.num_labels)
