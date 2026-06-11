"""
Load canonical Lasagne SonoNet .npz weights into the PyTorch port.

The canonical .npz files (SonoNet16/32/64.npz) at github.com/baumgach/SonoNet-weights
are produced by `lasagne.layers.get_all_param_values(net['output'])`. For SonoNet32
this returns 75 arrays in 15 conv-BN blocks of 5 arrays each. The 5-array layout
per block is:

    arr_{5n+0}: Conv W,        shape (out, in, kh, kw)
    arr_{5n+1}: BN beta,       shape (out,)   <- nn.BatchNorm2d.bias
    arr_{5n+2}: BN gamma,      shape (out,)   <- nn.BatchNorm2d.weight
    arr_{5n+3}: BN running_mean, shape (out,) <- nn.BatchNorm2d.running_mean
    arr_{5n+4}: BN inv_std,    shape (out,)   <- 1 / sqrt(var + eps_lasagne=1e-4)

Notes on the BN parameter ordering: Lasagne `BatchNormLayer` adds parameters
in the order beta, gamma, mean, inv_std (see lasagne/layers/normalization.py).
The conv bias is removed by the `batch_norm()` wrapper, so each conv block
contributes exactly 1 + 4 = 5 arrays to the .npz.

We absorb the Lasagne epsilon by storing `running_var = 1 / inv_std**2` and
setting `eps=0.0` in the PyTorch BatchNorm2d (see model.py for the math).
"""
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .model import SonoNet32


def load_npz_weights(model: SonoNet32, npz_path: Union[str, Path]) -> SonoNet32:
    """
    Load a canonical SonoNet .npz checkpoint into the given PyTorch SonoNet32.

    Args:
        model: an unloaded SonoNet32 instance with matching num_labels.
        npz_path: path to SonoNet32.npz (or compatible).

    Returns:
        The same model with weights loaded in-place.

    Raises:
        AssertionError on shape mismatch (defensive — fail loud, not silent).
    """
    npz_path = Path(npz_path)
    f = np.load(npz_path)
    arrays = [f["arr_%d" % i] for i in range(len(f.files))]
    f.close()

    # Walk the model in forward order, picking out (Conv2d, BatchNorm2d) pairs.
    modules = list(model.net.children())
    conv_bn_pairs = []
    i = 0
    while i < len(modules):
        if isinstance(modules[i], nn.Conv2d):
            assert i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm2d), (
                f"Expected BatchNorm2d after Conv2d at module index {i}; "
                f"got {type(modules[i + 1]).__name__ if i + 1 < len(modules) else 'EOF'}. "
                f"This usually means the model architecture has been edited."
            )
            conv_bn_pairs.append((modules[i], modules[i + 1]))
            i += 2
        else:
            i += 1

    expected_blocks = len(arrays) // 5
    assert len(conv_bn_pairs) == expected_blocks, (
        f"Block count mismatch: model has {len(conv_bn_pairs)} conv-BN pairs, "
        f".npz has {len(arrays)} arrays = {expected_blocks} blocks. "
        f"For SonoNet32 both should be 15."
    )
    assert len(arrays) % 5 == 0, (
        f".npz has {len(arrays)} arrays, not divisible by 5. "
        f"Expected the canonical 5-arrays-per-block layout."
    )

    for block_idx, (conv, bn) in enumerate(conv_bn_pairs):
        base = block_idx * 5
        W = torch.from_numpy(arrays[base + 0])
        beta = torch.from_numpy(arrays[base + 1])
        gamma = torch.from_numpy(arrays[base + 2])
        mean = torch.from_numpy(arrays[base + 3])
        inv_std = torch.from_numpy(arrays[base + 4])

        assert tuple(conv.weight.shape) == tuple(W.shape), (
            f"Block {block_idx} conv weight shape mismatch: "
            f"PyTorch model expects {tuple(conv.weight.shape)}, .npz has {tuple(W.shape)}. "
            f"Check architecture (channel widths, kernel sizes, padding)."
        )
        assert tuple(bn.weight.shape) == tuple(gamma.shape) == (W.shape[0],), (
            f"Block {block_idx} BN shape mismatch."
        )

        with torch.no_grad():
            conv.weight.copy_(W)
            # PyTorch BN: weight=gamma (scale), bias=beta (shift). Lasagne stores
            # them in the opposite order in the .npz, so we map by name not position.
            bn.weight.copy_(gamma)
            bn.bias.copy_(beta)
            bn.running_mean.copy_(mean)
            # Convert inv_std -> running_var. Combined with eps=0.0 in BN this
            # reproduces Lasagne's forward exactly.
            bn.running_var.copy_(1.0 / (inv_std ** 2))
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked.fill_(0)

    return model
