"""Small helpers for control tensor buffering and assignment.

This module centralises the control cache helpers so both the interactive
app and the headless benchmark use the exact same semantics and memory
layout for (B, C, F) control tensors.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import utils


def _control_view(cache: Dict[str, np.ndarray], key: str, frames: int) -> np.ndarray:
    """Return a view into the cached BCF buffer for ``key`` sized to ``frames``.

    If the cache does not contain a suitably-sized buffer it is (re)allocated
    with a power-of-two frame capacity to avoid frequent reallocations.
    """
    view = cache.get(key)
    if view is None or view.shape[2] < frames:
        new_frames = 1 << max(0, frames - 1).bit_length()
        view = np.zeros((1, 1, new_frames), dtype=utils.RAW_DTYPE)
        cache[key] = view
    return view[:, :, :frames]


def _assign_control(
    cache: Dict[str, np.ndarray], key: str, frames: int, value: float | np.ndarray
) -> np.ndarray:
    """Fill or copy ``value`` into the cache for ``key`` and return a (1,1,frames) view.

    Accepts the same value shapes the rest of the graph expects: scalar,
    1-D per-frame arrays, or a pre-formed (1,1,N) BCF buffer.
    """
    view = _control_view(cache, key, frames)
    array = np.asarray(value, dtype=utils.RAW_DTYPE)
    if array.ndim == 0:
        view.fill(float(array))
        return view
    if array.ndim == 1:
        if array.shape[0] != frames:
            raise ValueError(f"{key}: expected {frames} samples, got {array.shape[0]}")
        view[0, 0, :frames] = array
        return view
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] == 1 and array.shape[2] >= frames:
        view[...] = array[:, :, :frames]
        return view
    raise ValueError(f"Unsupported control shape for '{key}': {array.shape}")
