"""Reusable CFFI-backed memory blocks for graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .state import RAW_DTYPE
from . import c_kernels


@dataclass(slots=True)
class _Block:
    """Opaque memory region backing a ``float64`` numpy array."""

    size: int
    buffer: object
    array: np.ndarray

    def view(self, shape: Sequence[int]) -> np.ndarray:
        """Return a C-contiguous ``np.ndarray`` view of ``shape``."""

        total = int(np.prod(shape))
        if total > self.size:
            raise ValueError(
                f"Requested view of {total} elements exceeds block capacity {self.size}"
            )
        return self.array[:total].reshape(tuple(shape))


class BlockLease:
    """Represents a checked-out block view returned to a node."""

    __slots__ = ("_pool", "block", "view", "tag")

    def __init__(self, pool: "BlockPool", block: _Block, view: np.ndarray, tag: str) -> None:
        self._pool = pool
        self.block = block
        self.view = view
        self.tag = tag

    def release(self) -> None:
        self._pool.release(self.block)


class BlockPool:
    """Allocator that dispenses reusable float64 blocks backed by CFFI memory."""

    __slots__ = ("_free", "_dtype", "_itemsize")

    def __init__(self) -> None:
        self._dtype = np.dtype(RAW_DTYPE)
        self._itemsize = int(self._dtype.itemsize)
        self._free: Dict[int, List[_Block]] = {}

    def _allocate(self, size: int) -> _Block:
        size = int(size)
        if size <= 0:
            raise ValueError("Block size must be positive")
        ffi = getattr(c_kernels, "ffi", None)
        if ffi is not None:
            buf = ffi.new(f"double[{size}]")
            mem = ffi.buffer(buf, size * self._itemsize)
            array = np.frombuffer(mem, dtype=self._dtype)
        else:  # pragma: no cover - fallback
            array = np.empty(size, dtype=self._dtype)
            buf = array
        array = array.reshape((size,), order="C")
        return _Block(size=size, buffer=buf, array=array)

    def acquire(self, shape: Sequence[int], *, tag: str = "default") -> BlockLease:
        total = int(np.prod(shape))
        blocks = self._free.get(total)
        if blocks:
            block = blocks.pop()
        else:
            block = self._allocate(total)
        view = block.view(shape)
        return BlockLease(self, block, view, tag)

    def release(self, block: _Block) -> None:
        self._free.setdefault(block.size, []).append(block)


__all__ = ["BlockPool", "BlockLease"]

