"""Reusable CFFI-backed memory blocks for graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .state import RAW_DTYPE
from . import c_kernels


@dataclass(slots=True)
class _NodeBuffer:
    """Opaque C-ready memory region backing a ``float64`` numpy array for node buffers."""

    size: int
    c_buffer: object
    array: np.ndarray

    def view(self, shape: Sequence[int]) -> np.ndarray:
        """Return a C-contiguous node buffer (np.ndarray) view of ``shape``."""

        total = int(np.prod(shape))
        if total > self.size:
            raise ValueError(
                f"Requested view of {total} elements exceeds node buffer capacity {self.size}"
            )
        return self.array[:total].reshape(tuple(shape))


class BlockLease:
    """Represents a checked-out node buffer view returned to a node."""

    __slots__ = ("_pool", "node_buffer", "view", "tag")

    def __init__(self, pool: "BlockPool", node_buffer: _NodeBuffer, view: np.ndarray, tag: str) -> None:
        self._pool = pool
        self.node_buffer = node_buffer
        self.view = view
        self.tag = tag

    def release(self) -> None:
        self._pool.release(self.node_buffer)


class BlockPool:
    """Allocator that dispenses reusable C-ready node buffers (float64) backed by CFFI memory."""

    __slots__ = ("_free", "_dtype", "_itemsize")

    def __init__(self) -> None:
        self._dtype = np.dtype(RAW_DTYPE)
        self._itemsize = int(self._dtype.itemsize)
        self._free: Dict[int, List[_NodeBuffer]] = {}

    def _allocate(self, size: int) -> _NodeBuffer:
        size = int(size)
        if size <= 0:
            raise ValueError("Node buffer size must be positive")
        ffi = getattr(c_kernels, "ffi", None)
        if ffi is not None:
            c_buf = ffi.new(f"double[{size}]")
            mem = ffi.buffer(c_buf, size * self._itemsize)
            array = np.frombuffer(mem, dtype=self._dtype)
        else:  # pragma: no cover - fallback
            array = np.empty(size, dtype=self._dtype)
            c_buf = array
        array = array.reshape((size,), order="C")
        return _NodeBuffer(size=size, c_buffer=c_buf, array=array)

    def acquire(self, shape: Sequence[int], *, tag: str = "default") -> BlockLease:
        total = int(np.prod(shape))
        node_buffers = self._free.get(total)
        if node_buffers:
            node_buffer = node_buffers.pop()
        else:
            node_buffer = self._allocate(total)
        view = node_buffer.view(shape)
        return BlockLease(self, node_buffer, view, tag)

    def release(self, node_buffer: _NodeBuffer) -> None:
        self._free.setdefault(node_buffer.size, []).append(node_buffer)


__all__ = ["BlockPool", "BlockLease"]

