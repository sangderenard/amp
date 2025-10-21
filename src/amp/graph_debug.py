"""Bridge utilities for exporting audio graph metadata to C tooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import struct
from typing import Any

from .graph import AudioGraph

try:  # pragma: no cover - optional dependency
    import cffi
except Exception:  # pragma: no cover - optional dependency
    cffi = None

try:  # pragma: no cover - platform dependent
    from multiprocessing import shared_memory
except Exception:  # pragma: no cover - platform dependent
    shared_memory = None

_GRAPH_DEBUG_HEADER = struct.Struct("<IIII")

_CDEF = """
    typedef struct {
        uint32_t version;
        uint32_t node_bytes;
        uint32_t control_bytes;
        uint32_t reserved;
    } GraphDebugHeader;

    typedef struct {
        uint32_t type_id;
        uint32_t name_len;
        uint32_t type_len;
        uint32_t audio_input_count;
        uint32_t mod_input_count;
        uint32_t param_buffer_count;
        uint32_t buffer_shape_count;
        uint32_t params_json_len;
    } NodeDescriptorHeader;

    typedef struct {
        uint32_t source_len;
        uint32_t param_len;
        uint32_t mode_code;
        float scale;
        int32_t channel;
    } NodeModInputHeader;

    typedef struct {
        uint32_t name_len;
        uint32_t batches;
        uint32_t channels;
        uint32_t frames;
        uint64_t byte_len;
    } NodeParamBufferHeader;

    typedef struct {
        uint32_t key_len;
        uint32_t rank;
        uint64_t byte_len;
    } ControlSampleExtraHeader;

    typedef struct {
        uint32_t frames;
        uint32_t pitch_dim;
        uint32_t envelope_dim;
        uint32_t extras_count;
        uint32_t pcm_channels;
        double start_time;
        double update_hz;
        uint64_t times_offset;
        uint64_t pitch_offset;
        uint64_t envelope_offset;
        uint64_t control_offset;
        uint64_t pcm_offset;
        uint64_t extras_offset;
        uint64_t total_size;
    } ControlSampleHeader;
"""


@dataclass(slots=True)
class SharedMemoryHandle:
    """Descriptor returned when a bundle is written to shared memory."""

    name: str
    size: int


class GraphDebugBridge:
    """Emit binary graph descriptors consumable by C tooling via cffi."""

    VERSION = 1

    def __init__(self) -> None:
        if cffi is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("cffi is required to build the graph debug bridge")
        self.ffi = cffi.FFI()
        self.ffi.cdef(_CDEF)

    def build_bundle(
        self,
        graph: AudioGraph,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
    ) -> bytes:
        """Return a packed binary bundle combining node and control descriptors."""

        node_blob = graph.serialize_node_descriptors()
        control_blob = graph.control_delay.export_sample_block(
            start_time, frames, update_hz=update_hz
        )
        header = _GRAPH_DEBUG_HEADER.pack(self.VERSION, len(node_blob), len(control_blob), 0)
        return header + node_blob + control_blob

    def _as_c_buffer(self, payload: bytes) -> Any:
        size = len(payload)
        c_buf = self.ffi.new(f"char[{size}]")
        self.ffi.memmove(c_buf, payload, size)
        return c_buf

    def write_file(
        self,
        path: os.PathLike[str] | str,
        graph: AudioGraph,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
    ) -> str:
        """Write the bundle to ``path`` for offline inspection."""

        payload = self.build_bundle(graph, start_time, frames, update_hz=update_hz)
        c_buf = self._as_c_buffer(payload)
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as handle:
            handle.write(self.ffi.buffer(c_buf)[:])
        return str(file_path)

    def write_shared_memory(
        self,
        graph: AudioGraph,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
        name: str | None = None,
    ) -> SharedMemoryHandle:
        """Store the bundle in shared memory for live C inspection."""

        if shared_memory is None:  # pragma: no cover - optional dependency
            raise RuntimeError("Shared memory is not available on this platform")
        payload = self.build_bundle(graph, start_time, frames, update_hz=update_hz)
        c_buf = self._as_c_buffer(payload)
        shm = shared_memory.SharedMemory(create=True, size=len(payload), name=name)
        view = shm.buf
        try:
            shm_ptr = self.ffi.from_buffer(view)
            self.ffi.memmove(shm_ptr, c_buf, len(payload))
        finally:
            view.release()
        return SharedMemoryHandle(name=shm.name, size=len(payload))


__all__ = ["GraphDebugBridge", "SharedMemoryHandle"]

