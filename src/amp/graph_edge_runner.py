from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence
import struct

import numpy as np

from .graph import (
    AudioGraph,
    ModConnection,
    RAW_DTYPE,
    _MODE_CODES,
    _NODE_DESCRIPTOR_HEADER,
    _MOD_DESCRIPTOR_HEADER,
    _PARAM_DESCRIPTOR_HEADER,
    _assert_bcf,
)

try:  # pragma: no cover - optional dependency
    import cffi
except Exception:  # pragma: no cover - optional dependency
    cffi = None


_MODE_NAMES = {code: name for name, code in _MODE_CODES.items()}


@dataclass(slots=True)
class _ParsedNodeDescriptor:
    name: str
    audio_inputs: tuple[str, ...]
    mod_groups: tuple[tuple[str, tuple[ModConnection, ...]], ...]


@dataclass(slots=True)
class NodeInputHandle:
    """Holds the CFFI view for a node's C-ready input buffers."""

    node: str
    ffi: Any
    cdata: Any
    node_buffer: np.ndarray | None  # C-ready node buffer for audio inputs
    batches: int
    channels: int
    frames: int
    param_names: tuple[str, ...]
    param_buffers: tuple[np.ndarray, ...]  # C-ready parameter buffers
    keepers: tuple[Any, ...]

    @property
    def params(self) -> Dict[str, np.ndarray]:
        """Return a mapping of parameter names to their C-ready buffers."""
        return {name: buf for name, buf in zip(self.param_names, self.param_buffers)}


_EDGE_RUNNER_CDEF = """
    typedef struct {
        uint32_t has_audio;
        uint32_t batches;
        uint32_t channels;
        uint32_t frames;
        const double *data;
    } EdgeRunnerAudioView;

    typedef struct {
        const char *name;
        uint32_t batches;
        uint32_t channels;
        uint32_t frames;
        const double *data;
    } EdgeRunnerParamView;

    typedef struct {
        uint32_t count;
        EdgeRunnerParamView *items;
    } EdgeRunnerParamSet;

    typedef struct {
        EdgeRunnerAudioView audio;
        EdgeRunnerParamSet params;
    } EdgeRunnerNodeInputs;
"""


class CffiEdgeRunner:
    """Prepares C-ready node input buffers using the C-formatted graph descriptors."""

    def __init__(self, graph: AudioGraph) -> None:
        if cffi is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("cffi is required to build the graph edge runner")
        self._graph = graph
        self.ffi = cffi.FFI()
        self.ffi.cdef(_EDGE_RUNNER_CDEF)
        self._node_descriptors: tuple[_ParsedNodeDescriptor, ...] = ()
        self._descriptor_by_name: dict[str, _ParsedNodeDescriptor] = {}
        self._ordered_names: tuple[str, ...] = ()
        self._frames: int = 0
        self._sample_rate: float = float(graph.sample_rate)
        self._base_params: Dict[str, Dict[str, np.ndarray]] = {}
        self._caches: Dict[str, np.ndarray | None] = {}
        self._gather_handles: Dict[str, NodeInputHandle] = {}
        self._compiled = False

    @property
    def ordered_nodes(self) -> tuple[str, ...]:
        return self._ordered_names

    def begin_block(
        self,
        frames: int,
        sample_rate: float | None = None,
        base_params: Dict[str, Dict[str, np.ndarray]] | None = None,
    ) -> None:
        """
        Prepare for a new processing block (frame window), initializing all C-ready node buffers and caches.
        """
        if frames <= 0:
            raise ValueError("frames must be positive")
        self._frames = int(frames)
        self._sample_rate = float(sample_rate or self._graph.sample_rate)
        self._base_params = dict(base_params or {})
        # Descriptor parsing and cache allocation are performed in compile();
        # begin_block should be lightweight and only reset per-block state.
        if not self._compiled:
            raise RuntimeError("CffiEdgeRunner must be compiled before beginning a block")
        # Reset per-block caches (keep allocations) and gather handles map
        for name in tuple(self._caches.keys()):
            # preserve existing arrays; setting to None marks as not yet produced this block
            self._caches[name] = None
        self._gather_handles.clear()
        for node in self._graph._nodes.values():
            recycle = getattr(node, "recycle_blocks", None)
            if recycle is not None:
                recycle()

    def compile(self) -> None:
        """Compile the graph descriptors and preallocate reusable caches.

        This should be run once when the graph changes (or when the runner is created).
        """
        self._load_descriptors()
        # Preallocate per-node cache placeholders (actual arrays will be allocated by C or reused)
        self._caches = {name: None for name in self._graph._nodes}
        self._gather_handles.clear()
        # Allow nodes to recycle or preallocate any node-local buffers
        for node in self._graph._nodes.values():
            recycle = getattr(node, "recycle_blocks", None)
            if recycle is not None:
                recycle()
        self._compiled = True

    def gather_to(self, node_name: str) -> NodeInputHandle:
        """
        Gather all C-ready input buffers for the given node, including audio and parameter buffers.
        Returns a NodeInputHandle with CFFI pointers to C-ready buffers.
        """
        descriptor = self._descriptor_by_name.get(node_name)
        if descriptor is None:
            raise KeyError(f"Unknown node '{node_name}' in edge runner")
        audio_node_buffers = self._collect_audio_inputs(descriptor)
        if audio_node_buffers:
            batches = audio_node_buffers[0].shape[0]
            frame_count = audio_node_buffers[0].shape[2]
            node_buffer, channels = self._merge_audio(audio_node_buffers, batches, frame_count)
        else:
            node_buffer = None
            batches = int(self._base_params.get("_B", 1))
            channels = int(self._base_params.get("_C", 1))
            frame_count = self._frames
        param_buffers = self._prepare_base_params(node_name, batches, channels, frame_count)
        merged_param_buffers = self._apply_modulations(
            node_name, descriptor.mod_groups, param_buffers, batches, channels, frame_count
        )
        handle = self._build_handle(
            node_name, node_buffer, batches, channels, frame_count, merged_param_buffers
        )
        self._gather_handles[node_name] = handle
        return handle

    def set_node_output(self, node_name: str, output_buffer: np.ndarray | None) -> None:
        """
        Set the C-ready output buffer for a node after C evaluation.
        """
        if node_name not in self._caches:
            raise KeyError(f"Unknown node '{node_name}' in edge runner caches")
        if output_buffer is None:
            self._caches[node_name] = None
            return
        c_ready_buffer = np.asarray(output_buffer, dtype=RAW_DTYPE)
        c_ready_buffer = _assert_bcf(c_ready_buffer, name=f"{node_name}.out")
        if not c_ready_buffer.flags["C_CONTIGUOUS"]:
            c_ready_buffer = np.ascontiguousarray(c_ready_buffer, dtype=RAW_DTYPE)
        self._caches[node_name] = c_ready_buffer

    def get_cached_output(self, node_name: str) -> np.ndarray | None:
        """
        Get the C-ready output buffer for a node.
        """
        return self._caches.get(node_name)

    def _load_descriptors(self) -> None:
        blob = self._graph.serialize_node_descriptors()
        descriptors = list(self._parse_descriptors(blob))
        self._node_descriptors = tuple(descriptors)
        self._descriptor_by_name = {desc.name: desc for desc in descriptors}
        self._ordered_names = tuple(desc.name for desc in descriptors)

    def _parse_descriptors(self, blob: bytes) -> Iterable[_ParsedNodeDescriptor]:
        offset = 0
        (count,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        for _ in range(count):
            header = _NODE_DESCRIPTOR_HEADER.unpack_from(blob, offset)
            offset += _NODE_DESCRIPTOR_HEADER.size
            name_len = header[1]
            type_len = header[2]
            audio_count = header[3]
            mod_count = header[4]
            param_buffer_count = header[5]
            buffer_shape_count = header[6]
            params_json_len = header[7]

            name = blob[offset : offset + name_len].decode("utf-8")
            offset += name_len
            offset += type_len  # skip type name

            audio_inputs = []
            for _ in range(audio_count):
                (source_len,) = struct.unpack_from("<I", blob, offset)
                offset += 4
                source = blob[offset : offset + source_len].decode("utf-8")
                offset += source_len
                audio_inputs.append(source)

            grouped: Dict[str, list[ModConnection]] = {}
            order: list[str] = []
            for _ in range(mod_count):
                mod_header = _MOD_DESCRIPTOR_HEADER.unpack_from(blob, offset)
                offset += _MOD_DESCRIPTOR_HEADER.size
                source_len, param_len, mode_code, scale, channel = mod_header
                source = blob[offset : offset + source_len].decode("utf-8")
                offset += source_len
                param = blob[offset : offset + param_len].decode("utf-8")
                offset += param_len
                mode = _MODE_NAMES.get(mode_code, "add")
                channel_idx = None if channel < 0 else int(channel)
                connection = ModConnection(
                    source=source,
                    target=name,
                    param=param,
                    scale=float(scale),
                    mode=mode,
                    channel=channel_idx,
                )
                key = connection.param or "value"
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(connection)

            for _ in range(param_buffer_count):
                param_header = _PARAM_DESCRIPTOR_HEADER.unpack_from(blob, offset)
                offset += _PARAM_DESCRIPTOR_HEADER.size
                name_size = param_header[0]
                offset += name_size
                byte_len = param_header[4]
                offset += byte_len

            offset += buffer_shape_count * 12
            offset += params_json_len

            mod_groups = tuple((param, tuple(grouped[param])) for param in order)
            yield _ParsedNodeDescriptor(
                name=name,
                audio_inputs=tuple(audio_inputs),
                mod_groups=mod_groups,
            )

    def _collect_audio_inputs(self, descriptor: _ParsedNodeDescriptor) -> list[np.ndarray]:
        """
        Collect C-ready output buffers from upstream nodes for use as input node buffers.
        """
        node_buffers: list[np.ndarray] = []
        for source in descriptor.audio_inputs:
            output_buffer = self._caches.get(source)
            if output_buffer is None:
                continue
            node_buffers.append(_assert_bcf(output_buffer, name=f"{source}.out"))
        return node_buffers

    def _merge_audio(
        self,
        node_buffers: Sequence[np.ndarray],
        batches: int,
        frames: int,
    ) -> tuple[np.ndarray, int]:
        """
        Merge multiple C-ready node buffers into a single C-ready input buffer for the node.
        """
        if len(node_buffers) == 1:
            node_buffer = node_buffers[0]
            channels = node_buffer.shape[1]
            return node_buffer, channels
        total_channels = node_buffers[0].shape[1]
        for buf in node_buffers[1:]:
            if buf.shape[0] != batches or buf.shape[2] != frames:
                raise ValueError("Shape mismatch in node buffers during gather")
            total_channels += buf.shape[1]
        workspace = self._graph._acquire_audio_workspace((batches, total_channels, frames))
        offset = 0
        for buf in node_buffers:
            channels = buf.shape[1]
            target = workspace[:, offset : offset + channels, :]
            np.copyto(target, buf)
            offset += channels
        return workspace[:, :total_channels, :], total_channels

    def _prepare_base_params(
        self,
        node_name: str,
        batches: int,
        channels: int,
        frames: int,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare C-ready parameter buffers for the node.
        """
        param_buffers: Dict[str, np.ndarray] = {}
        node_params = self._base_params.get(node_name)
        if not node_params:
            return param_buffers
        for key, value in node_params.items():
            param_buffers[key] = self._graph._prepare_param_buffer(
                node_name, key, value, batches, channels, frames
            )
        return param_buffers

    def _apply_modulations(
        self,
        node_name: str,
        mod_groups: Sequence[tuple[str, Sequence[ModConnection]]],
        param_buffers: Dict[str, np.ndarray],
        batches: int,
        channels: int,
        frames: int,
    ) -> Dict[str, np.ndarray]:
        """
        Apply modulations to C-ready parameter buffers for the node.
        """
        merged = dict(param_buffers)
        if not mod_groups:
            return merged
        shape = (batches, channels, frames)
        scratch = self._graph._acquire_merge_scratch(shape)
        for param_name, connections in mod_groups:
            signals: list[tuple[np.ndarray, float, str]] = []
            for connection in connections:
                output_buffer = self._caches.get(connection.source)
                if output_buffer is None:
                    continue
                source_buf = _assert_bcf(output_buffer, name=f"{connection.source}.out")
                mod_signal = self._graph._prepare_mod_buffer(
                    node_name,
                    param_name,
                    connection,
                    source_buf,
                    batches,
                    channels,
                    frames,
                )
                signals.append((mod_signal, connection.scale, connection.mode))
            if not signals:
                continue
            base_source = merged.get(param_name)
            if base_source is None:
                base_source = self._graph._prepare_param_buffer(
                    node_name, param_name, 0.0, batches, channels, frames
                )
            base = np.require(base_source, dtype=RAW_DTYPE, requirements=("C",))
            if base is base_source:
                base = base.copy()
            for signal, scale, mode in signals:
                if mode == "add":
                    if scale == 0.0:
                        continue
                    if scale == 1.0:
                        np.add(base, signal, out=base)
                    else:
                        np.multiply(signal, scale, out=scratch)
                        np.add(base, scratch, out=base)
                else:
                    np.multiply(signal, scale, out=scratch)
                    np.add(scratch, 1.0, out=scratch)
                    np.multiply(base, scratch, out=base)
            merged[param_name] = base
        return merged

    def _build_handle(
        self,
        node_name: str,
        node_buffer: np.ndarray | None,
        batches: int,
        channels: int,
        frames: int,
        param_buffers: Dict[str, np.ndarray],
    ) -> NodeInputHandle:
        """
        Build a NodeInputHandle with CFFI pointers to all C-ready node and parameter buffers.
        """
        struct_obj = self.ffi.new("EdgeRunnerNodeInputs *")
        keepers: list[Any] = [struct_obj]
        c_node_buffer = None
        node_ptr = self.ffi.NULL
        if node_buffer is not None:
            c_node_buffer = np.require(node_buffer, dtype=RAW_DTYPE, requirements=("C",))
            node_ptr = self.ffi.from_buffer("double[]", c_node_buffer)
            keepers.append(node_ptr)
        struct_obj.audio.has_audio = 1 if c_node_buffer is not None else 0
        struct_obj.audio.batches = int(batches)
        struct_obj.audio.channels = int(channels)
        struct_obj.audio.frames = int(frames)
        struct_obj.audio.data = node_ptr

        param_names = tuple(param_buffers.keys())
        param_bufs: list[np.ndarray] = []
        if param_names:
            items = self.ffi.new("EdgeRunnerParamView[]", len(param_names))
            keepers.append(items)
            name_buffers: list[Any] = []
            for idx, name in enumerate(param_names):
                buf = np.require(param_buffers[name], dtype=RAW_DTYPE, requirements=("C",))
                param_bufs.append(buf)
                name_buf = self.ffi.new("char[]", name.encode("utf-8"))
                name_buffers.append(name_buf)
                data_ptr = self.ffi.from_buffer("double[]", buf)
                keepers.append(data_ptr)
                items[idx].name = name_buf
                items[idx].batches = int(buf.shape[0])
                items[idx].channels = int(buf.shape[1])
                items[idx].frames = int(buf.shape[2])
                items[idx].data = data_ptr
            keepers.extend(name_buffers)
            struct_obj.params.count = len(param_names)
            struct_obj.params.items = items
        else:
            struct_obj.params.count = 0
            struct_obj.params.items = self.ffi.NULL

        return NodeInputHandle(
            node=node_name,
            ffi=self.ffi,
            cdata=struct_obj,
            node_buffer=c_node_buffer,
            batches=int(batches),
            channels=int(channels),
            frames=int(frames),
            param_names=param_names,
            param_buffers=tuple(param_bufs),
            keepers=tuple(keepers),
        )

    def run_c_graph(self, control_history_blob: bytes) -> np.ndarray:
        """
        Invoke the C kernel for the entire graph using the serialized control history blob.
        Returns the output buffer as a C-ready numpy array.
        """
        # For now, also pass the node descriptors so C can know the graph structure
        node_descriptors = self._graph.serialize_node_descriptors()
        print("[CffiEdgeRunner] Passing control history blob of size:", len(control_history_blob))
        print("[CffiEdgeRunner] Passing node descriptors of size:", len(node_descriptors))
        print("[CffiEdgeRunner] Graph traversal order:", self.ordered_nodes)
        # CFFI call stub: replace with actual C call
        # Example: c_output = self.ffi.dlopen("_amp_ckernels_cffi").amp_cffi_run_graph(control_history_blob, node_descriptors, ...)
        # For now, just return a dummy output buffer
        batches = 1
        channels = self._graph.output_channels if hasattr(self._graph, 'output_channels') else 2
        frames = self._frames
        dummy = np.zeros((batches, channels, frames), dtype=RAW_DTYPE)
        print("[CffiEdgeRunner] (Stub) Returning dummy output buffer:", dummy.shape)
        return dummy

__all__ = ["CffiEdgeRunner", "NodeInputHandle"]
