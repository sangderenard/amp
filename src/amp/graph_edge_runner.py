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
    """Holds the CFFI view for a node's prepared inputs."""

    node: str
    ffi: Any
    cdata: Any
    audio: np.ndarray | None
    batches: int
    channels: int
    frames: int
    param_names: tuple[str, ...]
    param_arrays: tuple[np.ndarray, ...]
    keepers: tuple[Any, ...]

    @property
    def params(self) -> Dict[str, np.ndarray]:
        return {name: array for name, array in zip(self.param_names, self.param_arrays)}


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
    """Prepares node inputs using the C-formatted graph descriptors."""

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

    @property
    def ordered_nodes(self) -> tuple[str, ...]:
        return self._ordered_names

    def begin_block(
        self,
        frames: int,
        sample_rate: float | None = None,
        base_params: Dict[str, Dict[str, np.ndarray]] | None = None,
    ) -> None:
        if frames <= 0:
            raise ValueError("frames must be positive")
        self._frames = int(frames)
        self._sample_rate = float(sample_rate or self._graph.sample_rate)
        self._base_params = dict(base_params or {})
        self._load_descriptors()
        self._caches = {name: None for name in self._graph._nodes}
        self._gather_handles.clear()
        for node in self._graph._nodes.values():
            recycle = getattr(node, "recycle_blocks", None)
            if recycle is not None:
                recycle()

    def gather_to(self, node_name: str) -> NodeInputHandle:
        descriptor = self._descriptor_by_name.get(node_name)
        if descriptor is None:
            raise KeyError(f"Unknown node '{node_name}' in edge runner")
        audio_arrays = self._collect_audio_inputs(descriptor)
        if audio_arrays:
            batches = audio_arrays[0].shape[0]
            frame_count = audio_arrays[0].shape[2]
            audio_view, channels = self._merge_audio(audio_arrays, batches, frame_count)
        else:
            audio_view = None
            batches = int(self._base_params.get("_B", 1))
            channels = int(self._base_params.get("_C", 1))
            frame_count = self._frames
        node_params = self._prepare_base_params(node_name, batches, channels, frame_count)
        merged_params = self._apply_modulations(
            node_name, descriptor.mod_groups, node_params, batches, channels, frame_count
        )
        handle = self._build_handle(
            node_name, audio_view, batches, channels, frame_count, merged_params
        )
        self._gather_handles[node_name] = handle
        return handle

    def set_node_output(self, node_name: str, output: np.ndarray | None) -> None:
        if node_name not in self._caches:
            raise KeyError(f"Unknown node '{node_name}' in edge runner caches")
        if output is None:
            self._caches[node_name] = None
            return
        array = np.asarray(output, dtype=RAW_DTYPE)
        array = _assert_bcf(array, name=f"{node_name}.out")
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array, dtype=RAW_DTYPE)
        self._caches[node_name] = array

    def get_cached_output(self, node_name: str) -> np.ndarray | None:
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
        arrays: list[np.ndarray] = []
        for source in descriptor.audio_inputs:
            buffer = self._caches.get(source)
            if buffer is None:
                continue
            arrays.append(_assert_bcf(buffer, name=f"{source}.out"))
        return arrays

    def _merge_audio(
        self,
        audio_inputs: Sequence[np.ndarray],
        batches: int,
        frames: int,
    ) -> tuple[np.ndarray, int]:
        if len(audio_inputs) == 1:
            audio = audio_inputs[0]
            channels = audio.shape[1]
            return audio, channels
        total_channels = audio_inputs[0].shape[1]
        for buf in audio_inputs[1:]:
            if buf.shape[0] != batches or buf.shape[2] != frames:
                raise ValueError("Shape mismatch in audio inputs during gather")
            total_channels += buf.shape[1]
        workspace = self._graph._acquire_audio_workspace((batches, total_channels, frames))
        offset = 0
        for buf in audio_inputs:
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
        params: Dict[str, np.ndarray] = {}
        node_params = self._base_params.get(node_name)
        if not node_params:
            return params
        for key, value in node_params.items():
            params[key] = self._graph._prepare_param_buffer(
                node_name, key, value, batches, channels, frames
            )
        return params

    def _apply_modulations(
        self,
        node_name: str,
        mod_groups: Sequence[tuple[str, Sequence[ModConnection]]],
        node_params: Dict[str, np.ndarray],
        batches: int,
        channels: int,
        frames: int,
    ) -> Dict[str, np.ndarray]:
        merged = dict(node_params)
        if not mod_groups:
            return merged
        shape = (batches, channels, frames)
        scratch = self._graph._acquire_merge_scratch(shape)
        for param_name, connections in mod_groups:
            signals: list[tuple[np.ndarray, float, str]] = []
            for connection in connections:
                buffer = self._caches.get(connection.source)
                if buffer is None:
                    continue
                source_buf = _assert_bcf(buffer, name=f"{connection.source}.out")
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
        audio: np.ndarray | None,
        batches: int,
        channels: int,
        frames: int,
        params: Dict[str, np.ndarray],
    ) -> NodeInputHandle:
        struct_obj = self.ffi.new("EdgeRunnerNodeInputs *")
        keepers: list[Any] = [struct_obj]
        audio_array = None
        audio_ptr = self.ffi.NULL
        if audio is not None:
            audio_array = np.require(audio, dtype=RAW_DTYPE, requirements=("C",))
            audio_ptr = self.ffi.from_buffer("double[]", audio_array)
            keepers.append(audio_ptr)
        struct_obj.audio.has_audio = 1 if audio_array is not None else 0
        struct_obj.audio.batches = int(batches)
        struct_obj.audio.channels = int(channels)
        struct_obj.audio.frames = int(frames)
        struct_obj.audio.data = audio_ptr

        param_names = tuple(params.keys())
        param_arrays: list[np.ndarray] = []
        if param_names:
            items = self.ffi.new("EdgeRunnerParamView[]", len(param_names))
            keepers.append(items)
            name_buffers: list[Any] = []
            for idx, name in enumerate(param_names):
                array = np.require(params[name], dtype=RAW_DTYPE, requirements=("C",))
                param_arrays.append(array)
                name_buf = self.ffi.new("char[]", name.encode("utf-8"))
                name_buffers.append(name_buf)
                data_ptr = self.ffi.from_buffer("double[]", array)
                keepers.append(data_ptr)
                items[idx].name = name_buf
                items[idx].batches = int(array.shape[0])
                items[idx].channels = int(array.shape[1])
                items[idx].frames = int(array.shape[2])
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
            audio=audio_array,
            batches=int(batches),
            channels=int(channels),
            frames=int(frames),
            param_names=param_names,
            param_arrays=tuple(param_arrays),
            keepers=tuple(keepers),
        )


__all__ = ["CffiEdgeRunner", "NodeInputHandle"]
