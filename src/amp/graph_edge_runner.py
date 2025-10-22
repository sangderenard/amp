from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence
import json
import struct
import time

import numpy as np
import os

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
from . import quantizer
from .node_contracts import NodeContract, get_node_contract

try:  # pragma: no cover - optional dependency
    import cffi
except Exception:  # pragma: no cover - optional dependency
    cffi = None


_MODE_NAMES = {code: name for name, code in _MODE_CODES.items()}


def _extract_controller_signal(expression: str) -> str | None:
    """Return the referenced signal name when expression is a simple lookup."""

    expr = expression.strip()
    for prefix in ("signals[", "raw_signals["):
        if not expr.startswith(prefix):
            continue
        if not expr.endswith("]"):
            continue
        inner = expr[len(prefix) : -1].strip()
        if len(inner) >= 2 and inner[0] in {'"', "'"} and inner[-1] == inner[0]:
            return inner[1:-1]
    return None


@dataclass(slots=True)
class _ParsedNodeDescriptor:
    name: str
    type_name: str
    params_json: str
    audio_inputs: tuple[str, ...]
    mod_groups: tuple[tuple[str, tuple[ModConnection, ...]], ...]
    blob_offset: int
    blob_size: int


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
    /*
     * C execution contract shared with the compiled `_amp_ckernels_cffi` module.
     *
     * The Python runner provides node descriptors (static metadata), node inputs
     * (audio buffers + parameter views) and receives an owned output buffer.
     *
     * Ownership rules:
     *   - `amp_run_node` allocates `out_buffer` when it returns 0 (success).
     *   - Callers must release that buffer with `amp_free` after copying.
     *   - `amp_run_node` stores per-node state via the opaque `state` pointer.
     *     Python retains the pointer between invocations and eventually calls
     *     `amp_release_state` to free any associated allocations.
     *
     * Return codes from `amp_run_node`:
     *   0   -> success (buffer+channels populated)
     *  -1   -> allocation or contract violation (fatal)
     *  -3   -> node type not handled by the C backend (fallback to Python)
     */
    typedef unsigned char uint8_t;

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

    typedef struct {
        const char *name;
        size_t name_len;
        const char *type_name;
        size_t type_len;
        const char *params_json;
        size_t params_len;
    } EdgeRunnerNodeDescriptor;

    typedef struct {
        const char *name;
        uint32_t name_len;
        uint32_t offset;
        uint32_t span;
    } EdgeRunnerCompiledParam;

    typedef struct {
        const char *name;
        uint32_t name_len;
        uint32_t function_id;
        uint32_t audio_offset;
        uint32_t audio_span;
        uint32_t param_count;
        EdgeRunnerCompiledParam *params;
    } EdgeRunnerCompiledNode;

    typedef struct {
        uint32_t version;
        uint32_t node_count;
        EdgeRunnerCompiledNode *nodes;
    } EdgeRunnerCompiledPlan;

    typedef struct {
        char *name;
        uint32_t name_len;
        double *values;
        uint32_t value_count;
        double timestamp;
    } EdgeRunnerControlCurve;

    typedef struct {
        uint32_t frames_hint;
        uint32_t curve_count;
        EdgeRunnerControlCurve *curves;
    } EdgeRunnerControlHistory;

    int amp_run_node(
        const EdgeRunnerNodeDescriptor *descriptor,
        const EdgeRunnerNodeInputs *inputs,
        int batches,
        int channels,
        int frames,
        double sample_rate,
        double **out_buffer,
        int *out_channels,
        void **state,
        const EdgeRunnerControlHistory *history
    );

    void amp_free(double *buffer);
    void amp_release_state(void *state);

    EdgeRunnerCompiledPlan *amp_load_compiled_plan(
        const uint8_t *descriptor_blob,
        size_t descriptor_len,
        const uint8_t *plan_blob,
        size_t plan_len
    );

    void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan);

    EdgeRunnerControlHistory *amp_load_control_history(
        const uint8_t *blob,
        size_t blob_len,
        int frames_hint
    );

    void amp_release_control_history(EdgeRunnerControlHistory *history);
"""


class CffiEdgeRunner:
    """Prepares C-ready node input buffers using the C-formatted graph descriptors."""

    def __init__(self, graph: AudioGraph) -> None:
        if cffi is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("cffi is required to build the graph edge runner")
        self._graph = graph
        self.ffi = cffi.FFI()
        self.ffi.cdef(_EDGE_RUNNER_CDEF)
        self._descriptor_blob: bytes = b""
        self._node_descriptors: tuple[_ParsedNodeDescriptor, ...] = ()
        self._descriptor_by_name: dict[str, _ParsedNodeDescriptor] = {}
        self._ordered_names: tuple[str, ...] = ()
        self._frames: int = 0
        self._sample_rate: float = float(graph.sample_rate)
        self._base_params: Dict[str, Dict[str, np.ndarray]] = {}
        self._descriptor_params_cache: Dict[str, Dict[str, Any]] = {}
        self._caches: Dict[str, np.ndarray | None] = {}
        self._gather_handles: Dict[str, NodeInputHandle] = {}
        self._node_contracts: Dict[str, NodeContract | None] = {}
        self._compiled = False
        self._compiled_plan: bytes | None = None
        self._plan_handle: Any = self.ffi.NULL
        self._plan_names: tuple[str, ...] = ()
        self._c_kernel: Any | None = None
        self._node_states: Dict[str, Any] = {}
        self._python_fallback_counts: Dict[str, int] = {}
        self.compile()

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
        if self._node_states:
            self._release_states()
        self._release_plan()
        self._load_descriptors()
        # Preallocate per-node cache placeholders (actual arrays will be allocated by C or reused)
        self._caches = {name: None for name in self._graph._nodes}
        self._gather_handles.clear()
        self._python_fallback_counts.clear()
        # Allow nodes to recycle or preallocate any node-local buffers
        for node in self._graph._nodes.values():
            recycle = getattr(node, "recycle_blocks", None)
            if recycle is not None:
                recycle()
        self._compiled_plan = self._serialize_compiled_plan()
        self._load_compiled_plan_handle()
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
            channels = self._predict_output_channels(node_name, descriptor, batches)
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
        self._descriptor_blob = blob
        self._node_descriptors = tuple(descriptors)
        self._descriptor_by_name = {desc.name: desc for desc in descriptors}
        self._ordered_names = tuple(desc.name for desc in descriptors)
        self._descriptor_params_cache.clear()
        self._node_contracts = {desc.name: get_node_contract(desc.type_name) for desc in descriptors}

    def _release_plan(self) -> None:
        if self._plan_handle not in (None, self.ffi.NULL):
            lib = self._c_kernel
            if lib is not None:
                try:
                    lib.amp_release_compiled_plan(self._plan_handle)
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
            self._plan_handle = self.ffi.NULL
        self._plan_names = ()

    def _load_compiled_plan_handle(self) -> None:
        self._plan_names = self._ordered_names
        if not self._descriptor_blob or not self._compiled_plan:
            self._release_plan()
            return
        try:
            lib = self._ensure_c_kernel()
        except Exception:
            # If the C kernel isn't available during compiled-plan loading, just
            # release any plan handle and return early — compiled plans are only
            # meaningful when a C backend is present.
            self._release_plan()
            return
        desc_buf = self.ffi.new("uint8_t[]", self._descriptor_blob)
        plan_buf = self.ffi.new("uint8_t[]", self._compiled_plan)
        handle = lib.amp_load_compiled_plan(
            desc_buf,
            len(self._descriptor_blob),
            plan_buf,
            len(self._compiled_plan),
        )
        if handle == self.ffi.NULL:
            raise RuntimeError("C kernel rejected compiled plan blob")
        plan_struct = handle[0]
        names: list[str] = []
        for idx in range(int(plan_struct.node_count)):
            entry = plan_struct.nodes[idx]
            if entry.name == self.ffi.NULL:
                lib.amp_release_compiled_plan(handle)
                raise RuntimeError("Compiled plan entry is missing a node name")
            decoded = self.ffi.string(entry.name, entry.name_len).decode("utf-8")
            names.append(decoded)
        plan_names = tuple(names)
        if plan_names != self._ordered_names:
            lib.amp_release_compiled_plan(handle)
            raise RuntimeError("C kernel returned plan with mismatched node ordering")
        self._release_plan()
        self._plan_handle = handle
        self._plan_names = plan_names

    def describe_compiled_plan(self) -> Dict[str, Any]:
        if self._plan_handle in (None, self.ffi.NULL):
            return {
                "version": 0,
                "node_count": len(self._ordered_names),
                "nodes": tuple(
                    {
                        "name": name,
                        "function_id": idx,
                        "audio_offset": idx,
                        "audio_span": 1,
                        "params": tuple(),
                    }
                    for idx, name in enumerate(self._ordered_names)
                ),
            }
        plan_struct = self._plan_handle[0]
        nodes: list[Dict[str, Any]] = []
        for idx in range(int(plan_struct.node_count)):
            entry = plan_struct.nodes[idx]
            name = self.ffi.string(entry.name, entry.name_len).decode("utf-8")
            params: list[Dict[str, Any]] = []
            for param_idx in range(int(entry.param_count)):
                param_entry = entry.params[param_idx]
                param_name = self.ffi.string(param_entry.name, param_entry.name_len).decode("utf-8")
                params.append(
                    {
                        "name": param_name,
                        "offset": int(param_entry.offset),
                        "span": int(param_entry.span),
                    }
                )
            nodes.append(
                {
                    "name": name,
                    "function_id": int(entry.function_id),
                    "audio_offset": int(entry.audio_offset),
                    "audio_span": int(entry.audio_span),
                    "params": tuple(params),
                }
            )
        return {
            "version": int(plan_struct.version),
            "node_count": int(plan_struct.node_count),
            "nodes": tuple(nodes),
        }

    def python_fallback_summary(self) -> Dict[str, int]:
        """Return a copy of the per-node Python fallback invocation counts."""

        return dict(self._python_fallback_counts)

    def _ensure_c_kernel(self) -> Any:
        if self._c_kernel is not None:
            return self._c_kernel
        try:
            lib = self.ffi.dlopen("_amp_ckernels_cffi")
        except OSError as exc:  # pragma: no cover - depends on deployment environment
            try:
                from . import c_kernels  # noqa: WPS433 - runtime import to trigger build

                if getattr(c_kernels, "AVAILABLE", False):
                    module_path = getattr(getattr(c_kernels, "_impl", None), "__file__", None)
                    if module_path:
                        lib = self.ffi.dlopen(module_path)
                    else:
                        lib = self.ffi.dlopen("_amp_ckernels_cffi")
                else:  # pragma: no cover - depends on optional dependency
                    reason = getattr(c_kernels, "UNAVAILABLE_REASON", "unavailable")
                    raise RuntimeError(
                        "The compiled '_amp_ckernels_cffi' shared library is required for C graph execution"
                        f" (unavailable: {reason})"
                    ) from exc
            except Exception as build_exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    "The compiled '_amp_ckernels_cffi' shared library is required for C graph execution"
                ) from build_exc
        self._c_kernel = lib
        return lib

    def _serialize_compiled_plan(self) -> bytes:
        if not self._ordered_names:
            return b""
        payload = bytearray()
        payload.extend(b"AMPL")
        payload.extend(struct.pack("<II", 1, len(self._ordered_names)))
        audio_cursor = 0
        for function_id, name in enumerate(self._ordered_names):
            descriptor = self._descriptor_by_name[name]
            name_bytes = name.encode("utf-8")
            audio_offset = audio_cursor
            audio_span = 1
            audio_cursor += 1
            param_count = len(descriptor.mod_groups)
            payload.extend(
                struct.pack(
                    "<IIIII",
                    int(function_id),
                    len(name_bytes),
                    int(audio_offset),
                    int(audio_span),
                    int(param_count),
                )
            )
            payload.extend(name_bytes)
            param_cursor = 0
            for param_name, _ in descriptor.mod_groups:
                param_bytes = param_name.encode("utf-8")
                payload.extend(
                    struct.pack(
                        "<III",
                        len(param_bytes),
                        int(param_cursor),
                        0,
                    )
                )
                payload.extend(param_bytes)
                param_cursor += 1
        return bytes(payload)

    def _parse_descriptors(self, blob: bytes) -> Iterable[_ParsedNodeDescriptor]:
        offset = 0
        (count,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        for _ in range(count):
            start = offset
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
            type_name = blob[offset : offset + type_len].decode("utf-8")
            offset += type_len

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

            params_json = blob[offset - params_json_len : offset].decode("utf-8") if params_json_len else "{}"

            size = offset - start

            mod_groups = tuple((param, tuple(grouped[param])) for param in order)
            yield _ParsedNodeDescriptor(
                name=name,
                type_name=type_name,
                params_json=params_json,
                audio_inputs=tuple(audio_inputs),
                mod_groups=mod_groups,
                blob_offset=start,
                blob_size=size,
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

        # Defensive validation: ensure all C-ready buffers match the handle's
        # batches/frames the runner is passing to C. If a producer created a
        # buffer with an incorrect shape, fail fast and point to the offending
        # node/param so it can be fixed upstream.
        for p_name, p_buf in zip(param_names, param_bufs):
            try:
                if p_buf is None:
                    continue
                if int(p_buf.shape[0]) != int(batches) or int(p_buf.shape[2]) != int(frames):
                    raise RuntimeError(
                        f"Shape mismatch for node '{node_name}' param '{p_name}': "
                        f"buf.shape={p_buf.shape} vs handle (batches={int(batches)}, frames={int(frames)})"
                    )
            except Exception:
                # Re-raise with context for clarity
                raise

        if c_node_buffer is not None:
            try:
                if int(c_node_buffer.shape[0]) != int(batches) or int(c_node_buffer.shape[2]) != int(frames):
                    raise RuntimeError(
                        f"Shape mismatch for node '{node_name}' audio buffer: "
                        f"buf.shape={c_node_buffer.shape} vs handle (batches={int(batches)}, frames={int(frames)})"
                    )
            except Exception:
                raise

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
        if not self._compiled:
            raise RuntimeError("CffiEdgeRunner must be compiled before running the graph")
        try:
            lib = self._ensure_c_kernel()
        except Exception:
            lib = None

        if not self._descriptor_blob:
            raise RuntimeError("Descriptor blob not initialised for C execution")

        timings: Dict[str, float] | None = getattr(self._graph, "_last_node_timings", None)
        if timings is not None:
            timings.clear()

        result: np.ndarray | None = None
        sink_name = getattr(self._graph, "sink", None)
        execution_order = self._plan_names or self._ordered_names
        history_handle = self.ffi.NULL
        history_keepalive: tuple[Any, ...] = ()
        if lib is None:
            # No real C kernel available: try lightweight test module which implements
            # `test_run_graph` to validate handoff. If that is not available, fall
            # back to Python zero output.
            try:
                from .c_kernels_test import get_test_lib

                mod = get_test_lib()
                if mod is not None:
                    ffi, tlib = mod
                    batches = 1
                    channels = int(getattr(self._graph, "output_channels", 2) or 2)
                    frames = int(self._frames)
                    out_count = batches * channels * frames
                    out_ptr = ffi.new("double[]", out_count)
                    ctrl = ffi.new("uint8_t[]", control_history_blob)
                    desc = ffi.new("uint8_t[]", self._descriptor_blob)
                    tlib.test_run_graph(ctrl, len(control_history_blob), desc, len(self._descriptor_blob), out_ptr, batches, channels, frames)
                    arr = np.frombuffer(ffi.buffer(out_ptr, out_count * np.dtype(RAW_DTYPE).itemsize), dtype=RAW_DTYPE).copy()
                    arr = arr.reshape((batches, channels, frames))
                    return arr
            except Exception:
                # Fall through to regular path which will raise if necessary
                pass

        if control_history_blob:
            history_buf = self.ffi.new("uint8_t[]", control_history_blob)
            handle = lib.amp_load_control_history(history_buf, len(control_history_blob), int(self._frames))
            if handle == self.ffi.NULL:
                raise RuntimeError("C kernel rejected control history blob")
            history_handle = handle
            history_keepalive = (history_buf,)
        # Feature gate: verbose per-node logging to help diagnose C/kernel failures.
        verbose_nodes = os.environ.get("AMP_VERBOSE_NODES", "0") in ("1", "true", "True")
        try:
            for name in execution_order:
                descriptor = self._descriptor_by_name.get(name)
                if descriptor is None:
                    raise RuntimeError(f"Missing descriptor for node '{name}'")
                handle = self.gather_to(name)
                desc_struct, desc_keepalive = self._build_descriptor_struct(descriptor)
                _keepalive = (handle, desc_keepalive)
                out_ptr = self.ffi.new("double **")
                out_channels = self.ffi.new("int *")
                state_ptr = self.ffi.new(
                    "void **", self._node_states.get(name, self.ffi.NULL)
                )
                start_time = time.perf_counter()
                status = lib.amp_run_node(
                    desc_struct,
                    handle.cdata,
                    int(handle.batches),
                    int(handle.channels),
                    int(handle.frames),
                    float(self._sample_rate),
                    out_ptr,
                    out_channels,
                    state_ptr,
                    history_handle,
                )
                if verbose_nodes:
                    try:
                        print(f"[CffiEdgeRunner] node='{name}' -> status={status}", flush=True)
                    except Exception:
                        pass
                # Defensive diagnostics: surface unexpected C-kernel failures
                if status not in (0, -3):
                    try:
                        # Best-effort diagnostic to help identify C-side errors
                        print(f"[CffiEdgeRunner] amp_run_node returned status={status} for node='{name}'", flush=True)
                        print(f"  descriptor.type={descriptor.type_name} params={descriptor.params_json}", flush=True)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"C kernel failed while executing node '{name}' (status {status})"
                    )

                if status == -3:
                    contract = self._node_contracts.get(name)
                    message = (
                        "C kernel declined node '{name}' (type {type_name}); "
                        "runtime forbids Python fallbacks during graph execution"
                    ).format(name=name, type_name=descriptor.type_name)
                    if contract is not None and not contract.allow_python_fallback:
                        message += "; contract already marked this node as C-only"
                    raise RuntimeError(message)
                elif status == 0:
                    self._node_states[name] = state_ptr[0]
                    batches = int(handle.batches)
                    channels = int(out_channels[0])
                    frames = int(handle.frames)
                    # Diagnostic: detect param buffers whose frames differ from the runner frames
                    try:
                        if hasattr(handle, "param_names") and handle.param_names:
                            for p_name, p_buf in zip(handle.param_names, handle.param_buffers):
                                try:
                                    # p_buf is a numpy array of shape (B, C, F)
                                    if p_buf is not None and int(p_buf.shape[2]) != int(handle.frames):
                                        print(
                                            f"[CffiEdgeRunner][DIAG] node='{name}' param='{p_name}' buf_frames={int(p_buf.shape[2])} handle_frames={int(handle.frames)}",
                                            flush=True,
                                        )
                                except Exception:
                                    # best-effort only; do not interrupt normal execution
                                    pass
                    except Exception:
                        pass
                    total = batches * channels * frames
                    if total <= 0:
                        raise RuntimeError(f"Node '{name}' produced an empty buffer")
                    buffer = self.ffi.buffer(out_ptr[0], total * np.dtype(RAW_DTYPE).itemsize)
                    array = np.frombuffer(buffer, dtype=RAW_DTYPE).copy().reshape(
                        batches, channels, frames
                    )
                    lib.amp_free(out_ptr[0])
                else:
                    raise RuntimeError(
                        f"C kernel failed while executing node '{name}' (status {status})"
                    )
                end_time = time.perf_counter()
                if timings is not None:
                    timings[name] = float(end_time - start_time)
                self.set_node_output(name, array)
                if sink_name == name:
                    result = array
        finally:
            if history_handle not in (None, self.ffi.NULL):
                try:
                    lib.amp_release_control_history(history_handle)
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
        _ = history_keepalive
        if sink_name is None:
            raise RuntimeError("Graph sink has not been configured")
        if result is None:
            sink_buffer = self.get_cached_output(sink_name)
            if sink_buffer is None:
                raise RuntimeError("Sink node did not produce output")
            result = sink_buffer
        return np.require(result, dtype=RAW_DTYPE, requirements=("C",)).copy()

    def _predict_output_channels(
        self, node_name: str, descriptor: _ParsedNodeDescriptor, batches: int
    ) -> int:
        """Determine the expected output channel count for nodes without audio inputs."""

        def _coerce_positive_int(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, (int, np.integer)):
                return int(value) if int(value) > 0 else None
            if isinstance(value, (float, np.floating)) and not np.isnan(value):
                coerced = int(value)
                return coerced if coerced > 0 else None
            try:
                coerced = int(value)
            except Exception:
                return None
            return coerced if coerced > 0 else None

        contract = self._node_contracts.get(node_name)
        attr_keys: Sequence[str]
        param_keys: Sequence[str]
        stereo_params: Sequence[str]

        if contract is not None and contract.default_channels is not None:
            default_channels = int(contract.default_channels)
        else:
            default_channels = int(self._base_params.get("_C", 1))

        if contract is not None:
            attr_keys = contract.channel_attributes or ("channels", "out_channels")
            param_keys = contract.channel_params or ("channels", "out_channels")
            stereo_params = contract.stereo_params or ()
        else:
            attr_keys = ("channels", "out_channels")
            param_keys = ("channels", "out_channels")
            stereo_params = ()

        node_obj = self._graph._nodes.get(node_name)
        for attr_name in attr_keys:
            candidate = (
                _coerce_positive_int(getattr(node_obj, attr_name, None)) if node_obj else None
            )
            if candidate is not None:
                default_channels = candidate
                break

        params = self._descriptor_params_cache.get(node_name)
        if params is None:
            try:
                params = json.loads(descriptor.params_json) if descriptor.params_json else {}
                if not isinstance(params, dict):
                    params = {}
            except Exception:
                params = {}
            self._descriptor_params_cache[node_name] = params
        for key in param_keys:
            candidate = _coerce_positive_int(params.get(key))
            if candidate is not None:
                default_channels = candidate
                break

        if stereo_params and self._stereo_request(node_name, descriptor, stereo_params, batches):
            default_channels = max(default_channels, 2)

        if default_channels <= 0:
            return 1
        return int(default_channels)

    def _stereo_request(
        self,
        node_name: str,
        descriptor: _ParsedNodeDescriptor,
        stereo_params: Sequence[str],
        batches: int,
    ) -> bool:
        """Return True when the node contract requests promotion to stereo output."""

        if not stereo_params:
            return False
        node_params = self._base_params.get(node_name) or {}
        for param in stereo_params:
            if param in node_params:
                return True
        for param_name, connections in descriptor.mod_groups:
            if param_name not in stereo_params:
                continue
            for connection in connections:
                source_buffer = self._caches.get(connection.source)
                if source_buffer is not None and source_buffer.shape[0] == batches:
                    return True
        return False

    def _build_descriptor_struct(
        self, descriptor: _ParsedNodeDescriptor
    ) -> tuple[Any, tuple[Any, ...]]:
        struct_obj = self.ffi.new("EdgeRunnerNodeDescriptor *")
        name_bytes = descriptor.name.encode("utf-8")
        type_bytes = descriptor.type_name.encode("utf-8")
        node_obj = self._graph._nodes.get(descriptor.name)
        try:
            params_dict: Dict[str, Any] = json.loads(descriptor.params_json) if descriptor.params_json else {}
        except Exception:
            params_dict = {}
        extras: Dict[str, Any] = {}
        if descriptor.type_name == "MixNode" and node_obj is not None:
            channels = getattr(node_obj, "out_channels", None)
            if channels:
                extras.setdefault("channels", int(channels))
        if descriptor.type_name == "SafetyNode" and node_obj is not None:
            channels = getattr(node_obj, "channels", None)
            if channels:
                extras.setdefault("channels", int(channels))
            alpha = getattr(node_obj, "dc_alpha", None)
            if alpha is not None:
                extras.setdefault("dc_alpha", float(alpha))
        if descriptor.type_name == "SineOscillatorNode" and node_obj is not None:
            channels = getattr(node_obj, "channels", None)
            if channels:
                extras.setdefault("channels", int(channels))
            phase_state = getattr(node_obj, "_phase", None)
            try:
                initial_phase = float(phase_state[0, 0]) if phase_state is not None else None
            except Exception:  # pragma: no cover - defensive indexing
                initial_phase = None
            if initial_phase is not None:
                extras.setdefault("phase", float(initial_phase))
            freq = getattr(node_obj, "frequency", None)
            amp = getattr(node_obj, "amplitude", None)
            if freq is not None:
                extras.setdefault("frequency", float(freq))
            if amp is not None:
                extras.setdefault("amplitude", float(amp))
        if descriptor.type_name == "ControllerNode" and node_obj is not None:
            outputs = tuple(getattr(node_obj, "_output_order", ()))
            if outputs:
                extras.setdefault("__controller_outputs__", ",".join(outputs))
                mapping: Dict[str, Any] = {}
                params_cfg = getattr(node_obj, "params", {})
                outputs_cfg = params_cfg.get("outputs", {}) if isinstance(params_cfg, dict) else {}
                for out_name in outputs:
                    spec = outputs_cfg.get(out_name)
                    signal = None
                    if isinstance(spec, str):
                        signal = _extract_controller_signal(spec)
                    elif isinstance(spec, dict):
                        expr_val = spec.get("equation") or spec.get("expr") or spec.get("expression")
                        if isinstance(expr_val, str):
                            signal = _extract_controller_signal(expr_val)
                    if signal:
                        mapping[out_name] = signal
                if mapping:
                    extras.setdefault(
                        "__controller_sources__",
                        ",".join(f"{key}={value}" for key, value in mapping.items()),
                    )
        if descriptor.type_name == "LFONode" and node_obj is not None:
            extras.setdefault("wave", getattr(node_obj, "wave", "sine"))
            extras.setdefault("rate_hz", float(getattr(node_obj, "rate", 1.0)))
            extras.setdefault("depth", float(getattr(node_obj, "depth", 0.5)))
            extras.setdefault("use_input", 1 if getattr(node_obj, "use_input", False) else 0)
            extras.setdefault("slew_ms", float(getattr(node_obj, "slew_ms", 0.0)))
        if descriptor.type_name == "EnvelopeModulatorNode" and node_obj is not None:
            extras.setdefault("attack_ms", float(getattr(node_obj, "attack_ms", 0.0)))
            extras.setdefault("hold_ms", float(getattr(node_obj, "hold_ms", 0.0)))
            extras.setdefault("decay_ms", float(getattr(node_obj, "decay_ms", 0.0)))
            extras.setdefault("sustain_level", float(getattr(node_obj, "sustain_level", 0.0)))
            extras.setdefault("sustain_ms", float(getattr(node_obj, "sustain_ms", 0.0)))
            extras.setdefault("release_ms", float(getattr(node_obj, "release_ms", 0.0)))
            extras.setdefault("send_resets", 1 if getattr(node_obj, "send_resets", True) else 0)
        if descriptor.type_name == "PitchQuantizerNode" and node_obj is not None:
            token = getattr(node_obj, "effective_token", "12tet/full")
            extras.setdefault("effective_token", token)
            extras.setdefault("free_variant", getattr(node_obj, "free_variant", "continuous"))
            extras.setdefault("span_default", float(getattr(node_obj, "span_oct", 2.0)))
            extras.setdefault("slew", 1 if getattr(node_obj, "slew", True) else 0)
            try:
                grid = quantizer.get_reference_grid_cents(getattr(node_obj, "state", {}), token)
            except Exception:
                grid = [i * 100.0 for i in range(12)]
            extras.setdefault("grid_cents", ",".join(f"{float(val):.12g}" for val in grid))
            extras.setdefault("is_free_mode", 1 if quantizer.is_free_mode_token(token) else 0)
        if descriptor.type_name == "OscNode" and node_obj is not None:
            extras.setdefault("wave", getattr(node_obj, "wave", "sine"))
            extras.setdefault("accept_reset", 1 if getattr(node_obj, "accept_reset", True) else 0)
        if descriptor.type_name == "SubharmonicLowLifterNode" and node_obj is not None:
            extras.setdefault("band_lo", float(getattr(node_obj, "band_lo", 70.0)))
            extras.setdefault("band_hi", float(getattr(node_obj, "band_hi", 160.0)))
            extras.setdefault("mix", float(getattr(node_obj, "mix", 0.5)))
            extras.setdefault("drive", float(getattr(node_obj, "drive", 1.0)))
            extras.setdefault("out_hp", float(getattr(node_obj, "out_hp", 25.0)))
            extras.setdefault("use_div4", 1 if getattr(node_obj, "use_div4", False) else 0)
        if extras:
            merged = dict(params_dict)
            merged.update(extras)
        else:
            merged = params_dict
        params_json = json.dumps(merged, sort_keys=True) if merged else descriptor.params_json
        params_bytes = params_json.encode("utf-8")
        name_buf = self.ffi.new("char[]", name_bytes)
        type_buf = self.ffi.new("char[]", type_bytes)
        params_buf = self.ffi.new("char[]", params_bytes)
        struct_obj.name = name_buf
        struct_obj.name_len = len(name_bytes)
        struct_obj.type_name = type_buf
        struct_obj.type_len = len(type_bytes)
        struct_obj.params_json = params_buf
        struct_obj.params_len = len(params_bytes)
        keepers: tuple[Any, ...] = (struct_obj, name_buf, type_buf, params_buf)
        return struct_obj, keepers

    def _release_states(self) -> None:
        if not self._node_states:
            return
        lib = self._c_kernel
        if lib is None:
            self._node_states.clear()
            return
        for state in self._node_states.values():
            if state:
                lib.amp_release_state(state)
        self._node_states.clear()

__all__ = ["CffiEdgeRunner", "NodeInputHandle"]
