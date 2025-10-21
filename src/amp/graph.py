"""Unified audio graph runtime used across the application."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Mapping, Sequence, Tuple
import struct
import time
import zlib

import numpy as np

from .config import GraphConfig
from .nodes import NODE_TYPES, Node as AudioNode

if TYPE_CHECKING:
    from .graph_edge_runner import CffiEdgeRunner

RAW_DTYPE = np.float64


_NODE_DESCRIPTOR_HEADER = struct.Struct("<IIIIIIII")
_MOD_DESCRIPTOR_HEADER = struct.Struct("<IIIfi")
_PARAM_DESCRIPTOR_HEADER = struct.Struct("<IIIIQ")
_CONTROL_SAMPLE_HEADER = struct.Struct("<IIIII dd QQQQQQQ")
_EXTRA_ENTRY_HEADER = struct.Struct("<IIQ")

_MODE_CODES: dict[str, int] = {"add": 0, "mul": 1}


def _as_bcf(value, batches: int, channels: int, frames: int, *, name: str) -> np.ndarray:
    """Coerce ``value`` into ``(B, C, F)`` for parameter merging."""

    if np.isscalar(value):
        return np.full((batches, channels, frames), float(value), RAW_DTYPE)
    array = np.asarray(value, dtype=RAW_DTYPE)
    if array.ndim == 1 and array.shape[0] == frames:
        array = np.broadcast_to(array, (batches, channels, frames)).copy()
    elif array.ndim == 2:
        if array.shape == (batches, frames):
            array = array[:, None, :]
        elif array.shape == (channels, frames):
            array = array[None, :, :]
        else:
            raise ValueError(f"{name}: expected (*, {frames}), got {array.shape}")
    elif array.ndim == 3:
        pass
    else:
        raise ValueError(f"{name}: unsupported rank {array.ndim}")
    if array.shape[0] == 1 and batches > 1:
        array = np.broadcast_to(array, (batches, array.shape[1], frames)).copy()
    if array.shape[1] == 1 and channels > 1:
        array = np.broadcast_to(array, (array.shape[0], channels, frames)).copy()
    if array.shape != (batches, channels, frames):
        raise ValueError(f"{name}: got {array.shape}, expected {(batches, channels, frames)}")
    return array


def _assert_bcf(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=RAW_DTYPE)
    if array.ndim != 3:
        raise ValueError(f"{name}: expected (B, C, F); got rank {array.ndim}")
    return array


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    kind: str = "audio"


@dataclass(slots=True)
class ModConnection:
    """Control edge linking a modulation source to a target parameter."""

    source: str
    target: str
    param: str
    scale: float = 1.0
    mode: str = "add"
    channel: int | None = None


@dataclass(slots=True)
class ControlEvent:
    """Timestamped controller state used for read-ahead interpolation."""

    timestamp: float
    pitch: np.ndarray
    envelope: np.ndarray
    extras: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.pitch = np.atleast_1d(np.asarray(self.pitch, dtype=RAW_DTYPE))
        if self.pitch.ndim != 1:
            raise ValueError("pitch events must be 1-D sequences")
        self.envelope = np.atleast_1d(np.asarray(self.envelope, dtype=RAW_DTYPE))
        if self.envelope.ndim != 1:
            raise ValueError("envelope events must be 1-D sequences")
        if self.extras:
            self.extras = {key: np.asarray(value, dtype=RAW_DTYPE) for key, value in self.extras.items()}


@dataclass(slots=True)
class PcmChunk:
    """Block of PCM samples aligned to an absolute timeline."""

    timestamp: float
    data: np.ndarray
    sample_rate: float

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.sample_rate = float(self.sample_rate)
        array = np.asarray(self.data, dtype=RAW_DTYPE)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError("pcm data must be shaped (C, F)")
        self.data = array

    @property
    def frames(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def end_time(self) -> float:
        return self.timestamp + self.frames / self.sample_rate


class ControlDelay:
    """Retains controller history for pre-buffered read-ahead rendering."""

    def __init__(
        self,
        sample_rate: float,
        *,
        history_seconds: float = 1.0,
        lookahead_seconds: float = 0.25,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.history_seconds = float(history_seconds)
        self.lookahead_seconds = float(lookahead_seconds)
        self._events: Deque[ControlEvent] = deque()
        self._pcm: Deque[PcmChunk] = deque()
        self._latest_time: float = 0.0
        self._pitch_dim: int | None = None
        self._env_dim: int | None = None

    @property
    def events(self) -> Tuple[ControlEvent, ...]:
        return tuple(self._events)

    @property
    def pcm_chunks(self) -> Tuple[PcmChunk, ...]:
        return tuple(self._pcm)

    @property
    def latest_time(self) -> float:
        return self._latest_time

    def record_event(
        self,
        timestamp: float,
        *,
        pitch: np.ndarray | float,
        envelope: np.ndarray | float,
        extras: Mapping[str, np.ndarray] | None = None,
    ) -> ControlEvent:
        event = ControlEvent(timestamp, pitch, envelope, extras)
        self._insert_event(event)
        self._pitch_dim = event.pitch.shape[0] if self._pitch_dim is None else self._pitch_dim
        self._env_dim = event.envelope.shape[0] if self._env_dim is None else self._env_dim
        if event.pitch.shape[0] != self._pitch_dim:
            raise ValueError("pitch dimensionality mismatch in control history")
        if event.envelope.shape[0] != self._env_dim:
            raise ValueError("envelope dimensionality mismatch in control history")
        self._latest_time = max(self._latest_time, event.timestamp)
        self._trim_history()
        return event

    def add_pcm(
        self,
        timestamp: float,
        data: np.ndarray,
        *,
        sample_rate: float | None = None,
    ) -> PcmChunk:
        chunk = PcmChunk(timestamp, data, sample_rate or self.sample_rate)
        if not np.isclose(chunk.sample_rate, self.sample_rate):
            raise ValueError("PCM chunk sample rate must match the graph sample rate")
        self._insert_pcm(chunk)
        self._latest_time = max(self._latest_time, chunk.end_time)
        self._trim_history()
        return chunk

    def sample(
        self,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
    ) -> dict[str, np.ndarray]:
        if frames <= 0:
            raise ValueError("frames must be a positive integer")
        update_rate = float(update_hz or self.sample_rate)
        step = 1.0 / update_rate
        times = start_time + np.arange(frames, dtype=RAW_DTYPE) * step
        pitch = self._interpolate_series(times, "pitch", self._pitch_dim)
        envelope = self._interpolate_series(times, "envelope", self._env_dim)
        timestamps = times[:, None]
        controls = np.concatenate([pitch, envelope, timestamps], axis=1)
        pcm = self._gather_pcm(start_time, frames)

        # Provide a simple sampling of any extras attached to the latest
        # control event at or before the requested start_time. Headless
        # benchmarks record per-block arrays into event.extras, so this
        # returns those arrays unchanged when available. For interactive
        # inputs that record instantaneous snapshots, callers should
        # continue to provide fallbacks (this function avoids aggressive
        # interpretation and only returns a best-effort extras mapping).
        extras_out: dict[str, np.ndarray] = {}
        if self._events:
            # Prefer the most-recent event at or before start_time
            candidates = [e for e in self._events if e.timestamp <= start_time]
            latest = candidates[-1] if candidates else self._events[0]
            if latest.extras:
                for key, val in latest.extras.items():
                    arr = np.asarray(val, dtype=RAW_DTYPE)
                    # If the extras entry is a scalar, broadcast to frames
                    if arr.ndim == 0:
                        extras_out[key] = np.full(frames, float(arr), dtype=RAW_DTYPE)
                    elif arr.ndim == 1:
                        if arr.shape[0] == frames:
                            extras_out[key] = arr.copy()
                        elif arr.shape[0] == 1:
                            extras_out[key] = np.full(frames, float(arr[0]), dtype=RAW_DTYPE)
                        elif arr.shape[0] > frames:
                            # If the recorded extras contains a longer array,
                            # slice to the requested block length.
                            extras_out[key] = arr[:frames].copy()
                        else:
                            # Pad with the final value when shorter than frames.
                            pad = np.full(frames - arr.shape[0], float(arr[-1]), dtype=RAW_DTYPE)
                            extras_out[key] = np.concatenate([arr, pad])
                    else:
                        # For higher-rank extras, pass through as-is (caller
                        # must handle shape). Convert to RAW_DTYPE.
                        extras_out[key] = arr.copy()

        return {
            "times": times,
            "pitch": pitch,
            "envelope": envelope,
            "control_tensor": controls,
            "pcm": pcm,
            "extras": extras_out,
        }

    def export_sample_block(
        self,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
    ) -> bytes:
        """Serialize a sampled control block into a C-compatible layout."""

        block = self.sample(start_time, frames, update_hz=update_hz)
        times = np.ascontiguousarray(block["times"], dtype=RAW_DTYPE)
        pitch = np.ascontiguousarray(block["pitch"], dtype=RAW_DTYPE)
        envelope = np.ascontiguousarray(block["envelope"], dtype=RAW_DTYPE)
        control_tensor = np.ascontiguousarray(block["control_tensor"], dtype=RAW_DTYPE)
        pcm = np.ascontiguousarray(block["pcm"], dtype=RAW_DTYPE)

        pitch_dim = int(pitch.shape[1]) if pitch.ndim == 2 else 0
        envelope_dim = int(envelope.shape[1]) if envelope.ndim == 2 else 0
        extras = block.get("extras", {}) or {}
        pcm_channels = int(pcm.shape[0]) if pcm.ndim == 2 else 0

        payload = bytearray(_CONTROL_SAMPLE_HEADER.size)
        offset = _CONTROL_SAMPLE_HEADER.size

        def _append_array(array: np.ndarray) -> tuple[int, int]:
            nonlocal offset
            if array.size == 0:
                return 0, 0
            data = array.tobytes(order="C")
            start = offset
            payload.extend(data)
            offset += len(data)
            return start, len(data)

        times_offset, _ = _append_array(times)
        pitch_offset, _ = _append_array(pitch)
        envelope_offset, _ = _append_array(envelope)
        control_offset, _ = _append_array(control_tensor)
        pcm_offset, _ = _append_array(pcm)

        extras_offset = offset
        payload.extend(struct.pack("<I", len(extras)))
        offset += 4
        for key, value in extras.items():
            key_bytes = key.encode("utf-8")
            array = np.ascontiguousarray(value, dtype=RAW_DTYPE)
            rank = array.ndim
            dims = array.shape
            data = array.tobytes(order="C")
            payload.extend(_EXTRA_ENTRY_HEADER.pack(len(key_bytes), rank, len(data)))
            offset += _EXTRA_ENTRY_HEADER.size
            payload.extend(key_bytes)
            offset += len(key_bytes)
            if rank:
                payload.extend(struct.pack(f"<{rank}I", *dims))
                offset += rank * 4
            payload.extend(data)
            offset += len(data)

        total_size = len(payload)

        _CONTROL_SAMPLE_HEADER.pack_into(
            payload,
            0,
            int(frames),
            pitch_dim,
            envelope_dim,
            len(extras),
            pcm_channels,
            float(start_time),
            float(update_hz or self.sample_rate),
            times_offset,
            pitch_offset,
            envelope_offset,
            control_offset,
            pcm_offset,
            extras_offset,
            total_size,
        )
        return bytes(payload)

    def lookahead_events(self, start_time: float) -> Tuple[ControlEvent, ...]:
        window_end = start_time + self.lookahead_seconds
        return tuple(event for event in self._events if start_time <= event.timestamp <= window_end)

    def _insert_event(self, event: ControlEvent) -> None:
        if not self._events or event.timestamp >= self._events[-1].timestamp:
            self._events.append(event)
            return
        timestamps = [entry.timestamp for entry in self._events]
        idx = bisect_right(timestamps, event.timestamp)
        self._events.insert(idx, event)

    def _insert_pcm(self, chunk: PcmChunk) -> None:
        if not self._pcm or chunk.timestamp >= self._pcm[-1].timestamp:
            self._pcm.append(chunk)
            return
        timestamps = [entry.timestamp for entry in self._pcm]
        idx = bisect_right(timestamps, chunk.timestamp)
        self._pcm.insert(idx, chunk)

    def _trim_history(self) -> None:
        cutoff = self._latest_time - self.history_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()
        while self._pcm and self._pcm[0].end_time < cutoff:
            self._pcm.popleft()

    def _interpolate_series(
        self,
        times: np.ndarray,
        attr: str,
        expected_dim: int | None,
    ) -> np.ndarray:
        if not self._events:
            dim = expected_dim or 1
            return np.zeros((times.size, dim), dtype=RAW_DTYPE)
        event_times = np.array([event.timestamp for event in self._events], dtype=RAW_DTYPE)
        values = np.stack([getattr(event, attr) for event in self._events])
        if values.ndim == 1:
            values = values[:, None]
        result = np.zeros((times.size, values.shape[1]), dtype=RAW_DTYPE)
        for column in range(values.shape[1]):
            result[:, column] = np.interp(
                times,
                event_times,
                values[:, column],
                left=values[0, column],
                right=values[-1, column],
            )
        return result

    def _gather_pcm(self, start_time: float, frames: int) -> np.ndarray:
        if not self._pcm:
            return np.zeros((1, frames), dtype=RAW_DTYPE)
        out_channels = self._pcm[-1].channels
        output = np.zeros((out_channels, frames), dtype=RAW_DTYPE)
        frame_step = 1.0 / self.sample_rate
        end_time = start_time + frames * frame_step
        for chunk in self._pcm:
            chunk_start = chunk.timestamp
            chunk_end = chunk.end_time
            if chunk_end <= start_time or chunk_start >= end_time:
                continue
            overlap_start = max(chunk_start, start_time)
            overlap_end = min(chunk_end, end_time)
            local_start = int(round((overlap_start - chunk_start) * self.sample_rate))
            local_end = int(round((overlap_end - chunk_start) * self.sample_rate))
            out_start = int(round((overlap_start - start_time) * self.sample_rate))
            out_end = out_start + (local_end - local_start)
            if chunk.channels != out_channels:
                raise ValueError("PCM chunk channel count mismatch in history")
            output[:, out_start:out_end] = chunk.data[:, local_start:local_end]
        return output


@dataclass(slots=True)
class _ModGroupPlan:
    param: str
    connections: Tuple[ModConnection, ...]


@dataclass(slots=True)
class _NodeExecutionPlan:
    name: str
    audio_inputs: Tuple[str, ...]
    mod_groups: Tuple[_ModGroupPlan, ...]


class AudioGraph:
    """Directed audio processing graph supporting modulation links."""

    def __init__(self, sample_rate: int, output_channels: int | None = None) -> None:
        self.sample_rate = int(sample_rate)
        self.output_channels = int(output_channels) if output_channels is not None else None
        self._nodes: Dict[str, AudioNode] = {}
        self._audio_inputs: Dict[str, List[str]] = {}
        self._audio_successors: Dict[str, List[str]] = {}
        self._mod_inputs: Dict[str, List[ModConnection]] = {}
        self.sink: str | None = None
        self._levels_lock = Lock()
        self._last_node_levels: Dict[str, np.ndarray] = {}
        self.control_delay = ControlDelay(self.sample_rate)
        self._plan_dirty = True
        self._execution_plan: Tuple[_NodeExecutionPlan, ...] = ()
        self._ordered_node_names: Tuple[str, ...] = ()
        self._param_buffers: Dict[
            Tuple[str, str], Dict[Tuple[int, int, int], np.ndarray]
        ] = {}
        self._mod_buffers: Dict[
            Tuple[str, str, str, int | None], Dict[Tuple[int, int, int], np.ndarray]
        ] = {}
        self._merge_scratch: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._audio_workspaces: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._last_node_timings: Dict[str, float] = {}
        self._edge_runner: "CffiEdgeRunner" | None = None

    @classmethod
    def from_config(cls, config: GraphConfig, sample_rate: int, output_channels: int) -> "AudioGraph":
        graph = cls(sample_rate=sample_rate, output_channels=output_channels)
        for node_cfg in config.nodes:
            node_type = node_cfg.type.lower()
            try:
                node_cls = NODE_TYPES[node_type]
            except KeyError as exc:
                raise KeyError(f"Unknown node type '{node_cfg.type}'") from exc
            node = node_cls(node_cfg.name, node_cfg.params)
            graph.add_node(node)
        for connection in config.connections:
            kind = connection.kind.lower()
            if kind == "audio":
                graph.connect_audio(connection.source, connection.target)
                continue
            if kind in {"mod", "modulation"}:
                channel = connection.channel
                if isinstance(channel, str):
                    source_node = graph._nodes.get(connection.source)
                    if source_node is None:
                        raise ValueError(
                            f"Mod connection references unknown source '{connection.source}'"
                        )
                    if not hasattr(source_node, "output_index"):
                        raise ValueError(
                            f"Source node '{connection.source}' does not expose named outputs"
                        )
                    try:
                        channel = source_node.output_index(channel)
                    except Exception as exc:  # pragma: no cover - defensive
                        raise ValueError(
                            f"Source node '{connection.source}' has no output named '{connection.channel}'"
                        ) from exc
                graph.connect_mod(
                    connection.source,
                    connection.target,
                    connection.param or "",
                    scale=connection.scale,
                    mode=connection.mode,
                    channel=channel if channel is None else int(channel),
                )
                continue
            raise ValueError(f"Unsupported connection kind '{connection.kind}' in configuration")
        graph.set_sink(config.sink)
        return graph

    def add_node(self, node: AudioNode) -> None:
        self._nodes[node.name] = node
        self._audio_inputs.setdefault(node.name, [])
        self._audio_successors.setdefault(node.name, [])
        self._mod_inputs.setdefault(node.name, [])
        self._invalidate_plan()

    def connect_audio(self, source: str, target: str) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("Audio connections must reference defined nodes")
        self._audio_inputs.setdefault(target, []).append(source)
        self._audio_successors.setdefault(source, []).append(target)
        self._invalidate_plan()

    def connect_mod(
        self,
        source: str,
        target: str,
        param: str,
        *,
        scale: float = 1.0,
        mode: str = "add",
        channel: int | None = None,
    ) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("Mod connections must reference defined nodes")
        if channel is not None and channel < 0:
            raise ValueError("Mod channel index must be non-negative")
        connection = ModConnection(
            source=source,
            target=target,
            param=param,
            scale=float(scale),
            mode=mode,
            channel=int(channel) if channel is not None else None,
        )
        self._mod_inputs.setdefault(target, []).append(connection)
        self._invalidate_plan()

    def set_sink(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Unknown sink node '{name}'")
        self.sink = name
        # Sink changes do not affect topology but may change execution order queries.
        # Keep the cached plan to ensure downstream lookups remain consistent.

    @property
    def ordered_nodes(self) -> Sequence[AudioNode]:
        plan = self._ensure_execution_plan()
        return [self._nodes[entry.name] for entry in plan]

    def mod_connections(self, target: str) -> Sequence[ModConnection]:
        """Return modulation connections arriving at ``target``."""

        return tuple(self._mod_inputs.get(target, ()))

    def _topo_order(self) -> Iterable[str]:
        return tuple(entry.name for entry in self._ensure_execution_plan())

    def _invalidate_plan(self) -> None:
        self._plan_dirty = True
        self._edge_runner = None

    def _ensure_execution_plan(self) -> Tuple[_NodeExecutionPlan, ...]:
        if not self._plan_dirty:
            return self._execution_plan
        plan = self._build_execution_plan()
        self._execution_plan = plan
        self._ordered_node_names = tuple(entry.name for entry in plan)
        self._plan_dirty = False
        return plan

    def _build_execution_plan(self) -> Tuple[_NodeExecutionPlan, ...]:
        if not self._nodes:
            return ()
        incoming = {name: len(self._audio_inputs.get(name, [])) for name in self._nodes}
        outgoing: Dict[str, List[str]] = {
            name: list(self._audio_successors.get(name, [])) for name in self._nodes
        }
        for target, entries in self._mod_inputs.items():
            incoming[target] = incoming.get(target, 0) + len(entries)
            for entry in entries:
                outgoing.setdefault(entry.source, []).append(target)
        queue: Deque[str] = deque(name for name, count in incoming.items() if count == 0)
        order: List[str] = []
        while queue:
            name = queue.popleft()
            order.append(name)
            for successor in outgoing.get(name, []):
                incoming[successor] -= 1
                if incoming[successor] == 0:
                    queue.append(successor)
        if len(order) != len(self._nodes):
            raise ValueError("Graph contains cycles or disconnected nodes")
        plan: List[_NodeExecutionPlan] = []
        valid_mod_keys: set[Tuple[str, str, str, int | None]] = set()
        for name in order:
            audio_inputs = tuple(self._audio_inputs.get(name, ()))
            mod_entries = self._mod_inputs.get(name, ())
            groups: List[_ModGroupPlan] = []
            if mod_entries:
                grouped: Dict[str, List[ModConnection]] = {}
                param_order: List[str] = []
                for connection in mod_entries:
                    param = connection.param or "value"
                    if param not in grouped:
                        grouped[param] = []
                        param_order.append(param)
                    grouped[param].append(connection)
                    valid_mod_keys.add((connection.source, name, param, connection.channel))
                groups = [_ModGroupPlan(param, tuple(grouped[param])) for param in param_order]
            plan.append(_NodeExecutionPlan(name, audio_inputs, tuple(groups)))
        if self._mod_buffers:
            self._mod_buffers = {
                key: buffers
                for key, buffers in self._mod_buffers.items()
                if key in valid_mod_keys
            }
        return tuple(plan)

    def _prepare_param_buffer(
        self,
        node: str,
        param: str,
        value: np.ndarray | float,
        batches: int,
        channels: int,
        frames: int,
    ) -> np.ndarray:
        key = (node, param)
        target_shape = (batches, channels, frames)
        shape_buffers = self._param_buffers.setdefault(key, {})
        buffer = shape_buffers.get(target_shape)
        if buffer is None:
            buffer = np.empty(target_shape, dtype=RAW_DTYPE)
            shape_buffers[target_shape] = buffer
        array = np.asarray(value, dtype=RAW_DTYPE)
        self._copy_to_bcf(array, buffer, batches, channels, frames, name=f"{node}.{param}")
        return buffer

    def _copy_to_bcf(
        self,
        source: np.ndarray,
        dest: np.ndarray,
        batches: int,
        channels: int,
        frames: int,
        *,
        name: str,
    ) -> None:
        if source.ndim == 0:
            dest.fill(float(source))
            return
        if source.ndim == 1:
            if source.shape[0] != frames:
                raise ValueError(f"{name}: expected length {frames}, got {source.shape[0]}")
            dest[...] = source.reshape(1, 1, frames)
            return
        if source.ndim == 2:
            if source.shape[1] != frames:
                raise ValueError(f"{name}: expected (*, {frames}); got {source.shape}")
            if source.shape == (batches, frames):
                reshaped = source.reshape(batches, 1, frames)
            elif source.shape == (channels, frames):
                reshaped = source.reshape(1, channels, frames)
            else:
                raise ValueError(f"{name}: expected {(batches, frames)} or {(channels, frames)}; got {source.shape}")
            np.copyto(dest, reshaped)
            return
        if source.ndim == 3:
            if source.shape[2] != frames:
                raise ValueError(f"{name}: expected frame count {frames}; got {source.shape[2]}")
            np.copyto(dest, source)
            return
        raise ValueError(f"{name}: unsupported rank {source.ndim}")

    def _prepare_mod_buffer(
        self,
        target_node: str,
        param: str,
        connection: ModConnection,
        source_buffer: np.ndarray,
        batches: int,
        channels: int,
        frames: int,
    ) -> np.ndarray:
        key = (connection.source, target_node, param, connection.channel)
        target_shape = (batches, channels, frames)
        shape_buffers = self._mod_buffers.setdefault(key, {})
        buffer = shape_buffers.get(target_shape)
        if buffer is None:
            buffer = np.empty(target_shape, dtype=RAW_DTYPE)
            shape_buffers[target_shape] = buffer
        if connection.channel is not None:
            if connection.channel >= source_buffer.shape[1]:
                raise ValueError(
                    f"Mod channel {connection.channel} out of range for '{connection.source}'"
                )
            sliced = source_buffer[:, connection.channel : connection.channel + 1, :]
        else:
            sliced = source_buffer
        self._copy_to_bcf(sliced, buffer, batches, channels, frames, name=f"mod {connection.source}->{target_node}")
        return buffer

    def _acquire_merge_scratch(self, shape: Tuple[int, int, int]) -> np.ndarray:
        buffer = self._merge_scratch.get(shape)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=RAW_DTYPE)
            self._merge_scratch[shape] = buffer
        return buffer

    def _acquire_audio_workspace(self, shape: Tuple[int, int, int]) -> np.ndarray:
        buffer = self._audio_workspaces.get(shape)
        if buffer is None or buffer.shape != shape:
            buffer = np.empty(shape, dtype=RAW_DTYPE)
            self._audio_workspaces[shape] = buffer
        return buffer

    def _ensure_edge_runner(self) -> "CffiEdgeRunner":
        if self._edge_runner is None:
            try:
                from .graph_edge_runner import CffiEdgeRunner
            except Exception as exc:  # pragma: no cover - defensive import guard
                raise RuntimeError("cffi edge runner is required for graph rendering") from exc
            self._edge_runner = CffiEdgeRunner(self)
        return self._edge_runner

    def render_block(
        self,
        frames: int,
        sample_rate: int | None = None,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        if not self.sink:
            raise RuntimeError("Sink node has not been configured")
        sr = int(sample_rate or self.sample_rate)
        plan = self._ensure_execution_plan()
        payload = dict(base_params) if base_params is not None else {}
        runner = self._ensure_edge_runner()
        runner.begin_block(frames, sample_rate=sr, base_params=payload)
        node_timings: Dict[str, float] = {}
        for node in self._nodes.values():
            recycle = getattr(node, "recycle_blocks", None)
            if recycle is not None:
                recycle()
        for entry in plan:
            node = self._nodes[entry.name]
            handle = runner.gather_to(entry.name)
            params = handle.params
            audio_in = handle.audio
            start = time.perf_counter()
            output = node.process(handle.frames, sr, audio_in, {}, params)
            node_timings[entry.name] = time.perf_counter() - start
            runner.set_node_output(entry.name, output)
        caches = {name: runner.get_cached_output(name) for name in runner.ordered_nodes}
        sink_output = caches.get(self.sink)
        if sink_output is None:
            raise RuntimeError(f"Sink node '{self.sink}' produced no data")
        with self._levels_lock:
            self._last_node_levels = {
                name: np.max(np.abs(buf), axis=2)
                for name, buf in caches.items()
                if buf is not None
            }
            self._last_node_timings = dict(node_timings)
        return sink_output

    def render(
        self,
        frames: int,
        sample_rate: int | None = None,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        block = self.render_block(frames, sample_rate, base_params)
        if base_params is not None or block.shape[0] != 1:
            return block
        channels = block.shape[1]
        data = block[0]
        if self.output_channels is not None:
            if channels < self.output_channels:
                pad = np.zeros((self.output_channels - channels, frames), dtype=block.dtype)
                data = np.concatenate([data, pad], axis=0)
            elif channels > self.output_channels:
                data = data[: self.output_channels]
        return data

    def serialize_node_descriptors(self) -> bytes:
        """Serialize the graph topology and parameter buffers for C tooling."""

        plan = self._ensure_execution_plan()
        payload = bytearray(struct.pack("<I", len(plan)))
        for entry in plan:
            node = self._nodes[entry.name]
            type_name = type(node).__name__
            type_bytes = type_name.encode("utf-8")
            name_bytes = entry.name.encode("utf-8")
            type_id = zlib.crc32(type_bytes) & 0xFFFFFFFF

            audio_inputs = tuple(self._audio_inputs.get(entry.name, ()))
            mod_connections = list(self._mod_inputs.get(entry.name, ()))

            params_json = json.dumps(getattr(node, "params", {}), sort_keys=True).encode("utf-8")

            param_buffers: list[tuple[str, tuple[int, int, int], bytes]] = []
            for (node_name, param), buffers in self._param_buffers.items():
                if node_name != entry.name:
                    continue
                for shape, array in buffers.items():
                    if array is None:
                        continue
                    contiguous = np.ascontiguousarray(array, dtype=RAW_DTYPE)
                    param_buffers.append((param, tuple(map(int, shape)), contiguous.tobytes(order="C")))

            buffer_shapes: set[tuple[int, int, int]] = {shape for _, shape, _ in param_buffers}
            for (source, target, _param, _channel), buffers in self._mod_buffers.items():
                if target != entry.name:
                    continue
                for shape in buffers:
                    buffer_shapes.add(tuple(map(int, shape)))

            payload.extend(
                _NODE_DESCRIPTOR_HEADER.pack(
                    type_id,
                    len(name_bytes),
                    len(type_bytes),
                    len(audio_inputs),
                    len(mod_connections),
                    len(param_buffers),
                    len(buffer_shapes),
                    len(params_json),
                )
            )
            payload.extend(name_bytes)
            payload.extend(type_bytes)

            for source in audio_inputs:
                src_bytes = source.encode("utf-8")
                payload.extend(struct.pack("<I", len(src_bytes)))
                payload.extend(src_bytes)

            for connection in mod_connections:
                source_bytes = connection.source.encode("utf-8")
                param_bytes = (connection.param or "").encode("utf-8")
                mode_code = _MODE_CODES.get(connection.mode, 0xFFFFFFFF)
                channel = connection.channel if connection.channel is not None else -1
                payload.extend(
                    _MOD_DESCRIPTOR_HEADER.pack(
                        len(source_bytes),
                        len(param_bytes),
                        mode_code,
                        float(connection.scale),
                        int(channel),
                    )
                )
                payload.extend(source_bytes)
                payload.extend(param_bytes)

            for param_name, shape, blob in param_buffers:
                name_bytes = param_name.encode("utf-8")
                payload.extend(
                    _PARAM_DESCRIPTOR_HEADER.pack(
                        len(name_bytes),
                        int(shape[0]) if shape else 0,
                        int(shape[1]) if len(shape) > 1 else 0,
                        int(shape[2]) if len(shape) > 2 else 0,
                        len(blob),
                    )
                )
                payload.extend(name_bytes)
                payload.extend(blob)

            for shape in sorted(buffer_shapes):
                payload.extend(struct.pack("<III", *shape))

            payload.extend(params_json)

        return bytes(payload)

    @property
    def last_node_levels(self) -> Dict[str, np.ndarray]:
        with self._levels_lock:
            return {name: levels.copy() for name, levels in self._last_node_levels.items()}

    @property
    def last_node_timings(self) -> Dict[str, float]:
        with self._levels_lock:
            return dict(self._last_node_timings)

    def record_control_event(
        self,
        timestamp: float,
        *,
        pitch: np.ndarray | float,
        envelope: np.ndarray | float,
        extras: Mapping[str, np.ndarray] | None = None,
    ) -> ControlEvent:
        """Store a controller snapshot for future read-ahead sampling."""

        return self.control_delay.record_event(timestamp, pitch=pitch, envelope=envelope, extras=extras)

    def add_pcm_history(
        self,
        timestamp: float,
        data: np.ndarray,
        *,
        sample_rate: float | None = None,
    ) -> PcmChunk:
        """Append PCM history aligned to the graph timeline."""

        return self.control_delay.add_pcm(timestamp, data, sample_rate=sample_rate)

    def sample_control_tensor(
        self,
        start_time: float,
        frames: int,
        *,
        update_hz: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate an interpolated control tensor from retained history."""

        return self.control_delay.sample(start_time, frames, update_hz=update_hz)


__all__ = [
    "AudioGraph",
    "ControlDelay",
    "ControlEvent",
    "GraphEdge",
    "ModConnection",
    "PcmChunk",
]
