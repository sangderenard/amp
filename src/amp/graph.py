"""Unified audio graph runtime used across the application."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np

from .config import GraphConfig
from .nodes import NODE_TYPES, Node as AudioNode

RAW_DTYPE = np.float64


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
        return {
            "times": times,
            "pitch": pitch,
            "envelope": envelope,
            "control_tensor": controls,
            "pcm": pcm,
        }

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
        caches: Dict[str, np.ndarray | None] = {name: None for name in self._nodes}
        for entry in plan:
            name = entry.name
            audio_inputs: List[np.ndarray] = []
            for predecessor in entry.audio_inputs:
                buffer = caches.get(predecessor)
                if buffer is None:
                    continue
                audio_inputs.append(_assert_bcf(buffer, name=f"{predecessor}.out"))
            if audio_inputs:
                batches = audio_inputs[0].shape[0]
                frame_count = audio_inputs[0].shape[2]
                if len(audio_inputs) == 1:
                    audio_in = audio_inputs[0]
                    channels = audio_in.shape[1]
                else:
                    total_channels = audio_inputs[0].shape[1]
                    for buf in audio_inputs[1:]:
                        if buf.shape[0] != batches or buf.shape[2] != frame_count:
                            raise ValueError(f"Shape mismatch in inputs to '{name}'")
                        total_channels += buf.shape[1]
                    workspace = self._acquire_audio_workspace(
                        (batches, total_channels, frame_count)
                    )
                    offset = 0
                    for buf in audio_inputs:
                        channels_slice = buf.shape[1]
                        target = workspace[:, offset : offset + channels_slice, :]
                        np.copyto(target, buf)
                        offset += channels_slice
                    audio_in = workspace[:, :total_channels, :]
                    channels = total_channels
            else:
                audio_in = None
                batches = int(base_params.get("_B", 1)) if base_params else 1
                frame_count = frames
                channels = int(base_params.get("_C", 1)) if base_params else 1
            node_params: Dict[str, np.ndarray] = {}
            if base_params and name in base_params:
                for key, value in base_params[name].items():
                    node_params[key] = self._prepare_param_buffer(
                        name,
                        key,
                        value,
                        batches,
                        channels,
                        frame_count,
                    )
            mods: Dict[str, List[tuple[np.ndarray, float, str]]] = {}
            for group in entry.mod_groups:
                signals: List[tuple[np.ndarray, float, str]] = []
                for connection in group.connections:
                    buffer = caches.get(connection.source)
                    if buffer is None:
                        continue
                    source_buf = _assert_bcf(buffer, name=f"{connection.source}.out")
                    mod_signal = self._prepare_mod_buffer(
                        name,
                        group.param,
                        connection,
                        source_buf,
                        batches,
                        channels,
                        frame_count,
                    )
                    signals.append((mod_signal, connection.scale, connection.mode))
                if signals:
                    mods[group.param] = signals
            merged_params: Dict[str, np.ndarray] = dict(node_params)
            if mods:
                shape = (batches, channels, frame_count)
                scratch = self._acquire_merge_scratch(shape)
                for param_name, entries in mods.items():
                    base = merged_params.get(
                        param_name,
                        self._prepare_param_buffer(
                            name,
                            param_name,
                            0.0,
                            batches,
                            channels,
                            frame_count,
                        ),
                    )
                    for signal, scale, mode in entries:
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
                    merged_params[param_name] = base
            node = self._nodes[name]
            output = node.process(frame_count, sr, audio_in, mods, merged_params)
            caches[name] = _assert_bcf(output, name=f"{name}.out")
        with self._levels_lock:
            self._last_node_levels = {
                name: np.max(np.abs(buf), axis=2)
                for name, buf in caches.items()
                if buf is not None
            }
        sink_output = caches[self.sink]
        if sink_output is None:
            raise RuntimeError(f"Sink node '{self.sink}' produced no data")
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

    @property
    def last_node_levels(self) -> Dict[str, np.ndarray]:
        with self._levels_lock:
            return {name: levels.copy() for name, levels in self._last_node_levels.items()}

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
