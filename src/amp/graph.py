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

    def connect_audio(self, source: str, target: str) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("Audio connections must reference defined nodes")
        self._audio_inputs.setdefault(target, []).append(source)
        self._audio_successors.setdefault(source, []).append(target)

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

    def set_sink(self, name: str) -> None:
        if name not in self._nodes:
            raise ValueError(f"Unknown sink node '{name}'")
        self.sink = name

    @property
    def ordered_nodes(self) -> Sequence[AudioNode]:
        return [self._nodes[name] for name in self._topo_order()]

    def mod_connections(self, target: str) -> Sequence[ModConnection]:
        """Return modulation connections arriving at ``target``."""

        return tuple(self._mod_inputs.get(target, ()))

    def _topo_order(self) -> Iterable[str]:
        if not self._nodes:
            return []
        incoming = {name: len(self._audio_inputs.get(name, [])) for name in self._nodes}
        outgoing: Dict[str, List[str]] = {
            name: list(self._audio_successors.get(name, [])) for name in self._nodes
        }
        for target, entries in self._mod_inputs.items():
            incoming[target] = incoming.get(target, 0) + len(entries)
            for entry in entries:
                outgoing.setdefault(entry.source, []).append(target)
        queue: Deque[str] = deque(
            name for name, count in incoming.items() if count == 0
        )
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
        return order

    def render_block(
        self,
        frames: int,
        sample_rate: int | None = None,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        if not self.sink:
            raise RuntimeError("Sink node has not been configured")
        sr = int(sample_rate or self.sample_rate)
        order = self._topo_order()
        caches: Dict[str, np.ndarray | None] = {name: None for name in self._nodes}
        for name in order:
            audio_inputs = []
            for predecessor in self._audio_inputs.get(name, []):
                buffer = caches[predecessor]
                if buffer is None:
                    continue
                audio_inputs.append(_assert_bcf(buffer, name=f"{predecessor}.out"))
            if audio_inputs:
                batches = audio_inputs[0].shape[0]
                frame_count = audio_inputs[0].shape[2]
                for buf in audio_inputs[1:]:
                    if buf.shape[0] != batches or buf.shape[2] != frame_count:
                        raise ValueError(f"Shape mismatch in inputs to '{name}'")
                audio_in = np.concatenate(audio_inputs, axis=1)
            else:
                audio_in = None
                batches = int(base_params.get("_B", 1)) if base_params else 1
                frame_count = frames
            channels = audio_in.shape[1] if audio_in is not None else int(base_params.get("_C", 1)) if base_params else 1
            node_params: Dict[str, np.ndarray] = {}
            if base_params and name in base_params:
                for key, value in base_params[name].items():
                    node_params[key] = _as_bcf(value, batches, channels, frame_count, name=f"{name}.{key}")
            mods: Dict[str, list[tuple[np.ndarray, float, str]]] = {}
            for entry in self._mod_inputs.get(name, []):
                buffer = caches[entry.source]
                if buffer is None:
                    continue
                target_param = entry.param or "value"
                buf = _assert_bcf(buffer, name=f"{entry.source}.out")
                channel = entry.channel
                if channel is not None:
                    if channel >= buf.shape[1]:
                        raise ValueError(
                            f"Mod channel {channel} out of range for '{entry.source}'"
                        )
                    buf = buf[:, channel : channel + 1, :]
                mods.setdefault(target_param, []).append(
                    (
                        _as_bcf(
                            buf,
                            batches,
                            channels,
                            frame_count,
                            name=f"mod {entry.source}->{name}",
                        ),
                        entry.scale,
                        entry.mode,
                    )
                )
            merged_params: Dict[str, np.ndarray] = dict(node_params)
            for param_name, entries in mods.items():
                base = merged_params.get(param_name, np.zeros((batches, channels, frame_count), dtype=RAW_DTYPE))
                for signal, scale, mode in entries:
                    if mode == "add":
                        base = base + signal * scale
                    else:
                        base = base * (1.0 + signal * scale)
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
