# nodes.py
from __future__ import annotations

import math
import queue
import threading
from collections.abc import Mapping, Sequence

import numpy as np

from . import envelope, quantizer, utils
from .block_pool import BlockLease, BlockPool
from .utils import (
    as_BCF,
    assert_BCF,
    dc_block,
    make_wave_hq,
    osc_saw_blep,
    osc_square_blep,
    osc_triangle_blep,
    soft_clip,
)
from . import c_kernels
from .state import RAW_DTYPE, MAX_FRAMES


class _EnvelopeGroupState:
    """Runtime helper that distributes triggers across grouped envelopes."""

    def __init__(self) -> None:
        self.members: list[str] = []
        self._assignments: dict[str, dict[str, np.ndarray]] = {}
        self._next_voice = 0
        self._block_token: int | None = None
        self._latched_voice: np.ndarray = np.empty(0, dtype=np.int32)

    def register(self, name: str) -> None:
        if name not in self.members:
            self.members.append(name)

    def reset(self) -> None:
        self._assignments.clear()
        self._block_token = None
        self._latched_voice = np.empty(0, dtype=np.int32)
        self._next_voice = 0

    def _token(self, trigger: np.ndarray) -> int:
        ptr = trigger.__array_interface__["data"][0]
        return hash((ptr, trigger.shape))

    def assign(
        self,
        name: str,
        trigger: np.ndarray,
        gate: np.ndarray,
        drone: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.members or name not in self.members:
            return trigger, gate, drone, velocity

        token = self._token(trigger)
        if token != self._block_token:
            self._prepare_assignments(trigger, gate, drone, velocity, token)
        bundles = self._assignments.get(name)
        if bundles is None:
            zero = np.zeros_like(trigger)
            return zero, gate, drone, velocity
        return bundles["trigger"], bundles["gate"], bundles["drone"], bundles["velocity"]

    def _prepare_assignments(
        self,
        trigger: np.ndarray,
        gate: np.ndarray,
        drone: np.ndarray,
        velocity: np.ndarray,
        token: int,
    ) -> None:
        B, F = trigger.shape
        assignments: dict[str, dict[str, np.ndarray]] = {}
        dtype = trigger.dtype
        for member in self.members:
            assignments[member] = {
                "trigger": np.zeros((B, F), dtype=dtype),
                "gate": np.zeros((B, F), dtype=dtype),
                "drone": np.zeros((B, F), dtype=dtype),
                "velocity": np.zeros((B, F), dtype=dtype),
            }

        member_count = len(self.members)
        if member_count:
            if self._latched_voice.shape != (B,):
                self._latched_voice = np.full(B, -1, dtype=np.int32)

            triggers_active = trigger > 0.5
            gate_active = gate > 0.0
            drone_active = drone > 0.0
            sustain_active = gate_active | drone_active

            idx = self._next_voice % member_count
            flat_triggers = triggers_active.T.reshape(-1)
            trigger_count = int(np.count_nonzero(flat_triggers))
            trigger_voice_indices = np.full(flat_triggers.shape, -1, dtype=np.int32)
            if trigger_count:
                voice_sequence = (np.arange(trigger_count, dtype=np.int32) + idx) % member_count
                trigger_voice_indices[flat_triggers] = voice_sequence
            trigger_voice_indices = trigger_voice_indices.reshape(F, B).T

            latched_history = np.empty((B, F), dtype=np.int32)
            current_voice = self._latched_voice.copy()
            for frame in range(F):
                new_voice = trigger_voice_indices[:, frame]
                trig_frame = triggers_active[:, frame]
                candidate_voice = np.where(trig_frame, new_voice, current_voice)
                release_mask = (
                    (~trig_frame)
                    & (gate[:, frame] <= 0.5)
                    & (drone[:, frame] <= 0.5)
                )
                current_voice = np.where(release_mask, -1, candidate_voice)
                latched_history[:, frame] = current_voice

            trigger_voice_map = np.full((B, F), -1, dtype=np.int32)
            trigger_voice_map[triggers_active] = trigger_voice_indices[triggers_active]

            for member_idx, member in enumerate(self.members):
                latched_mask = latched_history == member_idx
                trigger_mask = trigger_voice_map == member_idx

                assignments[member]["trigger"][trigger_mask] = trigger[trigger_mask]
                velocity_mask = trigger_mask | (latched_mask & sustain_active)
                assignments[member]["velocity"][velocity_mask] = velocity[velocity_mask]
                gate_mask = latched_mask & gate_active
                assignments[member]["gate"][gate_mask] = gate[gate_mask]
                drone_mask = latched_mask & drone_active
                assignments[member]["drone"][drone_mask] = drone[drone_mask]

            self._latched_voice = current_voice
            self._next_voice = (idx + trigger_count) % member_count

        self._assignments = assignments
        self._block_token = token



def _ensure_bcf(audio_in, frames: int, *, name: str):
    if audio_in is None:
        return None, 1
    array = assert_BCF(audio_in, name=name)
    if array.shape[2] != frames:
        raise ValueError(f"{name}: expected {frames} frames, got {array.shape[2]}")
    return array, array.shape[0]


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if data.shape[1] > channels:
        return data[:, :channels, :]
    pad = np.zeros((data.shape[0], channels - data.shape[1], data.shape[2]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=1)



# =========================
# Graph nodes
# =========================
#
# Filters consume audio streams (`audio_in`) and transform them.
# Modulators emit control-rate signals that downstream nodes treat as parameters.
class Node:
    """Base class providing pooled CFFI-backed node buffers for graph nodes (C-ready)."""

    __slots__ = (
        "name",
        "_block_pool",
        "_leases",
        "params",
        "oversample_ratio",
        "declared_delay_frames",
        "supports_v2",
    )

    def __init__(self, name: str, *, block_pool: BlockPool | None = None) -> None:
        self.name = name
        self._block_pool = block_pool or BlockPool()
        self._leases: list[BlockLease] = []
        self.params: dict[str, object] = {}
        self.oversample_ratio = 1
        self.declared_delay_frames = 0
        self.supports_v2 = True

    def allocate_node_buffer(
        self,
        batches: int,
        channels: int,
        frames: int,
        *,
        tag: str = "out",
        zero: bool = False,
    ) -> np.ndarray:
        """Return a reusable C-ready node buffer shaped ``(B, C, F)`` for writing output."""

        lease = self._block_pool.acquire((int(batches), int(channels), int(frames)), tag=tag)
        self._leases.append(lease)
        node_buffer = lease.view
        if zero:
            node_buffer.fill(0.0)
        return node_buffer

    def recycle_node_buffers(self) -> None:
        """Return any leased node buffers to the pool for reuse."""

        for lease in self._leases:
            lease.release()
        self._leases.clear()

    def process(self, frames, sr, input_buffer, mods, params):
        raise NotImplementedError


class ConfigNode(Node):
    def __init__(self, name, params=None):
        super().__init__(name)
        self.params = dict(params or {})


class ControllerNode(ConfigNode):
    """Map controller actions to modulation signals via expressions."""

    def __init__(self, name, params=None):
        super().__init__(name, params)
        cfg = dict(params or {})
        outputs_cfg = cfg.get("outputs")
        if not isinstance(outputs_cfg, Mapping) or not outputs_cfg:
            raise ValueError(f"{self.name}: outputs configuration is required")

        self._output_order: list[str] = []
        self._compiled: list[object] = []
        for out_name, spec in outputs_cfg.items():
            if isinstance(spec, str):
                expression = spec
            elif isinstance(spec, Mapping):
                expr_value = (
                    spec.get("equation")
                    or spec.get("expr")
                    or spec.get("expression")
                )
                if expr_value is None:
                    raise ValueError(
                        f"{self.name}.{out_name}: expression is required"
                    )
                expression = str(expr_value)
            else:
                raise TypeError(
                    f"{self.name}.{out_name}: unsupported specification type {type(spec)!r}"
                )
            compiled = compile(expression, f"{self.name}.{out_name}", "eval")
            self._output_order.append(str(out_name))
            self._compiled.append(compiled)

        self.channels = len(self._output_order)
        self._output_index = {name: idx for idx, name in enumerate(self._output_order)}
        self._context_base = {"np": np, "math": math}
        self._task_queue: queue.Queue[
            tuple[dict[str, np.ndarray], int, int]
        ] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_output: np.ndarray | None = None
        self._latest_meta: tuple[int, int] | None = None
        self._last_error: Exception | None = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"{self.name}-controller",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _sanitise(name: str) -> str:
        safe = [
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in name
        ]
        token = "_".join("".join(part) for part in "".join(safe).split())
        return f"param_{token}" if token and token[0].isdigit() else token or "param"

    def output_index(self, name: str) -> int:
        try:
            return self._output_index[name]
        except KeyError as exc:
            raise KeyError(f"{self.name}: unknown output '{name}'") from exc

    def process(self, frames, sr, audio_in, mods, params):
        batches, param_frames = self._infer_dimensions(params, frames)
        expected_meta = (batches, param_frames)

        error: Exception | None = None
        with self._result_lock:
            if self._last_error is not None:
                error = self._last_error
                self._last_error = None
            cached_output = (
                self._latest_output
                if self._latest_meta == expected_meta
                else None
            )
        if error is not None:
            raise error

        computed_sync = False
        if cached_output is None:
            output, batches, param_frames = self._evaluate(params, frames, sr)
            with self._result_lock:
                self._latest_output = output
                self._latest_meta = (batches, param_frames)
            computed_sync = True
        else:
            output = cached_output

        if not computed_sync:
            self._submit_async(params, frames, sr)
        return output

    def _infer_dimensions(
        self, params: Mapping[str, np.ndarray], frames: int
    ) -> tuple[int, int]:
        example = None
        for value in params.values():
            if isinstance(value, np.ndarray):
                example = value
                break
        if example is not None:
            batches, _, param_frames = example.shape
        else:
            batches, param_frames = 1, frames
        return batches, param_frames

    def _evaluate(
        self,
        params: Mapping[str, np.ndarray],
        frames: int,
        sr: int,
    ) -> tuple[np.ndarray, int, int]:
        batches, param_frames = self._infer_dimensions(params, frames)

        def _zero_value() -> np.ndarray:
            return np.zeros((batches, param_frames), dtype=RAW_DTYPE)

        def _zero_raw() -> np.ndarray:
            return np.zeros((batches, 1, param_frames), dtype=RAW_DTYPE)

        class _DefaultDict(dict):
            def __init__(self, *args, factory):
                super().__init__(*args)
                self._factory = factory

            def __getitem__(self, key):
                if key not in self:
                    return self._factory()
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in self:
                    return super().get(key)
                return self._factory() if default is None else default

        values: dict[str, np.ndarray] = {}
        raw_values: dict[str, np.ndarray] = {}
        for key, value in params.items():
            array = assert_BCF(value, name=f"{self.name}.{key}")
            raw_values[key] = array
            if array.shape[1] == 1:
                values[key] = array[:, 0, :]
            else:
                values[key] = array

        signals = _DefaultDict(values, factory=_zero_value)
        raw_signals = _DefaultDict(raw_values, factory=_zero_raw)

        context = dict(self._context_base)
        context.update(
            {
                "signals": signals,
                "raw_signals": raw_signals,
                "frames": param_frames,
                "sample_rate": sr,
            }
        )
        for key, value in values.items():
            context[self._sanitise(key)] = value

        output = np.zeros((batches, self.channels, param_frames), dtype=RAW_DTYPE)
        for idx, code in enumerate(self._compiled):
            result = eval(code, context, {})
            value = as_BCF(
                result,
                batches,
                1,
                param_frames,
                name=f"{self.name}.{self._output_order[idx]}",
            )[:, 0, :]
            output[:, idx, :] = value
        return output, batches, param_frames

    def _worker_loop(self) -> None:
        while True:
            try:
                params, frames, sr = self._task_queue.get()
            except Exception:
                continue
            try:
                output, batches, param_frames = self._evaluate(params, frames, sr)
            except Exception as exc:  # pragma: no cover - defensive
                with self._result_lock:
                    self._last_error = exc
            else:
                with self._result_lock:
                    self._latest_output = output
                    self._latest_meta = (batches, param_frames)
            finally:
                self._task_queue.task_done()

    def _submit_async(
        self,
        params: Mapping[str, np.ndarray],
        frames: int,
        sr: int,
    ) -> None:
        cloned: dict[str, np.ndarray] = {
            key: np.array(value, copy=True)
            for key, value in params.items()
        }
        task = (cloned, frames, sr)
        try:
            self._task_queue.put_nowait(task)
        except queue.Full:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - defensive
                pass
            else:
                self._task_queue.task_done()
            try:
                self._task_queue.put_nowait(task)
            except queue.Full:  # pragma: no cover - defensive
                pass


class SilenceNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))

    def process(self, frames, sr, input_buffer, mods, params):
        _, batches = _ensure_bcf(input_buffer, frames, name=f"{self.name}.in")
        return self.allocate_node_buffer(batches, self.channels, frames, zero=True)


class ConstantNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))
        self.value = float(self.params.get("value", 0.0))

    def process(self, frames, sr, input_buffer, mods, params):
        _, batches = _ensure_bcf(input_buffer, frames, name=f"{self.name}.in")
        node_buffer = self.allocate_node_buffer(batches, self.channels, frames, zero=False)
        node_buffer.fill(self.value)
        return node_buffer


class SineOscillatorNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))
        self.frequency = float(self.params.get("frequency", 440.0))
        self.amplitude = float(self.params.get("amplitude", 0.5))
        phase = float(self.params.get("phase", 0.0)) % 1.0
        self._phase = np.array([[phase]], dtype=RAW_DTYPE)

    def process(self, frames, sr, input_buffer, mods, params):
        _, batches = _ensure_bcf(input_buffer, frames, name=f"{self.name}.in")
        channels = self.channels
        freq = as_BCF(
            params.get("frequency", self.frequency),
            batches,
            channels,
            frames,
            name=f"{self.name}.frequency",
        )
        amp = as_BCF(
            params.get("amplitude", self.amplitude),
            batches,
            channels,
            frames,
            name=f"{self.name}.amplitude",
        )
        if self._phase.shape != (batches, channels):
            self._phase = np.full((batches, channels), self._phase[0, 0], dtype=RAW_DTYPE)
        dphi = freq / float(sr)
        phase = (self._phase[..., None] + np.cumsum(dphi, axis=2)) % 1.0
        self._phase = phase[..., -1]
        node_buffer = self.allocate_node_buffer(batches, channels, frames, tag="sine")
        np.sin(2.0 * np.pi * phase, out=node_buffer)
        np.multiply(node_buffer, amp, out=node_buffer)
        return node_buffer


class SafetyNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 2))
        self.dc_alpha = float(self.params.get("dc_alpha", 0.995))
        # persistent per-batch/state array (B, C). Start empty; _ensure will size it.
        self._state: np.ndarray = np.zeros((0, self.channels), dtype=RAW_DTYPE)

    def process(self, frames, sr, audio_in, mods, params):
        audio, batches = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        out = self.allocate_block(batches, self.channels, frames, tag="safety", zero=False)
        if audio is None:
            out.fill(0.0)
            return out
        data = _match_channels(audio, self.channels)
        data = np.require(data, dtype=RAW_DTYPE, requirements=("C",))
        # There is no acceptable Python fallback path for runtime nodes: the
        # compiled CFFI kernel must run or we refuse to proceed.
        if not isinstance(self._state, np.ndarray) or self._state.shape != (batches, self.channels):
            self._state = np.zeros((batches, self.channels), dtype=RAW_DTYPE)
        if not c_kernels.AVAILABLE:
            raise RuntimeError(
                "SafetyNode requires compiled C kernels; Python fallbacks are forbidden."
            )
        try:
            c_kernels.dc_block_c(data, float(self.dc_alpha), self._state, out=out)
        except RuntimeError as exc:
            raise RuntimeError(
                "SafetyNode must execute the CFFI dc_block kernel; Python fallback is unacceptable."
            ) from exc
        np.clip(out, -1.0, 1.0, out=out)
        return out

class DelayNode(Node):
    def __init__(self,name,delay_samples=64):
        super().__init__(name)
        self.delay = delay_samples
        self.buf = np.zeros((1, 1, delay_samples), RAW_DTYPE)  # (B,C,D)
        self.w   = np.zeros((1, 1), dtype=int)                 # (B,C)

    def _ensure(self, B, C):
        if self.buf.shape[:2] != (B, C):
            self.buf = np.zeros((B, C, self.delay), RAW_DTYPE)
            self.w   = np.zeros((B, C), dtype=int)

    def process(self, frames, sr, audio_in, mods, params):
        if audio_in is None:
            x = self.allocate_block(1, 1, frames, tag="delay_in", zero=True)
        else:
            x = assert_BCF(audio_in, name="delay.in")
        B, C, F = x.shape
        self._ensure(B, C)

        out = self.allocate_block(B, C, F, tag="delay_out")
        idxs = (self.w[..., None] + np.arange(F)[None, None, :]) % self.delay  # (B,C,F)
        out[:] = self.buf.take(idxs, axis=2, mode='wrap')
        self.buf[np.arange(B)[:,None,None], np.arange(C)[None,:,None], idxs] = x
        self.w = (self.w + F) % self.delay
        return out

class LFONode(Node):
    def __init__(self,name,wave="sine",rate_hz=4.0,depth=0.5,use_input=False,slew_ms=0.0, slew_backend: str | None = None):
        super().__init__(name)
        self.wave=wave; self.rate=rate_hz; self.depth=depth
        self.use_input=use_input; self.slew_ms=slew_ms
        self.phase=0.0
        # slew processing must execute in C; there is no sanctioned Python fallback.
        backend = "auto" if slew_backend is None else str(slew_backend)
        if backend not in {"auto", "c"}:
            raise ValueError("LFONode only supports the CFFI slew backend")
        if not c_kernels.AVAILABLE:
            raise RuntimeError(
                "LFONode requires compiled C kernels; Python fallbacks are forbidden."
            )
        self.slew_backend = "c"
        # state for C-backed implementations
        self._slew_z0: np.ndarray | None = None
    def _make(self,ph):
        if self.wave=="sine": return np.sin(2*np.pi*ph,dtype=RAW_DTYPE)
        if self.wave=="square": return np.where((ph%1.0)<0.5,1.0,-1.0).astype(RAW_DTYPE)
        if self.wave=="saw": return (2.0*((ph%1.0)) - 1.0).astype(RAW_DTYPE)
        if self.wave=="triangle": return (2.0*np.abs(2.0*(ph%1.0)-1.0)-1.0).astype(RAW_DTYPE)
        return np.zeros_like(ph)
    def process(self,frames,sr,audio_in,mods,params):
        B = audio_in.shape[0] if self.use_input and audio_in is not None else 1
        C = 1; F = frames
        if self.use_input and audio_in is not None:
            x = assert_BCF(audio_in, name="lfo.in")
            m = np.maximum(1e-12, np.max(np.abs(x), axis=(1,2)))  # (B,)
            out = (x[:, :1, :] / m[:, None, None]) * float(self.depth)
        else:
            t = (self.phase + np.arange(F)*(self.rate/sr)) % 1.0
            self.phase = float(t[-1])
            wave = self._make(t)  # (F,)
            out = np.tile(wave, (B,1,1))  # (B,1,F)
            out *= float(self.depth)
        if self.slew_ms > 0:
            alpha = 1.0 - math.exp(-1.0/(sr*(self.slew_ms/1000.0)))
            # Exponential smoothing across time: z[n] = r*z[n-1] + alpha*x[n]
            # where r = 1-alpha. We can compute closed-form or use an iterative kernel.
            r = 1.0 - alpha
            if alpha >= 1.0 - 1e-15:
                # alpha==1 -> z = x (no smoothing)
                pass
            else:
                plane = np.require(out[:, 0, :], dtype=RAW_DTYPE, requirements=("C",))
                # ensure z0 exists for iterative backends
                if self._slew_z0 is None or self._slew_z0.shape[0] != plane.shape[0]:
                    self._slew_z0 = np.zeros(plane.shape[0], dtype=RAW_DTYPE)
                try:
                    c_kernels.lfo_slew_c(plane, r, alpha, self._slew_z0, out=plane)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "LFONode must execute the CFFI slew kernel; Python fallback is unacceptable."
                    ) from exc
        return out  # (B,1,F)

class EnvelopeModulatorNode(Node):
    """Multi-stage envelope generator that emits control signals."""

    _GROUPS: dict[str, _EnvelopeGroupState] = {}

    @classmethod
    def reset_groups(cls) -> None:
        cls._GROUPS.clear()

    _IDLE = 0
    _ATTACK = 1
    _HOLD = 2
    _DECAY = 3
    _SUSTAIN = 4
    _RELEASE = 5
    def __init__(
        self,
        name,
        *,
        attack_ms=8.0,
        hold_ms=10.0,
        decay_ms=80.0,
        sustain_level=0.7,
        sustain_ms=0.0,
        release_ms=160.0,
        send_resets=True,
        group: str | None = None,
    ):
        super().__init__(name)
        self.attack_ms = float(attack_ms)
        self.hold_ms = float(hold_ms)
        self.decay_ms = float(decay_ms)
        self.sustain_level = float(sustain_level)
        self.sustain_ms = float(sustain_ms)
        self.release_ms = float(release_ms)
        self.send_resets = bool(send_resets)
        self.group = group
        self._stage = None
        self._value = None
        self._timer = None
        self._velocity = None
        self._activation_count = None
        self._release_start = None
        self._gate_state = None
        self._drone_state = None
        self._kernel_planes: np.ndarray | None = None
        if self.group:
            state = self._GROUPS.setdefault(self.group, _EnvelopeGroupState())
            state.register(self.name)

    def _ensure(self, B: int) -> None:
        if (
            self._stage is None
            or self._stage.shape[0] != B
        ):
            self._stage = np.full(B, self._IDLE, dtype=np.int32)
            self._value = np.zeros(B, RAW_DTYPE)
            self._timer = np.zeros(B, RAW_DTYPE)
            self._velocity = np.zeros(B, RAW_DTYPE)
            self._activation_count = np.zeros(B, dtype=np.int64)
            self._release_start = np.zeros(B, RAW_DTYPE)
            self._gate_state = np.zeros(B, dtype=bool)
            self._drone_state = np.zeros(B, dtype=bool)

        if self._gate_state is None or self._gate_state.shape[0] != B:
            self._gate_state = np.zeros(B, dtype=bool)
        if self._drone_state is None or self._drone_state.shape[0] != B:
            self._drone_state = np.zeros(B, dtype=bool)

    def _stage_frames(self, ms: float, sr: int) -> int:
        if ms <= 0.0:
            return 0
        return max(1, int(round(ms * sr / 1000.0)))

    def process(self, frames, sr, audio_in, mods, params):
        B = audio_in.shape[0] if audio_in is not None else 1
        C = 1
        F = frames
        self._ensure(B)

        trigger = as_BCF(params.get("trigger", 0.0), B, C, F, name=f"{self.name}.trigger")[:, 0, :]
        gate = as_BCF(params.get("gate", 0.0), B, C, F, name=f"{self.name}.gate")[:, 0, :]
        drone = as_BCF(params.get("drone", 0.0), B, C, F, name=f"{self.name}.drone")[:, 0, :]
        velocity = as_BCF(params.get("velocity", 1.0), B, C, F, name=f"{self.name}.velocity")[:, 0, :]
        send_reset = as_BCF(
            params.get("send_reset", float(self.send_resets)),
            B,
            C,
            F,
            name=f"{self.name}.send_reset",
        )[:, 0, :]

        if self.group:
            state = self._GROUPS.setdefault(self.group, _EnvelopeGroupState())
            state.register(self.name)
            trigger, gate, drone, velocity = state.assign(self.name, trigger, gate, drone, velocity)

        if (
            self._stage.size
            and np.all(self._stage == self._IDLE)
            and not np.any(trigger > 0.5)
            and not np.any(gate > 0.5)
            and not np.any(drone > 0.5)
        ):
            return np.zeros((B, 2, F), dtype=RAW_DTYPE)

        atk_frames = self._stage_frames(self.attack_ms, sr)
        hold_frames = self._stage_frames(self.hold_ms, sr)
        dec_frames = self._stage_frames(self.decay_ms, sr)
        sus_frames = self._stage_frames(self.sustain_ms, sr)
        rel_frames = self._stage_frames(self.release_ms, sr)

        trigger_active = trigger > 0.5
        gate_active = gate > 0.5
        drone_active = drone > 0.5

        if (
            self._kernel_planes is None
            or self._kernel_planes.shape != (2, B, F)
        ):
            self._kernel_planes = np.empty((2, B, F), dtype=RAW_DTYPE)

        amp_plane = self._kernel_planes[0]
        reset_plane = self._kernel_planes[1]

        send_reset_flag = bool(send_reset.mean() > 0.5)

        try:
            amp, reset = c_kernels.envelope_process_c(
                trigger,
                gate,
                drone,
                velocity,
                atk_frames,
                hold_frames,
                dec_frames,
                sus_frames,
                rel_frames,
                self.sustain_level,
                send_reset_flag,
                self._stage,
                self._value,
                self._timer,
                self._velocity,
                self._activation_count,
                self._release_start,
                out_amp=amp_plane,
                out_reset=reset_plane,
            )
        except RuntimeError:
            amp, reset = c_kernels.envelope_process_py(
                trigger,
                gate,
                drone,
                velocity,
                atk_frames,
                hold_frames,
                dec_frames,
                sus_frames,
                rel_frames,
                self.sustain_level,
                send_reset_flag,
                self._stage,
                self._value,
                self._timer,
                self._velocity,
                self._activation_count,
                self._release_start,
                out_amp=amp_plane,
                out_reset=reset_plane,
            )

        np.copyto(self._gate_state, gate_active[:, -1], casting="no")
        np.copyto(self._drone_state, drone_active[:, -1], casting="no")

        out = np.empty((B, 2, F), dtype=RAW_DTYPE)
        out[:, 0, :] = amp
        out[:, 1, :] = reset
        return out


class PitchQuantizerNode(Node):
    """Convert controller input into quantised pitch values."""

    def __init__(self, name: str, state: Mapping[str, object], *, slew: bool = True) -> None:
        super().__init__(name)
        self.state = state
        self.slew = bool(slew)
        self.effective_token = "12tet/full"
        self.free_variant = "continuous"
        self.span_oct = 2.0
        self._last_freq: np.ndarray | None = None
        self._last_output: np.ndarray | None = None
        self._last_target: np.ndarray | None = None

    def update_mode(
        self,
        *,
        effective_token: str,
        free_variant: str,
        span_oct: float,
    ) -> None:
        self.effective_token = effective_token
        self.free_variant = free_variant
        self.span_oct = float(span_oct)

    @staticmethod
    def _midi_to_freq(midi: np.ndarray) -> np.ndarray:
        return 440.0 * np.power(2.0, (midi - 69.0) / 12.0, dtype=RAW_DTYPE)

    @property
    def last_output(self) -> np.ndarray | None:
        return None if self._last_output is None else self._last_output.copy()

    @property
    def last_target(self) -> np.ndarray | None:
        return None if self._last_target is None else self._last_target.copy()

    def process(self, frames, sr, audio_in, mods, params):
        B = audio_in.shape[0] if audio_in is not None else 1
        C = 1
        F = frames

        ctrl = as_BCF(params.get("input", 0.0), B, C, F, name=f"{self.name}.input")[:, 0, :]
        root_midi = as_BCF(params.get("root_midi", 60.0), B, C, F, name=f"{self.name}.root")[:, 0, :]
        span_arr = as_BCF(params.get("span_oct", self.span_oct), B, C, F, name=f"{self.name}.span")[:, 0, :]

        token = self.effective_token or str(self.state.get("base_token", "12tet/full"))
        grid = quantizer.get_reference_grid_cents(self.state, token)
        root_freq = self._midi_to_freq(root_midi)

        ctrl_scaled = ctrl * span_arr
        if quantizer.is_free_mode_token(token):
            grid_arr, _ = utils._grid_sorted(grid)
            N = max(1, int(grid_arr.size))
            variant = self.free_variant
            if variant == "continuous":
                cents = ctrl_scaled * 1200.0
            elif variant == "weighted":
                cents = quantizer.grid_warp_inverse(ctrl_scaled * N, grid)
            else:
                cents = quantizer.grid_warp_inverse(np.rint(ctrl_scaled * N), grid)
        else:
            cents_unq = ctrl_scaled * 1200.0
            u = quantizer.grid_warp_forward(cents_unq, grid)
            cents = quantizer.grid_warp_inverse(np.rint(u), grid)

        freq_target = root_freq * np.power(2.0, cents / 1200.0)

        if not self.slew:
            block = freq_target
        else:
            if self._last_freq is None or self._last_freq.shape[0] != B:
                self._last_freq = freq_target[:, 0].copy()
            # Vectorized cubic ramp across batches
            y0 = self._last_freq
            y1 = freq_target[:, -1]
            t = np.linspace(0.0, 1.0, F, endpoint=False, dtype=RAW_DTYPE)
            ramp = (3.0 * t * t - 2.0 * t * t * t)[None, :]
            block = y0[:, None] + (y1 - y0)[:, None] * ramp
            self._last_freq = block[:, -1].copy()

        self._last_output = block.copy()
        self._last_target = freq_target[:, -1].copy()

        return block[:, None, :]


class AmplifierModulatorNode(Node):
    """Combine velocity and control modulators into an amplitude control signal."""

    def __init__(self, name):
        super().__init__(name)

    def process(self, frames, sr, audio_in, mods, params):
        batches = audio_in.shape[0] if audio_in is not None else 1
        base_param = params.get("base")
        control_param = params.get("control")
        if base_param is None and control_param is not None:
            control_bcf = assert_BCF(control_param, name=f"{self.name}.control")
            base = np.zeros((control_bcf.shape[0], 1, frames), dtype=RAW_DTYPE)
            control = np.clip(control_bcf, 0.0, 1.0, out=control_bcf.copy())
        else:
            base = (
                np.zeros((batches, 1, frames), dtype=RAW_DTYPE)
                if base_param is None
                else assert_BCF(base_param, name=f"{self.name}.base")
            )
            if control_param is None:
                control = np.ones((base.shape[0], 1, frames), dtype=RAW_DTYPE)
            else:
                control = assert_BCF(control_param, name=f"{self.name}.control")
                if control.shape[1] != 1:
                    control = control[:, :1, :]
                control = np.clip(control, 0.0, 1.0, out=control.copy())
        out = base * control
        mod = params.get("mod")
        if mod is not None:
            mod = assert_BCF(mod, name=f"{self.name}.mod")
            if mod.shape[1] != 1:
                mod = mod[:, :1, :]
            out = out * (1.0 + mod)
        return out


class OscillatorPitchNode(Node):
    """Program pitch curves for driver handoff with optional slew limiting."""

    def __init__(
        self,
        name: str,
        *,
        min_freq: float = 0.0,
        default_slew: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.min_freq = float(min_freq)
        self.default_slew = float(default_slew)
        self.params.update({
            "min_freq": self.min_freq,
            "default_slew": self.default_slew,
        })
        self._last: np.ndarray | None = None

    def _ensure_curve(self, key: str, value, batches: int, frames: int) -> np.ndarray:
        return as_BCF(value, batches, 1, frames, name=key)[:, 0, :]

    def process(self, frames, sr, audio_in, mods, params):
        batches = audio_in.shape[0] if audio_in is not None else 1
        if batches <= 0:
            batches = 1
        freq_curve = None
        if "pitch_hz" in params:
            freq_curve = self._ensure_curve(
                f"{self.name}.pitch_hz", params.get("pitch_hz", 0.0), batches, frames
            ).astype(RAW_DTYPE, copy=True)
        else:
            root = self._ensure_curve(
                f"{self.name}.root_hz", params.get("root_hz", self.min_freq), batches, frames
            )
            offsets = self._ensure_curve(
                f"{self.name}.offset_cents", params.get("offset_cents", 0.0), batches, frames
            )
            freq_curve = (root * np.power(2.0, offsets / 1200.0)).astype(RAW_DTYPE, copy=True)
        add = params.get("add_hz")
        if add is not None:
            freq_curve = freq_curve + self._ensure_curve(
                f"{self.name}.add_hz", add, batches, frames
            )
        min_freq = float(params.get("min_freq", self.min_freq))
        if min_freq > 0.0:
            np.maximum(freq_curve, min_freq, out=freq_curve)
        slew_param = params.get("slew_hz_per_s")
        if slew_param is not None:
            slew_curve = np.maximum(
                self._ensure_curve(f"{self.name}.slew", slew_param, batches, frames),
                0.0,
            )
        elif self.default_slew > 0.0:
            slew_curve = np.full((batches, frames), self.default_slew, dtype=RAW_DTYPE)
        else:
            slew_curve = None

        if slew_curve is None or not np.any(slew_curve > 0.0):
            if self._last is None or self._last.shape[0] != batches:
                self._last = freq_curve[:, -1].copy()
            return freq_curve[:, None, :]

        if self._last is None or self._last.shape[0] != batches:
            self._last = freq_curve[:, 0].copy()

        output = np.empty_like(freq_curve, dtype=RAW_DTYPE)
        per_sample = slew_curve / float(sr)
        np.maximum(per_sample, 0.0, out=per_sample)

        for b in range(batches):
            current = float(self._last[b])
            base = freq_curve[b]
            limit = per_sample[b]
            for f in range(frames):
                target = float(base[f])
                lim = float(limit[f])
                if lim > 0.0:
                    delta = target - current
                    if delta > lim:
                        delta = lim
                    elif delta < -lim:
                        delta = -lim
                    current += delta
                else:
                    current = target
                if current < min_freq:
                    current = min_freq
                output[b, f] = current
            self._last[b] = current

        return output[:, None, :]


class OscNode(Node):
    def __init__(
        self,
        name: str,
        wave: str = "sine",
        *,
        mode: str = "polyblep",
        accept_reset: bool = True,
        integration_leak: float = 0.9995,
        integration_gain: float = 1.0,
        integration_clamp: float = 1.2,
        slew_rate: float = 12000.0,
        slew_clamp: float = 1.2,
    ) -> None:
        super().__init__(name)
        self.wave = str(wave)
        self.mode = str(mode)
        self.phase = None
        self.accept_reset = bool(accept_reset)
        self.integration_leak = float(integration_leak)
        self.integration_gain = float(integration_gain)
        self.integration_clamp = float(integration_clamp)
        self.slew_rate = float(slew_rate)
        self.slew_clamp = float(slew_clamp)
        self.params.update(
            {
                "wave": self.wave,
                "mode": self.mode,
                "accept_reset": self.accept_reset,
                "integration_leak": self.integration_leak,
                "integration_gain": self.integration_gain,
                "integration_clamp": self.integration_clamp,
                "slew_rate": self.slew_rate,
                "slew_clamp": self.slew_clamp,
            }
        )
        self._freq_state: np.ndarray | None = None
        self._voice_phase: dict[str, np.ndarray] = {}
        self._voice_phase_out: dict[str, np.ndarray] = {}
        self._phase_buffer: np.ndarray | None = None
        self._arp_step: np.ndarray | None = None
        self._arp_timer: np.ndarray | None = None
        self._last_arp_offsets: np.ndarray | None = None
        self._arp_offsets_buf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_array(self, key: str, data, batches: int, frames: int) -> np.ndarray:
        return as_BCF(data, batches, 1, frames, name=key)[:, 0, :]

    def _parse_voice_spec(self, spec, name: str) -> list[tuple[float, float]]:
        if spec is None:
            return []
        offsets: Sequence[float] | None = None
        mixes: Sequence[float] | None = None
        if isinstance(spec, Mapping):
            offsets = spec.get("offsets") or spec.get("voices") or ()
            mixes = spec.get("mix") or spec.get("mixes") or ()
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            if len(spec) == 0:
                return []
            offsets = spec[0] if len(spec) > 0 else ()
            mixes = spec[1] if len(spec) > 1 else ()
        else:
            raise TypeError(f"{name}: unsupported specification type {type(spec)!r}")

        offsets_array = [] if offsets is None else list(np.asarray(offsets, dtype=RAW_DTYPE).ravel())
        mixes_array = [] if mixes is None else list(np.asarray(mixes, dtype=RAW_DTYPE).ravel())
        if not offsets_array:
            return []
        if not mixes_array:
            mixes_array = [1.0] * len(offsets_array)
        if len(mixes_array) < len(offsets_array):
            mixes_array.extend([mixes_array[-1]] * (len(offsets_array) - len(mixes_array)))

        voices: list[tuple[float, float]] = []
        for idx, offset in enumerate(offsets_array):
            try:
                off = float(offset)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: invalid offset {offset!r}") from exc
            try:
                mix = float(mixes_array[idx])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: invalid mix {mixes_array[idx]!r}") from exc
            if abs(mix) <= 1e-12:
                continue
            ratio = float(np.power(2.0, off / 1200.0))
            voices.append((ratio, mix))
        return voices

    def _advance_phase(self, key: str, dphi: np.ndarray) -> np.ndarray:
        batches, frames = dphi.shape
        state = self._voice_phase.get(key)
        if state is None or state.shape[0] != batches:
            state = np.zeros(batches, dtype=RAW_DTYPE)
        buf = self._voice_phase_out.get(key)
        if buf is None or buf.shape != (batches, frames):
            buf = np.zeros((batches, frames), dtype=RAW_DTYPE)
        # Phase advance must run via CFFI; abandoning to Python is disallowed.
        if not c_kernels.AVAILABLE:
            raise RuntimeError(
                "OscNode requires compiled C kernels; Python fallbacks are forbidden."
            )
        try:
            ph = c_kernels.phase_advance_c(dphi, None, state, out=buf)
        except RuntimeError as exc:
            raise RuntimeError(
                "OscNode must execute the CFFI phase_advance kernel; Python fallback is unacceptable."
            ) from exc
        # state is updated in-place by the kernel
        self._voice_phase[key] = state
        self._voice_phase_out[key] = ph
        return ph

    def _render_wave(self, phase: np.ndarray, dphi: np.ndarray) -> np.ndarray:
        if self.wave == "sine":
            return np.sin(2 * np.pi * phase, dtype=RAW_DTYPE)
        if self.wave == "saw":
            return osc_saw_blep(phase, dphi)
        if self.wave == "square":
            return osc_square_blep(phase, dphi)
        if self.wave == "triangle":
            return osc_triangle_blep(phase, dphi)
        return np.zeros_like(phase)

    def _parse_arp_plan(self, plan, frames: int, sr: int) -> tuple[list[float], int] | None:
        if plan is None:
            return None
        sequence: Sequence[float] | None = None
        rate_hz = None
        frames_per_step = None
        if isinstance(plan, Mapping):
            sequence = plan.get("sequence") or plan.get("pattern") or plan.get("steps")
            rate_hz = plan.get("rate_hz") or plan.get("rate")
            frames_per_step = plan.get("frames_per_step") or plan.get("step_frames")
        elif isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
            if len(plan) == 0:
                sequence = []
            elif len(plan) == 1:
                sequence = plan[0]
            else:
                sequence = plan[0]
                frames_per_step = plan[1]
        else:
            raise TypeError("arp: unsupported plan specification")

        seq_vals = list(np.asarray(sequence if sequence is not None else [0.0], dtype=RAW_DTYPE).ravel())
        if not seq_vals:
            seq_vals = [0.0]
        if frames_per_step is not None:
            try:
                fps = int(frames_per_step)
            except (TypeError, ValueError) as exc:
                raise ValueError("arp: frames_per_step must be numeric") from exc
            fps = max(1, fps)
        else:
            try:
                rate = float(rate_hz) if rate_hz is not None else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError("arp: rate_hz must be numeric") from exc
            if rate <= 0.0:
                fps = frames
            else:
                fps = max(1, int(round(sr / rate)))
        return seq_vals, fps

    # ------------------------------------------------------------------
    def process(self, frames, sr, audio_in, mods, params):
        B = audio_in.shape[0] if audio_in is not None else 1
        C = 1
        F = frames
        mode = getattr(self, "mode", "polyblep")
        if mode not in ("polyblep", "integrator", "op_amp"):
            raise RuntimeError(f"{self.name}: unsupported oscillator mode '{mode}' in Python backend")
        if mode in ("integrator", "op_amp"):
            raise RuntimeError(
                f"{self.name}: oscillator mode '{mode}' requires the native runtime; Python fallback is not permitted."
            )

        f = self._ensure_array("osc.freq", params.get("freq", 0.0), B, F)
        a = self._ensure_array("osc.amp", params.get("amp", 1.0), B, F)

        pan = params.get("pan")
        pan_arr = None
        if pan is not None:
            pan_arr = np.clip(self._ensure_array("osc.pan", pan, B, F), -1.0, 1.0)

        port = params.get("port")
        port_mask = None
        if port is not None:
            port_vals = self._ensure_array("osc.port", port, B, F)
            port_mask = np.where(port_vals > 0.5, 1.0, 0.0).astype(RAW_DTYPE, copy=False)

        slide_time = np.zeros((B, F), dtype=RAW_DTYPE)
        slide_damp = np.zeros((B, F), dtype=RAW_DTYPE)
        slide_spec = params.get("slide")
        if slide_spec is not None:
            if isinstance(slide_spec, Sequence) and not isinstance(slide_spec, (str, bytes)) and len(slide_spec) >= 1:
                slide_time = self._ensure_array("osc.slide_time", slide_spec[0], B, F)
                if len(slide_spec) > 1:
                    slide_damp = self._ensure_array("osc.slide_damp", slide_spec[1], B, F)
            else:
                slide_arr = as_BCF(slide_spec, B, 2, F, name="osc.slide")
                slide_time = slide_arr[:, 0, :]
                if slide_arr.shape[1] > 1:
                    slide_damp = slide_arr[:, 1, :]

        chord_voices = self._parse_voice_spec(params.get("chord"), "chord")
        subharmonic_voices = self._parse_voice_spec(params.get("subharmonic"), "subharmonic")

        arp_plan = self._parse_arp_plan(params.get("arp"), F, sr) if "arp" in params else None
        freq_target = f.copy()
        if arp_plan is not None:
            seq_vals, fps = arp_plan
            if self._arp_step is None or self._arp_step.shape[0] != B:
                self._arp_step = np.zeros(B, dtype=np.int32)
                self._arp_timer = np.zeros(B, dtype=np.int32)
            assert self._arp_timer is not None
            seq_arr = np.asarray(seq_vals, dtype=RAW_DTYPE)
            if seq_arr.size == 0:
                seq_arr = np.array([0.0], dtype=RAW_DTYPE)
            if self._arp_offsets_buf is None or self._arp_offsets_buf.shape != (B, F):
                self._arp_offsets_buf = np.zeros((B, F), dtype=RAW_DTYPE)
            try:
                offsets = c_kernels.arp_advance_c(
                    seq_arr,
                    int(seq_arr.size),
                    B,
                    F,
                    self._arp_step,
                    self._arp_timer,
                    int(fps),
                    out=self._arp_offsets_buf,
                )
            except RuntimeError:
                offsets = c_kernels.arp_advance_py(
                    seq_arr,
                    int(seq_arr.size),
                    B,
                    F,
                    self._arp_step,
                    self._arp_timer,
                    int(fps),
                    out=self._arp_offsets_buf,
                )
            self._last_arp_offsets = offsets
            freq_target = freq_target * np.power(2.0, offsets / 1200.0)
        else:
            self._last_arp_offsets = None

        if port_mask is not None and np.any(port_mask > 0.5):
            if self._freq_state is None or self._freq_state.shape[0] != B:
                self._freq_state = freq_target[:, 0].copy()
            freq_target = np.require(freq_target, dtype=RAW_DTYPE, requirements=("C",))
            port_mask = np.require(port_mask, dtype=RAW_DTYPE, requirements=("C",))
            slide_time = np.require(slide_time, dtype=RAW_DTYPE, requirements=("C",))
            slide_damp = np.require(slide_damp, dtype=RAW_DTYPE, requirements=("C",))
            try:
                smoothed = c_kernels.portamento_smooth_c(
                    freq_target,
                    port_mask,
                    slide_time,
                    slide_damp,
                    sr,
                    self._freq_state,
                    out=freq_target,
                )
            except RuntimeError:
                smoothed = c_kernels.portamento_smooth_py(
                    freq_target,
                    port_mask,
                    slide_time,
                    slide_damp,
                    sr,
                    self._freq_state,
                    out=freq_target,
                )
            self._freq_state = self._freq_state
            freq_target = smoothed
        else:
            self._freq_state = freq_target[:, -1].copy()

        dphi = freq_target / float(sr)
        if self.phase is None or self.phase.shape[0] != B:
            self.phase = np.zeros(B, RAW_DTYPE)
        if self._phase_buffer is None or self._phase_buffer.shape != (B, F):
            self._phase_buffer = np.zeros((B, F), dtype=RAW_DTYPE)

        reset = None
        if self.accept_reset and "reset" in params:
            reset = np.require(
                self._ensure_array("osc.reset", params.get("reset", 0.0), B, F),
                dtype=RAW_DTYPE,
                requirements=("C",),
            )

        try:
            ph_base = c_kernels.phase_advance_c(dphi, reset, self.phase, out=self._phase_buffer)
        except RuntimeError:
            ph_base = c_kernels.phase_advance_py(dphi, reset, self.phase, out=self._phase_buffer)

        wave = self._render_wave(ph_base, dphi)

        for idx, (ratio, mix) in enumerate(chord_voices):
            dphi_voice = dphi * ratio
            ph_voice = self._advance_phase(f"chord{idx}", dphi_voice)
            wave += self._render_wave(ph_voice, dphi_voice) * mix

        for idx, (ratio, mix) in enumerate(subharmonic_voices):
            dphi_voice = dphi * ratio
            ph_voice = self._advance_phase(f"sub{idx}", dphi_voice)
            wave += self._render_wave(ph_voice, dphi_voice) * mix

        signal = wave * a

        if pan_arr is None:
            return signal[:, None, :]

        angle = (pan_arr + 1.0) * (math.pi / 4.0)
        left = signal * np.cos(angle)
        right = signal * np.sin(angle)
        out = np.empty((B, 2, F), dtype=RAW_DTYPE)
        out[:, 0, :] = left
        out[:, 1, :] = right
        return out

class ParametricDriverNode(Node):
    def __init__(self, name: str, *, mode: str = "quartz", harmonics: Sequence[float] | None = None) -> None:
        super().__init__(name)
        self.mode = str(mode)
        if harmonics is None:
            if self.mode == "piezo":
                harmonics = (1.0, 0.35, 0.12)
            else:
                harmonics = ()
        self.harmonics = tuple(float(h) for h in harmonics)
        params: dict[str, object] = {"mode": self.mode}
        if self.harmonics:
            params["harmonics"] = ",".join(f"{h:.9g}" for h in self.harmonics)
        self.params.update(params)
        self._phase: np.ndarray | None = None

    def process(self, frames, sr, audio_in, mods, params):
        B = audio_in.shape[0] if audio_in is not None else 1
        freq = as_BCF(params.get("frequency", 440.0), B, 1, frames, name=f"{self.name}.frequency")[:, 0, :]
        amp = as_BCF(params.get("amplitude", 1.0), B, 1, frames, name=f"{self.name}.amplitude")[:, 0, :]
        phase_offset = as_BCF(params.get("phase_offset", 0.0), B, 1, frames, name=f"{self.name}.phase_offset")[:, 0, :]
        blend = as_BCF(params.get("render_mode", 0.0), B, 1, frames, name=f"{self.name}.render_mode")[:, 0, :]
        if self._phase is None or self._phase.shape[0] != B:
            self._phase = np.zeros(B, dtype=RAW_DTYPE)
        out = np.zeros((B, 1, frames), dtype=RAW_DTYPE)
        harmonics = np.asarray(self.harmonics if self.harmonics else (1.0,), dtype=RAW_DTYPE)
        if audio_in is not None:
            driver_in = assert_BCF(audio_in, name=f"{self.name}.in")  # (B,C,F)
            streaming = np.mean(driver_in, axis=1)
        else:
            streaming = np.zeros((B, frames), dtype=RAW_DTYPE)
        for b in range(B):
            phase = float(self._phase[b])
            for f in range(frames):
                phase = (phase + float(freq[b, f]) / float(sr)) % 1.0
                ph = (phase + float(phase_offset[b, f])) % 1.0
                sample = 0.0
                for idx, coeff in enumerate(harmonics):
                    if coeff == 0.0:
                        continue
                    sample += float(coeff) * np.sin(2.0 * np.pi * ph * (idx + 1))
                mix = float(blend[b, f])
                if mix < 0.0:
                    mix = 0.0
                elif mix > 1.0:
                    mix = 1.0
                if streaming.size:
                    stream_val = float(streaming[b, f % streaming.shape[1]])
                else:
                    stream_val = 0.0
                combined = (1.0 - mix) * sample + mix * stream_val
                out[b, 0, f] = combined * float(amp[b, f])
            self._phase[b] = phase
        return out


class FFTDivisionNode(Node):
    """Frequency-division FFT node (native-only)."""

    def __init__(self, name: str, params: Mapping[str, object] | None = None) -> None:
        super().__init__(name)
        self.params = dict(params or {})

    def process(self, frames, sr, audio_in, mods, params):
        raise RuntimeError(
            "FFTDivisionNode must execute via the native runtime; Python fallback is not available."
        )


class SamplerNode(Node):
    def __init__(self, name, sampler):
        super().__init__(name)
        self.sampler = sampler
    def process(self,frames,sr,audio_in,mods,params):
        _, B = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        out = np.empty((B, 1, frames), RAW_DTYPE)
        rate = params.get("rate", 1.0)
        tr = params.get("transpose", 0.0)
        gain = params.get("gain", 1.0)
        # Vectorized: if params are arrays, use them per channel, else broadcast
        for b in range(B):
            self.sampler.render_into(
                out[b, 0],
                sr,
                rate[b] if isinstance(rate, np.ndarray) and rate.shape[0] == B else rate,
                tr[b] if isinstance(tr, np.ndarray) and tr.shape[0] == B else tr,
                gain[b] if isinstance(gain, np.ndarray) and gain.shape[0] == B else gain
            )
        return out

class MixNode(Node):
    """
    Mixes N input bundles (num_inputs, channels, frames) down to (channels, frames).
    Applies per-channel gain, expands mono to stereo if needed, and supports ALC and compression.
    """

    def __init__(self, name, params=None, *, out_channels=2, alc=True, compression="tanh"):
        super().__init__(name)
        cfg = dict(params or {}) if params is not None else {}
        if params is not None and not isinstance(params, dict):
            raise TypeError("MixNode params must be a mapping")
        self.out_channels = int(cfg.get("channels", out_channels))
        self.alc = bool(cfg.get("alc", alc))
        self.compression = cfg.get("compression", compression)
        self.stats = ClipStats()
        # For ALC
        self.rms_hist = [np.zeros(self.out_channels, dtype=RAW_DTYPE) for _ in range(256)]
        self.peak_hist = [np.zeros(self.out_channels, dtype=RAW_DTYPE) for _ in range(256)]
        self.alpha = 0.5
        self.attack = 16
        self.sustain = 128
        self.decay = 256

    def process(self, frames, sr, audio_in, mods, params):
        if audio_in is None:
            return np.zeros((1, self.out_channels, frames), dtype=RAW_DTYPE)
        x = assert_BCF(audio_in, name="mix.in")  # (B,C,F)
        B, C, F = x.shape
        y = np.sum(x, axis=1, keepdims=True)     # (B,1,F)
        if self.out_channels > 1:
            y = np.repeat(y, self.out_channels, axis=1)  # (B,outC,F)

        # (optional) ALC/compression can operate per-channel over y[:,c,:] here

        self.stats.update(y.reshape(-1, F))
        return y

class BiquadNode(Node):
    def __init__(self,name,fs,ftype="lowpass"):
        super().__init__(name); self.ftype=ftype; self.filt=FilterLPBiquad(fs)
        self.peaking_db=6.0
    def process(self,frames,sr,audio_in,mods,params):
        if audio_in is None: return np.zeros((1,frames),RAW_DTYPE)
        cutoff=params.get("cutoff", 1000.0)  # Default cutoff to 1000.0 Hz if not provided
        Q=params.get("Q", 0.707)  # Default Q to 0.707 if not provided
        # cutoff, Q: (B,F)
        B = audio_in.shape[0]
        self.filt.ensure_B(B)
        # Use last frame for filter update (per batch)
        self.filt.update(float(cutoff[-1,-1]), float(Q[-1,-1]), self.ftype, self.peaking_db)
        return self.filt.process(audio_in)

class GainNode(Node):
    def __init__(self,name,gain=1.0): super().__init__(name); self.gain=gain
    def process(self,frames,sr,audio_in,mods,params):
        if audio_in is None: return np.zeros((1,frames),RAW_DTYPE)
        g=params.get("gain",None)
        if g is None: return audio_in*self.gain
        return audio_in*g

class SourceSwitch(Node):
    def __init__(self,name,osc_node:OscNode,samp_node:SamplerNode|None, state):
        super().__init__(name); self.osc=osc_node; self.samp=samp_node; self.state=state
    def process(self,frames,sr,audio_in,mods,params):
        B = audio_in.shape[0] if audio_in is not None else 1
        if self.state["source_type"]=="sampler" and self.samp is not None:
            return self.samp.process(frames,sr,audio_in,mods,{"rate":1.0,"transpose":0.0,"gain":1.0})
        return self.osc.process(frames,sr,audio_in,mods,params)

class ClipStats:
    def __init__(self):
        self.last_max = 0.0
        self.last_min = 0.0
        self.last_clipped = False
    def update(self, arr):
        self.last_max = float(np.max(arr))
        self.last_min = float(np.min(arr))
        self.last_clipped = np.any(np.abs(arr) > 1.0)

class SafetyFilterNode(Node):
    def __init__(self, name, sr, n_ch=2):
        super().__init__(name)
        self.a = 0.995
        self.prev_in = np.zeros((1, n_ch), RAW_DTYPE)  # (B,C)
        self.prev_dc = np.zeros((1, n_ch), RAW_DTYPE)
        self._buffer = np.zeros((0, n_ch, 0), RAW_DTYPE)

    def _ensure(self, B, C):
        if self.prev_in.shape != (B, C):
            self.prev_in = np.zeros((B, C), RAW_DTYPE)
            self.prev_dc = np.zeros((B, C), RAW_DTYPE)
        if self._buffer.shape[:2] != (B, C):
            self._buffer = np.zeros((B, C, 0), RAW_DTYPE)

    def process(self, frames, sr, audio_in, mods, params):
        x = assert_BCF(audio_in, name="safety.in")  # (B,C,F)
        B, C, F = x.shape
        self._ensure(B, C)
        if self._buffer.shape != (B, C, F):
            self._buffer = np.zeros((B, C, F), RAW_DTYPE)
        x = np.require(x, dtype=RAW_DTYPE, requirements=("C",))
        try:
            y = c_kernels.safety_filter_c(x, float(self.a), self.prev_in, self.prev_dc, out=self._buffer)
        except RuntimeError:
            y = c_kernels.safety_filter_py(x, float(self.a), self.prev_in, self.prev_dc, out=self._buffer)
        return y

class NormalizerCompressorNode(Node):
    def __init__(self, name, n_ch=2, alpha=0.5, attack=16, sustain=128, decay=256):
        super().__init__(name)
        self.n_ch = n_ch
        self.alpha = alpha
        self.attack = attack
        self.sustain = sustain
        self.decay = decay
        self.rms_hist = [np.zeros(n_ch, dtype=RAW_DTYPE) for _ in range(decay)]
        self.peak_hist = [np.zeros(n_ch, dtype=RAW_DTYPE) for _ in range(decay)]
        self.stats = ClipStats()
    def process(self, frames, sr, audio_in, mods, params):
        # Compute RMS and peak over moving window (attack/sustain/decay)
        rms = np.sqrt(np.mean(audio_in**2, axis=1))
        peak = np.max(np.abs(audio_in), axis=1)
        self.rms_hist.pop(0)
        self.rms_hist.append(rms)
        self.peak_hist.pop(0)
        self.peak_hist.append(peak)
        # Weighted window: attack, sustain, decay
        def weighted_avg(hist, attack, sustain, decay):
            hist_arr = np.stack(hist)
            weights = np.concatenate([
                np.full(attack, 1.0),
                np.full(sustain, 0.5),
                np.full(decay, 0.25)
            ])
            weights = weights[:hist_arr.shape[0]]
            weights = weights / np.sum(weights)
            return np.sum(hist_arr * weights[:, None], axis=0)
        rms_val = weighted_avg(self.rms_hist, self.attack, self.sustain, self.decay)
        peak_val = weighted_avg(self.peak_hist, self.attack, self.sustain, self.decay)
        norm_val = self.alpha * rms_val + (1.0 - self.alpha) * peak_val + 1e-8
        # Normalize
        normed = audio_in / norm_val[:, None]
        # Tanh soft-knee compression
        compressed = np.tanh(normed)
        self.stats.update(compressed)
        return compressed

class SubharmonicLowLifterNode(Node):
    """Low-lifter style subharmonic enhancer operating purely on audio input."""

    def __init__(
        self,
        name,
        sample_rate,
        *,
        band_lo=70.0,
        band_hi=160.0,
        mix=0.5,
        drive=1.0,
        out_hp=25.0,
        use_div4=False,
    ):
        super().__init__(name)
        self.sr = float(sample_rate)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.mix = float(mix)
        self.drive = float(drive)
        self.out_hp = float(out_hp)
        self.use_div4 = bool(use_div4)
        self._init = False

    # --- coefficient helpers -------------------------------------------------
    def _alpha_lp(self, fc, sr):
        return 1.0 - math.exp(-2.0 * math.pi * max(fc, 1.0) / sr)

    def _alpha_hp(self, fc, sr):
        fc = max(fc, 1.0)
        rc = 1.0 / (2.0 * math.pi * fc)
        return rc / (rc + 1.0 / sr)

    def _ensure_state(self, B, C):
        if self._init and self.prev.shape == (B, C):
            return
        self.hp_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.lp_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.prev = np.zeros((B, C), dtype=RAW_DTYPE)
        self.sign = np.zeros((B, C), dtype=np.int8)
        self.ff2 = np.ones((B, C), dtype=np.int8)
        self.sub2_lp = np.zeros((B, C), dtype=RAW_DTYPE)
        self.env = np.zeros((B, C), dtype=RAW_DTYPE)
        self.hp_out_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.hp_out_x = np.zeros((B, C), dtype=RAW_DTYPE)
        if self.use_div4:
            self.ff4 = np.ones((B, C), dtype=np.int8)
            self.ff4_count = np.zeros((B, C), dtype=np.int32)
            self.sub4_lp = np.zeros((B, C), dtype=RAW_DTYPE)
        else:
            self.ff4 = None
            self.ff4_count = None
            self.sub4_lp = None
        self._init = True

    @staticmethod
    def _scalar_param(params, key, default):
        value = params.get(key)
        if value is None:
            return float(default)
        arr = np.asarray(value, dtype=RAW_DTYPE)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.reshape(-1)[-1])

    def process(self, frames, sr, audio_in, mods, params):
        if audio_in is None:
            return np.zeros((1, 1, frames), dtype=RAW_DTYPE)

        x = assert_BCF(audio_in, name=f"{self.name}.in")
        B, C, F = x.shape

        self._ensure_state(B, C)

        # Allow slow parameter modulation by sampling the last provided value.
        sr = float(sr or self.sr)
        band_lo = self._scalar_param(params, "band_lo", self.band_lo)
        band_hi = self._scalar_param(params, "band_hi", self.band_hi)
        mix = self._scalar_param(params, "mix", self.mix)
        drive = self._scalar_param(params, "drive", self.drive)
        out_hp = self._scalar_param(params, "out_hp", self.out_hp)

        a_hp_in = self._alpha_hp(band_lo, sr)
        a_lp_in = self._alpha_lp(band_hi, sr)
        a_sub2 = self._alpha_lp(max(band_hi / 3.0, 30.0), sr)
        a_sub4 = self._alpha_lp(max(band_hi / 5.0, 20.0), sr) if self.use_div4 else None
        a_env_attack = self._alpha_lp(100.0, sr)
        a_env_release = self._alpha_lp(5.0, sr)
        a_hp_out = self._alpha_hp(out_hp, sr)

        if not c_kernels.AVAILABLE:
            raise RuntimeError(
                "SubharmonicLowLifterNode requires compiled C kernels; Python fallbacks are forbidden."
            )
        y = np.empty_like(x)
        try:
            y = c_kernels.subharmonic_process_c(
                x,
                a_hp_in,
                a_lp_in,
                a_sub2,
                self.use_div4,
                a_sub4 if a_sub4 is not None else 0.0,
                a_env_attack,
                a_env_release,
                a_hp_out,
                drive,
                mix,
                self.hp_y,
                self.lp_y,
                self.prev,
                self.sign,
                self.ff2,
                self.ff4,
                self.ff4_count,
                self.sub2_lp,
                self.sub4_lp,
                self.env,
                self.hp_out_y,
                self.hp_out_x,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "SubharmonicLowLifterNode must execute the CFFI subharmonic kernel; Python fallback is unacceptable."
            ) from exc

        return y


NODE_TYPES = {
    "silence": SilenceNode,
    "constant": ConstantNode,
    "controller": ControllerNode,
    "sine": SineOscillatorNode,
    "sine_oscillator": SineOscillatorNode,
    "osc": OscNode,
    "oscillator": OscNode,
    "oscillator_pitch": OscillatorPitchNode,
    "OscillatorPitchNode": OscillatorPitchNode,
    "parametric_driver": ParametricDriverNode,
    "ParametricDriverNode": ParametricDriverNode,
    "fft_division": FFTDivisionNode,
    "FFTDivisionNode": FFTDivisionNode,
    "envelope": EnvelopeModulatorNode,
    "envelope_modulator": EnvelopeModulatorNode,
    "pitch_quantizer": PitchQuantizerNode,
    "amplifier_modulator": AmplifierModulatorNode,
    "mix": MixNode,
    "safety": SafetyNode,
    "subharmonic_low_lifter": SubharmonicLowLifterNode,
}


__all__ = [
    "NODE_TYPES",
    "Node",
    "SilenceNode",
    "ConstantNode",
    "ControllerNode",
    "SineOscillatorNode",
    "PitchQuantizerNode",
    "OscillatorPitchNode",
    "EnvelopeModulatorNode",
    "AmplifierModulatorNode",
    "OscNode",
    "ParametricDriverNode",
    "FFTDivisionNode",
    "MixNode",
    "SafetyNode",
    "SubharmonicLowLifterNode",
]
