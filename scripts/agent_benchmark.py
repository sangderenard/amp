#!/usr/bin/env python3
"""Headless benchmarking helper for AMP agents.

This script renders the default controller graph without initialising the UI or
sounddevice backend.  It repeatedly renders the runtime graph, collects the
per-node timings exported by :class:`amp.graph.AudioGraph` and exposes slow
moving averages so agents can spot long term regressions without staring at the
interactive HUD.  A virtual joystick performs smooth, expressive gestures so
agents can observe how controller-driven modulation affects performance.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from amp import app as amp_app
from amp import state as app_state
from amp import utils
from amp.graph import AudioGraph


class _StubPygame:
    """Minimal subset of pygame keycodes required by ``build_default_state``."""

    K_m = ord("m")
    K_k = ord("k")
    K_x = ord("x")
    K_y = ord("y")
    K_b = ord("b")
    K_n = ord("n")
    K_z = ord("z")
    K_PERIOD = ord(".")
    K_COMMA = ord(",")
    K_SLASH = ord("/")


@dataclass(frozen=True)
class _ControlTrack:
    """Piecewise automation used by :class:`PrewrittenJoystickScript`."""

    times: np.ndarray
    values: np.ndarray
    mode: str  # ``"linear"`` or ``"step"``

    def sample(self, *, start: float, duration: float, frames: int) -> np.ndarray:
        if frames <= 0:
            raise ValueError("frames must be positive")
        if self.times.size == 0:
            raise ValueError("control track must define at least one event")

        if frames == 1:
            sample_times = np.array([start], dtype=np.float64)
        else:
            sample_times = start + np.linspace(0.0, duration, frames, endpoint=False, dtype=np.float64)

        if self.mode == "step":
            indices = np.searchsorted(self.times, sample_times, side="right") - 1
            indices = np.clip(indices, 0, self.values.size - 1)
            samples = self.values[indices]
        else:
            samples = np.interp(sample_times, self.times, self.values)
        return samples.astype(utils.RAW_DTYPE, copy=False)


class PrewrittenJoystickScript:
    """Replay deterministic joystick gestures described by a JSON script."""

    def __init__(self, *, controls: Dict[str, _ControlTrack], mode: str | None) -> None:
        self._controls = controls
        self._mode = mode

    @property
    def mode(self) -> str | None:
        return self._mode

    def sample_block(
        self,
        *,
        start_time: float,
        duration: float,
        frames: int,
    ) -> Dict[str, np.ndarray]:
        block: Dict[str, np.ndarray] = {}
        for name, track in self._controls.items():
            block[name] = track.sample(start=start_time, duration=duration, frames=frames)
        return block


def _load_prewritten_script(path: pathlib.Path) -> PrewrittenJoystickScript:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Joystick script must contain a JSON object")

    controls_raw = data.get("controls")
    if not isinstance(controls_raw, dict) or not controls_raw:
        raise ValueError("Joystick script must provide a non-empty 'controls' object")

    tracks: Dict[str, _ControlTrack] = {}
    for name, spec in controls_raw.items():
        if isinstance(spec, dict):
            events = spec.get("events")
            mode = spec.get("mode", spec.get("type", "linear"))
        else:
            events = spec
            mode = "step" if name in {"gate", "drone"} else "linear"

        if mode not in {"linear", "step"}:
            raise ValueError(f"Control '{name}' has unsupported mode '{mode}'")
        if not isinstance(events, Sequence) or not events:
            raise ValueError(f"Control '{name}' must define a non-empty sequence of events")

        times: list[float] = []
        values: list[float] = []
        for entry in events:
            if isinstance(entry, Mapping):
                if "time" not in entry:
                    raise ValueError(f"Control '{name}' event missing 'time'")
                value = entry.get("value", entry.get("v"))
                if value is None:
                    raise ValueError(f"Control '{name}' event missing 'value'")
                time_val = float(entry["time"])
                value_val = float(value)
            elif isinstance(entry, Sequence) and len(entry) == 2:
                time_val = float(entry[0])
                value_val = float(entry[1])
            else:
                raise ValueError(
                    "Control events must be dicts with 'time'/'value' or 2-item sequences"
                )
            times.append(time_val)
            values.append(value_val)

        order = np.argsort(times)
        times_arr = np.asarray(times, dtype=np.float64)[order]
        values_arr = np.asarray(values, dtype=np.float64)[order]
        tracks[name] = _ControlTrack(times_arr, values_arr, mode)

    mode = data.get("mode")
    if mode is not None and mode not in {"switch", "axis"}:
        raise ValueError("Joystick script 'mode' must be 'switch', 'axis', or omitted")

    return PrewrittenJoystickScript(controls=tracks, mode=mode)


class VirtualJoystickPerformer:
    """Generate virtual joystick gestures mirroring AMP's runtime controls."""

    _SWITCH_MOMENTARY_LENGTH = 0.25
    _AXIS_THRESHOLD = 0.6

    def __init__(
        self,
        sample_rate: float,
        *,
        mode: str = "switch",
        script: PrewrittenJoystickScript | None = None,
    ) -> None:
        if mode not in {"switch", "axis"}:
            raise ValueError("mode must be 'switch' or 'axis'")
        if script is not None and script.mode is not None:
            mode = script.mode

        self._sample_rate = sample_rate
        self._rng = np.random.default_rng(0xA17)
        self._time = 0.0
        self._mode = mode
        self._script = script

        self._next_momentary = 0.0
        self._momentary_remaining = 0.0
        self._drone_active = False
        self._next_drone_toggle = 2.0

        self._axis_values: Dict[str, float] = {
            "cutoff": 1400.0,
            "q": 0.85,
            "pitch_input": 0.0,
        }
        self._axis_targets = dict(self._axis_values)
        self._axis_remaining: Dict[str, float] = {"cutoff": 0.0, "q": 0.0, "pitch_input": 0.0}

        self._momentary_axis = 0.0
        self._drone_axis = 0.0
        self._momentary_axis_target = 0.0
        self._drone_axis_target = 0.0
        self._momentary_axis_remaining = 0.0
        self._drone_axis_remaining = 0.0
        self._momentary_engaged = False
        self._drone_engaged = False
        self._momentary_charge = 0.0
        self._velocity_value = 0.55
        self._velocity_target = 0.55

        self._pitch_span = 2.0
        self._pitch_root = 60.0

    def _schedule_next_momentary(self) -> None:
        self._next_momentary = self._time + float(self._rng.uniform(0.65, 1.35))

    def _toggle_drone(self) -> None:
        self._drone_active = not self._drone_active
        if self._drone_active:
            self._next_drone_toggle = self._time + float(self._rng.uniform(4.0, 6.5))
        else:
            self._next_drone_toggle = self._time + float(self._rng.uniform(2.0, 3.5))

    def _maybe_refresh_axis_targets(self) -> None:
        for name, remaining in list(self._axis_remaining.items()):
            if remaining > 0.0:
                self._axis_remaining[name] = max(0.0, remaining)
                continue
            centre = {
                "cutoff": 1500.0,
                "q": 0.9,
                "pitch_input": 0.0,
            }[name]
            width = {
                "cutoff": 900.0,
                "q": 0.4,
                "pitch_input": 0.45,
            }[name]
            self._axis_targets[name] = centre + float(self._rng.uniform(-width, width))
            self._axis_remaining[name] = float(self._rng.uniform(1.8, 3.6))

    def _advance_axes(self, frames: int) -> Dict[str, np.ndarray]:
        self._maybe_refresh_axis_targets()
        duration = frames / self._sample_rate
        curves: Dict[str, np.ndarray] = {}
        for name in self._axis_values:
            start = self._axis_values[name]
            target = self._axis_targets[name]
            curve = np.linspace(start, target, frames, dtype=utils.RAW_DTYPE)
            curves[name] = curve
            self._axis_values[name] = float(curve[-1])
            self._axis_remaining[name] = max(0.0, self._axis_remaining[name] - duration)
        return curves

    def _render_axis_curves(
        self,
        *,
        frames: int,
        momentary_curve: np.ndarray,
        drone_curve: np.ndarray,
        prev_momentary: float,
    ) -> Dict[str, np.ndarray]:
        if momentary_curve.shape[0] != frames or drone_curve.shape[0] != frames:
            raise ValueError("Axis curves must match the requested frame count")

        threshold = self._AXIS_THRESHOLD
        gate_mask = momentary_curve >= threshold
        drone_mask = drone_curve >= threshold

        positive_delta = np.maximum(np.diff(momentary_curve, prepend=prev_momentary), 0.0)
        trigger = np.zeros(frames, dtype=utils.RAW_DTYPE)
        charge = self._momentary_charge
        engaged = self._momentary_engaged

        for idx, active in enumerate(gate_mask):
            if active and not engaged:
                trigger[idx] = 1.0
                strike = charge + float(positive_delta[idx])
                charge = 0.0
                target = 0.45 + min(0.5, strike * 1.2)
                self._velocity_target = max(self._velocity_target, target)
            elif not active:
                charge = min(2.0, charge + float(positive_delta[idx]))
            engaged = bool(active)

        self._momentary_charge = charge
        self._momentary_engaged = engaged
        self._drone_engaged = bool(drone_mask[-1]) if frames else self._drone_engaged

        if not gate_mask.any():
            self._velocity_target = max(0.45, self._velocity_target - 0.05)
        if drone_mask.any():
            self._velocity_target = max(self._velocity_target, 0.58)
        else:
            self._velocity_target = max(0.45, self._velocity_target - 0.02)

        velocity_curve = np.linspace(
            self._velocity_value,
            self._velocity_target,
            frames,
            dtype=utils.RAW_DTYPE,
        )
        if frames:
            self._velocity_value = float(velocity_curve[-1])

        return {
            "trigger": trigger,
            "gate": gate_mask.astype(utils.RAW_DTYPE, copy=False),
            "drone": drone_mask.astype(utils.RAW_DTYPE, copy=False),
            "velocity": velocity_curve,
        }

    def _advance_switch_controls(self, frames: int) -> Dict[str, np.ndarray | float]:
        block_duration = frames / self._sample_rate
        trigger = np.zeros(frames, dtype=utils.RAW_DTYPE)
        gate = np.zeros(frames, dtype=utils.RAW_DTYPE)

        if self._time >= self._next_momentary:
            self._momentary_remaining = self._SWITCH_MOMENTARY_LENGTH
            trigger[0] = 1.0
            self._schedule_next_momentary()

        if self._momentary_remaining > 0.0:
            samples_high = min(frames, int(round(self._momentary_remaining * self._sample_rate)))
            if samples_high > 0:
                gate[:samples_high] = 1.0
            self._momentary_remaining = max(0.0, self._momentary_remaining - block_duration)

        if self._time >= self._next_drone_toggle:
            self._toggle_drone()

        drone = np.full(frames, 1.0 if self._drone_active else 0.0, dtype=utils.RAW_DTYPE)

        velocity = np.full(frames, 0.48, dtype=utils.RAW_DTYPE)
        if np.any(gate):
            velocity = np.where(gate > 0.0, 0.82, velocity)
        if np.any(drone):
            velocity = np.maximum(velocity, 0.62)

        self._momentary_engaged = bool(gate[-1]) if frames else self._momentary_engaged
        self._drone_engaged = bool(drone[-1]) if frames else self._drone_engaged
        self._momentary_charge = 0.0
        if frames:
            self._velocity_target = float(velocity[-1])
            self._velocity_value = self._velocity_target

        axis_curves = self._advance_axes(frames)

        return {
            "trigger": trigger,
            "gate": gate,
            "drone": drone,
            "velocity": velocity,
            "cutoff": axis_curves["cutoff"],
            "q": axis_curves["q"],
            "pitch_input": axis_curves["pitch_input"],
            "pitch_span": float(self._pitch_span),
            "pitch_root": float(self._pitch_root),
        }

    def _advance_axis_controls(self, frames: int) -> Dict[str, np.ndarray | float]:
        duration = frames / self._sample_rate

        if self._momentary_axis_remaining <= 0.0:
            if self._momentary_engaged:
                self._momentary_axis_target = 0.0
                self._momentary_axis_remaining = float(self._rng.uniform(0.18, 0.4))
            else:
                self._momentary_axis_target = float(self._rng.uniform(0.7, 1.0))
                self._momentary_axis_remaining = float(self._rng.uniform(0.28, 0.55))

        if self._drone_axis_remaining <= 0.0:
            if self._drone_engaged:
                self._drone_axis_target = 0.0
                self._drone_axis_remaining = float(self._rng.uniform(0.35, 0.75))
            else:
                self._drone_axis_target = float(self._rng.uniform(0.55, 1.0))
                self._drone_axis_remaining = float(self._rng.uniform(0.9, 1.7))

        prev_momentary = self._momentary_axis

        momentary_curve = np.linspace(
            self._momentary_axis,
            self._momentary_axis_target,
            frames,
            dtype=utils.RAW_DTYPE,
        )
        drone_curve = np.linspace(
            self._drone_axis,
            self._drone_axis_target,
            frames,
            dtype=utils.RAW_DTYPE,
        )

        if frames:
            self._momentary_axis = float(momentary_curve[-1])
            self._drone_axis = float(drone_curve[-1])
        self._momentary_axis_remaining = max(0.0, self._momentary_axis_remaining - duration)
        self._drone_axis_remaining = max(0.0, self._drone_axis_remaining - duration)

        axis_output = self._render_axis_curves(
            frames=frames,
            momentary_curve=momentary_curve,
            drone_curve=drone_curve,
            prev_momentary=prev_momentary,
        )
        axis_curves = self._advance_axes(frames)
        axis_output.update(
            {
                "cutoff": axis_curves["cutoff"],
                "q": axis_curves["q"],
                "pitch_input": axis_curves["pitch_input"],
                "pitch_span": float(self._pitch_span),
                "pitch_root": float(self._pitch_root),
            }
        )
        return axis_output

    def _apply_script_block(
        self,
        frames: int,
        block: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray | float]:
        result: Dict[str, np.ndarray | float] = {}

        axis_curves = self._advance_axes(frames)

        momentary_axis = block.get("momentary_axis")
        drone_axis = block.get("drone_axis")
        axis_output: Dict[str, np.ndarray] | None = None
        if momentary_axis is not None or drone_axis is not None:
            if momentary_axis is None or drone_axis is None:
                raise ValueError("Joystick script must provide both momentary_axis and drone_axis")
            momentary_curve = np.asarray(momentary_axis, dtype=utils.RAW_DTYPE)
            drone_curve = np.asarray(drone_axis, dtype=utils.RAW_DTYPE)
            if momentary_curve.shape[0] != frames or drone_curve.shape[0] != frames:
                raise ValueError("Script axis curves must match the block frame count")
            prev_momentary = self._momentary_axis
            if frames:
                self._momentary_axis = float(momentary_curve[-1])
                self._drone_axis = float(drone_curve[-1])
            axis_output = self._render_axis_curves(
                frames=frames,
                momentary_curve=momentary_curve,
                drone_curve=drone_curve,
                prev_momentary=prev_momentary,
            )

        def _control_array(name: str, source: Any) -> np.ndarray:
            array = np.asarray(source, dtype=utils.RAW_DTYPE)
            if array.ndim == 0:
                array = np.full(frames, float(array), dtype=utils.RAW_DTYPE)
            if array.shape[0] != frames:
                raise ValueError(f"Control '{name}' expected {frames} samples, received {array.shape[0]}")
            return array

        for name in ("trigger", "gate", "drone", "velocity"):
            if name in block:
                arr = _control_array(name, block[name])
                if name == "velocity" and arr.size:
                    self._velocity_target = float(arr[-1])
                    self._velocity_value = self._velocity_target
                if name == "gate" and arr.size:
                    self._momentary_engaged = bool(arr[-1] >= 0.5)
                if name == "drone" and arr.size:
                    self._drone_engaged = bool(arr[-1] >= 0.5)
                result[name] = arr
            elif axis_output is not None and name in axis_output:
                result[name] = axis_output[name]
            else:
                fill = 0.0
                result[name] = np.full(frames, fill, dtype=utils.RAW_DTYPE)

        for name in ("cutoff", "q", "pitch_input"):
            if name in block:
                result[name] = _control_array(name, block[name])
            else:
                result[name] = axis_curves[name]

        result["pitch_span"] = float(block.get("pitch_span", self._pitch_span))
        result["pitch_root"] = float(block.get("pitch_root", self._pitch_root))
        return result

    def generate(self, frames: int) -> Dict[str, np.ndarray | float]:
        """Return simulated joystick control curves for a render block."""

        block_duration = frames / self._sample_rate

        if self._script is not None:
            script_block = self._script.sample_block(
                start_time=self._time,
                duration=block_duration,
                frames=frames,
            )
            curves = self._apply_script_block(frames, script_block)
        elif self._mode == "axis":
            curves = self._advance_axis_controls(frames)
        else:
            curves = self._advance_switch_controls(frames)

        self._time += block_duration
        return curves


def _control_view(cache: Dict[str, np.ndarray], key: str, frames: int) -> np.ndarray:
    view = cache.get(key)
    if view is None or view.shape[2] < frames:
        new_frames = 1 << max(0, frames - 1).bit_length()
        view = np.zeros((1, 1, new_frames), dtype=utils.RAW_DTYPE)
        cache[key] = view
    return view[:, :, :frames]


def _assign_control(
    cache: Dict[str, np.ndarray], key: str, frames: int, value: float | np.ndarray
) -> np.ndarray:
    view = _control_view(cache, key, frames)
    array = np.asarray(value, dtype=utils.RAW_DTYPE)
    if array.ndim == 0:
        view.fill(float(array))
        return view
    if array.ndim == 1:
        if array.shape[0] != frames:
            raise ValueError(f"{key}: expected {frames} samples, got {array.shape[0]}")
        view[0, 0, :frames] = array
        return view
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] == 1 and array.shape[2] >= frames:
        view[...] = array[:, :, :frames]
        return view
    raise ValueError(f"Unsupported control shape for '{key}': {array.shape}")


def _build_base_params(
    graph: AudioGraph,
    state: Dict[str, Any],
    frames: int,
    cache: Dict[str, np.ndarray],
    envelope_names: list[str],
    amp_mod_names: list[str],
    joystick_curves: Mapping[str, np.ndarray | float],
) -> Dict[str, Dict[str, np.ndarray]]:
    base_params: Dict[str, Dict[str, np.ndarray]] = {"_B": 1, "_C": 1}

    base_params["keyboard_ctrl"] = {
        "trigger": _assign_control(cache, "keyboard.trigger", frames, 0.0),
        "gate": _assign_control(cache, "keyboard.gate", frames, 0.0),
        "drone": _assign_control(cache, "keyboard.drone", frames, 0.0),
        "velocity": _assign_control(cache, "keyboard.velocity", frames, 0.0),
    }

    joystick_params: Dict[str, np.ndarray] = {}
    for key in (
        "trigger",
        "gate",
        "drone",
        "velocity",
        "cutoff",
        "q",
        "pitch_input",
        "pitch_span",
        "pitch_root",
    ):
        value = joystick_curves.get(key)
        if value is None:
            if key == "pitch_span":
                value = float(state.get("free_span_oct", 2.0))
            elif key == "pitch_root":
                value = float(state.get("root_midi", 60))
            else:
                value = 0.0
        joystick_params[key] = _assign_control(
            cache,
            f"joystick.{key}",
            frames,
            value,
        )

    base_params["joystick_ctrl"] = joystick_params

    pitch_node = graph._nodes.get("pitch")
    if pitch_node is not None:
        pitch_node.update_mode(
            effective_token=state.get("base_token", "12tet/full"),
            free_variant=state.get("free_variant", "continuous"),
            span_oct=float(state.get("free_span_oct", 2.0)),
        )

    osc_names = [name for name in ("osc1", "osc2", "osc3") if name in graph._nodes]
    for idx, name in enumerate(osc_names):
        freq = 110.0 * (idx + 2)
        amp = 0.3 if idx == 0 else 0.25
        base_params[name] = {
            "freq": _assign_control(cache, f"{name}.freq", frames, freq),
            "amp": _assign_control(cache, f"{name}.amp", frames, amp),
        }

    if envelope_names:
        send_reset = _assign_control(cache, "envelope.send_reset", frames, 1.0)
        for env_name in envelope_names:
            base_params[env_name] = {"send_reset": send_reset}

    if amp_mod_names:
        amp_base = joystick_params["velocity"]
        for mod_name in amp_mod_names:
            base_params[mod_name] = {"base": amp_base}

    return base_params


def benchmark_default_graph(
    *,
    frames: int,
    iterations: int,
    sample_rate: float,
    ema_alpha: float,
    warmup: int,
    joystick_mode: str,
    joystick_script: pathlib.Path | None,
    ) -> pd.DataFrame:
    state = app_state.build_default_state(joy=None, pygame=_StubPygame())
    graph, envelope_names, amp_mod_names = amp_app.build_runtime_graph(sample_rate, state)

    control_cache: Dict[str, np.ndarray] = {}
    ema: Dict[str, float] = {}
    peaks: Dict[str, float] = defaultdict(float)
    totals: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)
    if joystick_script is not None:
        try:
            script = _load_prewritten_script(joystick_script)
        except ValueError as exc:
            raise SystemExit(f"Failed to load joystick script: {exc}") from exc
    else:
        script = None
    virtual_joystick = VirtualJoystickPerformer(sample_rate, mode=joystick_mode, script=script)

    timeline_records: list[Dict[str, Any]] = []
    timeline_start = time.perf_counter()
    playhead_time = 0.0
    buffer_ahead = 0.0
    cumulative_gap = 0.0

    for iteration in range(iterations + warmup):
        joystick_curves = virtual_joystick.generate(frames)
        params = _build_base_params(
            graph,
            state,
            frames,
            control_cache,
            envelope_names,
            amp_mod_names,
            joystick_curves,
        )
        block_start_time = time.perf_counter()
        audio_block = graph.render_block(frames, sample_rate, params)
        block_end_time = time.perf_counter()

        render_duration = block_end_time - block_start_time
        block_duration = frames / sample_rate
        buffer_ahead += block_duration
        buffer_ahead -= render_duration
        underrun_gap = 0.0
        if buffer_ahead < 0.0:
            underrun_gap = -buffer_ahead
            cumulative_gap += underrun_gap
            buffer_ahead = 0.0

        scheduled_start = playhead_time
        scheduled_end = scheduled_start + block_duration
        realised_start = scheduled_start + cumulative_gap
        realised_end = realised_start + block_duration
        playhead_time = scheduled_end

        timings = graph.last_node_timings
        if timings:
            for name, duration in timings.items():
                peaks[name] = max(peaks[name], duration)
                totals[name] += duration
                count[name] += 1
                if iteration >= warmup:
                    previous = ema.get(name)
                    ema[name] = (
                        duration
                        if previous is None
                        else previous + ema_alpha * (duration - previous)
                    )

        audio_abs = np.abs(audio_block)
        audio_peak = float(np.max(audio_abs)) if audio_abs.size else 0.0
        audio_rms = float(np.sqrt(np.mean(np.square(audio_block)))) if audio_abs.size else 0.0
        channel_peaks: list[float] = []
        channel_rms: list[float] = []
        if audio_abs.size:
            per_channel_peaks = np.max(audio_abs, axis=2)
            per_channel_rms = np.sqrt(np.mean(np.square(audio_block), axis=2))
            channel_peaks = per_channel_peaks.flatten().tolist()
            channel_rms = per_channel_rms.flatten().tolist()

        wall_start = block_start_time - timeline_start
        wall_end = block_end_time - timeline_start

        def _curve_mean(value: Any) -> float:
            array = np.asarray(value, dtype=np.float64)
            if array.size == 0:
                return 0.0
            return float(np.mean(array))

        def _curve_max(value: Any) -> float:
            array = np.asarray(value, dtype=np.float64)
            if array.size == 0:
                return 0.0
            return float(np.max(array))

        gate_curve = joystick_curves.get("gate", 0.0)
        drone_curve = joystick_curves.get("drone", 0.0)
        velocity_curve = joystick_curves.get("velocity", 0.0)
        pitch_curve = joystick_curves.get("pitch_input", 0.0)

        record: Dict[str, Any] = {
            "iteration": iteration,
            "is_warmup": iteration < warmup,
            "scheduled_start_ms": scheduled_start * 1000.0,
            "scheduled_end_ms": scheduled_end * 1000.0,
            "realised_start_ms": realised_start * 1000.0,
            "realised_end_ms": realised_end * 1000.0,
            "wall_start_ms": wall_start * 1000.0,
            "wall_end_ms": wall_end * 1000.0,
            "render_ms": render_duration * 1000.0,
            "block_ms": block_duration * 1000.0,
            "buffer_ahead_ms": buffer_ahead * 1000.0,
            "underrun_gap_ms": underrun_gap * 1000.0,
            "cumulative_gap_ms": cumulative_gap * 1000.0,
            "audio_peak": audio_peak,
            "audio_rms": audio_rms,
            "audio_channel_peaks": channel_peaks,
            "audio_channel_rms": channel_rms,
            "momentary_active": bool(_curve_max(gate_curve) >= 0.5),
            "drone_active": bool(_curve_max(drone_curve) >= 0.5),
            "velocity_mean": _curve_mean(velocity_curve),
            "pitch_mean": _curve_mean(pitch_curve),
            "start_sample": iteration * frames,
            "end_sample": (iteration + 1) * frames,
        }

        for name, duration in timings.items():
            record[f"node_{name}_ms"] = duration * 1000.0

        timeline_records.append(record)

    produced_ms = frames / sample_rate * 1000.0
    print(f"Rendered {iterations} iterations of {frames} frames ({produced_ms:.2f} ms per block)")
    print()
    print(f"Moving averages (alpha={ema_alpha:.3f}) sorted by descending cost:")
    ordered = sorted(ema.items(), key=lambda item: item[1], reverse=True)
    for name, avg in ordered:
        peak = peaks.get(name, 0.0) * 1000.0
        mean = (totals[name] / max(1, count[name])) * 1000.0
        print(f"  {name:<24} avg {mean:7.3f} ms  ema {avg * 1000.0:7.3f} ms  peak {peak:7.3f} ms")

    timeline_df = pd.DataFrame.from_records(timeline_records)
    if not timeline_df.empty:
        warmup_df = timeline_df.loc[~timeline_df["is_warmup"]]
        underrun_count = int((warmup_df["underrun_gap_ms"] > 0.0).sum()) if not warmup_df.empty else 0
        total_gap = float(warmup_df["underrun_gap_ms"].sum()) if not warmup_df.empty else 0.0
        print()
        print(
            "Real-time timeline summary:"
            f" {len(warmup_df)} measured blocks, {underrun_count} underruns"
            f" totalling {total_gap:.3f} ms"
        )
        preview_count = min(6, len(timeline_df))
        print()
        print("Timeline preview (first rows):")
        preview = timeline_df.head(preview_count)
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(preview.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    return timeline_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the default AMP controller graph headlessly")
    parser.add_argument("--frames", type=int, default=256, help="Frames per render block")
    parser.add_argument("--iterations", type=int, default=512, help="Number of benchmark iterations (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=32, help="Warmup iterations to discard from EMA")
    parser.add_argument("--rate", type=float, default=44100.0, help="Sample rate in Hz")
    parser.add_argument("--alpha", type=float, default=0.02, help="EMA smoothing factor (0-1)")
    parser.add_argument(
        "--joystick-mode",
        choices=("switch", "axis"),
        default="switch",
        help="Virtual joystick style: 'switch' uses on/off buttons, 'axis' emulates analog strikes",
    )
    parser.add_argument(
        "--joystick-script",
        type=pathlib.Path,
        help="Optional JSON file containing prewritten joystick automation",
    )
    args = parser.parse_args()

    if args.frames <= 0:
        raise SystemExit("Frames must be positive")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("EMA alpha must be in the interval (0, 1]")
    if args.iterations <= 0:
        raise SystemExit("Iterations must be positive")

    benchmark_default_graph(
        frames=args.frames,
        iterations=args.iterations,
        sample_rate=args.rate,
        ema_alpha=args.alpha,
        warmup=max(0, args.warmup),
        joystick_mode=args.joystick_mode,
        joystick_script=args.joystick_script,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
