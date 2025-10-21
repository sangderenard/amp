"""Virtual joystick performer and prewritten script loader.

This module exposes a deterministic virtual controller used by the
headless benchmark.  Extracting it from the script makes it reusable by the
interactive application and other tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from . import utils


@dataclass(frozen=True)
class _ControlTrack:
    times: np.ndarray
    values: np.ndarray
    mode: str

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
    def __init__(self, *, controls: Dict[str, _ControlTrack], mode: str | None) -> None:
        self._controls = controls
        self._mode = mode

    @property
    def mode(self) -> str | None:
        return self._mode

    def sample_block(self, *, start_time: float, duration: float, frames: int) -> Dict[str, np.ndarray]:
        block: Dict[str, np.ndarray] = {}
        for name, track in self._controls.items():
            block[name] = track.sample(start=start_time, duration=duration, frames=frames)
        return block


def _load_prewritten_script(path) -> PrewrittenJoystickScript:
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
        if not isinstance(events, (list, tuple)) or not events:
            raise ValueError(f"Control '{name}' must define a non-empty sequence of events")

        times: list[float] = []
        values: list[float] = []
        for entry in events:
            if isinstance(entry, dict):
                if "time" not in entry:
                    raise ValueError(f"Control '{name}' event missing 'time'")
                value = entry.get("value", entry.get("v"))
                if value is None:
                    raise ValueError(f"Control '{name}' event missing 'value'")
                time_val = float(entry["time"])
                value_val = float(value)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                time_val = float(entry[0])
                value_val = float(entry[1])
            else:
                raise ValueError("Control events must be dicts with 'time'/'value' or 2-item sequences")
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

    # The rest of the implementation mirrors the benchmark script and is
    # intentionally kept unchanged to preserve deterministic behaviour.
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

    def _render_axis_curves(self, *, frames: int, momentary_curve: np.ndarray, drone_curve: np.ndarray, prev_momentary: float) -> Dict[str, np.ndarray]:
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

        velocity_curve = np.linspace(self._velocity_value, self._velocity_target, frames, dtype=utils.RAW_DTYPE)
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

        momentary_curve = np.linspace(self._momentary_axis, self._momentary_axis_target, frames, dtype=utils.RAW_DTYPE)
        drone_curve = np.linspace(self._drone_axis, self._drone_axis_target, frames, dtype=utils.RAW_DTYPE)

        if frames:
            self._momentary_axis = float(momentary_curve[-1])
            self._drone_axis = float(drone_curve[-1])
        self._momentary_axis_remaining = max(0.0, self._momentary_axis_remaining - duration)
        self._drone_axis_remaining = max(0.0, self._drone_axis_remaining - duration)

        axis_output = self._render_axis_curves(frames=frames, momentary_curve=momentary_curve, drone_curve=drone_curve, prev_momentary=prev_momentary)
        axis_curves = self._advance_axes(frames)
        axis_output.update({
            "cutoff": axis_curves["cutoff"],
            "q": axis_curves["q"],
            "pitch_input": axis_curves["pitch_input"],
            "pitch_span": float(self._pitch_span),
            "pitch_root": float(self._pitch_root),
        })
        return axis_output

    def _apply_script_block(self, frames: int, block: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | float]:
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
            axis_output = self._render_axis_curves(frames=frames, momentary_curve=momentary_curve, drone_curve=drone_curve, prev_momentary=prev_momentary)

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
        block_duration = frames / self._sample_rate

        if self._script is not None:
            script_block = self._script.sample_block(start_time=self._time, duration=block_duration, frames=frames)
            curves = self._apply_script_block(frames, script_block)
        elif self._mode == "axis":
            curves = self._advance_axis_controls(frames)
        else:
            curves = self._advance_switch_controls(frames)

        self._time += block_duration
        return curves


__all__ = ["VirtualJoystickPerformer", "PrewrittenJoystickScript", "_load_prewritten_script"]
