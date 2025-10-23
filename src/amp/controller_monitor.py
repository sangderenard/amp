"""Polling helper that records controller events into the control history."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import threading
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class _Snapshot:
    """Lightweight container for the most recent controller state."""

    trigger: float
    gate: float
    drone: float
    velocity: float
    cutoff: float
    q: float
    pitch_input: float
    pitch_span: float
    pitch_root: float
    momentary_axis: float
    drone_axis: float


class ControllerMonitor:
    """Poll a controller and write derived controls into the graph history."""

    def __init__(
        self,
        poll_fn: Callable[[], object],
        control_history,
        poll_interval: float = 0.005,
        audio_frame_rate: float | None = None,
        *,
        state: Mapping[str, object] | None = None,
        axis_map: Mapping[str, int] | None = None,
        button_map: Mapping[str, int] | None = None,
    ) -> None:
        if poll_fn is None or not callable(poll_fn):
            raise ValueError("poll_fn must be a callable returning controller input")
        self.poll_fn = poll_fn
        self.control_history = control_history
        self.poll_interval = float(poll_interval)
        self.audio_frame_rate = float(audio_frame_rate) if audio_frame_rate else None
        self.running = False
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_snapshot: _Snapshot | None = None
        self._state = dict(state or {})
        self._axis_map = dict(axis_map or {})
        self._button_map = dict(button_map or {})
        self._gate_active = False
        self._drone_latched = False
        self._prev_drone_button = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, name="ControllerMonitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self.thread = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_latest_snapshot(self) -> _Snapshot | None:
        with self._lock:
            return self._latest_snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_axis_index(self, name: str) -> int | None:
        if name == "cutoff":
            return int(self._state.get("filter_axis_cutoff", 3))
        if name == "q":
            return int(self._state.get("filter_axis_q", 4))
        defaults = {
            "momentary": 2,
            "drone": 5,
            "velocity": 2,
            "pitch_input": 0,
        }
        return defaults.get(name)

    def _axis_value(self, axes: Sequence[float], name: str) -> float:
        index = self._axis_map.get(name, self._default_axis_index(name))
        if index is None:
            return 0.0
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return 0.0
        if idx < 0 or idx >= len(axes):
            return 0.0
        try:
            return float(axes[idx])
        except (TypeError, ValueError):
            return 0.0

    def _button_pressed(self, buttons: Sequence[float], name: str) -> bool:
        index = self._button_map.get(name)
        if index is None:
            defaults = {"momentary": 0, "drone_toggle": 1}
            index = defaults.get(name)
        if index is None:
            return False
        try:
            idx = int(index)
        except (TypeError, ValueError):
            return False
        if idx < 0 or idx >= len(buttons):
            return False
        try:
            value = float(buttons[idx])
        except (TypeError, ValueError):
            return False
        return value >= 0.5

    @staticmethod
    def _normalise_trigger(value: float) -> float:
        # Map controller trigger range [-1, 1] to [0, 1]
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, (value + 1.0) * 0.5))

    def _derive_snapshot(self, axes: Sequence[float], buttons: Sequence[float]) -> _Snapshot:
        momentary_axis = self._normalise_trigger(self._axis_value(axes, "momentary"))
        gate_button = self._button_pressed(buttons, "momentary")
        gate_now = gate_button or momentary_axis >= 0.6
        trigger_now = gate_now and not self._gate_active
        self._gate_active = gate_now

        drone_axis = self._normalise_trigger(self._axis_value(axes, "drone"))
        drone_button = self._button_pressed(buttons, "drone_toggle")
        if drone_button and not self._prev_drone_button:
            self._drone_latched = not self._drone_latched
        self._prev_drone_button = drone_button
        drone_now = self._drone_latched or drone_axis >= 0.6

        velocity_axis = self._normalise_trigger(self._axis_value(axes, "velocity"))
        pitch_axis = float(self._axis_value(axes, "pitch_input"))
        cutoff_axis = float(self._axis_value(axes, "cutoff"))
        q_axis = float(self._axis_value(axes, "q"))

        # Rescale into musically-useful ranges.
        velocity = float(velocity_axis)
        pitch_input = float(np.clip(pitch_axis, -1.0, 1.0))
        cutoff = 1500.0 + 1000.0 * float(np.clip(cutoff_axis, -1.0, 1.0))
        q_value = 0.9 + 0.3 * float(np.clip(q_axis, -1.0, 1.0))
        pitch_span = float(self._state.get("free_span_oct", 2.0))
        pitch_root = float(self._state.get("root_midi", 60))

        return _Snapshot(
            trigger=1.0 if trigger_now else 0.0,
            gate=1.0 if gate_now else 0.0,
            drone=1.0 if drone_now else 0.0,
            velocity=velocity,
            cutoff=cutoff,
            q=q_value,
            pitch_input=pitch_input,
            pitch_span=pitch_span,
            pitch_root=pitch_root,
            momentary_axis=float(momentary_axis),
            drone_axis=float(drone_axis),
        )

    def _record_snapshot(
        self, timestamp: float, axes: Sequence[float], buttons: Sequence[float]
    ) -> None:
        """Derive control extras from the latest axes/buttons sample and store them."""

        snapshot = self._derive_snapshot(axes, buttons)
        extras = {
            "trigger": np.asarray([snapshot.trigger], dtype=np.float64),
            "gate": np.asarray([snapshot.gate], dtype=np.float64),
            "drone": np.asarray([snapshot.drone], dtype=np.float64),
            "velocity": np.asarray([snapshot.velocity], dtype=np.float64),
            "cutoff": np.asarray([snapshot.cutoff], dtype=np.float64),
            "q": np.asarray([snapshot.q], dtype=np.float64),
            "pitch_input": np.asarray([snapshot.pitch_input], dtype=np.float64),
            "pitch_span": np.asarray([snapshot.pitch_span], dtype=np.float64),
            "pitch_root": np.asarray([snapshot.pitch_root], dtype=np.float64),
            "momentary_axis": np.asarray([snapshot.momentary_axis], dtype=np.float64),
            "drone_axis": np.asarray([snapshot.drone_axis], dtype=np.float64),
            "axes": np.asarray(list(axes), dtype=np.float64),
            "buttons": np.asarray(list(buttons), dtype=np.float64),
        }
        pitch = np.zeros(1, dtype=np.float64)
        envelope = np.zeros(1, dtype=np.float64)
        self.control_history.record_control_event(
            float(timestamp), pitch=pitch, envelope=envelope, extras=extras
        )
        with self._lock:
            self._latest_snapshot = snapshot

    def _coerce_axes_buttons(self, payload: object) -> tuple[list[float], list[float]]:
        if hasattr(payload, "axes") and hasattr(payload, "buttons"):
            axes = getattr(payload, "axes")
            buttons = getattr(payload, "buttons")
        elif isinstance(payload, Mapping):
            axes = payload.get("axes", [])
            buttons = payload.get("buttons", [])
        elif isinstance(payload, Sequence) and len(payload) >= 2:
            axes, buttons = payload[0], payload[1]
        else:
            axes, buttons = [], []

        def _to_list(values: Iterable[float]) -> list[float]:
            if values is None:
                return []
            return [float(v) for v in values]

        return _to_list(axes), _to_list(buttons)

    def _run(self) -> None:
        while self.running:
            timestamp = time.perf_counter()
            try:
                payload = self.poll_fn()
            except Exception:
                print("ControllerMonitor poll_fn raised an exception", file=sys.stderr)
                import traceback
                traceback.print_exc()
                time.sleep(self.poll_interval)
                continue

            axes, buttons = self._coerce_axes_buttons(payload)
            try:
                self._record_snapshot(timestamp, axes, buttons)
            except Exception:
                print(
                    "ControllerMonitor failed to record derived controller state",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc()
                # Fallback to recording raw controller payload so history still advances
                try:
                    extras = {
                        "axes": np.asarray(list(axes), dtype=np.float64),
                        "buttons": np.asarray(list(buttons), dtype=np.float64),
                    }
                    pitch = np.zeros(1, dtype=np.float64)
                    envelope = np.zeros(1, dtype=np.float64)
                    self.control_history.record_control_event(
                        float(timestamp), pitch=pitch, envelope=envelope, extras=extras
                    )
                except Exception:
                    print(
                        "ControllerMonitor failed to record raw controller state",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
            time.sleep(self.poll_interval)


__all__ = ["ControllerMonitor"]
