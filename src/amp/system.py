"""Shared runtime runner utilities (benchmarks, headless runs)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from . import app as amp_app
from . import c_kernels, native_runtime
from .virtual_joystick import VirtualJoystickPerformer, _load_prewritten_script
from .graph import AudioGraph


def require_native_graph_runtime() -> None:
    """Ensure the native C graph runtime is available before rendering.

    All benchmark and diagnostic entry points *must* route through the C edge
    runner.  Python fallbacks exist solely for development convenience and are
    unsuitable for performance measurements or crash triage because they bypass
    the compiled execution plan.  This guard keeps every public entry point
    aligned with that policy by refusing to proceed when the native bindings are
    missing.
    """

    if not c_kernels.AVAILABLE:
        reason = c_kernels.UNAVAILABLE_REASON or "unknown reason"
        raise RuntimeError(
            "C kernels are required for graph benchmarking and diagnostics; "
            f"unavailable ({reason})."
        )
    if not native_runtime.AVAILABLE:
        reason = native_runtime.UNAVAILABLE_REASON or "unknown reason"
        raise RuntimeError(
            "Native graph runtime is unavailable; all graph operations must run in C. "
            f"Loader reported: {reason}."
        )


def benchmark_default_graph(
    *,
    frames: int,
    iterations: int,
    sample_rate: float,
    ema_alpha: float,
    warmup: int,
    joystick_mode: str,
    joystick_script_path,
) -> pd.DataFrame:
    """Run the default graph headlessly using a virtual controller.

    Returns a pandas DataFrame matching the shape produced by the original
    script-based benchmark helper.
    """

    require_native_graph_runtime()

    from . import state as app_state

    class _StubPygame:
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

    state = app_state.build_default_state(joy=None, pygame=_StubPygame())
    graph, envelope_names, amp_mod_names = amp_app.build_runtime_graph(sample_rate, state)

    control_cache: Dict[str, np.ndarray] = {}
    ema: Dict[str, float] = {}
    peaks: Dict[str, float] = defaultdict(float)
    totals: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)

    if joystick_script_path is not None:
        try:
            script = _load_prewritten_script(joystick_script_path)
        except ValueError as exc:
            raise
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
        timestamp = time.perf_counter()

        def _control_array(value: Any) -> np.ndarray:
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 0:
                return np.full(frames, float(array), dtype=np.float64)
            if array.shape[0] != frames:
                raise ValueError(
                    f"Control array expected {frames} samples, received {array.shape[0]}"
                )
            return array

        extras: Dict[str, Any] | None
        try:
            control_arrays: Dict[str, np.ndarray] = {
                key: _control_array(val)
                for key, val in joystick_curves.items()
            }

            # Derive synthetic raw axes/buttons mirroring ControllerMonitor's sampling.
            def _frame_or_default(key: str, default: float = 0.0) -> np.ndarray:
                array = control_arrays.get(key)
                if array is None:
                    return np.full(frames, default, dtype=np.float64)
                return array

            pitch_input_curve = np.clip(_frame_or_default("pitch_input"), -1.0, 1.0)
            velocity_curve = np.clip(_frame_or_default("velocity"), 0.0, 1.0)
            momentary_axis_curve = np.clip(_frame_or_default("momentary_axis", 0.0), 0.0, 1.0)
            drone_axis_curve = np.clip(
                _frame_or_default("drone_axis", 0.0),
                0.0,
                1.0,
            )
            cutoff_curve = _frame_or_default("cutoff", 1500.0)
            q_curve = _frame_or_default("q", 0.9)

            axes = np.zeros(6, dtype=np.float64)
            buttons = np.zeros(2, dtype=np.float64)

            if frames:
                axes[0] = float(pitch_input_curve[-1])
                axes[2] = float(momentary_axis_curve[-1] * 2.0 - 1.0)
                axes[3] = float(np.clip((cutoff_curve[-1] - 1500.0) / 1000.0, -1.0, 1.0))
                axes[4] = float(np.clip((q_curve[-1] - 0.9) / 0.3, -1.0, 1.0))
                axes[5] = float(np.clip(drone_axis_curve[-1] * 2.0 - 1.0, -1.0, 1.0))
                buttons[0] = float((_frame_or_default("trigger")[-1]) >= 0.5)
                buttons[1] = float((_frame_or_default("drone")[-1]) >= 0.5)

            def _snapshot(value: np.ndarray) -> np.ndarray:
                arr = np.asarray(value, dtype=np.float64)
                if arr.size == 0:
                    return np.asarray([0.0], dtype=np.float64)
                return np.asarray([float(arr.reshape(-1)[-1])], dtype=np.float64)

            extras = {"axes": axes, "buttons": buttons}
            for key, array in control_arrays.items():
                extras[key] = _snapshot(array)
        except Exception:
            extras = None

        pitch_stub = np.zeros(1, dtype=np.float64)
        envelope_stub = np.zeros(1, dtype=np.float64)
        graph.record_control_event(
            timestamp,
            pitch=pitch_stub,
            envelope=envelope_stub,
            extras=extras,
        )

        # Sample from the graph's retained history to obtain per-frame
        # pitch/envelope and any extras. This enforces the invariant that
        # the renderer only consumes data derived from ControlDelay.
        start_time = timestamp
        sampled = graph.sample_control_tensor(start_time, frames)
        sampled_pitch = sampled.get("pitch")
        sampled_envelope = sampled.get("envelope")
        sampled_extras = sampled.get("extras", {})

    # Build joystick_curves strictly from sampled history. This ensures
    # the headless path uses the exact same interpolation/control-delay
    # semantics as the interactive application.
        joystick_curves_from_history: dict = {}
        # Keys the interactive app expects as joystick_curves
        expected_keys = (
            "trigger",
            "gate",
            "drone",
            "velocity",
            "cutoff",
            "q",
            "pitch_input",
            "pitch_span",
            "pitch_root",
        )
        def _coerce_history_value(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                array = np.asarray(value)
                if array.ndim == 0:
                    return float(array)
                if array.ndim == 1:
                    return array
                # Flatten extra dimensions by taking the leading frame axis.
                return array.reshape(array.shape[0], -1)[:, 0]
            return value

        for key in expected_keys:
            if key in sampled_extras:
                joystick_curves_from_history[key] = _coerce_history_value(sampled_extras[key])
            else:
                if key == "pitch_input":
                    if sampled_pitch is None:
                        joystick_curves_from_history[key] = np.zeros(frames, dtype=float)
                    else:
                        pitch_array = np.asarray(sampled_pitch, dtype=float)
                        if pitch_array.ndim == 1:
                            joystick_curves_from_history[key] = pitch_array
                        else:
                            pitch_array = pitch_array.reshape(pitch_array.shape[0], -1)
                            joystick_curves_from_history[key] = pitch_array[:, 0]
                elif key == "pitch_span":
                    joystick_curves_from_history[key] = float(state.get("free_span_oct", 2.0))
                elif key == "pitch_root":
                    joystick_curves_from_history[key] = float(state.get("root_midi", 60))
                else:
                    joystick_curves_from_history[key] = np.zeros(frames, dtype=float)

        from .runner import render_audio_block
        block_start_time = time.perf_counter()
        audio_block, meta = render_audio_block(
            graph,
            block_start_time,
            frames,
            sample_rate,
            joystick_curves_from_history,
            state,
            envelope_names,
            amp_mod_names,
            control_cache,
        )
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

        timings = meta.get("node_timings", {})
        if timings:
            for name, duration in timings.items():
                peaks[name] = max(peaks[name], duration)
                totals[name] += duration
                count[name] += 1
                if iteration >= warmup:
                    previous = ema.get(name)
                    ema[name] = duration if previous is None else previous + ema_alpha * (duration - previous)

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

        gate_curve = joystick_curves_from_history.get("gate", 0.0)
        drone_curve = joystick_curves_from_history.get("drone", 0.0)
        velocity_curve = joystick_curves_from_history.get("velocity", 0.0)
        pitch_curve = joystick_curves_from_history.get("pitch_input", 0.0)

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

    timeline_df = pd.DataFrame.from_records(timeline_records)
    return timeline_df


__all__ = ["benchmark_default_graph", "require_native_graph_runtime"]
