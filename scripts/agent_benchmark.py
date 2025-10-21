#!/usr/bin/env python3
"""Headless benchmarking helper for AMP agents.

This script renders the default controller graph without initialising the UI or
sounddevice backend.  It repeatedly renders the runtime graph, collects the
per-node timings exported by :class:`amp.graph.AudioGraph` and exposes slow
moving averages so agents can spot long term regressions without staring at the
interactive HUD.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

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
    sample_rate: float,
    cache: Dict[str, np.ndarray],
    iteration: int,
    envelope_names: list[str],
    amp_mod_names: list[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    base_params: Dict[str, Dict[str, np.ndarray]] = {"_B": 1, "_C": 1}

    base_params["keyboard_ctrl"] = {
        "trigger": _assign_control(cache, "keyboard.trigger", frames, 0.0),
        "gate": _assign_control(cache, "keyboard.gate", frames, 0.0),
        "drone": _assign_control(cache, "keyboard.drone", frames, 0.0),
        "velocity": _assign_control(cache, "keyboard.velocity", frames, 0.0),
    }

    trigger = _assign_control(cache, "joystick.trigger", frames, 0.0)
    trigger.fill(0.0)
    if iteration % 32 == 0:
        trigger[0, 0, 0] = 1.0
    gate = _assign_control(cache, "joystick.gate", frames, 1.0)
    drone = _assign_control(cache, "joystick.drone", frames, 0.0)

    velocity = _assign_control(cache, "joystick.velocity", frames, 0.75)
    cutoff = _assign_control(cache, "joystick.cutoff", frames, 1500.0)
    resonance = _assign_control(cache, "joystick.q", frames, 0.9)
    pitch_input = _assign_control(cache, "joystick.pitch_input", frames, 0.0)
    pitch_span = _assign_control(
        cache, "joystick.pitch_span", frames, float(state.get("free_span_oct", 2.0))
    )
    pitch_root = _assign_control(cache, "joystick.pitch_root", frames, float(state.get("root_midi", 60)))

    base_params["joystick_ctrl"] = {
        "trigger": trigger,
        "gate": gate,
        "drone": drone,
        "velocity": velocity,
        "cutoff": cutoff,
        "q": resonance,
        "pitch_input": pitch_input,
        "pitch_span": pitch_span,
        "pitch_root": pitch_root,
    }

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
        amp_base = _assign_control(cache, "amp_mod.base", frames, velocity)
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
) -> None:
    state = app_state.build_default_state(joy=None, pygame=_StubPygame())
    graph, envelope_names, amp_mod_names = amp_app.build_runtime_graph(sample_rate, state)

    control_cache: Dict[str, np.ndarray] = {}
    ema: Dict[str, float] = {}
    peaks: Dict[str, float] = defaultdict(float)
    totals: Dict[str, float] = defaultdict(float)
    count: Dict[str, int] = defaultdict(int)

    for iteration in range(iterations + warmup):
        params = _build_base_params(
            graph,
            state,
            frames,
            sample_rate,
            control_cache,
            iteration,
            envelope_names,
            amp_mod_names,
        )
        graph.render_block(frames, sample_rate, params)
        timings = graph.last_node_timings
        if not timings:
            continue
        for name, duration in timings.items():
            peaks[name] = max(peaks[name], duration)
            totals[name] += duration
            count[name] += 1
            if iteration >= warmup:
                previous = ema.get(name)
                ema[name] = duration if previous is None else previous + ema_alpha * (duration - previous)

    produced_ms = frames / sample_rate * 1000.0
    print(f"Rendered {iterations} iterations of {frames} frames ({produced_ms:.2f} ms per block)")
    print()
    print(f"Moving averages (alpha={ema_alpha:.3f}) sorted by descending cost:")
    ordered = sorted(ema.items(), key=lambda item: item[1], reverse=True)
    for name, avg in ordered:
        peak = peaks.get(name, 0.0) * 1000.0
        mean = (totals[name] / max(1, count[name])) * 1000.0
        print(f"  {name:<24} avg {mean:7.3f} ms  ema {avg * 1000.0:7.3f} ms  peak {peak:7.3f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the default AMP controller graph headlessly")
    parser.add_argument("--frames", type=int, default=256, help="Frames per render block")
    parser.add_argument("--iterations", type=int, default=512, help="Number of benchmark iterations (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=32, help="Warmup iterations to discard from EMA")
    parser.add_argument("--rate", type=float, default=44100.0, help="Sample rate in Hz")
    parser.add_argument("--alpha", type=float, default=0.02, help="EMA smoothing factor (0-1)")
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
