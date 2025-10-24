import threading
import time
import queue
import math
import numpy as np
import pandas as pd

def run_producer_thread(
    graph,
    sample_rate: float,
    block_frames: int,
    batch_blocks: int,
    joystick_curves_fn,
    state,
    envelope_names,
    amp_mod_names,
    control_cache,
    pcm_queue: queue.Queue,
    stop_event: threading.Event,
    timing_samples: list,
):
    """
    Shared producer thread for both interactive and headless runs.
    Renders C-ready output node buffers in batches, enqueues PCM, and records timing.
    joystick_curves_fn(frames) -> dict for each block.
    All per-node data and audio blocks are C-ready buffers.
    """
    # Use ControllerMonitor for synthetic input, writing to control history
    from .controller_monitor import ControllerMonitor
    controller_monitor = ControllerMonitor(
        poll_fn=joystick_curves_fn,  # For synthetic input, this returns a dict of curves
        control_history=graph,
        poll_interval=1.0 / sample_rate,
        audio_frame_rate=sample_rate,
        synthetic=True
    )
    controller_monitor.start()
    try:
        while not stop_event.is_set():
            total_frames = block_frames * batch_blocks
            start_time = time.perf_counter()
            # ControllerMonitor writes synthetic input to history; sample history for params
            # joystick_curves is now always sourced from history
            joystick_curves = {}  # Not used directly; build_base_params will sample from history
            output_node_buffer, meta = render_audio_block(
                graph,
                start_time,
                total_frames,
                sample_rate,
                joystick_curves,
                state,
                envelope_names,
                amp_mod_names,
                control_cache,
            )
            end_time = time.perf_counter()
            meta["render_duration"] = end_time - start_time
            meta["produced_frames"] = total_frames
            meta["sample_rate"] = sample_rate
            meta["batch_blocks"] = batch_blocks
            meta["block_frames"] = block_frames
            timing_samples.append(meta)
            # Split into output node buffers and enqueue
            for idx in range(batch_blocks):
                node_buffer = output_node_buffer[:, :, idx * block_frames : (idx + 1) * block_frames]
                pcm_queue.put((node_buffer, meta))
    finally:
        controller_monitor.stop()

def run_benchmark(
    graph,
    sample_rate: float,
    block_frames: int,
    batch_blocks: int,
    iterations: int,
    joystick_curves_fn,
    state,
    envelope_names,
    amp_mod_names,
    control_cache,
) -> pd.DataFrame:
    """
    Shared headless benchmark runner.
    Runs render_audio_block in a loop, collects timing/efficiency stats.
    All audio blocks are C-ready output node buffers.
    Returns a pandas DataFrame of results.
    """
    timing_samples = []
    from .controller_monitor import ControllerMonitor
    controller_monitor = ControllerMonitor(
        poll_fn=joystick_curves_fn,
        control_history=graph,
        poll_interval=1.0 / sample_rate,
        audio_frame_rate=sample_rate,
        synthetic=True
    )
    controller_monitor.start()
    try:
        for i in range(iterations):
            start_time = time.perf_counter()
            joystick_curves = {}  # Not used directly; build_base_params will sample from history
            output_node_buffer, meta = render_audio_block(
                graph,
                start_time,
                block_frames,
                sample_rate,
                joystick_curves,
                state,
                envelope_names,
                amp_mod_names,
                control_cache,
            )
            end_time = time.perf_counter()
            meta["render_duration"] = end_time - start_time
            meta["produced_frames"] = block_frames
            meta["sample_rate"] = sample_rate
            meta["iteration"] = i
            timing_samples.append(meta)
    finally:
        controller_monitor.stop()
    return pd.DataFrame(timing_samples)
"""Shared audio block renderer for interactive and benchmark flows.
Ready for CFFI migration: uses only simple types and numpy arrays.
"""
from typing import Any, Dict
import numpy as np

def render_audio_block(
    graph,
    start_time: float,
    frames: int,
    sample_rate: float,
    joystick_curves: Dict[str, Any],
    state: Dict[str, Any],
    envelope_names: list,
    amp_mod_names: list,
    control_cache: Dict[str, np.ndarray],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Shared block renderer for both interactive and benchmark runs.
    - Records control events (if needed) before rendering.
    - Samples history and builds params.
    - Calls graph.render_block and returns a C-ready output node buffer + timing metadata.
    - All inputs/outputs are simple types or C-ready numpy arrays.
    """
    # Build base params using shared builder
    from .app import build_base_params
    output_frames = int(frames)
    output_sample_rate = float(sample_rate or graph.sample_rate)
    dsp_rate = float(getattr(graph, "dsp_sample_rate", output_sample_rate) or output_sample_rate)
    if output_sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    ratio = dsp_rate / output_sample_rate
    dsp_frames = int(math.ceil(output_frames * ratio)) if ratio > 0 else output_frames

    base_params = build_base_params(
        graph,
        state,
        dsp_frames,
        control_cache,
        envelope_names,
        amp_mod_names,
        joystick_curves,
        start_time=start_time,
        update_hz=dsp_rate,
    )
    try:
        try:
            with open("logs/py_c_calls.log", "a") as _pf:
                _pf.write(f"{time.time()} {threading.get_ident()} render_audio_block.enter frames={frames} sample_rate={sample_rate} base_params_keys={list((base_params or {}).keys())}\n")
        except Exception:
            pass
        output_node_buffer = graph.render_block(
            dsp_frames,
            dsp_rate,
            base_params,
            output_frames=output_frames,
            output_sample_rate=output_sample_rate,
            dsp_sample_rate=dsp_rate,
        )
        try:
            with open("logs/py_c_calls.log", "a") as _pf:
                _pf.write(f"{time.time()} {threading.get_ident()} render_audio_block.exit frames={frames} output_shape={getattr(output_node_buffer, 'shape', None)}\n")
        except Exception:
            pass
    except Exception:
        raise
    runner = getattr(graph, "_edge_runner", None)
    if runner is not None:
        fallback_summary = getattr(runner, "python_fallback_summary", lambda: {})()
        if fallback_summary:
            details = ", ".join(
                f"{name}={count}" for name, count in sorted(fallback_summary.items()) if count
            )
            raise RuntimeError(
                "Graph execution invoked Python fallbacks; "
                "runtime requires the C backend for all nodes"
                + (f" (fallbacks: {details})" if details else "")
            )
    meta = {
        "render_duration": None,  # caller can fill timing
        "produced_frames": output_frames,
        "sample_rate": output_sample_rate,
        "node_timings": getattr(graph, "last_node_timings", {}),
    }
    return output_node_buffer, meta