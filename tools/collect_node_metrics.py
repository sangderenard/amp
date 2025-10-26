from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import importlib.util  # noqa: E402  (deferred import for repo path setup)

from amp.native_runtime import NativeGraphExecutor, UNAVAILABLE_REASON  # noqa: E402


def _load_demo_module():
    script_path = REPO_ROOT / "scripts" / "demo_kpn_kpn_native_correct.py"
    spec = importlib.util.spec_from_file_location("amp_demo_kpn", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_param_block(values: Dict[str, np.ndarray], total_frames: int) -> Dict[str, np.ndarray]:
    block: Dict[str, np.ndarray] = {}
    for key, value in values.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(total_frames, float(arr), dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != total_frames:
                raise ValueError(f"param '{key}' length {arr.shape[0]} != frames {total_frames}")
            arr = arr[np.newaxis, np.newaxis, :]
        elif arr.ndim == 2:
            if arr.shape[1] != total_frames:
                raise ValueError(f"param '{key}' shape {arr.shape} incompatible with frames {total_frames}")
            arr = arr[:, np.newaxis, :]
        elif arr.ndim == 3:
            if arr.shape[2] != total_frames:
                raise ValueError(f"param '{key}' shape {arr.shape} incompatible with frames {total_frames}")
        else:
            raise ValueError(f"param '{key}' must be 1D, 2D or 3D (got ndim={arr.ndim})")
        block[str(key)] = np.require(arr, requirements=("C",))
    return block


def _prepare_base_params(module, total_frames: int) -> Dict[str, Dict[str, np.ndarray]]:
    base_freq = 220.0
    driver_freq = np.full(total_frames, base_freq, dtype=np.float64)
    driver_amp = np.full(total_frames, 0.75, dtype=np.float64)
    render_mode = np.full(total_frames, 0.5, dtype=np.float64)
    pitch_curve = driver_freq.copy()
    pitch_slew = np.full(total_frames, 8.0, dtype=np.float64)
    osc_amp = np.full(total_frames, 0.8, dtype=np.float64)
    osc_slew = np.full(total_frames, 0.35, dtype=np.float64)
    ratio_curve = np.ones(total_frames, dtype=np.float64)

    fft_params = {
        "divisor": np.ones(total_frames, dtype=np.float64),
        "divisor_imag": np.zeros(total_frames, dtype=np.float64),
        "phase_offset": np.zeros(total_frames, dtype=np.float64),
        "lower_band": np.zeros(total_frames, dtype=np.float64),
        "upper_band": np.ones(total_frames, dtype=np.float64),
        "filter_intensity": np.ones(total_frames, dtype=np.float64),
        "stabilizer": np.full(total_frames, 1.0e-9, dtype=np.float64),
    }

    params: Dict[str, Dict[str, np.ndarray]] = {
        module.pitch.name: _create_param_block(
            {
                "pitch_hz": pitch_curve,
                "slew_hz_per_s": pitch_slew,
            },
            total_frames,
        ),
        module.driver.name: _create_param_block(
            {
                "frequency": driver_freq,
                "amplitude": driver_amp,
                "render_mode": render_mode,
            },
            total_frames,
        ),
        module.oscillator.name: _create_param_block(
            {
                "freq": driver_freq,
                "amp": osc_amp,
                "slew": osc_slew,
            },
            total_frames,
        ),
        "fft": _create_param_block(fft_params, total_frames),
    }

    pitch_shift = getattr(module, "pitch_shift", None)
    if pitch_shift is not None:
        params[pitch_shift.name] = _create_param_block({"ratio": ratio_curve}, total_frames)
    return params


def _param_to_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("parameter tensors must be 3D (B, C, F)")
    plane = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    image = np.zeros_like(plane, dtype=np.uint8)
    finite_mask = np.isfinite(plane)
    if np.any(finite_mask):
        finite_values = plane[finite_mask]
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        if not math.isclose(vmin, vmax):
            scaled = (plane - vmin) / (vmax - vmin)
            scaled = np.clip(scaled, 0.0, 1.0)
            image = (scaled * 255.0).astype(np.uint8)
            image[~finite_mask] = 0
    return image


def _spectrogram_image(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 1:
        signal = np.reshape(signal, -1)
    if signal.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    window = min(2048, signal.size)
    if window < 64:
        window = signal.size
    window = max(16, min(window, signal.size))
    if window <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    hop = max(1, window // 4)
    frame_count = 1 if signal.size <= window else int(math.ceil((signal.size - window) / hop)) + 1
    if frame_count <= 0:
        frame_count = 1
    window_func = np.hanning(window)
    freq_bins = window // 2 + 1
    spectrogram = np.zeros((freq_bins, frame_count), dtype=np.float64)
    for idx in range(frame_count):
        start = idx * hop
        if start >= signal.size:
            break
        end = start + window
        segment = np.zeros(window, dtype=np.float64)
        slice_end = min(end, signal.size)
        chunk = signal[start:slice_end]
        segment[: chunk.size] = chunk
        segment *= window_func
        spectrum = np.fft.rfft(segment, n=window)
        spectrogram[:, idx] = np.abs(spectrum)
    spectrogram = np.log10(spectrogram + 1.0e-12)
    finite = np.isfinite(spectrogram)
    if np.any(finite):
        vmin = float(np.min(spectrogram[finite]))
        vmax = float(np.max(spectrogram[finite]))
        if math.isclose(vmin, vmax):
            normalised = np.zeros_like(spectrogram)
        else:
            normalised = (spectrogram - vmin) / (vmax - vmin)
    else:
        normalised = np.zeros_like(spectrogram)
    normalised = np.clip(normalised, 0.0, 1.0)
    image = (normalised * 255.0).astype(np.uint8)
    return np.flipud(image)


def _process_node_dumps(dump_root: Path, writer) -> None:
    if not dump_root.exists():
        return
    meta_suffix = ".meta.json"
    for meta_path in sorted(dump_root.rglob(f"*{meta_suffix}")):
        if not meta_path.name.endswith(meta_suffix):
            continue
        base_name = meta_path.name[: -len(meta_suffix)]
        raw_path = meta_path.with_name(base_name + ".raw")
        if not raw_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        batches = int(metadata.get("batches", 0))
        channels = int(metadata.get("channels", 0))
        frames = int(metadata.get("frames", 0))
        expected = batches * channels * frames
        if expected <= 0:
            continue
        data = np.fromfile(raw_path, dtype=np.float32)
        if data.size != expected:
            continue
        wave = data.reshape((batches, channels, frames))
        npy_path = meta_path.with_name(base_name + ".npy")
        np.save(npy_path, wave)
        flat = wave.reshape((-1, frames))[0]
        spec_image = _spectrogram_image(flat)
        writer(meta_path.with_name(base_name + "_spectrogram.png"), spec_image)


def _save_param_artifacts(base_params: Dict[str, Dict[str, np.ndarray]], out_dir: Path, writer) -> None:
    params_dir = out_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for node_name, tensors in base_params.items():
        if node_name.startswith("_"):
            continue
        node_dir = params_dir / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        for param_name, tensor in tensors.items():
            np.save(node_dir / f"{param_name}.npy", tensor)
            try:
                image = _param_to_image(tensor)
            except ValueError:
                continue
            writer(node_dir / f"{param_name}.png", image)


def collect(
    *,
    duration: float = 5.0,
    sr: int = 48000,
    block_size: int = 256,
    out_dir: Path | str = Path("output") / "collect_metrics",
    dump_dir: Path | str | None = None,
) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dump_path = Path(dump_dir) if dump_dir is not None else out_path / "node_dumps"
    if dump_path.exists():
        shutil.rmtree(dump_path)
    dump_path.mkdir(parents=True, exist_ok=True)

    module = _load_demo_module()
    grayscale_writer = getattr(module, "_write_grayscale_png", None)
    if grayscale_writer is None:
        raise RuntimeError("demo module does not expose _write_grayscale_png")

    graph, demo_module = module.build_graph(int(sr))
    audio_rate = float(sr)
    total_frames = int(round(duration * audio_rate))
    dsp_rate = float(getattr(graph, "dsp_sample_rate", audio_rate) or audio_rate)
    ratio = dsp_rate / audio_rate if audio_rate > 0.0 else 1.0
    dsp_frames = int(math.ceil(total_frames * ratio)) if ratio > 0.0 else total_frames
    base_params = _prepare_base_params(demo_module, dsp_frames)

    os.environ["AMP_NODE_DUMP_DIR"] = str(dump_path)
    try:
        executor = NativeGraphExecutor(graph)
    except Exception as exc:  # pragma: no cover - interactive feedback
        os.environ.pop("AMP_NODE_DUMP_DIR", None)
        if UNAVAILABLE_REASON:
            print("Native runtime unavailable:", UNAVAILABLE_REASON)
        print("Native runtime failed to initialise:", exc)
        return 2

    control_history_blob: bytes = b""
    try:
        control_history_blob = graph.export_control_history_blob(0.0, float(duration))
    except Exception:
        control_history_blob = b""

    try:
        with executor:
            streamer = executor.create_streamer(
                total_frames=total_frames,
                sample_rate=float(sr),
                base_params=base_params,
                control_history_blob=control_history_blob,
                ring_frames=total_frames,
                block_frames=block_size,
            )
            with streamer:
                streamer.start()
                start_time = time.time()
                produced = 0
                while True:
                    produced, consumed, status = streamer.status()
                    if status != 0:
                        raise RuntimeError(f"streamer error status {status}")
                    if produced >= total_frames:
                        break
                    if time.time() - start_time >= duration + 0.5:
                        break
                    time.sleep(0.05)
                streamer.stop()
                _ = streamer.collect(total_frames)

            summaries: Dict[str, dict] = {}
            for node in graph.ordered_nodes:
                summary = executor.ffi.new("AmpGraphNodeSummary *")
                rc = executor.lib.amp_graph_runtime_describe_node(
                    executor._runtime, node.name.encode("utf-8"), summary
                )
                if int(rc) != 0:
                    continue
                metrics = None
                if summary.has_metrics:
                    m = summary.metrics
                    metrics = {
                        "measured_delay_frames": int(m.measured_delay_frames),
                        "accumulated_heat": float(m.accumulated_heat),
                        "processing_time_seconds": float(getattr(m, "processing_time_seconds", 0.0)),
                        "logging_time_seconds": float(getattr(m, "logging_time_seconds", 0.0)),
                        "total_time_seconds": float(getattr(m, "total_time_seconds", 0.0)),
                        "thread_cpu_time_seconds": float(getattr(m, "thread_cpu_time_seconds", 0.0)),
                    }
                summaries[node.name] = {
                    "declared_delay_frames": int(summary.declared_delay_frames),
                    "oversample_ratio": int(summary.oversample_ratio),
                    "supports_v2": bool(summary.supports_v2),
                    "has_metrics": bool(summary.has_metrics),
                    "metrics": metrics,
                    "total_heat_accumulated": float(summary.total_heat_accumulated),
                }

            summaries_path = out_path / "node_summaries.json"
            summaries_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print("Wrote node summaries to", summaries_path)
    finally:
        os.environ.pop("AMP_NODE_DUMP_DIR", None)

    _process_node_dumps(dump_path, grayscale_writer)
    _save_param_artifacts(base_params, out_path, grayscale_writer)
    print("Processed node dumps in", dump_path)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("output") / "collect_metrics")
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()
    exit_code = collect(
        duration=args.duration,
        sr=args.sr,
        block_size=args.block_size,
        out_dir=args.out_dir,
        dump_dir=args.dump_dir,
    )
    sys.exit(exit_code)
