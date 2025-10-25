"""Generate buttery FFT noise gradients via the native KPN runtime."""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import wave
from pathlib import Path
from typing import Iterable

import numpy as np


HEADER_STRUCT = struct.Struct("<IIdI")
PARAM_ORDER = (
    "audio",
    "divisor",
    "divisor_imag",
    "phase",
    "lower",
    "upper",
    "filter",
    "stabilizer",
)


def load_fft_interface():
    """Return callable that provides (ffi, lib) bound to amp_run_node_v2."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_fft_spectral_node",
        Path(__file__).resolve().parents[1] / "tests" / "test_fft_spectral_node.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_fft_interface  # type: ignore[attr-defined]


def generate_parameters(frames: int, window: int, seed: int = 49152) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float64)

    raw = rng.normal(0.0, 1.0, frames)
    audio = np.empty_like(raw)
    prev = 0.0
    alpha = 0.02
    for i, sample in enumerate(raw):
        prev = (1.0 - alpha) * prev + alpha * sample
        audio[i] = prev

    lower = np.clip(0.05 + 0.4 * t, 0.0, 1.0)
    span = 0.15 + 0.25 * (1.0 - np.cos(2.0 * np.pi * t))
    upper = np.clip(lower + span, 0.05, 0.99)
    filter_intensity = np.clip(0.35 + 0.45 * np.sin(np.pi * t), 0.05, 0.95)
    phase = 0.5 * np.pi + 0.5 * np.pi * np.sin(2.0 * np.pi * t)
    divisor = 0.85 + 0.15 * np.cos(4.0 * np.pi * t)
    divisor_imag = np.zeros_like(divisor)
    stabilizer = np.full_like(divisor, 1.0e-9)

    return {
        "audio": audio,
        "divisor": divisor,
        "divisor_imag": divisor_imag,
        "phase": phase,
        "lower": lower,
        "upper": upper,
        "filter": filter_intensity,
        "stabilizer": stabilizer,
    }


def write_blob(path: Path, params: dict[str, np.ndarray], frames: int, window: int, sr: float, oversample: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(HEADER_STRUCT.pack(frames, window, float(sr), oversample))
        for key in PARAM_ORDER:
            array = np.asarray(params[key], dtype=np.float64)
            if array.size != frames:
                raise ValueError(f"{key}: expected {frames} samples, got {array.size}")
            fh.write(array.tobytes(order="C"))


def locate_executable(build_root: Path) -> Path:
    candidates = [
        build_root / "native" / "Release" / "test_fft_noise_gradient.exe",
        build_root / "native" / "Debug" / "test_fft_noise_gradient.exe",
        build_root / "native" / "test_fft_noise_gradient",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Unable to locate test_fft_noise_gradient executable; build the native targets first.")


def _ensure_param(name: str, params: dict[str, np.ndarray], frames: int) -> np.ndarray:
    array = np.asarray(params[name], dtype=np.float64)
    if array.size != frames:
        raise ValueError(f"{name}: expected {frames} samples, received {array.size}")
    return np.ascontiguousarray(array, dtype=np.float64)


def render_with_native(
    params: dict[str, np.ndarray],
    frames: int,
    window: int,
    sample_rate: float,
    oversample: int,
    algorithm: str | None,
) -> np.ndarray:
    loader = load_fft_interface()
    ffi, lib = loader()

    descriptor = ffi.new("EdgeRunnerNodeDescriptor *")
    descriptor.name = ffi.new("char[]", b"fft_noise_gradient")
    descriptor.name_len = len(b"fft_noise_gradient")
    descriptor.type_name = ffi.new("char[]", b"FFTDivisionNode")
    descriptor.type_len = len(b"FFTDivisionNode")

    params_json = (
        "{"
        + f"\"window_size\":{window},"
        + "\"stabilizer\":1e-9,"
        + "\"epsilon\":1e-9,"
        + f"\"declared_delay\":{window - 1 if window > 0 else 0},"
        + f"\"oversample_ratio\":{max(1, oversample)},"
        + "\"supports_v2\":true"
    )
    if algorithm:
        params_json += f",\"algorithm\":\"{algorithm}\""
    params_json += "}"
    descriptor.params_json = ffi.new("char[]", params_json.encode("utf-8"))
    descriptor.params_len = len(params_json)

    batches = 1
    channels = 1

    audio = _ensure_param("audio", params, frames)
    audio_ptr = ffi.from_buffer("double[]", audio)
    audio_view = ffi.new("EdgeRunnerAudioView *")
    audio_view.has_audio = 1
    audio_view.batches = batches
    audio_view.channels = channels
    audio_view.frames = frames
    audio_view.data = audio_ptr

    param_map = {
        "divisor": "divisor",
        "divisor_imag": "divisor_imag",
        "phase": "phase_offset",
        "lower": "lower_band",
        "upper": "upper_band",
        "filter": "filter_intensity",
        "stabilizer": "stabilizer",
    }

    keepalive: list[Iterable[object]] = [audio, audio_ptr]
    param_views = ffi.new("EdgeRunnerParamView[]", len(param_map))
    for idx, (key, runtime_name) in enumerate(param_map.items()):
        arr = _ensure_param(key, params, frames)
        buf = ffi.from_buffer("double[]", arr)
        name_buf = ffi.new("char[]", runtime_name.encode("utf-8"))
        view = param_views[idx]
        view.name = name_buf
        view.batches = batches
        view.channels = channels
        view.frames = frames
        view.data = buf
        keepalive.extend((arr, buf, name_buf))

    param_set = ffi.new("EdgeRunnerParamSet *")
    param_set.count = len(param_map)
    param_set.items = param_views

    inputs = ffi.new("EdgeRunnerNodeInputs *")
    inputs.audio = audio_view[0]
    inputs.params = param_set[0]

    out_buffer = ffi.new("double **")
    out_channels = ffi.new("int *")
    state_ptr = ffi.new("void **")
    metrics = ffi.new("AmpNodeMetrics *")

    try:
        rc = lib.amp_run_node_v2(
            descriptor,
            inputs,
            batches,
            channels,
            frames,
            float(sample_rate),
            out_buffer,
            out_channels,
            state_ptr,
            ffi.NULL,
            ffi.cast("AmpExecutionMode", 1),  # backward mode for gradient synthesis
            metrics,
        )
        if int(rc) != 0 or out_buffer[0] == ffi.NULL:
            raise RuntimeError(f"amp_run_node_v2 failed with rc={int(rc)}")
        total = int(out_channels[0]) * frames
        pcm = np.frombuffer(
            ffi.buffer(out_buffer[0], total * np.dtype(np.float64).itemsize),
            dtype=np.float64,
        ).copy()
    finally:
        if out_buffer[0] != ffi.NULL:
            lib.amp_free(out_buffer[0])
        if state_ptr[0] != ffi.NULL:
            lib.amp_release_state(state_ptr[0])

    return pcm


def write_wav(path: Path, samples: np.ndarray, sample_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.0:
        normalised = samples / peak * 0.98
    else:
        normalised = samples
    pcm16 = np.clip(np.rint(normalised * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate buttery FFT noise gradients via the native KPN runtime.")
    parser.add_argument("--frames", type=int, default=32768, help="Number of frames to synthesise (default: 32768)")
    parser.add_argument("--window", type=int, default=1024, help="FFT window size (default: 1024)")
    parser.add_argument("--sample-rate", type=float, default=48000.0, help="Sample rate for the render (default: 48000)")
    parser.add_argument("--oversample", type=int, default=4, help="Oversample ratio recorded in descriptor metadata (default: 4)")
    parser.add_argument("--seed", type=int, default=49152, help="Random seed for the noise profile")
    parser.add_argument("--blob", type=Path, default=Path("build") / "fft_noise_params.bin", help="Parameter blob output")
    parser.add_argument("--output", type=Path, default=Path("output.wav"), help="Destination WAV file")
    parser.add_argument("--native-output", type=Path, help="Optional WAV path for the in-process native render")
    parser.add_argument("--build-root", type=Path, default=Path("build"), help="Root directory that contains the native build outputs")
    parser.add_argument(
        "--engine",
        choices=("exe", "native", "both"),
        default="exe",
        help="Select rendering backend: compiled test executable, in-process native runtime, or both (default: exe)",
    )
    parser.add_argument(
        "--algorithm",
        choices=("radix2", "dft", "nufft", "czt", "dynamic_oscillators"),
        help="Override the FFTDivisionNode algorithm used by the native renderer",
    )
    args = parser.parse_args()

    params = generate_parameters(args.frames, args.window, seed=args.seed)
    write_blob(args.blob, params, args.frames, args.window, args.sample_rate, max(1, args.oversample))

    if args.engine in {"native", "both"}:
        try:
            native_samples = render_with_native(
                params,
                args.frames,
                args.window,
                args.sample_rate,
                max(1, args.oversample),
                args.algorithm,
            )
        except Exception as exc:  # pragma: no cover - propagation handled by caller
            raise RuntimeError("Native in-process FFT render failed") from exc

        target = args.native_output
        if target is None:
            if args.engine == "both":
                target = args.output.with_name(f"{args.output.stem}_native{args.output.suffix}")
            else:
                target = args.output
        write_wav(target, native_samples, args.sample_rate)
        print(f"[fft_noise_gradient] wrote native runtime WAV to {target}")

    if args.engine in {"exe", "both"}:
        exe = locate_executable(args.build_root)
        cmd = [
            str(exe),
            "--params",
            str(args.blob),
            "--output",
            str(args.output),
        ]
        if args.algorithm:
            cmd.extend(["--algorithm", args.algorithm])
        env = os.environ.copy()
        print(f"[fft_noise_gradient] invoking native renderer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        print(f"[fft_noise_gradient] wrote WAV to {args.output}")


if __name__ == "__main__":
    main()
