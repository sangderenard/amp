from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate buttery FFT noise gradients via the native KPN runtime.")
    parser.add_argument("--frames", type=int, default=32768, help="Number of frames to synthesise (default: 32768)")
    parser.add_argument("--window", type=int, default=1024, help="FFT window size (default: 1024)")
    parser.add_argument("--sample-rate", type=float, default=48000.0, help="Sample rate for the render (default: 48000)")
    parser.add_argument("--oversample", type=int, default=4, help="Oversample ratio recorded in descriptor metadata (default: 4)")
    parser.add_argument("--seed", type=int, default=49152, help="Random seed for the noise profile")
    parser.add_argument("--blob", type=Path, default=Path("build") / "fft_noise_params.bin", help="Parameter blob output")
    parser.add_argument("--output", type=Path, default=Path("output.wav"), help="Destination WAV file")
    parser.add_argument("--build-root", type=Path, default=Path("build"), help="Root directory that contains the native build outputs")
    args = parser.parse_args()

    params = generate_parameters(args.frames, args.window, seed=args.seed)
    write_blob(args.blob, params, args.frames, args.window, args.sample_rate, max(1, args.oversample))

    exe = locate_executable(args.build_root)
    cmd = [
        str(exe),
        "--params",
        str(args.blob),
        "--output",
        str(args.output),
    ]
    env = os.environ.copy()
    print(f"[fft_noise_gradient] invoking native renderer: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    print(f"[fft_noise_gradient] wrote WAV to {args.output}")


if __name__ == "__main__":
    main()
