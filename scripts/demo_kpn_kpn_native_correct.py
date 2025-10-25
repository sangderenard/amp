"""Native-only KPN demo: driver -> oscillator -> PCM sink with FFT tap."""
from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from amp.graph import AudioGraph
from amp.native_runtime import NativeGraphExecutor, UNAVAILABLE_REASON
from amp.nodes import FFTDivisionNode, MixNode, OscNode, ParametricDriverNode


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    import struct
    import zlib

    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    height, width = image.shape
    header = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    idat = zlib.compress(scanlines, level=6)

    with path.open("wb") as stream:
        stream.write(header)
        stream.write(chunk(b"IHDR", ihdr))
        stream.write(chunk(b"IDAT", idat))
        stream.write(chunk(b"IEND", b""))


def build_graph(sample_rate: int) -> AudioGraph:
    graph = AudioGraph(sample_rate=sample_rate, output_channels=1)

    driver = ParametricDriverNode("driver", mode="piezo")
    osc = OscNode(
        "osc",
        wave="saw",
        mode="integrator",
        accept_reset=False,
        integration_leak=0.997,
        integration_gain=0.5,
        integration_clamp=1.2,
    )
    mix = MixNode("mix", params={"channels": 1})
    fft = FFTDivisionNode(
        "fft",
        params={
            "window_size": 512,
            "oversample_ratio": 1,
            "declared_delay": 511,
            "supports_v2": True,
            "enable_remainder": True,
            "algorithm": "radix2",
        },
    )

    graph.add_node(driver)
    graph.add_node(osc)
    graph.add_node(mix)
    graph.add_node(fft)

    graph.connect_mod("driver", "osc", "freq", scale=40.0, mode="add")
    graph.connect_audio("osc", "mix")
    graph.connect_audio("mix", "fft")
    graph.set_sink("mix")
    return graph


def ensure_native_kernels(executor: NativeGraphExecutor, node_names: Iterable[str]) -> None:
    ffi, lib = executor.ffi, executor.lib
    for name in node_names:
        summary = ffi.new("AmpGraphNodeSummary *")
        rc = lib.amp_graph_runtime_describe_node(executor._runtime, name.encode("utf-8"), summary)
        if int(rc) != 0:
            raise RuntimeError(f"native runtime cannot describe node '{name}' (rc={int(rc)})")
        if not summary.supports_v2:
            raise RuntimeError(f"node '{name}' does not have a native ABI implementation (supports_v2=0)")


def create_param_block(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    block: Dict[str, np.ndarray] = {}
    for key, array in values.items():
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, np.newaxis, :]
        elif arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        block[key] = np.require(arr, requirements=("C",))
    return block


def generate_driver_curves(total_frames: int, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(total_frames, dtype=np.float64) / sample_rate
    frequency = np.full(total_frames, 2.0, dtype=np.float64)  # 2 Hz modulation
    amplitude = 0.65 + 0.35 * np.sin(2.0 * math.pi * 0.25 * t)
    return frequency, amplitude


def compute_spectrogram(
    pcm: np.ndarray,
    sample_rate: float,
    window_size: int,
    hop: int,
) -> np.ndarray:
    if pcm.ndim != 1:
        raise ValueError("pcm must be one-dimensional")
    if pcm.size < window_size:
        pad = np.zeros(window_size - pcm.size, dtype=np.float64)
        pcm = np.concatenate([pcm, pad])
    window = np.hanning(window_size)
    segment_count = 1 + (pcm.size - window_size) // hop
    spectra = np.empty((segment_count, window_size // 2 + 1), dtype=np.float64)
    for idx in range(segment_count):
        start = idx * hop
        segment = pcm[start : start + window_size]
        tapered = segment * window
        fft = np.fft.rfft(tapered)
        magnitude = np.abs(fft)
        spectra[idx] = magnitude

    with np.errstate(divide="ignore"):
        log_spectra = 20.0 * np.log10(np.maximum(spectra, 1.0e-12))
    log_spectra -= log_spectra.max()
    min_val = float(log_spectra.min())
    if math.isclose(min_val, 0.0, abs_tol=1.0e-12):
        min_val = -1.0
    scaled = np.clip(log_spectra / min_val, 0.0, 1.0)
    image = np.flipud((1.0 - scaled).T)
    return (image * 255.0).astype(np.uint8)


def write_wav(path: Path, pcm: np.ndarray, sample_rate: float) -> None:
    pcm = np.asarray(pcm, dtype=np.float64)
    peak = np.max(np.abs(pcm)) if pcm.size else 0.0
    scaled = pcm / peak * 0.98 if peak > 0 else pcm
    pcm16 = np.clip(np.rint(scaled * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0, help="Render duration in seconds (default: 2.0)")
    parser.add_argument("--sr", type=float, default=48_000.0, help="Sample rate in Hz (default: 48000)")
    parser.add_argument("--block-size", type=int, default=512, help="Block size in frames (default: 512)")
    parser.add_argument("--out-dir", type=Path, default=Path("output") / "demo_kpn_spectro", help="Output directory")
    parser.add_argument("--play", action="store_true", help="Attempt realtime playback (not implemented)")
    parser.add_argument("--display", action="store_true", help="Display spectrogram window (not implemented)")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if args.play:
        print("[demo] --play requested but audio playback is not implemented in this demo.")
    if args.display:
        print("[demo] --display requested but GUI display is not implemented in this demo.")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "demo.log"
    log_path.write_text("")

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    log("[demo] Constructing audio graph...")
    graph = build_graph(int(args.sr))

    try:
        log("[demo] Initialising native graph runtime...")
        executor = NativeGraphExecutor(graph)
    except Exception as exc:
        if UNAVAILABLE_REASON:
            print(f"[demo] Native runtime unavailable: {UNAVAILABLE_REASON}", file=sys.stderr)
        print(f"[demo] Failed to create native runtime: {exc}", file=sys.stderr)
        print(
            "[demo] Build instructions:\n"
            "  cmake -S . -B build -G \"Visual Studio 17 2022\" -A x64\n"
            "  cmake --build build --config Release",
            file=sys.stderr,
        )
        return 2

    with executor:
        log("[demo] Verifying native node coverage...")
        ensure_native_kernels(executor, [node.name for node in graph.ordered_nodes])

        total_frames = int(round(args.duration * args.sr))
        if total_frames <= 0:
            raise ValueError("duration must produce at least one sample")
        block_size = int(args.block_size)
        if block_size <= 0:
            raise ValueError("block-size must be positive")

        log(f"[demo] Rendering {total_frames} frames (block size {block_size})...")
        driver_freq_curve, driver_amp_curve = generate_driver_curves(total_frames, args.sr)

        base_freq = 330.0
        base_amp = 0.4

        pcm_blocks: list[np.ndarray] = []
        metrics_log: list[Tuple[int, float]] = []

        produced = 0
        block_index = 0
        while produced < total_frames:
            frames_this = min(block_size, total_frames - produced)
            sl = slice(produced, produced + frames_this)
            log(f"[demo] Block {block_index}: rendering {frames_this} frames (produced={produced})")
            driver_params = create_param_block(
                {
                    "frequency": driver_freq_curve[sl],
                    "amplitude": driver_amp_curve[sl],
                }
            )
            osc_params = create_param_block(
                {
                    "freq": np.full(frames_this, base_freq, dtype=np.float64),
                    "amp": np.full(frames_this, base_amp, dtype=np.float64),
                }
            )
            fft_params = create_param_block(
                {
                    "divisor": np.ones(frames_this, dtype=np.float64),
                    "divisor_imag": np.zeros(frames_this, dtype=np.float64),
                    "phase_offset": np.zeros(frames_this, dtype=np.float64),
                    "lower_band": np.zeros(frames_this, dtype=np.float64),
                    "upper_band": np.ones(frames_this, dtype=np.float64),
                    "filter_intensity": np.ones(frames_this, dtype=np.float64),
                    "stabilizer": np.full(frames_this, 1.0e-9, dtype=np.float64),
                }
            )

            base_params = {
                "driver": driver_params,
                "osc": osc_params,
                "fft": fft_params,
            }
            try:
                block_pcm = executor.run_block(frames_this, float(args.sr), base_params=base_params)
            except Exception as exc:
                log(f"[demo] Native execution failed at block {block_index}: {exc}")
                raise
            log(f"[demo] Block {block_index}: completed")
            pcm_blocks.append(block_pcm.reshape(-1))

            summary = executor.ffi.new("AmpGraphNodeSummary *")
            rc = executor.lib.amp_graph_runtime_describe_node(executor._runtime, b"fft", summary)
            if int(rc) == 0 and summary.has_metrics:
                metrics_log.append((block_index, float(summary.metrics.accumulated_heat)))

            produced += frames_this
            block_index += 1

    pcm = np.concatenate(pcm_blocks)

    log("[demo] Rendering spectrogram...")
    window_size = int(graph._nodes["fft"].params.get("window_size", 512))
    hop = max(1, window_size // 4)
    image = compute_spectrogram(pcm, args.sr, window_size, hop)
    png_path = out_dir / "spectrogram.png"
    _write_grayscale_png(png_path, image)

    log("[demo] Writing PCM output...")
    wav_path = out_dir / "output.wav"
    write_wav(wav_path, pcm, args.sr)

    log(f"[demo] Wrote spectrogram: {png_path}")
    log(f"[demo] Wrote PCM: {wav_path}")
    if metrics_log:
        avg_heat = sum(m[1] for m in metrics_log) / len(metrics_log)
        log(f"[demo] FFT node average accumulated heat per block: {avg_heat:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
