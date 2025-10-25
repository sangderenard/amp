"""Native-only KPN demo: stream oscillator -> driver -> op-amp oscillator -> PCM sink with FFT tap."""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import sympy as sp

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "src"))

from amp.graph import AudioGraph
from amp.native_runtime import NativeGraphExecutor, UNAVAILABLE_REASON
from amp.nodes import (
    FFTDivisionNode,
    MixNode,
    OscNode,
    OscillatorPitchNode,
    ParametricDriverNode,
)


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


@dataclass(frozen=True)
class PitchProgram:
    oscillator_freq: np.ndarray
    driver_freq: np.ndarray
    driver_amp: np.ndarray
    normalized: np.ndarray
    render_blend: np.ndarray
    raw_expression: np.ndarray


@dataclass(frozen=True)
class PitchDriverOscModule:
    pitch: OscillatorPitchNode
    driver: ParametricDriverNode
    oscillator: OscNode

    @classmethod
    def install(
        cls,
        graph: AudioGraph,
        *,
        pitch_name: str = "pitch_programmer",
        driver_name: str = "driver",
        oscillator_name: str = "osc_master",
    ) -> "PitchDriverOscModule":
        pitch = OscillatorPitchNode(pitch_name, min_freq=0.0, default_slew=0.0)
        driver = ParametricDriverNode(driver_name, mode="piezo")
        osc = OscNode(
            oscillator_name,
            wave="saw",
            mode="op_amp",
            accept_reset=False,
            integration_leak=0.997,
            integration_gain=0.5,
            integration_clamp=1.2,
        )

        graph.add_node(pitch)
        graph.add_node(driver)
        graph.add_node(osc)

        graph.connect_mod(pitch.name, driver.name, "frequency", scale=1.0, mode="add")
        graph.connect_audio(driver.name, osc.name)

        return cls(pitch=pitch, driver=driver, oscillator=osc)


def build_graph(sample_rate: int) -> Tuple[AudioGraph, PitchDriverOscModule]:
    graph = AudioGraph(sample_rate=sample_rate, output_channels=1)

    module = PitchDriverOscModule.install(graph)
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

    graph.add_node(mix)
    graph.add_node(fft)

    graph.connect_audio(module.oscillator.name, "mix")
    graph.connect_audio("mix", "fft")
    graph.set_sink("mix")
    return graph, module


def _json_safe(value):
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return repr(value)


def export_network_map(graph: AudioGraph, path: Path) -> None:
    nodes = []
    for node in graph.ordered_nodes:
        params = dict(getattr(node, "params", {}))
        safe_params = {str(key): _json_safe(val) for key, val in params.items()}
        nodes.append(
            {
                "name": node.name,
                "type": type(node).__name__,
                "params": safe_params,
                "oversample_ratio": int(getattr(node, "oversample_ratio", 1) or 1),
                "declared_delay": int(getattr(node, "declared_delay_frames", 0) or 0),
                "supports_v2": bool(getattr(node, "supports_v2", True)),
            }
        )
    audio_edges = []
    for target, sources in graph._audio_inputs.items():  # type: ignore[attr-defined]
        for source in sources:
            audio_edges.append({"source": source, "target": target})
    mod_edges = []
    for target, entries in graph._mod_inputs.items():  # type: ignore[attr-defined]
        for connection in entries:
            mod_edges.append(
                {
                    "source": connection.source,
                    "target": target,
                    "param": connection.param,
                    "scale": connection.scale,
                    "mode": connection.mode,
                    "channel": connection.channel,
                }
            )
    payload = {
        "nodes": nodes,
        "audio_edges": audio_edges,
        "mod_edges": mod_edges,
        "sink": graph.sink,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _evaluate_pitch_expression(expr: str, t: np.ndarray) -> np.ndarray:
    symbol_t = sp.Symbol("t", real=True)
    try:
        parsed = sp.sympify(expr, locals={"pi": sp.pi})
    except sp.SympifyError as exc:
        raise ValueError(f"invalid SymPy expression '{expr}': {exc}") from exc
    extra_symbols = parsed.free_symbols.difference({symbol_t})
    if extra_symbols:
        names = ", ".join(sorted(str(sym) for sym in extra_symbols))
        raise ValueError(f"unsupported symbols in pitch expression: {names}")
    func = sp.lambdify((symbol_t,), parsed, modules=["numpy"])
    values = func(t)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(t.shape, float(arr), dtype=np.float64)
    else:
        arr = np.broadcast_to(arr, t.shape).astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("pitch expression produced non-finite values")
    return arr


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


def generate_pitch_schedule(
    total_frames: int,
    sample_rate: float,
    expression: str,
    *,
    base_freq: float,
    pitch_depth: float,
    driver_min_freq: float,
    log: Callable[[str], None],
) -> PitchProgram:
    t = np.arange(total_frames, dtype=np.float64) / sample_rate
    try:
        raw_curve = _evaluate_pitch_expression(expression, t)
    except Exception as exc:  # noqa: BLE001 - fall back to a stable modulation
        log(
            "[demo] Pitch expression evaluation failed: "
            f"{exc}. Falling back to neutral modulation."
        )
        raw_curve = np.zeros(total_frames, dtype=np.float64)
    centered = raw_curve - np.mean(raw_curve)
    span = float(np.max(np.abs(centered))) if centered.size else 0.0
    if not math.isfinite(span) or span < 1.0e-9:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / span
    pitch_offsets = normalized * float(pitch_depth)
    osc_frequency = np.asarray(base_freq + pitch_offsets, dtype=np.float64)
    osc_frequency = np.maximum(osc_frequency, driver_min_freq)
    driver_frequency = np.maximum(osc_frequency, driver_min_freq)
    driver_amplitude = 0.65 + 0.35 * np.tanh(normalized)
    render_mode = np.clip(0.5 + 0.5 * normalized, 0.0, 1.0)
    return PitchProgram(
        oscillator_freq=osc_frequency,
        driver_freq=driver_frequency,
        driver_amp=driver_amplitude,
        normalized=normalized,
        render_blend=render_mode,
        raw_expression=raw_curve,
    )


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
    parser.add_argument(
        "--pitch-modulation",
        type=str,
        default="2.0",
        help=(
            "SymPy expression of time 't' describing the oscillator pitch program prior to driver handoff "
            "(default: '2.0')."
        ),
    )
    parser.add_argument(
        "--pitch-depth",
        type=float,
        default=40.0,
        help="Depth in Hz applied to the evaluated expression before delivering pitch to the driver (default: 40.0).",
    )
    parser.add_argument(
        "--pitch-direct-depth",
        type=float,
        default=0.0,
        help=(
            "Additional depth in Hz applied directly inside the oscillator after the driver-rendered waveform is received "
            "(default: 0.0)."
        ),
    )
    parser.add_argument(
        "--base-freq",
        type=float,
        default=330.0,
        help="Base oscillator frequency in Hz before modulation is applied (default: 330.0).",
    )
    parser.add_argument(
        "--driver-min-freq",
        type=float,
        default=0.1,
        help="Lower clamp in Hz applied to the evaluated driver frequency curve (default: 0.1).",
    )
    parser.add_argument(
        "--pitch-slew",
        type=float,
        default=0.0,
        help="Slew limit in Hz/s applied by the pitch programmer before values reach the driver (default: 0.0).",
    )
    parser.add_argument(
        "--op-amp-slew",
        type=float,
        default=12000.0,
        help="Slew rate in Hz/s applied by the op-amp oscillator when chasing the driver signal (default: 12000.0).",
    )
    parser.add_argument(
        "--oscillator-pitch-mode",
        choices=("follow", "hold"),
        default="follow",
        help=(
            "Whether the oscillator should follow the programmed pitch curve or hold a static frequency "
            "derived from --base-freq (default: follow)."
        ),
    )
    parser.add_argument(
        "--driver-pitch-mode",
        choices=("follow", "hold"),
        default="follow",
        help=(
            "Whether the driver should follow the pitch program or hold its initial frequency "
            "(default: follow)."
        ),
    )
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
    graph, module = build_graph(int(args.sr))
    pitch_node = module.pitch
    pitch_node.params["default_slew"] = max(0.0, float(args.pitch_slew))
    pitch_node.params["min_freq"] = max(0.0, float(args.driver_min_freq))
    map_path = out_dir / "network_map.json"
    export_network_map(graph, map_path)
    log(f"[demo] Exported network map: {map_path}")

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
        pitch_program = generate_pitch_schedule(
            total_frames,
            args.sr,
            args.pitch_modulation,
            base_freq=float(args.base_freq),
            pitch_depth=float(args.pitch_depth),
            driver_min_freq=max(1.0e-6, float(args.driver_min_freq)),
            log=log,
        )

        pitch_schedule = pitch_program.oscillator_freq
        driver_freq_curve = pitch_program.driver_freq
        driver_amp_curve = pitch_program.driver_amp
        normalized_pitch = pitch_program.normalized
        render_mode_curve = pitch_program.render_blend

        base_amp = 0.4
        if args.pitch_direct_depth != 0.0:
            master_freq_curve = pitch_schedule + normalized_pitch * float(args.pitch_direct_depth)
        else:
            master_freq_curve = pitch_schedule.copy()
        if args.driver_pitch_mode == "hold":
            initial_driver = float(driver_freq_curve[0]) if driver_freq_curve.size else float(args.base_freq)
            driver_freq_curve = np.full(total_frames, initial_driver, dtype=np.float64)
        if args.oscillator_pitch_mode == "hold":
            hold_freq = float(master_freq_curve[0]) if master_freq_curve.size else float(args.base_freq)
            master_freq_curve = np.full(total_frames, hold_freq, dtype=np.float64)
        slew_curve = np.full(total_frames, max(0.0, float(args.op_amp_slew)), dtype=np.float64)
        pitch_slew_curve = np.full(total_frames, max(0.0, float(args.pitch_slew)), dtype=np.float64)
        log(
            "[demo] Pitch program stats: pitch[min={:.4f}, max={:.4f}] Hz, raw[min={:.4f}, max={:.4f}], "
            "norm[min={:.4f}, max={:.4f}]".format(
                float(pitch_schedule.min()),
                float(pitch_schedule.max()),
                float(pitch_program.raw_expression.min()),
                float(pitch_program.raw_expression.max()),
                float(normalized_pitch.min()),
                float(normalized_pitch.max()),
            )
        )
        if driver_freq_curve.size:
            log(
                "[demo] Driver stats: freq[min={:.4f}, max={:.4f}] Hz, amp[min={:.4f}, max={:.4f}], "
                "blend[min={:.4f}, max={:.4f}] mode={}".format(
                    float(driver_freq_curve.min()),
                    float(driver_freq_curve.max()),
                    float(driver_amp_curve.min()),
                    float(driver_amp_curve.max()),
                    float(render_mode_curve.min()),
                    float(render_mode_curve.max()),
                    args.driver_pitch_mode,
                )
            )
        if master_freq_curve.size:
            log(
                "[demo] Oscillator stats: freq[min={:.4f}, max={:.4f}] Hz mode={}".format(
                    float(master_freq_curve.min()),
                    float(master_freq_curve.max()),
                    args.oscillator_pitch_mode,
                )
            )

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
                    "render_mode": render_mode_curve[sl],
                }
            )
            pitch_params = create_param_block(
                {
                    "pitch_hz": pitch_schedule[sl],
                    "slew_hz_per_s": pitch_slew_curve[sl],
                }
            )
            osc_params = create_param_block(
                {
                    "freq": master_freq_curve[sl],
                    "amp": np.full(frames_this, base_amp, dtype=np.float64),
                    "slew": slew_curve[sl],
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
                module.pitch.name: pitch_params,
                module.driver.name: driver_params,
                module.oscillator.name: osc_params,
                "fft": fft_params,
            }
            try:
                block_pcm = executor.run_block(frames_this, float(args.sr), base_params=base_params)
            except Exception as exc:
                log(f"[demo] Native execution failed at block {block_index}: {exc}")
                err = executor.last_error()
                if err:
                    stage = err.get("stage") or "<unknown>"
                    node_name = err.get("node") or "<none>"
                    detail = err.get("detail") or ""
                    log(
                        "[demo] Last native error: "
                        f"code={err.get('code')} stage={stage} node={node_name} detail={detail}"
                    )
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
