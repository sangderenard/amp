"""Native-only KPN demo: stream oscillator -> driver -> op-amp oscillator -> PCM sink with FFT tap."""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

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
    PitchShiftNode,
    ContinuousTimebaseNode,
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
    pitch_shift: PitchShiftNode
    timebase: ContinuousTimebaseNode | None

    @classmethod
    def install(
        cls,
        graph: AudioGraph,
        *,
        pitch_name: str = "pitch_programmer",
        driver_name: str = "driver",
        oscillator_name: str = "osc_master",
        timebase_name: str = "timebase_bridge",
        pitch_shift_name: str = "pitch_shift_bridge",
        enable_timebase: bool = False,
        timebase_params: Mapping[str, object] | None = None,
        pitch_shift_params: Mapping[str, object] | None = None,
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

        pitch_shift_kwargs: Dict[str, object] = {}
        if pitch_shift_params is not None:
            if "window_size" in pitch_shift_params:
                pitch_shift_kwargs["window_size"] = int(pitch_shift_params["window_size"])
            if "hop_size" in pitch_shift_params:
                pitch_shift_kwargs["hop_size"] = int(pitch_shift_params["hop_size"])
            if "resynthesis_hop" in pitch_shift_params:
                pitch_shift_kwargs["resynthesis_hop"] = int(pitch_shift_params["resynthesis_hop"])

        pitch_shift = PitchShiftNode(pitch_shift_name, **pitch_shift_kwargs)
        graph.add_node(pitch_shift)

        if pitch_shift_params is not None:
            for key, value in pitch_shift_params.items():
                if key in {"window_size", "hop_size", "resynthesis_hop"}:
                    continue
                pitch_shift.params[str(key)] = value

        timebase: ContinuousTimebaseNode | None = None
        if enable_timebase:
            timebase = ContinuousTimebaseNode(timebase_name, params=timebase_params)
            graph.add_node(timebase)
            graph.connect_audio(driver.name, pitch_shift.name)
            graph.connect_audio(pitch_shift.name, timebase.name)
            graph.connect_audio(timebase.name, osc.name)
            graph.connect_mod(pitch.name, timebase.name, "pitch_in", scale=1.0, mode="add")
        else:
            graph.connect_audio(driver.name, pitch_shift.name)
            graph.connect_audio(pitch_shift.name, osc.name)

        graph.connect_mod(pitch.name, driver.name, "frequency", scale=1.0, mode="add")

        return cls(pitch=pitch, driver=driver, oscillator=osc, pitch_shift=pitch_shift, timebase=timebase)


def build_graph(
    sample_rate: int,
    *,
    enable_timebase: bool = False,
    timebase_params: Mapping[str, object] | None = None,
    pitch_shift_params: Mapping[str, object] | None = None,
) -> Tuple[AudioGraph, PitchDriverOscModule]:
    graph = AudioGraph(sample_rate=sample_rate, output_channels=1)

    module = PitchDriverOscModule.install(
        graph,
        enable_timebase=enable_timebase,
        timebase_params=timebase_params,
        pitch_shift_params=pitch_shift_params,
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


def ensure_native_kernels(
    executor: NativeGraphExecutor,
    node_names: Iterable[str],
    *,
    allow_missing: Iterable[str] = (),
) -> None:
    ffi, lib = executor.ffi, executor.lib
    allowed = {str(name) for name in allow_missing}
    for name in node_names:
        summary = ffi.new("AmpGraphNodeSummary *")
        rc = lib.amp_graph_runtime_describe_node(executor._runtime, name.encode("utf-8"), summary)
        if int(rc) != 0:
            if name in allowed:
                continue
            raise RuntimeError(f"native runtime cannot describe node '{name}' (rc={int(rc)})")
        if not summary.supports_v2:
            if name in allowed:
                continue
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


def _apply_hold_mask(curve: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if curve.ndim != 1:
        raise ValueError("curve must be one-dimensional for hold application")
    if mask.shape != curve.shape:
        raise ValueError("mask shape must match curve shape")
    if not np.any(mask):
        return curve
    held = curve.copy()
    current = float(curve[0]) if curve.size else 0.0
    for idx in range(curve.size):
        if mask[idx]:
            held[idx] = current
        else:
            current = float(curve[idx])
    return held


def _parse_semitone_list(text: str, *, fallback: Iterable[int]) -> List[int]:
    entries = [part.strip() for part in str(text).split(",")]
    values: List[int] = []
    for entry in entries:
        if not entry:
            continue
        try:
            values.append(int(entry))
        except ValueError as exc:
            raise ValueError(f"invalid semitone offset '{entry}' in '{text}'") from exc
    if not values:
        values = [int(v) for v in fallback]
    return values


def _generate_expression_schedule(
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


def _generate_arpeggio_schedule(
    total_frames: int,
    sample_rate: float,
    *,
    base_freq: float,
    driver_min_freq: float,
    arpeggio_intervals: Iterable[int],
    chord_intervals: Iterable[int],
    whole_note_seconds: float,
    chord_hold_notes: int,
    log: Callable[[str], None],
) -> PitchProgram:
    intervals = list(arpeggio_intervals)
    chord = list(chord_intervals)
    if not intervals:
        intervals = [0]
    if not chord:
        chord = [intervals[0]]
    frames_per_note = max(1, int(round(max(whole_note_seconds, 1.0e-3) * sample_rate)))
    chord_hold_notes = max(1, int(chord_hold_notes))
    arpeggio_frames = frames_per_note * len(intervals)
    chord_frames = frames_per_note * chord_hold_notes
    cycle_frames = arpeggio_frames + chord_frames
    raw_offsets = np.zeros(total_frames, dtype=np.float64)
    driver_offsets = np.zeros(total_frames, dtype=np.float64)
    render_blend = np.zeros(total_frames, dtype=np.float64)
    driver_amp = np.zeros(total_frames, dtype=np.float64)

    for frame in range(total_frames):
        pos = frame % cycle_frames
        if pos < arpeggio_frames:
            note_idx = pos // frames_per_note
            offset = intervals[note_idx]
            raw_offsets[frame] = float(offset)
            driver_offsets[frame] = float(intervals[0])
            render_blend[frame] = 0.2
            driver_amp[frame] = 0.55 + 0.35 * (note_idx / max(1, len(intervals) - 1))
        else:
            local = pos - arpeggio_frames
            chord_note_idx = (local // frames_per_note) % len(chord)
            offset = chord[chord_note_idx]
            raw_offsets[frame] = float(offset)
            driver_offsets[frame] = float(chord[0])
            render_blend[frame] = 0.85
            driver_amp[frame] = 0.75

    root_multiplier = np.power(2.0, raw_offsets / 12.0)
    osc_frequency = np.asarray(base_freq * root_multiplier, dtype=np.float64)
    driver_multiplier = np.power(2.0, driver_offsets / 12.0)
    driver_frequency = np.asarray(base_freq * driver_multiplier, dtype=np.float64)
    np.maximum(osc_frequency, driver_min_freq, out=osc_frequency)
    np.maximum(driver_frequency, driver_min_freq, out=driver_frequency)

    centered = raw_offsets - np.mean(raw_offsets)
    span = float(np.max(np.abs(centered))) if centered.size else 0.0
    if not math.isfinite(span) or span < 1.0e-9:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / span

    driver_amp = np.clip(driver_amp, 0.1, 1.1)
    render_blend = np.clip(render_blend, 0.0, 1.0)

    log(
        "[demo] Arpeggio program: intervals=%s chord=%s whole_note=%.3fs chord_hold=%d notes"
        % (intervals, chord, whole_note_seconds, chord_hold_notes)
    )

    return PitchProgram(
        oscillator_freq=osc_frequency,
        driver_freq=driver_frequency,
        driver_amp=driver_amp,
        normalized=normalized,
        render_blend=render_blend,
        raw_expression=raw_offsets,
    )


def generate_pitch_schedule(
    total_frames: int,
    sample_rate: float,
    expression: str,
    *,
    base_freq: float,
    pitch_depth: float,
    driver_min_freq: float,
    program: str,
    arpeggio_intervals: Iterable[int],
    chord_intervals: Iterable[int],
    whole_note_seconds: float,
    chord_hold_notes: int,
    log: Callable[[str], None],
) -> PitchProgram:
    mode = program.lower()
    if mode == "expression":
        return _generate_expression_schedule(
            total_frames,
            sample_rate,
            expression,
            base_freq=base_freq,
            pitch_depth=pitch_depth,
            driver_min_freq=driver_min_freq,
            log=log,
        )
    if mode == "arpeggio":
        return _generate_arpeggio_schedule(
            total_frames,
            sample_rate,
            base_freq=base_freq,
            driver_min_freq=driver_min_freq,
            arpeggio_intervals=arpeggio_intervals,
            chord_intervals=chord_intervals,
            whole_note_seconds=whole_note_seconds,
            chord_hold_notes=chord_hold_notes,
            log=log,
        )
    raise ValueError(f"unsupported pitch program '{program}'")


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
        "--pitch-program",
        choices=("arpeggio", "expression"),
        default="arpeggio",
        help="Select the high-level pitch program to render (default: arpeggio).",
    )
    parser.add_argument(
        "--pitch-modulation",
        type=str,
        default="2.0",
        help=(
            "SymPy expression of time 't' describing the oscillator pitch program prior to driver handoff. "
            "Only used when --pitch-program=expression (default: '2.0')."
        ),
    )
    parser.add_argument(
        "--pitch-depth",
        type=float,
        default=40.0,
        help="Depth in Hz applied to the evaluated expression before delivering pitch to the driver (expression mode only).",
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
        default=75.0,
        help="Slew limit in Hz/s applied by the pitch programmer before values reach the driver (default: 75.0).",
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
    parser.add_argument(
        "--pitch-authority",
        choices=("both", "oscillator", "driver", "manual"),
        default="both",
        help=(
            "Chooses which element leads the pitch program across the render. 'both' splits the render "
            "between oscillator-led then driver-led halves; 'manual' honours the follow/hold switches."
        ),
    )
    parser.add_argument(
        "--timebase-mode",
        choices=("disabled", "enabled"),
        default="disabled",
        help=(
            "Enable the pending continuous timebase bridge node for graph export and native experiments "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--timebase-algorithm",
        choices=("sinc_resample", "phase_vocoder"),
        default="sinc_resample",
        help=(
            "Continuous timebase algorithm to configure when --timebase-mode=enabled (default: sinc_resample)."
        ),
    )
    parser.add_argument(
        "--timebase-analysis-window",
        type=int,
        default=512,
        help="Analysis window size in frames for the continuous timebase node (default: 512).",
    )
    parser.add_argument(
        "--timebase-synthesis-window",
        type=int,
        default=512,
        help="Synthesis window size in frames for the continuous timebase node (default: 512).",
    )
    parser.add_argument(
        "--timebase-hop",
        type=int,
        default=128,
        help="Hop size in frames for overlap scheduling inside the timebase node (default: 128).",
    )
    parser.add_argument(
        "--timebase-similarity-span",
        type=int,
        default=4,
        help="Number of candidate windows scanned for SOLA/WSOLA alignment (default: 4).",
    )
    parser.add_argument(
        "--timebase-authority-bias",
        type=float,
        default=0.5,
        help="Authority bias (0=oscillator, 1=driver) applied to timebase alignment (default: 0.5).",
    )
    parser.add_argument(
        "--timebase-slew-limit",
        type=float,
        default=1.5,
        help="Maximum stretch ratio delta applied per block by the timebase node (default: 1.5).",
    )
    parser.add_argument(
        "--timebase-blend-crossfade",
        type=int,
        default=128,
        help="Crossfade length in frames used when the timebase authority toggles (default: 128).",
    )
    parser.add_argument(
        "--pitch-shift-algorithm",
        choices=("sola", "wsola", "phase_vocoder"),
        default="sola",
        help=(
            "Label the pitch-shift implementation for diagnostics and export (default: sola)."
        ),
    )
    parser.add_argument(
        "--pitch-track-mode",
        choices=("hybrid", "oscillator", "driver"),
        default="hybrid",
        help=(
            "Derive the pitch-shift ratio from oscillator-led, driver-led, or blended tracking "
            "(default: hybrid)."
        ),
    )
    parser.add_argument(
        "--portamento-ms",
        type=float,
        default=12.0,
        help="Smoothing window (ms) applied to pitch-shift ratio transitions (default: 12.0).",
    )
    parser.add_argument(
        "--pitch-shift-window",
        type=int,
        default=PitchShiftNode.ANALYSIS_WINDOW,
        help="Analysis window size in frames for the pitch-shift node (default: 1024).",
    )
    parser.add_argument(
        "--pitch-shift-hop",
        type=int,
        default=PitchShiftNode.HOP_SIZE,
        help="Hop size in frames for the pitch-shift node (default: 256).",
    )
    parser.add_argument(
        "--pitch-shift-resynthesis-hop",
        type=int,
        default=PitchShiftNode.RESYNTHESIS_HOP,
        help="Resynthesis hop size in frames for the pitch-shift node (default: 256).",
    )
    parser.add_argument(
        "--pitch-shift-min-ratio",
        type=float,
        default=0.25,
        help="Lower clamp applied to the computed pitch-shift ratio (default: 0.25).",
    )
    parser.add_argument(
        "--pitch-shift-max-ratio",
        type=float,
        default=4.0,
        help="Upper clamp applied to the computed pitch-shift ratio (default: 4.0).",
    )
    parser.add_argument(
        "--arpeggio-intervals",
        type=str,
        default="0,4,7",
        help=(
            "Comma separated semitone offsets (relative to --base-freq) used for the ascending arpeggio "
            "when --pitch-program=arpeggio (default: '0,4,7')."
        ),
    )
    parser.add_argument(
        "--arpeggio-chord",
        type=str,
        default="0,4,7",
        help=(
            "Comma separated semitone offsets that describe the sustained chord segment when "
            "--pitch-program=arpeggio (default: '0,4,7')."
        ),
    )
    parser.add_argument(
        "--arpeggio-whole-note",
        type=float,
        default=1.0,
        help="Duration in seconds of each whole note during the arpeggio program (default: 1.0).",
    )
    parser.add_argument(
        "--arpeggio-chord-hold",
        type=int,
        default=2,
        help="Number of whole notes to sustain the chord segment in the arpeggio program (default: 2).",
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
    enable_timebase = args.timebase_mode == "enabled"
    timebase_params: Mapping[str, object] | None = None
    if enable_timebase:
        timebase_params = {
            "algorithm": args.timebase_algorithm,
            "analysis_window": args.timebase_analysis_window,
            "synthesis_window": args.timebase_synthesis_window,
            "hop_size": args.timebase_hop,
            "similarity_span": args.timebase_similarity_span,
            "authority_bias": args.timebase_authority_bias,
            "slew_limit": args.timebase_slew_limit,
            "blend_crossfade": args.timebase_blend_crossfade,
        }

    pitch_shift_params: Mapping[str, object] | None = {
        "window_size": int(args.pitch_shift_window),
        "hop_size": int(args.pitch_shift_hop),
        "resynthesis_hop": int(args.pitch_shift_resynthesis_hop),
        "algorithm": str(args.pitch_shift_algorithm),
        "track_mode": str(args.pitch_track_mode),
        "portamento_ms": float(args.portamento_ms),
        "ratio_min": float(args.pitch_shift_min_ratio),
        "ratio_max": float(args.pitch_shift_max_ratio),
    }

    graph, module = build_graph(
        int(args.sr),
        enable_timebase=enable_timebase,
        timebase_params=timebase_params,
        pitch_shift_params=pitch_shift_params,
    )
    pitch_node = module.pitch
    pitch_node.params["default_slew"] = max(0.0, float(args.pitch_slew))
    pitch_node.params["min_freq"] = max(0.0, float(args.driver_min_freq))
    module.pitch_shift.params.setdefault("algorithm", str(args.pitch_shift_algorithm))
    module.pitch_shift.params.setdefault("track_mode", str(args.pitch_track_mode))
    module.pitch_shift.params.setdefault("portamento_ms", float(args.portamento_ms))
    module.pitch_shift.params.setdefault("ratio_min", float(args.pitch_shift_min_ratio))
    module.pitch_shift.params.setdefault("ratio_max", float(args.pitch_shift_max_ratio))
    log(
        "[demo] Pitch-shift node '%s' bridges driver '%s' → oscillator '%s' (algorithm=%s, track_mode=%s, portamento_ms=%.2f)"
        % (
            module.pitch_shift.name,
            module.driver.name,
            module.oscillator.name,
            module.pitch_shift.params.get("algorithm", "<unset>"),
            module.pitch_shift.params.get("track_mode", "<unset>"),
            float(module.pitch_shift.params.get("portamento_ms", 0.0)),
        )
    )
    if module.timebase is not None:
        log(
            "[demo] Continuous timebase node '%s' configured (algorithm=%s)"
            % (module.timebase.name, module.timebase.params.get("algorithm", "<unset>"))
        )
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
        pending_native: List[str] = []
        if module.timebase is not None:
            pending_native.append(module.timebase.name)
        ensure_native_kernels(
            executor,
            [node.name for node in graph.ordered_nodes],
            allow_missing=pending_native,
        )

        total_frames = int(round(args.duration * args.sr))
        if total_frames <= 0:
            raise ValueError("duration must produce at least one sample")
        block_size = int(args.block_size)
        if block_size <= 0:
            raise ValueError("block-size must be positive")

        log(f"[demo] Rendering {total_frames} frames (block size {block_size})...")
        arpeggio_intervals = _parse_semitone_list(args.arpeggio_intervals, fallback=(0,))
        chord_intervals = _parse_semitone_list(args.arpeggio_chord, fallback=arpeggio_intervals)
        pitch_program = generate_pitch_schedule(
            total_frames,
            args.sr,
            args.pitch_modulation,
            base_freq=float(args.base_freq),
            pitch_depth=float(args.pitch_depth),
            driver_min_freq=max(1.0e-6, float(args.driver_min_freq)),
            program=args.pitch_program,
            arpeggio_intervals=arpeggio_intervals,
            chord_intervals=chord_intervals,
            whole_note_seconds=float(args.arpeggio_whole_note),
            chord_hold_notes=int(args.arpeggio_chord_hold),
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

        authority_mode = str(args.pitch_authority)
        driver_hold_mask = np.zeros(total_frames, dtype=bool)
        osc_hold_mask = np.zeros(total_frames, dtype=bool)
        if authority_mode == "manual":
            if args.driver_pitch_mode == "hold":
                driver_hold_mask[:] = True
            if args.oscillator_pitch_mode == "hold":
                osc_hold_mask[:] = True
            log(
                "[demo] Manual authority: driver=%s oscillator=%s"
                % (args.driver_pitch_mode, args.oscillator_pitch_mode)
            )
        elif authority_mode == "oscillator":
            driver_hold_mask[:] = True
            log("[demo] Authority: oscillator-led (driver holds, oscillator follows)")
        elif authority_mode == "driver":
            osc_hold_mask[:] = True
            log("[demo] Authority: driver-led (oscillator holds, driver follows)")
        elif authority_mode == "both":
            split = max(1, total_frames // 2)
            driver_hold_mask[:split] = True
            osc_hold_mask[split:] = True
            render_mode_curve[:split] = np.minimum(render_mode_curve[:split], 0.3)
            render_mode_curve[split:] = np.maximum(render_mode_curve[split:], 0.7)
            log(
                "[demo] Authority split: first half oscillator-led, second half driver-led (split frame %d)"
                % split
            )
        else:
            raise ValueError(f"unsupported pitch authority '{authority_mode}'")

        if np.any(driver_hold_mask):
            driver_freq_curve = _apply_hold_mask(driver_freq_curve, driver_hold_mask)
            driver_amp_curve = _apply_hold_mask(driver_amp_curve, driver_hold_mask)
        if np.any(osc_hold_mask):
            master_freq_curve = _apply_hold_mask(master_freq_curve, osc_hold_mask)

        slew_curve = np.full(total_frames, max(0.0, float(args.op_amp_slew)), dtype=np.float64)
        pitch_slew_curve = np.full(total_frames, max(0.0, float(args.pitch_slew)), dtype=np.float64)
        ratio_curve = np.ones(total_frames, dtype=np.float64)
        if module.pitch_shift is not None:
            driver_freq_safe = np.maximum(driver_freq_curve, 1.0e-6)
            osc_freq_safe = np.maximum(master_freq_curve, 1.0e-6)
            osc_ratio = np.divide(osc_freq_safe, driver_freq_safe, out=np.ones_like(driver_freq_safe), where=driver_freq_safe > 0.0)
            min_ratio = float(module.pitch_shift.params.get("ratio_min", args.pitch_shift_min_ratio))
            max_ratio = float(module.pitch_shift.params.get("ratio_max", args.pitch_shift_max_ratio))
            min_ratio = min_ratio if min_ratio > 0.0 else 0.25
            max_ratio = max(max_ratio, min_ratio)
            track_mode = str(args.pitch_track_mode)
            if track_mode == "driver":
                ratio_curve.fill(1.0)
            elif track_mode == "oscillator":
                ratio_curve = osc_ratio.copy()
            else:
                blend = np.clip(render_mode_curve, 0.0, 1.0)
                ratio_curve = blend * 1.0 + (1.0 - blend) * osc_ratio
            if np.any(driver_hold_mask):
                ratio_curve[driver_hold_mask] = osc_ratio[driver_hold_mask]
            if np.any(osc_hold_mask):
                ratio_curve[osc_hold_mask] = 1.0
            ratio_curve = np.clip(ratio_curve, min_ratio, max_ratio)
            portamento_ms = max(0.0, float(args.portamento_ms))
            if portamento_ms > 0.0 and ratio_curve.size > 1:
                samples = max(1, int(round(portamento_ms * float(args.sr) / 1000.0)))
                alpha = max(0.0, min(1.0, 1.0 / float(samples)))
                smoothed = ratio_curve.copy()
                current = float(smoothed[0])
                for idx in range(1, smoothed.size):
                    target = float(ratio_curve[idx])
                    current += (target - current) * alpha
                    smoothed[idx] = current
                ratio_curve = smoothed
            ratio_min = float(ratio_curve.min()) if ratio_curve.size else 1.0
            ratio_max_val = float(ratio_curve.max()) if ratio_curve.size else 1.0
            log(
                "[demo] Pitch-shift ratio prepared: min={:.5f}, max={:.5f}, mode={}, portamento_ms={:.2f}"
                .format(ratio_min, ratio_max_val, track_mode, portamento_ms)
            )
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
            if np.all(driver_hold_mask):
                driver_mode_label = "hold"
            elif not np.any(driver_hold_mask):
                driver_mode_label = "follow"
            else:
                driver_mode_label = "split"
            log(
                "[demo] Driver stats: freq[min={:.4f}, max={:.4f}] Hz, amp[min={:.4f}, max={:.4f}], "
                "blend[min={:.4f}, max={:.4f}] mode={}".format(
                    float(driver_freq_curve.min()),
                    float(driver_freq_curve.max()),
                    float(driver_amp_curve.min()),
                    float(driver_amp_curve.max()),
                    float(render_mode_curve.min()),
                    float(render_mode_curve.max()),
                    driver_mode_label,
                )
            )
        if master_freq_curve.size:
            if np.all(osc_hold_mask):
                osc_mode_label = "hold"
            elif not np.any(osc_hold_mask):
                osc_mode_label = "follow"
            else:
                osc_mode_label = "split"
            log(
                "[demo] Oscillator stats: freq[min={:.4f}, max={:.4f}] Hz mode={}".format(
                    float(master_freq_curve.min()),
                    float(master_freq_curve.max()),
                    osc_mode_label,
                )
            )

        if module.pitch_shift is not None:
            if np.any(driver_hold_mask):
                osc_map_path = out_dir / "network_map_oscillator_led.json"
                export_network_map(graph, osc_map_path)
                log(
                    "[demo] Exported oscillator-led network map (pitch shift between driver and oscillator): %s"
                    % osc_map_path
                )
            if np.any(osc_hold_mask):
                driver_map_path = out_dir / "network_map_driver_led.json"
                export_network_map(graph, driver_map_path)
                log(
                    "[demo] Exported driver-led network map (pitch shift bridge preserved): %s"
                    % driver_map_path
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
            if module.pitch_shift is not None:
                pitch_shift_block = create_param_block({"ratio": ratio_curve[sl]})
                base_params[module.pitch_shift.name] = pitch_shift_block
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
