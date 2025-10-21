"""Utilities for benchmarking individual nodes across varying batch sizes.

This module provides a small harness that instantiates each registered node
type, feeds it synthetic data, and records the processing latency for a set of
batch sizes.  The intent is to make it easy to compare the raw performance of
nodes without the rest of the audio graph machinery so we can reason about
queueing or buffer copying overhead when diagnosing glitches.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np

from . import nodes
from .state import RAW_DTYPE


# ---------------------------------------------------------------------------
# Benchmark specification helpers


PrepareFn = Callable[[np.random.Generator, int, int, float], tuple[np.ndarray | None, dict[str, np.ndarray]]]
RunnerFn = Callable[[nodes.Node, int, float, np.ndarray | None, dict[str, np.ndarray]], np.ndarray | None]
ResetFn = Callable[[nodes.Node], None]


@dataclass(slots=True)
class NodeBenchmarkSpec:
    """Definition for how to benchmark a particular node type."""

    factory: Callable[[float], nodes.Node]
    prepare: PrepareFn
    runner: RunnerFn | None = None
    reset: ResetFn | None = None


@dataclass(slots=True)
class BenchmarkStats:
    """Simple statistics captured for each (node, batch size) pair."""

    mean_seconds: float
    stdev_seconds: float
    min_seconds: float
    max_seconds: float


def _random_bcf(
    rng: np.random.Generator,
    batches: int,
    channels: int,
    frames: int,
    *,
    low: float,
    high: float,
) -> np.ndarray:
    return rng.uniform(low, high, size=(batches, channels, frames)).astype(RAW_DTYPE, copy=False)


def _zero_bcf(batches: int, channels: int, frames: int) -> np.ndarray:
    return np.zeros((batches, channels, frames), dtype=RAW_DTYPE)


def _controller_runner(
    node: nodes.ControllerNode,
    frames: int,
    sample_rate: float,
    audio_in: np.ndarray | None,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    output, _, _ = node._evaluate(params, frames, int(sample_rate))  # type: ignore[attr-defined]
    return output


def _controller_reset(node: nodes.ControllerNode) -> None:
    with node._result_lock:  # type: ignore[attr-defined]
        node._latest_output = None  # type: ignore[attr-defined]
        node._latest_meta = None  # type: ignore[attr-defined]
        node._last_error = None  # type: ignore[attr-defined]
    try:
        while True:
            node._task_queue.get_nowait()  # type: ignore[attr-defined]
    except Exception:
        pass


def _envelope_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    trigger = (rng.random((batches, 1, frames)) > 0.95).astype(RAW_DTYPE, copy=False)
    gate = (rng.random((batches, 1, frames)) > 0.7).astype(RAW_DTYPE, copy=False)
    drone = (rng.random((batches, 1, frames)) > 0.9).astype(RAW_DTYPE, copy=False)
    velocity = _random_bcf(rng, batches, 1, frames, low=0.4, high=1.0)
    send_reset = np.ones((batches, 1, frames), dtype=RAW_DTYPE)
    return audio, {
        "trigger": trigger,
        "gate": gate,
        "drone": drone,
        "velocity": velocity,
        "send_reset": send_reset,
    }


def _quantizer_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    ctrl = _random_bcf(rng, batches, 1, frames, low=-1.0, high=1.0)
    root = np.full((batches, 1, frames), 60.0, dtype=RAW_DTYPE)
    span = np.full((batches, 1, frames), 2.0, dtype=RAW_DTYPE)
    return audio, {"input": ctrl, "root_midi": root, "span_oct": span}


def _amplifier_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    base = _random_bcf(rng, batches, 1, frames, low=0.0, high=1.0)
    control = _random_bcf(rng, batches, 1, frames, low=0.0, high=1.0)
    mod = _random_bcf(rng, batches, 1, frames, low=-0.5, high=0.5)
    return audio, {"base": base, "control": control, "mod": mod}


def _osc_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    freq = _random_bcf(rng, batches, 1, frames, low=55.0, high=880.0)
    amp = _random_bcf(rng, batches, 1, frames, low=0.1, high=0.9)
    pan = _random_bcf(rng, batches, 1, frames, low=-1.0, high=1.0)
    return audio, {"freq": freq, "amp": amp, "pan": pan}


def _sine_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    freq = _random_bcf(rng, batches, 1, frames, low=110.0, high=660.0)
    amp = _random_bcf(rng, batches, 1, frames, low=0.2, high=0.8)
    return audio, {"frequency": freq, "amplitude": amp}


def _controller_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    a = _random_bcf(rng, batches, 1, frames, low=-1.0, high=1.0)
    b = _random_bcf(rng, batches, 1, frames, low=-1.0, high=1.0)
    c = _random_bcf(rng, batches, 1, frames, low=0.0, high=1.0)
    return audio, {"a": a, "b": b, "c": c}


def _mix_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _random_bcf(rng, batches, 4, frames, low=-0.8, high=0.8)
    return audio, {}


def _safety_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _random_bcf(rng, batches, 2, frames, low=-1.2, high=1.2)
    return audio, {}


def _subharm_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _random_bcf(rng, batches, 2, frames, low=-1.0, high=1.0)
    return audio, {}


def _constant_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    return audio, {}


def _silence_prepare(
    rng: np.random.Generator, batches: int, frames: int, _: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    audio = _zero_bcf(batches, 1, frames)
    return audio, {}


NODE_BENCHMARKS: dict[str, NodeBenchmarkSpec] = {
    "silence": NodeBenchmarkSpec(
        factory=lambda sr: nodes.SilenceNode("silence"),
        prepare=_silence_prepare,
    ),
    "constant": NodeBenchmarkSpec(
        factory=lambda sr: nodes.ConstantNode("constant"),
        prepare=_constant_prepare,
    ),
    "controller": NodeBenchmarkSpec(
        factory=lambda sr: nodes.ControllerNode(
            "controller",
            params={
                "outputs": {
                    "sum": "signals['a'] + signals['b']",
                    "diff": "signals['a'] - signals['b']",
                    "clamp": "np.clip(signals['c'], 0.0, 1.0)",
                }
            },
        ),
        prepare=_controller_prepare,
        runner=_controller_runner,
        reset=_controller_reset,
    ),
    "sine": NodeBenchmarkSpec(
        factory=lambda sr: nodes.SineOscillatorNode("sine"),
        prepare=_sine_prepare,
    ),
    "sine_oscillator": NodeBenchmarkSpec(
        factory=lambda sr: nodes.SineOscillatorNode("sine"),
        prepare=_sine_prepare,
    ),
    "osc": NodeBenchmarkSpec(
        factory=lambda sr: nodes.OscNode("osc"),
        prepare=_osc_prepare,
    ),
    "oscillator": NodeBenchmarkSpec(
        factory=lambda sr: nodes.OscNode("osc"),
        prepare=_osc_prepare,
    ),
    "envelope": NodeBenchmarkSpec(
        factory=lambda sr: nodes.EnvelopeModulatorNode("env"),
        prepare=_envelope_prepare,
    ),
    "envelope_modulator": NodeBenchmarkSpec(
        factory=lambda sr: nodes.EnvelopeModulatorNode("env"),
        prepare=_envelope_prepare,
    ),
    "pitch_quantizer": NodeBenchmarkSpec(
        factory=lambda sr: nodes.PitchQuantizerNode("quant", state={"base_token": "12tet/full"}),
        prepare=_quantizer_prepare,
    ),
    "amplifier_modulator": NodeBenchmarkSpec(
        factory=lambda sr: nodes.AmplifierModulatorNode("amp"),
        prepare=_amplifier_prepare,
    ),
    "mix": NodeBenchmarkSpec(
        factory=lambda sr: nodes.MixNode("mix"),
        prepare=_mix_prepare,
    ),
    "safety": NodeBenchmarkSpec(
        factory=lambda sr: nodes.SafetyNode("safety"),
        prepare=_safety_prepare,
    ),
    "subharmonic_low_lifter": NodeBenchmarkSpec(
        factory=lambda sr: nodes.SubharmonicLowLifterNode("sub", sr),
        prepare=_subharm_prepare,
    ),
}


def run_node_benchmarks(
    batch_sizes: Iterable[int],
    *,
    frames: int = 256,
    sample_rate: float = 48_000.0,
    iterations: int = 5,
    node_names: Iterable[str] | None = None,
    seed: int = 0,
) -> dict[str, dict[int, BenchmarkStats]]:
    """Execute the benchmark suite and return summary statistics."""

    selected = list(node_names) if node_names is not None else list(NODE_BENCHMARKS)
    unknown = sorted(name for name in selected if name not in NODE_BENCHMARKS)
    if unknown:
        raise KeyError(f"Unknown node types requested: {', '.join(unknown)}")

    results: dict[str, dict[int, BenchmarkStats]] = {}
    for node_name in selected:
        spec = NODE_BENCHMARKS[node_name]
        node_results: dict[int, BenchmarkStats] = {}
        for batch in batch_sizes:
            if batch <= 0:
                raise ValueError("batch sizes must be positive integers")
            rng_seed = seed + hash((node_name, batch)) & 0xFFFFFFFFFFFF
            rng = np.random.default_rng(rng_seed)
            node = spec.factory(sample_rate)

            audio_in, params = spec.prepare(rng, batch, frames, sample_rate)
            runner = spec.runner or (lambda n, f, sr, a, p: n.process(f, sr, a, {}, p))
            if spec.reset is not None:
                spec.reset(node)
            runner(node, frames, sample_rate, audio_in, params)

            times: list[float] = []
            for _ in range(iterations):
                audio_in, params = spec.prepare(rng, batch, frames, sample_rate)
                if spec.reset is not None:
                    spec.reset(node)
                start = time.perf_counter()
                output = runner(node, frames, sample_rate, audio_in, params)
                if isinstance(output, np.ndarray):
                    _ = float(np.sum(output))
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            times_arr = np.array(times, dtype=RAW_DTYPE)
            node_results[batch] = BenchmarkStats(
                mean_seconds=float(times_arr.mean()),
                stdev_seconds=float(times_arr.std(ddof=0)),
                min_seconds=float(times_arr.min()),
                max_seconds=float(times_arr.max()),
            )
        results[node_name] = node_results
    return results


def _format_table(results: Mapping[str, Mapping[int, BenchmarkStats]]) -> str:
    if not results:
        return "No results"
    batches = sorted({batch for node in results.values() for batch in node})
    header = ["Node"] + [f"B={batch}" for batch in batches]
    widths = [max(len(header[0]), 12)] + [max(len(h), 12) for h in header[1:]]
    lines = [" ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append(" ".join("-" * w for w in widths))
    for name in sorted(results):
        row = [name.ljust(widths[0])]
        node_results = results[name]
        for idx, batch in enumerate(batches, start=1):
            stats = node_results.get(batch)
            if stats is None:
                cell = "n/a"
            else:
                mean_ms = stats.mean_seconds * 1e3
                stdev_ms = stats.stdev_seconds * 1e3
                cell = f"{mean_ms:6.3f}±{stdev_ms:5.3f} ms"
            row.append(cell.ljust(widths[idx]))
        lines.append(" ".join(row))
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark AMP nodes across batch sizes")
    parser.add_argument("--frames", type=int, default=256, help="Frames per render block")
    parser.add_argument("--sample-rate", type=float, default=48_000.0, help="Sample rate")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Samples per measurement")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    parser.add_argument(
        "--nodes",
        nargs="*",
        default=None,
        help="Optional subset of node names to benchmark",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available node names and exit",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        for name in sorted(NODE_BENCHMARKS):
            print(name)
        return 0

    results = run_node_benchmarks(
        args.batch_sizes,
        frames=args.frames,
        sample_rate=args.sample_rate,
        iterations=args.iterations,
        node_names=args.nodes,
        seed=args.seed,
    )
    print(_format_table(results))
    return 0


__all__ = [
    "BenchmarkStats",
    "NODE_BENCHMARKS",
    "NodeBenchmarkSpec",
    "main",
    "run_node_benchmarks",
]


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())

