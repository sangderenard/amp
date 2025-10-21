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
    block_sizes: Iterable[int] | None = None,
    sample_rate: float = 48_000.0,
    iterations: int = 5,
    node_names: Iterable[str] | None = None,
    seed: int = 0,
) -> dict[str, dict[int, dict[int, BenchmarkStats]]]:
    """Execute the benchmark suite and return summary statistics.

    The benchmark now sweeps both the batch size and the render block size
    (expressed in audio frames).  For backwards compatibility ``frames`` can be
    provided to benchmark a single block size, while ``block_sizes`` allows
    specifying multiple values.
    """

    selected = list(node_names) if node_names is not None else list(NODE_BENCHMARKS)
    unknown = sorted(name for name in selected if name not in NODE_BENCHMARKS)
    if unknown:
        raise KeyError(f"Unknown node types requested: {', '.join(unknown)}")

    block_sizes_list = list(block_sizes if block_sizes is not None else [frames])
    if not block_sizes_list:
        raise ValueError("at least one block size must be provided")

    for block in block_sizes_list:
        if block <= 0:
            raise ValueError("block sizes must be positive integers")

    results: dict[str, dict[int, dict[int, BenchmarkStats]]] = {}
    for node_name in selected:
        spec = NODE_BENCHMARKS[node_name]
        node_block_results: dict[int, dict[int, BenchmarkStats]] = {}
        for block in block_sizes_list:
            node_results: dict[int, BenchmarkStats] = {}
            for batch in batch_sizes:
                if batch <= 0:
                    raise ValueError("batch sizes must be positive integers")
                rng_seed = seed + hash((node_name, batch, block)) & 0xFFFFFFFFFFFF
                rng = np.random.default_rng(rng_seed)
                node = spec.factory(sample_rate)

                audio_in, params = spec.prepare(rng, batch, block, sample_rate)
                runner = spec.runner or (lambda n, f, sr, a, p: n.process(f, sr, a, {}, p))
                if spec.reset is not None:
                    spec.reset(node)
                runner(node, block, sample_rate, audio_in, params)

                times: list[float] = []
                for _ in range(iterations):
                    audio_in, params = spec.prepare(rng, batch, block, sample_rate)
                    if spec.reset is not None:
                        spec.reset(node)
                    start = time.perf_counter()
                    output = runner(node, block, sample_rate, audio_in, params)
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
            node_block_results[block] = node_results
        results[node_name] = node_block_results
    return results


def _format_table(results: Mapping[str, Mapping[int, Mapping[int, BenchmarkStats]]]) -> str:
    if not results:
        return "No results"

    blocks = sorted({block for node in results.values() for block in node})
    batches = sorted({batch for node in results.values() for block in node.values() for batch in block})
    if not batches or not blocks:
        return "No results"

    tables: list[str] = []
    for block in blocks:
        header = [f"Block={block}"] + [f"B={batch}" for batch in batches]
        widths = [max(len(header[0]), 12)] + [max(len(h), 12) for h in header[1:]]
        lines = [" ".join(h.ljust(w) for h, w in zip(header, widths))]
        lines.append(" ".join("-" * w for w in widths))
        for name in sorted(results):
            row = [name.ljust(widths[0])]
            node_results = results[name].get(block, {})
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
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


def _surface_grid(
    node_results: Mapping[int, Mapping[int, BenchmarkStats]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid-friendly arrays for plotting surfaces."""

    block_sizes = sorted(node_results)
    batch_sizes = sorted({batch for block in node_results.values() for batch in block})
    if not block_sizes or not batch_sizes:
        raise ValueError("benchmark results did not include any batch or block sizes")

    surface = np.full((len(block_sizes), len(batch_sizes)), np.nan, dtype=float)
    for i, block in enumerate(block_sizes):
        batches = node_results[block]
        for j, batch in enumerate(batch_sizes):
            stats = batches.get(batch)
            if stats is not None:
                surface[i, j] = stats.mean_seconds * 1e3  # milliseconds

    return (
        np.asarray(batch_sizes, dtype=float),
        np.asarray(block_sizes, dtype=float),
        surface,
    )


def plot_benchmark_surface(
    node_results: Mapping[int, Mapping[int, BenchmarkStats]],
    *,
    node_name: str,
    output_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """Render a 3D surface plot of the benchmark data for a single node.

    Parameters
    ----------
    node_results:
        Mapping of block size to per-batch benchmark statistics for a node.
    node_name:
        Name of the node the results correspond to.  Used for labelling.
    output_path:
        Optional location to save the generated figure.  When ``None`` the
        figure is still created and returned to the caller but not saved.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers projection
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "matplotlib is required to render benchmark surface plots"
        ) from exc

    batch_sizes, block_sizes, surface = _surface_grid(node_results)
    batch_grid, block_grid = np.meshgrid(batch_sizes, block_sizes)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = ax.plot_surface(batch_grid, block_grid, surface, cmap=cm.viridis, edgecolor="none")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Block size (frames)")
    ax.set_zlabel("Mean execution time (ms)")
    ax.set_title(f"Benchmark Surface for {node_name}")
    fig.colorbar(mesh, ax=ax, shrink=0.6, label="Mean execution time (ms)")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark AMP nodes across batch sizes and block sizes")
    parser.add_argument(
        "--frames",
        type=int,
        default=256,
        help="Frames per render block (deprecated, use --block-sizes)",
    )
    parser.add_argument("--sample-rate", type=float, default=48_000.0, help="Sample rate")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="*",
        default=[32, 64, 128, 256, 512],
        help="Block sizes (in frames) to benchmark",
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
    parser.add_argument(
        "--surface-node",
        type=str,
        default=None,
        help="Render a surface plot for the specified node",
    )
    parser.add_argument(
        "--surface-output",
        type=str,
        default=None,
        help="Optional output path for the surface plot (defaults to <node>_surface.png)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        for name in sorted(NODE_BENCHMARKS):
            print(name)
        return 0

    block_sizes = args.block_sizes if args.block_sizes else [args.frames]

    results = run_node_benchmarks(
        args.batch_sizes,
        block_sizes=block_sizes,
        frames=args.frames,
        sample_rate=args.sample_rate,
        iterations=args.iterations,
        node_names=args.nodes,
        seed=args.seed,
    )
    print(_format_table(results))

    if args.surface_node is not None:
        node_name = args.surface_node
        if node_name not in results:
            raise KeyError(f"Surface requested for unknown node '{node_name}'")
        node_results = results[node_name]
        output_path = args.surface_output or f"{node_name}_surface.png"
        plot_benchmark_surface(node_results, node_name=node_name, output_path=output_path)
        print(f"Saved surface plot for '{node_name}' to {output_path}")
    return 0


__all__ = [
    "BenchmarkStats",
    "NODE_BENCHMARKS",
    "NodeBenchmarkSpec",
    "main",
    "plot_benchmark_surface",
    "run_node_benchmarks",
]


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(main())

