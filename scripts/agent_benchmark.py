"""Headless benchmarking helper for AMP agents.

This script delegates to :func:`amp.system.benchmark_default_graph` so the
interactive and headless execution paths share the same CFFI-backed graph
runtime.  It prints a concise textual summary and returns the timeline as a
:class:`pandas.DataFrame` for further analysis.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

# Ensure we can import the package from the local src directory.
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amp.system import (
    benchmark_default_graph as run_benchmark_default_graph,
    require_native_graph_runtime,
    summarise_benchmark_timeline,
)


def benchmark_default_graph(
    *,
    frames: int,
    iterations: int,
    sample_rate: float,
    ema_alpha: float,
    warmup: int,
    joystick_mode: str,
    joystick_script: pathlib.Path | None,
    batch_blocks: int,
) -> pd.DataFrame:
    df = run_benchmark_default_graph(
        frames=frames,
        iterations=iterations,
        sample_rate=sample_rate,
        ema_alpha=ema_alpha,
        warmup=warmup,
        joystick_mode=joystick_mode,
        joystick_script_path=joystick_script,
        batch_blocks=batch_blocks,
    )
    for line in summarise_benchmark_timeline(df, ema_alpha=ema_alpha):
        print(line)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the default AMP controller graph headlessly")
    parser.add_argument("--frames", type=int, default=256, help="Frames per render block")
    parser.add_argument("--iterations", type=int, default=256, help="Number of benchmark iterations (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=32, help="Warmup iterations to discard from EMA")
    parser.add_argument("--rate", type=float, default=44100.0, help="Sample rate in Hz")
    parser.add_argument("--alpha", type=float, default=0.02, help="EMA smoothing factor (0-1)")
    parser.add_argument(
        "--joystick-mode",
        choices=("switch", "axis"),
        default="switch",
        help="Virtual joystick style: 'switch' uses on/off buttons, 'axis' emulates analog strikes",
    )
    parser.add_argument(
        "--joystick-script",
        type=pathlib.Path,
        help="Optional JSON file containing prewritten joystick automation",
    )
    parser.add_argument(
        "--batch-blocks",
        type=int,
        default=1,
        help="Number of callback-sized blocks rendered per benchmark iteration",
    )
    args = parser.parse_args()

    try:
        require_native_graph_runtime()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if args.frames <= 0:
        raise SystemExit("Frames must be positive")
    if not (0.0 < args.alpha <= 1.0):
        raise SystemExit("EMA alpha must be in the interval (0, 1]")
    if args.iterations <= 0:
        raise SystemExit("Iterations must be positive")
    if args.batch_blocks <= 0:
        raise SystemExit("Batch blocks must be positive")

    benchmark_default_graph(
        frames=args.frames,
        iterations=args.iterations,
        sample_rate=args.rate,
        ema_alpha=args.alpha,
        warmup=max(0, args.warmup),
        joystick_mode=args.joystick_mode,
        joystick_script=args.joystick_script,
        batch_blocks=args.batch_blocks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
