#!/usr/bin/env python3
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
from typing import Dict

import numpy as np
import pandas as pd

# Ensure we can import the package from the local src directory.
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amp.system import (
    benchmark_default_graph as run_benchmark_default_graph,
    require_native_graph_runtime,
)

def _summarise_timeline(df: pd.DataFrame, ema_alpha: float) -> None:
    if df.empty:
        print("No benchmark samples were recorded.")
        return

    produced_ms = float(df["block_ms"].iloc[0]) if "block_ms" in df else float("nan")
    iterations = int((~df["is_warmup"]).sum()) if "is_warmup" in df else len(df)

    print(
        f"Rendered {iterations} iterations of {int(df['end_sample'].iloc[0] - df['start_sample'].iloc[0]) if 'end_sample' in df else 'unknown'} frames "
        f"({produced_ms:.2f} ms per block)"
    )

    if "is_warmup" in df:
        measured = df.loc[~df["is_warmup"]]
    else:
        measured = df

    node_columns = [col for col in df.columns if col.startswith("node_") and col.endswith("_ms")]
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    peaks: Dict[str, float] = {}
    ema: Dict[str, float] = {}

    for col in node_columns:
        node = col[len("node_") : -len("_ms")]
        if measured.empty:
            totals[node] = 0.0
            counts[node] = 0
            peaks[node] = 0.0
            ema[node] = 0.0
            continue
        values_ms = measured[col].to_numpy(dtype=float)
        totals[node] = float(np.sum(values_ms)) / 1000.0
        counts[node] = int(values_ms.size)
        peaks[node] = float(np.max(values_ms))

    if measured.empty:
        ema.clear()
    else:
        for _, row in measured.iterrows():
            for col in node_columns:
                node = col[len("node_") : -len("_ms")]
                value_sec = float(row[col]) / 1000.0
                previous = ema.get(node)
                ema[node] = value_sec if previous is None else previous + ema_alpha * (value_sec - previous)

    print()
    print(f"Moving averages (alpha={ema_alpha:.3f}) sorted by descending EMA:")
    ordered_nodes = sorted(ema, key=lambda key: ema[key], reverse=True)
    for node in ordered_nodes:
        mean_sec = totals.get(node, 0.0) / max(1, counts.get(node, 1))
        ema_ms = ema[node] * 1000.0
        peak_ms = peaks.get(node, 0.0)
        print(f"  {node:<24} avg {mean_sec * 1000.0:7.3f} ms  ema {ema_ms:7.3f} ms  peak {peak_ms:7.3f} ms")

    if measured.empty:
        underrun_count = 0
        total_gap = 0.0
    else:
        underrun_count = int((measured.get("underrun_gap_ms", 0.0) > 0.0).sum())
        total_gap = float(measured.get("underrun_gap_ms", pd.Series(dtype=float)).sum()) if "underrun_gap_ms" in measured else 0.0

    print()
    print(
        "Real-time timeline summary:",
        f" {len(measured)} measured blocks, {underrun_count} underruns",
        f" totalling {total_gap:.3f} ms",
    )

    preview = df.head(min(6, len(df)))
    if not preview.empty:
        print()
        print("Timeline preview (first rows):")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(preview.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


def benchmark_default_graph(
    *,
    frames: int,
    iterations: int,
    sample_rate: float,
    ema_alpha: float,
    warmup: int,
    joystick_mode: str,
    joystick_script: pathlib.Path | None,
) -> pd.DataFrame:
    df = run_benchmark_default_graph(
        frames=frames,
        iterations=iterations,
        sample_rate=sample_rate,
        ema_alpha=ema_alpha,
        warmup=warmup,
        joystick_mode=joystick_mode,
        joystick_script_path=joystick_script,
    )
    _summarise_timeline(df, ema_alpha)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the default AMP controller graph headlessly")
    parser.add_argument("--frames", type=int, default=256, help="Frames per render block")
    parser.add_argument("--iterations", type=int, default=512, help="Number of benchmark iterations (excluding warmup)")
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

    benchmark_default_graph(
        frames=args.frames,
        iterations=args.iterations,
        sample_rate=args.rate,
        ema_alpha=args.alpha,
        warmup=max(0, args.warmup),
        joystick_mode=args.joystick_mode,
        joystick_script=args.joystick_script,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
