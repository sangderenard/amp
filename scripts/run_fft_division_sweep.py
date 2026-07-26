#!/usr/bin/env python3
"""FFTDivisionNode test battery with timeout and permutation table.

This helper drives the native ``test_fft_division_node`` binary through a
structured grid of parameter combinations (window power, hop or overlap,
working-wheel overrides, and warm-up policy). Every run exercises the real
runtime path—there are no Python fallbacks or smoke harnesses.

Features
--------
* 60-second timeout per invocation (override via ``--timeout``); timeouts are
    treated as failures and annotated in the final table.
* Deterministic parameter grid plus optional filtering via ``--cases``.
* Summary permutation table that lists each configuration and its PASS/FAIL
    status so correlations are obvious at a glance.
* Agent policy reminder: only the native runtime path is acceptable—Python
    fallback harnesses or "smoke" shims are explicitly forbidden.
* Exhaustive mode now honors per-axis CLI filters, so you can lock specific
    dimensions (e.g. ``--working-window-values 128``) or trim ranges without
    editing the script.
* Curated sweep presets (``--sweep-preset``) cover small-window half-overlap,
    large-window half-overlap, and a sparse-but-diverse sampler between the
    curated defaults and the true exhaustive grid.
* Timeout classification now consumes native stage counters/timestamps to decide
    whether a timeout was idle or active, so budgeting decisions follow real
    pipeline activity rather than textual heuristics.
* A continuous "stats heartbeat" log line mirrors the exit-code metadata so
    forced timeouts still carry the latest stage/timestamp snapshot even if the
    harness is killed before emitting the exit diagnostics.
* Structured JSON output (``--summary-json``) captures every case result so
    ``scripts/plot_fft_division_sweep.py`` can render visual diagnostics.
* Two-bit result flags capture whether a run finished before timeout and whether
    it represents a "good" outcome (pass or active timeout) or "bad" outcome
    (fail or idle timeout), enabling lightweight automation hooks.
* ``--min-execution-time`` enforces a per-case budget floor so only a subset of
    permutations (as chosen by the scheduler) can run before patience is spent.
* An entropy-aware scheduler pre-allocates every parameter permutation, then
    continually re-sorts the pending pool using a lightweight network that chases
    high-entropy regions while sprinkling jitter to break ties.

Examples
--------
        python scripts/run_fft_division_sweep.py
        python scripts/run_fft_division_sweep.py --cases overlap_low,grid_prefill --passthrough
        python scripts/run_fft_division_sweep.py --timeout 120 --stop-on-failure

The binary is auto-discovered under ``build/`` (Debug/Release buckets on
Windows). Build beforehand via ``cmake --build build --config <cfg>``."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

KNOWN_TIMEOUT_BUCKETS: Tuple[str, ...] = ("idle", "active", "unknown")

FLAG_COMBO_LABELS: dict[Tuple[int, int], str] = {
    (0, 0): "timeout_bad",
    (0, 1): "timeout_active",
    (1, 0): "finished_bad",
    (1, 1): "finished_good",
}

FLAG_LABEL_UNIVERSE: Tuple[str, ...] = tuple(sorted(set(FLAG_COMBO_LABELS.values()) | {"unknown_flag_combo"}))
FLAG_LABEL_ENTROPY_DENOM = (
    math.log(len(FLAG_LABEL_UNIVERSE), 2)
    if len(FLAG_LABEL_UNIVERSE) > 1
    else 1.0
)

DEFAULT_VERBOSITY = "detail"
DEFAULT_EXIT_CODE_ARGS: Tuple[str, ...] = (
    "--exit-code-default",
    "64",
    "--exit-code-stage",
    "idle=65",
    "--exit-code-stage",
    "stage1_ingest=66",
    "--exit-code-stage",
    "stage2_wheel=67",
    "--exit-code-stage",
    "stage3_operator=68",
    "--exit-code-stage",
    "stage4_emit=69",
    "--exit-code-stage",
    "stage5_pcm=70",
    "--exit-code-stage",
    "worker_drain=71",
    "--exit-code-step-mod",
    "256",
    "--exit-code-many-threshold",
    "16",
)

EXIT_CODE_DIAG_PATTERN = re.compile(
    r"\[EXIT-CODE\]\s+"
    r"stage=(?P<stage>[\w-]+)\((?P<stage_code>\d+)\)\s+"
    r"step_raw=(?P<step_raw>\d+)\s+step_mod=(?P<step_mod>\d+)\s+"
    r"once_mask=0x(?P<once_mask>[0-9A-Fa-f]+)\s+"
    r"many_mask=0x(?P<many_mask>[0-9A-Fa-f]+)\s+"
    r"base=(?P<base>-?\d+)\s+packed=(?P<packed>\d+)\s+"
    r"last_tick_ns=(?P<last_tick>\d+)\s+exit_code=(?P<exit_code>-?\d+)"
)

STATS_HEARTBEAT_PATTERN = re.compile(
    r"\[STATS-HEARTBEAT\]\s+"
    r"tag=(?P<tag>[^\s]+)\s+"
    r"stage=(?P<stage_code>\d+)\s+"
    r"step=(?P<step>\d+)\s+"
    r"last_tick_ns=(?P<last_tick>\d+)\s+"
    r"once_mask=0x(?P<once_mask>[0-9A-Fa-f]+)\s+"
    r"many_mask=0x(?P<many_mask>[0-9A-Fa-f]+)\s+"
    r"attempt_mask=0x(?P<attempt_mask>[0-9A-Fa-f]+)\s+"
    r"idle_loops=(?P<idle_loops>-?\d+)"
)


def _parse_comma_list(
    raw: str | None,
    *,
    cast,
    allow_none: bool = False,
    none_token: str = "none",
) -> Sequence:
    if raw is None:
        return ()
    items: List = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        lowered = token.lower()
        if allow_none and lowered == none_token:
            items.append(None)
        else:
            items.append(cast(token))
    if not items:
        raise SystemExit("Override lists cannot be empty once provided.")
    return tuple(items)


def _parse_bool(token: str) -> bool:
    lowered = token.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise SystemExit(f"Invalid boolean token '{token}'. Use true/false/none values.")


@dataclass(frozen=True)
class ExhaustiveSpace:
    window_powers: Sequence[int | None]
    frames: Sequence[int | None]
    working_windows: Sequence[int | None]
    working_hops: Sequence[int | None]
    working_overlaps: Sequence[float | None]
    hop_values: Sequence[int]
    overlap_values: Sequence[float]
    prefill_values: Sequence[bool | None]
    warmup_values: Sequence[bool | None]


@dataclass(frozen=True)
class SweepCase:
    """Single invocation of the FFT division node harness."""

    label: str
    window_power: int | None = 4
    tolerance: float | None = 1e-4
    hop: int | None = None
    overlap: float | None = None
    working_window: int | None = None
    working_hop: int | None = None
    working_overlap: float | None = None
    frames: int | None = None
    wheel_prefill: bool | None = None
    warmup_passthrough: bool | None = None
    verbosity: str | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)

    def _effective_working_hop(self) -> int | None:
        if self.working_overlap is None:
            return self.working_hop
        if self.working_window is None or self.working_window <= 0:
            return self.working_hop
        hop_fraction = max(0.0, min(1.0, 1.0 - self.working_overlap))
        derived = int(round(self.working_window * hop_fraction))
        if derived <= 0:
            derived = 1
        return derived if self.working_hop is None else self.working_hop

    def build_argv(self, binary: Path) -> List[str]:
        argv = [str(binary)]
        if self.window_power is not None:
            argv.append(str(self.window_power))
            if self.tolerance is not None:
                argv.append(f"{self.tolerance:g}")
        if self.frames is not None:
            argv.extend(["--frames", str(self.frames)])
        if self.hop is not None:
            argv.extend(["--hop", str(self.hop)])
        if self.overlap is not None:
            argv.extend(["--overlap", f"{self.overlap:.4f}"])
        if self.working_window is not None:
            argv.extend(["--wwin", str(self.working_window)])
        effective_whop = self._effective_working_hop()
        if effective_whop is not None:
            argv.extend(["--whop", str(effective_whop)])
        if self.wheel_prefill is True:
            argv.append("--wheel-prefill")
        elif self.wheel_prefill is False:
            argv.append("--no-wheel-prefill")
        if self.warmup_passthrough is True:
            argv.append("--warmup-passthrough")
        elif self.warmup_passthrough is False:
            argv.append("--warmup-hold")
        verbosity = self.verbosity or DEFAULT_VERBOSITY
        if verbosity:
            argv.extend(["--verbosity", verbosity])
        argv.extend(DEFAULT_EXIT_CODE_ARGS)
        argv.extend(str(arg) for arg in self.extra_args)
        return argv

    def summary_row(self) -> List[str]:
        def _fmt_bool(value: bool | None) -> str:
            if value is None:
                return "-"
            return "Y" if value else "N"

        def _fmt_float(value: float | None) -> str:
            if value is None:
                return "-"
            return f"{value:.2f}"

        effective_whop = self._effective_working_hop()

        return [
            self.label,
            str(self.window_power) if self.window_power is not None else "-",
            str(self.hop) if self.hop is not None else "-",
            _fmt_float(self.overlap),
            str(self.working_window) if self.working_window is not None else "-",
            str(effective_whop) if effective_whop is not None else "-",
            _fmt_float(self.working_overlap),
            str(self.frames) if self.frames is not None else "-",
            _fmt_bool(self.wheel_prefill),
            _fmt_bool(self.warmup_passthrough),
        ]


DEFAULT_CASES: Sequence[SweepCase] = (
    SweepCase(
        label="baseline",
        window_power=None,
        tolerance=None,
    ),
    SweepCase(
        label="overlap_low",
        window_power=4,
        overlap=0.10,
        working_window=16,
        working_hop=2,
        wheel_prefill=False,
        warmup_passthrough=False,
    ),
    SweepCase(
        label="overlap_high",
        window_power=5,
        overlap=0.80,
        working_window=24,
        working_hop=2,
        wheel_prefill=True,
        warmup_passthrough=True,
    ),
    SweepCase(
        label="hop_override",
        window_power=6,
        hop=2,
        working_window=32,
        working_hop=4,
        wheel_prefill=True,
        warmup_passthrough=False,
    ),
    SweepCase(
        label="small_window",
        window_power=2,
        working_window=4,
        working_hop=1,
        wheel_prefill=False,
        warmup_passthrough=True,
    ),
    SweepCase(
        label="dense_wheel",
        window_power=7,
        overlap=0.60,
        working_window=48,
        working_hop=3,
        wheel_prefill=True,
        warmup_passthrough=True,
        verbosity="trace",
    ),
    SweepCase(
        label="grid_prefill",
        window_power=5,
        overlap=0.33,
        working_window=32,
        working_hop=2,
        wheel_prefill=True,
        warmup_passthrough=False,
    ),
    SweepCase(
        label="grid_passthrough",
        window_power=6,
        hop=3,
        working_window=24,
        working_hop=3,
        wheel_prefill=False,
        warmup_passthrough=True,
    ),
)


WINDOW_POWER_SWEEP = [None, 2, 4, 6, 8, 9, 10, 11]
FRAMES_SWEEP = [None, 16, 64, 256, 1024, 2048, 4096, 8092, 8192]
WORKING_WINDOW_SWEEP = [None, 1, 4, 16, 64, 128, 256, 512]
WORKING_HOP_SWEEP = [None, 1, 2, 4, 8]
WORKING_OVERLAP_SWEEP = [None, 0.25, 0.5, 0.75]
HOP_ONLY_VALUES = [1, 2, 3, 4]
OVERLAP_ONLY_VALUES = [0.10, 0.33, 0.50, 0.80]
PREFILL_SWEEP = [None, True, False]
WARMUP_SWEEP = [None, True, False]


DEFAULT_EXHAUSTIVE_SPACE = ExhaustiveSpace(
    window_powers=tuple(WINDOW_POWER_SWEEP),
    frames=tuple(FRAMES_SWEEP),
    working_windows=tuple(WORKING_WINDOW_SWEEP),
    working_hops=tuple(WORKING_HOP_SWEEP),
    working_overlaps=tuple(WORKING_OVERLAP_SWEEP),
    hop_values=tuple(HOP_ONLY_VALUES),
    overlap_values=tuple(OVERLAP_ONLY_VALUES),
    prefill_values=tuple(PREFILL_SWEEP),
    warmup_values=tuple(WARMUP_SWEEP),
)


PRESET_SPACES = {
    "small-windows-half": ExhaustiveSpace(
        window_powers=(2, 3, 4, 5),
        frames=(None, 64, 256),
        working_windows=(1, 4, 16),
        working_hops=(None,),
        working_overlaps=(0.5,),
        hop_values=(1, 2),
        overlap_values=(0.5,),
        prefill_values=tuple(PREFILL_SWEEP),
        warmup_values=tuple(WARMUP_SWEEP),
    ),
    "large-windows-half": ExhaustiveSpace(
        window_powers=(8, 9, 10, 11),
        frames=(512, 2048, 8192),
        working_windows=(64, 128, 256, 512),
        working_hops=(None,),
        working_overlaps=(0.5,),
        hop_values=(2, 4, 8),
        overlap_values=(0.5,),
        prefill_values=tuple(PREFILL_SWEEP),
        warmup_values=tuple(WARMUP_SWEEP),
    ),
    "sparse-diverse": ExhaustiveSpace(
        window_powers=(None, 3, 6, 9),
        frames=(None, 64, 512, 2048),
        working_windows=(None, 16, 128, 512),
        working_hops=(None, 2, 4),
        working_overlaps=(None, 0.25, 0.75),
        hop_values=(1, 3, 4),
        overlap_values=(0.1, 0.5, 0.8),
        prefill_values=tuple(PREFILL_SWEEP),
        warmup_values=tuple(WARMUP_SWEEP),
    ),
}


def _resolve_axis(
    raw: str | None,
    *,
    base: Sequence,
    cast,
    allow_none: bool,
) -> Sequence:
    if raw is None:
        return tuple(base)
    return _parse_comma_list(raw, cast=cast, allow_none=allow_none)


def resolve_exhaustive_space(args: argparse.Namespace) -> ExhaustiveSpace:
    base_space = (
        PRESET_SPACES.get(args.sweep_preset)
        if args.sweep_preset
        else DEFAULT_EXHAUSTIVE_SPACE
    )
    window_powers = _resolve_axis(
        args.window_powers, base=base_space.window_powers, cast=int, allow_none=True
    )
    frames = _resolve_axis(
        args.frames_values, base=base_space.frames, cast=int, allow_none=True
    )
    working_windows = _resolve_axis(
        args.working_window_values,
        base=base_space.working_windows,
        cast=int,
        allow_none=True,
    )
    working_hops = _resolve_axis(
        args.working_hop_values, base=base_space.working_hops, cast=int, allow_none=True
    )
    working_overlaps = _resolve_axis(
        args.working_overlap_values,
        base=base_space.working_overlaps,
        cast=float,
        allow_none=True,
    )
    hop_values_raw = _resolve_axis(
        args.hop_values, base=base_space.hop_values, cast=int, allow_none=True
    )
    hop_values = tuple(value for value in hop_values_raw if value is not None)
    overlap_values = _resolve_axis(
        args.overlap_values,
        base=base_space.overlap_values,
        cast=float,
        allow_none=False,
    )
    prefill_values = _resolve_axis(
        args.prefill_values,
        base=base_space.prefill_values,
        cast=_parse_bool,
        allow_none=True,
    )
    warmup_values = _resolve_axis(
        args.warmup_values,
        base=base_space.warmup_values,
        cast=_parse_bool,
        allow_none=True,
    )

    return ExhaustiveSpace(
        window_powers=window_powers,
        frames=frames,
        working_windows=working_windows,
        working_hops=working_hops,
        working_overlaps=working_overlaps,
        hop_values=hop_values,
        overlap_values=overlap_values,
        prefill_values=prefill_values,
        warmup_values=warmup_values,
    )


def discover_default_binary(repo_root: Path) -> Path:
    """Heuristically locate the compiled test binary."""

    build_dir = repo_root / "build"
    candidates: List[Path] = []
    if platform.system() == "Windows":
        candidates.extend(
            [
                build_dir / "Debug" / "test_fft_division_node.exe",
                build_dir / "RelWithDebInfo" / "test_fft_division_node.exe",
                build_dir / "Release" / "test_fft_division_node.exe",
            ]
        )
    else:
        candidates.extend(
            [
                build_dir / "test_fft_division_node",
                build_dir / "Debug" / "test_fft_division_node",
            ]
        )
    candidates.append(build_dir / "test_fft_division_node.exe")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback to first guess even if missing so argparse can report a helpful message later
    return candidates[0]


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_binary = discover_default_binary(repo_root)
    parser.add_argument(
        "--binary",
        type=Path,
        default=default_binary,
        help=f"Path to test_fft_division_node binary (default: {default_binary})",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case labels to run (defaults to all).",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Abort the sweep after the first failing configuration.",
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Stream stdout/stderr directly instead of capturing for the summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-case timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--timeout-headroom",
        type=float,
        default=1.75,
        help=(
            "Multiplier applied to the rolling average of explicit runtimes when "
            "deriving the adaptive timeout ceiling (default: 1.75)."
        ),
    )
    parser.add_argument(
        "--retry-cap-multiplier",
        type=float,
        default=1.25,
        help=(
            "Factor applied to the adaptive timeout cap whenever active timeouts are "
            "re-queued for another sweep pass (default: 1.25)."
        ),
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help=(
            "Total patience budget in wall-clock seconds shared across the sweep. "
            "When set, each case continuously rebalances against whatever "
            "patience remains so late cases inherit any slack that earlier "
            "runs returned."
        ),
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run the exhaustive permutation table instead of the curated defaults.",
    )
    parser.add_argument(
        "--sweep-preset",
        type=str,
        choices=[
            "small-windows-half",
            "large-windows-half",
            "sparse-diverse",
        ],
        default=None,
        help=(
            "Optional shorthand for curated exhaustive subsets (overrides still apply). "
            "Choices: small-windows-half, large-windows-half, sparse-diverse."
        ),
    )
    parser.add_argument(
        "--window-powers",
        type=str,
        default=None,
        help="Comma-separated window powers (use 'none' to include the default pass-through).",
    )
    parser.add_argument(
        "--frames-values",
        type=str,
        default=None,
        help="Comma-separated frame counts (use 'none' for default / unset).",
    )
    parser.add_argument(
        "--working-window-values",
        type=str,
        default=None,
        help="Comma-separated working window sizes (use 'none' for unset).",
    )
    parser.add_argument(
        "--working-hop-values",
        type=str,
        default=None,
        help="Comma-separated working hop overrides (use 'none' for unset).",
    )
    parser.add_argument(
        "--working-overlap-values",
        type=str,
        default=None,
        help="Comma-separated working overlap fractions (0..1, use 'none' for unset).",
    )
    parser.add_argument(
        "--hop-values",
        type=str,
        default=None,
        help="Comma-separated hop overrides (integers, use 'none' to disable explicit hops).",
    )
    parser.add_argument(
        "--overlap-values",
        type=str,
        default=None,
        help="Comma-separated overlap fractions (0..1).",
    )
    parser.add_argument(
        "--prefill-values",
        type=str,
        default=None,
        help="Comma-separated wheel prefill toggles (true,false,none).",
    )
    parser.add_argument(
        "--warmup-values",
        type=str,
        default=None,
        help="Comma-separated warmup passthrough toggles (true,false,none).",
    )
    parser.add_argument(
        "--patience-exempt-timeouts",
        type=str,
        default="idle",
        help=(
            "Comma-separated timeout buckets that should not burn patience. "
            "Choices: idle,active,unknown."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="output/fft_division_sweep.json",
        help=(
            "Path to write the structured JSON summary (default: output/fft_division_sweep.json). "
            "Pass an empty string to skip writing JSON output."
        ),
    )
    parser.add_argument(
        "--case-order-seed",
        type=int,
        default=0,
        help="Seed controlling the adaptive case ordering jitter (default: 0).",
    )
    parser.add_argument(
        "--disable-adaptive-ordering",
        action="store_true",
        help="Process cases in the declared order instead of entropy-prioritized scheduling.",
    )
    parser.add_argument(
        "--entropy-reprioritize-interval",
        type=int,
        default=8,
        help=(
            "Completed-case interval between adaptive reprioritizations (default: 8). "
            "Set to 1 to recompute priorities after every result as before."
        ),
    )
    parser.add_argument(
        "--min-execution-time",
        type=float,
        default=0.0,
        help="Floor (seconds) applied to every case timeout budget before scheduling (default: 0).",
    )
    parser.add_argument(
        "--patience-overbook-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the automatic reservation headroom used when --worker-count > 1. "
            "Headroom allows dispatching extra cases while existing workers still hold reservations. "
            "Set to 0 to disable overbooking (default: 1)."
        ),
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help=(
            "Number of concurrent workers used to run sweep cases (default: 1). "
            "Increase to exercise multiple native test instances in parallel."
        ),
    )
    parser.add_argument(
        "--live-dashboard",
        action="store_true",
        help=(
            "Silence per-case stdout/stderr dumps and render a compact top/bottom pending "
            "view with live sweep statistics."
        ),
    )
    parser.add_argument(
        "--dashboard-topk",
        type=int,
        default=5,
        help=(
            "How many entries appear in the top/bottom pending columns when --live-dashboard is enabled "
            "(default: 5)."
        ),
    )
    return parser.parse_args()


def select_cases(filters: Iterable[str]) -> List[SweepCase]:
    wanted = {label.strip() for label in filters if label.strip()}
    if not wanted:
        return list(DEFAULT_CASES)
    case_map = {case.label: case for case in DEFAULT_CASES}
    missing = sorted(label for label in wanted if label not in case_map)
    if missing:
        raise SystemExit(f"Unknown case labels: {', '.join(missing)}")
    return [case_map[label] for label in wanted]


def build_exhaustive_cases(space: ExhaustiveSpace) -> List[SweepCase]:
    cases: List[SweepCase] = []
    hop_modes: List[Tuple[str, float | int | None]] = [("none", None)]
    hop_modes.extend(("hop", value) for value in space.hop_values)
    hop_modes.extend(("overlap", value) for value in space.overlap_values)
    case_id = 0
    for window_power in space.window_powers:
        for frames in space.frames:
            for mode, value in hop_modes:
                hop_value = value if mode == "hop" else None
                overlap_value = value if mode == "overlap" else None
                for working_window in space.working_windows:
                    for working_hop in space.working_hops:
                        for working_overlap in space.working_overlaps:
                            if working_overlap is not None:
                                if working_window is None:
                                    continue
                                if working_hop is not None:
                                    continue
                            for wheel_prefill in space.prefill_values:
                                for warmup in space.warmup_values:
                                    case_id += 1
                                    label = f"exh_{case_id:05d}"
                                    cases.append(
                                        SweepCase(
                                            label=label,
                                            window_power=window_power,
                                            tolerance=1e-4,
                                            hop=int(hop_value) if isinstance(hop_value, int) else None,
                                            overlap=float(overlap_value) if isinstance(overlap_value, float) else None,
                                            working_window=working_window,
                                            working_hop=working_hop,
                                            working_overlap=working_overlap,
                                            frames=frames,
                                            wheel_prefill=wheel_prefill,
                                            warmup_passthrough=warmup,
                                        )
                                    )
    return cases


@dataclass
class TimeoutInference:
    bucket: str
    reason: str
    evidence: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"timeout:{self.bucket}"


def _parse_timeout_bucket_list(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    invalid = tokens.difference(KNOWN_TIMEOUT_BUCKETS)
    if invalid:
        choices = ", ".join(KNOWN_TIMEOUT_BUCKETS)
        raise SystemExit(
            "Invalid timeout bucket(s): " + ", ".join(sorted(invalid)) + f". Choices: {choices}."
        )
    return tokens


AMP_FFTDIV_STAGE_IDLE_CODE = 0


def _infer_timeout_from_exit_metadata(result: CaseResult) -> TimeoutInference | None:
    has_metadata = any(
        field is not None
        for field in (
            result.exit_stage_code,
            result.exit_once_mask,
            result.exit_many_mask,
            result.exit_last_tick_ns,
        )
    )
    if not has_metadata:
        return None
    evidence: List[str] = []
    if result.exit_stage_code is not None:
        evidence.append(f"stage_code={result.exit_stage_code}")
    if result.exit_stage:
        evidence.append(f"stage_label={result.exit_stage}")
    if result.exit_once_mask is not None:
        evidence.append(f"once_mask=0x{result.exit_once_mask:02X}")
    if result.exit_many_mask is not None:
        evidence.append(f"many_mask=0x{result.exit_many_mask:02X}")
    if result.exit_last_tick_ns is not None:
        evidence.append(f"last_tick_ns={result.exit_last_tick_ns}")
    activity_mask = (result.exit_once_mask or 0) | (result.exit_many_mask or 0)
    if activity_mask == 0:
        return TimeoutInference(
            "idle",
            "Stage counters never recorded work while logging was enabled",
            tuple(evidence),
        )
    idle_stage = False
    if result.exit_stage_code is not None and result.exit_stage_code == AMP_FFTDIV_STAGE_IDLE_CODE:
        idle_stage = True
    elif result.exit_stage and result.exit_stage.lower().startswith("idle"):
        idle_stage = True
    if idle_stage:
        return TimeoutInference(
            "idle",
            "Pipeline snapshot captured while worker reported an idle stage",
            tuple(evidence),
        )
    if result.exit_last_tick_ns is None or result.exit_last_tick_ns == 0:
        return TimeoutInference(
            "idle",
            "No valid pipeline timestamp was captured at timeout",
            tuple(evidence),
        )
    tick_seconds = float(result.exit_last_tick_ns) * 1e-9
    return TimeoutInference(
        "active",
        (
            "Stage counters recorded work and a recent pipeline timestamp "
            f"({tick_seconds:.6f}s) prior to the timeout"
        ),
        tuple(evidence),
    )


def _infer_timeout_from_heartbeat(result: CaseResult) -> TimeoutInference | None:
    if result.heartbeat_last_tick_ns is None:
        return None
    evidence: List[str] = []
    if result.heartbeat_tag:
        evidence.append(f"tag={result.heartbeat_tag}")
    if result.heartbeat_stage_code is not None:
        evidence.append(f"stage_code={result.heartbeat_stage_code}")
    if result.heartbeat_once_mask is not None:
        evidence.append(f"once_mask=0x{result.heartbeat_once_mask:02X}")
    if result.heartbeat_many_mask is not None:
        evidence.append(f"many_mask=0x{result.heartbeat_many_mask:02X}")
    if result.heartbeat_attempt_mask is not None:
        evidence.append(f"attempt_mask=0x{result.heartbeat_attempt_mask:02X}")
    if result.heartbeat_idle_loops is not None:
        evidence.append(f"idle_loops={result.heartbeat_idle_loops}")
    once_mask = result.heartbeat_once_mask or 0
    many_mask = result.heartbeat_many_mask or 0
    stage_code = result.heartbeat_stage_code
    if once_mask == 0 and many_mask == 0:
        return TimeoutInference(
            "idle",
            "Heartbeat indicated zero work recorded across all stages",
            tuple(evidence),
        )
    if stage_code is not None and stage_code == AMP_FFTDIV_STAGE_IDLE_CODE:
        return TimeoutInference(
            "idle",
            "Heartbeat snapshot captured while worker reported the idle stage",
            tuple(evidence),
        )
    tick_seconds = float(result.heartbeat_last_tick_ns) * 1e-9
    reason = (
        "Heartbeat recorded stage activity with pipeline timestamp"
        f" {tick_seconds:.6f}s before timeout"
    )
    return TimeoutInference("active", reason, tuple(evidence))


def infer_timeout_behavior(result: CaseResult) -> TimeoutInference:
    metadata_inference = _infer_timeout_from_exit_metadata(result)
    if metadata_inference is not None:
        return metadata_inference
    heartbeat_inference = _infer_timeout_from_heartbeat(result)
    if heartbeat_inference is not None:
        return heartbeat_inference
    if result.stdout or result.stderr:
        return TimeoutInference(
            "unknown",
            "Captured diagnostics never emitted a STATS heartbeat before shutdown",
            (),
        )
    return TimeoutInference(
        "unknown",
        "Logging never produced diagnostics before timeout",
        (),
    )


@dataclass
class CaseResult:
    case: SweepCase
    returncode: int
    duration: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    allocated_timeout: float | None = None
    timeout_inference: TimeoutInference | None = None
    patience_charge: float | None = None
    notes_tokens: Tuple[str, ...] = field(default_factory=tuple)
    result_type: str | None = None
    finished_flag: int | None = None
    quality_flag: int | None = None
    flag_label: str | None = None
    exit_stage: str | None = None
    exit_stage_code: int | None = None
    exit_once_mask: int | None = None
    exit_many_mask: int | None = None
    exit_step_raw: int | None = None
    exit_step_mod: int | None = None
    exit_packed_bits: int | None = None
    exit_base_code: int | None = None
    exit_last_tick_ns: int | None = None
    heartbeat_tag: str | None = None
    heartbeat_stage_code: int | None = None
    heartbeat_step: int | None = None
    heartbeat_once_mask: int | None = None
    heartbeat_many_mask: int | None = None
    heartbeat_attempt_mask: int | None = None
    heartbeat_last_tick_ns: int | None = None
    heartbeat_idle_loops: int | None = None

    @property
    def ok(self) -> bool:
        return (not self.timed_out) and self.returncode == 0

    @property
    def status(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        return "PASS" if self.ok else "FAIL"




def _extract_exit_code_metadata(stdout: str, stderr: str) -> dict | None:
    payload = "\n".join(part for part in (stdout, stderr) if part)
    if not payload:
        return None
    match = None
    for candidate in EXIT_CODE_DIAG_PATTERN.finditer(payload):
        match = candidate
    if match is None:
        return None
    groups = match.groupdict()
    try:
        return {
            "stage": groups["stage"],
            "stage_code": int(groups["stage_code"]),
            "step_raw": int(groups["step_raw"]),
            "step_mod": int(groups["step_mod"]),
            "once_mask": int(groups["once_mask"], 16),
            "many_mask": int(groups["many_mask"], 16),
            "base_code": int(groups["base"]),
            "packed_bits": int(groups["packed"]),
            "last_tick_ns": int(groups["last_tick"]),
            "exit_code": int(groups["exit_code"]),
        }
    except (ValueError, KeyError):
        return None


def _extract_latest_heartbeat(stdout: str, stderr: str) -> dict | None:
    payload = "\n".join(part for part in (stdout, stderr) if part)
    if not payload:
        return None
    last_match = None
    for candidate in STATS_HEARTBEAT_PATTERN.finditer(payload):
        last_match = candidate
    if last_match is None:
        return None
    groups = last_match.groupdict()
    try:
        return {
            "tag": groups.get("tag", ""),
            "stage_code": int(groups["stage_code"]),
            "step": int(groups["step"]),
            "last_tick_ns": int(groups["last_tick"]),
            "once_mask": int(groups["once_mask"], 16),
            "many_mask": int(groups["many_mask"], 16),
            "attempt_mask": int(groups["attempt_mask"], 16),
            "idle_loops": int(groups["idle_loops"]),
        }
    except (ValueError, KeyError):
        return None


def attach_exit_code_metadata(result: CaseResult) -> None:
    exit_metadata = _extract_exit_code_metadata(result.stdout, result.stderr)
    heartbeat = _extract_latest_heartbeat(result.stdout, result.stderr)
    if exit_metadata:
        result.exit_stage = exit_metadata.get("stage")
        result.exit_stage_code = exit_metadata.get("stage_code")
        result.exit_once_mask = exit_metadata.get("once_mask")
        result.exit_many_mask = exit_metadata.get("many_mask")
        result.exit_step_raw = exit_metadata.get("step_raw")
        result.exit_step_mod = exit_metadata.get("step_mod")
        result.exit_packed_bits = exit_metadata.get("packed_bits")
        result.exit_base_code = exit_metadata.get("base_code")
        result.exit_last_tick_ns = exit_metadata.get("last_tick_ns")
    if heartbeat:
        result.heartbeat_tag = heartbeat.get("tag")
        result.heartbeat_stage_code = heartbeat.get("stage_code")
        result.heartbeat_step = heartbeat.get("step")
        result.heartbeat_once_mask = heartbeat.get("once_mask")
        result.heartbeat_many_mask = heartbeat.get("many_mask")
        result.heartbeat_attempt_mask = heartbeat.get("attempt_mask")
        result.heartbeat_last_tick_ns = heartbeat.get("last_tick_ns")
        result.heartbeat_idle_loops = heartbeat.get("idle_loops")


def annotate_case_result(result: CaseResult) -> None:
    note_tokens: List[str]
    if result.notes_tokens:
        note_tokens = list(result.notes_tokens)
    else:
        note_tokens = []
        if result.timed_out:
            note_tokens.append("timeout")
            if result.timeout_inference:
                note_tokens.append(result.timeout_inference.label)
        elif not result.ok:
            note_tokens.append(f"rc={result.returncode}")
        if result.patience_charge is not None:
            note_tokens.append(f"patience={result.patience_charge:.2f}")
        if not note_tokens:
            note_tokens.append("none")
        result.notes_tokens = tuple(note_tokens)
    def _append_note(token: str) -> None:
        if token not in note_tokens:
            note_tokens.append(token)
    if result.exit_stage:
        _append_note(f"exit_stage={result.exit_stage}")
    elif result.heartbeat_stage_code is not None:
        _append_note(f"hb_stage={result.heartbeat_stage_code}")
    if result.exit_once_mask is not None:
        _append_note(f"once=0x{result.exit_once_mask:02X}")
    if result.exit_many_mask is not None:
        _append_note(f"many=0x{result.exit_many_mask:02X}")
    if result.exit_step_mod is not None:
        _append_note(f"step={result.exit_step_mod}")
    if result.heartbeat_idle_loops is not None and result.heartbeat_idle_loops > 0:
        _append_note(f"hb_idle={result.heartbeat_idle_loops}")
    classification_tokens = [token for token in note_tokens if not token.startswith("patience=")]
    if not classification_tokens:
        classification_tokens = ["none"]
    result.result_type = "|".join([result.status] + classification_tokens)
    finished_flag = 1 if not result.timed_out else 0
    if result.status == "PASS":
        quality_flag = 1
    elif result.status == "FAIL":
        quality_flag = 0
    else:
        bucket = result.timeout_inference.bucket if result.timeout_inference else None
        quality_flag = 1 if bucket == "active" else 0
    result.finished_flag = finished_flag
    result.quality_flag = quality_flag
    result.flag_label = FLAG_COMBO_LABELS.get((finished_flag, quality_flag), "unknown_flag_combo")
    result.notes_tokens = tuple(note_tokens)


def is_active_timeout(result: CaseResult) -> bool:
    """Return True when diagnostics show the timeout occurred during active work."""
    if not result.timed_out:
        return False
    bucket = result.timeout_inference.bucket if result.timeout_inference else None
    if bucket == "active":
        return True
    if result.flag_label == "timeout_active":
        return True
    activity_mask = 0
    activity_mask |= result.exit_once_mask or 0
    activity_mask |= result.exit_many_mask or 0
    activity_mask |= result.heartbeat_once_mask or 0
    activity_mask |= result.heartbeat_many_mask or 0
    activity_mask |= result.heartbeat_attempt_mask or 0
    if activity_mask:
        return True
    if (result.exit_last_tick_ns and result.exit_last_tick_ns > 0) or (
        result.heartbeat_last_tick_ns and result.heartbeat_last_tick_ns > 0
    ):
        return True
    return False


def serialize_case_result(result: CaseResult) -> dict:
    case = result.case
    inference = result.timeout_inference
    timeout_bucket = inference.bucket if inference else None
    timeout_reason = inference.reason if inference else None
    timeout_evidence = list(inference.evidence) if inference else []
    return {
        "label": case.label,
        "window_power": case.window_power,
        "tolerance": case.tolerance,
        "hop": case.hop,
        "overlap": case.overlap,
        "working_window": case.working_window,
        "working_hop": case.working_hop,
        "working_overlap": case.working_overlap,
        "frames": case.frames,
        "wheel_prefill": case.wheel_prefill,
        "warmup_passthrough": case.warmup_passthrough,
        "extra_args": list(case.extra_args),
        "duration": result.duration,
        "allocated_timeout": result.allocated_timeout,
        "patience_charge": result.patience_charge,
        "timed_out": result.timed_out,
        "timeout_bucket": timeout_bucket,
        "timeout_reason": timeout_reason,
        "timeout_evidence": timeout_evidence,
        "status": result.status,
        "ok": result.ok,
        "notes_tokens": list(result.notes_tokens),
        "result_type": result.result_type,
        "finished_flag": result.finished_flag,
        "quality_flag": result.quality_flag,
        "flag_label": result.flag_label,
        "returncode": result.returncode,
        "exit_stage": result.exit_stage,
        "exit_stage_code": result.exit_stage_code,
        "exit_once_mask": result.exit_once_mask,
        "exit_many_mask": result.exit_many_mask,
        "exit_step_raw": result.exit_step_raw,
        "exit_step_mod": result.exit_step_mod,
        "exit_packed_bits": result.exit_packed_bits,
        "exit_base_code": result.exit_base_code,
        "exit_last_tick_ns": result.exit_last_tick_ns,
        "heartbeat_tag": result.heartbeat_tag,
        "heartbeat_stage_code": result.heartbeat_stage_code,
        "heartbeat_step": result.heartbeat_step,
        "heartbeat_once_mask": result.heartbeat_once_mask,
        "heartbeat_many_mask": result.heartbeat_many_mask,
        "heartbeat_attempt_mask": result.heartbeat_attempt_mask,
        "heartbeat_last_tick_ns": result.heartbeat_last_tick_ns,
        "heartbeat_idle_loops": result.heartbeat_idle_loops,
    }


@dataclass
class AllocationSnapshot:
    applied_timeout: float
    base_timeout: float
    share_ceiling: float | None = None
    dynamic_cap: float | None = None
    last_completed_result: CaseResult | None = None


@dataclass
class ActiveCaseEntry:
    case: SweepCase
    future: Future
    allocation: AllocationSnapshot
    reserved_budget: float


class LiveDashboard:
    def __init__(self, top_k: int, *, enabled: bool) -> None:
        self.top_k = max(0, top_k)
        self.enabled = enabled
        self._isatty = sys.stdout.isatty()
        self._warned_non_tty = False

    def update(
        self,
        *,
        top_cases: Sequence[Tuple[SweepCase, float | None]],
        bottom_cases: Sequence[Tuple[SweepCase, float | None]],
        completed: int,
        total_cases: int,
        pass_count: int,
        active_timeout_count: int,
        fail_count: int,
        idle_timeout_count: int,
        pending_count: int,
        patience_enabled: bool,
        patience_remaining: float | None,
        patience_reserved: float | None,
        patience_headroom: float | None,
        patience_savings: float | None,
        min_exec_time: float,
        dynamic_cap: float | None,
        last_case: CaseResult | None,
        allocation: AllocationSnapshot | None,
        status_line: str | None,
    ) -> None:
        if not self.enabled:
            return
        block = self._compose_block(
            top_cases=top_cases,
            bottom_cases=bottom_cases,
            completed=completed,
            total_cases=total_cases,
            pass_count=pass_count,
            active_timeout_count=active_timeout_count,
            fail_count=fail_count,
            idle_timeout_count=idle_timeout_count,
            pending_count=pending_count,
            patience_enabled=patience_enabled,
            patience_remaining=patience_remaining,
            patience_reserved=patience_reserved,
            patience_headroom=patience_headroom,
            patience_savings=patience_savings,
            min_exec_time=min_exec_time,
            dynamic_cap=dynamic_cap,
            last_case=last_case,
            allocation=allocation,
            status_line=status_line,
        )
        if self._isatty:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(block)
            sys.stdout.flush()
        else:
            if not self._warned_non_tty:
                print("[live-dashboard] stdout is not a TTY; falling back to repeated snapshots.")
                self._warned_non_tty = True
            print(block)

    def close(self) -> None:
        if not self.enabled or not self._isatty:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _compose_block(
        self,
        *,
        top_cases: Sequence[Tuple[SweepCase, float | None]],
        bottom_cases: Sequence[Tuple[SweepCase, float | None]],
        completed: int,
        total_cases: int,
        pass_count: int,
        active_timeout_count: int,
        fail_count: int,
        idle_timeout_count: int,
        pending_count: int,
        patience_enabled: bool,
        patience_remaining: float | None,
        patience_reserved: float | None,
        patience_headroom: float | None,
        patience_savings: float | None,
        min_exec_time: float,
        dynamic_cap: float | None,
        last_case: CaseResult | None,
        allocation: AllocationSnapshot | None,
        status_line: str | None,
    ) -> str:
        term_width = shutil.get_terminal_size(fallback=(120, 40)).columns
        col_width = max(32, (term_width - 6) // 3)
        top_lines = self._column_lines("Top Pending", top_cases)
        bottom_lines = self._column_lines("Bottom Pending", bottom_cases)
        stats_lines = self._stats_lines(
            completed=completed,
            total_cases=total_cases,
            pass_count=pass_count,
            active_timeout_count=active_timeout_count,
            fail_count=fail_count,
            idle_timeout_count=idle_timeout_count,
            pending_count=pending_count,
            patience_enabled=patience_enabled,
            patience_remaining=patience_remaining,
            patience_reserved=patience_reserved,
            patience_headroom=patience_headroom,
            patience_savings=patience_savings,
            min_exec_time=min_exec_time,
            dynamic_cap=dynamic_cap,
            last_case=last_case,
            allocation=allocation,
            status_line=status_line,
        )
        max_rows = max(len(top_lines), len(bottom_lines), len(stats_lines))
        lines: List[str] = []
        for idx in range(max_rows):
            top_line = top_lines[idx] if idx < len(top_lines) else ""
            bottom_line = bottom_lines[idx] if idx < len(bottom_lines) else ""
            stats_line = stats_lines[idx] if idx < len(stats_lines) else ""
            lines.append(
                f"{top_line:<{col_width}} | {bottom_line:<{col_width}} | {stats_line:<{col_width}}"
            )
        header = f"{'=' * (col_width * 3 + 6)}"
        return "\n".join([header, *lines, header])

    def _column_lines(
        self,
        title: str,
        entries: Sequence[Tuple[SweepCase, float | None]],
    ) -> List[str]:
        lines = [title]
        if not entries:
            lines.append("(none)")
            return lines
        for idx, (case, priority) in enumerate(entries, start=1):
            prefix = f"{idx:>2}. "
            lines.append(prefix + self._case_brief(case, priority))
        return lines

    def _case_brief(self, case: SweepCase, priority: float | None) -> str:
        def _fmt(value):
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

        brief = (
            f"{case.label} win={_fmt(case.window_power)} hop={_fmt(case.hop)} "
            f"ov={_fmt(case.overlap)} ww={_fmt(case.working_window)}"
        )
        if priority is not None:
            brief += f" p={priority:.2f}"
        return brief

    def _stats_lines(
        self,
        *,
        completed: int,
        total_cases: int,
        pass_count: int,
        active_timeout_count: int,
        fail_count: int,
        idle_timeout_count: int,
        pending_count: int,
        patience_enabled: bool,
        patience_remaining: float | None,
        patience_reserved: float | None,
        patience_headroom: float | None,
        patience_savings: float | None,
        min_exec_time: float,
        dynamic_cap: float | None,
        last_case: CaseResult | None,
        allocation: AllocationSnapshot | None,
        status_line: str | None,
    ) -> List[str]:
        lines = ["Sweep Stats"]
        lines.append(f"Completed: {completed}/{total_cases}")
        lines.append(
            "Pass/ActiveTO/Fail/IdleTO: "
            f"{pass_count}/{active_timeout_count}/{fail_count}/{idle_timeout_count}"
        )
        lines.append(f"Pending: {pending_count}")
        if patience_enabled:
            if patience_remaining is None:
                lines.append("Patience: unknown")
            else:
                lines.append(f"Patience left: {patience_remaining:.2f}s")
                if patience_reserved is not None and patience_reserved > 0:
                    lines.append(f"Reserved: {patience_reserved:.2f}s")
                if patience_headroom is not None and patience_headroom > 0:
                    lines.append(f"Headroom: {patience_headroom:.2f}s")
                if patience_savings is not None and patience_savings > 0:
                    lines.append(f"Savings: {patience_savings:.2f}s")
        else:
            lines.append("Patience: disabled")
        if dynamic_cap is not None:
            lines.append(f"Dynamic cap: {dynamic_cap:.2f}s")
        else:
            lines.append("Dynamic cap: pending")
        lines.append(f"Min exec: {min_exec_time:.2f}s")
        if allocation is not None:
            lines.append(
                f"Applied timeout: {allocation.applied_timeout:.2f}s (base {allocation.base_timeout:.2f}s)"
            )
            if allocation.share_ceiling is not None:
                lines.append(f"Share ceiling: {allocation.share_ceiling:.2f}s")
            if allocation.dynamic_cap is not None:
                lines.append(f"Cap after data: {allocation.dynamic_cap:.2f}s")
        if last_case is not None:
            lines.append(
                f"Last: {last_case.case.label} -> {last_case.status} ({last_case.duration:.2f}s)"
            )
        if status_line:
            lines.append(f"Status: {status_line}")
        return lines

class EntropyTracker:
    def __init__(self) -> None:
        self._global_counts: Counter[str] = Counter()
        self._axis_counts: defaultdict[Tuple[str, str], Counter[str]] = defaultdict(Counter)

    def observe(self, case: SweepCase, flag_label: str) -> float:
        normalized_label = flag_label or "unknown_flag_combo"
        for key in self._case_axes(case):
            bucket = self._axis_counts[key]
            bucket[normalized_label] += 1
        self._global_counts[normalized_label] += 1
        return self.case_entropy(case)

    def case_entropy(self, case: SweepCase) -> float:
        entropies: List[float] = [self._normalized_entropy(self._global_counts)]
        for key in self._case_axes(case):
            entropies.append(self._normalized_entropy(self._axis_counts.get(key, Counter())))
        return sum(entropies) / len(entropies)

    def _case_axes(self, case: SweepCase) -> Tuple[Tuple[str, str], ...]:
        return (
            ("label", case.label),
            ("window_power", self._norm(case.window_power)),
            ("frames", self._norm(case.frames)),
            ("working_window", self._norm(case.working_window)),
            ("working_hop", self._norm(case.working_hop)),
            ("working_overlap", self._norm(case.working_overlap)),
            ("hop", self._norm(case.hop)),
            ("overlap", self._norm(case.overlap)),
            ("wheel_prefill", self._norm(case.wheel_prefill)),
            ("warmup", self._norm(case.warmup_passthrough)),
        )

    @staticmethod
    def _norm(value, default: str = "default") -> str:
        if value is None:
            return default
        if isinstance(value, bool):
            return "Y" if value else "N"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    @staticmethod
    def _normalized_entropy(counts: Counter[str]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 1.0
        entropy = 0.0
        for tally in counts.values():
            if tally <= 0:
                continue
            p = tally / total
            entropy -= p * math.log2(p)
        if FLAG_LABEL_ENTROPY_DENOM <= 0:
            return float(entropy)
        normalized = entropy / FLAG_LABEL_ENTROPY_DENOM
        return max(0.0, min(1.0, normalized))


class SimpleEntropyNetwork:
    def __init__(self, tracker: EntropyTracker, *, rng: random.Random | None = None) -> None:
        self._tracker = tracker
        self._rng = rng or random.Random()
        self._recent_entropy = 0.5
        self._bias = 0.0
        self._weight_entropy = 1.25
        self._feature_weights = {
            "window_power": 0.35,
            "working_window": 0.25,
            "working_hop": 0.15,
            "frames": 0.2,
            "hop": 0.15,
            "overlap": 0.15,
            "working_overlap": 0.15,
            "prefill": 0.1,
            "warmup": 0.1,
        }

    def priority(self, case: SweepCase) -> float:
        entropy_estimate = self._tracker.case_entropy(case)
        anticipatory_entropy = 0.65 * entropy_estimate + 0.35 * self._recent_entropy
        projection = self._feature_projection(case)
        score = self._weight_entropy * anticipatory_entropy + projection + self._bias
        return math.tanh(score)

    def observe_result(self, case: SweepCase, entropy_signal: float) -> None:
        projection = self._feature_projection(case)
        self._recent_entropy = 0.9 * self._recent_entropy + 0.1 * entropy_signal
        self._bias = 0.98 * self._bias + 0.02 * (entropy_signal - 0.5 + 0.1 * projection)

    def _feature_projection(self, case: SweepCase) -> float:
        features = {
            "window_power": self._scale(case.window_power, 12.0),
            "working_window": self._scale(case.working_window, 512.0),
            "working_hop": self._scale(case.working_hop, 32.0),
            "frames": self._scale(case.frames, 8192.0),
            "hop": self._scale(case.hop, 32.0),
            "overlap": self._clip(case.overlap),
            "working_overlap": self._clip(case.working_overlap),
            "prefill": 1.0 if case.wheel_prefill else 0.0,
            "warmup": 1.0 if case.warmup_passthrough else 0.0,
        }
        return sum(self._feature_weights[k] * features[k] for k in features)

    @staticmethod
    def _scale(value, scale: float) -> float:
        if value is None or scale == 0:
            return 0.0
        return float(value) / scale

    @staticmethod
    def _clip(value: float | None) -> float:
        if value is None:
            return 0.0
        return float(max(-1.0, min(1.0, value)))


@dataclass
class PendingCaseEntry:
    case: SweepCase
    base_index: int
    priority: float = 0.0
    jitter: float = 0.0


class AdaptiveCasePool:
    _JITTER_WIDTH = 1e-3

    def __init__(self, cases: Sequence[SweepCase], network: SimpleEntropyNetwork, rng: random.Random) -> None:
        self._network = network
        self._rng = rng
        self._entries: List[PendingCaseEntry] = [
            PendingCaseEntry(case=case, base_index=index) for index, case in enumerate(cases)
        ]
        self._refresh()

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._entries)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def remaining(self) -> int:
        return len(self._entries)

    def pop_next(self) -> SweepCase:
        if not self._entries:
            raise IndexError("No pending cases remain")
        entry = self._entries.pop(0)
        return entry.case

    def reprioritize(self) -> None:
        if not self._entries:
            return
        self._refresh()

    def peek_cases(
        self, count: int
    ) -> Tuple[List[Tuple[SweepCase, float]], List[Tuple[SweepCase, float]]]:
        if count <= 0 or not self._entries:
            return ([], [])
        top_entries = self._entries[:count]
        bottom_entries = list(reversed(self._entries[-count:]))
        top = [(entry.case, entry.priority) for entry in top_entries]
        bottom = [(entry.case, entry.priority) for entry in bottom_entries]
        return (top, bottom)

    def _refresh(self) -> None:
        for entry in self._entries:
            entry.priority = self._network.priority(entry.case)
            entry.jitter = self._rng.uniform(-self._JITTER_WIDTH, self._JITTER_WIDTH)
        self._entries.sort(key=lambda item: (-(item.priority + item.jitter), item.base_index))


def run_case(
    case: SweepCase,
    binary: Path,
    passthrough: bool,
    timeout: float,
) -> CaseResult:
    argv = case.build_argv(binary)
    start = time.perf_counter()

    try:
        if passthrough:
            completed = subprocess.run(argv, check=False, timeout=timeout)
            stdout = ""
            stderr = ""
        else:
            completed = subprocess.run(
                argv,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        duration = time.perf_counter() - start
        result = CaseResult(
            case,
            completed.returncode,
            duration,
            stdout,
            stderr,
            timed_out=False,
            allocated_timeout=timeout,
        )
        attach_exit_code_metadata(result)
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = (exc.stdout or "") if not passthrough else ""
        stderr = (exc.stderr or "") if not passthrough else ""
        result = CaseResult(
            case,
            returncode=-999,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            allocated_timeout=timeout,
        )
        attach_exit_code_metadata(result)
        return result


def _linear_pending_views(
    pending: Sequence[SweepCase], count: int
) -> Tuple[List[Tuple[SweepCase, float | None]], List[Tuple[SweepCase, float | None]]]:
    if count <= 0 or not pending:
        return ([], [])
    top = [(case, None) for case in list(pending[:count])]
    bottom_seq = list(pending[-count:])
    bottom_seq.reverse()
    bottom = [(case, None) for case in bottom_seq]
    return (top, bottom)


def ensure_binary(binary: Path) -> None:
    if not binary.exists():
        raise SystemExit(
            f"Binary '{binary}' was not found. Build the project (cmake --build ...) and retry."
        )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(repo_root)
    patience_exempt_buckets = _parse_timeout_bucket_list(args.patience_exempt_timeouts)
    summary_json_arg = (args.summary_json or "").strip()
    summary_json_path: Path | None = None
    if summary_json_arg:
        candidate = Path(summary_json_arg)
        summary_json_path = candidate if candidate.is_absolute() else (repo_root / candidate)
    if args.exhaustive:
        space = resolve_exhaustive_space(args)
        cases = build_exhaustive_cases(space)
    else:
        cases = select_cases(args.cases.split(","))
    binary = args.binary if args.binary.is_absolute() else (repo_root / args.binary)
    ensure_binary(binary)

    print(f"Running {len(cases)} FFTDivisionNode configurations using {binary}")

    if args.dry_run:
        for case in cases:
            print(f"[DRY-RUN] {case.label}: {' '.join(case.build_argv(binary))}")
        return 0

    patience_enabled = args.patience is not None and args.patience > 0
    patience_remaining = float(args.patience) if patience_enabled else None
    initial_patience_budget = patience_remaining if patience_enabled else None
    patience_reserved = 0.0 if patience_enabled else None
    patience_savings = 0.0 if patience_enabled else None
    min_exec_time = max(0.0, float(args.min_execution_time))
    total_cases = len(cases)
    if patience_enabled and total_cases == 0:
        raise SystemExit("Patience budgeting requested but there are zero cases to schedule.")
    if patience_enabled and (patience_remaining is None or patience_remaining <= 0):
        raise SystemExit("Patience budget must be positive if provided.")

    worker_count = max(1, int(args.worker_count))
    if worker_count > 1:
        print(
            "Concurrency enabled: dispatching up to"
            f" {worker_count} FFTDivisionNode workers in parallel."
        )

    reservation_headroom = 0.0
    if patience_enabled:
        overbook_factor = max(0.0, float(args.patience_overbook_factor))
        reservation_headroom = max(0, worker_count - 1) * args.timeout * overbook_factor
        if reservation_headroom > 0:
            print(
                "Patience reservation headroom:" 
                f" allowing up to {reservation_headroom:.2f}s of overlapping reservations"
            )

    results: List[CaseResult] = []
    result_slots: dict[str, int] = {}
    flag_combo_counts: Counter[str] = Counter()
    exit_stage_counts: Counter[str] = Counter()
    pass_count = fail_count = 0
    active_timeout_count = idle_timeout_count = 0
    explicit_count = 0
    explicit_sum = 0.0
    explicit_max = 0.0
    dynamic_cap: float | None = None
    patience_warned = False
    dashboard_status: str | None = None
    use_live_dashboard = bool(args.live_dashboard)
    dashboard = LiveDashboard(args.dashboard_topk, enabled=use_live_dashboard)
    verbose_output = not use_live_dashboard

    adaptive_enabled = not args.disable_adaptive_ordering
    tracker = EntropyTracker() if adaptive_enabled else None
    rng = random.Random(args.case_order_seed) if adaptive_enabled else None
    network = SimpleEntropyNetwork(tracker, rng=rng) if tracker and rng else None
    cases_for_iteration = list(cases)
    current_pool: AdaptiveCasePool | None = None
    current_pending_linear: List[SweepCase] = [] if adaptive_enabled else list(cases_for_iteration)
    current_cap_multiplier = 1.0
    reprioritize_interval = max(1, int(args.entropy_reprioritize_interval)) if adaptive_enabled else 0
    completed_since_reprioritize = 0
    last_patience_tick = time.perf_counter()

    def _flag_label_token(result: CaseResult) -> str:
        return result.flag_label or "unknown_flag_combo"

    def _timeout_bucket(result: CaseResult) -> str | None:
        if not result.timed_out:
            return None
        if result.timeout_inference is None:
            return "unknown"
        return result.timeout_inference.bucket

    def apply_result_stats(result: CaseResult) -> None:
        nonlocal pass_count, fail_count, active_timeout_count, idle_timeout_count
        if result.exit_stage:
            exit_stage_counts[result.exit_stage] += 1
        flag_label = _flag_label_token(result)
        flag_combo_counts[flag_label] += 1
        bucket = _timeout_bucket(result)
        if bucket is None:
            if result.ok:
                pass_count += 1
            else:
                fail_count += 1
            return
        if bucket == "active":
            active_timeout_count += 1
        else:
            idle_timeout_count += 1

    def remove_result_stats(result: CaseResult) -> None:
        nonlocal pass_count, fail_count, active_timeout_count, idle_timeout_count
        if result.exit_stage and exit_stage_counts.get(result.exit_stage):
            exit_stage_counts[result.exit_stage] -= 1
            if exit_stage_counts[result.exit_stage] <= 0:
                del exit_stage_counts[result.exit_stage]
        flag_label = _flag_label_token(result)
        if flag_combo_counts.get(flag_label):
            flag_combo_counts[flag_label] -= 1
            if flag_combo_counts[flag_label] <= 0:
                del flag_combo_counts[flag_label]
        bucket = _timeout_bucket(result)
        if bucket is None:
            if result.ok and pass_count > 0:
                pass_count -= 1
            elif (not result.ok) and fail_count > 0:
                fail_count -= 1
            return
        if bucket == "active" and active_timeout_count > 0:
            active_timeout_count -= 1
        elif bucket != "active" and idle_timeout_count > 0:
            idle_timeout_count -= 1

    def burn_wall_time_budget() -> None:
        nonlocal patience_remaining, patience_warned, dashboard_status, last_patience_tick
        now = time.perf_counter()
        delta = max(0.0, now - last_patience_tick)
        last_patience_tick = now
        if not patience_enabled or patience_remaining is None or delta <= 0:
            return
        patience_remaining = max(0.0, patience_remaining - delta)
        if patience_remaining <= 0 and not patience_warned:
            dashboard_status = "Patience budget consumed"
            print(
                "Patience budget fully consumed; remaining cases (if any) will be skipped"
                " until additional budget is provided."
            )
            patience_warned = True

    def effective_dynamic_cap() -> float | None:
        if dynamic_cap is None:
            return None
        return dynamic_cap * current_cap_multiplier

    def available_patience_budget() -> float | None:
        if not patience_enabled or patience_remaining is None:
            return None
        reserved_now = patience_reserved or 0.0
        return max(0.0, patience_remaining - reserved_now + reservation_headroom)

    def render_dashboard(
        last_case: CaseResult | None,
        allocation: AllocationSnapshot | None,
        *,
        pool_ref: AdaptiveCasePool | None,
        pending_ref: Sequence[SweepCase],
        dynamic_cap_value: float | None,
    ) -> None:
        if not use_live_dashboard:
            return
        if adaptive_enabled and pool_ref is not None:
            top_cases, bottom_cases = pool_ref.peek_cases(dashboard.top_k)
            pending_count = pool_ref.remaining()
        else:
            top_cases, bottom_cases = _linear_pending_views(list(pending_ref), dashboard.top_k)
            pending_count = len(pending_ref)
        dashboard.update(
            top_cases=top_cases,
            bottom_cases=bottom_cases,
            completed=len(results),
            total_cases=total_cases,
            pass_count=pass_count,
            active_timeout_count=active_timeout_count,
            fail_count=fail_count,
            idle_timeout_count=idle_timeout_count,
            pending_count=pending_count,
            patience_enabled=patience_enabled,
            patience_remaining=patience_remaining,
            patience_reserved=patience_reserved if patience_enabled else None,
            patience_headroom=reservation_headroom if patience_enabled else None,
            patience_savings=patience_savings if patience_enabled else None,
            min_exec_time=min_exec_time,
            dynamic_cap=dynamic_cap_value,
            last_case=last_case,
            allocation=allocation,
            status_line=dashboard_status,
        )

    retry_round = 0
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while cases_for_iteration:
            current_cap_multiplier = args.retry_cap_multiplier ** retry_round if retry_round > 0 else 1.0
            active_retry_cases: dict[str, SweepCase] = {}
            iteration_completed = True
            completed_since_reprioritize = 0
            if adaptive_enabled and network is not None and rng is not None:
                current_pool = AdaptiveCasePool(cases_for_iteration, network, rng)
                current_pending_linear = []
            else:
                current_pool = None
                current_pending_linear = list(cases_for_iteration)

            render_dashboard(
                None,
                None,
                pool_ref=current_pool,
                pending_ref=current_pending_linear,
                dynamic_cap_value=effective_dynamic_cap(),
            )

            active_entries: List[ActiveCaseEntry] = []
            stop_dispatch = False

            def dequeue_case() -> Tuple[int, SweepCase] | None:
                nonlocal current_pool, current_pending_linear
                if adaptive_enabled:
                    if current_pool is None or not current_pool:
                        return None
                    remaining = current_pool.remaining()
                    if remaining <= 0:
                        return None
                    return remaining, current_pool.pop_next()
                if not current_pending_linear:
                    return None
                remaining = len(current_pending_linear)
                if remaining <= 0:
                    return None
                return remaining, current_pending_linear.pop(0)

            while True:
                burn_wall_time_budget()
                if patience_enabled and patience_remaining is not None and patience_remaining <= 0:
                    stop_dispatch = True
                while not stop_dispatch and len(active_entries) < worker_count:
                    payload = dequeue_case()
                    if payload is None:
                        break
                    cases_left, case = payload
                    base_ceiling = args.timeout
                    current_dynamic_cap = effective_dynamic_cap()
                    if current_dynamic_cap is not None:
                        base_ceiling = min(base_ceiling, max(current_dynamic_cap, min_exec_time))
                    timeout = base_ceiling
                    share_ceiling = None
                    equitable_share = None
                    if patience_enabled and patience_remaining is not None:
                        reserved_before = patience_reserved or 0.0
                        available_patience = available_patience_budget() or 0.0
                        if available_patience <= 0:
                            dashboard_status = "Patience exhausted before scheduling"
                            print(
                                "Patience budget exhausted before scheduling remaining"
                                f" {cases_left} cases; halting sweep."
                            )
                            iteration_completed = False
                            stop_dispatch = True
                            break
                        equitable_share = available_patience / cases_left
                        if equitable_share <= 0:
                            dashboard_status = "Patience share collapsed"
                            print(
                                "Patience share collapsed to zero while"
                                f" {cases_left} cases remain; halting sweep."
                            )
                            iteration_completed = False
                            stop_dispatch = True
                            break
                        share_ceiling = min(timeout, equitable_share, available_patience)
                        timeout = share_ceiling
                    timeout = max(timeout, min_exec_time)
                    if patience_enabled and patience_remaining is not None:
                        available_patience = available_patience_budget() or 0.0
                        timeout = min(timeout, available_patience)
                        if timeout <= 0:
                            dashboard_status = "Patience exhausted before scheduling"
                            iteration_completed = False
                            stop_dispatch = True
                            break
                    allocation_snapshot = AllocationSnapshot(
                        applied_timeout=timeout,
                        base_timeout=args.timeout,
                        share_ceiling=share_ceiling,
                        dynamic_cap=current_dynamic_cap,
                    )
                    if patience_enabled and patience_remaining is not None and verbose_output:
                        reserved_before = patience_reserved or 0.0
                        available_patience = available_patience_budget()
                        share_bits = []
                        if equitable_share is not None:
                            share_bits.append(
                                f"share={equitable_share:.2f}s"
                            )
                        share_bits.append(f"remaining={patience_remaining:.2f}s")
                        if reserved_before > 0:
                            share_bits.append(f"reserved={reserved_before:.2f}s")
                        if reservation_headroom > 0:
                            share_bits.append(f"headroom={reservation_headroom:.2f}s")
                        if available_patience is not None:
                            share_bits.append(f"avail={available_patience:.2f}s")
                        share_bits.append(f"cases={cases_left}")
                        allocation_parts = [
                            f"--timeout={args.timeout:.2f}s",
                            f"cap={max(current_dynamic_cap, min_exec_time):.2f}s"
                            if current_dynamic_cap is not None
                            else None,
                            " ".join(share_bits),
                            f"min={min_exec_time:.2f}s" if min_exec_time > 0 else None,
                        ]
                        allocation_text = ", ".join(part for part in allocation_parts if part)
                        print(f"Patience allocation: {timeout:.2f}s for {case.label} ({allocation_text})")
                    elif min_exec_time > 0 and verbose_output:
                        allocation_parts = [
                            f"--timeout={args.timeout:.2f}s",
                            f"cap={max(current_dynamic_cap, min_exec_time):.2f}s"
                            if current_dynamic_cap is not None
                            else None,
                            f"min={min_exec_time:.2f}s",
                        ]
                        allocation_text = ", ".join(part for part in allocation_parts if part)
                        print(f"Budget floor: allocating {timeout:.2f}s to {case.label} ({allocation_text})")
                    if verbose_output:
                        print(f"\n=== Case: {case.label} ===")
                        print("Command:", " ".join(case.build_argv(binary)))
                    future = executor.submit(run_case, case, binary, args.passthrough, timeout)
                    active_entries.append(
                        ActiveCaseEntry(
                            case=case,
                            future=future,
                            allocation=allocation_snapshot,
                            reserved_budget=timeout,
                        )
                    )
                    if patience_enabled and patience_reserved is not None:
                        patience_reserved += timeout

                if not active_entries:
                    pending_remaining = (
                        current_pool.remaining()
                        if adaptive_enabled and current_pool is not None
                        else len(current_pending_linear)
                    )
                    if stop_dispatch or pending_remaining <= 0:
                        break
                    continue

                done, _ = wait([entry.future for entry in active_entries], return_when=FIRST_COMPLETED)
                burn_wall_time_budget()
                if patience_enabled and patience_remaining is not None and patience_remaining <= 0:
                    stop_dispatch = True
                for entry in list(active_entries):
                    if entry.future not in done:
                        continue
                    result = entry.future.result()
                    active_entries.remove(entry)
                    if patience_enabled and patience_reserved is not None:
                        patience_reserved = max(0.0, patience_reserved - entry.reserved_budget)
                    if not result.timed_out:
                        explicit_count += 1
                        explicit_sum += result.duration
                        explicit_max = max(explicit_max, result.duration)
                        avg_duration = explicit_sum / explicit_count
                        suggested_cap = max(explicit_max, avg_duration * args.timeout_headroom)
                        if dynamic_cap is None or abs(suggested_cap - dynamic_cap) > 1e-6:
                            dynamic_cap = suggested_cap
                            adjusted_cap = effective_dynamic_cap()
                            if adjusted_cap is not None:
                                dashboard_status = f"Adaptive cap adjusted to {adjusted_cap:.2f}s"
                            else:
                                dashboard_status = "Adaptive cap deriving"
                            if verbose_output:
                                print(
                                    "Updated adaptive timeout cap to"
                                    f" {dynamic_cap:.2f}s based on explicit pass/fail durations"
                                )
                    if result.timed_out:
                        if args.passthrough:
                            inference = TimeoutInference(
                                bucket="unknown",
                                reason="stdout/stderr passthrough prevented timeout classification",
                                evidence=(),
                            )
                        else:
                            inference = infer_timeout_behavior(result)
                        result.timeout_inference = inference
                        dashboard_status = f"Timeout {inference.label}: {inference.reason}"
                        if verbose_output:
                            print(f"Timeout classification: {inference.label} ({inference.reason})")
                            for evidence_line in inference.evidence:
                                print("  evidence:", evidence_line)

                    reserved_budget = entry.reserved_budget
                    observed = min(result.duration, reserved_budget)
                    patience_charge: float | None = observed
                    if patience_enabled and patience_savings is not None:
                        saved = max(0.0, reserved_budget - observed)
                        patience_savings += saved
                    if (
                        patience_enabled
                        and patience_remaining is not None
                        and result.timed_out
                        and result.timeout_inference
                        and result.timeout_inference.bucket in patience_exempt_buckets
                    ):
                        concurrency_for_credit = max(1, len(active_entries) + 1)
                        credit = observed / float(concurrency_for_credit)
                        if initial_patience_budget is not None:
                            patience_remaining = min(
                                initial_patience_budget,
                                patience_remaining + credit,
                            )
                        patience_charge = 0.0
                        dashboard_status = (
                            f"Patience credit {credit:.2f}s for {result.timeout_inference.label}"
                        )
                        if verbose_output:
                            print(
                                "Patience credit: refunded"
                                f" {credit:.2f}s due to {result.timeout_inference.label}"
                            )
                    result.patience_charge = patience_charge
                    annotate_case_result(result)
                    flag_label = result.flag_label or "unknown_flag_combo"
                    if is_active_timeout(result):
                        active_retry_cases[result.case.label] = result.case

                    slot = result_slots.get(result.case.label)
                    if slot is not None:
                        remove_result_stats(results[slot])
                        results[slot] = result
                    else:
                        slot = len(results)
                        results.append(result)
                        result_slots[result.case.label] = slot
                    apply_result_stats(result)

                    if tracker and network and current_pool is not None:
                        entropy_signal = tracker.observe(result.case, flag_label)
                        network.observe_result(result.case, entropy_signal)
                        if current_pool:
                            completed_since_reprioritize += 1
                            if reprioritize_interval <= 1 or completed_since_reprioritize >= reprioritize_interval:
                                current_pool.reprioritize()
                                completed_since_reprioritize = 0
                    last_completed_result = result
                    status = result.status if not result.ok else "PASS"
                    if result.timed_out:
                        status = "TIMEOUT"
                    elif not result.ok:
                        status = f"FAIL (rc={result.returncode})"
                    if verbose_output:
                        print(f"Result: {status} in {result.duration:.2f}s")
                        if not args.passthrough:
                            if result.stdout:
                                print("-- stdout --")
                                print(result.stdout.rstrip())
                            if result.stderr:
                                print("-- stderr --")
                                print(result.stderr.rstrip())
                    if args.stop_on_failure and not result.ok:
                        iteration_completed = False
                        stop_dispatch = True

                    render_dashboard(
                        result,
                        entry.allocation,
                        pool_ref=current_pool,
                        pending_ref=current_pending_linear,
                        dynamic_cap_value=effective_dynamic_cap(),
                    )

                if not active_entries:
                    pending_remaining = (
                        current_pool.remaining()
                        if adaptive_enabled and current_pool is not None
                        else len(current_pending_linear)
                    )
                    if stop_dispatch or pending_remaining <= 0:
                        break
                    continue

            render_dashboard(
                last_completed_result,
                None,
                pool_ref=current_pool,
                pending_ref=current_pending_linear,
                dynamic_cap_value=effective_dynamic_cap(),
            )

            if not iteration_completed:
                cases_for_iteration = []
                break

            if (
                patience_enabled
                and patience_remaining is not None
                and patience_remaining > 0
                and active_retry_cases
            ):
                retry_round += 1
                cases_for_iteration = list(active_retry_cases.values())
                next_multiplier = args.retry_cap_multiplier ** retry_round
                dashboard_status = (
                    f"Retrying {len(cases_for_iteration)} active timeouts (round {retry_round})"
                )
                if verbose_output:
                    print(
                        "Retry pass {round_idx}: re-queueing {count} cases with cap multiplier x{mult:.2f}"
                        .format(round_idx=retry_round, count=len(cases_for_iteration), mult=next_multiplier)
                    )
                continue

            cases_for_iteration = []
            break

    burn_wall_time_budget()
    render_dashboard(
        last_completed_result,
        None,
        pool_ref=current_pool,
        pending_ref=current_pending_linear,
        dynamic_cap_value=effective_dynamic_cap(),
    )
    dashboard.close()

    passed = sum(1 for r in results if r.ok)
    print("\n=== Summary ===")
    header = [
        "case",
        "win",
        "hop",
        "overlap",
        "wwin",
        "whop",
        "wovr",
        "frames",
        "prefill",
        "warmup",
        "status",
        "notes",
    ]
    print("| " + " | ".join(f"{h:>8}" for h in header) + " |")
    print("|" + "----------|" * len(header))
    result_type_counts: Counter[str] = Counter()
    for result in results:
        annotate_case_result(result)
        row = result.case.summary_row()
        notes = ",".join(result.notes_tokens)
        row.extend([result.status, notes])
        rtype = result.result_type or "unknown"
        result_type_counts[rtype] += 1
        print("| " + " | ".join(f"{cell:>8}" for cell in row) + " |")
    print(f"Total: {passed}/{len(results)} cases succeeded")
    if active_timeout_count or idle_timeout_count:
        print(
            "Timeout breakdown: active={}/idle={}".format(
                active_timeout_count,
                idle_timeout_count,
            )
        )
    if result_type_counts:
        print("Result type breakdown:")
        for rtype, count in result_type_counts.most_common():
            print(f"  {rtype}: {count}")
    if exit_stage_counts:
        print("Exit stage breakdown:")
        total_reported = sum(exit_stage_counts.values())
        for stage, count in exit_stage_counts.most_common():
            percentage = (count / total_reported) * 100.0 if total_reported else 0.0
            print(f"  {stage}: {count} ({percentage:.1f}% of recorded exits)")
    if explicit_count:
        avg_duration = explicit_sum / explicit_count
        max_duration = explicit_max
        suggested_timeout = max(max_duration, avg_duration * args.timeout_headroom)
        print(
            "Explicit results stats: "
            f"n={explicit_count}, avg={avg_duration:.2f}s, max={max_duration:.2f}s"
        )
        print(
            "Suggested default timeout ceiling ≈ "
            f"{suggested_timeout:.2f}s (current --timeout={args.timeout:.2f}s)"
        )
        if args.timeout > suggested_timeout:
            print(
                "Current timeout exceeds the suggested ceiling; consider lowering it "
                "to avoid spending the full patience budget by default."
            )
    else:
        print("No explicit pass/fail durations were recorded (all cases timed out).")

    if summary_json_path:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_epoch": time.time(),
            "binary": str(binary),
            "timeout": args.timeout,
            "patience": args.patience,
            "timeout_headroom": args.timeout_headroom,
            "flag_combo_counts": dict(flag_combo_counts),
            "exit_stage_counts": dict(exit_stage_counts),
            "active_timeout_count": active_timeout_count,
            "idle_timeout_count": idle_timeout_count,
            "cases": [serialize_case_result(result) for result in results],
        }
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"JSON summary written to {summary_json_path}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
