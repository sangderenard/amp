"""Render a joystick automation script to audio via the headless runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from amp.config import DEFAULT_CONFIG_PATH
from amp.app import run as run_app


def _default_output_path(script_path: Path) -> Path:
    stem = script_path.stem or script_path.name
    return script_path.with_name(f"{stem}.wav")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a joystick automation JSON script using the headless renderer "
            "and write a 16-bit WAV file."
        )
    )
    parser.add_argument(
        "joystick_script",
        type=Path,
        help="Path to the joystick automation JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination WAV path (defaults to <script>.wav)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Configuration file passed to the synthesiser runtime",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Override the callback frame count for the render",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of callback batches to render",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="Warmup batches discarded from EMA calculations",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Callback-sized blocks rendered per iteration",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="EMA smoothing factor (0-1)",
    )
    parser.add_argument(
        "--joystick-mode",
        choices=("switch", "axis"),
        default="switch",
        help="Virtual joystick performer mode",
    )
    parser.add_argument(
        "--allow-no-joystick",
        action="store_true",
        help="Permit running even when no physical joystick is connected",
    )
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Initialise the audio backend (disabled by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    script_path = args.joystick_script.expanduser().resolve()
    if not script_path.is_file():
        parser.error(f"Joystick automation script not found: {script_path}")

    output_path = (args.output.expanduser() if args.output else _default_output_path(script_path)).resolve()
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")

    exit_code = run_app(
        allow_no_joystick=bool(args.allow_no_joystick),
        no_audio=not args.with_audio,
        headless=True,
        config_path=str(args.config),
        headless_frames=args.frames,
        headless_iterations=args.iterations,
        headless_warmup=args.warmup,
        headless_batch=args.batch,
        headless_alpha=args.alpha,
        headless_output=str(output_path),
        headless_joystick_mode=args.joystick_mode,
        headless_joystick_script=str(script_path),
    )
    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
