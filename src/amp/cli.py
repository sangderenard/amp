"""Command line entry point for the synthesiser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMP synthesiser entry point")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to configuration file")
    parser.add_argument(
        "--allow-no-joystick",
        action="store_true",
        help="Permit launching the application without a connected joystick",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip initialising audio output (useful in CI or debugging)",
    )
    parser.add_argument(
        "--interactive-adaptive-batching",
        dest="interactive_adaptive_batching",
        action="store_true",
        help="Enable adaptive batching during interactive playback",
    )
    parser.add_argument(
        "--no-interactive-adaptive-batching",
        dest="interactive_adaptive_batching",
        action="store_false",
        help="Disable adaptive batching during interactive playback",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render the configured graph without launching the interactive UI",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        help="Override the callback-sized frame count used for headless diagnostics",
    )
    parser.add_argument(
        "--headless-iterations",
        type=int,
        help="Number of headless diagnostic iterations to run",
    )
    parser.add_argument(
        "--headless-warmup",
        type=int,
        help="Warmup iterations to discard from headless EMA calculations",
    )
    parser.add_argument(
        "--headless-batch",
        type=int,
        help="Number of callback-sized blocks rendered per headless iteration",
    )
    parser.add_argument(
        "--headless-alpha",
        type=float,
        help="EMA smoothing factor for headless diagnostics (0-1)",
    )
    parser.add_argument(
        "--headless-adaptive-batching",
        dest="headless_adaptive_batching",
        action="store_true",
        help="Enable adaptive batching during headless diagnostics",
    )
    parser.add_argument(
        "--no-headless-adaptive-batching",
        dest="headless_adaptive_batching",
        action="store_false",
        help="Disable adaptive batching during headless diagnostics",
    )
    parser.add_argument(
        "--headless-output",
        type=Path,
        help=(
            "Optional path to write rendered audio. Paths ending in .wav are"
            " written as 16-bit WAV; other suffixes receive raw float32"
            " frames (little-endian)."
        ),
    )
    parser.add_argument(
        "--headless-joystick-mode",
        choices=("switch", "axis"),
        help="Select the virtual joystick style used during headless runs",
    )
    parser.add_argument(
        "--headless-joystick-script",
        type=Path,
        help="Path to a JSON script describing virtual joystick automation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.set_defaults(
        interactive_adaptive_batching=None,
        headless_adaptive_batching=None,
    )

    args = parser.parse_args(argv)

    from .app import run as run_app

    return run_app(
        allow_no_joystick=args.allow_no_joystick,
        no_audio=args.no_audio,
        headless=args.headless,
        config_path=str(args.config),
        interactive_adaptive_batching=args.interactive_adaptive_batching,
        headless_frames=args.headless_frames,
        headless_iterations=args.headless_iterations,
        headless_warmup=args.headless_warmup,
        headless_batch=args.headless_batch,
        headless_alpha=args.headless_alpha,
        headless_output=str(args.headless_output) if args.headless_output else None,
        headless_joystick_mode=args.headless_joystick_mode,
        headless_joystick_script=(
            str(args.headless_joystick_script)
            if args.headless_joystick_script
            else None
        ),
        headless_adaptive_batching=args.headless_adaptive_batching,
    )


__all__ = ["main", "build_parser"]
