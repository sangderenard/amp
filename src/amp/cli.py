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
        "--headless",
        action="store_true",
        help="Render the configured graph without launching the interactive UI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from .app import run as run_app

    return run_app(
        allow_no_joystick=args.allow_no_joystick,
        no_audio=args.no_audio,
        headless=args.headless,
        config_path=str(args.config),
    )


__all__ = ["main", "build_parser"]
