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
        help="Run without pygame/audio output; uses a neutral virtual joystick",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.headless:
        from .application import SynthApplication
        from .config import load_configuration

        config = load_configuration(args.config)
        app = SynthApplication.from_config(config)
        print(app.summary())
        buffer = app.render()
        peak = float(buffer.max())
        trough = float(buffer.min())
        print(
            f"Rendered {buffer.shape[1]} frames @ {config.sample_rate} Hz "
            f"(peak {peak:.3f}, trough {trough:.3f})"
        )
        if app.joystick_error and not app.joystick:
            print(f"Warning: {app.joystick_error}")
        return 0

    from .interactive import run_app

    allow_no_joystick = args.allow_no_joystick
    no_audio = args.no_audio

    return run_app(
        allow_no_joystick=allow_no_joystick,
        no_audio=no_audio,
        headless=False,
        config_path=str(args.config),
    )


__all__ = ["main", "build_parser"]
