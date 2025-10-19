"""Command line entry point for the synthesiser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .application import SynthApplication
from .config import DEFAULT_CONFIG_PATH, load_configuration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an audio graph configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to configuration file")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to render")
    parser.add_argument("--summary", action="store_true", help="Print the graph summary before rendering")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_configuration(args.config)
    app = SynthApplication.from_config(config)

    if args.summary or config.runtime.log_summary:
        print(app.summary())
        if app.joystick_error and not app.joystick:
            print(f"Warning: {app.joystick_error}")

    buffer = app.render(frames=args.frames)
    # We do not write to an audio device; simply report success and stats.
    peak = float(buffer.max())
    trough = float(buffer.min())
    print(f"Rendered {buffer.shape[1]} frames @ {config.sample_rate} Hz (peak {peak:.3f}, trough {trough:.3f})")
    return 0


__all__ = ["main", "build_parser"]
