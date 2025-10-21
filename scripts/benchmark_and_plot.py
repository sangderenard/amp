"""
Script to run the agent benchmark and display the results using Plotly.
"""


import sys
import pathlib
import pandas as pd

# Ensure we can import the package from the local src directory
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amp.config import DEFAULT_FRAMES_PER_BLOCK
from amp.agent_benchmark_viz import create_timeline_figure
from scripts.agent_benchmark import benchmark_default_graph

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agent benchmark and plot results.")
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES_PER_BLOCK)
    parser.add_argument('--sample-rate', type=float, default=44100)
    parser.add_argument('--ema-alpha', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--joystick-mode', type=str, default='switch')
    parser.add_argument('--joystick-script', type=pathlib.Path, default=pathlib.Path('scripts/joystick_song_10s.json'))
    args = parser.parse_args()

    # 10 seconds of audio: iterations = ceil(10 * sample_rate / frames)
    import math
    iterations = math.ceil(10 * args.sample_rate / args.frames)

    df = benchmark_default_graph(
        frames=args.frames,
        iterations=iterations,
        sample_rate=args.sample_rate,
        ema_alpha=args.ema_alpha,
        warmup=args.warmup,
        joystick_mode=args.joystick_mode,
        joystick_script=args.joystick_script,
    )

    fig = create_timeline_figure(df)
    fig.show()

if __name__ == "__main__":
    main()
