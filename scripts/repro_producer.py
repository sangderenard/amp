from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amp.app import build_runtime_graph
from amp.state import build_default_state

# Provide a minimal fake pygame object required by build_default_state
class _FakePygame:
    K_m = 109
    K_k = 107
    K_x = 120
    K_y = 121
    K_b = 98
    K_n = 110
    K_z = 122
    K_PERIOD = 190
    K_COMMA = 188
    K_SLASH = 191

fake_pygame = _FakePygame()
from amp.runner import render_audio_block

def main():
    state = build_default_state(joy=None, pygame=fake_pygame)
    graph, envs, amps = build_runtime_graph(44100, state)
    iterations = 200
    frames = 256
    sample_rate = 44100
    start = perf_counter()
    for i in range(iterations):
        try:
            buffer, meta = render_audio_block(
                graph,
                perf_counter(),
                frames,
                sample_rate,
                {},
                state,
                envs,
                amps,
                {},
            )
        except Exception as exc:
            print(f"Iteration {i}: renderer raised: {exc!r}")
            raise
        if buffer is None:
            print(f"Iteration {i}: renderer returned None")
            raise RuntimeError("renderer returned None")
    end = perf_counter()
    print(f"Completed {iterations} iterations in {end-start:.3f}s")

if __name__ == '__main__':
    main()
