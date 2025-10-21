import sys
import pathlib
import traceback

# Ensure we can import the package from the local src directory (same as
# scripts/_import_check.py).
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from amp.system import benchmark_default_graph
except Exception as exc:
    print('ImportError:', exc)
    traceback.print_exc()
    raise


if __name__ == '__main__':
    try:
        df = benchmark_default_graph(frames=32, iterations=2, sample_rate=44100, ema_alpha=0.1, warmup=1, joystick_mode='switch', joystick_script_path=None)
        print('rows=', len(df))
    except Exception as exc:
        print('Runtime error:', exc)
        traceback.print_exc()
        raise
