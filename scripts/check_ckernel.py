import sys
import types
import traceback
from pathlib import Path

# Ensure the project's `src` directory is on sys.path so `import amp` works when run from the repo root
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

from amp import c_kernels
print("c_kernels.AVAILABLE:", c_kernels.AVAILABLE)

from amp.app import build_runtime_graph
from amp.native_runtime import AVAILABLE as RUNTIME_AVAILABLE
from amp.native_runtime import NativeGraphExecutor
from amp.state import build_default_state

fake_pygame = types.SimpleNamespace(
    K_m=0, K_k=1, K_x=2, K_y=3, K_b=4, K_n=5, K_z=6, K_PERIOD=7, K_COMMA=8, K_SLASH=9
)

state = build_default_state(joy=None, pygame=fake_pygame)
graph, envs, amps = build_runtime_graph(44100, state)

print("graph nodes:", list(graph._nodes.keys()))
print("native runtime available:", RUNTIME_AVAILABLE)

try:
    with NativeGraphExecutor(graph) as executor:
        out = executor.run_block(256, sample_rate=44100.0, base_params={}, control_history_blob=b"")
        print("NativeGraphExecutor run succeeded, output shape:", out.shape)
except Exception:
    print("NativeGraphExecutor failed:")
    traceback.print_exc()
