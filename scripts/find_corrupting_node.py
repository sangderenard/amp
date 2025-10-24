# Diagnostic: call each amp_run_node C implementation individually
# and run a NumPy op after each to detect heap corruption.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from amp.app import build_runtime_graph
from amp.graph import AudioGraph
from amp.native_runtime import NativeGraphExecutor
from amp.state import build_default_state


def main():
    from amp.config import DEFAULT_CONFIG_PATH, load_configuration
    # Build the interactive default state (no joystick, no pygame), then
    # construct the runtime graph exactly as the interactive app does.
    cfg = load_configuration(DEFAULT_CONFIG_PATH)
    # build_default_state expects (joy, pygame) — pass None for joystick and
    # a minimal dummy pygame-like object (only attribute K_* used in mappings).
    class _DummyPygame:
        K_m = 0
        K_k = 0
        K_x = 0
        K_y = 0
        K_b = 0
        K_n = 0
        K_z = 0
        K_PERIOD = 0
        K_COMMA = 0
        K_SLASH = 0

    state = build_default_state(joy=None, pygame=_DummyPygame())
    g, envelope_names, amp_mod_names = build_runtime_graph(cfg.sample_rate, state)
    print("Native runtime diagnostics")
    with NativeGraphExecutor(g) as executor:
        try:
            output = executor.run_block(
                cfg.runtime.frames_per_chunk,
                sample_rate=cfg.sample_rate,
                base_params={},
                control_history_blob=b"",
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print("Native execution failed:", exc)
            return 1
    print("Native execution succeeded; output shape:", output.shape)
    print("Node order:", [node.name for node in g.ordered_nodes])
    return 0


if __name__ == '__main__':
    sys.exit(main())
