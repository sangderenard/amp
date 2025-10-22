# Diagnostic: call each amp_run_node C implementation individually
# and run a NumPy op after each to detect heap corruption.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from amp.graph_edge_runner import CffiEdgeRunner
from amp.node_contracts import get_node_contract
from amp.graph import AudioGraph
from amp.app import build_runtime_graph
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
    runner = CffiEdgeRunner(g)
    # Use the runtime-configured frames per chunk from the loaded configuration
    runner.begin_block(frames=cfg.runtime.frames_per_chunk)
    try:
        lib = runner._ensure_c_kernel()
    except Exception as e:
        print("C kernel not available:", e)
        return 1

    # Sanity check: does the compiled library expose our debug getter?
    has_getter = hasattr(lib, "amp_last_alloc_count_get")
    print(f"C kernel loaded; amp_last_alloc_count_get available: {has_getter}")

    print("C kernel loaded; testing nodes one-by-one")
    for name in runner._plan_names or runner.ordered_nodes:
        print(f"Testing node: {name}")
        try:
            handle = runner.gather_to(name)
            desc_struct, desc_keep = runner._build_descriptor_struct(runner._descriptor_by_name[name])
            contract = get_node_contract(runner._descriptor_by_name[name].type_name)
            if contract is not None:
                print(
                    "  contract: allow_python_fallback=%s stereo_params=%s" % (
                        contract.allow_python_fallback,
                        ",".join(contract.stereo_params) if contract.stereo_params else "(none)",
                    )
                )
            out_ptr = runner.ffi.new("double **")
            out_channels = runner.ffi.new("int *")
            state_ptr = runner.ffi.new("void **", runner._node_states.get(name, runner.ffi.NULL))
            status = lib.amp_run_node(
                desc_struct,
                handle.cdata,
                int(handle.batches),
                int(handle.channels),
                int(handle.frames),
                float(runner._sample_rate),
                out_ptr,
                out_channels,
                state_ptr,
                runner.ffi.NULL,
            )
            print(f"  status={status}")
            if status == 0:
                # copy back into numpy to be safe
                total = int(handle.batches) * int(out_channels[0]) * int(handle.frames)
                # ask C how many elements it allocated in its last malloc
                c_alloc = None
                try:
                    impl_lib = None
                    # prefer getting the symbol from the dlopened lib, else fallback
                    if hasattr(lib, 'amp_last_alloc_count_get'):
                        c_alloc = lib.amp_last_alloc_count_get()
                    else:
                        try:
                            import amp.c_kernels as c_kernels
                            impl = getattr(c_kernels, '_impl', None)
                            impl_lib = getattr(impl, 'lib', None)
                            if impl_lib is not None and hasattr(impl_lib, 'amp_last_alloc_count_get'):
                                c_alloc = impl_lib.amp_last_alloc_count_get()
                        except Exception:
                            pass
                    if c_alloc is not None:
                        b = int(handle.batches)
                        f = int(handle.frames)
                        py_ch = int(out_channels[0])
                        print(f"  python: batches={b}, out_channels={py_ch}, frames={f}, expected={total}")
                        if handle.node_buffer is None and py_ch != handle.channels:
                            raise RuntimeError(
                                f"gather_to() predicted {handle.channels} channel(s) but the C kernel reported {py_ch}"
                            )
                        print(f"  C reported alloc elements={c_alloc}")
                        try:
                            # infer how many channels C allocated (integer division)
                            inferred_c = int(c_alloc // (b * f)) if (b * f) > 0 else None
                        except Exception:
                            inferred_c = None
                        print(f"  inferred C channels={inferred_c}")
                    else:
                        print("  amp_last_alloc_count_get not available on C lib")
                except Exception as e:
                    print(f"  calling amp_last_alloc_count_get() raised: {e!r}")
                array = None
                if total > 0:
                    buf = runner.ffi.buffer(out_ptr[0], total * np.dtype(np.float64).itemsize)
                    array = np.frombuffer(buf, dtype=np.float64).copy().reshape((int(handle.batches), int(out_channels[0]), int(handle.frames)))
                runner.set_node_output(name, array)
                lib.amp_free(out_ptr[0])
            # run a large numpy op to try to trigger heap-corruption detection quickly
            try:
                a = np.linspace(0.0, 1.0, 200000)
                s = a.sum()
                print(f"  numpy check OK (sum={s:.6f})")
            except Exception as ne:
                print(f"  numpy op after node {name} failed: {ne!r}")
                raise
        except Exception as exc:
            print(f"  Exception while testing node {name}: {exc!r}")
            raise
    print("All nodes tested without triggering immediate numpy failure")
    fallback_summary = runner.python_fallback_summary()
    if fallback_summary:
        print("Python fallbacks detected:")
        for node, count in fallback_summary.items():
            print(f"  {node}: {count}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
