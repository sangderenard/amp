import numpy as np
from amp.nodes import LFONode


def _make_params(frames, B=2, depth=0.5):
    # simple params to drive LFONode without input
    return {}


def run_node(node: LFONode, frames=128, sr=48000):
    # audio_in None -> node will generate internal waveform
    out = node.process(frames, sr, None, None, {})
    return out.copy()


def test_lfo_slew_backends_agree():
    frames = 128
    sr = 48000
    # Vector backend
    n_vec = LFONode("lfo_vec", slew_ms=10.0, slew_backend="vector")
    n_iter = LFONode("lfo_iter", slew_ms=10.0, slew_backend="iter")
    n_c = LFONode("lfo_c", slew_ms=10.0, slew_backend="c")

    out_vec = run_node(n_vec, frames, sr)
    out_iter = run_node(n_iter, frames, sr)
    out_c = None
    try:
        out_c = run_node(n_c, frames, sr)
    except Exception:
        # C backend may be unavailable in CI; ignore
        out_c = None

    # Compare vector and iterative (should be close)
    assert np.allclose(out_vec, out_iter, rtol=1e-9, atol=1e-12)
    if out_c is not None:
        assert np.allclose(out_vec, out_c, rtol=1e-9, atol=1e-12)
