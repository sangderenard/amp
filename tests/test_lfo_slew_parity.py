import pytest

from amp import c_kernels
from amp.nodes import LFONode


def run_node(node: LFONode, frames=128, sr=48000):
    # audio_in None -> node will generate internal waveform
    out = node.process(frames, sr, None, None, {})
    return out.copy()


def test_lfo_slew_requires_c_backend():
    with pytest.raises(ValueError):
        LFONode("lfo_vec", slew_ms=10.0, slew_backend="vector")
    with pytest.raises(ValueError):
        LFONode("lfo_iter", slew_ms=10.0, slew_backend="iter")


def test_lfo_slew_runs_when_c_available():
    if not c_kernels.AVAILABLE:
        with pytest.raises(RuntimeError):
            LFONode("lfo_auto", slew_ms=10.0)
        return

    node = LFONode("lfo_c", slew_ms=10.0, slew_backend="c")
    out = run_node(node)
    assert out.shape[2] == 128
