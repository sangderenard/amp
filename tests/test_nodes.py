from __future__ import annotations

import numpy as np
import pytest

from amp import nodes


def _make_param(value, frames):
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (1, 1, frames))


def test_oscillator_pan_stereo_output():
    osc = nodes.OscNode("osc", wave="sine")
    frames = 16
    params = {
        "freq": _make_param(440.0, frames),
        "amp": _make_param(0.2, frames),
        "pan": _make_param(1.0, frames),
    }

    out = osc.process(frames, 48000, None, {}, params)

    assert out.shape == (1, 2, frames)
    assert np.allclose(out[0, 0], 0.0, atol=1e-6)
    assert np.max(np.abs(out[0, 1])) > 0.0


def test_oscillator_portamento_slide_retains_state():
    osc = nodes.OscNode("osc", wave="sine")
    frames = 8
    freq = np.concatenate([
        np.full((1, 1, frames // 2), 10.0, dtype=np.float64),
        np.full((1, 1, frames - frames // 2), 40.0, dtype=np.float64),
    ], axis=2)
    params = {
        "freq": freq,
        "amp": _make_param(0.5, frames),
        "port": _make_param(1.0, frames),
        "slide": (
            _make_param(0.5, frames),
            _make_param(0.0, frames),
        ),
    }

    osc.process(frames, 80, None, {}, params)

    assert osc._freq_state is not None
    assert osc._freq_state.shape[0] == 1
    assert osc._freq_state[0] < 40.0


def test_oscillator_registers_chord_and_subharmonic_voices():
    osc = nodes.OscNode("osc", wave="sine")
    frames = 12
    params = {
        "freq": _make_param(55.0, frames),
        "amp": _make_param(0.8, frames),
        "chord": ([0.0, 1200.0], [1.0, 0.5]),
        "subharmonic": ([-1200.0], [0.3]),
    }

    osc.process(frames, 48000, None, {}, params)

    assert any(key.startswith("chord") for key in osc._voice_phase)
    assert any(key.startswith("sub") for key in osc._voice_phase)


def test_oscillator_applies_arpeggiation_plan():
    osc = nodes.OscNode("osc", wave="sine")
    frames = 4
    params = {
        "freq": _make_param(220.0, frames),
        "amp": _make_param(0.3, frames),
        "arp": {"sequence": [0.0, 1200.0], "frames_per_step": 1},
    }

    osc.process(frames, 48000, None, {}, params)

    assert osc._last_arp_offsets is not None
    assert np.allclose(
        osc._last_arp_offsets[0],
        np.array([0.0, 1200.0, 0.0, 1200.0], dtype=np.float64),
    )


def test_controller_node_evaluates_expressions():
    frames = 8
    ctrl = nodes.ControllerNode(
        "ctrl",
        params={
            "outputs": {
                "sum": "signals['a'] + 2.0 * signals['b']",
                "clipped": "np.clip(signals['a'] - signals['b'], 0.0, 1.0)",
            }
        },
    )
    a = np.full((1, 1, frames), 0.25, dtype=np.float64)
    b = np.linspace(0.0, 0.5, frames, dtype=np.float64)[None, None, :]
    params = {"a": a, "b": b}

    output = ctrl.process(frames, 48000, None, {}, params)

    assert output.shape == (1, 2, frames)
    sum_idx = ctrl.output_index("sum")
    clip_idx = ctrl.output_index("clipped")
    expected_sum = a[:, 0, :] + 2.0 * b[:, 0, :]
    expected_clip = np.clip(a[:, 0, :] - b[:, 0, :], 0.0, 1.0)
    assert np.allclose(output[:, sum_idx, :], expected_sum)
    assert np.allclose(output[:, clip_idx, :], expected_clip)
    with pytest.raises(KeyError):
        ctrl.output_index("missing")
