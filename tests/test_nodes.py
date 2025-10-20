from __future__ import annotations

import numpy as np

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
