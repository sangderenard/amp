import numpy as np

from amp.graph import ControlDelay


def test_control_delay_interpolates_and_emits_tensor():
    delay = ControlDelay(48000, history_seconds=0.2, control_delay_seconds=0.05)

    delay.record_event(0.0, pitch=60.0, envelope=0.0)
    delay.record_event(0.01, pitch=62.0, envelope=0.5)
    delay.record_event(0.02, pitch=65.0, envelope=1.0)

    pcm_chunk = np.vstack(
        [
            np.linspace(-1.0, 1.0, 96, dtype=float),
            np.linspace(1.0, -1.0, 96, dtype=float),
        ]
    )
    delay.add_pcm(0.015, pcm_chunk)

    block = delay.sample(0.015, 48)

    assert block["control_tensor"].shape == (48, 3)
    assert np.allclose(block["pitch"][0, 0], 63.5, atol=1e-3)
    assert np.allclose(block["envelope"][0, 0], 0.75, atol=1e-3)
    assert np.allclose(block["envelope"][-1, 0], 0.798958, atol=1e-3)
    assert block["pcm"].shape == (2, 48)
    assert np.isclose(block["pcm"][0, 0], -1.0, atol=1e-6)
    assert np.isclose(block["pcm"][1, -1], 0.010526, atol=1e-6)


def test_control_delay_trims_history():
    delay = ControlDelay(44100, history_seconds=0.05)

    delay.record_event(0.0, pitch=60.0, envelope=0.0)
    delay.record_event(0.1, pitch=61.0, envelope=0.5)

    assert len(delay.events) == 1

    chunk = np.ones(64)
    delay.add_pcm(0.2, chunk)
    assert len(delay.pcm_chunks) == 1
