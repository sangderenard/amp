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


def test_control_delay_consumes_cached_pcm():
    delay = ControlDelay(48000, history_seconds=0.5)

    pcm = np.vstack(
        [
            np.linspace(0.0, 1.0, 96, dtype=float),
            np.linspace(1.0, 0.0, 96, dtype=float),
        ]
    )
    delay.add_pcm(0.0, pcm)

    first = delay.consume_pcm(0.0, 32)
    assert first is not None
    assert first.shape == (2, 32)
    assert np.isclose(first[0, 0], 0.0)
    assert np.isclose(first[0, -1], pcm[0, 31])

    second_start = 32 / 48000.0
    second = delay.consume_pcm(second_start, 32)
    assert second is not None
    assert second.shape == (2, 32)
    assert np.isclose(second[0, 0], pcm[0, 32])
    assert np.isclose(second[1, -1], pcm[1, 63])

    # Remaining chunk should have 32 frames left starting at the new timestamp
    remaining = delay.consume_pcm(64 / 48000.0, 32)
    assert remaining is not None
    assert remaining.shape == (2, 32)

    # Cache exhausted -> subsequent consume returns None
    assert delay.consume_pcm(96 / 48000.0, 16) is None


def test_control_delay_invalidation_truncates_future_pcm():
    delay = ControlDelay(44100, history_seconds=1.0)

    pcm = np.vstack(
        [
            np.linspace(-1.0, 1.0, 128, dtype=float),
            np.linspace(1.0, -1.0, 128, dtype=float),
        ]
    )
    delay.add_pcm(0.0, pcm)

    # Record an event mid-way through the cached PCM. Future audio past the
    # event timestamp should be discarded.
    event_time = 32 / 44100.0
    delay.record_event(event_time, pitch=60.0, envelope=0.0)

    cached = delay.pcm_chunks
    assert len(cached) == 1
    chunk = cached[0]
    assert chunk.timestamp == 0.0
    # The retained chunk should end no later than the event timestamp
    assert chunk.end_time <= event_time + (1.0 / delay.sample_rate)
