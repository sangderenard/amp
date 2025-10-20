import numpy as np
import pytest

from amp import utils


def _reference_dc(sig, a, y_prev, x_prev):
    sig = np.asarray(sig, dtype=utils.RAW_DTYPE)
    out = np.empty(sig.shape[0], dtype=utils.RAW_DTYPE)
    for i, x in enumerate(sig):
        y_prev = a * y_prev + x - x_prev
        x_prev = x
        out[i] = y_prev
    return out, float(y_prev), float(x_prev)


def test_scratch_buffers_grow_to_power_of_two():
    scratch = utils._ScratchBuffers()

    scratch.ensure(3)
    assert scratch.capacity == 4
    assert scratch.tmp.shape == (4,)
    assert scratch.f.shape == (4,)

    tmp_before = scratch.tmp
    scratch.ensure(2)
    assert scratch.capacity == 4
    assert scratch.tmp is tmp_before

    scratch.ensure(9)
    assert scratch.capacity == 16
    assert scratch.q.shape == (16,)


def test_dc_block_allocates_scratch(monkeypatch):
    scratch = utils._ScratchBuffers()
    monkeypatch.setattr(utils, "_scratch", scratch)

    signal = np.linspace(-1.0, 1.0, 7, dtype=utils.RAW_DTYPE)
    out = utils.dc_block(signal)

    assert out.shape == (7,)
    assert scratch.capacity >= 7
    assert np.shares_memory(out, scratch.tmp)


def test_dc_block_matches_reference_short_buffer(monkeypatch):
    scratch = utils._ScratchBuffers()
    monkeypatch.setattr(utils, "_scratch", scratch)
    monkeypatch.setattr(utils, "_dc_prev", 0.0, raising=False)
    monkeypatch.setattr(utils, "_prev_in", 0.0, raising=False)

    a = 0.97
    signal = np.array([0.3, -0.2, 0.5, -0.7, 0.1], dtype=utils.RAW_DTYPE)
    expected, expected_prev, expected_in = _reference_dc(signal, a, 0.0, 0.0)

    out = utils.dc_block(signal, a=a)

    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-12)
    assert utils._dc_prev == pytest.approx(expected_prev)
    assert utils._prev_in == pytest.approx(expected_in)
    assert np.shares_memory(out, scratch.tmp)


def test_dc_block_matches_reference_long_buffer_and_reuses_scratch(monkeypatch):
    scratch = utils._ScratchBuffers()
    monkeypatch.setattr(utils, "_scratch", scratch)
    monkeypatch.setattr(utils, "_dc_prev", 0.0, raising=False)
    monkeypatch.setattr(utils, "_prev_in", 0.0, raising=False)

    a = 0.995
    long_signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 2048, dtype=utils.RAW_DTYPE))
    expected_long, y_prev, x_prev = _reference_dc(long_signal, a, 0.0, 0.0)

    out_long = utils.dc_block(long_signal, a=a)

    np.testing.assert_allclose(out_long, expected_long, atol=1e-12, rtol=1e-12)
    assert utils._dc_prev == pytest.approx(y_prev)
    assert utils._prev_in == pytest.approx(x_prev)
    assert np.shares_memory(out_long, scratch.tmp)

    tmp_buffer = scratch.tmp

    next_signal = np.cos(np.linspace(0.0, 4.0 * np.pi, 512, dtype=utils.RAW_DTYPE))
    expected_next, y_prev, x_prev = _reference_dc(next_signal, a, y_prev, x_prev)

    out_next = utils.dc_block(next_signal, a=a)

    np.testing.assert_allclose(out_next, expected_next, atol=1e-12, rtol=1e-12)
    assert utils._dc_prev == pytest.approx(y_prev)
    assert utils._prev_in == pytest.approx(x_prev)
    assert scratch.tmp is tmp_buffer
    assert np.shares_memory(out_next, scratch.tmp)
