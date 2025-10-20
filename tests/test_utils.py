import numpy as np

from amp import utils


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
