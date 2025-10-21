from __future__ import annotations

import numpy as np
import pytest

from amp import c_kernels
from amp.nodes import EnvelopeModulatorNode
from amp.state import RAW_DTYPE


def _random_envelope_params(rng: np.random.Generator, batches: int, frames: int) -> dict[str, np.ndarray]:
    trigger = (rng.random((batches, 1, frames)) > 0.8).astype(RAW_DTYPE, copy=False)
    gate = (rng.random((batches, 1, frames)) > 0.5).astype(RAW_DTYPE, copy=False)
    drone = (rng.random((batches, 1, frames)) > 0.7).astype(RAW_DTYPE, copy=False)
    velocity = rng.uniform(0.2, 1.0, size=(batches, 1, frames)).astype(RAW_DTYPE, copy=False)
    send_reset = np.ones((batches, 1, frames), dtype=RAW_DTYPE)
    return {
        "trigger": trigger,
        "gate": gate,
        "drone": drone,
        "velocity": velocity,
        "send_reset": send_reset,
    }


def test_envelope_kernel_matches_python():
    if not c_kernels.AVAILABLE:
        pytest.skip("C envelope kernel unavailable in this environment")

    rng = np.random.default_rng(1234)
    batches = 3
    frames = 96
    params = _random_envelope_params(rng, batches, frames)

    atk = 40
    hold = 10
    dec = 30
    sus = 0
    rel = 60
    sustain = 0.65
    send_resets = True

    trigger = params["trigger"][:, 0, :]
    gate = params["gate"][:, 0, :]
    drone = params["drone"][:, 0, :]
    velocity = params["velocity"][:, 0, :]

    init_stage = np.zeros(batches, dtype=np.int32)
    init_value = np.zeros(batches, dtype=RAW_DTYPE)
    init_timer = np.zeros(batches, dtype=RAW_DTYPE)
    init_vel_state = np.zeros(batches, dtype=RAW_DTYPE)
    init_activ = np.zeros(batches, dtype=np.int64)
    init_release = np.zeros(batches, dtype=RAW_DTYPE)

    args_py = (
        trigger,
        gate,
        drone,
        velocity,
        atk,
        hold,
        dec,
        sus,
        rel,
        sustain,
        send_resets,
        init_stage.copy(),
        init_value.copy(),
        init_timer.copy(),
        init_vel_state.copy(),
        init_activ.copy(),
        init_release.copy(),
    )

    out_amp_py = np.empty((batches, frames), dtype=RAW_DTYPE)
    out_reset_py = np.empty((batches, frames), dtype=RAW_DTYPE)
    amp_py, reset_py = c_kernels.envelope_process_py(*args_py, out_amp=out_amp_py, out_reset=out_reset_py)

    args_c = (
        trigger,
        gate,
        drone,
        velocity,
        atk,
        hold,
        dec,
        sus,
        rel,
        sustain,
        send_resets,
        init_stage.copy(),
        init_value.copy(),
        init_timer.copy(),
        init_vel_state.copy(),
        init_activ.copy(),
        init_release.copy(),
    )

    out_amp_c = np.empty((batches, frames), dtype=RAW_DTYPE)
    out_reset_c = np.empty((batches, frames), dtype=RAW_DTYPE)
    amp_c, reset_c = c_kernels.envelope_process_c(*args_c, out_amp=out_amp_c, out_reset=out_reset_c)

    np.testing.assert_allclose(amp_c, amp_py, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(reset_c, reset_py, rtol=0.0, atol=0.0)


def test_envelope_node_process_multi_batch():
    rng = np.random.default_rng(4321)
    node = EnvelopeModulatorNode("env")
    batches = 5
    frames = 128
    sr = 48_000

    params = _random_envelope_params(rng, batches, frames)
    audio = np.zeros((batches, 1, frames), dtype=RAW_DTYPE)

    out = node.process(frames, sr, audio, {}, params)

    assert out.shape == (batches, 2, frames)
    assert np.isfinite(out).all()
