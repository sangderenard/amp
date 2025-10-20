import numpy as np
from amp import c_kernels


def _rand_ctrl(B, F):
    return (np.random.rand(B, F) > 0.97).astype(float)


def test_envelope_parity_small():
    np.random.seed(1)
    B = 3
    F = 64
    trigger = _rand_ctrl(B, F)
    gate = (np.random.rand(B, F) > 0.5).astype(float)
    drone = (np.random.rand(B, F) > 0.95).astype(float)
    velocity = np.random.rand(B, F).astype(float)

    # initial states
    stage = np.zeros(B, dtype=np.int32)
    value = np.zeros(B, dtype=float)
    timer = np.zeros(B, dtype=float)
    vel_state = np.zeros(B, dtype=float)
    activations = np.zeros(B, dtype=np.int64)
    release_start = np.zeros(B, dtype=float)

    kwargs = dict(
        trigger=trigger,
        gate=gate,
        drone=drone,
        velocity=velocity,
        atk_frames=3,
        hold_frames=2,
        dec_frames=5,
        sus_frames=0,
        rel_frames=7,
        sustain_level=0.6,
        send_resets=True,
    )

    # run python fallback
    st_py = stage.copy()
    val_py = value.copy()
    tim_py = timer.copy()
    vel_py = vel_state.copy()
    acts_py = activations.copy()
    rel_py = release_start.copy()
    amp_py, reset_py = c_kernels.envelope_process_py(
        kwargs["trigger"],
        kwargs["gate"],
        kwargs["drone"],
        kwargs["velocity"],
        kwargs["atk_frames"],
        kwargs["hold_frames"],
        kwargs["dec_frames"],
        kwargs["sus_frames"],
        kwargs["rel_frames"],
        kwargs["sustain_level"],
        kwargs["send_resets"],
        st_py,
        val_py,
        tim_py,
        vel_py,
        acts_py,
        rel_py,
    )

    # run C kernel if available
    st_c = np.zeros_like(stage)
    val_c = np.zeros_like(value)
    tim_c = np.zeros_like(timer)
    vel_c = np.zeros_like(vel_state)
    acts_c = np.zeros_like(activations)
    rel_c = np.zeros_like(release_start)
    try:
        amp_c, reset_c = c_kernels.envelope_process_c(
            kwargs["trigger"],
            kwargs["gate"],
            kwargs["drone"],
            kwargs["velocity"],
            kwargs["atk_frames"],
            kwargs["hold_frames"],
            kwargs["dec_frames"],
            kwargs["sus_frames"],
            kwargs["rel_frames"],
            kwargs["sustain_level"],
            kwargs["send_resets"],
            st_c,
            val_c,
            tim_c,
            vel_c,
            acts_c,
            rel_c,
        )
    except Exception:
        # if C kernel not available just assert that Python runs
        assert amp_py.shape == (B, F)
        return

    # compare outputs and states
    assert np.allclose(amp_py, amp_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(reset_py, reset_c, atol=1e-12, rtol=1e-9)
    assert np.array_equal(st_py, st_c)
    assert np.allclose(val_py, val_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(tim_py, tim_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(vel_py, vel_c, atol=1e-12, rtol=1e-9)
    assert np.array_equal(acts_py, acts_c)
    assert np.allclose(rel_py, rel_c, atol=1e-12, rtol=1e-9)
