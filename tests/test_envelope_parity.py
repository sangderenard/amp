import numpy as np
import pytest
from amp import c_kernels


def _rand_ctrl(B, F):
    return (np.random.rand(B, F) > 0.97).astype(float)


def envelope_process_reference(
    trigger,
    gate,
    drone,
    velocity,
    atk_frames,
    hold_frames,
    dec_frames,
    sus_frames,
    rel_frames,
    sustain_level,
    send_resets,
    stage,
    value,
    timer,
    vel_state,
    activations,
    release_start,
):
    B, F = trigger.shape
    amp = np.zeros((B, F), dtype=float)
    reset = np.zeros((B, F), dtype=float)

    for b in range(B):
        st = int(stage[b])
        val = float(value[b])
        tim = float(timer[b])
        vel = float(vel_state[b])
        acts = int(activations[b])
        rel_start = float(release_start[b])
        trig_line = trigger[b] > 0.5
        gate_line = gate[b] > 0.5
        drone_line = drone[b] > 0.5
        for i in range(F):
            trig = bool(trig_line[i])
            gate_on = bool(gate_line[i])
            drone_on = bool(drone_line[i])
            if trig:
                st = 1
                tim = 0.0
                val = 0.0
                vel = max(0.0, float(velocity[b, i]))
                rel_start = vel
                acts += 1
                if send_resets:
                    reset[b, i] = 1.0
            elif st == 0 and (gate_on or drone_on):
                st = 1
                tim = 0.0
                val = 0.0
                vel = max(0.0, float(velocity[b, i]))
                rel_start = vel
                acts += 1
                if send_resets:
                    reset[b, i] = 1.0

            if st == 1:
                if atk_frames <= 0:
                    val = vel
                    st = 2 if hold_frames > 0 else (3 if dec_frames > 0 else 4)
                    tim = 0.0
                else:
                    val += vel / max(atk_frames, 1)
                    if val > vel:
                        val = vel
                    tim += 1.0
                    if tim >= atk_frames:
                        val = vel
                        st = 2 if hold_frames > 0 else (3 if dec_frames > 0 else 4)
                        tim = 0.0
            elif st == 2:
                val = vel
                if hold_frames <= 0:
                    st = 3 if dec_frames > 0 else 4
                    tim = 0.0
                else:
                    tim += 1.0
                    if tim >= hold_frames:
                        st = 3 if dec_frames > 0 else 4
                        tim = 0.0
            elif st == 3:
                target = vel * sustain_level
                if dec_frames <= 0:
                    val = target
                    st = 4
                    tim = 0.0
                else:
                    delta = (vel - target) / max(dec_frames, 1)
                    val = max(target, val - delta)
                    tim += 1.0
                    if tim >= dec_frames:
                        val = target
                        st = 4
                        tim = 0.0
            elif st == 4:
                val = vel * sustain_level
                if sus_frames > 0:
                    tim += 1.0
                    if tim >= sus_frames:
                        st = 5
                        rel_start = val
                        tim = 0.0
                elif not gate_on and not drone_on:
                    st = 5
                    rel_start = val
                    tim = 0.0
            elif st == 5:
                if rel_frames <= 0:
                    val = 0.0
                    st = 0
                    tim = 0.0
                else:
                    step = rel_start / max(rel_frames, 1)
                    val = max(0.0, val - step)
                    tim += 1.0
                    if tim >= rel_frames:
                        val = 0.0
                        st = 0
                        tim = 0.0
                if gate_on or drone_on:
                    st = 1
                    tim = 0.0
                    val = 0.0
                    vel = max(0.0, float(velocity[b, i]))
                    rel_start = vel
                    acts += 1
                    if send_resets:
                        reset[b, i] = 1.0

            val = max(0.0, val)
            amp[b, i] = val

        stage[b] = st
        value[b] = val
        timer[b] = tim
        vel_state[b] = vel
        activations[b] = acts
        release_start[b] = rel_start

    return amp, reset


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

    # run Python reference implementation
    st_py = stage.copy()
    val_py = value.copy()
    tim_py = timer.copy()
    vel_py = vel_state.copy()
    acts_py = activations.copy()
    rel_py = release_start.copy()
    amp_py, reset_py = envelope_process_reference(
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

    if not c_kernels.AVAILABLE or c_kernels._impl is None:
        pytest.skip("C kernels not available")

    # run C kernel
    st_c = stage.copy()
    val_c = value.copy()
    tim_c = timer.copy()
    vel_c = vel_state.copy()
    acts_c = activations.copy()
    rel_c = release_start.copy()
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

    # compare outputs and states
    assert np.allclose(amp_py, amp_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(reset_py, reset_c, atol=1e-12, rtol=1e-9)
    assert np.array_equal(st_py, st_c)
    assert np.allclose(val_py, val_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(tim_py, tim_c, atol=1e-12, rtol=1e-9)
    assert np.allclose(vel_py, vel_c, atol=1e-12, rtol=1e-9)
    assert np.array_equal(acts_py, acts_c)
    assert np.allclose(rel_py, rel_c, atol=1e-12, rtol=1e-9)
