"""Blockwise envelope generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import RAW_DTYPE


@dataclass
class EnvelopeParams:
    """Static parameters that describe an ADSHR envelope."""

    attack_frames: int
    hold_frames: int
    decay_frames: int
    sustain_frames: int
    release_frames: int
    sustain_level: float
    send_resets: bool


def _start_attack(
    *,
    velocity: float,
    send_resets: bool,
    reset: np.ndarray,
    index: int,
    activations: int,
    stage_attack: int,
) -> tuple[int, float, float, float, float, int]:
    """Initialise a new attack stage and return updated state values."""

    vel = velocity if velocity > 0.0 else 0.0
    if send_resets:
        reset[index] = 1.0
    return (
        stage_attack,
        0.0,
        0.0,
        vel,
        vel,
        activations + 1,
    )


def _process_voice(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    params: EnvelopeParams,
    *,
    stage_state: int,
    value_state: float,
    timer_state: float,
    velocity_state: float,
    activation_state: int,
    release_start_state: float,
    stage_constants: dict[str, int],
    amp_out: np.ndarray,
    reset_out: np.ndarray,
) -> tuple[int, float, float, float, int, float]:
    """Process a single voice worth of envelope data."""

    atk_frames = params.attack_frames
    hold_frames = params.hold_frames
    dec_frames = params.decay_frames
    sus_frames = params.sustain_frames
    rel_frames = params.release_frames
    sustain_level = params.sustain_level
    send_resets = params.send_resets

    stage_attack = stage_constants["attack"]
    stage_hold = stage_constants["hold"]
    stage_decay = stage_constants["decay"]
    stage_sustain = stage_constants["sustain"]
    stage_release = stage_constants["release"]
    stage_idle = stage_constants["idle"]

    F = trigger.shape[0]

    # Pre-compute boundaries where controller state changes.
    trig_indices = np.flatnonzero(trigger > 0.5)
    gate_bool = gate > 0.5
    drone_bool = drone > 0.5
    gate_changes = np.flatnonzero(gate_bool[1:] != gate_bool[:-1]) + 1 if F > 1 else np.empty(0, dtype=np.int64)
    drone_changes = (
        np.flatnonzero(drone_bool[1:] != drone_bool[:-1]) + 1 if F > 1 else np.empty(0, dtype=np.int64)
    )

    boundaries = np.concatenate((
        np.array([0, F], dtype=np.int64),
        trig_indices.astype(np.int64, copy=False),
        gate_changes.astype(np.int64, copy=False),
        drone_changes.astype(np.int64, copy=False),
    ))
    boundaries = np.unique(boundaries)

    # Local state copies.
    st = int(stage_state)
    val = float(value_state)
    tim = float(timer_state)
    vel_state = float(velocity_state)
    acts = int(activation_state)
    rel_start = float(release_start_state)

    # Reset output line to zero as we write events explicitly.
    if reset_out is not None:
        reset_out.fill(0.0)

    def start_attack_at(index: int) -> None:
        nonlocal st, tim, val, vel_state, rel_start, acts
        st, tim, val, vel_state, rel_start, acts = _start_attack(
            velocity=float(velocity[index]),
            send_resets=send_resets,
            reset=reset_out,
            index=index,
            activations=acts,
            stage_attack=stage_attack,
        )

    boundary_iter = zip(boundaries[:-1], boundaries[1:])
    trig_ptr = 0

    for start, stop in boundary_iter:
        if start >= F:
            break

        # Process any triggers scheduled at the start of this segment.
        while trig_ptr < trig_indices.size and trig_indices[trig_ptr] == start:
            start_attack_at(start)
            trig_ptr += 1

        seg_end = min(stop, F)
        t = start
        while t < seg_end:
            gate_on = bool(gate_bool[t] or drone_bool[t])

            # Resolve zero-duration stages before emitting any samples.
            changed = True
            while changed:
                changed = False
                if st == stage_attack and atk_frames <= 0:
                    val = vel_state
                    st = stage_hold if hold_frames > 0 else (stage_decay if dec_frames > 0 else stage_sustain)
                    tim = 0.0
                    changed = True
                    continue
                if st == stage_hold and hold_frames <= 0:
                    st = stage_decay if dec_frames > 0 else stage_sustain
                    tim = 0.0
                    changed = True
                    continue
                if st == stage_decay and dec_frames <= 0:
                    val = vel_state * sustain_level
                    st = stage_sustain
                    tim = 0.0
                    changed = True
                    continue
                if st == stage_release and rel_frames <= 0:
                    val = 0.0
                    st = stage_idle
                    tim = 0.0
                    changed = True
                    continue

            if st == stage_idle:
                if gate_on:
                    start_attack_at(t)
                    continue
                amp_out[t:seg_end] = 0.0
                val = 0.0
                tim = 0.0
                t = seg_end
                continue

            if st == stage_attack:
                if atk_frames <= 0:
                    continue
                remaining = int(max(atk_frames - tim, 1))
                seg_len = min(seg_end - t, remaining)
                step = vel_state / float(max(atk_frames, 1))
                idxs = np.arange(1, seg_len + 1, dtype=RAW_DTYPE)
                seg_vals = val + step * idxs
                if vel_state >= 0.0:
                    np.minimum(seg_vals, vel_state, out=seg_vals)
                amp_out[t : t + seg_len] = np.maximum(seg_vals, 0.0)
                val = float(amp_out[t + seg_len - 1])
                tim += float(seg_len)
                if atk_frames > 0 and tim >= atk_frames:
                    val = vel_state
                    st = stage_hold if hold_frames > 0 else (stage_decay if dec_frames > 0 else stage_sustain)
                    tim = 0.0
                t += seg_len
                continue

            if st == stage_hold:
                if hold_frames <= 0:
                    continue
                remaining = int(max(hold_frames - tim, 1))
                seg_len = min(seg_end - t, remaining)
                amp_out[t : t + seg_len] = vel_state
                val = vel_state
                tim += float(seg_len)
                if tim >= hold_frames:
                    st = stage_decay if dec_frames > 0 else stage_sustain
                    tim = 0.0
                t += seg_len
                continue

            if st == stage_decay:
                if dec_frames <= 0:
                    continue
                remaining = int(max(dec_frames - tim, 1))
                seg_len = min(seg_end - t, remaining)
                target = vel_state * sustain_level
                delta = (vel_state - target) / float(max(dec_frames, 1))
                idxs = np.arange(1, seg_len + 1, dtype=RAW_DTYPE)
                seg_vals = val - delta * idxs
                np.maximum(seg_vals, target, out=seg_vals)
                amp_out[t : t + seg_len] = np.maximum(seg_vals, 0.0)
                val = float(amp_out[t + seg_len - 1])
                tim += float(seg_len)
                if tim >= dec_frames:
                    val = target
                    st = stage_sustain
                    tim = 0.0
                t += seg_len
                continue

            if st == stage_sustain:
                sustain_val = vel_state * sustain_level
                if sus_frames > 0:
                    remaining = int(max(sus_frames - tim, 1))
                    seg_len = min(seg_end - t, remaining)
                    amp_out[t : t + seg_len] = sustain_val
                    val = sustain_val
                    tim += float(seg_len)
                    if tim >= sus_frames:
                        st = stage_release
                        rel_start = val
                        tim = 0.0
                    t += seg_len
                    continue
                # No sustain timer – release when gate/drone drop.
                seg_len = min(seg_end - t, 1 if not gate_on else seg_end - t)
                if seg_len <= 0:
                    seg_len = 1
                amp_out[t : t + seg_len] = sustain_val
                val = sustain_val
                if not gate_on:
                    st = stage_release
                    rel_start = val
                    tim = 0.0
                else:
                    tim = 0.0
                t += seg_len
                continue

            if st == stage_release:
                if rel_frames <= 0:
                    continue
                remaining = int(max(rel_frames - tim, 1))
                seg_len = min(seg_end - t, 1 if gate_on else remaining)
                step = rel_start / float(max(rel_frames, 1))
                idxs = np.arange(1, seg_len + 1, dtype=RAW_DTYPE)
                seg_vals = val - step * idxs
                np.maximum(seg_vals, 0.0, out=seg_vals)
                amp_out[t : t + seg_len] = seg_vals
                val = float(amp_out[t + seg_len - 1])
                tim += float(seg_len)
                if tim >= rel_frames:
                    val = 0.0
                    st = stage_idle
                    tim = 0.0
                t_next = t + seg_len
                if gate_on:
                    # Restart attack for the next frame using the velocity at the event index.
                    restart_index = min(t_next - 1, F - 1)
                    st, tim, val, vel_state, rel_start, acts = _start_attack(
                        velocity=float(velocity[restart_index]),
                        send_resets=send_resets,
                        reset=reset_out,
                        index=restart_index,
                        activations=acts,
                        stage_attack=stage_attack,
                    )
                t = t_next
                continue

        if seg_end >= F:
            break

    return st, val, tim, vel_state, acts, rel_start


def envelope_process_block(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    *,
    params: EnvelopeParams,
    stage: np.ndarray,
    value: np.ndarray,
    timer: np.ndarray,
    vel_state: np.ndarray,
    activations: np.ndarray,
    release_start: np.ndarray,
    stage_constants: dict[str, int],
    out_amp: np.ndarray | None = None,
    out_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an envelope for an entire block using vectorised segments."""

    trigger = np.asarray(trigger, dtype=RAW_DTYPE)
    gate = np.asarray(gate, dtype=RAW_DTYPE)
    drone = np.asarray(drone, dtype=RAW_DTYPE)
    velocity = np.asarray(velocity, dtype=RAW_DTYPE)

    B, F = trigger.shape

    if out_amp is None:
        out_amp = np.empty((B, F), dtype=RAW_DTYPE)
    else:
        if out_amp.shape != (B, F):
            raise ValueError("out_amp has incorrect shape")
    if out_reset is None:
        out_reset = np.zeros((B, F), dtype=RAW_DTYPE)
    else:
        if out_reset.shape != (B, F):
            raise ValueError("out_reset has incorrect shape")
        out_reset.fill(0.0)

    for b in range(B):
        st, val, tim, vel, acts, rel = _process_voice(
            trigger[b],
            gate[b],
            drone[b],
            velocity[b],
            params,
            stage_state=int(stage[b]),
            value_state=float(value[b]),
            timer_state=float(timer[b]),
            velocity_state=float(vel_state[b]),
            activation_state=int(activations[b]),
            release_start_state=float(release_start[b]),
            stage_constants=stage_constants,
            amp_out=out_amp[b],
            reset_out=out_reset[b],
        )
        stage[b] = st
        value[b] = val
        timer[b] = tim
        vel_state[b] = vel
        activations[b] = acts
        release_start[b] = rel

    return out_amp, out_reset

