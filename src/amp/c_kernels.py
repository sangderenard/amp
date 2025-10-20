"""Optional C-backed kernels for tight loops.

This module attempts to build a small C kernel using cffi. If compilation
is not available in the environment (no compiler or cffi not installed),
the module exposes python fallbacks so callers can transparently fall back
to pure-Python/numpy implementations.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

AVAILABLE = False
_impl = None

try:
    import cffi
    ffi = cffi.FFI()
    ffi.cdef("""
    void lfo_slew(const double* x, double* out, int B, int F, double r, double alpha, double* z0);
    void safety_filter(const double* x, double* y, int B, int C, int F, double a, double* prev_in, double* prev_dc);
    void dc_block(const double* x, double* out, int B, int C, int F, double a, double* state);
    void subharmonic_process(
        const double* x,
        double* y,
        int B,
        int C,
        int F,
        double a_hp_in,
        double a_lp_in,
        double a_sub2,
        int use_div4,
        double a_sub4,
        double a_env_attack,
        double a_env_release,
        double a_hp_out,
        double drive,
        double mix,
        double* hp_y,
        double* lp_y,
        double* prev,
        int8_t* sign,
        int8_t* ff2,
        int8_t* ff4,
        int32_t* ff4_count,
        double* sub2_lp,
        double* sub4_lp,
        double* env,
        double* hp_out_y,
        double* hp_out_x
    );
    void envelope_process(
        const double* trigger,
        const double* gate,
        const double* drone,
        const double* velocity,
        int B,
        int F,
        int atk_frames,
        int hold_frames,
        int dec_frames,
        int sus_frames,
        int rel_frames,
        double sustain_level,
        int send_resets,
        int* stage,
        double* value,
        double* timer,
        double* vel_state,
        int64_t* activations,
        double* release_start,
        double* amp_out,
        double* reset_out
    );
    void phase_advance(const double* dphi, double* phase_out, int B, int F, double* phase_state, const double* reset);
    void portamento_smooth(const double* freq_target, const double* port_mask, const double* slide_time, const double* slide_damp, int B, int F, int sr, double* freq_state, double* out);
    void arp_advance(const double* seq, int seq_len, double* offsets_out, int B, int F, int* step_state, int* timer_state, int fps);
    void polyblep_arr(const double* t, const double* dt, double* out, int N);
    void osc_saw_blep_c(const double* ph, const double* dphi, double* out, int B, int F);
    void osc_square_blep_c(const double* ph, const double* dphi, double pw, double* out, int B, int F);
    void osc_triangle_blep_c(const double* ph, const double* dphi, double* out, int B, int F, double* tri_state);
    """)
    C_SRC = r"""
#include <math.h>
#include <stdint.h>
void lfo_slew(const double* x, double* out, int B, int F, double r, double alpha, double* z0) {
    for (int b = 0; b < B; ++b) {
        double state = 0.0;
        if (z0 != NULL) state = z0[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double xi = x[base + i];
            state = r * state + alpha * xi;
            out[base + i] = state;
        }
        if (z0 != NULL) z0[b] = state;
    }
}

void safety_filter(const double* x, double* y, int B, int C, int F, double a, double* prev_in, double* prev_dc) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int chan = b * C + c;
            double pi = 0.0;
            double pd = 0.0;
            if (prev_in != NULL) pi = prev_in[chan];
            if (prev_dc != NULL) pd = prev_dc[chan];
            int base = chan * F;
            for (int i = 0; i < F; ++i) {
                double xin = x[base + i];
                double diff;
                if (i == 0) diff = xin - pi;
                else diff = xin - x[base + i - 1];
                pd = a * pd + diff;
                y[base + i] = pd;
            }
            if (prev_in != NULL) prev_in[chan] = x[base + F - 1];
            if (prev_dc != NULL) prev_dc[chan] = y[base + F - 1];
        }
    }
}

void dc_block(const double* x, double* out, int B, int C, int F, double a, double* state) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            int chan = b * C + c;
            double dc = 0.0;
            if (state != NULL) dc = state[chan];
            int base = chan * F;
            for (int i = 0; i < F; ++i) {
                double xin = x[base + i];
                dc = a * dc + (1.0 - a) * xin;
                out[base + i] = xin - dc;
            }
            if (state != NULL) state[chan] = dc;
        }
    }
}

void subharmonic_process(
    const double* x,
    double* y,
    int B,
    int C,
    int F,
    double a_hp_in,
    double a_lp_in,
    double a_sub2,
    int use_div4,
    double a_sub4,
    double a_env_attack,
    double a_env_release,
    double a_hp_out,
    double drive,
    double mix,
    double* hp_y,
    double* lp_y,
    double* prev,
    int8_t* sign,
    int8_t* ff2,
    int8_t* ff4,
    int32_t* ff4_count,
    double* sub2_lp,
    double* sub4_lp,
    double* env,
    double* hp_out_y,
    double* hp_out_x
) {
    // Layout: arrays are flattened per-channel: index = (b*C + c) * F + t
    for (int t = 0; t < F; ++t) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C; ++c) {
                int chan = b * C + c;
                int base = chan * F;
                double xt = x[base + t];

                // Bandpass driver: simple HP then LP
                double hp_y_val = hp_y[chan];
                double prev_val = prev[chan];
                double lp_y_val = lp_y[chan];
                double hp_in = a_hp_in * (hp_y_val + xt - prev_val);
                hp_y[chan] = hp_in;
                prev[chan] = xt;
                double bp = lp_y_val + a_lp_in * (hp_in - lp_y_val);
                lp_y[chan] = bp;

                // env
                double abs_bp = fabs(bp);
                double env_val = env[chan];
                if (abs_bp > env_val) env_val = env_val + a_env_attack * (abs_bp - env_val);
                else env_val = env_val + a_env_release * (abs_bp - env_val);
                env[chan] = env_val;

                // sign and flip-flops
                int8_t prev_sign = sign[chan];
                int8_t sign_now = (bp > 0.0) ? 1 : -1;
                int pos_zc = (prev_sign < 0) && (sign_now > 0);
                sign[chan] = sign_now;

                if (pos_zc) ff2[chan] = -ff2[chan];

                if (use_div4) {
                    if (pos_zc) ff4_count[chan] = ff4_count[chan] + 1;
                    int toggle4 = (pos_zc && (ff4_count[chan] >= 2));
                    if (toggle4) ff4[chan] = -ff4[chan];
                    if (toggle4) ff4_count[chan] = 0;
                }

                double sq2 = (double) ff2[chan];
                double sub2_lp_val = sub2_lp[chan];
                sub2_lp_val = sub2_lp_val + a_sub2 * (sq2 - sub2_lp_val);
                sub2_lp[chan] = sub2_lp_val;
                double sub_val = sub2_lp_val;

                if (use_div4) {
                    double sq4 = (double) ff4[chan];
                    double sub4_lp_val = sub4_lp[chan];
                    sub4_lp_val = sub4_lp_val + a_sub4 * (sq4 - sub4_lp_val);
                    sub4_lp[chan] = sub4_lp_val;
                    sub_val = sub_val + 0.6 * sub4_lp_val;
                }

                double sub = tanh(drive * sub_val) * (env_val + 1e-6);

                double dry = xt;
                double wet = sub;
                double out_t = (1.0 - mix) * dry + mix * wet;

                double y_prev = hp_out_y[chan];
                double x_prev = hp_out_x[chan];
                double hp = a_hp_out * (y_prev + out_t - x_prev);
                hp_out_y[chan] = hp;
                hp_out_x[chan] = out_t;
                y[base + t] = hp;
            }
        }
    }
}

void envelope_process(
    const double* trigger,
    const double* gate,
    const double* drone,
    const double* velocity,
    int B,
    int F,
    int atk_frames,
    int hold_frames,
    int dec_frames,
    int sus_frames,
    int rel_frames,
    double sustain_level,
    int send_resets,
    int* stage,
    double* value,
    double* timer,
    double* vel_state,
    int64_t* activations,
    double* release_start,
    double* amp_out,
    double* reset_out
) {
    for (int b = 0; b < B; ++b) {
        int st = stage[b];
        double val = value[b];
        double tim = timer[b];
        double vel = vel_state[b];
        int64_t acts = activations[b];
        double rel_start = release_start[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            int idx = base + i;
            int trig = trigger[idx] > 0.5 ? 1 : 0;
            int gate_on = gate[idx] > 0.5 ? 1 : 0;
            int drone_on = drone[idx] > 0.5 ? 1 : 0;
            int send_reset_line = send_resets ? (reset_out != NULL && reset_out[idx]) : 0; // placeholder, will handle below

            if (trig) {
                st = 1; // ATTACK
                tim = 0.0;
                val = 0.0;
                vel = velocity[idx] > 0.0 ? velocity[idx] : 0.0;
                rel_start = vel;
                acts += 1;
                if (send_resets && reset_out != NULL) reset_out[idx] = 1.0;
            } else if (st == 0 && (gate_on || drone_on)) {
                st = 1;
                tim = 0.0;
                val = 0.0;
                vel = velocity[idx] > 0.0 ? velocity[idx] : 0.0;
                rel_start = vel;
                acts += 1;
                if (send_resets && reset_out != NULL) reset_out[idx] = 1.0;
            }

            if (st == 1) { // ATTACK
                if (atk_frames <= 0) {
                    val = vel;
                    if (hold_frames > 0) st = 2; else if (dec_frames > 0) st = 3; else st = 4;
                    tim = 0.0;
                } else {
                    val += vel / (double)atk_frames;
                    if (val > vel) val = vel;
                    tim += 1.0;
                    if (tim >= atk_frames) {
                        val = vel;
                        if (hold_frames > 0) st = 2; else if (dec_frames > 0) st = 3; else st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 2) { // HOLD
                val = vel;
                if (hold_frames <= 0) {
                    if (dec_frames > 0) st = 3; else st = 4;
                    tim = 0.0;
                } else {
                    tim += 1.0;
                    if (tim >= hold_frames) { if (dec_frames > 0) st = 3; else st = 4; tim = 0.0; }
                }
            } else if (st == 3) { // DECAY
                double target = vel * sustain_level;
                if (dec_frames <= 0) { val = target; st = 4; tim = 0.0; }
                else {
                    double delta = (vel - target) / (double)(dec_frames > 0 ? dec_frames : 1);
                    if (val - delta < target) val = target; else val = val - delta;
                    tim += 1.0;
                    if (tim >= dec_frames) { val = target; st = 4; tim = 0.0; }
                }
            } else if (st == 4) { // SUSTAIN
                val = vel * sustain_level;
                if (sus_frames > 0) {
                    tim += 1.0;
                    if (tim >= sus_frames) { st = 5; rel_start = val; tim = 0.0; }
                } else if (!gate_on && !drone_on) {
                    st = 5; rel_start = val; tim = 0.0;
                }
            } else if (st == 5) { // RELEASE
                if (rel_frames <= 0) { val = 0.0; st = 0; tim = 0.0; }
                else {
                    double step = rel_start / (double)(rel_frames > 0 ? rel_frames : 1);
                    if (val - step < 0.0) val = 0.0; else val = val - step;
                    tim += 1.0;
                    if (tim >= rel_frames) { val = 0.0; st = 0; tim = 0.0; }
                }
                if (gate_on || drone_on) {
                    st = 1; tim = 0.0; val = 0.0; vel = velocity[idx] > 0.0 ? velocity[idx] : 0.0; rel_start = vel; acts += 1;
                    if (send_resets && reset_out != NULL) reset_out[idx] = 1.0;
                }
            }

            if (val < 0.0) val = 0.0;
            amp_out[idx] = val;
        }
        stage[b] = st;
        value[b] = val;
        timer[b] = tim;
        vel_state[b] = vel;
        activations[b] = acts;
        release_start[b] = rel_start;
    }
}

// Advance phase per frame with optional reset line. dphi and phase_state are arrays of length B*F and B respectively
void phase_advance(const double* dphi, double* phase_out, int B, int F, double* phase_state, const double* reset) {
    for (int b = 0; b < B; ++b) {
        double cur = 0.0;
        if (phase_state != NULL) cur = phase_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            if (reset != NULL && reset[base + i] > 0.5) cur = 0.0;
            cur = cur + dphi[base + i];
            // wrap into [0,1)
            cur = cur - floor(cur);
            phase_out[base + i] = cur;
        }
        if (phase_state != NULL) phase_state[b] = cur;
    }
}

// Portamento smoothing: per-frame smoothing with alpha derived from slide_time and slide_damp
void portamento_smooth(const double* freq_target, const double* port_mask, const double* slide_time, const double* slide_damp, int B, int F, int sr, double* freq_state, double* out) {
    for (int b = 0; b < B; ++b) {
        double cur = 0.0;
        if (freq_state != NULL) cur = freq_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double target = freq_target[base + i];
            int active = port_mask != NULL && port_mask[base + i] > 0.5 ? 1 : 0;
            double frames_const = slide_time != NULL ? slide_time[base + i] * (double)sr : 1.0;
            if (frames_const < 1.0) frames_const = 1.0;
            double alpha = exp(-1.0 / frames_const);
            if (slide_damp != NULL) alpha = pow(alpha, 1.0 + fmax(0.0, slide_damp[base + i]));
            if (active) cur = alpha * cur + (1.0 - alpha) * target;
            else cur = target;
            out[base + i] = cur;
        }
        if (freq_state != NULL) freq_state[b] = cur;
    }
}

// Arp advance: write offsets per frame, update step/timer states
void arp_advance(const double* seq, int seq_len, double* offsets_out, int B, int F, int* step_state, int* timer_state, int fps) {
    for (int b = 0; b < B; ++b) {
        int step = 0;
        int timer = 0;
        if (step_state != NULL) step = step_state[b];
        if (timer_state != NULL) timer = timer_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            int idx = step % (seq_len > 0 ? seq_len : 1);
            offsets_out[base + i] = seq[idx];
            timer += 1;
            if (timer >= fps) {
                timer = 0;
                step = (step + 1) % (seq_len > 0 ? seq_len : 1);
            }
        }
        if (step_state != NULL) step_state[b] = step;
        if (timer_state != NULL) timer_state[b] = timer;
    }
}

void polyblep_arr(const double* t, const double* dt, double* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = 0.0;
    }
    for (int i = 0; i < N; ++i) {
        double ti = t[i];
        double dti = dt[i];
        if (ti < dti) {
            double x = ti / (dti > 0.0 ? dti : 1e-20);
            out[i] = x + x - x * x - 1.0;
        } else if (ti > 1.0 - dti) {
            double x = (ti - 1.0) / (dti > 0.0 ? dti : 1e-20);
            out[i] = x * x + x + x + 1.0;
        } else {
            out[i] = 0.0;
        }
    }
}

void osc_saw_blep_c(const double* ph, const double* dphi, double* out, int B, int F) {
    int N = B * F;
    for (int i = 0; i < N; ++i) {
        double t = ph[i];
        double y = 2.0 * t - 1.0;
        double pb = 0.0;
        double dti = dphi[i];
        if (t < dti) {
            double x = t / (dti > 0.0 ? dti : 1e-20);
            pb = x + x - x * x - 1.0;
        } else if (t > 1.0 - dti) {
            double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb = x * x + x + x + 1.0;
        }
        out[i] = y - pb;
    }
}

void osc_square_blep_c(const double* ph, const double* dphi, double pw, double* out, int B, int F) {
    int N = B * F;
    for (int i = 0; i < N; ++i) {
        double t = ph[i];
        double y = (t < pw) ? 1.0 : -1.0;
        // subtract polyblep at rising edge
        double pb1 = 0.0;
        double dti = dphi[i];
        if (t < dti) {
            double x = t / (dti > 0.0 ? dti : 1e-20);
            pb1 = x + x - x * x - 1.0;
        } else if (t > 1.0 - dti) {
            double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb1 = x * x + x + x + 1.0;
        }
        // add polyblep at falling edge (t + (1-pw))%1
        double t2 = t + (1.0 - pw);
        if (t2 >= 1.0) t2 -= 1.0;
        double pb2 = 0.0;
        if (t2 < dti) {
            double x = t2 / (dti > 0.0 ? dti : 1e-20);
            pb2 = x + x - x * x - 1.0;
        } else if (t2 > 1.0 - dti) {
            double x = (t2 - 1.0) / (dti > 0.0 ? dti : 1e-20);
            pb2 = x * x + x + x + 1.0;
        }
        out[i] = y - pb1 + pb2;
    }
}

void osc_triangle_blep_c(const double* ph, const double* dphi, double* out, int B, int F, double* tri_state) {
    int N = B * F;
    // Use square -> leaky integrator per-batch sequence
    for (int b = 0; b < B; ++b) {
        double s = 0.0;
        if (tri_state != NULL) s = tri_state[b];
        int base = b * F;
        for (int i = 0; i < F; ++i) {
            double t = ph[base + i];
            // square
            double y = (t < 0.5) ? 1.0 : -1.0;
            // blep corrections around edges
            double dti = dphi[base + i];
            double pb1 = 0.0;
            if (t < dti) {
                double x = t / (dti > 0.0 ? dti : 1e-20);
                pb1 = x + x - x * x - 1.0;
            } else if (t > 1.0 - dti) {
                double x = (t - 1.0) / (dti > 0.0 ? dti : 1e-20);
                pb1 = x * x + x + x + 1.0;
            }
            double t2 = t + 0.5; if (t2 >= 1.0) t2 -= 1.0;
            double pb2 = 0.0;
            if (t2 < dti) {
                double x = t2 / (dti > 0.0 ? dti : 1e-20);
                pb2 = x + x - x * x - 1.0;
            } else if (t2 > 1.0 - dti) {
                double x = (t2 - 1.0) / (dti > 0.0 ? dti : 1e-20);
                pb2 = x * x + x + x + 1.0;
            }
            double sq = y - pb1 + pb2;
            double leak = 0.9995;
            s = leak * s + (1.0 - leak) * sq;
            out[base + i] = s;
        }
        if (tri_state != NULL) tri_state[b] = s;
    }
}
"""
    try:
        ffi.set_source("_amp_ckernels_cffi", C_SRC)
        # compile lazy; this will create a module in-place
        ffi.compile(verbose=False)
        import importlib
        _impl = importlib.import_module("_amp_ckernels_cffi")
        AVAILABLE = True
    except Exception:
        # any compile/import error -> disable C backend
        AVAILABLE = False
except Exception:
    AVAILABLE = False


def lfo_slew_c(x: np.ndarray, r: float, alpha: float, z0: Optional[np.ndarray]) -> np.ndarray:
    """Call the compiled C kernel to compute exponential smoothing.

    x: (B, F) contiguous C-order array of doubles
    r: feedback coefficient
    alpha: feed coefficient
    z0: optional (B,) array of initial states (modified in-place)

    Returns out (B, F) same dtype.
    Raises RuntimeError if C backend is unavailable.
    """
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    B, F = x.shape
    out = np.empty_like(x)
    zbuf = None
    if z0 is not None:
        if z0.shape[0] != B:
            raise ValueError("z0 must have shape (B,)")
        if not z0.flags['C_CONTIGUOUS']:
            z0 = np.ascontiguousarray(z0)
        zbuf = z0
    # get C pointers
    x_ptr = ffi.cast("const double *", x.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    if zbuf is not None:
        z_ptr = ffi.cast("double *", zbuf.ctypes.data)
    else:
        z_ptr = ffi.cast("double *", ffi.NULL)
    _impl.lib.lfo_slew(x_ptr, out_ptr, int(B), int(F), float(r), float(alpha), z_ptr)
    # if zbuf provided, copy back into original z0
    if zbuf is not None and z0 is not zbuf:
        z0[:] = zbuf
    return out


def lfo_slew_py(x: np.ndarray, r: float, alpha: float, z0: Optional[np.ndarray]) -> np.ndarray:
    """Pure-Python sample-sequential fallback (fast with numpy per-row ops).

    Semantics: iterative recurrence z[n] = r*z[n-1] + alpha * x[n].
    """
    B, F = x.shape
    out = np.empty_like(x)
    if z0 is None:
        z = np.zeros(B, dtype=x.dtype)
    else:
        z = z0.copy()
    for i in range(F):
        xi = x[:, i]
        z = r * z + alpha * xi
        out[:, i] = z
    if z0 is not None:
        z0[:] = z
    return out


def lfo_slew_vector(x: np.ndarray, r: float, alpha: float, z0: Optional[np.ndarray]) -> np.ndarray:
    """Vectorized closed-form solution equivalent to iterative recurrence.

    z[n] = r^n * z0 + alpha * r^n * sum_{k=0..n} r^{-k} * x[k]
    Implemented using np.cumsum on axis 1.
    """
    B, F = x.shape
    idx = np.arange(F, dtype=x.dtype)
    r_pow = r ** idx
    # handle r==0
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.where(r == 0.0, 0.0, r ** (-idx))
    accum = np.cumsum(x * r_inv[None, :], axis=1)
    out = (r_pow[None, :] * (alpha * accum))
    if z0 is not None:
        out = out + (r_pow[None, :] * z0[:, None])
        z0[:] = out[:, -1]
    return out


def safety_filter_c(x: np.ndarray, a: float, prev_in: Optional[np.ndarray], prev_dc: Optional[np.ndarray]) -> np.ndarray:
    """Call compiled safety_filter kernel. x shape (B,C,F) -> returns y (B,C,F)
    prev_in and prev_dc are optional (B,C) arrays and will be updated in-place if provided.
    """
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, C, F = x.shape
    out = np.empty_like(x)
    xb = np.ascontiguousarray(x)
    outb = np.ascontiguousarray(out)
    prev_in_buf = None
    prev_dc_buf = None
    if prev_in is not None:
        prev_in_buf = np.ascontiguousarray(prev_in)
    if prev_dc is not None:
        prev_dc_buf = np.ascontiguousarray(prev_dc)
    x_ptr = ffi.cast("const double *", xb.ctypes.data)
    y_ptr = ffi.cast("double *", outb.ctypes.data)
    prev_in_ptr = ffi.cast("double *", prev_in_buf.ctypes.data) if prev_in_buf is not None else ffi.cast("double *", ffi.NULL)
    prev_dc_ptr = ffi.cast("double *", prev_dc_buf.ctypes.data) if prev_dc_buf is not None else ffi.cast("double *", ffi.NULL)
    _impl.lib.safety_filter(x_ptr, y_ptr, int(B), int(C), int(F), float(a), prev_in_ptr, prev_dc_ptr)
    if prev_in_buf is not None and prev_in is not prev_in_buf:
        prev_in[:] = prev_in_buf
    if prev_dc_buf is not None and prev_dc is not prev_dc_buf:
        prev_dc[:] = prev_dc_buf
    return outb


def safety_filter_py(x: np.ndarray, a: float, prev_in: Optional[np.ndarray], prev_dc: Optional[np.ndarray]) -> np.ndarray:
    B, C, F = x.shape
    out = np.empty_like(x)
    pi = np.zeros((B, C), dtype=x.dtype) if prev_in is None else prev_in.copy()
    pd = np.zeros((B, C), dtype=x.dtype) if prev_dc is None else prev_dc.copy()
    for b in range(B):
        for c in range(C):
            if F <= 0:
                continue
            # compute diffs
            diffs = np.empty(F, dtype=x.dtype)
            diffs[0] = x[b, c, 0] - pi[b, c]
            if F > 1:
                diffs[1:] = x[b, c, 1:] - x[b, c, :-1]
            powers = a ** np.arange(F, dtype=x.dtype)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_p = 1.0 / powers
            accum = np.cumsum(diffs * inv_p) + (a * pd[b, c])
            y = accum * powers
            out[b, c, :] = y
            pi[b, c] = x[b, c, -1]
            pd[b, c] = y[-1]
    if prev_in is not None:
        prev_in[:] = pi
    if prev_dc is not None:
        prev_dc[:] = pd
    return out


def dc_block_c(x: np.ndarray, a: float, state: Optional[np.ndarray]) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, C, F = x.shape
    out = np.empty_like(x)
    xb = np.ascontiguousarray(x)
    outb = np.ascontiguousarray(out)
    state_buf = None
    if state is not None:
        state_buf = np.ascontiguousarray(state)
    x_ptr = ffi.cast("const double *", xb.ctypes.data)
    out_ptr = ffi.cast("double *", outb.ctypes.data)
    state_ptr = ffi.cast("double *", state_buf.ctypes.data) if state_buf is not None else ffi.cast("double *", ffi.NULL)
    _impl.lib.dc_block(x_ptr, out_ptr, int(B), int(C), int(F), float(a), state_ptr)
    if state_buf is not None and state is not state_buf:
        state[:] = state_buf
    return outb


def dc_block_py(x: np.ndarray, a: float, state: Optional[np.ndarray]) -> np.ndarray:
    B, C, F = x.shape
    out = np.empty_like(x)
    st = np.zeros((B, C), dtype=x.dtype) if state is None else state.copy()
    for b in range(B):
        for c in range(C):
            dc = st[b, c]
            for i in range(F):
                xi = x[b, c, i]
                dc = a * dc + (1.0 - a) * xi
                out[b, c, i] = xi - dc
            st[b, c] = dc
    if state is not None:
        state[:] = st
    return out


def subharmonic_process_c(
    x: np.ndarray,
    a_hp_in: float,
    a_lp_in: float,
    a_sub2: float,
    use_div4: bool,
    a_sub4: float,
    a_env_attack: float,
    a_env_release: float,
    a_hp_out: float,
    drive: float,
    mix: float,
    hp_y: np.ndarray,
    lp_y: np.ndarray,
    prev: np.ndarray,
    sign: np.ndarray,
    ff2: np.ndarray,
    ff4: np.ndarray | None,
    ff4_count: np.ndarray | None,
    sub2_lp: np.ndarray,
    sub4_lp: np.ndarray | None,
    env: np.ndarray,
    hp_out_y: np.ndarray,
    hp_out_x: np.ndarray,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, C, F = x.shape
    xb = np.ascontiguousarray(x)
    out = np.empty_like(xb)
    outb = np.ascontiguousarray(out)

    # ensure buffers
    hp_y_b = np.ascontiguousarray(hp_y)
    lp_y_b = np.ascontiguousarray(lp_y)
    prev_b = np.ascontiguousarray(prev)
    sign_b = np.ascontiguousarray(sign.astype(np.int8))
    ff2_b = np.ascontiguousarray(ff2.astype(np.int8))
    ff4_b = np.ascontiguousarray(ff4.astype(np.int8)) if ff4 is not None else ffi.cast("int8_t *", ffi.NULL)
    ff4_count_b = np.ascontiguousarray(ff4_count.astype(np.int32)) if ff4_count is not None else ffi.cast("int32_t *", ffi.NULL)
    sub2_lp_b = np.ascontiguousarray(sub2_lp)
    sub4_lp_b = np.ascontiguousarray(sub4_lp) if sub4_lp is not None else ffi.cast("double *", ffi.NULL)
    env_b = np.ascontiguousarray(env)
    hp_out_y_b = np.ascontiguousarray(hp_out_y)
    hp_out_x_b = np.ascontiguousarray(hp_out_x)

    x_ptr = ffi.cast("const double *", xb.ctypes.data)
    y_ptr = ffi.cast("double *", outb.ctypes.data)
    hp_y_ptr = ffi.cast("double *", hp_y_b.ctypes.data)
    lp_y_ptr = ffi.cast("double *", lp_y_b.ctypes.data)
    prev_ptr = ffi.cast("double *", prev_b.ctypes.data)
    sign_ptr = ffi.cast("int8_t *", sign_b.ctypes.data)
    ff2_ptr = ffi.cast("int8_t *", ff2_b.ctypes.data)
    ff4_ptr = ffi.cast("int8_t *", ff4_b.ctypes.data) if ff4 is not None else ffi.cast("int8_t *", ffi.NULL)
    ff4_count_ptr = ffi.cast("int32_t *", ff4_count_b.ctypes.data) if ff4_count is not None else ffi.cast("int32_t *", ffi.NULL)
    sub2_lp_ptr = ffi.cast("double *", sub2_lp_b.ctypes.data)
    sub4_lp_ptr = ffi.cast("double *", sub4_lp_b.ctypes.data) if sub4_lp is not None else ffi.cast("double *", ffi.NULL)
    env_ptr = ffi.cast("double *", env_b.ctypes.data)
    hp_out_y_ptr = ffi.cast("double *", hp_out_y_b.ctypes.data)
    hp_out_x_ptr = ffi.cast("double *", hp_out_x_b.ctypes.data)

    _impl.lib.subharmonic_process(
        x_ptr,
        y_ptr,
        int(B),
        int(C),
        int(F),
        float(a_hp_in),
        float(a_lp_in),
        float(a_sub2),
        int(1 if use_div4 else 0),
        float(a_sub4),
        float(a_env_attack),
        float(a_env_release),
        float(a_hp_out),
        float(drive),
        float(mix),
        hp_y_ptr,
        lp_y_ptr,
        prev_ptr,
        sign_ptr,
        ff2_ptr,
        ff4_ptr,
        ff4_count_ptr,
        sub2_lp_ptr,
        sub4_lp_ptr,
        env_ptr,
        hp_out_y_ptr,
        hp_out_x_ptr,
    )

    # copy back mutable state
    hp_y[:] = hp_y_b
    lp_y[:] = lp_y_b
    prev[:] = prev_b
    sign[:] = sign_b
    ff2[:] = ff2_b
    if ff4 is not None:
        ff4[:] = ff4_b
    if ff4_count is not None:
        ff4_count[:] = ff4_count_b
    sub2_lp[:] = sub2_lp_b
    if sub4_lp is not None:
        sub4_lp[:] = sub4_lp_b
    env[:] = env_b
    hp_out_y[:] = hp_out_y_b
    hp_out_x[:] = hp_out_x_b

    return outb


def subharmonic_process_py(
    x: np.ndarray,
    a_hp_in: float,
    a_lp_in: float,
    a_sub2: float,
    use_div4: bool,
    a_sub4: float,
    a_env_attack: float,
    a_env_release: float,
    a_hp_out: float,
    drive: float,
    mix: float,
    hp_y: np.ndarray,
    lp_y: np.ndarray,
    prev: np.ndarray,
    sign: np.ndarray,
    ff2: np.ndarray,
    ff4: np.ndarray | None,
    ff4_count: np.ndarray | None,
    sub2_lp: np.ndarray,
    sub4_lp: np.ndarray | None,
    env: np.ndarray,
    hp_out_y: np.ndarray,
    hp_out_x: np.ndarray,
) -> np.ndarray:
    B, C, F = x.shape
    y = np.empty_like(x)
    for t in range(F):
        xt = x[:, :, t]

        # Bandpass driver: simple HP then LP
        hp_y[:] = a_hp_in * (hp_y + xt - prev)
        prev[:] = xt
        bp = lp_y + a_lp_in * (hp_y - lp_y)
        lp_y[:] = bp

        abs_bp = np.abs(bp)
        env[:] = np.where(
            abs_bp > env,
            env + a_env_attack * (abs_bp - env),
            env + a_env_release * (abs_bp - env),
        )

        prev_sign = sign.copy()
        sign_now = (bp > 0.0).astype(np.int8) * 2 - 1
        pos_zc = (prev_sign < 0) & (sign_now > 0)
        sign[:] = sign_now

        ff2[:] = np.where(pos_zc, -ff2, ff2)

        if use_div4 and ff4 is not None and ff4_count is not None:
            ff4_count[:] = np.where(pos_zc, ff4_count + 1, ff4_count)
            toggle4 = pos_zc & (ff4_count >= 2)
            ff4[:] = np.where(toggle4, -ff4, ff4)
            ff4_count[:] = np.where(toggle4, 0, ff4_count)

        sq2 = ff2.astype(x.dtype)
        sub2_lp[:] = sub2_lp + a_sub2 * (sq2 - sub2_lp)
        sub = sub2_lp.copy()

        if use_div4 and sub4_lp is not None and ff4 is not None:
            sq4 = ff4.astype(x.dtype)
            sub4_lp[:] = sub4_lp + a_sub4 * (sq4 - sub4_lp)
            sub = sub + 0.6 * sub4_lp

        sub = np.tanh(drive * sub) * (env + 1e-6)

        dry = xt
        wet = sub
        out_t = (1.0 - mix) * dry + mix * wet

        y_prev = hp_out_y.copy()
        x_prev = hp_out_x.copy()
        hp = a_hp_out * (y_prev + out_t - x_prev)
        hp_out_y[:] = hp
        hp_out_x[:] = out_t
        y[:, :, t] = hp

    return y


def phase_advance_c(dphi: np.ndarray, reset: np.ndarray | None, phase_state: np.ndarray | None) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    if not dphi.flags['C_CONTIGUOUS']:
        dphi = np.ascontiguousarray(dphi)
    B, F = dphi.shape
    out = np.empty_like(dphi)
    dphi_ptr = ffi.cast("const double *", dphi.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    state_ptr = ffi.cast("double *", np.ascontiguousarray(phase_state).ctypes.data) if phase_state is not None else ffi.cast("double *", ffi.NULL)
    reset_buf = np.ascontiguousarray(reset) if reset is not None else None
    reset_ptr = ffi.cast("const double *", reset_buf.ctypes.data) if reset_buf is not None else ffi.cast("const double *", ffi.NULL)
    _impl.lib.phase_advance(dphi_ptr, out_ptr, int(B), int(F), state_ptr, reset_ptr)
    if phase_state is not None:
        phase_state[:] = np.ascontiguousarray(state_ptr)[:B]
    return out


def phase_advance_py(dphi: np.ndarray, reset: np.ndarray | None, phase_state: np.ndarray | None) -> np.ndarray:
    B, F = dphi.shape
    out = np.empty_like(dphi)
    if phase_state is None:
        cur = np.zeros(B, dtype=dphi.dtype)
    else:
        cur = phase_state.copy()
    for i in range(F):
        if reset is not None:
            mask = reset[:, i] > 0.5
            if np.any(mask):
                cur = np.where(mask, 0.0, cur)
        cur = (cur + dphi[:, i]) % 1.0
        out[:, i] = cur
    if phase_state is not None:
        phase_state[:] = cur
    return out


def portamento_smooth_c(freq_target: np.ndarray, port_mask: np.ndarray | None, slide_time: np.ndarray | None, slide_damp: np.ndarray | None, sr: int, freq_state: np.ndarray | None) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = freq_target.shape
    out = np.empty_like(freq_target)
    ft_ptr = ffi.cast("const double *", np.ascontiguousarray(freq_target).ctypes.data)
    port_ptr = ffi.cast("const double *", np.ascontiguousarray(port_mask).ctypes.data) if port_mask is not None else ffi.cast("const double *", ffi.NULL)
    st_ptr = ffi.cast("const double *", np.ascontiguousarray(slide_time).ctypes.data) if slide_time is not None else ffi.cast("const double *", ffi.NULL)
    sd_ptr = ffi.cast("const double *", np.ascontiguousarray(slide_damp).ctypes.data) if slide_damp is not None else ffi.cast("const double *", ffi.NULL)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    state_ptr = ffi.cast("double *", np.ascontiguousarray(freq_state).ctypes.data) if freq_state is not None else ffi.cast("double *", ffi.NULL)
    _impl.lib.portamento_smooth(ft_ptr, port_ptr, st_ptr, sd_ptr, int(B), int(F), int(sr), state_ptr, out_ptr)
    if freq_state is not None:
        freq_state[:] = np.ascontiguousarray(state_ptr)[:B]
    return out


def portamento_smooth_py(freq_target: np.ndarray, port_mask: np.ndarray | None, slide_time: np.ndarray | None, slide_damp: np.ndarray | None, sr: int, freq_state: np.ndarray | None) -> np.ndarray:
    B, F = freq_target.shape
    out = np.empty_like(freq_target)
    cur = np.zeros(B, dtype=freq_target.dtype) if freq_state is None else freq_state.copy()
    for i in range(F):
        target = freq_target[:, i]
        active = port_mask[:, i] if port_mask is not None else np.zeros(B, dtype=bool)
        frames_const = np.maximum(slide_time[:, i] * float(sr) if slide_time is not None else 1.0, 1.0)
        alpha = np.exp(-1.0 / frames_const)
        if slide_damp is not None:
            alpha = alpha ** (1.0 + np.clip(slide_damp[:, i], 0.0, None))
        cur = np.where(active, alpha * cur + (1.0 - alpha) * target, target)
        out[:, i] = cur
    if freq_state is not None:
        freq_state[:] = cur
    return out


def arp_advance_c(seq: np.ndarray, seq_len: int, B: int, F: int, step_state: np.ndarray, timer_state: np.ndarray, fps: int) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    seq_buf = np.ascontiguousarray(seq.astype(np.float64))
    out = np.empty((B, F), dtype=np.float64)
    seq_ptr = ffi.cast("const double *", seq_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    step_ptr = ffi.cast("int *", np.ascontiguousarray(step_state).ctypes.data)
    timer_ptr = ffi.cast("int *", np.ascontiguousarray(timer_state).ctypes.data)
    _impl.lib.arp_advance(seq_ptr, int(seq_len), out_ptr, int(B), int(F), step_ptr, timer_ptr, int(fps))
    step_state[:] = np.ascontiguousarray(step_ptr)[:B]
    timer_state[:] = np.ascontiguousarray(timer_ptr)[:B]
    return out


def arp_advance_py(seq: np.ndarray, seq_len: int, B: int, F: int, step_state: np.ndarray, timer_state: np.ndarray, fps: int) -> np.ndarray:
    out = np.empty((B, F), dtype=float)
    seq_list = list(np.asarray(seq, dtype=float).ravel())
    if len(seq_list) == 0:
        seq_list = [0.0]
    step = step_state.copy()
    timer = timer_state.copy()
    for i in range(F):
        idx = step % (len(seq_list) if len(seq_list) > 0 else 1)
        out[:, i] = np.asarray(seq_list)[idx]
        timer += 1
        reached = timer >= fps
        if np.any(reached):
            timer[reached] = 0
            step[reached] = (step[reached] + 1) % len(seq_list)
    step_state[:] = step
    timer_state[:] = timer
    return out


def _polyblep_arr_c(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    t_b = np.ascontiguousarray(t)
    dt_b = np.ascontiguousarray(dt)
    out = np.empty_like(t_b)
    _impl.lib.polyblep_arr(ffi.cast("const double *", t_b.ctypes.data), ffi.cast("const double *", dt_b.ctypes.data), ffi.cast("double *", out.ctypes.data), int(out.size))
    return out


def osc_saw_blep_c(ph: np.ndarray, dphi: np.ndarray) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    _impl.lib.osc_saw_blep_c(ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data), ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data), ffi.cast("double *", out.ctypes.data), int(B), int(F))
    return out


def osc_saw_blep_py(ph: np.ndarray, dphi: np.ndarray) -> np.ndarray:
    t = ph
    y = 2.0 * t - 1.0
    # reuse _polyblep_arr implementation
    pb = _polyblep_arr_py(t, dphi)
    return y - pb


def osc_square_blep_c(ph: np.ndarray, dphi: np.ndarray, pw: float = 0.5) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    _impl.lib.osc_square_blep_c(ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data), ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data), float(pw), ffi.cast("double *", out.ctypes.data), int(B), int(F))
    return out


def osc_square_blep_py(ph: np.ndarray, dphi: np.ndarray, pw: float = 0.5) -> np.ndarray:
    t = ph
    y = np.where(t < pw, 1.0, -1.0)
    y = y - _polyblep_arr_py(t, dphi)
    t2 = (t + (1.0 - pw)) % 1.0
    y = y + _polyblep_arr_py(t2, dphi)
    return y


def osc_triangle_blep_c(ph: np.ndarray, dphi: np.ndarray, tri_state: np.ndarray | None = None) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B, F = ph.shape
    out = np.empty((B, F), dtype=ph.dtype)
    if tri_state is not None:
        tri_buf = np.ascontiguousarray(tri_state.astype(np.float64))
        tri_ptr = ffi.cast("double *", tri_buf.ctypes.data)
    else:
        tri_buf = None
        tri_ptr = ffi.cast("double *", ffi.NULL)
    _impl.lib.osc_triangle_blep_c(
        ffi.cast("const double *", np.ascontiguousarray(ph).ctypes.data),
        ffi.cast("const double *", np.ascontiguousarray(dphi).ctypes.data),
        ffi.cast("double *", out.ctypes.data),
        int(B),
        int(F),
        tri_ptr,
    )
    if tri_state is not None:
        tri_state[:] = tri_buf[:B]
    return out


def _polyblep_arr_py(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    m = t < dt
    if np.any(m):
        x = t[m] / np.maximum(dt[m], 1e-20)
        out[m] = x + x - x * x - 1.0
    m = t > (1.0 - dt)
    if np.any(m):
        x = (t[m] - 1.0) / np.maximum(dt[m], 1e-20)
        out[m] = x * x + x + x + 1.0
    return out



def envelope_process_c(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    atk_frames: int,
    hold_frames: int,
    dec_frames: int,
    sus_frames: int,
    rel_frames: int,
    sustain_level: float,
    send_resets: bool,
    stage: np.ndarray,
    value: np.ndarray,
    timer: np.ndarray,
    vel_state: np.ndarray,
    activations: np.ndarray,
    release_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B = trigger.shape[0]
    F = trigger.shape[1]
    out_amp = np.zeros((B, F), dtype=trigger.dtype)
    out_reset = np.zeros((B, F), dtype=trigger.dtype)
    trig_b = np.ascontiguousarray(trigger)
    gate_b = np.ascontiguousarray(gate)
    drone_b = np.ascontiguousarray(drone)
    vel_b = np.ascontiguousarray(velocity)
    stage_b = np.ascontiguousarray(stage.astype(np.int32))
    value_b = np.ascontiguousarray(value.astype(np.float64))
    timer_b = np.ascontiguousarray(timer.astype(np.float64))
    vel_state_b = np.ascontiguousarray(vel_state.astype(np.float64))
    activ_b = np.ascontiguousarray(activations.astype(np.int64))
    rel_b = np.ascontiguousarray(release_start.astype(np.float64))
    amp_ptr = ffi.cast("double *", out_amp.ctypes.data)
    reset_ptr = ffi.cast("double *", out_reset.ctypes.data)
    trig_ptr = ffi.cast("const double *", trig_b.ctypes.data)
    gate_ptr = ffi.cast("const double *", gate_b.ctypes.data)
    drone_ptr = ffi.cast("const double *", drone_b.ctypes.data)
    vel_ptr = ffi.cast("const double *", vel_b.ctypes.data)
    stage_ptr = ffi.cast("int *", stage_b.ctypes.data)
    value_ptr = ffi.cast("double *", value_b.ctypes.data)
    timer_ptr = ffi.cast("double *", timer_b.ctypes.data)
    vel_state_ptr = ffi.cast("double *", vel_state_b.ctypes.data)
    activ_ptr = ffi.cast("int64_t *", activ_b.ctypes.data)
    rel_ptr = ffi.cast("double *", rel_b.ctypes.data)

    _impl.lib.envelope_process(
        trig_ptr,
        gate_ptr,
        drone_ptr,
        vel_ptr,
        int(B),
        int(F),
        int(atk_frames),
        int(hold_frames),
        int(dec_frames),
        int(sus_frames),
        int(rel_frames),
        float(sustain_level),
        int(1 if send_resets else 0),
        stage_ptr,
        value_ptr,
        timer_ptr,
        vel_state_ptr,
        activ_ptr,
        rel_ptr,
        amp_ptr,
        reset_ptr,
    )

    # copy back mutable states
    stage[:] = stage_b
    value[:] = value_b
    timer[:] = timer_b
    vel_state[:] = vel_state_b
    activations[:] = activ_b
    release_start[:] = rel_b
    return out_amp, out_reset


def envelope_process_py(
    trigger: np.ndarray,
    gate: np.ndarray,
    drone: np.ndarray,
    velocity: np.ndarray,
    atk_frames: int,
    hold_frames: int,
    dec_frames: int,
    sus_frames: int,
    rel_frames: int,
    sustain_level: float,
    send_resets: bool,
    stage: np.ndarray,
    value: np.ndarray,
    timer: np.ndarray,
    vel_state: np.ndarray,
    activations: np.ndarray,
    release_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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



