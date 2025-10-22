"""Optional C-backed kernels for tight loops.

This module attempts to build a small C kernel using cffi. If compilation
is not available in the environment (no compiler or cffi not installed),
the module exposes python fallbacks so callers can transparently fall back
to pure-Python/numpy implementations.
"""
from __future__ import annotations

import traceback
from typing import Optional

import numpy as np

AVAILABLE = False
_impl = None
UNAVAILABLE_REASON: str | None = None

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
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int *boundaries;
    int *trig_indices;
    int8_t *gate_bool;
    int8_t *drone_bool;
    size_t boundary_cap;
    size_t trig_cap;
    size_t bool_cap;
} envelope_scratch_t;

static envelope_scratch_t envelope_scratch = { NULL, NULL, NULL, NULL, 0, 0, 0 };

/*
 * Edge runner contract (mirrors `_EDGE_RUNNER_CDEF` in Python).
 *
 * The runtime passes node descriptors/inputs to `amp_run_node`, which may
 * allocate per-node state (returned via `state`) and a heap-owned audio buffer
 * (`out_buffer`).
 *
 * Return codes:
 *   0   -> success
 *  -1   -> allocation failure / invalid contract usage
 *  -3   -> unsupported node kind (caller should fall back to Python)
 */
typedef struct {
    uint32_t has_audio;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    const double *data;
} EdgeRunnerAudioView;

typedef struct {
    const char *name;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    const double *data;
} EdgeRunnerParamView;

typedef struct {
    uint32_t count;
    EdgeRunnerParamView *items;
} EdgeRunnerParamSet;

typedef struct {
    EdgeRunnerAudioView audio;
    EdgeRunnerParamSet params;
} EdgeRunnerNodeInputs;

typedef struct {
    const char *name;
    size_t name_len;
    const char *type_name;
    size_t type_len;
    const char *params_json;
    size_t params_len;
} EdgeRunnerNodeDescriptor;

typedef struct {
    char *name;
    uint32_t name_len;
    uint32_t offset;
    uint32_t span;
} EdgeRunnerCompiledParam;

typedef struct {
    char *name;
    uint32_t name_len;
    uint32_t function_id;
    uint32_t audio_offset;
    uint32_t audio_span;
    uint32_t param_count;
    EdgeRunnerCompiledParam *params;
} EdgeRunnerCompiledNode;

typedef struct {
    uint32_t version;
    uint32_t node_count;
    EdgeRunnerCompiledNode *nodes;
} EdgeRunnerCompiledPlan;

typedef struct {
    char *name;
    uint32_t name_len;
    double *values;
    uint32_t value_count;
    double timestamp;
} EdgeRunnerControlCurve;

typedef struct {
    uint32_t frames_hint;
    uint32_t curve_count;
    EdgeRunnerControlCurve *curves;
} EdgeRunnerControlHistory;

static void destroy_compiled_plan(EdgeRunnerCompiledPlan *plan) {
    if (plan == NULL) {
        return;
    }
    if (plan->nodes != NULL) {
        for (uint32_t i = 0; i < plan->node_count; ++i) {
            EdgeRunnerCompiledNode *node = &plan->nodes[i];
            if (node->params != NULL) {
                for (uint32_t j = 0; j < node->param_count; ++j) {
                    EdgeRunnerCompiledParam *param = &node->params[j];
                    if (param->name != NULL) {
                        free(param->name);
                        param->name = NULL;
                    }
                }
                free(node->params);
                node->params = NULL;
            }
            if (node->name != NULL) {
                free(node->name);
                node->name = NULL;
            }
        }
        free(plan->nodes);
        plan->nodes = NULL;
    }
    free(plan);
}

static int read_u32_le(const uint8_t **cursor, size_t *remaining, uint32_t *out_value) {
    if (cursor == NULL || remaining == NULL || out_value == NULL) {
        return 0;
    }
    if (*remaining < 4) {
        return 0;
    }
    const uint8_t *ptr = *cursor;
    *out_value = (uint32_t)ptr[0]
        | ((uint32_t)ptr[1] << 8)
        | ((uint32_t)ptr[2] << 16)
        | ((uint32_t)ptr[3] << 24);
    *cursor += 4;
    *remaining -= 4;
    return 1;
}

EdgeRunnerCompiledPlan *amp_load_compiled_plan(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    if (descriptor_blob == NULL || plan_blob == NULL) {
        return NULL;
    }
    if (descriptor_len < 4 || plan_len < 12) {
        return NULL;
    }

    const uint8_t *descriptor_cursor = descriptor_blob;
    size_t descriptor_remaining = descriptor_len;
    uint32_t descriptor_count = 0;
    if (!read_u32_le(&descriptor_cursor, &descriptor_remaining, &descriptor_count)) {
        return NULL;
    }

    const uint8_t *cursor = plan_blob;
    size_t remaining = plan_len;
    if (remaining < 4) {
        return NULL;
    }
    if (cursor[0] != 'A' || cursor[1] != 'M' || cursor[2] != 'P' || cursor[3] != 'L') {
        return NULL;
    }
    cursor += 4;
    remaining -= 4;

    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!read_u32_le(&cursor, &remaining, &version) || !read_u32_le(&cursor, &remaining, &node_count)) {
        return NULL;
    }
    if (descriptor_count != node_count) {
        return NULL;
    }

    EdgeRunnerCompiledPlan *plan = (EdgeRunnerCompiledPlan *)calloc(1, sizeof(EdgeRunnerCompiledPlan));
    if (plan == NULL) {
        return NULL;
    }
    plan->version = version;
    plan->node_count = node_count;

    if (node_count == 0) {
        if (remaining != 0) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        return plan;
    }

    plan->nodes = (EdgeRunnerCompiledNode *)calloc(node_count, sizeof(EdgeRunnerCompiledNode));
    if (plan->nodes == NULL) {
        destroy_compiled_plan(plan);
        return NULL;
    }

    for (uint32_t idx = 0; idx < node_count; ++idx) {
        EdgeRunnerCompiledNode *node = &plan->nodes[idx];
        uint32_t function_id = 0;
        uint32_t name_len = 0;
        uint32_t audio_offset = 0;
        uint32_t audio_span = 0;
        uint32_t param_count = 0;
        if (!read_u32_le(&cursor, &remaining, &function_id)
            || !read_u32_le(&cursor, &remaining, &name_len)
            || !read_u32_le(&cursor, &remaining, &audio_offset)
            || !read_u32_le(&cursor, &remaining, &audio_span)
            || !read_u32_le(&cursor, &remaining, &param_count)) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        if (remaining < name_len) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        node->name = (char *)malloc((size_t)name_len + 1);
        if (node->name == NULL) {
            destroy_compiled_plan(plan);
            return NULL;
        }
        memcpy(node->name, cursor, name_len);
        node->name[name_len] = '\0';
        node->name_len = name_len;
        cursor += name_len;
        remaining -= name_len;
        node->function_id = function_id;
        node->audio_offset = audio_offset;
        node->audio_span = audio_span;
        node->param_count = param_count;
        if (param_count > 0) {
            node->params = (EdgeRunnerCompiledParam *)calloc(param_count, sizeof(EdgeRunnerCompiledParam));
            if (node->params == NULL) {
                destroy_compiled_plan(plan);
                return NULL;
            }
        }
        for (uint32_t param_idx = 0; param_idx < param_count; ++param_idx) {
            EdgeRunnerCompiledParam *param = &node->params[param_idx];
            uint32_t param_name_len = 0;
            uint32_t param_offset = 0;
            uint32_t param_span = 0;
            if (!read_u32_le(&cursor, &remaining, &param_name_len)
                || !read_u32_le(&cursor, &remaining, &param_offset)
                || !read_u32_le(&cursor, &remaining, &param_span)) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            if (remaining < param_name_len) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            param->name = (char *)malloc((size_t)param_name_len + 1);
            if (param->name == NULL) {
                destroy_compiled_plan(plan);
                return NULL;
            }
            memcpy(param->name, cursor, param_name_len);
            param->name[param_name_len] = '\0';
            param->name_len = param_name_len;
            param->offset = param_offset;
            param->span = param_span;
            cursor += param_name_len;
            remaining -= param_name_len;
        }
    }

    if (remaining != 0) {
        destroy_compiled_plan(plan);
        return NULL;
    }

    return plan;
}

void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan) {
    destroy_compiled_plan(plan);
}

static void destroy_control_history(EdgeRunnerControlHistory *history) {
    if (history == NULL) {
        return;
    }
    if (history->curves != NULL) {
        for (uint32_t i = 0; i < history->curve_count; ++i) {
            EdgeRunnerControlCurve *curve = &history->curves[i];
            if (curve->name != NULL) {
                free(curve->name);
                curve->name = NULL;
            }
            if (curve->values != NULL) {
                free(curve->values);
                curve->values = NULL;
            }
            curve->value_count = 0;
        }
        free(history->curves);
        history->curves = NULL;
    }
    free(history);
}

static const EdgeRunnerControlCurve *find_history_curve(
    const EdgeRunnerControlHistory *history,
    const char *name,
    size_t name_len
) {
    if (history == NULL || name == NULL || name_len == 0) {
        return NULL;
    }
    for (uint32_t i = 0; i < history->curve_count; ++i) {
        const EdgeRunnerControlCurve *curve = &history->curves[i];
        if (curve->name_len == name_len && curve->name != NULL && strncmp(curve->name, name, name_len) == 0) {
            return curve;
        }
    }
    return NULL;
}

static void apply_history_curve(
    double *dest,
    int batches,
    int frames,
    const EdgeRunnerControlCurve *curve
) {
    if (dest == NULL || curve == NULL || curve->values == NULL || curve->value_count == 0) {
        return;
    }
    int count = (int)curve->value_count;
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            double value = 0.0;
            if (count >= frames) {
                if (f < count) {
                    value = curve->values[f];
                } else {
                    value = curve->values[count - 1];
                }
            } else if (count == 1) {
                value = curve->values[0];
            } else {
                if (f < count) {
                    value = curve->values[f];
                } else {
                    value = curve->values[count - 1];
                }
            }
            dest[((size_t)b * (size_t)frames) + (size_t)f] = value;
        }
    }
}

EdgeRunnerControlHistory *amp_load_control_history(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
) {
    if (blob == NULL || blob_len < 8) {
        return NULL;
    }
    const uint8_t *cursor = blob;
    size_t remaining = blob_len;
    uint32_t event_count = 0;
    uint32_t key_count = 0;
    if (!read_u32_le(&cursor, &remaining, &event_count) || !read_u32_le(&cursor, &remaining, &key_count)) {
        return NULL;
    }
    EdgeRunnerControlHistory *history = (EdgeRunnerControlHistory *)calloc(1, sizeof(EdgeRunnerControlHistory));
    if (history == NULL) {
        return NULL;
    }
    history->frames_hint = frames_hint > 0 ? (uint32_t)frames_hint : 0U;
    history->curve_count = key_count;
    if (key_count > 0) {
        history->curves = (EdgeRunnerControlCurve *)calloc(key_count, sizeof(EdgeRunnerControlCurve));
        if (history->curves == NULL) {
            destroy_control_history(history);
            return NULL;
        }
    }
    if (key_count == 0) {
        return history;
    }
    uint32_t *name_lengths = (uint32_t *)calloc(key_count, sizeof(uint32_t));
    if (name_lengths == NULL) {
        destroy_control_history(history);
        return NULL;
    }
    for (uint32_t i = 0; i < key_count; ++i) {
        if (!read_u32_le(&cursor, &remaining, &name_lengths[i])) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
    }
    for (uint32_t i = 0; i < key_count; ++i) {
        uint32_t name_len = name_lengths[i];
        if (remaining < name_len) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
        EdgeRunnerControlCurve *curve = &history->curves[i];
        curve->name = (char *)malloc((size_t)name_len + 1);
        if (curve->name == NULL) {
            free(name_lengths);
            destroy_control_history(history);
            return NULL;
        }
        memcpy(curve->name, cursor, name_len);
        curve->name[name_len] = '\0';
        curve->name_len = name_len;
        curve->value_count = 0;
        curve->values = NULL;
        curve->timestamp = -DBL_MAX;
        cursor += name_len;
        remaining -= name_len;
    }
    free(name_lengths);
    for (uint32_t event_idx = 0; event_idx < event_count; ++event_idx) {
        if (remaining < sizeof(double)) {
            destroy_control_history(history);
            return NULL;
        }
        double timestamp = 0.0;
        memcpy(&timestamp, cursor, sizeof(double));
        cursor += sizeof(double);
        remaining -= sizeof(double);
        for (uint32_t key_idx = 0; key_idx < key_count; ++key_idx) {
            uint32_t value_count = 0;
            if (!read_u32_le(&cursor, &remaining, &value_count)) {
                destroy_control_history(history);
                return NULL;
            }
            double *values_copy = NULL;
            if (value_count > 0) {
                size_t bytes = (size_t)value_count * sizeof(double);
                if (remaining < bytes) {
                    destroy_control_history(history);
                    return NULL;
                }
                values_copy = (double *)malloc(bytes);
                if (values_copy == NULL) {
                    destroy_control_history(history);
                    return NULL;
                }
                memcpy(values_copy, cursor, bytes);
                cursor += bytes;
                remaining -= bytes;
            }
            EdgeRunnerControlCurve *curve = &history->curves[key_idx];
            if (value_count > 0 && (curve->values == NULL || timestamp >= curve->timestamp)) {
                if (curve->values != NULL) {
                    free(curve->values);
                }
                curve->values = values_copy;
                curve->value_count = value_count;
                curve->timestamp = timestamp;
                values_copy = NULL;
            }
            if (values_copy != NULL) {
                free(values_copy);
            }
        }
    }
    return history;
}

void amp_release_control_history(EdgeRunnerControlHistory *history) {
    destroy_control_history(history);
}

static int envelope_reserve_scratch(int F) {
    size_t needed_boundaries = (size_t)(4 * F + 4);
    size_t needed_trig = (size_t)(F > 0 ? F : 1);
    size_t needed_bool = (size_t)(F > 0 ? F : 1);

    if (envelope_scratch.boundary_cap < needed_boundaries) {
        int *new_boundaries = (int *)realloc(envelope_scratch.boundaries, needed_boundaries * sizeof(int));
        if (new_boundaries == NULL) {
            return 0;
        }
        envelope_scratch.boundaries = new_boundaries;
        envelope_scratch.boundary_cap = needed_boundaries;
    }

    if (envelope_scratch.trig_cap < needed_trig) {
        int *new_trig = (int *)realloc(envelope_scratch.trig_indices, needed_trig * sizeof(int));
        if (new_trig == NULL) {
            return 0;
        }
        envelope_scratch.trig_indices = new_trig;
        envelope_scratch.trig_cap = needed_trig;
    }

    if (envelope_scratch.bool_cap < needed_bool) {
        int8_t *gate_ptr = envelope_scratch.gate_bool;
        int8_t *drone_ptr = envelope_scratch.drone_bool;
        int8_t *new_gate = (int8_t *)realloc(gate_ptr, needed_bool * sizeof(int8_t));
        int8_t *new_drone = (int8_t *)realloc(drone_ptr, needed_bool * sizeof(int8_t));
        if (new_gate == NULL || new_drone == NULL) {
            if (new_gate != NULL) {
                envelope_scratch.gate_bool = new_gate;
            }
            if (new_drone != NULL) {
                envelope_scratch.drone_bool = new_drone;
            }
            return 0;
        }
        envelope_scratch.gate_bool = new_gate;
        envelope_scratch.drone_bool = new_drone;
        envelope_scratch.bool_cap = needed_bool;
    }

    return 1;
}

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

static void envelope_start_attack(
    int index,
    const double* velocity,
    int send_resets,
    double* reset_line,
    int* stage,
    double* timer,
    double* value,
    double* vel_state,
    double* release_start,
    int64_t* activations
) {
    double vel = velocity[index];
    if (vel < 0.0) vel = 0.0;
    *stage = 1;
    *timer = 0.0;
    *value = 0.0;
    *vel_state = vel;
    *release_start = vel;
    *activations += 1;
    if (send_resets && reset_line != NULL) {
        reset_line[index] = 1.0;
    }
}

static void envelope_process_simple(
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

            if (trig) {
                envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
            } else if (st == 0 && (gate_on || drone_on)) {
                envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
            }

            if (st == 1) {
                if (atk_frames <= 0) {
                    val = vel;
                    if (hold_frames > 0) st = 2;
                    else if (dec_frames > 0) st = 3;
                    else st = 4;
                    tim = 0.0;
                } else {
                    val += vel / (double)(atk_frames > 0 ? atk_frames : 1);
                    if (val > vel) val = vel;
                    tim += 1.0;
                    if (tim >= atk_frames) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 2) {
                val = vel;
                if (hold_frames <= 0) {
                    if (dec_frames > 0) st = 3;
                    else st = 4;
                    tim = 0.0;
                } else {
                    tim += 1.0;
                    if (tim >= hold_frames) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 3) {
                double target = vel * sustain_level;
                if (dec_frames <= 0) {
                    val = target;
                    st = 4;
                    tim = 0.0;
                } else {
                    double delta = (vel - target) / (double)(dec_frames > 0 ? dec_frames : 1);
                    double candidate = val - delta;
                    if (candidate < target) candidate = target;
                    val = candidate;
                    tim += 1.0;
                    if (tim >= dec_frames) {
                        val = target;
                        st = 4;
                        tim = 0.0;
                    }
                }
            } else if (st == 4) {
                val = vel * sustain_level;
                if (sus_frames > 0) {
                    tim += 1.0;
                    if (tim >= sus_frames) {
                        st = 5;
                        rel_start = val;
                        tim = 0.0;
                    }
                } else if (!gate_on && !drone_on) {
                    st = 5;
                    rel_start = val;
                    tim = 0.0;
                }
            } else if (st == 5) {
                if (rel_frames <= 0) {
                    val = 0.0;
                    st = 0;
                    tim = 0.0;
                } else {
                    double step = rel_start / (double)(rel_frames > 0 ? rel_frames : 1);
                    double candidate = val - step;
                    if (candidate < 0.0) candidate = 0.0;
                    val = candidate;
                    tim += 1.0;
                    if (tim >= rel_frames) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                    }
                }
                if (gate_on || drone_on) {
                    envelope_start_attack(i, velocity + base, send_resets, reset_out != NULL ? reset_out + base : NULL, &st, &tim, &val, &vel, &rel_start, &acts);
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
    if (reset_out != NULL) {
        size_t total = (size_t)B * (size_t)F;
        memset(reset_out, 0, total * sizeof(double));
    }
    if (B <= 0 || F <= 0) {
        return;
    }

    if (!envelope_reserve_scratch(F)) {
        envelope_process_simple(
            trigger,
            gate,
            drone,
            velocity,
            B,
            F,
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
            amp_out,
            reset_out
        );
        return;
    }

    int* boundaries = envelope_scratch.boundaries;
    int* trig_indices = envelope_scratch.trig_indices;
    int8_t* gate_bool = envelope_scratch.gate_bool;
    int8_t* drone_bool = envelope_scratch.drone_bool;

    for (int b = 0; b < B; ++b) {
        int st = stage[b];
        double val = value[b];
        double tim = timer[b];
        double vel = vel_state[b];
        int64_t acts = activations[b];
        double rel_start = release_start[b];

        const double* trig_line = trigger + b * F;
        const double* gate_line = gate + b * F;
        const double* drone_line = drone + b * F;
        const double* vel_line = velocity + b * F;
        double* amp_line = amp_out + b * F;
        double* reset_line = reset_out != NULL ? reset_out + b * F : NULL;

        int trig_count = 0;
        for (int i = 0; i < F; ++i) {
            if (trig_line[i] > 0.5) {
                trig_indices[trig_count++] = i;
            }
            gate_bool[i] = gate_line[i] > 0.5 ? 1 : 0;
            drone_bool[i] = drone_line[i] > 0.5 ? 1 : 0;
        }

        int boundary_count = 0;
        boundaries[boundary_count++] = 0;
        boundaries[boundary_count++] = F;
        for (int i = 0; i < trig_count; ++i) {
            boundaries[boundary_count++] = trig_indices[i];
        }
        for (int i = 1; i < F; ++i) {
            if (gate_bool[i] != gate_bool[i - 1]) {
                boundaries[boundary_count++] = i;
            }
            if (drone_bool[i] != drone_bool[i - 1]) {
                boundaries[boundary_count++] = i;
            }
        }

        for (int i = 1; i < boundary_count; ++i) {
            int key = boundaries[i];
            int j = i - 1;
            while (j >= 0 && boundaries[j] > key) {
                boundaries[j + 1] = boundaries[j];
                --j;
            }
            boundaries[j + 1] = key;
        }

        int unique_count = 0;
        for (int i = 0; i < boundary_count; ++i) {
            int val_b = boundaries[i];
            if (val_b < 0) val_b = 0;
            if (val_b > F) val_b = F;
            if (unique_count == 0 || boundaries[unique_count - 1] != val_b) {
                boundaries[unique_count++] = val_b;
            }
        }
        if (unique_count < 2) {
            boundaries[0] = 0;
            boundaries[1] = F;
            unique_count = 2;
        }

        int trig_ptr = 0;
        for (int seg = 0; seg < unique_count - 1; ++seg) {
            int start = boundaries[seg];
            int stop = boundaries[seg + 1];
            if (start >= F) {
                break;
            }
            if (stop > F) {
                stop = F;
            }
            if (stop <= start) {
                continue;
            }

            while (trig_ptr < trig_count && trig_indices[trig_ptr] == start) {
                envelope_start_attack(
                    start,
                    vel_line,
                    send_resets,
                    reset_line,
                    &st,
                    &tim,
                    &val,
                    &vel,
                    &rel_start,
                    &acts
                );
                trig_ptr++;
            }

            int t = start;
            while (t < stop) {
                int gate_on = (gate_bool[t] != 0) || (drone_bool[t] != 0);

                int changed = 1;
                while (changed) {
                    changed = 0;
                    if (st == 1 && atk_frames <= 0) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 2 && hold_frames <= 0) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 3 && dec_frames <= 0) {
                        val = vel * sustain_level;
                        st = 4;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                    if (st == 5 && rel_frames <= 0) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                        changed = 1;
                        continue;
                    }
                }

                if (st == 0) {
                    if (gate_on) {
                        envelope_start_attack(
                            t,
                            vel_line,
                            send_resets,
                            reset_line,
                            &st,
                            &tim,
                            &val,
                            &vel,
                            &rel_start,
                            &acts
                        );
                        continue;
                    }
                    int seg_len = stop - t;
                    for (int k = 0; k < seg_len; ++k) {
                        amp_line[t + k] = 0.0;
                    }
                    val = 0.0;
                    tim = 0.0;
                    t = stop;
                    continue;
                }

                if (st == 1) {
                    if (atk_frames <= 0) {
                        continue;
                    }
                    int remaining = atk_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    double step = vel / (atk_frames > 0 ? (double)atk_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val + step * (double)(k + 1);
                        if (vel >= 0.0 && sample > vel) sample = vel;
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (atk_frames > 0 && tim >= atk_frames) {
                        val = vel;
                        if (hold_frames > 0) st = 2;
                        else if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 2) {
                    if (hold_frames <= 0) {
                        continue;
                    }
                    int remaining = hold_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    for (int k = 0; k < seg_len; ++k) {
                        amp_line[t + k] = vel;
                    }
                    val = vel;
                    tim += (double)seg_len;
                    if (tim >= hold_frames) {
                        if (dec_frames > 0) st = 3;
                        else st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 3) {
                    if (dec_frames <= 0) {
                        continue;
                    }
                    int remaining = dec_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        t = stop;
                        continue;
                    }
                    double target = vel * sustain_level;
                    double delta = (vel - target) / (dec_frames > 0 ? (double)dec_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val - delta * (double)(k + 1);
                        if (sample < target) sample = target;
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (tim >= dec_frames) {
                        val = target;
                        st = 4;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                if (st == 4) {
                    double sustain_val = vel * sustain_level;
                    if (sus_frames > 0) {
                        int remaining = sus_frames - (int)tim;
                        if (remaining <= 0) remaining = 1;
                        int seg_len = stop - t;
                        if (seg_len > remaining) seg_len = remaining;
                        if (seg_len <= 0) {
                            t = stop;
                            continue;
                        }
                        for (int k = 0; k < seg_len; ++k) {
                            amp_line[t + k] = sustain_val;
                        }
                        val = sustain_val;
                        tim += (double)seg_len;
                        if (tim >= sus_frames) {
                            st = 5;
                            rel_start = val;
                            tim = 0.0;
                        }
                        t += seg_len;
                        continue;
                    } else {
                        int seg_len = stop - t;
                        if (!gate_on && seg_len > 1) {
                            seg_len = 1;
                        }
                        if (seg_len <= 0) {
                            seg_len = 1;
                            if (t + seg_len > stop) seg_len = stop - t;
                        }
                        if (seg_len <= 0) {
                            break;
                        }
                        for (int k = 0; k < seg_len; ++k) {
                            amp_line[t + k] = sustain_val;
                        }
                        val = sustain_val;
                        if (!gate_on) {
                            st = 5;
                            rel_start = val;
                            tim = 0.0;
                        } else {
                            tim = 0.0;
                        }
                        t += seg_len;
                        continue;
                    }
                }

                if (st == 5) {
                    if (gate_on) {
                        amp_line[t] = 0.0;
                        envelope_start_attack(
                            t,
                            vel_line,
                            send_resets,
                            reset_line,
                            &st,
                            &tim,
                            &val,
                            &vel,
                            &rel_start,
                            &acts
                        );
                        t += 1;
                        continue;
                    }
                    if (rel_frames <= 0) {
                        continue;
                    }
                    int remaining = rel_frames - (int)tim;
                    if (remaining <= 0) remaining = 1;
                    int seg_len = stop - t;
                    if (seg_len > remaining) seg_len = remaining;
                    if (seg_len <= 0) {
                        seg_len = remaining;
                        if (seg_len <= 0) seg_len = 1;
                        if (t + seg_len > stop) seg_len = stop - t;
                        if (seg_len <= 0) {
                            break;
                        }
                    }
                    double step = rel_start / (rel_frames > 0 ? (double)rel_frames : 1.0);
                    for (int k = 0; k < seg_len; ++k) {
                        double sample = val - step * (double)(k + 1);
                        if (sample < 0.0) sample = 0.0;
                        amp_line[t + k] = sample;
                    }
                    val = amp_line[t + seg_len - 1];
                    tim += (double)seg_len;
                    if (tim >= rel_frames) {
                        val = 0.0;
                        st = 0;
                        tim = 0.0;
                    }
                    t += seg_len;
                    continue;
                }

                // Unknown stage -> silence and exit segment.
                for (int k = t; k < stop; ++k) {
                    amp_line[k] = 0.0;
                }
                val = 0.0;
                tim = 0.0;
                st = 0;
                t = stop;
            }
        }

        if (val < 0.0) val = 0.0;
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

typedef enum {
    NODE_KIND_UNKNOWN = 0,
    NODE_KIND_CONSTANT,
    NODE_KIND_GAIN,
    NODE_KIND_MIX,
    NODE_KIND_SAFETY,
    NODE_KIND_SINE_OSC,
    NODE_KIND_CONTROLLER,
    NODE_KIND_LFO,
    NODE_KIND_ENVELOPE,
    NODE_KIND_PITCH,
    NODE_KIND_OSC,
    NODE_KIND_SUBHARM,
} node_kind_t;

typedef struct {
    node_kind_t kind;
    union {
        struct {
            double value;
            int channels;
        } constant;
        struct {
            int out_channels;
        } mix;
        struct {
            double *state;
            int batches;
            int channels;
            double alpha;
        } safety;
        struct {
            double *phase;
            int batches;
            int channels;
            double base_phase;
        } sine;
        struct {
            double *phase;
            double *phase_buffer;
            double *wave_buffer;
            double *dphi_buffer;
            double *tri_state;
            int batches;
            int channels;
            double base_phase;
            int stereo;
        } osc;
        struct {
            double *slew_state;
            int batches;
            double phase;
        } lfo;
        struct {
            int *stage;
            double *value;
            double *timer;
            double *velocity;
            int64_t *activations;
            double *release_start;
            int batches;
        } envelope;
        struct {
            double *last_freq;
            int batches;
        } pitch;
        struct {
            double *hp_y;
            double *lp_y;
            double *prev;
            int8_t *sign;
            int8_t *ff2;
            int8_t *ff4;
            int32_t *ff4_count;
            double *sub2_lp;
            double *sub4_lp;
            double *env;
            double *hp_out_y;
            double *hp_out_x;
            int batches;
            int channels;
            int use_div4;
        } subharm;
    } u;
} node_state_t;

static void release_node_state(node_state_t *state) {
    if (state == NULL) {
        return;
    }
    if (state->kind == NODE_KIND_SAFETY && state->u.safety.state != NULL) {
        free(state->u.safety.state);
        state->u.safety.state = NULL;
        state->u.safety.batches = 0;
        state->u.safety.channels = 0;
        state->u.safety.alpha = 0.0;
    }
    if (state->kind == NODE_KIND_SINE_OSC && state->u.sine.phase != NULL) {
        free(state->u.sine.phase);
        state->u.sine.phase = NULL;
        state->u.sine.batches = 0;
        state->u.sine.channels = 0;
        state->u.sine.base_phase = 0.0;
    }
    if (state->kind == NODE_KIND_OSC) {
        free(state->u.osc.phase);
        free(state->u.osc.phase_buffer);
        free(state->u.osc.wave_buffer);
        free(state->u.osc.dphi_buffer);
        free(state->u.osc.tri_state);
        state->u.osc.phase = NULL;
        state->u.osc.phase_buffer = NULL;
        state->u.osc.wave_buffer = NULL;
        state->u.osc.dphi_buffer = NULL;
        state->u.osc.tri_state = NULL;
        state->u.osc.batches = 0;
        state->u.osc.channels = 0;
        state->u.osc.stereo = 0;
    }
    if (state->kind == NODE_KIND_LFO) {
        free(state->u.lfo.slew_state);
        state->u.lfo.slew_state = NULL;
        state->u.lfo.batches = 0;
        state->u.lfo.phase = 0.0;
    }
    if (state->kind == NODE_KIND_ENVELOPE) {
        free(state->u.envelope.stage);
        free(state->u.envelope.value);
        free(state->u.envelope.timer);
        free(state->u.envelope.velocity);
        free(state->u.envelope.activations);
        free(state->u.envelope.release_start);
        state->u.envelope.stage = NULL;
        state->u.envelope.value = NULL;
        state->u.envelope.timer = NULL;
        state->u.envelope.velocity = NULL;
        state->u.envelope.activations = NULL;
        state->u.envelope.release_start = NULL;
        state->u.envelope.batches = 0;
    }
    if (state->kind == NODE_KIND_PITCH) {
        free(state->u.pitch.last_freq);
        state->u.pitch.last_freq = NULL;
        state->u.pitch.batches = 0;
    }
    if (state->kind == NODE_KIND_SUBHARM) {
        free(state->u.subharm.hp_y);
        free(state->u.subharm.lp_y);
        free(state->u.subharm.prev);
        free(state->u.subharm.sign);
        free(state->u.subharm.ff2);
        free(state->u.subharm.ff4);
        free(state->u.subharm.ff4_count);
        free(state->u.subharm.sub2_lp);
        free(state->u.subharm.sub4_lp);
        free(state->u.subharm.env);
        free(state->u.subharm.hp_out_y);
        free(state->u.subharm.hp_out_x);
        state->u.subharm.hp_y = NULL;
        state->u.subharm.lp_y = NULL;
        state->u.subharm.prev = NULL;
        state->u.subharm.sign = NULL;
        state->u.subharm.ff2 = NULL;
        state->u.subharm.ff4 = NULL;
        state->u.subharm.ff4_count = NULL;
        state->u.subharm.sub2_lp = NULL;
        state->u.subharm.sub4_lp = NULL;
        state->u.subharm.env = NULL;
        state->u.subharm.hp_out_y = NULL;
        state->u.subharm.hp_out_x = NULL;
        state->u.subharm.batches = 0;
        state->u.subharm.channels = 0;
        state->u.subharm.use_div4 = 0;
    }
    free(state);
}

static node_kind_t determine_node_kind(const EdgeRunnerNodeDescriptor *descriptor) {
    if (descriptor == NULL || descriptor->type_name == NULL) {
        return NODE_KIND_UNKNOWN;
    }
    if (strcmp(descriptor->type_name, "ConstantNode") == 0) {
        return NODE_KIND_CONSTANT;
    }
    if (strcmp(descriptor->type_name, "GainNode") == 0) {
        return NODE_KIND_GAIN;
    }
    if (strcmp(descriptor->type_name, "MixNode") == 0) {
        return NODE_KIND_MIX;
    }
    if (strcmp(descriptor->type_name, "SafetyNode") == 0) {
        return NODE_KIND_SAFETY;
    }
    if (strcmp(descriptor->type_name, "SineOscillatorNode") == 0) {
        return NODE_KIND_SINE_OSC;
    }
    if (strcmp(descriptor->type_name, "ControllerNode") == 0) {
        return NODE_KIND_CONTROLLER;
    }
    if (strcmp(descriptor->type_name, "LFONode") == 0) {
        return NODE_KIND_LFO;
    }
    if (strcmp(descriptor->type_name, "EnvelopeModulatorNode") == 0) {
        return NODE_KIND_ENVELOPE;
    }
    if (strcmp(descriptor->type_name, "PitchQuantizerNode") == 0) {
        return NODE_KIND_PITCH;
    }
    if (strcmp(descriptor->type_name, "OscNode") == 0) {
        return NODE_KIND_OSC;
    }
    if (strcmp(descriptor->type_name, "SubharmonicLowLifterNode") == 0) {
        return NODE_KIND_SUBHARM;
    }
    return NODE_KIND_UNKNOWN;
}

static double json_get_double(const char *json, size_t json_len, const char *key, double default_value) {
    (void)json_len;
    if (json == NULL || key == NULL) {
        return default_value;
    }
    size_t key_len = strlen(key);
    if (key_len == 0) {
        return default_value;
    }
    char pattern[128];
    if (key_len + 3 >= sizeof(pattern)) {
        return default_value;
    }
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = json;
    size_t pattern_len = strlen(pattern);
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        cursor += pattern_len;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != ':') {
            continue;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        char *endptr = NULL;
        double value = strtod(cursor, &endptr);
        if (endptr == cursor) {
            cursor = endptr;
            continue;
        }
        return value;
    }
    return default_value;
}

static int json_get_int(const char *json, size_t json_len, const char *key, int default_value) {
    double value = json_get_double(json, json_len, key, (double)default_value);
    if (value >= 0.0) {
        return (int)(value + 0.5);
    }
    return (int)(value - 0.5);
}

static int json_get_bool(const char *json, size_t json_len, const char *key, int default_value) {
    double value = json_get_double(json, json_len, key, default_value ? 1.0 : 0.0);
    return value >= 0.5 ? 1 : 0;
}

static int json_copy_string(const char *json, size_t json_len, const char *key, char *out, size_t out_len) {
    (void)json_len;
    if (out == NULL || out_len == 0) {
        return 0;
    }
    out[0] = '\0';
    if (json == NULL || key == NULL) {
        return 0;
    }
    size_t key_len = strlen(key);
    if (key_len == 0) {
        return 0;
    }
    char pattern[128];
    if (key_len + 3 >= sizeof(pattern)) {
        return 0;
    }
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = json;
    size_t pattern_len = strlen(pattern);
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        cursor += pattern_len;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != ':') {
            continue;
        }
        cursor++;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') {
            cursor++;
        }
        if (*cursor != '"') {
            continue;
        }
        cursor++;
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != '"') {
            cursor++;
        }
        size_t length = (size_t)(cursor - start);
        if (length >= out_len) {
            length = out_len - 1;
        }
        memcpy(out, start, length);
        out[length] = '\0';
        return 1;
    }
    return 0;
}

static const EdgeRunnerParamView *find_param(const EdgeRunnerNodeInputs *inputs, const char *name) {
    if (inputs == NULL || name == NULL) {
        return NULL;
    }
    uint32_t count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < count; ++i) {
        const EdgeRunnerParamView *view = &items[i];
        if (view->name != NULL && strcmp(view->name, name) == 0) {
            return view;
        }
    }
    return NULL;
}

typedef struct {
    char output[64];
    char source[64];
} controller_source_t;

static int parse_csv_tokens(const char *csv, char tokens[][64], int max_tokens) {
    if (csv == NULL || tokens == NULL || max_tokens <= 0) {
        return 0;
    }
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_tokens) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            cursor++;
        }
        size_t len = (size_t)(cursor - start);
        if (len >= 63) {
            len = 63;
        }
        memcpy(tokens[count], start, len);
        tokens[count][len] = '\0';
        count++;
    }
    return count;
}

static int parse_controller_sources(const char *csv, controller_source_t *items, int max_items) {
    if (csv == NULL || items == NULL || max_items <= 0) {
        return 0;
    }
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_items) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        const char *eq = strchr(cursor, '=');
        if (eq == NULL) {
            break;
        }
        size_t key_len = (size_t)(eq - cursor);
        if (key_len >= sizeof(items[count].output)) {
            key_len = sizeof(items[count].output) - 1;
        }
        memcpy(items[count].output, cursor, key_len);
        items[count].output[key_len] = '\0';
        cursor = eq + 1;
        const char *end = strchr(cursor, ',');
        if (end == NULL) {
            end = cursor + strlen(cursor);
        }
        size_t value_len = (size_t)(end - cursor);
        if (value_len >= sizeof(items[count].source)) {
            value_len = sizeof(items[count].source) - 1;
        }
        memcpy(items[count].source, cursor, value_len);
        items[count].source[value_len] = '\0';
        cursor = end;
        count++;
    }
    return count;
}

static int parse_csv_doubles(const char *csv, double *values, int max_values) {
    if (csv == NULL || values == NULL || max_values <= 0) {
        return 0;
    }
    int count = 0;
    const char *cursor = csv;
    while (*cursor != '\0' && count < max_values) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }
        char *endptr = NULL;
        double value = strtod(cursor, &endptr);
        if (endptr == cursor) {
            break;
        }
        values[count++] = value;
        cursor = endptr;
    }
    return count;
}

static const double *ensure_param_plane(
    const EdgeRunnerParamView *view,
    int batches,
    int frames,
    double default_value,
    double **owned_out
) {
    if (owned_out != NULL) {
        *owned_out = NULL;
    }
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)frames;
    if (view == NULL || view->data == NULL) {
        if (owned_out == NULL) {
            return NULL;
        }
        double *buf = (double *)malloc(total * sizeof(double));
        if (buf == NULL) {
            return NULL;
        }
        for (size_t i = 0; i < total; ++i) {
            buf[i] = default_value;
        }
        *owned_out = buf;
        return buf;
    }
    int vb = view->batches > 0 ? (int)view->batches : batches;
    int vc = view->channels > 0 ? (int)view->channels : 1;
    int vf = view->frames > 0 ? (int)view->frames : frames;
    if (vb == batches && vc == 1 && vf == frames) {
        return view->data;
    }
    if (owned_out == NULL) {
        return NULL;
    }
    double *buf = (double *)malloc(total * sizeof(double));
    if (buf == NULL) {
        return NULL;
    }
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            size_t idx = (size_t)b * (size_t)frames + (size_t)f;
            double value = default_value;
            if (b < vb && f < vf) {
                size_t src_idx = ((size_t)b * (size_t)vc) * (size_t)vf + (size_t)f;
                if (vc > 0) {
                    src_idx = ((size_t)b * (size_t)vc + 0) * (size_t)vf + (size_t)f;
                }
                size_t span = (size_t)vb * (size_t)vc * (size_t)vf;
                if (src_idx < span) {
                    value = view->data[src_idx];
                }
            }
            buf[idx] = value;
        }
    }
    *owned_out = buf;
    return buf;
}

static double read_scalar_param(const EdgeRunnerParamView *view, double default_value) {
    if (view == NULL || view->data == NULL) {
        return default_value;
    }
    size_t total = (size_t)(view->batches ? view->batches : 1)
        * (size_t)(view->channels ? view->channels : 1)
        * (size_t)(view->frames ? view->frames : 1);
    if (total == 0) {
        return default_value;
    }
    return view->data[total - 1];
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static int build_sorted_grid(const double *values, int count, double *sorted, double *ext) {
    if (values == NULL || sorted == NULL || ext == NULL || count <= 0) {
        return 0;
    }
    int n = count;
    if (n < 2) {
        n = 12;
        for (int i = 0; i < n; ++i) {
            sorted[i] = (double)i * 100.0;
        }
    } else {
        memcpy(sorted, values, (size_t)n * sizeof(double));
        qsort(sorted, (size_t)n, sizeof(double), compare_double);
    }
    for (int i = 0; i < n; ++i) {
        ext[i] = sorted[i];
    }
    ext[n] = sorted[0] + 1200.0;
    return n;
}

static double grid_warp_forward_value(double cents, const double *grid, const double *grid_ext, int N) {
    double octs = floor(cents / 1200.0);
    double c_mod = fmod(cents, 1200.0);
    if (c_mod < 0.0) {
        c_mod += 1200.0;
    }
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        if (c_mod >= grid_ext[i] && c_mod < grid_ext[i + 1]) {
            idx = i;
            break;
        }
        if (i == N - 1) {
            idx = N - 1;
        }
    }
    double lower = grid_ext[idx];
    double upper = grid_ext[idx + 1];
    double denom = upper - lower;
    if (fabs(denom) < 1e-9) {
        denom = 1e-9;
    }
    double t = (c_mod - lower) / denom;
    double u_mod = (double)idx + t;
    return octs * (double)N + u_mod;
}

static double grid_warp_inverse_value(double u, const double *grid, const double *grid_ext, int N) {
    double octs = floor(u / (double)N);
    double u_mod = u - octs * (double)N;
    int idx = (int)floor(u_mod);
    if (idx < 0) {
        idx = 0;
    }
    if (idx >= N) {
        idx = N - 1;
    }
    double frac = u_mod - (double)idx;
    double lower = grid_ext[idx];
    double upper = grid_ext[idx + 1];
    double cents = lower + frac * (upper - lower);
    return octs * 1200.0 + cents;
}

static int run_constant_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    double value = json_get_double(descriptor->params_json, descriptor->params_len, "value", 0.0);
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    for (size_t i = 0; i < total; ++i) {
        buffer[i] = value;
    }
    if (state != NULL) {
        state->u.constant.value = value;
        state->u.constant.channels = channels;
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static double render_lfo_wave(const char *wave, double phase) {
    if (wave != NULL) {
        if (strcmp(wave, "square") == 0) {
            return phase < 0.5 ? 1.0 : -1.0;
        }
        if (strcmp(wave, "saw") == 0) {
            double t = phase - floor(phase);
            return 2.0 * t - 1.0;
        }
        if (strcmp(wave, "triangle") == 0) {
            double t = phase - floor(phase);
            return 2.0 * fabs(2.0 * t - 1.0) - 1.0;
        }
    }
    return sin(phase * 2.0 * M_PI);
}

static int run_controller_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    const EdgeRunnerControlHistory *history
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    char outputs_csv[256];
    char sources_csv[512];
    char output_names[32][64];
    controller_source_t mappings[32];
    int output_count = 0;
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_outputs__", outputs_csv, sizeof(outputs_csv))) {
        output_count = parse_csv_tokens(outputs_csv, output_names, 32);
    }
    if (output_count <= 0 && inputs != NULL) {
        uint32_t count = inputs->params.count;
        EdgeRunnerParamView *items = inputs->params.items;
        for (uint32_t i = 0; i < count && i < 32U; ++i) {
            if (items[i].name != NULL) {
                strncpy(output_names[output_count], items[i].name, sizeof(output_names[output_count]) - 1);
                output_names[output_count][sizeof(output_names[output_count]) - 1] = '\0';
                output_count++;
            }
        }
    }
    if (output_count <= 0) {
        return -1;
    }
    int mapping_count = 0;
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "__controller_sources__", sources_csv, sizeof(sources_csv))) {
        mapping_count = parse_controller_sources(sources_csv, mappings, 32);
    }
    int resolved_channels = output_count;
    if (inputs != NULL && inputs->params.count > 0) {
        const EdgeRunnerParamView *view = &inputs->params.items[0];
        if (batches <= 0 && view->batches > 0) {
            batches = (int)view->batches;
        }
        if (frames <= 0 && view->frames > 0) {
            frames = (int)view->frames;
        }
    }
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)resolved_channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    for (int c = 0; c < resolved_channels; ++c) {
        const char *source_name = output_names[c];
        for (int m = 0; m < mapping_count; ++m) {
            if (strcmp(mappings[m].output, output_names[c]) == 0) {
                source_name = mappings[m].source;
                break;
            }
        }
        const EdgeRunnerParamView *view = find_param(inputs, source_name);
        int view_missing = (view == NULL || view->data == NULL);
        double *owned = NULL;
        const double *data = ensure_param_plane(view, batches, frames, 0.0, &owned);
        if (data == NULL) {
            free(buffer);
            return -1;
        }
        if (view_missing && owned != NULL && history != NULL) {
            const EdgeRunnerControlCurve *curve = find_history_curve(history, source_name, strlen(source_name));
            if (curve != NULL) {
                apply_history_curve(owned, batches, frames, curve);
            }
            data = owned;
        }
        for (int b = 0; b < batches; ++b) {
            for (int f = 0; f < frames; ++f) {
                size_t src_idx = (size_t)b * (size_t)frames + (size_t)f;
                size_t dst_idx = ((size_t)b * (size_t)resolved_channels + (size_t)c) * (size_t)frames + (size_t)f;
                buffer[dst_idx] = data[src_idx];
            }
        }
        if (owned != NULL) {
            free(owned);
        }
    }
    *out_buffer = buffer;
    *out_channels = resolved_channels;
    return 0;
}

static int run_lfo_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    char wave_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "wave", wave_buf, sizeof(wave_buf))) {
        strcpy(wave_buf, "sine");
    }
    double rate_hz = json_get_double(descriptor->params_json, descriptor->params_len, "rate_hz", 1.0);
    double depth = json_get_double(descriptor->params_json, descriptor->params_len, "depth", 0.5);
    double slew_ms = json_get_double(descriptor->params_json, descriptor->params_len, "slew_ms", 0.0);
    int use_input = json_get_bool(descriptor->params_json, descriptor->params_len, "use_input", 0);
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    int audio_channels = 0;
    const double *audio_data = NULL;
    if (use_input && inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        B = inputs->audio.batches > 0 ? (int)inputs->audio.batches : B;
        F = inputs->audio.frames > 0 ? (int)inputs->audio.frames : F;
        audio_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
        audio_data = inputs->audio.data;
    }
    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (use_input && audio_data != NULL) {
        for (int b = 0; b < B; ++b) {
            double max_abs = 0.0;
            for (int c = 0; c < audio_channels; ++c) {
                for (int f = 0; f < F; ++f) {
                    size_t idx = ((size_t)b * (size_t)audio_channels + (size_t)c) * (size_t)F + (size_t)f;
                    double val = fabs(audio_data[idx]);
                    if (val > max_abs) {
                        max_abs = val;
                    }
                }
            }
            if (max_abs < 1e-12) {
                max_abs = 1.0;
            }
            for (int f = 0; f < F; ++f) {
                size_t src_idx = ((size_t)b * (size_t)audio_channels) * (size_t)F + (size_t)f;
                double sample = audio_data[src_idx];
                buffer[(size_t)b * (size_t)F + (size_t)f] = (sample / max_abs) * depth;
            }
        }
    } else {
        if (sample_rate <= 0.0) {
            sample_rate = 48000.0;
        }
        double step = rate_hz / sample_rate;
        double phase = 0.0;
        if (state != NULL) {
            phase = state->u.lfo.phase;
        }
        for (int b = 0; b < B; ++b) {
            double local_phase = phase;
            for (int f = 0; f < F; ++f) {
                double value = render_lfo_wave(wave_buf, local_phase) * depth;
                buffer[(size_t)b * (size_t)F + (size_t)f] = value;
                local_phase += step;
                local_phase -= floor(local_phase);
            }
            if (state != NULL) {
                phase = local_phase;
            }
        }
        if (state != NULL) {
            state->u.lfo.phase = phase;
        }
    }
    if (slew_ms > 0.0 && state != NULL) {
        if (sample_rate <= 0.0) {
            sample_rate = 48000.0;
        }
        double alpha = 1.0 - exp(-1.0 / (sample_rate * (slew_ms / 1000.0)));
        if (alpha < 1.0 - 1e-15) {
            double r = 1.0 - alpha;
            if (state->u.lfo.slew_state == NULL || state->u.lfo.batches != B) {
                free(state->u.lfo.slew_state);
                state->u.lfo.slew_state = (double *)calloc((size_t)B, sizeof(double));
                state->u.lfo.batches = B;
            }
            if (state->u.lfo.slew_state != NULL) {
                lfo_slew(buffer, buffer, B, F, r, alpha, state->u.lfo.slew_state);
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

static int run_envelope_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *trigger_view = find_param(inputs, "trigger");
    const EdgeRunnerParamView *gate_view = find_param(inputs, "gate");
    const EdgeRunnerParamView *drone_view = find_param(inputs, "drone");
    const EdgeRunnerParamView *velocity_view = find_param(inputs, "velocity");
    const EdgeRunnerParamView *send_reset_view = find_param(inputs, "send_reset");
    if (trigger_view != NULL && trigger_view->batches > 0) {
        B = (int)trigger_view->batches;
    }
    if (trigger_view != NULL && trigger_view->frames > 0) {
        F = (int)trigger_view->frames;
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }
    double attack_ms = json_get_double(descriptor->params_json, descriptor->params_len, "attack_ms", 12.0);
    double hold_ms = json_get_double(descriptor->params_json, descriptor->params_len, "hold_ms", 8.0);
    double decay_ms = json_get_double(descriptor->params_json, descriptor->params_len, "decay_ms", 90.0);
    double sustain_level = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_level", 0.7);
    double sustain_ms = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_ms", 0.0);
    double release_ms = json_get_double(descriptor->params_json, descriptor->params_len, "release_ms", 220.0);
    int send_resets_default = json_get_bool(descriptor->params_json, descriptor->params_len, "send_resets", 1);
    int atk_frames = (int)lrint((attack_ms / 1000.0) * sample_rate);
    int hold_frames = (int)lrint((hold_ms / 1000.0) * sample_rate);
    int dec_frames = (int)lrint((decay_ms / 1000.0) * sample_rate);
    int sus_frames = (int)lrint((sustain_ms / 1000.0) * sample_rate);
    int rel_frames = (int)lrint((release_ms / 1000.0) * sample_rate);
    if (atk_frames < 0) atk_frames = 0;
    if (hold_frames < 0) hold_frames = 0;
    if (dec_frames < 0) dec_frames = 0;
    if (sus_frames < 0) sus_frames = 0;
    if (rel_frames < 0) rel_frames = 0;
    double *owned_trigger = NULL;
    double *owned_gate = NULL;
    double *owned_drone = NULL;
    double *owned_velocity = NULL;
    const double *trigger = ensure_param_plane(trigger_view, B, F, 0.0, &owned_trigger);
    const double *gate = ensure_param_plane(gate_view, B, F, 0.0, &owned_gate);
    const double *drone = ensure_param_plane(drone_view, B, F, 0.0, &owned_drone);
    const double *velocity = ensure_param_plane(velocity_view, B, F, 1.0, &owned_velocity);
    if (trigger == NULL || gate == NULL || drone == NULL || velocity == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double send_reset_value = read_scalar_param(send_reset_view, (double)send_resets_default);
    int send_reset_flag = send_reset_value >= 0.5 ? 1 : 0;
    size_t total = (size_t)B * (size_t)F * 2;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double *amp_plane = buffer;
    double *reset_plane = buffer + (size_t)B * (size_t)F;
    if (state == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.batches != B) {
        free(state->u.envelope.stage);
        free(state->u.envelope.value);
        free(state->u.envelope.timer);
        free(state->u.envelope.velocity);
        free(state->u.envelope.activations);
        free(state->u.envelope.release_start);
        state->u.envelope.stage = (int *)calloc((size_t)B, sizeof(int));
        state->u.envelope.value = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.timer = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.velocity = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.activations = (int64_t *)calloc((size_t)B, sizeof(int64_t));
        state->u.envelope.release_start = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.batches = B;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.value == NULL || state->u.envelope.timer == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    envelope_process(
        trigger,
        gate,
        drone,
        velocity,
        B,
        F,
        atk_frames,
        hold_frames,
        dec_frames,
        sus_frames,
        rel_frames,
        sustain_level,
        send_reset_flag,
        state->u.envelope.stage,
        state->u.envelope.value,
        state->u.envelope.timer,
        state->u.envelope.velocity,
        state->u.envelope.activations,
        state->u.envelope.release_start,
        amp_plane,
        reset_plane
    );
    free(owned_trigger);
    free(owned_gate);
    free(owned_drone);
    free(owned_velocity);
    *out_buffer = buffer;
    *out_channels = 2;
    return 0;
}

static int run_pitch_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    char grid_csv[1024];
    double grid_values[256];
    double grid_sorted_vals[256];
    double grid_ext[257];
    int grid_count = 0;
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "grid_cents", grid_csv, sizeof(grid_csv))) {
        grid_count = parse_csv_doubles(grid_csv, grid_values, 256);
    }
    if (grid_count <= 0) {
        for (int i = 0; i < 12; ++i) {
            grid_values[i] = (double)i * 100.0;
        }
        grid_count = 12;
    }
    int grid_size = build_sorted_grid(grid_values, grid_count, grid_sorted_vals, grid_ext);
    int is_free_mode = json_get_bool(descriptor->params_json, descriptor->params_len, "is_free_mode", 0);
    char variant_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "free_variant", variant_buf, sizeof(variant_buf))) {
        strcpy(variant_buf, "continuous");
    }
    double span_default = json_get_double(descriptor->params_json, descriptor->params_len, "span_default", 2.0);
    int slew_enabled = json_get_bool(descriptor->params_json, descriptor->params_len, "slew", 1);
    const EdgeRunnerParamView *input_view = find_param(inputs, "input");
    const EdgeRunnerParamView *root_view = find_param(inputs, "root_midi");
    const EdgeRunnerParamView *span_view = find_param(inputs, "span_oct");
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    if (input_view != NULL && input_view->batches > 0) {
        B = (int)input_view->batches;
    }
    if (input_view != NULL && input_view->frames > 0) {
        F = (int)input_view->frames;
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;
    double *owned_input = NULL;
    double *owned_root = NULL;
    double *owned_span = NULL;
    const double *ctrl = ensure_param_plane(input_view, B, F, 0.0, &owned_input);
    const double *root = ensure_param_plane(root_view, B, F, 60.0, &owned_root);
    const double *span = ensure_param_plane(span_view, B, F, span_default, &owned_span);
    if (ctrl == NULL || root == NULL || span == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    size_t total = (size_t)B * (size_t)F;
    double *freq_target = (double *)malloc(total * sizeof(double));
    if (freq_target == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            size_t idx = (size_t)b * (size_t)F + (size_t)f;
            double ctrl_scaled = ctrl[idx] * span[idx];
            double root_midi = root[idx];
            double root_freq = 440.0 * pow(2.0, (root_midi - 69.0) / 12.0);
            double cents = 0.0;
            if (is_free_mode) {
                if (strcmp(variant_buf, "weighted") == 0) {
                    double u = ctrl_scaled * (double)grid_size;
                    cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                } else if (strcmp(variant_buf, "stepped") == 0) {
                    double u = round(ctrl_scaled * (double)grid_size);
                    cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                } else {
                    cents = ctrl_scaled * 1200.0;
                }
            } else {
                double cents_unq = ctrl_scaled * 1200.0;
                double u = grid_warp_forward_value(cents_unq, grid_sorted_vals, grid_ext, grid_size);
                double u_round = round(u);
                cents = grid_warp_inverse_value(u_round, grid_sorted_vals, grid_ext, grid_size);
            }
            freq_target[idx] = root_freq * pow(2.0, cents / 1200.0);
        }
    }
    free(owned_input);
    free(owned_root);
    free(owned_span);
    double *output = (double *)malloc(total * sizeof(double));
    if (output == NULL) {
        free(freq_target);
        return -1;
    }
    if (slew_enabled) {
        if (state->u.pitch.last_freq == NULL || state->u.pitch.batches != B) {
            free(state->u.pitch.last_freq);
            state->u.pitch.last_freq = (double *)calloc((size_t)B, sizeof(double));
            state->u.pitch.batches = B;
        }
        if (state->u.pitch.last_freq == NULL) {
            free(freq_target);
            free(output);
            return -1;
        }
        for (int b = 0; b < B; ++b) {
            double y0 = state->u.pitch.last_freq[b];
            double y1 = freq_target[(size_t)b * (size_t)F + (size_t)(F - 1)];
            for (int f = 0; f < F; ++f) {
                double t = (double)f / (double)F;
                double ramp = 3.0 * t * t - 2.0 * t * t * t;
                output[(size_t)b * (size_t)F + (size_t)f] = y0 + (y1 - y0) * ramp;
            }
            state->u.pitch.last_freq[b] = y1;
        }
    } else {
        memcpy(output, freq_target, total * sizeof(double));
    }
    free(freq_target);
    *out_buffer = output;
    *out_channels = 1;
    return 0;
}

static double alpha_lp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    return 1.0 - exp(-2.0 * M_PI * fc / sr);
}

static double alpha_hp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    double rc = 1.0 / (2.0 * M_PI * fc);
    return rc / (rc + 1.0 / sr);
}

static int run_subharm_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (inputs == NULL || !inputs->audio.has_audio || inputs->audio.data == NULL) {
        return -1;
    }
    int B = inputs->audio.batches > 0 ? (int)inputs->audio.batches : (batches > 0 ? batches : 1);
    int C = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
    int F = inputs->audio.frames > 0 ? (int)inputs->audio.frames : (frames > 0 ? frames : 1);
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    const double *audio = inputs->audio.data;
    double band_lo = json_get_double(descriptor->params_json, descriptor->params_len, "band_lo", 70.0);
    double band_hi = json_get_double(descriptor->params_json, descriptor->params_len, "band_hi", 160.0);
    double mix = json_get_double(descriptor->params_json, descriptor->params_len, "mix", 0.5);
    double drive = json_get_double(descriptor->params_json, descriptor->params_len, "drive", 1.0);
    double out_hp = json_get_double(descriptor->params_json, descriptor->params_len, "out_hp", 25.0);
    int use_div4 = json_get_bool(descriptor->params_json, descriptor->params_len, "use_div4", 0);
    double a_hp_in = alpha_hp(band_lo, sample_rate);
    double a_lp_in = alpha_lp(band_hi, sample_rate);
    double a_sub2 = alpha_lp(fmax(band_hi / 3.0, 30.0), sample_rate);
    double a_sub4 = use_div4 ? alpha_lp(fmax(band_hi / 5.0, 20.0), sample_rate) : 0.0;
    double a_env_attack = alpha_lp(100.0, sample_rate);
    double a_env_release = alpha_lp(5.0, sample_rate);
    double a_hp_out = alpha_hp(out_hp, sample_rate);
    size_t total = (size_t)B * (size_t)C * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    int need_resize = state->u.subharm.batches != B || state->u.subharm.channels != C || state->u.subharm.use_div4 != use_div4;
    if (need_resize) {
        free(state->u.subharm.hp_y);
        free(state->u.subharm.lp_y);
        free(state->u.subharm.prev);
        free(state->u.subharm.sign);
        free(state->u.subharm.ff2);
        free(state->u.subharm.ff4);
        free(state->u.subharm.ff4_count);
        free(state->u.subharm.sub2_lp);
        free(state->u.subharm.sub4_lp);
        free(state->u.subharm.env);
        free(state->u.subharm.hp_out_y);
        free(state->u.subharm.hp_out_x);
        state->u.subharm.hp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.lp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.prev = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.sign = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.ff2 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.sub2_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.env = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_x = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        if (use_div4) {
            state->u.subharm.ff4 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
            state->u.subharm.ff4_count = (int32_t *)calloc((size_t)B * (size_t)C, sizeof(int32_t));
            state->u.subharm.sub4_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        } else {
            free(state->u.subharm.ff4);
            free(state->u.subharm.ff4_count);
            free(state->u.subharm.sub4_lp);
            state->u.subharm.ff4 = NULL;
            state->u.subharm.ff4_count = NULL;
            state->u.subharm.sub4_lp = NULL;
        }
        state->u.subharm.batches = B;
        state->u.subharm.channels = C;
        state->u.subharm.use_div4 = use_div4;
    }
    if (state->u.subharm.hp_y == NULL || state->u.subharm.lp_y == NULL || state->u.subharm.prev == NULL || state->u.subharm.sign == NULL || state->u.subharm.ff2 == NULL || state->u.subharm.sub2_lp == NULL || state->u.subharm.env == NULL || state->u.subharm.hp_out_y == NULL || state->u.subharm.hp_out_x == NULL) {
        free(buffer);
        return -1;
    }
    subharmonic_process(
        audio,
        buffer,
        B,
        C,
        F,
        a_hp_in,
        a_lp_in,
        a_sub2,
        use_div4,
        a_sub4,
        a_env_attack,
        a_env_release,
        a_hp_out,
        drive,
        mix,
        state->u.subharm.hp_y,
        state->u.subharm.lp_y,
        state->u.subharm.prev,
        state->u.subharm.sign,
        state->u.subharm.ff2,
        state->u.subharm.ff4,
        state->u.subharm.ff4_count,
        state->u.subharm.sub2_lp,
        state->u.subharm.sub4_lp,
        state->u.subharm.env,
        state->u.subharm.hp_out_y,
        state->u.subharm.hp_out_x
    );
    *out_buffer = buffer;
    *out_channels = C;
    return 0;
}

static int run_osc_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    char wave_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "wave", wave_buf, sizeof(wave_buf))) {
        strcpy(wave_buf, "sine");
    }
    int accept_reset = json_get_bool(descriptor->params_json, descriptor->params_len, "accept_reset", 1);
    const EdgeRunnerParamView *freq_view = find_param(inputs, "freq");
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amp");
    const EdgeRunnerParamView *pan_view = find_param(inputs, "pan");
    const EdgeRunnerParamView *reset_view = accept_reset ? find_param(inputs, "reset") : NULL;
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    if (freq_view != NULL && freq_view->batches > 0) {
        B = (int)freq_view->batches;
    }
    if (freq_view != NULL && freq_view->frames > 0) {
        F = (int)freq_view->frames;
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;
    double *owned_freq = NULL;
    double *owned_amp = NULL;
    double *owned_pan = NULL;
    double *owned_reset = NULL;
    const double *freq = ensure_param_plane(freq_view, B, F, 0.0, &owned_freq);
    const double *amp = ensure_param_plane(amp_view, B, F, 1.0, &owned_amp);
    const double *pan = ensure_param_plane(pan_view, B, F, 0.0, &owned_pan);
    const double *reset = ensure_param_plane(reset_view, B, F, 0.0, &owned_reset);
    if (freq == NULL || amp == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        return -1;
    }
    size_t total = (size_t)B * (size_t)F;
    if (state->u.osc.phase == NULL || state->u.osc.batches != B) {
        free(state->u.osc.phase);
        state->u.osc.phase = (double *)calloc((size_t)B, sizeof(double));
        state->u.osc.batches = B;
    }
    if (state->u.osc.phase_buffer == NULL || state->u.osc.batches != B || state->u.osc.channels != 1) {
        free(state->u.osc.phase_buffer);
        state->u.osc.phase_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.wave_buffer == NULL || state->u.osc.batches != B || state->u.osc.channels != 1) {
        free(state->u.osc.wave_buffer);
        state->u.osc.wave_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.dphi_buffer == NULL || state->u.osc.batches != B || state->u.osc.channels != 1) {
        free(state->u.osc.dphi_buffer);
        state->u.osc.dphi_buffer = (double *)malloc(total * sizeof(double));
    }
    if (strcmp(wave_buf, "triangle") == 0) {
        if (state->u.osc.tri_state == NULL || state->u.osc.batches != B) {
            free(state->u.osc.tri_state);
            state->u.osc.tri_state = (double *)calloc((size_t)B, sizeof(double));
        }
    }
    if (state->u.osc.phase == NULL || state->u.osc.phase_buffer == NULL || state->u.osc.wave_buffer == NULL || state->u.osc.dphi_buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        return -1;
    }
    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            size_t idx = (size_t)b * (size_t)F + (size_t)f;
            state->u.osc.dphi_buffer[idx] = freq[idx] / sample_rate;
        }
    }
    const double *reset_ptr = accept_reset ? reset : NULL;
    phase_advance(state->u.osc.dphi_buffer, state->u.osc.phase_buffer, B, F, state->u.osc.phase, reset_ptr);
    if (strcmp(wave_buf, "saw") == 0) {
        osc_saw_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F);
    } else if (strcmp(wave_buf, "square") == 0) {
        osc_square_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, 0.5, state->u.osc.wave_buffer, B, F);
    } else if (strcmp(wave_buf, "triangle") == 0) {
        osc_triangle_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F, state->u.osc.tri_state);
    } else {
        for (int b = 0; b < B; ++b) {
            for (int f = 0; f < F; ++f) {
                size_t idx = (size_t)b * (size_t)F + (size_t)f;
                state->u.osc.wave_buffer[idx] = sin(state->u.osc.phase_buffer[idx] * 2.0 * M_PI);
            }
        }
    }
    int stereo = (pan_view != NULL && pan_view->data != NULL) ? 1 : 0;
    int channels = stereo ? 2 : 1;
    size_t total_out = (size_t)B * (size_t)channels * (size_t)F;
    double *buffer = (double *)malloc(total_out * sizeof(double));
    if (buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        return -1;
    }
    for (int b = 0; b < B; ++b) {
        for (int f = 0; f < F; ++f) {
            size_t idx = (size_t)b * (size_t)F + (size_t)f;
            double sample = state->u.osc.wave_buffer[idx] * amp[idx];
            if (stereo) {
                double pan_val = pan[idx];
                if (pan_val < -1.0) pan_val = -1.0;
                if (pan_val > 1.0) pan_val = 1.0;
                double angle = (pan_val + 1.0) * (M_PI / 4.0);
                double left = sample * cos(angle);
                double right = sample * sin(angle);
                buffer[((size_t)b * 2) * (size_t)F + (size_t)f] = left;
                buffer[((size_t)b * 2 + 1) * (size_t)F + (size_t)f] = right;
            } else {
                buffer[(size_t)b * (size_t)F + (size_t)f] = sample;
            }
        }
    }
    free(owned_freq);
    free(owned_amp);
    free(owned_pan);
    free(owned_reset);
    state->u.osc.channels = 1;
    state->u.osc.stereo = stereo;
    *out_buffer = buffer;
    *out_channels = stereo ? 2 : 1;
    return 0;
}

static int run_gain_node(
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int channels = (int)inputs->audio.channels;
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || channels <= 0) {
        channels = channels > 0 ? channels : 1;
        size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
        double *buffer = (double *)calloc(total, sizeof(double));
        if (buffer == NULL) {
            return -1;
        }
        *out_buffer = buffer;
        *out_channels = channels;
        return 0;
    }
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    const double *audio = inputs->audio.data;
    const EdgeRunnerParamView *gain_view = find_param(inputs, "gain");
    const double *gain = (gain_view != NULL) ? gain_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t base = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)frames;
            for (int f = 0; f < frames; ++f) {
                double sample = audio[base + (size_t)f];
                double g = gain != NULL ? gain[base + (size_t)f] : 1.0;
                buffer[base + (size_t)f] = sample * g;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static int run_mix_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int target_channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (target_channels <= 0) {
        target_channels = 1;
    }
    size_t total = (size_t)batches * (size_t)target_channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || inputs->audio.channels == 0) {
        memset(buffer, 0, total * sizeof(double));
        *out_buffer = buffer;
        *out_channels = target_channels;
        return 0;
    }
    int in_channels = (int)inputs->audio.channels;
    const double *audio = inputs->audio.data;
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            double sum = 0.0;
            for (int c = 0; c < in_channels; ++c) {
                size_t idx = ((size_t)b * (size_t)in_channels + (size_t)c) * (size_t)frames + (size_t)f;
                sum += audio[idx];
            }
            for (int oc = 0; oc < target_channels; ++oc) {
                size_t out_idx = ((size_t)b * (size_t)target_channels + (size_t)oc) * (size_t)frames + (size_t)f;
                buffer[out_idx] = sum;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = target_channels;
    return 0;
}

static int run_sine_osc_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    double initial_phase = json_get_double(descriptor->params_json, descriptor->params_len, "phase", 0.0);
    double normalized_phase = initial_phase - floor(initial_phase);
    if (state != NULL) {
        if (state->u.sine.phase == NULL || state->u.sine.batches != batches || state->u.sine.channels != channels) {
            free(state->u.sine.phase);
            state->u.sine.phase = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.sine.phase == NULL) {
                free(buffer);
                state->u.sine.batches = 0;
                state->u.sine.channels = 0;
                return -1;
            }
            state->u.sine.batches = batches;
            state->u.sine.channels = channels;
            state->u.sine.base_phase = normalized_phase;
            for (size_t idx = 0; idx < (size_t)batches * (size_t)channels; ++idx) {
                state->u.sine.phase[idx] = normalized_phase;
            }
        }
    }
    double *phase = state != NULL ? state->u.sine.phase : NULL;
    double base_freq = json_get_double(descriptor->params_json, descriptor->params_len, "frequency", 440.0);
    double base_amp = json_get_double(descriptor->params_json, descriptor->params_len, "amplitude", 0.5);
    const EdgeRunnerParamView *freq_view = find_param(inputs, "frequency");
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amplitude");
    const double *freq_data = freq_view != NULL ? freq_view->data : NULL;
    const double *amp_data = amp_view != NULL ? amp_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t bc = (size_t)b * (size_t)channels + (size_t)c;
            double phase_acc = phase != NULL ? phase[bc] : normalized_phase;
            for (int f = 0; f < frames; ++f) {
                size_t idx = bc * (size_t)frames + (size_t)f;
                double freq = freq_data != NULL ? freq_data[idx] : base_freq;
                double amp = amp_data != NULL ? amp_data[idx] : base_amp;
                double step = freq / sample_rate;
                phase_acc += step;
                phase_acc -= floor(phase_acc);
                buffer[idx] = sin(phase_acc * 2.0 * M_PI) * amp;
            }
            if (phase != NULL) {
                phase[bc] = phase_acc;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static int run_safety_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", (int)inputs->audio.channels);
    if (channels <= 0) {
        channels = (int)inputs->audio.channels;
    }
    if (channels <= 0) {
        channels = 1;
    }
    double alpha = json_get_double(descriptor->params_json, descriptor->params_len, "dc_alpha", 0.995);
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (state != NULL) {
        if (state->u.safety.state == NULL || state->u.safety.batches != batches || state->u.safety.channels != channels) {
            free(state->u.safety.state);
            state->u.safety.state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.safety.state == NULL) {
                free(buffer);
                state->u.safety.batches = 0;
                state->u.safety.channels = 0;
                return -1;
            }
            state->u.safety.batches = batches;
            state->u.safety.channels = channels;
        }
        state->u.safety.alpha = alpha;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL) {
        memset(buffer, 0, total * sizeof(double));
    } else {
        double *dc_state = NULL;
        if (state != NULL) {
            dc_state = state->u.safety.state;
        }
        if (dc_state == NULL) {
            dc_state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (dc_state == NULL) {
                free(buffer);
                return -1;
            }
            if (state != NULL) {
                state->u.safety.state = dc_state;
                state->u.safety.batches = batches;
                state->u.safety.channels = channels;
            }
        }
        dc_block(inputs->audio.data, buffer, batches, channels, frames, alpha, dc_state);
        for (size_t i = 0; i < total; ++i) {
            double v = buffer[i];
            if (v > 1.0) {
                buffer[i] = 1.0;
            } else if (v < -1.0) {
                buffer[i] = -1.0;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

int amp_run_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history
) {
    (void)channels;
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    node_kind_t kind = determine_node_kind(descriptor);
    if (kind == NODE_KIND_UNKNOWN) {
        return -3;
    }
    node_state_t *node_state = NULL;
    if (state != NULL && *state != NULL) {
        node_state = (node_state_t *)(*state);
    }
    if (node_state != NULL && node_state->kind != kind) {
        release_node_state(node_state);
        node_state = NULL;
        if (state != NULL) {
            *state = NULL;
        }
    }
    if (node_state == NULL) {
        node_state = (node_state_t *)calloc(1, sizeof(node_state_t));
        if (node_state == NULL) {
            return -1;
        }
        node_state->kind = kind;
        if (state != NULL) {
            *state = node_state;
        }
    }

    int rc = 0;
    switch (kind) {
        case NODE_KIND_CONSTANT:
            rc = run_constant_node(descriptor, batches, frames, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_GAIN:
            rc = run_gain_node(inputs, batches, frames, out_buffer, out_channels);
            break;
        case NODE_KIND_MIX:
            rc = run_mix_node(descriptor, inputs, batches, frames, out_buffer, out_channels);
            break;
        case NODE_KIND_SAFETY:
            rc = run_safety_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_SINE_OSC:
            rc = run_sine_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_CONTROLLER:
            rc = run_controller_node(descriptor, inputs, batches, frames, out_buffer, out_channels, history);
            break;
        case NODE_KIND_LFO:
            rc = run_lfo_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_ENVELOPE:
            rc = run_envelope_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_PITCH:
            rc = run_pitch_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_OSC:
            rc = run_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        case NODE_KIND_SUBHARM:
            rc = run_subharm_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state);
            break;
        default:
            rc = -3;
            break;
    }
    return rc;
}

void amp_free(double *buffer) {
    if (buffer != NULL) {
        free(buffer);
    }
}

void amp_release_state(void *state_ptr) {
    if (state_ptr == NULL) {
        return;
    }
    node_state_t *node_state = (node_state_t *)state_ptr;
    release_node_state(node_state);
}
"""
    try:
        ffi.set_source("_amp_ckernels_cffi", C_SRC)
        # compile lazy; this will create a module in-place and return its path
        module_path = ffi.compile(verbose=False)
        import importlib.util
        import sys
        from pathlib import Path

        compiled_path = Path(module_path)
        target_path = Path(__file__).resolve().parent / compiled_path.name
        if compiled_path.exists() and compiled_path != target_path:
            try:
                target_path.write_bytes(compiled_path.read_bytes())
            except Exception:
                # best-effort copy; continue even if it fails so runtime can still load from module_path
                target_path = compiled_path
        else:
            target_path = compiled_path

        spec = importlib.util.spec_from_file_location("_amp_ckernels_cffi", str(target_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load compiled module from {target_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_amp_ckernels_cffi"] = module
        spec.loader.exec_module(module)
        _impl = module
        AVAILABLE = True
        UNAVAILABLE_REASON = None
    except Exception as exc:
        # any compile/import error -> disable C backend
        AVAILABLE = False
        detail = traceback.format_exc()
        UNAVAILABLE_REASON = (
            "Failed to compile C kernels via cffi: "
            f"{exc}\n{detail}"
        )
except ModuleNotFoundError as exc:
    AVAILABLE = False
    UNAVAILABLE_REASON = f"cffi is not installed ({exc})"
except Exception as exc:
    AVAILABLE = False
    UNAVAILABLE_REASON = (
        "Unexpected error initialising cffi for C kernels: "
        f"{exc}"
    )


def _require_ctypes_ready(arr: np.ndarray, dtype: np.dtype, *, writable: bool) -> np.ndarray:
    """Validate that ``arr`` can be passed directly to a C kernel."""

    if arr.dtype != dtype:
        raise TypeError(f"expected dtype {dtype}, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("arrays passed to C kernels must be C-contiguous")
    if writable and not arr.flags.writeable:
        raise ValueError("writable arrays passed to C kernels must be writeable")
    return arr


DTYPE_FLOAT = np.dtype(np.float64)
DTYPE_INT32 = np.dtype(np.int32)
DTYPE_INT64 = np.dtype(np.int64)


def lfo_slew_c(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
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
    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, F = x_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if z0 is not None:
        if z0.shape != (B,):
            raise ValueError("z0 must have shape (B,)")
        z_buf = _require_ctypes_ready(z0, DTYPE_FLOAT, writable=True)
        z_ptr = ffi.cast("double *", z_buf.ctypes.data)
    else:
        z_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.lfo_slew(x_ptr, out_ptr, int(B), int(F), float(r), float(alpha), z_ptr)
    return out


def lfo_slew_py(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-Python sample-sequential fallback (fast with numpy per-row ops).

    Semantics: iterative recurrence z[n] = r*z[n-1] + alpha * x[n].
    """
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, F = x_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_py")
    if z0 is None:
        z = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        z = np.asarray(z0, dtype=DTYPE_FLOAT)
    for i in range(F):
        xi = x_buf[:, i]
        z = r * z + alpha * xi
        out[:, i] = z
    if z0 is not None:
        z0[:] = z
    return out


def lfo_slew_vector(
    x: np.ndarray,
    r: float,
    alpha: float,
    z0: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized closed-form solution equivalent to iterative recurrence.

    z[n] = r^n * z0 + alpha * r^n * sum_{k=0..n} r^{-k} * x[k]
    Implemented using np.cumsum on axis 1.
    """
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, F = x_buf.shape
    idx = np.arange(F, dtype=DTYPE_FLOAT)
    r_pow = r ** idx
    # handle r==0
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.where(r == 0.0, 0.0, r ** (-idx))
    accum = np.cumsum(x_buf * r_inv[None, :], axis=1)
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for lfo_slew_vector")
    out[:] = r_pow[None, :] * (alpha * accum)
    if z0 is not None:
        out += r_pow[None, :] * z0[:, None]
        z0[:] = out[:, -1]
    return out


def safety_filter_c(
    x: np.ndarray,
    a: float,
    prev_in: Optional[np.ndarray],
    prev_dc: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Call compiled ``safety_filter`` kernel without intermediate copies."""

    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for safety_filter_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if prev_in is not None:
        if prev_in.shape != (B, C):
            raise ValueError("prev_in must have shape (B, C)")
        prev_in_buf = _require_ctypes_ready(prev_in, DTYPE_FLOAT, writable=True)
        prev_in_ptr = ffi.cast("double *", prev_in_buf.ctypes.data)
    else:
        prev_in_ptr = ffi.cast("double *", ffi.NULL)

    if prev_dc is not None:
        if prev_dc.shape != (B, C):
            raise ValueError("prev_dc must have shape (B, C)")
        prev_dc_buf = _require_ctypes_ready(prev_dc, DTYPE_FLOAT, writable=True)
        prev_dc_ptr = ffi.cast("double *", prev_dc_buf.ctypes.data)
    else:
        prev_dc_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.safety_filter(x_ptr, out_ptr, int(B), int(C), int(F), float(a), prev_in_ptr, prev_dc_ptr)
    return out


def safety_filter_py(
    x: np.ndarray,
    a: float,
    prev_in: Optional[np.ndarray],
    prev_dc: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for safety_filter_py")
    pi = np.zeros((B, C), dtype=DTYPE_FLOAT) if prev_in is None else np.asarray(prev_in, dtype=DTYPE_FLOAT)
    pd = np.zeros((B, C), dtype=DTYPE_FLOAT) if prev_dc is None else np.asarray(prev_dc, dtype=DTYPE_FLOAT)
    for b in range(B):
        for c in range(C):
            if F <= 0:
                continue
            # compute diffs
            diffs = np.empty(F, dtype=DTYPE_FLOAT)
            diffs[0] = x_buf[b, c, 0] - pi[b, c]
            if F > 1:
                diffs[1:] = x_buf[b, c, 1:] - x_buf[b, c, :-1]
            powers = a ** np.arange(F, dtype=DTYPE_FLOAT)
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_p = 1.0 / powers
            accum = np.cumsum(diffs * inv_p) + (a * pd[b, c])
            y = accum * powers
            out[b, c, :] = y
            pi[b, c] = x_buf[b, c, -1]
            pd[b, c] = y[-1]
    if prev_in is not None:
        prev_in[:] = pi
    if prev_dc is not None:
        prev_dc[:] = pd
    return out


def dc_block_c(
    x: np.ndarray,
    a: float,
    state: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    x_buf = _require_ctypes_ready(np.asarray(x), DTYPE_FLOAT, writable=False)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for dc_block_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if state is not None:
        if state.shape != (B, C):
            raise ValueError("state must have shape (B, C)")
        state_buf = _require_ctypes_ready(state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", state_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    x_ptr = ffi.cast("const double *", x_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.dc_block(x_ptr, out_ptr, int(B), int(C), int(F), float(a), state_ptr)
    return out


def dc_block_py(
    x: np.ndarray,
    a: float,
    state: Optional[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    x_buf = np.asarray(x, dtype=DTYPE_FLOAT)
    B, C, F = x_buf.shape
    if out is None:
        out = np.empty((B, C, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, C, F):
            raise ValueError("out has incorrect shape for dc_block_py")
    st = np.zeros((B, C), dtype=DTYPE_FLOAT) if state is None else np.asarray(state, dtype=DTYPE_FLOAT)
    for b in range(B):
        for c in range(C):
            dc = st[b, c]
            for i in range(F):
                xi = x_buf[b, c, i]
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


def phase_advance_c(
    dphi: np.ndarray,
    reset: np.ndarray | None,
    phase_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    dphi_buf = _require_ctypes_ready(np.asarray(dphi), DTYPE_FLOAT, writable=False)
    B, F = dphi_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for phase_advance_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if phase_state is not None:
        if phase_state.shape != (B,):
            raise ValueError("phase_state must have shape (B,)")
        phase_buf = _require_ctypes_ready(phase_state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", phase_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    if reset is not None:
        reset_buf = _require_ctypes_ready(np.asarray(reset), DTYPE_FLOAT, writable=False)
        if reset_buf.shape != (B, F):
            raise ValueError("reset must have shape (B, F)")
        reset_ptr = ffi.cast("const double *", reset_buf.ctypes.data)
    else:
        reset_ptr = ffi.cast("const double *", ffi.NULL)

    dphi_ptr = ffi.cast("const double *", dphi_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.phase_advance(dphi_ptr, out_ptr, int(B), int(F), state_ptr, reset_ptr)
    return out


def phase_advance_py(
    dphi: np.ndarray,
    reset: np.ndarray | None,
    phase_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    dphi_buf = np.asarray(dphi, dtype=DTYPE_FLOAT)
    B, F = dphi_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for phase_advance_py")
    if phase_state is None:
        cur = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        cur = np.asarray(phase_state, dtype=DTYPE_FLOAT)
    if reset is not None:
        reset_buf = np.asarray(reset, dtype=DTYPE_FLOAT)
    else:
        reset_buf = None
    for i in range(F):
        if reset_buf is not None:
            mask = reset_buf[:, i] > 0.5
            if np.any(mask):
                cur = np.where(mask, 0.0, cur)
        cur = (cur + dphi_buf[:, i]) % 1.0
        out[:, i] = cur
    if phase_state is not None:
        phase_state[:] = cur
    return out


def portamento_smooth_c(
    freq_target: np.ndarray,
    port_mask: np.ndarray | None,
    slide_time: np.ndarray | None,
    slide_damp: np.ndarray | None,
    sr: int,
    freq_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    freq_buf = _require_ctypes_ready(np.asarray(freq_target), DTYPE_FLOAT, writable=False)
    B, F = freq_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for portamento_smooth_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    if port_mask is not None:
        port_buf = _require_ctypes_ready(np.asarray(port_mask), DTYPE_FLOAT, writable=False)
        if port_buf.shape != (B, F):
            raise ValueError("port_mask must have shape (B, F)")
        port_ptr = ffi.cast("const double *", port_buf.ctypes.data)
    else:
        port_ptr = ffi.cast("const double *", ffi.NULL)

    if slide_time is not None:
        st_buf = _require_ctypes_ready(np.asarray(slide_time), DTYPE_FLOAT, writable=False)
        if st_buf.shape != (B, F):
            raise ValueError("slide_time must have shape (B, F)")
        st_ptr = ffi.cast("const double *", st_buf.ctypes.data)
    else:
        st_ptr = ffi.cast("const double *", ffi.NULL)

    if slide_damp is not None:
        sd_buf = _require_ctypes_ready(np.asarray(slide_damp), DTYPE_FLOAT, writable=False)
        if sd_buf.shape != (B, F):
            raise ValueError("slide_damp must have shape (B, F)")
        sd_ptr = ffi.cast("const double *", sd_buf.ctypes.data)
    else:
        sd_ptr = ffi.cast("const double *", ffi.NULL)

    if freq_state is not None:
        if freq_state.shape != (B,):
            raise ValueError("freq_state must have shape (B,)")
        state_buf = _require_ctypes_ready(freq_state, DTYPE_FLOAT, writable=True)
        state_ptr = ffi.cast("double *", state_buf.ctypes.data)
    else:
        state_ptr = ffi.cast("double *", ffi.NULL)

    ft_ptr = ffi.cast("const double *", freq_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    _impl.lib.portamento_smooth(ft_ptr, port_ptr, st_ptr, sd_ptr, int(B), int(F), int(sr), state_ptr, out_ptr)
    return out


def portamento_smooth_py(
    freq_target: np.ndarray,
    port_mask: np.ndarray | None,
    slide_time: np.ndarray | None,
    slide_damp: np.ndarray | None,
    sr: int,
    freq_state: np.ndarray | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    freq_buf = np.asarray(freq_target, dtype=DTYPE_FLOAT)
    B, F = freq_buf.shape
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for portamento_smooth_py")
    if freq_state is None:
        cur = np.zeros(B, dtype=DTYPE_FLOAT)
    else:
        cur = np.asarray(freq_state, dtype=DTYPE_FLOAT)
    port_buf = None if port_mask is None else np.asarray(port_mask, dtype=DTYPE_FLOAT)
    st_buf = None if slide_time is None else np.asarray(slide_time, dtype=DTYPE_FLOAT)
    sd_buf = None if slide_damp is None else np.asarray(slide_damp, dtype=DTYPE_FLOAT)
    for i in range(F):
        target = freq_buf[:, i]
        if port_buf is not None:
            active = port_buf[:, i] > 0.5
        else:
            active = np.zeros(B, dtype=bool)
        frames_const = np.maximum(st_buf[:, i] * float(sr) if st_buf is not None else 1.0, 1.0)
        alpha = np.exp(-1.0 / frames_const)
        if sd_buf is not None:
            alpha = alpha ** (1.0 + np.clip(sd_buf[:, i], 0.0, None))
        cur = np.where(active, alpha * cur + (1.0 - alpha) * target, target)
        out[:, i] = cur
    if freq_state is not None:
        freq_state[:] = cur
    return out


def arp_advance_c(
    seq: np.ndarray,
    seq_len: int,
    B: int,
    F: int,
    step_state: np.ndarray,
    timer_state: np.ndarray,
    fps: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")

    seq_buf = _require_ctypes_ready(np.asarray(seq), DTYPE_FLOAT, writable=False)
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for arp_advance_c")
        _require_ctypes_ready(out, DTYPE_FLOAT, writable=True)

    step_buf = _require_ctypes_ready(np.asarray(step_state), DTYPE_INT32, writable=True)
    timer_buf = _require_ctypes_ready(np.asarray(timer_state), DTYPE_INT32, writable=True)

    seq_ptr = ffi.cast("const double *", seq_buf.ctypes.data)
    out_ptr = ffi.cast("double *", out.ctypes.data)
    step_ptr = ffi.cast("int *", step_buf.ctypes.data)
    timer_ptr = ffi.cast("int *", timer_buf.ctypes.data)
    _impl.lib.arp_advance(seq_ptr, int(seq_len), out_ptr, int(B), int(F), step_ptr, timer_ptr, int(fps))
    return out


def arp_advance_py(
    seq: np.ndarray,
    seq_len: int,
    B: int,
    F: int,
    step_state: np.ndarray,
    timer_state: np.ndarray,
    fps: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        out = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out.shape != (B, F):
            raise ValueError("out has incorrect shape for arp_advance_py")
        out.fill(0.0)
    seq_list = list(np.asarray(seq, dtype=DTYPE_FLOAT).ravel())
    if len(seq_list) == 0:
        seq_list = [0.0]
    seq_vals = np.asarray(seq_list, dtype=DTYPE_FLOAT)
    step = np.asarray(step_state, dtype=DTYPE_INT32)
    timer = np.asarray(timer_state, dtype=DTYPE_INT32)
    for i in range(F):
        idx = step % seq_vals.size
        out[:, i] = seq_vals[idx]
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
    *,
    out_amp: np.ndarray | None = None,
    out_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not AVAILABLE or _impl is None:
        raise RuntimeError("C kernel not available")
    B = trigger.shape[0]
    F = trigger.shape[1]
    dtype_float = np.dtype(np.float64)
    dtype_stage = np.dtype(np.int32)
    dtype_acts = np.dtype(np.int64)

    trig_b = _require_ctypes_ready(trigger, dtype_float, writable=False)
    gate_b = _require_ctypes_ready(gate, dtype_float, writable=False)
    drone_b = _require_ctypes_ready(drone, dtype_float, writable=False)
    vel_b = _require_ctypes_ready(velocity, dtype_float, writable=False)
    stage_b = _require_ctypes_ready(stage, dtype_stage, writable=True)
    value_b = _require_ctypes_ready(value, dtype_float, writable=True)
    timer_b = _require_ctypes_ready(timer, dtype_float, writable=True)
    vel_state_b = _require_ctypes_ready(vel_state, dtype_float, writable=True)
    activ_b = _require_ctypes_ready(activations, dtype_acts, writable=True)
    rel_b = _require_ctypes_ready(release_start, dtype_float, writable=True)

    if out_amp is None:
        out_amp = np.empty((B, F), dtype=dtype_float)
    else:
        if out_amp.shape != (B, F):
            raise ValueError("out_amp has incorrect shape")
        _require_ctypes_ready(out_amp, dtype_float, writable=True)
    if out_reset is None:
        out_reset = np.empty((B, F), dtype=dtype_float)
    else:
        if out_reset.shape != (B, F):
            raise ValueError("out_reset has incorrect shape")
        _require_ctypes_ready(out_reset, dtype_float, writable=True)

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
    *,
    out_amp: np.ndarray | None = None,
    out_reset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback envelope processor."""

    trigger_buf = np.asarray(trigger, dtype=DTYPE_FLOAT)
    gate_buf = np.asarray(gate, dtype=DTYPE_FLOAT)
    drone_buf = np.asarray(drone, dtype=DTYPE_FLOAT)
    velocity_buf = np.asarray(velocity, dtype=DTYPE_FLOAT)

    B, F = trigger_buf.shape

    if stage.dtype != DTYPE_INT32:
        raise TypeError("stage must have dtype int32")
    stage_buf = stage
    if value.dtype != DTYPE_FLOAT:
        raise TypeError("value must have dtype float64")
    value_buf = value
    if timer.dtype != DTYPE_FLOAT:
        raise TypeError("timer must have dtype float64")
    timer_buf = timer
    if vel_state.dtype != DTYPE_FLOAT:
        raise TypeError("vel_state must have dtype float64")
    vel_state_buf = vel_state
    if activations.dtype != DTYPE_INT64:
        raise TypeError("activations must have dtype int64")
    activ_buf = activations
    if release_start.dtype != DTYPE_FLOAT:
        raise TypeError("release_start must have dtype float64")
    rel_buf = release_start

    if out_amp is None:
        out_amp = np.empty((B, F), dtype=DTYPE_FLOAT)
    else:
        if out_amp.shape != (B, F):
            raise ValueError("out_amp has incorrect shape")
        out_amp = np.asarray(out_amp, dtype=DTYPE_FLOAT)

    if out_reset is None:
        out_reset = np.zeros((B, F), dtype=DTYPE_FLOAT)
    else:
        if out_reset.shape != (B, F):
            raise ValueError("out_reset has incorrect shape")
        out_reset = np.asarray(out_reset, dtype=DTYPE_FLOAT)
        out_reset.fill(0.0)

    for b in range(B):
        st = int(stage_buf[b])
        val = float(value_buf[b])
        tim = float(timer_buf[b])
        vel = float(vel_state_buf[b])
        acts = int(activ_buf[b])
        rel_start_val = float(rel_buf[b])

        trig_line = trigger_buf[b] > 0.5
        gate_line = gate_buf[b] > 0.5
        drone_line = drone_buf[b] > 0.5

        for i in range(F):
            trig = bool(trig_line[i])
            gate_on = bool(gate_line[i])
            drone_on = bool(drone_line[i])

            if trig:
                st = 1
                tim = 0.0
                val = 0.0
                vel = float(velocity_buf[b, i])
                if vel < 0.0:
                    vel = 0.0
                rel_start_val = vel
                acts += 1
                if send_resets:
                    out_reset[b, i] = 1.0
            elif st == 0 and (gate_on or drone_on):
                st = 1
                tim = 0.0
                val = 0.0
                vel = float(velocity_buf[b, i])
                if vel < 0.0:
                    vel = 0.0
                rel_start_val = vel
                acts += 1
                if send_resets:
                    out_reset[b, i] = 1.0

            if st == 1:
                if atk_frames <= 0:
                    val = vel
                    if hold_frames > 0:
                        st = 2
                    elif dec_frames > 0:
                        st = 3
                    else:
                        st = 4
                    tim = 0.0
                else:
                    step = vel / float(atk_frames if atk_frames > 0 else 1)
                    val += step
                    if val > vel:
                        val = vel
                    tim += 1.0
                    if tim >= atk_frames:
                        val = vel
                        if hold_frames > 0:
                            st = 2
                        elif dec_frames > 0:
                            st = 3
                        else:
                            st = 4
                        tim = 0.0
            elif st == 2:
                val = vel
                if hold_frames <= 0:
                    if dec_frames > 0:
                        st = 3
                    else:
                        st = 4
                    tim = 0.0
                else:
                    tim += 1.0
                    if tim >= hold_frames:
                        if dec_frames > 0:
                            st = 3
                        else:
                            st = 4
                        tim = 0.0
            elif st == 3:
                target = vel * sustain_level
                if dec_frames <= 0:
                    val = target
                    st = 4
                    tim = 0.0
                else:
                    delta = (vel - target) / float(dec_frames if dec_frames > 0 else 1)
                    candidate = val - delta
                    if candidate < target:
                        candidate = target
                    val = candidate
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
                        rel_start_val = val
                        tim = 0.0
                elif not gate_on and not drone_on:
                    st = 5
                    rel_start_val = val
                    tim = 0.0
            elif st == 5:
                if rel_frames <= 0:
                    val = 0.0
                    st = 0
                    tim = 0.0
                else:
                    step = rel_start_val / float(rel_frames if rel_frames > 0 else 1)
                    candidate = val - step
                    if candidate < 0.0:
                        candidate = 0.0
                    val = candidate
                    tim += 1.0
                    if tim >= rel_frames:
                        val = 0.0
                        st = 0
                        tim = 0.0
                if gate_on or drone_on:
                    st = 1
                    tim = 0.0
                    val = 0.0
                    vel = float(velocity_buf[b, i])
                    if vel < 0.0:
                        vel = 0.0
                    rel_start_val = vel
                    acts += 1
                    if send_resets:
                        out_reset[b, i] = 1.0

            if val < 0.0:
                val = 0.0
            out_amp[b, i] = val

        stage_buf[b] = st
        value_buf[b] = val
        timer_buf[b] = tim
        vel_state_buf[b] = vel
        activ_buf[b] = acts
        rel_buf[b] = rel_start_val

    return out_amp, out_reset


