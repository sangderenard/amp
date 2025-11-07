
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include <ctype.h>
#include <errno.h>
#include <float.h>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__cplusplus)
#include <Eigen/Dense>
#endif
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#if defined(__GNUC__) && !defined(_WIN32) && !defined(_WIN64)
#include <execinfo.h>
#endif
/* POSIX mkdir() */
#include <sys/stat.h>
#include <sys/types.h>

#include "amp_native.h"
#include "amp_fft_backend.h"
#include "amp_debug_alloc.h"

#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458176568
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
/* Debug: track last allocation element count for diagnostics. */
static size_t amp_last_alloc_count = 0;

AMP_CAPI size_t amp_last_alloc_count_get(void) {
    return amp_last_alloc_count;
}

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
/* Persistent log file handles and allocator wrappers now reside in amp_debug_alloc.c. */

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

AMP_CAPI EdgeRunnerCompiledPlan *amp_load_compiled_plan(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    AMP_LOG_NATIVE_CALL("amp_load_compiled_plan", descriptor_len, plan_len);
    AMP_LOG_GENERATED("amp_load_compiled_plan", (size_t)descriptor_blob, (size_t)plan_blob);
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

AMP_CAPI void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan) {
    AMP_LOG_NATIVE_CALL("amp_release_compiled_plan", (size_t)(plan != NULL), 0);
    AMP_LOG_GENERATED("amp_release_compiled_plan", (size_t)plan, 0);
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

AMP_CAPI EdgeRunnerControlHistory *amp_load_control_history(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
) {
    AMP_LOG_NATIVE_CALL("amp_load_control_history", blob_len, (size_t)frames_hint);
    AMP_LOG_GENERATED("amp_load_control_history", (size_t)blob, (size_t)frames_hint);
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

AMP_CAPI void amp_release_control_history(EdgeRunnerControlHistory *history) {
    AMP_LOG_NATIVE_CALL("amp_release_control_history", (size_t)(history != NULL), 0);
    AMP_LOG_GENERATED("amp_release_control_history", (size_t)history, 0);
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
    NODE_KIND_PITCH_SHIFT,
    NODE_KIND_OSC,
    NODE_KIND_OSC_PITCH,
    NODE_KIND_DRIVER,
    NODE_KIND_SUBHARM,
    NODE_KIND_RESAMPLER,
    NODE_KIND_FFT_DIV,
    NODE_KIND_SPECTRAL_DRIVE,
} node_kind_t;

typedef struct {
    node_kind_t kind;
    union {
        struct {
            double value;
            int channels;
        } constant;
        struct {
            double *last_freq;
            int out_channels;
        } mix;
        struct {
            double *state;
            int batches;
            int channels;
            double alpha;
            double *last_freq;
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
            double *integrator_state;
            double *op_amp_state;
            int batches;
            int channels;
            double base_phase;
            int stereo;
            int driver_channels;
            int mode;
        } osc;
        struct {
            double *phase;
            double *harmonics;
            int harmonic_count;
            int batches;
            int mode;
        } driver;
        struct {
            double *analysis_window;
            double *synthesis_window;
            double *prev_phase;
            double *phase_accum;
            int window_size;
            int hop_size;
            int resynthesis_hop;
        } pitch_shift;
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
            double *last_value;
            int batches;
        } osc_pitch;
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
        struct {
            double *input_buffer;
            double *divisor_buffer;
            double *divisor_imag_buffer;
            double *phase_buffer;
            double *lower_buffer;
            double *upper_buffer;
            double *filter_buffer;
            double *window;
            double *work_real;
            double *work_imag;
            double *div_real;
            double *div_imag;
            double *ifft_real;
            double *ifft_imag;
            double *result_buffer;
            double *div_fft_real;
            double *div_fft_imag;
            double *recomb_buffer;
            double *batch_input_real;
            double *batch_input_imag;
            double *batch_div_real;
            double *batch_div_imag;
            int *batch_slot_map;
            size_t *batch_slot_offset;
            size_t *batch_output_index;
            double *batch_phase;
            double *batch_lower;
            double *batch_upper;
            double *batch_filter;
            double *batch_epsilon;
            int window_size;
            int algorithm;
            int window_kind;
            int filled;
            int position;
            double epsilon;
            int batches;
            int channels;
            int slots;
            int recomb_filled;
            double last_phase;
            double last_lower;
            double last_upper;
            double last_filter;
            double total_heat;
            int dynamic_carrier_band_count;
            double dynamic_carrier_last_sum;
            double *dynamic_phase;
            double *dynamic_step_re;
            double *dynamic_step_im;
            int dynamic_k_active;
            int enable_remainder;
            double remainder_energy;
            int max_batch_windows;
            int batch_capacity;
            int backend_mode;
            int backend_hop;
            void **backend_stream_signal;
            void **backend_stream_div_real;
            void **backend_stream_div_imag;
            int backend_stream_window_size;
            int backend_stream_window_kind;
            int backend_stream_hop;
        } fftdiv;
        struct {
            int mode;
        } spectral_drive;
        struct {
            double *last_values;
            double *rate_window;
            uint32_t window_size;
            uint32_t window_index;
            uint32_t window_count;
            double window_sum;
            double ema_alpha;
            double last_rate;
            double fixed_sample_rate;
            int free_rate;
            uint32_t channels;
            uint32_t batches;
        } resampler;
    } u;
} node_state_t;

typedef enum {
    OSC_MODE_POLYBLEP = 0,
    OSC_MODE_INTEGRATOR = 1,
    OSC_MODE_OP_AMP = 2
} osc_mode_t;

static int json_copy_string(const char *json, size_t json_len, const char *key, char *out, size_t out_len);

static int parse_osc_mode(const char *json, size_t json_len, int default_mode) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "mode", buffer, sizeof(buffer))) {
        return default_mode;
    }
    for (char *p = buffer; *p != '\0'; ++p) {
        if (*p >= 'A' && *p <= 'Z') {
            *p = (char)(*p - 'A' + 'a');
        }
    }
    if (strcmp(buffer, "integrator") == 0 || strcmp(buffer, "blep_integrator") == 0) {
        return OSC_MODE_INTEGRATOR;
    }
    if (strcmp(buffer, "op_amp") == 0 || strcmp(buffer, "opamp") == 0 || strcmp(buffer, "slew_opamp") == 0) {
        return OSC_MODE_OP_AMP;
    }
    return default_mode;
}

typedef enum {
    DRIVER_MODE_QUARTZ = 0,
    DRIVER_MODE_PIEZO = 1,
    DRIVER_MODE_CUSTOM = 2
} driver_mode_t;

static int parse_driver_mode(const char *json, size_t json_len, int default_mode) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "mode", buffer, sizeof(buffer))) {
        return default_mode;
    }
    for (char *p = buffer; *p != '\0'; ++p) {
        if (*p >= 'A' && *p <= 'Z') {
            *p = (char)(*p - 'A' + 'a');
        }
    }
    if (strcmp(buffer, "piezo") == 0 || strcmp(buffer, "piezoelectric") == 0) {
        return DRIVER_MODE_PIEZO;
    }
    if (strcmp(buffer, "custom") == 0 || strcmp(buffer, "harmonic") == 0) {
        return DRIVER_MODE_CUSTOM;
    }
    return DRIVER_MODE_QUARTZ;
}

static void fft_state_free_buffers(node_state_t *state);

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
        free(state->u.osc.integrator_state);
        free(state->u.osc.op_amp_state);
        state->u.osc.phase = NULL;
        state->u.osc.phase_buffer = NULL;
        state->u.osc.wave_buffer = NULL;
        state->u.osc.dphi_buffer = NULL;
        state->u.osc.tri_state = NULL;
        state->u.osc.integrator_state = NULL;
        state->u.osc.op_amp_state = NULL;
        state->u.osc.batches = 0;
        state->u.osc.channels = 0;
        state->u.osc.stereo = 0;
        state->u.osc.driver_channels = 0;
        state->u.osc.mode = OSC_MODE_POLYBLEP;
    }
    if (state->kind == NODE_KIND_DRIVER) {
        free(state->u.driver.phase);
        free(state->u.driver.harmonics);
        state->u.driver.phase = NULL;
        state->u.driver.harmonics = NULL;
        state->u.driver.harmonic_count = 0;
        state->u.driver.batches = 0;
        state->u.driver.mode = DRIVER_MODE_QUARTZ;
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
    if (state->kind == NODE_KIND_PITCH_SHIFT) {
        free(state->u.pitch_shift.analysis_window);
        free(state->u.pitch_shift.synthesis_window);
        free(state->u.pitch_shift.prev_phase);
        free(state->u.pitch_shift.phase_accum);
        state->u.pitch_shift.analysis_window = NULL;
        state->u.pitch_shift.synthesis_window = NULL;
        state->u.pitch_shift.prev_phase = NULL;
        state->u.pitch_shift.phase_accum = NULL;
        state->u.pitch_shift.window_size = 0;
        state->u.pitch_shift.hop_size = 0;
        state->u.pitch_shift.resynthesis_hop = 0;
    }
    if (state->kind == NODE_KIND_OSC_PITCH) {
        free(state->u.osc_pitch.last_value);
        state->u.osc_pitch.last_value = NULL;
        state->u.osc_pitch.batches = 0;
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
    if (state->kind == NODE_KIND_RESAMPLER) {
        free(state->u.resampler.last_values);
        free(state->u.resampler.rate_window);
        state->u.resampler.last_values = NULL;
        state->u.resampler.rate_window = NULL;
        state->u.resampler.window_size = 0;
        state->u.resampler.window_index = 0;
        state->u.resampler.window_count = 0;
        state->u.resampler.window_sum = 0.0;
        state->u.resampler.ema_alpha = 0.0;
        state->u.resampler.last_rate = 0.0;
        state->u.resampler.fixed_sample_rate = 0.0;
        state->u.resampler.free_rate = 0;
        state->u.resampler.channels = 0;
        state->u.resampler.batches = 0;
    }
    if (state->kind == NODE_KIND_FFT_DIV) {
        fft_state_free_buffers(state);
        state->u.fftdiv.total_heat = 0.0;
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
    if (strcmp(descriptor->type_name, "PitchShiftNode") == 0
        || strcmp(descriptor->type_name, "pitch_shift") == 0) {
        return NODE_KIND_PITCH_SHIFT;
    }
    if (strcmp(descriptor->type_name, "OscillatorPitchNode") == 0
        || strcmp(descriptor->type_name, "oscillator_pitch") == 0) {
        return NODE_KIND_OSC_PITCH;
    }
    if (strcmp(descriptor->type_name, "OscNode") == 0) {
        return NODE_KIND_OSC;
    }
    if (strcmp(descriptor->type_name, "ParametricDriverNode") == 0
        || strcmp(descriptor->type_name, "parametric_driver") == 0) {
        return NODE_KIND_DRIVER;
    }
    if (strcmp(descriptor->type_name, "SubharmonicLowLifterNode") == 0) {
        return NODE_KIND_SUBHARM;
    }
    if (strcmp(descriptor->type_name, "ResamplerNode") == 0
        || strcmp(descriptor->type_name, "resampler") == 0) {
        return NODE_KIND_RESAMPLER;
    }
    if (strcmp(descriptor->type_name, "FFTDivisionNode") == 0) {
        return NODE_KIND_FFT_DIV;
    }
    if (strcmp(descriptor->type_name, "SpectralDriveNode") == 0
        || strcmp(descriptor->type_name, "spectral_drive") == 0) {
        return NODE_KIND_SPECTRAL_DRIVE;
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
    /* Register destination buffer so mem-op logging can correlate writes. */
    amp_debug_register_alloc(out, out_len);
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
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            /* log attempted oversize copy */
            void *stack_probe = (void *)&pattern;
            AMP_DEBUG_LOG_MEMOPS("PRECOPY json_copy_string base=%p dest=%p dest_cap=%zu req_len=%zu stack=%p\n", out, out, out_len, length, stack_probe);
#endif
            length = out_len > 0 ? out_len - 1 : 0;
        } else {
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            void *stack_probe = (void *)&pattern;
            AMP_DEBUG_LOG_MEMOPS("PRECOPY json_copy_string base=%p dest=%p dest_cap=%zu req_len=%zu stack=%p\n", out, out, out_len, length, stack_probe);
#endif
        }
        /* use safe copy that respects the provided destination capacity */
        if (out_len > 0 && length > 0) {
            memcpy(out, start, length);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            /* POSTCOPY: record that we actually wrote to 'out' */
            AMP_DEBUG_LOG_MEMOPS("POSTCOPY json_copy_string dest=%p wrote=%zu\n", out, length);
#endif
        }
        out[length] = '\0';
        amp_debug_unregister_alloc(out);
        return 1;
    }
    amp_debug_unregister_alloc(out);
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
    /* Register tokens buffer for correlation of PRECOPY/MEMCPY events */
    amp_debug_register_alloc(tokens, (size_t)max_tokens * 64);
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
#if defined(AMP_NATIVE_ENABLE_LOGGING)
    void *tokens_base = (void *)tokens;
    AMP_DEBUG_LOG_MEMOPS("PRECOPY parse_csv_tokens base=%p dest=%p dest_cap=%d req_len=%zu\n", tokens_base, tokens[count], 64, len);
#endif
    if (len >= 63) {
        len = 63;
    }
    if (len > 0) {
        /* write with postcopy logging and bounds assertion */
        memcpy(tokens[count], start, len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
        AMP_DEBUG_LOG_MEMOPS("POSTCOPY parse_csv_tokens dest=%p wrote=%zu token_idx=%d\n", tokens[count], len, count);
#endif
    }
    tokens[count][len] = '\0';
    count++;
    }
    amp_debug_unregister_alloc(tokens);
    return count;
}

static int parse_controller_sources(const char *csv, controller_source_t *items, int max_items) {
    if (csv == NULL || items == NULL || max_items <= 0) {
        return 0;
    }
    /* Register items buffer so writes to items->output/source are tracked */
    amp_debug_register_alloc(items, (size_t)max_items * sizeof(controller_source_t));
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
#if defined(AMP_NATIVE_ENABLE_LOGGING)
        void *items_base = (void *)items;
        AMP_DEBUG_LOG_MEMOPS("PRECOPY parse_controller_sources.output base=%p dest=%p dest_cap=%zu req_len=%zu\n", items_base, items[count].output, (size_t)sizeof(items[count].output), key_len);
#endif
        if (key_len >= sizeof(items[count].output)) {
            key_len = sizeof(items[count].output) - 1;
        }
        if (key_len > 0) {
            memcpy(items[count].output, cursor, key_len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            AMP_DEBUG_LOG_MEMOPS("POSTCOPY parse_controller_sources.output dest=%p wrote=%zu idx=%d\n", items[count].output, key_len, count);
#endif
        }
        items[count].output[key_len] = '\0';
        cursor = eq + 1;
        const char *end = strchr(cursor, ',');
        if (end == NULL) {
            end = cursor + strlen(cursor);
        }
        size_t value_len = (size_t)(end - cursor);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
        void *items_base_src = (void *)items;
        AMP_DEBUG_LOG_MEMOPS("PRECOPY parse_controller_sources.source base=%p dest=%p dest_cap=%zu req_len=%zu\n", items_base_src, items[count].source, (size_t)sizeof(items[count].source), value_len);
#endif
        if (value_len >= sizeof(items[count].source)) {
            value_len = sizeof(items[count].source) - 1;
        }
        if (value_len > 0) {
            memcpy(items[count].source, cursor, value_len);
#if defined(AMP_NATIVE_ENABLE_LOGGING)
            AMP_DEBUG_LOG_MEMOPS("POSTCOPY parse_controller_sources.source dest=%p wrote=%zu idx=%d\n", items[count].source, value_len, count);
#endif
        }
        items[count].source[value_len] = '\0';
        cursor = end;
        count++;
    }
    amp_debug_unregister_alloc(items);
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
    amp_last_alloc_count = total;
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

#define FFT_ALGORITHM_EIGEN 0
#define FFT_ALGORITHM_DFT 1
#define FFT_ALGORITHM_DYNAMIC_OSCILLATORS 2
#define FFT_ALGORITHM_HOOK 3

#define PITCH_SHIFT_DEFAULT_WINDOW 1024
#define PITCH_SHIFT_DEFAULT_HOP 256
#define PITCH_SHIFT_DEFAULT_RESYNTH_HOP 256

#define FFT_WINDOW_RECTANGULAR 0
#define FFT_WINDOW_HANN 1
#define FFT_WINDOW_HAMMING 2

#define FFT_DYNAMIC_CARRIER_LIMIT 64U

static int round_to_int(double value) {
    if (value >= 0.0) {
        return (int)(value + 0.5);
    }
    return (int)(value - 0.5);
}

static int clamp_algorithm_kind(int kind) {
    switch (kind) {
        case FFT_ALGORITHM_EIGEN:
        case FFT_ALGORITHM_DFT:
        case FFT_ALGORITHM_DYNAMIC_OSCILLATORS:
        case FFT_ALGORITHM_HOOK:
            return kind;
        default:
            break;
    }
    return FFT_ALGORITHM_EIGEN;
}

static int clamp_window_kind(int kind) {
    switch (kind) {
        case FFT_WINDOW_RECTANGULAR:
        case FFT_WINDOW_HANN:
        case FFT_WINDOW_HAMMING:
            return kind;
        default:
            break;
    }
    return FFT_WINDOW_HANN;
}

static double clamp_unit_double(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double wrap_phase_two_pi(double phase) {
    double wrapped = fmod(phase, 2.0 * M_PI);
    if (wrapped < 0.0) {
        wrapped += 2.0 * M_PI;
    }
    return wrapped;
}

static double compute_band_gain(double ratio, double lower, double upper, double intensity) {
    /* The minimum inside/outside gain is clamped to 1e-6 to avoid hard
       muting. Documented here so callers know we intentionally leave a floor
       for numerical stability. */
    double lower_clamped = clamp_unit_double(lower);
    double upper_clamped = clamp_unit_double(upper);
    if (upper_clamped < lower_clamped) {
        double tmp = lower_clamped;
        lower_clamped = upper_clamped;
        upper_clamped = tmp;
    }
    double intensity_clamped = clamp_unit_double(intensity);
    double inside_gain = intensity_clamped;
    if (inside_gain < 1e-6) {
        inside_gain = 1e-6;
    }
    double outside_gain = 1.0 - intensity_clamped;
    if (outside_gain < 1e-6) {
        outside_gain = 1e-6;
    }
    if (ratio >= lower_clamped && ratio <= upper_clamped) {
        return inside_gain;
    }
    return outside_gain;
}

static int solve_linear_system(double *matrix, double *rhs, int dim) {
    if (matrix == NULL || rhs == NULL || dim <= 0) {
        return -1;
    }
    for (int col = 0; col < dim; ++col) {
        int pivot = col;
        double max_val = fabs(matrix[col * dim + col]);
        for (int row = col + 1; row < dim; ++row) {
            double candidate = fabs(matrix[row * dim + col]);
            if (candidate > max_val) {
                max_val = candidate;
                pivot = row;
            }
        }
        if (max_val < 1e-18) {
            return -1;
        }
        if (pivot != col) {
            for (int k = col; k < dim; ++k) {
                double tmp = matrix[col * dim + k];
                matrix[col * dim + k] = matrix[pivot * dim + k];
                matrix[pivot * dim + k] = tmp;
            }
            double rhs_tmp = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = rhs_tmp;
        }
        double diag = matrix[col * dim + col];
        for (int row = col + 1; row < dim; ++row) {
            double factor = matrix[row * dim + col] / diag;
            matrix[row * dim + col] = 0.0;
            for (int k = col + 1; k < dim; ++k) {
                matrix[row * dim + k] -= factor * matrix[col * dim + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for (int row = dim - 1; row >= 0; --row) {
        double accum = rhs[row];
        for (int k = row + 1; k < dim; ++k) {
            accum -= matrix[row * dim + k] * rhs[k];
        }
        double diag = matrix[row * dim + row];
        if (fabs(diag) < 1e-18) {
            return -1;
        }
        rhs[row] = accum / diag;
    }
    return 0;
}

static size_t param_total_count(const EdgeRunnerParamView *view) {
    if (view == NULL) {
        return 0;
    }
    size_t batches = view->batches > 0U ? view->batches : 1U;
    size_t channels = view->channels > 0U ? view->channels : 1U;
    size_t frames = view->frames > 0U ? view->frames : 1U;
    return batches * channels * frames;
}

static double read_param_value(const EdgeRunnerParamView *view, size_t index, double default_value) {
    if (view == NULL || view->data == NULL) {
        return default_value;
    }
    size_t total = param_total_count(view);
    if (total == 0) {
        return default_value;
    }
    if (index >= total) {
        index = total - 1;
    }
    return view->data[index];
}

typedef struct {
    uint32_t band_count;
    double last_sum;
} fft_dynamic_carrier_summary_t;

static int parse_dynamic_carrier_index(const char *name, uint32_t *index_out) {
    if (name == NULL || index_out == NULL) {
        return 0;
    }
    const char *prefix = "carrier_band";
    size_t prefix_len = strlen(prefix);
    if (strncmp(name, prefix, prefix_len) != 0) {
        return 0;
    }
    const char *cursor = name + prefix_len;
    if (*cursor == '_' || *cursor == '-') {
        cursor++;
    }
    if (*cursor == '\0') {
        return 0;
    }
    for (const char *probe = cursor; *probe != '\0'; ++probe) {
        if (!isdigit((unsigned char)*probe)) {
            return 0;
        }
    }
    unsigned long parsed = strtoul(cursor, NULL, 10);
    if (parsed >= FFT_DYNAMIC_CARRIER_LIMIT) {
        return 0;
    }
    *index_out = (uint32_t)parsed;
    return 1;
}

static fft_dynamic_carrier_summary_t summarize_dynamic_carriers(const EdgeRunnerNodeInputs *inputs) {
    fft_dynamic_carrier_summary_t summary;
    summary.band_count = 0U;
    summary.last_sum = 0.0;
    if (inputs == NULL) {
        return summary;
    }
    uint32_t param_count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < param_count; ++i) {
        const EdgeRunnerParamView *view = &items[i];
        uint32_t index = 0U;
        if (!parse_dynamic_carrier_index(view->name, &index)) {
            continue;
        }
        if (index + 1U > summary.band_count) {
            summary.band_count = index + 1U;
        }
        size_t total = param_total_count(view);
        if (total > 0U) {
            double last_value = read_param_value(view, total - 1U, 0.0);
            summary.last_sum += last_value;
        }
    }
    return summary;
}

static uint32_t collect_dynamic_carrier_views(
    const EdgeRunnerNodeInputs *inputs,
    const EdgeRunnerParamView **views,
    uint32_t limit
) {
    if (views == NULL || limit == 0U) {
        return 0U;
    }
    for (uint32_t i = 0; i < limit; ++i) {
        views[i] = NULL;
    }
    if (inputs == NULL) {
        return 0U;
    }
    uint32_t max_index = 0U;
    uint32_t param_count = inputs->params.count;
    EdgeRunnerParamView *items = inputs->params.items;
    for (uint32_t i = 0; i < param_count; ++i) {
        EdgeRunnerParamView *view = &items[i];
        uint32_t index = 0U;
        if (!parse_dynamic_carrier_index(view->name, &index)) {
            continue;
        }
        if (index >= limit) {
            continue;
        }
        views[index] = view;
        if (index + 1U > max_index) {
            max_index = index + 1U;
        }
    }
    return max_index;
}

static int parse_algorithm_string(const char *json, size_t json_len, int default_value) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "algorithm", buffer, sizeof(buffer))) {
        return default_value;
    }
    for (size_t i = 0; buffer[i] != '\0'; ++i) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
    }
    if (
        strcmp(buffer, "fft") == 0
        || strcmp(buffer, "eigen") == 0
        || strcmp(buffer, "radix2") == 0
        || strcmp(buffer, "cooleytukey") == 0
        || strcmp(buffer, "nufft") == 0
        || strcmp(buffer, "nonuniform") == 0
        || strcmp(buffer, "czt") == 0
        || strcmp(buffer, "chirpz") == 0
        || strcmp(buffer, "chirpzt") == 0
    ) {
        return FFT_ALGORITHM_EIGEN;
    }
    if (strcmp(buffer, "hook") == 0 || strcmp(buffer, "custom_fft") == 0) {
        return FFT_ALGORITHM_HOOK;
    }
    if (strcmp(buffer, "dft") == 0 || strcmp(buffer, "direct") == 0 || strcmp(buffer, "slow") == 0) {
        return FFT_ALGORITHM_DFT;
    }
    if (
        strcmp(buffer, "dynamic") == 0
        || strcmp(buffer, "dynamic_oscillators") == 0
        || strcmp(buffer, "dynamicoscillators") == 0
    ) {
        return FFT_ALGORITHM_DYNAMIC_OSCILLATORS;
    }
    return default_value;
}

static int parse_window_string(const char *json, size_t json_len, int default_value) {
    char buffer[32];
    if (!json_copy_string(json, json_len, "window", buffer, sizeof(buffer))) {
        return default_value;
    }
    for (size_t i = 0; buffer[i] != '\0'; ++i) {
        buffer[i] = (char)tolower((unsigned char)buffer[i]);
    }
    if (strcmp(buffer, "rect") == 0 || strcmp(buffer, "rectangular") == 0) {
        return FFT_WINDOW_RECTANGULAR;
    }
    if (strcmp(buffer, "hann") == 0 || strcmp(buffer, "hanning") == 0) {
        return FFT_WINDOW_HANN;
    }
    if (strcmp(buffer, "hamming") == 0) {
        return FFT_WINDOW_HAMMING;
    }
    return default_value;
}

static int is_power_of_two_int(int value) {
    if (value <= 0) {
        return 0;
    }
    return (value & (value - 1)) == 0;
}

static void fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    amp_fft_backend_transform(in_real, in_imag, out_real, out_imag, n, inverse);
}

static void compute_dft(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    if (n <= 0 || out_real == NULL || out_imag == NULL) {
        return;
    }
    double sign = inverse != 0 ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (int t = 0; t < n; ++t) {
            double real = in_real != NULL ? in_real[t] : 0.0;
            double imag = in_imag != NULL ? in_imag[t] : 0.0;
            double angle = sign * 2.0 * M_PI * (double)k * (double)t / (double)n;
            double c = cos(angle);
            double s = sin(angle);
            sum_real += real * c - imag * s;
            sum_imag += real * s + imag * c;
        }
        if (inverse != 0) {
            sum_real /= (double)n;
            sum_imag /= (double)n;
        }
        out_real[k] = sum_real;
        out_imag[k] = sum_imag;
    }
}

typedef void (*fft_algorithm_impl_fn)(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
);

typedef struct {
    int kind;
    const char *label;
    fft_algorithm_impl_fn forward;
    fft_algorithm_impl_fn inverse;
    int requires_power_of_two;
    int supports_dynamic_carriers;
    int requires_hook;
} fft_algorithm_class_t;

static void fft_algorithm_backend_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_backend_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 1);
}

static void fft_algorithm_dft_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    compute_dft(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_dft_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    compute_dft(in_real, in_imag, out_real, out_imag, n, 1);
}

static void fft_algorithm_dynamic_forward(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 0);
}

static void fft_algorithm_dynamic_inverse(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n
) {
    fft_backend_transform(in_real, in_imag, out_real, out_imag, n, 1);
}

static const fft_algorithm_class_t FFT_ALGORITHM_CLASSES[] = {
    {
        FFT_ALGORITHM_EIGEN,
        "fft",
        fft_algorithm_backend_forward,
        fft_algorithm_backend_inverse,
        0,
        0,
        0,
    },
    {
        FFT_ALGORITHM_HOOK,
        "hook",
        fft_algorithm_backend_forward,
        fft_algorithm_backend_inverse,
        0,
        0,
        1,
    },
    {
        FFT_ALGORITHM_DFT,
        "dft",
        fft_algorithm_dft_forward,
        fft_algorithm_dft_inverse,
        0,
        0,
        0,
    },
    {
        FFT_ALGORITHM_DYNAMIC_OSCILLATORS,
        "dynamic_oscillators",
        fft_algorithm_dynamic_forward,
        fft_algorithm_dynamic_inverse,
        0,
        1,
        0,
    },
};

static const fft_algorithm_class_t *select_fft_algorithm(int kind) {
    size_t count = sizeof(FFT_ALGORITHM_CLASSES) / sizeof(FFT_ALGORITHM_CLASSES[0]);
    for (size_t i = 0; i < count; ++i) {
        if (FFT_ALGORITHM_CLASSES[i].kind == kind) {
            return &FFT_ALGORITHM_CLASSES[i];
        }
    }
    return NULL;
}

static void fft_algorithm_batched_forward(
    const fft_algorithm_class_t *cls,
    double *in_real,
    double *in_imag,
    int n,
    int slots
) {
    if (cls == NULL || cls->forward == NULL || n <= 0 || slots <= 0) {
        return;
    }
    /* If the selected algorithm is the backend (fftfree), prefer the
       batched backend transform which can accept STFT framing hints. This
       lets the runtime provide window/hop instructions to the backend
       without changing the per-frame algebra.
       Fall back to per-slot calls if the extended batched call fails. */
    if (cls->forward == fft_algorithm_backend_forward) {
        /* Request STFT framing from the backend: window = n, hop = 1 (sliding)
           so the backend will assemble overlapping frames the same way AMP
           previously did when it applied the analysis window itself. */
        int ok = amp_fft_backend_transform_many_ex(
            in_real,
            in_imag,
            in_real,
            in_imag,
            n,
            slots,
            0, /* forward */
            n, /* window */
            1, /* hop */
            1, /* stft_mode: enable */
            0, /* apply_windows */
            AMP_FFT_WINDOW_RECT,
            AMP_FFT_WINDOW_RECT
        );
        if (ok) {
            return;
        }
        /* fall through to per-slot if backend batched helper failed */
    }

    for (int s = 0; s < slots; ++s) {
        double *r = in_real + (size_t)s * (size_t)n;
        double *i = in_imag + (size_t)s * (size_t)n;
        cls->forward(r, i, r, i, n);
    }
}

static void fft_algorithm_batched_inverse(
    const fft_algorithm_class_t *cls,
    double *in_real,
    double *in_imag,
    int n,
    int slots
) {
    if (cls == NULL || cls->inverse == NULL || n <= 0 || slots <= 0) {
        return;
    }
    for (int s = 0; s < slots; ++s) {
        double *r = in_real + (size_t)s * (size_t)n;
        double *i = in_imag + (size_t)s * (size_t)n;
        cls->inverse(r, i, r, i, n);
    }
}

static void fftdiv_flush_batch(
    node_state_t *state,
    const fft_algorithm_class_t *algorithm_class,
    int window_size,
    int batch_count,
    double *output_buffer
) {
    if (state == NULL || algorithm_class == NULL || batch_count <= 0 || window_size <= 0 || output_buffer == NULL) {
        return;
    }
    double *sig_real = state->u.fftdiv.batch_input_real;
    double *sig_imag = state->u.fftdiv.batch_input_imag;
    double *div_real = state->u.fftdiv.batch_div_real;
    double *div_imag = state->u.fftdiv.batch_div_imag;
    int *slot_map = state->u.fftdiv.batch_slot_map;
    size_t *slot_offset_map = state->u.fftdiv.batch_slot_offset;
    size_t *output_index_map = state->u.fftdiv.batch_output_index;
    double *phase_map = state->u.fftdiv.batch_phase;
    double *lower_map = state->u.fftdiv.batch_lower;
    double *upper_map = state->u.fftdiv.batch_upper;
    double *filter_map = state->u.fftdiv.batch_filter;
    double *epsilon_map = state->u.fftdiv.batch_epsilon;
    if (
        sig_real == NULL || sig_imag == NULL || div_real == NULL || div_imag == NULL
        || slot_map == NULL || slot_offset_map == NULL || output_index_map == NULL
        || phase_map == NULL || lower_map == NULL || upper_map == NULL
        || filter_map == NULL || epsilon_map == NULL
    ) {
        return;
    }

    if (algorithm_class->forward == fft_algorithm_backend_forward) {
        if (state->u.fftdiv.backend_mode == 0) {
            fft_algorithm_batched_forward(algorithm_class, sig_real, sig_imag, window_size, batch_count);
            fft_algorithm_batched_forward(algorithm_class, div_real, div_imag, window_size, batch_count);
        } else if (state->u.fftdiv.backend_mode == 1) {
            int ok_sig = amp_fft_backend_transform_many_ex(
                sig_real,
                sig_imag,
                sig_real,
                sig_imag,
                window_size,
                batch_count,
                0,
                window_size,
                state->u.fftdiv.backend_hop,
                1,
                1,
                state->u.fftdiv.window_kind,
                state->u.fftdiv.window_kind
            );
            if (!ok_sig) {
                fft_algorithm_batched_forward(algorithm_class, sig_real, sig_imag, window_size, batch_count);
            }
            int ok_div = amp_fft_backend_transform_many_ex(
                div_real,
                div_imag,
                div_real,
                div_imag,
                window_size,
                batch_count,
                0,
                window_size,
                state->u.fftdiv.backend_hop,
                1,
                1,
                state->u.fftdiv.window_kind,
                state->u.fftdiv.window_kind
            );
            if (!ok_div) {
                fft_algorithm_batched_forward(algorithm_class, div_real, div_imag, window_size, batch_count);
            }
        } else {
            /* Streaming mode already wrote spectra into the batch buffers. */
        }
    } else {
        fft_algorithm_batched_forward(algorithm_class, sig_real, sig_imag, window_size, batch_count);
        fft_algorithm_batched_forward(algorithm_class, div_real, div_imag, window_size, batch_count);
    }
    /* Diagnostic dump for batches: print mapping and batched arrays when window_size is small
       This helps compare batched inputs/divisors against the simulator. */
    if (window_size > 0 && window_size <= 16) {
        fprintf(stderr, "[diag] fftdiv_flush_batch: window_size=%d batch_count=%d\n", window_size, batch_count);
        for (int b = 0; b < batch_count; ++b) {
            size_t batch_offset = (size_t)b * (size_t)window_size;
            fprintf(stderr, "[diag] batch %d: slot_map=%d slot_offset=%zu output_index=%zu phase=% .9g lower=% .9g upper=% .9g filter=% .9g epsilon=% .9g\n",
                b, slot_map[b], slot_offset_map[b], output_index_map[b], phase_map[b], lower_map[b], upper_map[b], filter_map[b], epsilon_map[b]);
            fprintf(stderr, "[diag]  time sig_real: ");
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g", sig_real[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[diag]  time sig_imag: ");
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g", sig_imag[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[diag]  time div_real: ");
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g", div_real[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[diag]  time div_imag: ");
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g", div_imag[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
        }
    }

    /* After forward transforms, print the frequency-domain bins for sig and divisor. */
    if (window_size > 0 && window_size <= 16) {
        for (int b = 0; b < batch_count; ++b) {
            size_t batch_offset = (size_t)b * (size_t)window_size;
            fprintf(stderr, "[diag] freq sig (batch %d): ", b);
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g+% .9gi", sig_real[batch_offset + (size_t)i], sig_imag[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[diag] freq div (batch %d): ", b);
            for (int i = 0; i < window_size; ++i) {
                fprintf(stderr, " % .9g+% .9gi", div_real[batch_offset + (size_t)i], div_imag[batch_offset + (size_t)i]);
            }
            fprintf(stderr, "\n");
        }
    }

    for (int b = 0; b < batch_count; ++b) {
        size_t batch_offset = (size_t)b * (size_t)window_size;
        double *sig_real_base = sig_real + batch_offset;
        double *sig_imag_base = sig_imag + batch_offset;
        double *div_real_base = div_real + batch_offset;
        double *div_imag_base = div_imag + batch_offset;
        size_t slot_offset = slot_offset_map[b];
        double phase_mod = phase_map[b];
        double lower_mod = lower_map[b];
        double upper_mod = upper_map[b];
        double filter_mod = filter_map[b];
        double epsilon_frame = epsilon_map[b];
        double lower_clamped = clamp_unit_double(lower_mod);
        double upper_clamped = clamp_unit_double(upper_mod);
        if (upper_clamped < lower_clamped) {
            double tmp_bounds = lower_clamped;
            lower_clamped = upper_clamped;
            upper_clamped = tmp_bounds;
        }
        double intensity_clamped = clamp_unit_double(filter_mod);
        double cos_phase = cos(phase_mod);
        double sin_phase = sin(phase_mod);
        for (int i = 0; i < window_size; ++i) {
            double a = sig_real_base[i];
            double b_im = sig_imag_base[i];
            double c = div_real_base[i];
            double d = div_imag_base[i];
            state->u.fftdiv.div_fft_real[slot_offset + (size_t)i] = c;
            state->u.fftdiv.div_fft_imag[slot_offset + (size_t)i] = d;
            double denom = c * c + d * d;
            if (denom < epsilon_frame) {
                denom = epsilon_frame;
            }
            double real = (a * c + b_im * d) / denom;
            double imag = (b_im * c - a * d) / denom;
            double ratio = window_size > 1 ? (double)i / (double)(window_size - 1) : 0.0;
            double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
            double gated_real = real * gain;
            double gated_imag = imag * gain;
            double rotated_real = gated_real * cos_phase - gated_imag * sin_phase;
            double rotated_imag = gated_real * sin_phase + gated_imag * cos_phase;
            sig_real_base[i] = rotated_real;
            sig_imag_base[i] = rotated_imag;
        }
    }

    fft_algorithm_batched_inverse(algorithm_class, sig_real, sig_imag, window_size, batch_count);

    for (int b = 0; b < batch_count; ++b) {
        size_t slot_offset = slot_offset_map[b];
        size_t batch_offset = (size_t)b * (size_t)window_size;
        double *sig_real_base = sig_real + batch_offset;
        for (int i = 0; i < window_size; ++i) {
            state->u.fftdiv.result_buffer[slot_offset + (size_t)i] = sig_real_base[i];
        }
        size_t output_index = output_index_map[b];
        output_buffer[output_index] = sig_real_base[window_size > 0 ? window_size - 1 : 0];
        state->u.fftdiv.batch_slot_map[b] = slot_map[b];
    }
}

static void fill_window_weights(double *window, int window_size, int window_kind) {
    if (window == NULL || window_size <= 0) {
        return;
    }
    if (window_size == 1) {
        window[0] = 1.0;
        return;
    }
    for (int i = 0; i < window_size; ++i) {
        double value = 1.0;
        double phase = (double)i / (double)(window_size - 1);
        switch (window_kind) {
            case FFT_WINDOW_RECTANGULAR:
                value = 1.0;
                break;
            case FFT_WINDOW_HANN:
                value = 0.5 * (1.0 - cos(2.0 * M_PI * phase));
                break;
            case FFT_WINDOW_HAMMING:
                value = 0.54 - 0.46 * cos(2.0 * M_PI * phase);
                break;
            default:
                value = 1.0;
                break;
        }
        window[i] = value;
    }
}

static void fftdiv_destroy_stream_handles(node_state_t *state) {
    if (state == NULL) {
        return;
    }
    const int slot_count = state->u.fftdiv.slots > 0 ? state->u.fftdiv.slots : state->u.fftdiv.batch_capacity;
    if (state->u.fftdiv.backend_stream_signal != NULL) {
        for (int i = 0; i < slot_count; ++i) {
            if (state->u.fftdiv.backend_stream_signal[i] != NULL) {
                amp_fft_backend_stream_destroy(state->u.fftdiv.backend_stream_signal[i]);
                state->u.fftdiv.backend_stream_signal[i] = NULL;
            }
        }
    }
    if (state->u.fftdiv.backend_stream_div_real != NULL) {
        for (int i = 0; i < slot_count; ++i) {
            if (state->u.fftdiv.backend_stream_div_real[i] != NULL) {
                amp_fft_backend_stream_destroy(state->u.fftdiv.backend_stream_div_real[i]);
                state->u.fftdiv.backend_stream_div_real[i] = NULL;
            }
        }
    }
    if (state->u.fftdiv.backend_stream_div_imag != NULL) {
        for (int i = 0; i < slot_count; ++i) {
            if (state->u.fftdiv.backend_stream_div_imag[i] != NULL) {
                amp_fft_backend_stream_destroy(state->u.fftdiv.backend_stream_div_imag[i]);
                state->u.fftdiv.backend_stream_div_imag[i] = NULL;
            }
        }
    }
}

static void *fftdiv_ensure_stream_handle(
    node_state_t *state,
    void **table,
    int slot,
    int window_size,
    int hop,
    int window_kind
) {
    if (state == NULL || table == NULL || slot < 0) {
        return NULL;
    }
    if (slot >= state->u.fftdiv.slots) {
        return NULL;
    }
    if (table[slot] != NULL) {
        return table[slot];
    }
    void *handle = amp_fft_backend_stream_create(window_size, window_size, hop, window_kind);
    table[slot] = handle;
    return handle;
}

static void fft_state_free_buffers(node_state_t *state) {
    if (state == NULL) {
        return;
    }
    fftdiv_destroy_stream_handles(state);
    free(state->u.fftdiv.input_buffer);
    free(state->u.fftdiv.divisor_buffer);
    free(state->u.fftdiv.divisor_imag_buffer);
    free(state->u.fftdiv.phase_buffer);
    free(state->u.fftdiv.lower_buffer);
    free(state->u.fftdiv.upper_buffer);
    free(state->u.fftdiv.filter_buffer);
    free(state->u.fftdiv.window);
    free(state->u.fftdiv.work_real);
    free(state->u.fftdiv.work_imag);
    free(state->u.fftdiv.div_real);
    free(state->u.fftdiv.div_imag);
    free(state->u.fftdiv.ifft_real);
    free(state->u.fftdiv.ifft_imag);
    free(state->u.fftdiv.result_buffer);
    free(state->u.fftdiv.div_fft_real);
    free(state->u.fftdiv.div_fft_imag);
    free(state->u.fftdiv.recomb_buffer);
    free(state->u.fftdiv.batch_input_real);
    free(state->u.fftdiv.batch_input_imag);
    free(state->u.fftdiv.batch_div_real);
    free(state->u.fftdiv.batch_div_imag);
    free(state->u.fftdiv.batch_slot_map);
    free(state->u.fftdiv.batch_slot_offset);
    free(state->u.fftdiv.batch_output_index);
    free(state->u.fftdiv.batch_phase);
    free(state->u.fftdiv.batch_lower);
    free(state->u.fftdiv.batch_upper);
    free(state->u.fftdiv.batch_filter);
    free(state->u.fftdiv.batch_epsilon);
    free(state->u.fftdiv.dynamic_phase);
    free(state->u.fftdiv.dynamic_step_re);
    free(state->u.fftdiv.dynamic_step_im);
    free(state->u.fftdiv.backend_stream_signal);
    free(state->u.fftdiv.backend_stream_div_real);
    free(state->u.fftdiv.backend_stream_div_imag);
    state->u.fftdiv.input_buffer = NULL;
    state->u.fftdiv.divisor_buffer = NULL;
    state->u.fftdiv.divisor_imag_buffer = NULL;
    state->u.fftdiv.phase_buffer = NULL;
    state->u.fftdiv.lower_buffer = NULL;
    state->u.fftdiv.upper_buffer = NULL;
    state->u.fftdiv.filter_buffer = NULL;
    state->u.fftdiv.window = NULL;
    state->u.fftdiv.work_real = NULL;
    state->u.fftdiv.work_imag = NULL;
    state->u.fftdiv.div_real = NULL;
    state->u.fftdiv.div_imag = NULL;
    state->u.fftdiv.ifft_real = NULL;
    state->u.fftdiv.ifft_imag = NULL;
    state->u.fftdiv.result_buffer = NULL;
    state->u.fftdiv.div_fft_real = NULL;
    state->u.fftdiv.div_fft_imag = NULL;
    state->u.fftdiv.recomb_buffer = NULL;
    state->u.fftdiv.batch_input_real = NULL;
    state->u.fftdiv.batch_input_imag = NULL;
    state->u.fftdiv.batch_div_real = NULL;
    state->u.fftdiv.batch_div_imag = NULL;
    state->u.fftdiv.batch_slot_map = NULL;
    state->u.fftdiv.batch_slot_offset = NULL;
    state->u.fftdiv.batch_output_index = NULL;
    state->u.fftdiv.batch_phase = NULL;
    state->u.fftdiv.batch_lower = NULL;
    state->u.fftdiv.batch_upper = NULL;
    state->u.fftdiv.batch_filter = NULL;
    state->u.fftdiv.batch_epsilon = NULL;
    state->u.fftdiv.dynamic_phase = NULL;
    state->u.fftdiv.dynamic_step_re = NULL;
    state->u.fftdiv.dynamic_step_im = NULL;
    state->u.fftdiv.backend_stream_signal = NULL;
    state->u.fftdiv.backend_stream_div_real = NULL;
    state->u.fftdiv.backend_stream_div_imag = NULL;
    state->u.fftdiv.window_size = 0;
    state->u.fftdiv.algorithm = FFT_ALGORITHM_EIGEN;
    state->u.fftdiv.window_kind = -1;
    state->u.fftdiv.filled = 0;
    state->u.fftdiv.position = 0;
    state->u.fftdiv.batches = 0;
    state->u.fftdiv.channels = 0;
    state->u.fftdiv.slots = 0;
    state->u.fftdiv.epsilon = 0.0;
    state->u.fftdiv.recomb_filled = 0;
    state->u.fftdiv.last_phase = 0.0;
    state->u.fftdiv.last_lower = 0.0;
    state->u.fftdiv.last_upper = 1.0;
    state->u.fftdiv.last_filter = 1.0;
    state->u.fftdiv.dynamic_carrier_band_count = 0;
    state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
    state->u.fftdiv.dynamic_k_active = 0;
    state->u.fftdiv.enable_remainder = 1;
    state->u.fftdiv.remainder_energy = 0.0;
    state->u.fftdiv.max_batch_windows = 0;
    state->u.fftdiv.batch_capacity = 0;
    state->u.fftdiv.backend_mode = 0;
    state->u.fftdiv.backend_hop = 1;
    state->u.fftdiv.backend_stream_window_size = 0;
    state->u.fftdiv.backend_stream_window_kind = -1;
    state->u.fftdiv.backend_stream_hop = 1;
}

static int ensure_fft_state_buffers(node_state_t *state, int slots, int window_size, int max_batch_windows) {
    if (state == NULL) {
        return -1;
    }
    if (slots <= 0) {
        slots = 1;
    }
    if (window_size <= 0) {
        window_size = 1;
    }
    if (max_batch_windows <= 0) {
        max_batch_windows = 1;
    }
    if (
        state->u.fftdiv.input_buffer != NULL
        && state->u.fftdiv.window_size == window_size
        && state->u.fftdiv.slots == slots
        && state->u.fftdiv.max_batch_windows == max_batch_windows
    ) {
        return 0;
    }
    double preserved_heat = state->u.fftdiv.total_heat;
    fft_state_free_buffers(state);
    size_t total = (size_t)window_size * (size_t)slots;
    if (total == 0) {
        state->u.fftdiv.total_heat = preserved_heat;
        return -1;
    }
    size_t batch_capacity = (size_t)max_batch_windows;
    size_t batch_total = (size_t)window_size * batch_capacity;
    state->u.fftdiv.input_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.divisor_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.divisor_imag_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.phase_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.lower_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.upper_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.filter_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.window = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.work_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.work_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.div_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.div_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.ifft_real = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.ifft_imag = (double *)calloc((size_t)window_size, sizeof(double));
    state->u.fftdiv.result_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.div_fft_real = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.div_fft_imag = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.recomb_buffer = (double *)calloc(total, sizeof(double));
    state->u.fftdiv.batch_input_real = (double *)calloc(batch_total, sizeof(double));
    state->u.fftdiv.batch_input_imag = (double *)calloc(batch_total, sizeof(double));
    state->u.fftdiv.batch_div_real = (double *)calloc(batch_total, sizeof(double));
    state->u.fftdiv.batch_div_imag = (double *)calloc(batch_total, sizeof(double));
    state->u.fftdiv.batch_slot_map = (int *)calloc(batch_capacity, sizeof(int));
    state->u.fftdiv.batch_slot_offset = (size_t *)calloc(batch_capacity, sizeof(size_t));
    state->u.fftdiv.batch_output_index = (size_t *)calloc(batch_capacity, sizeof(size_t));
    state->u.fftdiv.batch_phase = (double *)calloc(batch_capacity, sizeof(double));
    state->u.fftdiv.batch_lower = (double *)calloc(batch_capacity, sizeof(double));
    state->u.fftdiv.batch_upper = (double *)calloc(batch_capacity, sizeof(double));
    state->u.fftdiv.batch_filter = (double *)calloc(batch_capacity, sizeof(double));
    state->u.fftdiv.batch_epsilon = (double *)calloc(batch_capacity, sizeof(double));
    state->u.fftdiv.dynamic_phase = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    state->u.fftdiv.dynamic_step_re = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    state->u.fftdiv.dynamic_step_im = (double *)calloc((size_t)slots * FFT_DYNAMIC_CARRIER_LIMIT, sizeof(double));
    state->u.fftdiv.backend_stream_signal = (void **)calloc((size_t)slots, sizeof(void *));
    state->u.fftdiv.backend_stream_div_real = (void **)calloc((size_t)slots, sizeof(void *));
    state->u.fftdiv.backend_stream_div_imag = (void **)calloc((size_t)slots, sizeof(void *));
    if (state->u.fftdiv.input_buffer == NULL || state->u.fftdiv.divisor_buffer == NULL || state->u.fftdiv.divisor_imag_buffer == NULL
        || state->u.fftdiv.phase_buffer == NULL || state->u.fftdiv.lower_buffer == NULL || state->u.fftdiv.upper_buffer == NULL
        || state->u.fftdiv.filter_buffer == NULL || state->u.fftdiv.window == NULL || state->u.fftdiv.work_real == NULL
        || state->u.fftdiv.work_imag == NULL || state->u.fftdiv.div_real == NULL || state->u.fftdiv.div_imag == NULL
        || state->u.fftdiv.ifft_real == NULL || state->u.fftdiv.ifft_imag == NULL || state->u.fftdiv.result_buffer == NULL
        || state->u.fftdiv.div_fft_real == NULL || state->u.fftdiv.div_fft_imag == NULL || state->u.fftdiv.recomb_buffer == NULL
        || state->u.fftdiv.batch_input_real == NULL || state->u.fftdiv.batch_input_imag == NULL
        || state->u.fftdiv.batch_div_real == NULL || state->u.fftdiv.batch_div_imag == NULL
        || state->u.fftdiv.batch_slot_map == NULL || state->u.fftdiv.batch_slot_offset == NULL
        || state->u.fftdiv.batch_output_index == NULL || state->u.fftdiv.batch_phase == NULL
        || state->u.fftdiv.batch_lower == NULL || state->u.fftdiv.batch_upper == NULL
        || state->u.fftdiv.batch_filter == NULL || state->u.fftdiv.batch_epsilon == NULL
        || state->u.fftdiv.dynamic_phase == NULL || state->u.fftdiv.dynamic_step_re == NULL || state->u.fftdiv.dynamic_step_im == NULL
        || state->u.fftdiv.backend_stream_signal == NULL || state->u.fftdiv.backend_stream_div_real == NULL
        || state->u.fftdiv.backend_stream_div_imag == NULL) {
        fft_state_free_buffers(state);
        state->u.fftdiv.total_heat = preserved_heat;
        return -1;
    }
    state->u.fftdiv.window_size = window_size;
    state->u.fftdiv.slots = slots;
    state->u.fftdiv.max_batch_windows = max_batch_windows;
    state->u.fftdiv.batch_capacity = (int)batch_capacity;
    state->u.fftdiv.filled = 0;
    state->u.fftdiv.position = window_size > 0 ? window_size - 1 : 0;
    state->u.fftdiv.algorithm = FFT_ALGORITHM_EIGEN;
    state->u.fftdiv.window_kind = -1;
    state->u.fftdiv.batches = 0;
    state->u.fftdiv.channels = 0;
    state->u.fftdiv.epsilon = 1e-9;
    state->u.fftdiv.recomb_filled = 0;
    state->u.fftdiv.last_phase = 0.0;
    state->u.fftdiv.last_lower = 0.0;
    state->u.fftdiv.last_upper = 1.0;
    state->u.fftdiv.last_filter = 1.0;
    state->u.fftdiv.total_heat = preserved_heat;
    state->u.fftdiv.dynamic_carrier_band_count = 0;
    state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
    state->u.fftdiv.dynamic_k_active = 0;
    state->u.fftdiv.enable_remainder = 1;
    state->u.fftdiv.remainder_energy = 0.0;
    state->u.fftdiv.backend_mode = 0;
    state->u.fftdiv.backend_hop = 1;
    state->u.fftdiv.backend_stream_window_size = window_size;
    state->u.fftdiv.backend_stream_window_kind = -1;
    state->u.fftdiv.backend_stream_hop = 1;
    return 0;
}

#include "nodes/constant/constant_node.inc"
#include "nodes/controller/controller_node.inc"
#include "nodes/lfo/lfo_node.inc"
#include "nodes/envelope/envelope_node.inc"
#include "nodes/pitch/pitch_node.inc"
#include "nodes/oscillator_pitch/oscillator_pitch_node.inc"
#include "nodes/subharmonic/subharmonic_node.inc"
#include "nodes/oscillator/oscillator_node.inc"
#include "nodes/resampler/resampler_node.inc"
#include "nodes/parametric_driver/parametric_driver_node.inc"
#include "nodes/pitch_shift/pitch_shift_node.inc"
#include "nodes/gain/gain_node.inc"
#include "nodes/fft_division/fft_division_nodes.inc"
#include "nodes/mix/mix_node.inc"
#include "nodes/sine_osc/sine_osc_node.inc"
#include "nodes/safety/safety_node.inc"
static void amp_reset_metrics(AmpNodeMetrics *metrics) {
    if (metrics == NULL) {
        return;
    }
    metrics->measured_delay_frames = 0U;
    metrics->accumulated_heat = 0.0f;
    metrics->processing_time_seconds = 0.0;
    metrics->logging_time_seconds = 0.0;
    metrics->total_time_seconds = 0.0;
    metrics->thread_cpu_time_seconds = 0.0;
    for (size_t i = 0; i < sizeof(metrics->reserved) / sizeof(metrics->reserved[0]); ++i) {
        metrics->reserved[i] = 0.0;
    }
}

/* Thread-local accumulators used to separate time spent in logging helpers
   from the node processing time. We use clock() to measure CPU time which is
   sufficient for profiling relative contributions of logging vs processing. */
#if defined(_MSC_VER)
__declspec(thread) static double _tl_thread_cpu_start = 0.0;
#else
static __thread double _tl_thread_cpu_start = 0.0;
#endif

static inline double _now_thread_cpu_seconds(void) {
#if defined(_WIN32) || defined(_WIN64)
    FILETIME creation_time, exit_time, kernel_time, user_time;
    HANDLE thread = GetCurrentThread();
    if (GetThreadTimes(thread, &creation_time, &exit_time, &kernel_time, &user_time)) {
        ULARGE_INTEGER kernel_ticks;
        ULARGE_INTEGER user_ticks;
        kernel_ticks.LowPart = kernel_time.dwLowDateTime;
        kernel_ticks.HighPart = kernel_time.dwHighDateTime;
        user_ticks.LowPart = user_time.dwLowDateTime;
        user_ticks.HighPart = user_time.dwHighDateTime;
        unsigned long long total_ticks = kernel_ticks.QuadPart + user_ticks.QuadPart;
        return (double)total_ticks * 1.0e-7; /* FILETIME is 100-ns units */
    }
    return 0.0;
#else
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    clock_t ticks = clock();
    if (ticks == (clock_t)-1) {
        return 0.0;
    }
    return (double)ticks / (double)CLOCKS_PER_SEC;
#endif
}

typedef struct {
    double total_seconds;
    double processing_seconds;
    double logging_seconds;
    double thread_cpu_seconds;
} node_timing_info;

static double _node_timing_begin(const char *node_name) {
    amp_debug_logging_accum = 0.0;
    _tl_thread_cpu_start = _now_thread_cpu_seconds();
    amp_debug_current_node = node_name;
    return amp_debug_now_seconds();
}

static node_timing_info _node_timing_end(double start_clock) {
    node_timing_info info;
    double end_clock = amp_debug_now_seconds();
    info.total_seconds = end_clock - start_clock;
    if (info.total_seconds < 0.0) {
        info.total_seconds = 0.0;
    }
    double logging = amp_debug_logging_accum;
    if (logging < 0.0) logging = 0.0;
    info.logging_seconds = logging;
    info.processing_seconds = info.total_seconds - logging;
    if (info.processing_seconds < 0.0) {
        info.processing_seconds = 0.0;
    }
    double cpu_elapsed = _now_thread_cpu_seconds() - _tl_thread_cpu_start;
    if (cpu_elapsed < 0.0) {
        cpu_elapsed = 0.0;
    }
    info.thread_cpu_seconds = cpu_elapsed;
    amp_debug_current_node = NULL;
    amp_debug_logging_accum = 0.0;
    _tl_thread_cpu_start = 0.0;
    return info;
}

static const char *_node_dump_root_dir(void) {
    static int initialised = 0;
    static char root_dir[1024];
    if (!initialised) {
        const char *env = getenv("AMP_NODE_DUMP_DIR");
        if (env != NULL && env[0] != '\0') {
            size_t len = strlen(env);
            if (len >= sizeof(root_dir)) {
                len = sizeof(root_dir) - 1;
            }
            memcpy(root_dir, env, len);
            root_dir[len] = '\0';
        } else {
            root_dir[0] = '\0';
        }
        initialised = 1;
    }
    if (root_dir[0] == '\0') {
        return NULL;
    }
    return root_dir;
}

static void _sanitize_node_name(const char *name, char *out, size_t out_len) {
    if (out_len == 0) {
        return;
    }
    const char *src = (name != NULL && name[0] != '\0') ? name : "unnamed";
    size_t written = 0;
    for (; src[0] != '\0' && written + 1 < out_len; ++src) {
        unsigned char ch = (unsigned char)*src;
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out[written++] = (char)ch;
        } else if (ch == '-' || ch == '_') {
            out[written++] = (char)ch;
        } else {
            out[written++] = '_';
        }
    }
    if (written == 0) {
        out[written++] = 'n';
        if (written + 1 < out_len) {
            out[written++] = 'o';
        }
        if (written + 1 < out_len) {
            out[written++] = 'd';
        }
    }
    out[written] = '\0';
}

static void _write_json_string(FILE *stream, const char *text) {
    if (stream == NULL) {
        return;
    }
    if (text == NULL) {
        fputs("null", stream);
        return;
    }
    fputc('"', stream);
    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        unsigned char ch = *p;
        switch (ch) {
            case '\\':
                fputs("\\\\", stream);
                break;
            case '"':
                fputs("\\\"", stream);
                break;
            case '\n':
                fputs("\\n", stream);
                break;
            case '\r':
                fputs("\\r", stream);
                break;
            case '\t':
                fputs("\\t", stream);
                break;
            default:
                if (ch < 0x20U) {
                    fprintf(stream, "\\u%04x", ch);
                } else {
                    fputc((int)ch, stream);
                }
                break;
        }
    }
    fputc('"', stream);
}

#if defined(_WIN32) || defined(_WIN64)
static volatile LONG64 _node_dump_sequence_counter = 0;
#else
static volatile uint64_t _node_dump_sequence_counter = 0;
#endif

static uint64_t _next_dump_sequence(void) {
#if defined(_WIN32) || defined(_WIN64)
    LONG64 value = InterlockedIncrement64(&_node_dump_sequence_counter);
    if (value <= 0) {
        return 0ULL;
    }
    return (uint64_t)(value - 1);
#else
    return __sync_fetch_and_add(&_node_dump_sequence_counter, 1ULL);
#endif
}

static void maybe_dump_node_output(
    const char *node_name,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    const double *buffer,
    const AmpNodeMetrics *metrics,
    node_timing_info timing
) {
    (void)metrics;
    if (buffer == NULL) {
        return;
    }
    if (batches <= 0 || channels <= 0 || frames <= 0) {
        return;
    }
    const char *root = _node_dump_root_dir();
    if (root == NULL) {
        return;
    }
    char safe_name[128];
    _sanitize_node_name(node_name, safe_name, sizeof(safe_name));
    char node_dir[1024];
    if (snprintf(node_dir, sizeof(node_dir), "%s/%s", root, safe_name) >= (int)sizeof(node_dir)) {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (!CreateDirectoryA(node_dir, NULL)) {
        DWORD err = GetLastError();
        if (err != ERROR_ALREADY_EXISTS) {
            return;
        }
    }
#else
    if (mkdir(node_dir, 0775) != 0 && errno != EEXIST) {
        return;
    }
#endif
    uint64_t sequence = _next_dump_sequence();
    char base_path[1024];
    if (snprintf(base_path, sizeof(base_path), "%s/%s/%s_%06llu", root, safe_name, safe_name, (unsigned long long)sequence)
        >= (int)sizeof(base_path)) {
        return;
    }
    char raw_path[1024];
    if (snprintf(raw_path, sizeof(raw_path), "%s.raw", base_path) >= (int)sizeof(raw_path)) {
        return;
    }
    char meta_path[1024];
    if (snprintf(meta_path, sizeof(meta_path), "%s.meta.json", base_path) >= (int)sizeof(meta_path)) {
        return;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    float *temp = (float *)malloc(total * sizeof(float));
    if (temp == NULL) {
        return;
    }
    for (size_t i = 0; i < total; ++i) {
        temp[i] = (float)buffer[i];
    }
    FILE *raw_file = fopen(raw_path, "wb");
    if (raw_file == NULL) {
        free(temp);
        return;
    }
    size_t written = fwrite(temp, sizeof(float), total, raw_file);
    fclose(raw_file);
    free(temp);
    if (written != total) {
        remove(raw_path);
        return;
    }
    FILE *meta_file = fopen(meta_path, "w");
    if (meta_file == NULL) {
        return;
    }
    fprintf(meta_file, "{\n");
    fprintf(meta_file, "  \"node_name\": ");
    _write_json_string(meta_file, node_name);
    fprintf(meta_file, ",\n  \"safe_node_name\": ");
    _write_json_string(meta_file, safe_name);
    fprintf(meta_file, ",\n  \"sequence\": %llu,\n", (unsigned long long)sequence);
    fprintf(meta_file, "  \"batches\": %d,\n", batches);
    fprintf(meta_file, "  \"channels\": %d,\n", channels);
    fprintf(meta_file, "  \"frames\": %d,\n", frames);
    fprintf(meta_file, "  \"sample_rate\": %.9f,\n", sample_rate);
    fprintf(meta_file, "  \"total_time_seconds\": %.12g,\n", timing.total_seconds);
    fprintf(meta_file, "  \"processing_time_seconds\": %.12g,\n", timing.processing_seconds);
    fprintf(meta_file, "  \"logging_time_seconds\": %.12g,\n", timing.logging_seconds);
    fprintf(meta_file, "  \"thread_cpu_time_seconds\": %.12g,\n", timing.thread_cpu_seconds);
    fprintf(meta_file, "  \"dtype\": \"float32\",\n");
    fprintf(meta_file, "  \"layout\": \"BCF\"\n");
    fprintf(meta_file, "}\n");
    fclose(meta_file);
}

static int amp_run_node_impl(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
) {
    amp_reset_metrics(metrics);
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
    /* Begin per-node timing window: set current node context and mark start. */
    double _node_start_clock = _node_timing_begin(descriptor != NULL ? descriptor->name : NULL);
    switch (kind) {
        case NODE_KIND_CONSTANT:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_constant_node(descriptor, batches, frames, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_GAIN:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_gain_node(inputs, batches, frames, out_buffer, out_channels)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_MIX:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_mix_node(descriptor, inputs, batches, frames, out_buffer, out_channels)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SAFETY:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_safety_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SINE_OSC:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_sine_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_CONTROLLER:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_controller_node(descriptor, inputs, batches, frames, out_buffer, out_channels, history)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_LFO:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_lfo_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_ENVELOPE:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_envelope_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_PITCH:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_pitch_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_PITCH_SHIFT:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_pitch_shift_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_OSC_PITCH:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_oscillator_pitch_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_OSC:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_osc_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_DRIVER:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_parametric_driver_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_RESAMPLER:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_resampler_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state, metrics)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_SUBHARM:
            rc = (mode == AMP_EXECUTION_MODE_FORWARD)
                ? run_subharm_node(descriptor, inputs, batches, frames, sample_rate, out_buffer, out_channels, node_state)
                : AMP_E_UNSUPPORTED;
            break;
        case NODE_KIND_FFT_DIV:
            if (mode == AMP_EXECUTION_MODE_BACKWARD) {
                rc = run_fft_division_node_backward(
                    descriptor,
                    inputs,
                    batches,
                    channels,
                    frames,
                    sample_rate,
                    out_buffer,
                    out_channels,
                    node_state,
                    metrics
                );
            } else {
                rc = run_fft_division_node(
                    descriptor,
                    inputs,
                    batches,
                    channels,
                    frames,
                    sample_rate,
                    out_buffer,
                    out_channels,
                    node_state,
                    metrics
                );
            }
            break;
        case NODE_KIND_SPECTRAL_DRIVE:
            rc = AMP_E_UNSUPPORTED;
            break;
        default:
            rc = -3;
            break;
    }
    /* End timing window and attribute timing to metrics if requested. */
    node_timing_info _timing = _node_timing_end(_node_start_clock);
    if (metrics != NULL) {
        if (rc == 0 && kind != NODE_KIND_FFT_DIV) {
            metrics->measured_delay_frames = 0U;
            metrics->accumulated_heat = 0.0f;
        }
        metrics->processing_time_seconds = _timing.processing_seconds;
        metrics->logging_time_seconds = _timing.logging_seconds;
        metrics->total_time_seconds = _timing.total_seconds;
        metrics->thread_cpu_time_seconds = _timing.thread_cpu_seconds;
    }
    maybe_dump_node_output(
        descriptor != NULL ? descriptor->name : NULL,
        batches,
        (out_channels != NULL) ? *out_channels : 0,
        frames,
        sample_rate,
        (out_buffer != NULL && *out_buffer != NULL) ? *out_buffer : NULL,
        metrics,
        _timing
    );
    return rc;
}

AMP_CAPI int amp_run_node(
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
    AMP_LOG_NATIVE_CALL("amp_run_node", (size_t)batches, (size_t)frames);
    return amp_run_node_impl(
        descriptor,
        inputs,
        batches,
        channels,
        frames,
        sample_rate,
        out_buffer,
        out_channels,
        state,
        history,
        AMP_EXECUTION_MODE_FORWARD,
        NULL
    );
}

AMP_CAPI int amp_run_node_v2(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
) {
    AMP_LOG_NATIVE_CALL("amp_run_node_v2", (size_t)batches, (size_t)frames);
    return amp_run_node_impl(
        descriptor,
        inputs,
        batches,
        channels,
        frames,
        sample_rate,
        out_buffer,
        out_channels,
        state,
        history,
        mode,
        metrics
    );
}

AMP_CAPI void amp_free(double *buffer) {
    AMP_LOG_NATIVE_CALL("amp_free", (size_t)(buffer != NULL), 0);
    AMP_LOG_GENERATED("amp_free", (size_t)buffer, 0);
    if (buffer != NULL) {
        free(buffer);
    }
}

AMP_CAPI void amp_release_state(void *state_ptr) {
    AMP_LOG_NATIVE_CALL("amp_release_state", (size_t)(state_ptr != NULL), 0);
    AMP_LOG_GENERATED("amp_release_state", (size_t)state_ptr, 0);
    if (state_ptr == NULL) {
        return;
    }
    node_state_t *node_state = (node_state_t *)state_ptr;
    release_node_state(node_state);
}
