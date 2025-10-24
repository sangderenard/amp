#ifndef AMP_NATIVE_H
#define AMP_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(_WIN64)
#  define AMP_CAPI __declspec(dllexport)
#else
#  define AMP_CAPI
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

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

/*
 * Node/oscillator contract (notes)
 * --------------------------------
 * The runtime and native node implementations (oscillators, gains, drivers)
 * agree on the following expectations. These are documentation-only clarifications
 * to make the API contract explicit for implementers of `amp_run_node` and
 * oscillator-like nodes.
 *
 * - Per-frame invocation: The runtime currently drives nodes in an inner
 *   frame loop (see graph_runtime.cpp) and commonly calls `amp_run_node` with
 *   a single-frame slice (frames==1). Implementations SHOULD accept larger
 *   `frames` values and behave correctly when batch-processing multiple frames
 *   in a single call, but may assume callers will also call in single-frame
 *   increments for deterministic scheduling.
 *
 * - Inputs layout: Audio and parameter views follow a (batch,channel,frame)
 *   logical layout. For slice-based calls the `data` pointer refers to the
 *   beginning of the frame slice and `frames` indicates how many frames are
 *   available starting at that pointer. The runtime may provide audio with
 *   channels aggregated across upstream sources (fan-in) and nodes must
 *   respect the provided `channels` count.
 *
 * - Output contract: Nodes must allocate an output buffer (double*) using the
 *   runtime allocator conventions (the runtime expects to call `amp_free` on
 *   returned pointers). The `out_channels` out-parameter must be set to the
 *   number of channels produced for the provided frame slice. The runtime will
 *   copy per-frame outputs into its tensors and then free the buffer.
 *
 * - State handling: The `state` argument is an opaque pointer that the node
 *   may read and replace. If the node modifies or replaces the state it should
 *   return the new state via `*state`. The runtime will call `amp_release_state`
 *   on any previous state pointer if it is replaced.
 *
 * - Control history: When provided, the `history` pointer offers time-indexed
 *   control curves that nodes may sample for deterministic control-driven
 *   modulation. Implementations should treat this object as read-only for the
 *   duration of the `amp_run_node` call.
 *
 * - Modulation semantics: Modulation sources (passed through channels) are
 *   presented as tensors with the same batch/frame layout as parameters. Nodes
 *   should follow the runtime's modulation application rules: either add or
 *   multiply (mode), support per-channel addressing and scaling, and honour
 *   channel-indexing behaviour when the source has fewer channels than the
 *   destination.
 *
 * - Performance: Node implementations should be vectorised where possible and
 *   avoid per-sample allocation. The runtime expects low-latency execution and
 *   will call nodes repeatedly in tight loops.
 */

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

typedef struct AmpGraphRuntime AmpGraphRuntime;
typedef EdgeRunnerControlHistory AmpGraphControlHistory;

void lfo_slew(const double *x, double *out, int B, int F, double r, double alpha, double *z0);
void safety_filter(const double *x, double *y, int B, int C, int F, double a, double *prev_in, double *prev_dc);
void dc_block(const double *x, double *out, int B, int C, int F, double a, double *state);
void subharmonic_process(
    const double *x,
    double *y,
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
    double *hp_y,
    double *lp_y,
    double *prev,
    int8_t *sign,
    int8_t *ff2,
    int8_t *ff4,
    int32_t *ff4_count,
    double *sub2_lp,
    double *sub4_lp,
    double *env,
    double *hp_out_y,
    double *hp_out_x
);
void envelope_process(
    const double *trigger,
    const double *gate,
    const double *drone,
    const double *velocity,
    int B,
    int F,
    int atk_frames,
    int hold_frames,
    int dec_frames,
    int sus_frames,
    int rel_frames,
    double sustain_level,
    int send_resets,
    int *stage,
    double *value,
    double *timer,
    double *vel_state,
    int64_t *activations,
    double *release_start,
    double *amp_out,
    double *reset_out
);
void phase_advance(const double *dphi, double *phase_out, int B, int F, double *phase_state, const double *reset);
void portamento_smooth(
    const double *freq_target,
    const double *port_mask,
    const double *slide_time,
    const double *slide_damp,
    int B,
    int F,
    int sr,
    double *freq_state,
    double *out
);
void arp_advance(const double *seq, int seq_len, double *offsets_out, int B, int F, int *step_state, int *timer_state, int fps);
void polyblep_arr(const double *t, const double *dt, double *out, int N);
void osc_saw_blep_c(const double *ph, const double *dphi, double *out, int B, int F);
void osc_square_blep_c(const double *ph, const double *dphi, double pw, double *out, int B, int F);
void osc_triangle_blep_c(const double *ph, const double *dphi, double *out, int B, int F, double *tri_state);

AMP_CAPI size_t amp_last_alloc_count_get(void);
AMP_CAPI int amp_native_logging_enabled(void);
AMP_CAPI void amp_native_logging_set(int enabled);
AMP_CAPI void amp_log_generated(const char *fn, void *py_ts, size_t a, size_t b);
AMP_CAPI void amp_log_native_call_external(const char *fn, size_t a, size_t b);
AMP_CAPI EdgeRunnerCompiledPlan *amp_load_compiled_plan(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
);
AMP_CAPI void amp_release_compiled_plan(EdgeRunnerCompiledPlan *plan);
AMP_CAPI EdgeRunnerControlHistory *amp_load_control_history(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
);
AMP_CAPI void amp_release_control_history(EdgeRunnerControlHistory *history);
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
);
AMP_CAPI void amp_free(double *buffer);
AMP_CAPI void amp_release_state(void *state);
AMP_CAPI AmpGraphRuntime *amp_graph_runtime_create(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
);
AMP_CAPI void amp_graph_runtime_destroy(AmpGraphRuntime *runtime);
AMP_CAPI int amp_graph_runtime_configure(AmpGraphRuntime *runtime, uint32_t batches, uint32_t frames);
AMP_CAPI void amp_graph_runtime_set_dsp_sample_rate(AmpGraphRuntime *runtime, double sample_rate);
AMP_CAPI void amp_graph_runtime_clear_params(AmpGraphRuntime *runtime);
AMP_CAPI int amp_graph_runtime_set_param(
    AmpGraphRuntime *runtime,
    const char *node_name,
    const char *param_name,
    const double *data,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
);
AMP_CAPI int amp_graph_runtime_execute(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
AMP_CAPI void amp_graph_runtime_buffer_free(double *buffer);
AMP_CAPI AmpGraphControlHistory *amp_graph_history_load(const uint8_t *blob, size_t blob_len, int frames_hint);
AMP_CAPI void amp_graph_history_destroy(AmpGraphControlHistory *history);

#ifdef __cplusplus
}
#endif

#endif /* AMP_NATIVE_H */
