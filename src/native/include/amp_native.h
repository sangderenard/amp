#ifndef AMP_NATIVE_H
#define AMP_NATIVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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

typedef struct {
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
} EdgeRunnerTensorShape;

typedef struct {
    const char *tap_name;
    const char *buffer_class;
    EdgeRunnerTensorShape shape;
    size_t frame_stride;
    double *data;
} EdgeRunnerTapBuffer;

typedef struct {
    EdgeRunnerTapBuffer *items;
    uint32_t count;
} EdgeRunnerTapBufferSet;

typedef struct {
    const char *tap_name;
    uint32_t connected;
    uint32_t subscriber_count;
    uint32_t primary_consumer_present;
} EdgeRunnerTapStatus;

typedef struct {
    EdgeRunnerTapStatus *items;
    uint32_t count;
} EdgeRunnerTapStatusSet;

typedef struct {
    EdgeRunnerTapBufferSet outputs;
    EdgeRunnerTapStatusSet status;
} EdgeRunnerTapContext;

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
    EdgeRunnerTapContext taps;
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

typedef enum {
    AMP_SCHEDULER_ORDERED = 0,
    AMP_SCHEDULER_LEARNED = 1
} AmpGraphSchedulerMode;

typedef struct {
    double early_bias;
    double late_bias;
    double saturation_bias;
} AmpGraphSchedulerParams;

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
typedef enum {
    AMP_EXECUTION_MODE_FORWARD = 0,
    AMP_EXECUTION_MODE_BACKWARD = 1
} AmpExecutionMode;

typedef struct {
    uint32_t measured_delay_frames;
    float accumulated_heat;
    /* Per-node timing (seconds) accumulated during the most recent invocation.
       - processing_time: wall time spent performing the node's work (excluding logging)
       - logging_time: time spent inside native logging helpers while the node was active
       - total_time: total wall time (processing + logging)
       - thread_cpu_time: CPU time consumed by the executing thread during the invocation
    */
    double processing_time_seconds;
    double logging_time_seconds;
    double total_time_seconds;
    double thread_cpu_time_seconds;
    double reserved[6];
} AmpNodeMetrics;

typedef struct {
    uint32_t declared_delay_frames;
    uint32_t oversample_ratio;
    int supports_v2;
    int has_metrics;
    AmpNodeMetrics metrics;
    double total_heat_accumulated;
} AmpGraphNodeSummary;

typedef struct {
    int code;
    const char *stage;
    const char *node;
    const char *detail;
} AmpGraphRuntimeErrorInfo;

#define AMP_E_UNSUPPORTED (-4)
#define AMP_E_PENDING (-5)

#define AMP_GRAPH_NODE_MAX_TAPS 8

typedef struct {
    char name[32];
    uint32_t ring_capacity;
    uint32_t ring_size;
    uint32_t reader_count;
    uint32_t head_position;
    uint32_t tail_position;
    uint64_t produced_total;
} AmpGraphNodeTapDebugEntry;

typedef struct {
    char name[64];
    uint32_t ring_capacity;
    uint32_t ring_size;
    uint32_t reader_count;
    uint32_t declared_delay_frames;
    uint32_t oversample_ratio;
    uint8_t supports_v2;
    uint8_t prefill_only;
    float last_heat;
    double last_processing_time_seconds;
    double last_total_time_seconds;
    double total_heat_accumulated;
    uint64_t debug_sequence;
    uint64_t debug_sample_count;
    uint64_t debug_total_frames;
    uint64_t debug_total_batches;
    uint64_t debug_total_channels;
    uint64_t debug_metrics_samples;
    uint64_t debug_last_timestamp_millis;
    double debug_sum_processing_seconds;
    double debug_sum_logging_seconds;
    double debug_sum_total_seconds;
    double debug_sum_thread_cpu_seconds;
    uint32_t debug_last_frames;
    uint32_t debug_last_batches;
    uint32_t debug_last_channels;
    uint32_t debug_min_frames;
    uint32_t debug_preferred_frames;
    uint32_t debug_max_frames;
    double debug_priority_weight;
    uint32_t debug_channel_expand;
    uint8_t fifo_simultaneous_availability;
    uint8_t fifo_release_policy;
    uint32_t fifo_primary_consumer;
    uint32_t tap_count;
    AmpGraphNodeTapDebugEntry taps[AMP_GRAPH_NODE_MAX_TAPS];
} AmpGraphNodeDebugEntry;

typedef struct {
    uint32_t version;
    uint32_t node_count;
    uint32_t sink_index;
    double sample_rate;
    uint32_t scheduler_mode;
    uint64_t produced_frames;
    uint64_t consumed_frames;
    uint32_t ring_capacity;
    uint32_t ring_size;
    uint32_t dump_queue_depth;
} AmpGraphDebugSnapshot;

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
);
AMP_CAPI int amp_wait_node_completion(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    void **state,
    double **out_buffer,
    int *out_channels,
    AmpNodeMetrics *metrics
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
AMP_CAPI int amp_graph_runtime_set_scheduler_mode(AmpGraphRuntime *runtime, AmpGraphSchedulerMode mode);
AMP_CAPI int amp_graph_runtime_set_scheduler_params(AmpGraphRuntime *runtime, const AmpGraphSchedulerParams *params);
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
AMP_CAPI int amp_graph_runtime_describe_node(
    AmpGraphRuntime *runtime,
    const char *node_name,
    AmpGraphNodeSummary *summary
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
AMP_CAPI int amp_graph_runtime_execute_with_history(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
AMP_CAPI int amp_graph_runtime_execute_into(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
AMP_CAPI int amp_graph_runtime_execute_history_into(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
AMP_CAPI int amp_graph_runtime_last_error(
    AmpGraphRuntime *runtime,
    AmpGraphRuntimeErrorInfo *out_error
);
AMP_CAPI void amp_graph_runtime_buffer_free(double *buffer);
AMP_CAPI AmpGraphControlHistory *amp_graph_history_load(const uint8_t *blob, size_t blob_len, int frames_hint);
AMP_CAPI void amp_graph_history_destroy(AmpGraphControlHistory *history);

typedef struct AmpGraphStreamer AmpGraphStreamer;

AMP_CAPI AmpGraphStreamer *amp_graph_streamer_create(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    uint32_t ring_frames,
    uint32_t block_frames
);
AMP_CAPI int amp_graph_streamer_start(AmpGraphStreamer *streamer);
AMP_CAPI void amp_graph_streamer_stop(AmpGraphStreamer *streamer);
AMP_CAPI void amp_graph_streamer_destroy(AmpGraphStreamer *streamer);
AMP_CAPI int amp_graph_streamer_available(AmpGraphStreamer *streamer, uint64_t *out_frames);
AMP_CAPI int amp_graph_streamer_read(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
);
AMP_CAPI int amp_graph_streamer_dump_count(AmpGraphStreamer *streamer, uint32_t *out_count);
AMP_CAPI int amp_graph_streamer_pop_dump(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
);
AMP_CAPI int amp_graph_streamer_status(
    AmpGraphStreamer *streamer,
    uint64_t *out_produced_frames,
    uint64_t *out_consumed_frames
);

typedef struct {
    uint64_t target_produced_frames;
    uint64_t target_consumed_frames;
    uint32_t maximum_inflight_frames;
    uint32_t maximum_dump_depth;
    uint32_t idle_timeout_millis;
    uint32_t total_timeout_millis;
    int require_ring_drain;
    int require_dump_drain;
} AmpGraphStreamerCompletionContract;

typedef struct {
    uint64_t produced_frames;
    uint64_t consumed_frames;
    uint32_t ring_size;
    uint32_t ring_capacity;
    uint32_t dump_queue_depth;
    uint64_t elapsed_millis;
    uint64_t since_producer_progress_millis;
    uint64_t since_consumer_progress_millis;
    uint64_t since_dump_progress_millis;
    int running;
} AmpGraphStreamerCompletionState;

typedef struct {
    int contract_satisfied;
    int producer_goal_met;
    int consumer_goal_met;
    int ring_drained;
    int dump_drained;
    int timed_out;
    int idle_timeout_triggered;
    int total_timeout_triggered;
    int inflight_limit_exceeded;
    int dump_limit_exceeded;
} AmpGraphStreamerCompletionVerdict;

AMP_CAPI int amp_graph_streamer_evaluate_completion(
    AmpGraphStreamer *streamer,
    const AmpGraphStreamerCompletionContract *contract,
    AmpGraphStreamerCompletionState *out_state,
    AmpGraphStreamerCompletionVerdict *out_verdict
);

AMP_CAPI int amp_graph_runtime_debug_snapshot(
    AmpGraphRuntime *runtime,
    AmpGraphStreamer *streamer,
    AmpGraphNodeDebugEntry *node_entries,
    uint32_t node_capacity,
    AmpGraphDebugSnapshot *snapshot
);

typedef struct {
    uint32_t refresh_millis;
    int ansi_only;
    int clear_on_exit; /* 0 = leave final frame on screen, 1 = clear cursor+screen on shutdown */
    int enable_free_clock; /* 1 = compute/display EMA free-running Hz, 0 = skip */
} AmpKpnOverlayConfig;

typedef struct AmpKpnOverlay AmpKpnOverlay;

AMP_CAPI AmpKpnOverlay *amp_kpn_overlay_create(
    AmpGraphStreamer *streamer,
    const AmpKpnOverlayConfig *config
);
AMP_CAPI int amp_kpn_overlay_start(AmpKpnOverlay *overlay);
AMP_CAPI void amp_kpn_overlay_stop(AmpKpnOverlay *overlay);
AMP_CAPI void amp_kpn_overlay_destroy(AmpKpnOverlay *overlay);

#ifdef __cplusplus
}
#endif

#endif /* AMP_NATIVE_H */
