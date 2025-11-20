#include "fft_division_delay_math.h"
#if defined(__cplusplus)
#include <cstddef>
#include <vector>
#include "fft_division_types.h"
#endif

static double read_param_tensor3(
    const EdgeRunnerParamView *view,
    int batch,
    int channel,
    int frame,
    double default_value
) {
    if (view == NULL || view->data == NULL) {
        return default_value;
    }
    if (batch < 0 || channel < 0 || frame < 0) {
        return default_value;
    }
    size_t batches = view->batches > 0U ? view->batches : 1U;
    size_t channels = view->channels > 0U ? view->channels : 1U;
    size_t frames = view->frames > 0U ? view->frames : 1U;
    if ((size_t)batch >= batches || (size_t)channel >= channels || (size_t)frame >= frames) {
        return default_value;
    }
    size_t index = ((size_t)batch * channels + (size_t)channel) * frames + (size_t)frame;
    return view->data[index];
}

static void write_param_tensor3(
    const EdgeRunnerParamView *view,
    int batch,
    int channel,
    int frame,
    double value
) {
    if (view == NULL || view->data == NULL) {
        return;
    }
    if (batch < 0 || channel < 0 || frame < 0) {
        return;
    }
    size_t batches = view->batches > 0U ? view->batches : 1U;
    size_t channels = view->channels > 0U ? view->channels : 1U;
    size_t frames = view->frames > 0U ? view->frames : 1U;
    if ((size_t)batch >= batches || (size_t)channel >= channels || (size_t)frame >= frames) {
        return;
    }
    size_t index = ((size_t)batch * channels + (size_t)channel) * frames + (size_t)frame;
    double *mutable_data = (double *)(view->data);
    mutable_data[index] = value;
}

typedef struct {
    const char *name;
    int window_size;
    int hop;
    int freq_bins;
    int time_slices;
    int pcm_block_frames;
    int backlog_cycles;
} FftDivPresetSpec;

#ifndef FFTDIV_TRACE_ENABLED
#define FFTDIV_TRACE_ENABLED 0
#endif

#if FFTDIV_TRACE_ENABLED
#define FFTDIV_TRACE(...)                                                        \
    do {                                                                         \
        fprintf(stderr, __VA_ARGS__);                                            \
        fprintf(stderr, "\n");                                                \
        fflush(stderr);                                                          \
    } while (0)
#else
#define FFTDIV_TRACE(...) ((void)0)
#endif

#ifndef FFTDIV_LOG_LEVEL_NONE
#define FFTDIV_LOG_LEVEL_NONE 0
#define FFTDIV_LOG_LEVEL_SUMMARY 1
#define FFTDIV_LOG_LEVEL_DETAIL 2
#define FFTDIV_LOG_LEVEL_TRACE 3
#endif

static int fftdiv_clamp_log_level(int level) {
    if (level < FFTDIV_LOG_LEVEL_NONE) {
        return FFTDIV_LOG_LEVEL_NONE;
    }
    if (level > FFTDIV_LOG_LEVEL_TRACE) {
        return FFTDIV_LOG_LEVEL_TRACE;
    }
    return level;
}

static int fftdiv_parse_log_level_string(const char *value, int fallback) {
    if (value == NULL || *value == '\0') {
        return fftdiv_clamp_log_level(fallback);
    }
    char buffer[16];
    size_t len = strlen(value);
    if (len >= sizeof(buffer)) {
        len = sizeof(buffer) - 1;
    }
    for (size_t i = 0; i < len; ++i) {
        char ch = value[i];
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char)(ch - 'A' + 'a');
        }
        buffer[i] = ch;
    }
    buffer[len] = '\0';
    if (strcmp(buffer, "silent") == 0 || strcmp(buffer, "quiet") == 0) {
        return FFTDIV_LOG_LEVEL_NONE;
    }
    if (strcmp(buffer, "summary") == 0 || strcmp(buffer, "info") == 0 || strcmp(buffer, "normal") == 0) {
        return FFTDIV_LOG_LEVEL_SUMMARY;
    }
    if (strcmp(buffer, "detail") == 0 || strcmp(buffer, "verbose") == 0) {
        return FFTDIV_LOG_LEVEL_DETAIL;
    }
    if (strcmp(buffer, "trace") == 0 || strcmp(buffer, "debug") == 0) {
        return FFTDIV_LOG_LEVEL_TRACE;
    }
    char *end = NULL;
    long numeric = strtol(buffer, &end, 10);
    if (end != buffer && *end == '\0') {
        return fftdiv_clamp_log_level((int)numeric);
    }
    return fftdiv_clamp_log_level(fallback);
}

static int fftdiv_should_log(const node_state_t *state, int required_level) {
    if (state == NULL) {
        return 0;
    }
    int configured = state->u.fftdiv.log_level;
    if (configured < required_level) {
        return 0;
    }
    return 1;
}

#define FFTDIV_LOG(state_ptr, level, ...)                                      \
    do {                                                                       \
        if (fftdiv_should_log((state_ptr), (level))) {                          \
            fprintf(stderr, __VA_ARGS__);                                       \
        }                                                                      \
    } while (0)

#if defined(__cplusplus)
static int fftdiv_window_effective_span(const node_state_t *state, int wheel_length) {
    if (wheel_length <= 0) {
        return 0;
    }
    const int configured_hop = (state != nullptr && state->u.fftdiv.wheel_hop > 0)
        ? state->u.fftdiv.wheel_hop
        : 1;
    if (state == nullptr) {
        int span = wheel_length;
        if (span < configured_hop) {
            span = configured_hop;
        }
        if (span > wheel_length) {
            span = wheel_length;
        }
        return span;
    }
    int span = state->u.fftdiv.wheel_active_window_span;
    if (span <= 0) {
        span = wheel_length;
    }
    if (span < configured_hop) {
        span = configured_hop;
    }
    if (span > wheel_length) {
        span = wheel_length;
    }
    return span;
}

static bool fftdiv_window_release_tail(
    node_state_t *state,
    int wheel_length,
    int *wheel_tail,
    int *wheel_filled,
    const char *reason
) {
    if (state == nullptr || wheel_tail == nullptr || wheel_filled == nullptr || wheel_length <= 0) {
        return false;
    }
    auto &meta = state->u.fftdiv.wheel_slice_metadata;
    if (meta.size() != (size_t)wheel_length) {
        return false;
    }
    bool released = false;
    while (*wheel_filled > 0) {
        FftDivFilledSlice &tail_slice = meta[(size_t)(*wheel_tail)];
        if (!tail_slice.valid || !tail_slice.stage4_emitted) {
            break;
        }
        tail_slice.valid = false;
        tail_slice.stage4_emitted = false;
        *wheel_tail = (*wheel_tail + 1) % wheel_length;
        *wheel_filled -= 1;
        released = true;
    }
    if (released && fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
               "[STAGE4-WHEEL-DELIVERY] reason=%s tail=%d filled=%d len=%d\n",
                   (reason != NULL) ? reason : "unspecified",
                   *wheel_tail,
                   *wheel_filled,
                   wheel_length);
    }
    return released;
}

static void fftdiv_window_reserve_leading_edge(
    node_state_t *state,
    int slot,
    int frame_index,
    int wheel_length,
    int *wheel_tail,
    int *wheel_filled
) {
    if (state == nullptr || wheel_tail == nullptr || wheel_filled == nullptr || wheel_length <= 0) {
        return;
    }
    const int active_span = fftdiv_window_effective_span(state, wheel_length);
    if (*wheel_filled >= active_span && fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                   "[STAGE2-RESERVE] slot=%d frame=%d waiting headroom tail=%d filled=%d span=%d\n",
                   slot,
                   frame_index,
                   *wheel_tail,
                   *wheel_filled,
                   active_span);
    }
}
#endif

static int fftdiv_names_equal(const char *lhs, const char *rhs) {
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        int la = tolower((unsigned char)(*lhs));
        int rb = tolower((unsigned char)(*rhs));
        if (la != rb) {
            return 0;
        }
        lhs += 1;
        rhs += 1;
    }
    return *lhs == '\0' && *rhs == '\0';
}

static const FftDivPresetSpec *fftdiv_find_preset(const char *name) {
    if (name == NULL || *name == '\0') {
        return NULL;
    }
    static const FftDivPresetSpec kPresets[] = {
        { "analysis_1024", 2048, 1024, 1024, 1024, 2048, 512 },
        { "analysis_512", 1024, 256, 1024, 2048, 1024, 512 },
        { "low_latency", 512, 128, 512, 1024, 512, 256 }
    };
    const size_t count = sizeof(kPresets) / sizeof(kPresets[0]);
    for (size_t i = 0; i < count; ++i) {
        if (fftdiv_names_equal(name, kPresets[i].name)) {
            return &kPresets[i];
        }
    }
    return NULL;
}

static const EdgeRunnerTapBuffer *find_tap_buffer(
    const EdgeRunnerTapContext *context,
    const char *tap_name
) {
    if (context == NULL || tap_name == NULL) {
        return NULL;
    }
    if (context->outputs.items == NULL) {
        return NULL;
    }
    for (uint32_t i = 0; i < context->outputs.count; ++i) {
        const EdgeRunnerTapBuffer *buffer = &context->outputs.items[i];
        if (buffer->tap_name == NULL) {
            continue;
        }
        if (strcmp(buffer->tap_name, tap_name) == 0) {
            return buffer;
        }
    }
    return NULL;
}

static void tap_buffer_write_row(
    const EdgeRunnerTapBuffer *buffer,
    int batch,
    int frame_index,
    const double *src,
    int value_count
) {
    if (buffer == NULL || buffer->data == NULL || src == NULL) {
        return;
    }
    if (batch < 0 || frame_index < 0 || value_count <= 0) {
        return;
    }
    size_t batches = buffer->shape.batches > 0U ? buffer->shape.batches : 1U;
    size_t channels = buffer->shape.channels > 0U ? buffer->shape.channels : 1U;
    if ((size_t)batch >= batches) {
        return;
    }
    if (buffer->shape.frames > 0U && (uint32_t)frame_index >= buffer->shape.frames) {
        return;
    }
    size_t stride = buffer->frame_stride > 0U ? buffer->frame_stride : (batches * channels);
    size_t copy = (size_t)value_count;
    if (copy > channels) {
        copy = channels;
    }
    double *frame_ptr = buffer->data + (size_t)frame_index * stride;
    double *batch_ptr = frame_ptr + (size_t)batch * channels;
#if FFTDIV_TRACE_ENABLED
        fprintf(stderr, "[TAP-WRITE] buf=%p frames=%u chans=%u stride=%zu batch=%d frame=%d dst=%p copy=%zu src=%p\n",
            (void*)buffer->data, buffer->shape.frames, buffer->shape.channels, stride, batch, frame_index, (void*)batch_ptr, (size_t)(value_count <= (int)channels ? value_count : channels), (const void*)src);
        double before0 = batch_ptr[0];
#endif
        memcpy(batch_ptr, src, copy * sizeof(double));
#if FFTDIV_TRACE_ENABLED
        double after0 = batch_ptr[0];
        fprintf(stderr, "[TAP-WRITE-VERIFY] frame=%d dst[0] before=%.6g after=%.6g src0=%.6g\n",
            frame_index, before0, after0, src[0]);
#endif
    if (copy < channels) {
        size_t remaining = channels - copy;
        memset(batch_ptr + copy, 0, remaining * sizeof(double));
    }
}

static void zero_tensor_slice(
    FftWorkingTensor *tensor,
    int tensor_page,
    int slot,
    int tensor_slice,
    int tensor_freq_bins
) {
    if (tensor == NULL || tensor_freq_bins <= 0) {
        return;
    }
    for (int bin = 0; bin < tensor_freq_bins; ++bin) {
        (*tensor)(tensor_page, slot, bin, tensor_slice) = std::complex<double>(0.0, 0.0);
    }
}

#if defined(__cplusplus)
static double *fftdiv_spectral_scratch_real_ptr(node_state_t *state, int lane, int tensor_slice) {
    if (state == NULL) {
        return NULL;
    }
    auto &scratch = state->u.fftdiv.spectral_scratch;
    if (lane < 0 || tensor_slice < 0) {
        return NULL;
    }
    if (lane >= scratch.lanes || tensor_slice >= scratch.time_slices) {
        return NULL;
    }
    if (scratch.freq_bins <= 0) {
        return NULL;
    }
    const size_t offset = (((size_t)lane * (size_t)scratch.time_slices) + (size_t)tensor_slice) * (size_t)scratch.freq_bins;
    if (offset >= scratch.real.size()) {
        return NULL;
    }
    return scratch.real.data() + offset;
}

static double *fftdiv_spectral_scratch_imag_ptr(node_state_t *state, int lane, int tensor_slice) {
    if (state == NULL) {
        return NULL;
    }
    auto &scratch = state->u.fftdiv.spectral_scratch;
    if (lane < 0 || tensor_slice < 0) {
        return NULL;
    }
    if (lane >= scratch.lanes || tensor_slice >= scratch.time_slices) {
        return NULL;
    }
    if (scratch.freq_bins <= 0) {
        return NULL;
    }
    const size_t offset = (((size_t)lane * (size_t)scratch.time_slices) + (size_t)tensor_slice) * (size_t)scratch.freq_bins;
    if (offset >= scratch.imag.size()) {
        return NULL;
    }
    return scratch.imag.data() + offset;
}

static int fftdiv_spectral_scratch_bins(const node_state_t *state) {
    if (state == NULL) {
        return 0;
    }
    return state->u.fftdiv.spectral_scratch.freq_bins;
}

static void fftdiv_ring_ensure_capacity(
    std::vector<double> &buffer,
    std::size_t required
) {
    if (buffer.size() < required) {
        buffer.resize(required, 0.0);
    }
}

#if defined(__cplusplus)
static size_t fftdiv_pcm_backlog_limit(const node_state_t *state) {
    if (state == NULL) {
        return 0U;
    }
    size_t max_block = state->u.fftdiv.stream_max_pcm_block;
    if (max_block == 0U) {
        max_block = (state->u.fftdiv.window_size > 0) ? (size_t)state->u.fftdiv.window_size : 1U;
    }
    size_t backlog_cycles = state->u.fftdiv.stream_backlog_cycles > 0U
        ? state->u.fftdiv.stream_backlog_cycles
        : 1U;
    return max_block * backlog_cycles;
}

static void fftdiv_trim_pcm_backlog(
    node_state_t *state,
    std::vector<double> &pcm_backlog,
    std::size_t &pcm_consumed_samples,
    size_t window_size
) {
    if (state == NULL || window_size == 0U || pcm_backlog.empty()) {
        return;
    }
    const size_t guard = window_size;
    size_t trim_target = 0U;
    if (pcm_consumed_samples > guard) {
        trim_target = pcm_consumed_samples - guard;
    }
    size_t backlog_limit = fftdiv_pcm_backlog_limit(state);
    if (backlog_limit > 0U && backlog_limit < guard) {
        backlog_limit = guard;
    }
    if (backlog_limit > 0U && pcm_backlog.size() > backlog_limit) {
        const size_t overflow = pcm_backlog.size() - backlog_limit;
        if (overflow > trim_target) {
            trim_target = overflow;
        }
    }
    if (trim_target == 0U) {
        return;
    }
    if (trim_target > pcm_consumed_samples) {
        trim_target = pcm_consumed_samples;
    }
    if (trim_target > pcm_backlog.size()) {
        trim_target = pcm_backlog.size();
    }
    pcm_backlog.erase(pcm_backlog.begin(), pcm_backlog.begin() + trim_target);
    pcm_consumed_samples -= trim_target;
}
#endif

template <typename SlotState>
static void fftdiv_ring_append_frames(
    node_state_t *state,
    SlotState &slot,
    const double *stage_real,
    const double *stage_imag,
    size_t frames,
    int window_size
) {
    if (state == nullptr || window_size <= 0 || stage_real == nullptr || stage_imag == nullptr) {
        return;
    }
    if (frames == 0) {
        return;
    }
    const std::size_t capacity_frames = slot.forward_ring_capacity_frames > 0
        ? slot.forward_ring_capacity_frames
        : 1U;
    if (slot.forward_real.size() < capacity_frames * (std::size_t)window_size) {
        fftdiv_ring_ensure_capacity(slot.forward_real, capacity_frames * (std::size_t)window_size);
    }
    #if defined(__cplusplus)
    if (slot.forward_imag.size() < capacity_frames * (std::size_t)window_size) {
        fftdiv_ring_ensure_capacity(slot.forward_imag, capacity_frames * (std::size_t)window_size);
    }
    std::size_t write = slot.forward_ring_write % capacity_frames;
    std::size_t read = slot.forward_ring_read % capacity_frames;
    std::size_t filled = slot.forward_ring_filled;
    #endif

    const std::size_t bins = (std::size_t)window_size;

    if (frames >= capacity_frames) {
        // Keep only the most recent frames.
        const std::size_t skip = frames - capacity_frames;
        stage_real += skip * bins;
        stage_imag += skip * bins;
        frames = capacity_frames;
        read = 0U;
        write = 0U;
        filled = 0U;
        slot.forward_ring_wrapped = true;
    }

    const std::size_t free_frames = (filled < capacity_frames) ? (capacity_frames - filled) : 0U;
    if (frames > free_frames) {
        const std::size_t drop = frames - free_frames;
        if (drop < filled) {
            read = (read + drop) % capacity_frames;
            filled -= drop;
        } else {
            read = write;
            filled = 0U;
        }
        slot.forward_ring_wrapped = true;
    }

    std::size_t remaining = frames;
    while (remaining > 0) {
        const std::size_t contiguous = capacity_frames - write;
        const std::size_t write_frames = (remaining < contiguous) ? remaining : contiguous;
        const std::size_t copy_bins = write_frames * bins;
        const std::size_t write_offset = write * bins;
        const std::size_t stage_offset = (frames - remaining) * bins;
        std::memcpy(
            slot.forward_real.data() + write_offset,
            stage_real + stage_offset,
            copy_bins * sizeof(double));
        std::memcpy(
            slot.forward_imag.data() + write_offset,
            stage_imag + stage_offset,
            copy_bins * sizeof(double));
        write = (write + write_frames) % capacity_frames;
        remaining -= write_frames;
    }

    filled += frames;
    if (filled > capacity_frames) {
        filled = capacity_frames;
        slot.forward_ring_wrapped = true;
        read = write;
    }

    slot.forward_ring_write = write;
    slot.forward_ring_read = read;
    slot.forward_ring_filled = filled;
}

template <typename SlotState>
static const double *fftdiv_ring_frame_real(
    const SlotState &slot,
    size_t frame_offset,
    int window_size
) {
    if (window_size <= 0 || slot.forward_ring_capacity_frames == 0U) {
        return nullptr;
    }
    if (slot.forward_ring_filled == 0U || frame_offset >= slot.forward_ring_filled) {
        return nullptr;
    }
    const std::size_t capacity_frames = slot.forward_ring_capacity_frames;
    const std::size_t index = (slot.forward_ring_read + frame_offset) % capacity_frames;
    const std::size_t offset = index * (std::size_t)window_size;
    if (offset + (std::size_t)window_size > slot.forward_real.size()) {
        return nullptr;
    }
    return slot.forward_real.data() + offset;
}

template <typename SlotState>
static const double *fftdiv_ring_frame_imag(
    const SlotState &slot,
    size_t frame_offset,
    int window_size
) {
    if (window_size <= 0 || slot.forward_ring_capacity_frames == 0U) {
        return nullptr;
    }
    if (slot.forward_ring_filled == 0U || frame_offset >= slot.forward_ring_filled) {
        return nullptr;
    }
    const std::size_t capacity_frames = slot.forward_ring_capacity_frames;
    const std::size_t index = (slot.forward_ring_read + frame_offset) % capacity_frames;
    const std::size_t offset = index * (std::size_t)window_size;
    if (offset + (std::size_t)window_size > slot.forward_imag.size()) {
        return nullptr;
    }
    return slot.forward_imag.data() + offset;
}

#if defined(__cplusplus)
typedef struct {
    size_t pcm_frames;
    int window_size;
    int backend_hop;
    int working_span;
    int working_hop;
    int istft_tail_frames;
} FftDivTailPlan;

static FftDivTailPlan fftdiv_calculate_tail_plan(
    const node_state_t *state,
    bool pipeline_active_hint
) {
    FftDivTailPlan plan{};
    plan.window_size = 1;
    plan.backend_hop = 1;
    plan.working_span = 1;
    plan.working_hop = 1;
    if (state == NULL) {
        plan.pcm_frames = 1U;
        plan.istft_tail_frames = 0;
        return plan;
    }
    const auto &fftdiv = state->u.fftdiv;
    plan.window_size = (fftdiv.window_size > 0) ? fftdiv.window_size : 1;
    plan.backend_hop = (fftdiv.backend_hop > 0) ? fftdiv.backend_hop : 1;
    plan.working_span = (fftdiv.wheel_active_window_span > 0) ? fftdiv.wheel_active_window_span : 1;
    plan.working_hop = (fftdiv.wheel_hop > 0) ? fftdiv.wheel_hop : 1;

    const int64_t l_istft = fftdiv_delay_compute_l_istft(
        plan.window_size,
        plan.backend_hop,
        plan.working_hop);
    plan.istft_tail_frames = (l_istft > 0) ? static_cast<int>(l_istft) : 0;

    size_t pcm_frames = fftdiv_delay_frames_for_sample(
        0U,
        plan.window_size,
        plan.backend_hop,
        plan.working_hop,
        plan.window_size,
        plan.istft_tail_frames);
    if (pcm_frames == 0U) {
        pcm_frames = pipeline_active_hint
            ? (plan.window_size > 1 ? (size_t)(plan.window_size - 1) : 1U)
            : 1U;
    }
    plan.pcm_frames = (pcm_frames > 0U) ? pcm_frames : 1U;
    return plan;
}

template <typename SlotState>
static void fftdiv_stage1_append_zero_tail(
    node_state_t *state,
    SlotState &slot,
    size_t tail_frames
) {
    if (state == NULL || tail_frames == 0U) {
        return;
    }
    if (slot.zero_tail_enqueued) {
        return;
    }
    const size_t reserve_target = slot.pcm_backlog.size() + tail_frames;
    if (reserve_target > slot.pcm_backlog.capacity()) {
        try {
            slot.pcm_backlog.reserve(reserve_target);
        } catch (...) {
            // Ignore reserve failures and fall back to incremental growth.
        }
    }
    slot.pcm_backlog.insert(slot.pcm_backlog.end(), tail_frames, 0.0);
    slot.total_ingested_samples += tail_frames;
    slot.tail_injected_samples = tail_frames;
    slot.zero_tail_enqueued = true;
    slot.ingest_finalized = true;
    (void)state;
}
#endif

struct FftDivActiveWindowView {
    std::complex<double> *base{nullptr};
    int lanes{0};
    int freq_bins{0};
    int wheel_length{0};
    int window_span{0};
    int window_start{0};

    bool valid() const {
        return base != nullptr && lanes > 0 && freq_bins > 0 && wheel_length > 0 && window_span > 0;
    }

    std::complex<double> *element(int time, int freq, int lane) const {
        if (!valid() || time < 0 || time >= window_span || freq < 0 || freq >= freq_bins || lane < 0 || lane >= lanes) {
            return nullptr;
        }
        const int slice = (window_start + time) % wheel_length;
        const size_t lane_stride = static_cast<size_t>(freq_bins) * static_cast<size_t>(wheel_length);
        const size_t freq_stride = static_cast<size_t>(wheel_length);
        const size_t index =
            static_cast<size_t>(lane) * lane_stride +
            static_cast<size_t>(freq) * freq_stride +
            static_cast<size_t>(slice);
        return base + index;
    }

    std::complex<double> *time_slice(int time) const {
        if (!valid() || time < 0 || time >= window_span) {
            return nullptr;
        }
        const int slice = (window_start + time) % wheel_length;
        return base + static_cast<size_t>(slice);
    }

    size_t time_stride() const {
        return 1U;
    }

    size_t freq_stride() const {
        return static_cast<size_t>(wheel_length);
    }

    size_t lane_stride() const {
        return static_cast<size_t>(freq_bins) * static_cast<size_t>(wheel_length);
    }
};

static FftDivActiveWindowView fftdiv_build_active_window_view_internal(
    node_state_t *state,
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_slice,
    int filled_override
) {
    FftDivActiveWindowView view;
    if (state == nullptr || tensor == nullptr) {
        return view;
    }

    const int lane_count = state->u.fftdiv.working_tensor_lanes > 0
        ? state->u.fftdiv.working_tensor_lanes
        : state->u.fftdiv.default_lane_count;
    const int freq_bins = state->u.fftdiv.working_tensor_freq_bins;
    const int wheel_length = state->u.fftdiv.working_tensor_time_slices;
    if (lane_count <= 0 || freq_bins <= 0 || wheel_length <= 0) {
        return view;
    }

    if (tensor_page < 0) {
        tensor_page = 0;
    }
    if (tensor_slice < 0) {
        tensor_slice = 0;
    }
    tensor_slice %= wheel_length;

    int requested_span = state->u.fftdiv.wheel_active_window_span;
    if (requested_span <= 0) {
        requested_span = wheel_length;
    }
    if (requested_span > wheel_length) {
        requested_span = wheel_length;
    }

    int available = (filled_override >= 0) ? filled_override : state->u.fftdiv.wheel_filled_slices;
    if (available < 0) {
        available = 0;
    }
    if (available > wheel_length) {
        available = wheel_length;
    }

    view.base = tensor->data() + static_cast<size_t>(tensor_page) * static_cast<size_t>(lane_count) * static_cast<size_t>(freq_bins) * static_cast<size_t>(wheel_length);
    view.lanes = lane_count;
    view.freq_bins = freq_bins;
    view.wheel_length = wheel_length;
    view.window_start = tensor_slice % wheel_length;

    if (available == 0) {
        view.window_span = 0;
        return view;
    }

    int window_span = requested_span;
    if (window_span > available) {
        window_span = available;
    }
    if (window_span <= 0) {
        window_span = available;
    }

    int window_start = tensor_slice - (window_span - 1);
    while (window_start < 0) {
        window_start += wheel_length;
    }
    window_start %= wheel_length;

    view.window_span = window_span;
    view.window_start = window_start;
    return view;
}

static FftDivActiveWindowView fftdiv_build_active_window_view(
    node_state_t *state,
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_slice
) {
    return fftdiv_build_active_window_view_internal(state, tensor, tensor_page, tensor_slice, -1);
}

static int fftdiv_last_hop_slice_from_read_cursor(
    int read_cursor,
    int wheel_length,
    int hop_length,
    int fallback_slice
) {
    if (wheel_length <= 0) {
        return fallback_slice;
    }
    int normalized_read = read_cursor % wheel_length;
    if (normalized_read < 0) {
        normalized_read += wheel_length;
    }
    int effective_hop = (hop_length > 0) ? hop_length : 1;
    if (effective_hop > wheel_length) {
        effective_hop = wheel_length;
    }
    /* Read cursor names the start of data already eligible for release; advance by one hop
       so Stage 3 emits the most recent hop-aligned slice. */
    int slice = normalized_read + (effective_hop - 1);
    if (slice >= wheel_length) {
        slice %= wheel_length;
    }
    return slice;
}

struct FftDivOperatorLaneBinding {
    int slot_index{-1};
    int tensor_lane{-1};
    bool enable_pcm_in{false};
    bool enable_pcm_out{false};
    bool enable_spectral_in{false};
    bool enable_spectral_out{false};
    bool active{false};
};

struct FftDivOperatorContext {
    FftDivActiveWindowView window;
    const FftDivOperatorLaneBinding *lanes{nullptr};
    size_t lane_count{0U};
    int hop{0};
    int window_size{0};
    int wheel_length{0};
    int wheel_head{0};
    int wheel_tail{0};
    int64_t frame_index{0};
    int64_t pcm_sample_index{0};
    size_t slice_index{0U};
    size_t pcm_sample_stride{0U};
    double sample_rate{0.0};
    double timeline_seconds{0.0};
    double hop_seconds{0.0};
};
#endif

static void fftdiv_copy_spectrum_to_working(
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_lane,
    int tensor_slice,
    int tensor_freq_bins,
    const double *source_real,
    const double *source_imag,
    int source_bins
) {
    if (tensor == NULL || tensor_freq_bins <= 0) {
        return;
    }
    if (source_real == NULL || source_imag == NULL || source_bins <= 0) {
        for (int bin = 0; bin < tensor_freq_bins; ++bin) {
            (*tensor)(tensor_page, tensor_lane, bin, tensor_slice) = std::complex<double>(0.0, 0.0);
        }
        return;
    }
    int limit = tensor_freq_bins < source_bins ? tensor_freq_bins : source_bins;
    for (int bin = 0; bin < limit; ++bin) {
        (*tensor)(tensor_page, tensor_lane, bin, tensor_slice) = std::complex<double>(source_real[bin], source_imag[bin]);
    }
    for (int bin = limit; bin < tensor_freq_bins; ++bin) {
        (*tensor)(tensor_page, tensor_lane, bin, tensor_slice) = std::complex<double>(0.0, 0.0);
    }
}

static int stage_ingest_spectrum_input(
    double *spectral_real,
    double *spectral_imag,
    int window_size,
    const EdgeRunnerParamView *spectral_real_view,
    const EdgeRunnerParamView *spectral_imag_view,
    int tap_slot,
    int frame_index,
    int reset_buffer,
    int reset_scratch,
    double *scratch_real_target,
    double *scratch_imag_target,
    int scratch_bins
) {
    int contributed = 0;
    (void)spectral_real;
    (void)spectral_imag;
    if ((spectral_real_view == NULL || spectral_real_view->data == NULL) &&
        (spectral_imag_view == NULL || spectral_imag_view->data == NULL)) {
        if (reset_scratch && scratch_real_target != NULL && scratch_imag_target != NULL && scratch_bins > 0) {
            memset(scratch_real_target, 0, (size_t)scratch_bins * sizeof(double));
            memset(scratch_imag_target, 0, (size_t)scratch_bins * sizeof(double));
        }
        return 0;
    }

    if (reset_scratch && scratch_real_target != NULL && scratch_imag_target != NULL && scratch_bins > 0) {
        memset(scratch_real_target, 0, (size_t)scratch_bins * sizeof(double));
        memset(scratch_imag_target, 0, (size_t)scratch_bins * sizeof(double));
    }

    if (scratch_real_target == NULL || scratch_imag_target == NULL || scratch_bins <= 0) {
        (void)spectral_real;
        (void)spectral_imag;
        return 0;
    }

    const int process_bins = window_size;
    const int copy_bins = scratch_bins < process_bins ? scratch_bins : process_bins;
    for (int bin = 0; bin < copy_bins; ++bin) {
        double add_real = read_param_tensor3(spectral_real_view, tap_slot, bin, frame_index, 0.0);
        double add_imag = read_param_tensor3(spectral_imag_view, tap_slot, bin, frame_index, 0.0);
        if (add_real != 0.0 || add_imag != 0.0) {
            scratch_real_target[bin] += add_real;
            scratch_imag_target[bin] += add_imag;
            contributed = 1;
        }
    }

    if (reset_scratch && scratch_bins > process_bins) {
        memset(scratch_real_target + process_bins, 0, (size_t)(scratch_bins - process_bins) * sizeof(double));
        memset(scratch_imag_target + process_bins, 0, (size_t)(scratch_bins - process_bins) * sizeof(double));
    }

    return contributed;
}

#if defined(__cplusplus)
static void fftdiv_copy_working_to_slot_buffer(
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_lane,
    int tensor_slice,
    int tensor_freq_bins,
    double *slot_real,
    double *slot_imag,
    int slot_bins
) {
    if (tensor == NULL || slot_real == NULL || slot_imag == NULL || slot_bins <= 0) {
        return;
    }
    const int limit = tensor_freq_bins < slot_bins ? tensor_freq_bins : slot_bins;
    for (int bin = 0; bin < limit; ++bin) {
        const std::complex<double> value = (*tensor)(tensor_page, tensor_lane, bin, tensor_slice);
        slot_real[bin] = value.real();
        slot_imag[bin] = value.imag();
    }
    for (int bin = limit; bin < slot_bins; ++bin) {
        slot_real[bin] = 0.0;
        slot_imag[bin] = 0.0;
    }
}
#endif

#if defined(__cplusplus)
static void fftdiv_run_operator_stack(
    node_state_t *state,
    const FftDivOperatorContext &context
) {
    if (state == nullptr) {
        return;
    }
    if (!context.window.valid()) {
        return;
    }
    if (state->u.fftdiv.operator_steps.empty()) {
        return;
    }
    (void)state;
    (void)context;
    /* TODO: execute operator_steps using operator_arena tensors and context. */
}
#endif

static void stage_emit_spectral(
    node_state_t *state,
    double *spectral_real,
    double *spectral_imag,
    int window_size,
    int tap_slot,
    int frame_index,
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_lane,
    int tensor_slice,
    int tensor_freq_bins
) {
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
               "[STAGE4-EMIT-PRE] frame=%d slot=%d before_copy re[0]=%.6g im[0]=%.6g\n",
               frame_index, tap_slot, spectral_real[0], spectral_imag[0]);
    if (tensor != NULL && tensor_freq_bins > 0) {
        const int copy_bins = tensor_freq_bins < window_size ? tensor_freq_bins : window_size;
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                   "[STAGE4-EMIT-TENSOR] frame=%d slot=%d tensor=%p bins=%d page=%d lane=%d slice=%d\n",
                   frame_index, tap_slot, (void*)tensor, copy_bins, tensor_page, tensor_lane, tensor_slice);
        for (int bin = 0; bin < copy_bins; ++bin) {
            const std::complex<double> value = (*tensor)(tensor_page, tensor_lane, bin, tensor_slice);
            spectral_real[bin] = value.real();
            spectral_imag[bin] = value.imag();
            if (bin < 4) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                           "[STAGE4-EMIT-COPY] frame=%d slot=%d bin=%d tensor_re=%.6g tensor_im=%.6g\n",
                           frame_index, tap_slot, bin, value.real(), value.imag());
            }
        }
        for (int bin = copy_bins; bin < window_size; ++bin) {
            spectral_real[bin] = 0.0;
            spectral_imag[bin] = 0.0;
        }
    }
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
               "[STAGE4-EMIT-POST] frame=%d slot=%d after_copy re[0]=%.6g im[0]=%.6g\n",
               frame_index, tap_slot, spectral_real[0], spectral_imag[0]);
    
    // Push spectral data to mailbox
    AmpSpectralMailboxEntry *entry = amp_spectral_mailbox_entry_create(
        tap_slot, frame_index, window_size, spectral_real, spectral_imag);
    if (entry != NULL) {
        amp_node_spectral_mailbox_push(state, entry);
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                   "[STAGE4-MAILBOX-PUSH] frame=%d slot=%d window_size=%d pushed to spectral mailbox\n",
                   frame_index, tap_slot, window_size);
    }
}

#if defined(__cplusplus)
static int fftdiv_realize_operator_arena(node_state_t *state) {
    if (state == nullptr) {
        return 0;
    }
    auto &arena = state->u.fftdiv.operator_arena;
    for (auto &entry : arena) {
        auto &spec = entry.spec;
        const bool spec_valid = (
            spec.cache_pages > 0 &&
            spec.lanes > 0 &&
            spec.freq_bins > 0 &&
            spec.time_slices > 0);
        if (!spec_valid) {
            entry.tensor.reset();
            continue;
        }
        const bool allocate_new = !entry.tensor
            || entry.tensor->dimension(0) != spec.cache_pages
            || entry.tensor->dimension(1) != spec.lanes
            || entry.tensor->dimension(2) != spec.freq_bins
            || entry.tensor->dimension(3) != spec.time_slices;
        if (allocate_new) {
            using EigenIndex = Eigen::Index;
            entry.tensor = std::unique_ptr<FftWorkingTensor>(
                new (std::nothrow) FftWorkingTensor(
                    static_cast<EigenIndex>(spec.cache_pages),
                    static_cast<EigenIndex>(spec.lanes),
                    static_cast<EigenIndex>(spec.freq_bins),
                    static_cast<EigenIndex>(spec.time_slices)));
            if (!entry.tensor) {
                return -1;
            }
            entry.tensor->setZero();
        }
    }
    return 0;
}

static void fftdiv_prepare_operator_frame(node_state_t *state) {
    if (state == nullptr) {
        return;
    }
    for (auto &entry : state->u.fftdiv.operator_arena) {
        if (!entry.tensor) {
            continue;
        }
        if (!entry.spec.persistent) {
            entry.tensor->setZero();
        }
    }
}
#endif

#include "fft_division_stage_lock.h"
static FftDivStageLockSnapshot fftdiv_snapshot_stage_locks(const node_state_t *state);
#if defined(__cplusplus)
static void fftdiv_prepare_lane_plan(
    node_state_t *state,
    int slot_count,
    const EdgeRunnerNodeInputs *inputs,
    const EdgeRunnerParamView *spectral_input_real_view,
    const EdgeRunnerParamView *spectral_input_imag_view,
    const EdgeRunnerTapBuffer *spectral_real_tap,
    const EdgeRunnerTapBuffer *spectral_imag_tap
) {
    if (state == nullptr) {
        return;
    }
    auto &plan = state->u.fftdiv.lane_plan;
    if (plan.size() < static_cast<size_t>(slot_count)) {
        plan.resize(static_cast<size_t>(slot_count));
    }

    // Spectral output is always available via mailbox
    const bool spectral_out_available = true;

    const bool pcm_input_connected =
        inputs != nullptr &&
        inputs->audio.has_audio &&
        inputs->audio.data != nullptr;

    const uint32_t audio_batches = (inputs != nullptr && inputs->audio.batches > 0U)
        ? inputs->audio.batches
        : 1U;
    const uint32_t audio_channels = (inputs != nullptr && inputs->audio.channels > 0U)
        ? inputs->audio.channels
        : 1U;
    const uint32_t pcm_slot_limit = audio_batches * audio_channels;

    const uint32_t real_batches = (spectral_input_real_view != nullptr && spectral_input_real_view->batches > 0U)
        ? spectral_input_real_view->batches
        : 0U;
    const uint32_t imag_batches = (spectral_input_imag_view != nullptr && spectral_input_imag_view->batches > 0U)
        ? spectral_input_imag_view->batches
        : 0U;

    for (int slot = 0; slot < slot_count; ++slot) {
        // Determine if this lane could still make forward progress anywhere in the pipeline
        bool has_ring_data = false;
        bool has_wheel_frame = false;
        bool has_pcm_backlog = false;
        bool has_pending_output = false;
        if ((size_t)slot < state->u.fftdiv.stream_slots.size()) {
            const auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            has_ring_data = (slot_state.forward_ring_filled > 0U);
            has_pcm_backlog = (slot_state.pcm_backlog.size() > slot_state.pcm_consumed_samples);
            if (!slot_state.pending_spectra.empty() || !slot_state.inverse_queue.empty() ||
                (slot_state.inverse_handle != nullptr && amp_fft_backend_stream_pending_pcm(slot_state.inverse_handle) > 0U)) {
                has_pending_output = true;
            }
            // Wheel: check for any unemitted slice where this lane has a ready frame
            for (const auto &slice_meta : state->u.fftdiv.wheel_slice_metadata) {
                if (!slice_meta.valid || slice_meta.stage4_emitted) continue;
                if (slice_meta.lanes.size() > (size_t)slot && slice_meta.lanes[(size_t)slot].frame_ready) {
                    has_wheel_frame = true;
                    break;
                }
            }
            fprintf(stderr,
                    "[LANE-PLAN] slot=%d ring=%zu backlog=%zu wheel_ready=%d pending_out=%d\n",
                    slot,
                    (size_t)slot_state.forward_ring_filled,
                    (size_t)(slot_state.pcm_backlog.size() > slot_state.pcm_consumed_samples
                        ? (slot_state.pcm_backlog.size() - slot_state.pcm_consumed_samples)
                        : 0U),
                    has_wheel_frame ? 1 : 0,
                    has_pending_output ? 1 : 0);
        }
        auto &lane = plan[(size_t)slot];
        lane.slot_index = slot;
        lane.tensor_lane = slot;

        const bool has_pcm_feed = pcm_input_connected && (uint32_t)slot < pcm_slot_limit;
        const bool spect_real_in =
            spectral_input_real_view != nullptr &&
            spectral_input_real_view->data != nullptr &&
            (real_batches == 0U || (uint32_t)slot < real_batches);
        const bool spect_imag_in =
            spectral_input_imag_view != nullptr &&
            spectral_input_imag_view->data != nullptr &&
            (imag_batches == 0U || (uint32_t)slot < imag_batches);
        const bool has_spectral_in = spect_real_in || spect_imag_in || has_wheel_frame;

        // Lane should be active if there's anything left that could be used
        const bool could_use_anything = has_pcm_feed || has_spectral_in || has_ring_data || has_wheel_frame || has_pcm_backlog || has_pending_output;
        const bool enable_pcm_in = has_pcm_feed || has_spectral_in || has_ring_data || has_pcm_backlog;
        const bool enable_pcm_out = could_use_anything;

        lane.enable_pcm_in = enable_pcm_in;
        lane.enable_pcm_out = enable_pcm_out;
        lane.enable_spectral_in = has_spectral_in;
        lane.enable_spectral_out = spectral_out_available;
        lane.active = (lane.enable_pcm_in || lane.enable_spectral_in) &&
            (lane.enable_pcm_out || lane.enable_spectral_out);
    }
}
#endif

static int fftdiv_execute_block(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    size_t runtime_pcm_block,
    const FftDivStageLockSnapshot *stage_locks,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics,
    int flush_mode
) {
        if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL || frames <= 0) {
        return -1;
    }
#if !defined(__cplusplus)
    (void)stage_locks;
#endif
    if (flush_mode < AMP_FFT_STREAM_FLUSH_NONE || flush_mode > AMP_FFT_STREAM_FLUSH_FINAL) {
        flush_mode = AMP_FFT_STREAM_FLUSH_NONE;
    }
    const bool flush_request = (flush_mode != AMP_FFT_STREAM_FLUSH_NONE);
    if (frames <= 0) {
        frames = 1;
    }
    if (batches <= 0) {
        batches = 1;
    }
    int input_channels = channels;
    if (input_channels <= 0) {
        if (inputs->audio.channels > 0U) {
            input_channels = (int)inputs->audio.channels;
        } else {
            input_channels = 1;
        }
    }
    int slot_count = batches * input_channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }

    int default_window_size = 8;
    int default_hop = (default_window_size > 1) ? default_window_size / 2 : 1;
    int default_freq_bins = default_window_size;
    int default_time_slices = frames > 0 ? frames : 1;
    int default_pcm_block_frames = (int)runtime_pcm_block;
    size_t default_backlog_cycles = state->u.fftdiv.stream_backlog_cycles > 0U
        ? state->u.fftdiv.stream_backlog_cycles
        : 1U;

    char preset_name[64];
    preset_name[0] = '\0';
    if (json_copy_string(
            descriptor->params_json,
            descriptor->params_len,
            "stream_preset",
            preset_name,
            sizeof(preset_name)) != 0) {
        const FftDivPresetSpec *preset = fftdiv_find_preset(preset_name);
        if (preset != NULL) {
            if (preset->window_size > 0) {
                default_window_size = preset->window_size;
            }
            if (preset->hop > 0) {
                default_hop = preset->hop;
            } else if (default_window_size > 1) {
                default_hop = default_window_size / 2;
            } else {
                default_hop = 1;
            }
            if (preset->freq_bins > 0) {
                default_freq_bins = preset->freq_bins;
            } else {
                default_freq_bins = default_window_size;
            }
            if (preset->time_slices > 0) {
                default_time_slices = preset->time_slices;
            }
            if (preset->pcm_block_frames > 0) {
                default_pcm_block_frames = preset->pcm_block_frames;
            }
            if (preset->backlog_cycles > 0) {
                default_backlog_cycles = (size_t)preset->backlog_cycles;
            }
        }
    }

    int window_size = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "window_size",
        default_window_size
    );
    if (window_size <= 0) {
        window_size = 1;
    }

    int default_algorithm = parse_algorithm_string(descriptor->params_json, descriptor->params_len, FFT_ALGORITHM_EIGEN);
    default_algorithm = clamp_algorithm_kind(default_algorithm);
    int default_window_kind = parse_window_string(descriptor->params_json, descriptor->params_len, FFT_WINDOW_HANN);
    default_window_kind = clamp_window_kind(default_window_kind);

    if (ensure_fft_state_buffers(state, slot_count, window_size, 1) != 0) {
        return -1;
    }
    state->u.fftdiv.window_kind = default_window_kind;
    state->u.fftdiv.window_size = window_size;

    char log_verbosity_buffer[16];
    log_verbosity_buffer[0] = '\0';
    int default_log_level = FFTDIV_LOG_LEVEL_DETAIL;
    if (json_copy_string(
            descriptor->params_json,
            descriptor->params_len,
            "log_verbosity",
            log_verbosity_buffer,
            sizeof(log_verbosity_buffer)) != 0) {
        default_log_level = fftdiv_parse_log_level_string(log_verbosity_buffer, default_log_level);
    }
    int configured_log_level = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "log_level",
        default_log_level
    );
    configured_log_level = fftdiv_clamp_log_level(configured_log_level);
    state->u.fftdiv.log_level = configured_log_level;
    const int default_slice_cap = (configured_log_level >= FFTDIV_LOG_LEVEL_TRACE) ? 12 : 0;
    int configured_slice_cap = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "log_slice_bin_cap",
        default_slice_cap
    );
    if (configured_slice_cap < 0) {
        configured_slice_cap = 0;
    }
    state->u.fftdiv.log_slice_bin_cap = configured_slice_cap;

    // Default working frequency bins should follow the effective window_size
    // so the tensor's bin dimension matches the STFT output bins unless explicitly overridden.
    int working_freq_bins = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_frequency_bins",
        window_size
    );
    if (working_freq_bins <= 0) {
        working_freq_bins = window_size;
    }
    int working_time_slices = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_time_slices",
        json_get_int(
            descriptor->params_json,
            descriptor->params_len,
            "working_ft_duration_frames",
            (default_time_slices > 0) ? default_time_slices : (frames > 0 ? frames : 1))
    );
    if (working_time_slices <= 0) {
        working_time_slices = 1;
    }
    // Derive default working hop from the effective window_size, not the earlier default_hop.
    // This ensures tensor slice advancement matches the configured window and avoids misalignment
    // when a preset or earlier defaults set default_hop from a different window size.
    int working_hop = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_hop",
        (window_size > 1 ? window_size / 2 : 1)
    );
    if (working_hop <= 0) {
        working_hop = 1;
    }
    if (working_time_slices == 1) {
        working_hop = 1;
    } else if (working_hop > working_time_slices) {
        working_hop = working_time_slices;
    }
    int working_active_window_span = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_active_window_span",
        working_time_slices
    );
    if (working_active_window_span <= 0) {
        working_active_window_span = working_time_slices;
    }
    if (working_active_window_span > working_time_slices) {
        working_active_window_span = working_time_slices;
    }
    if (working_active_window_span < working_hop) {
        /* Enforce overlap so the active span always covers the most recent hop. */
        working_active_window_span = working_hop;
    }
    int required_wheel_slices = working_active_window_span + working_hop;
    if (required_wheel_slices < 0) {
        required_wheel_slices = 1;
    }
    if (working_time_slices < required_wheel_slices) {
        /* Wheel must fit the active window plus a trailing hop for the read cursor. */
        working_time_slices = required_wheel_slices;
    }
    int configured_pcm_block_param = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "stream_pcm_block_frames",
        (default_pcm_block_frames > 0) ? default_pcm_block_frames : (int)runtime_pcm_block
    );
    if (configured_pcm_block_param <= 0) {
        configured_pcm_block_param = (int)runtime_pcm_block;
    }
    size_t configured_pcm_block = (size_t)configured_pcm_block_param;
    size_t requested_pcm_block = runtime_pcm_block;
    if (configured_pcm_block > requested_pcm_block) {
        requested_pcm_block = configured_pcm_block;
    }
    int backlog_cycles_param = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "stream_backlog_cycles",
        (int)(default_backlog_cycles > 0U ? default_backlog_cycles : 1U)
    );
    size_t backlog_cycles = (backlog_cycles_param > 0) ? (size_t)backlog_cycles_param : 1U;
    state->u.fftdiv.stream_backlog_cycles = backlog_cycles;
    size_t hop_stride = (size_t)(working_hop > 0 ? working_hop : 1);
    size_t frames_per_block = (requested_pcm_block + hop_stride - 1U) / hop_stride;
    if (frames_per_block == 0U) {
        frames_per_block = 1U;
    }
    size_t desired_frame_capacity = frames_per_block * state->u.fftdiv.stream_backlog_cycles;
    if (desired_frame_capacity == 0U) {
        desired_frame_capacity = frames_per_block;
    }
    state->u.fftdiv.stream_max_pcm_block = requested_pcm_block;
    state->u.fftdiv.stream_max_fft_frames = desired_frame_capacity;
    // Configure backend hop: prefer explicit backend_hop; default to working_hop
    int backend_hop = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "backend_hop",
        working_hop
    );
    if (backend_hop <= 0) {
        backend_hop = 1;
    }
    if (backend_hop > window_size) {
        backend_hop = window_size;
    }
    state->u.fftdiv.backend_hop = backend_hop;
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
        "[FFT-CONFIG] W=%d working_freq_bins=%d working_time_slices=%d working_hop=%d backend_hop=%d pcm_block=%zu stage_frames=%zu ring_frames=%zu wheel_len=%d\n",
        window_size,
        working_freq_bins,
        working_time_slices,
        working_hop,
        backend_hop,
        state->u.fftdiv.stream_max_pcm_block,
        state->u.fftdiv.stream_max_fft_frames,
        state->u.fftdiv.spectral_ring_capacity_frames,
        working_time_slices
    );
    if (ensure_fft_working_tensor(state, slot_count, working_freq_bins, working_time_slices) != 0) {
        return -1;
    }

    if (ensure_fft_spectral_scratch(state, slot_count, window_size, working_time_slices) != 0) {
        return -1;
    }
    state->u.fftdiv.wheel_hop = working_hop;
    state->u.fftdiv.wheel_length = working_time_slices;
    state->u.fftdiv.wheel_active_window_span = working_active_window_span;
    size_t spectral_ring_frames = desired_frame_capacity;
    size_t wheel_frames = (working_time_slices > 0) ? (size_t)working_time_slices : 0U;
    if (spectral_ring_frames < wheel_frames) {
        spectral_ring_frames = wheel_frames;
    }
    if (spectral_ring_frames == 0U) {
        spectral_ring_frames = 1U;
    }
    state->u.fftdiv.spectral_ring_capacity_frames = spectral_ring_frames;

    if (ensure_fft_stream_slots(state, slot_count, window_size, default_window_kind) != 0) {
        return -1;
    }
#if defined(__cplusplus)
    if (state->u.fftdiv.stream_slots.size() < static_cast<size_t>(slot_count)) {
        return -1;
    }
    state->u.fftdiv.default_lane_count = slot_count;
    state->u.fftdiv.lane_plan.resize(static_cast<size_t>(slot_count));
#endif

    char spectral_aggregation_mode[32];
    spectral_aggregation_mode[0] = '\0';
    int preserve_tensor_on_ingest = 0;
    if (json_copy_string(
            descriptor->params_json,
            descriptor->params_len,
            "spectral_input_aggregation",
            spectral_aggregation_mode,
            sizeof(spectral_aggregation_mode)) != 0) {
        for (size_t i = 0; spectral_aggregation_mode[i] != '\0'; ++i) {
            spectral_aggregation_mode[i] = (char)tolower((unsigned char)spectral_aggregation_mode[i]);
        }
        if (strcmp(spectral_aggregation_mode, "accumulate") == 0 ||
            strcmp(spectral_aggregation_mode, "aggregate") == 0 ||
            strcmp(spectral_aggregation_mode, "sum") == 0 ||
            strcmp(spectral_aggregation_mode, "preserve") == 0 ||
            strcmp(spectral_aggregation_mode, "buffered_fill") == 0) {
            preserve_tensor_on_ingest = 1;
        }
    }
    int preserve_tensor_flag = json_get_bool(
        descriptor->params_json,
        descriptor->params_len,
        "preserve_spectral_tensor_on_ingest",
        preserve_tensor_on_ingest
    );
    state->u.fftdiv.preserve_tensor_on_ingest = preserve_tensor_flag ? 1 : 0;

    size_t total_samples = (size_t)slot_count * (size_t)frames;
    double *buffer = (double *)malloc(total_samples * sizeof(double));
    amp_last_alloc_count = total_samples;
    if (buffer == NULL) {
        return -1;
    }
    const double *audio_base = (inputs->audio.has_audio && inputs->audio.data != NULL) ? inputs->audio.data : NULL;
    const EdgeRunnerParamView *spectral_input_real_view = find_param(inputs, "spectral_input_real");
    const EdgeRunnerParamView *spectral_input_imag_view = find_param(inputs, "spectral_input_imag");
    const EdgeRunnerTapBuffer *spectral_real_tap = find_tap_buffer(&inputs->taps, "spectral_real");
    const EdgeRunnerTapBuffer *spectral_imag_tap = find_tap_buffer(&inputs->taps, "spectral_imag");
    const bool final_delivery = (inputs->audio.has_audio & EDGE_RUNNER_AUDIO_FLAG_FINAL) != 0;
    FftDivTailPlan final_tail_plan{};
    if (final_delivery) {
        final_tail_plan = fftdiv_calculate_tail_plan(state, true);
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
                   "[STAGE1-FINAL-FLAG] W_fft=%d H_fft=%d H_work=%d L_istft=%d frame0_delay=%zu\n",
                   final_tail_plan.window_size,
                   final_tail_plan.backend_hop,
                   final_tail_plan.working_hop,
                   final_tail_plan.istft_tail_frames,
                   final_tail_plan.pcm_frames);
    }

    FftWorkingTensor *working_tensor = state->u.fftdiv.working_tensor;
    const int tensor_time_slices = state->u.fftdiv.working_tensor_time_slices > 0
        ? state->u.fftdiv.working_tensor_time_slices
        : 1;
    const int tensor_freq_bins = state->u.fftdiv.working_tensor_freq_bins > 0
        ? state->u.fftdiv.working_tensor_freq_bins
        : window_size;
    int wheel_length = state->u.fftdiv.wheel_length > 0 ? state->u.fftdiv.wheel_length : tensor_time_slices;
    if (wheel_length <= 0) {
        wheel_length = 1;
    }
    int wheel_head = state->u.fftdiv.wheel_head;
    int wheel_tail = state->u.fftdiv.wheel_tail;
    int wheel_filled = state->u.fftdiv.wheel_filled_slices;
    int wheel_hop = state->u.fftdiv.wheel_hop > 0 ? state->u.fftdiv.wheel_hop : 1;
    if (wheel_head < 0 || wheel_head >= wheel_length) {
        wheel_head = 0;
    }
    if (wheel_tail < 0 || wheel_tail >= wheel_length) {
        wheel_tail = 0;
    }
    if (wheel_filled < 0) {
        wheel_filled = 0;
    }
    const int tensor_page = 0;
    int scratch_time_cursor = state->u.fftdiv.spectral_scratch.time_cursor;
    const int scratch_time_slices = state->u.fftdiv.spectral_scratch.time_slices > 0
        ? state->u.fftdiv.spectral_scratch.time_slices
        : 1;
    if (scratch_time_cursor < 0 || scratch_time_cursor >= scratch_time_slices) {
        scratch_time_cursor = 0;
    }
#if defined(__cplusplus)
    fftdiv_prepare_lane_plan(
        state,
        slot_count,
        inputs,
        spectral_input_real_view,
        spectral_input_imag_view,
        spectral_real_tap,
        spectral_imag_tap);

    std::vector<FftDivOperatorLaneBinding> operator_lane_bindings;
    operator_lane_bindings.resize(static_cast<size_t>(slot_count));
    for (int slot = 0; slot < slot_count; ++slot) {
        const auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
        auto &binding = operator_lane_bindings[(size_t)slot];
        binding.slot_index = lane.slot_index;
        binding.tensor_lane = lane.tensor_lane;
        binding.enable_pcm_in = lane.enable_pcm_in;
        binding.enable_pcm_out = lane.enable_pcm_out;
        binding.enable_spectral_in = lane.enable_spectral_in;
        binding.enable_spectral_out = lane.enable_spectral_out;
        binding.active = lane.active;
    }

    size_t pcm_expected_samples = 0U;
    for (const auto &lane_meta : state->u.fftdiv.lane_plan) {
        if (lane_meta.active && lane_meta.enable_pcm_out) {
            // Account for STFT latency: with window W, we get (W-1) samples delay
            // So for N input frames, expect max(0, N - (W-1)) output samples in steady state
            const int latency_samples = (window_size > 1) ? (window_size - 1) : 0;
            const size_t expected_for_lane = (frames > latency_samples) 
                ? static_cast<size_t>(frames - latency_samples)
                : 0U;
            pcm_expected_samples += expected_for_lane;
        }
    }

    bool made_progress = false;
    size_t pcm_written_samples = 0U;
    size_t queue_drains_without_emission = 0U;

    int active_lane_count = 0;
    for (const auto &lane_meta : state->u.fftdiv.lane_plan) {
        if (lane_meta.active) {
            active_lane_count += 1;
        }
    }
    if (fftdiv_realize_operator_arena(state) != 0) {
        free(buffer);
        return -1;
    }
#endif

    const int64_t frame_counter = state->u.fftdiv.wheel_frame_counter;
    int frame_index_int = (int)frame_counter;
    size_t base_index = 0;
    int metrics_window_span = 0;
#if defined(__cplusplus)
        fftdiv_prepare_operator_frame(state);
    const int scratch_slice = scratch_time_cursor;

    std::vector<size_t> lane_frames_emitted(static_cast<size_t>(slot_count), 0U);
    std::vector<double> lane_pcm_batch(static_cast<size_t>(slot_count) * static_cast<size_t>(frames), 0.0);

    FftDivStageLockSnapshot live_stage_locks = {0};
    const FftDivStageLockSnapshot *locks_for_stage = stage_locks;
    const bool incoming_pcm_available =
        inputs != nullptr &&
        inputs->audio.has_audio &&
        inputs->audio.data != nullptr &&
        frames > 0;
    if (locks_for_stage != nullptr && incoming_pcm_available) {
        live_stage_locks = *locks_for_stage;
        live_stage_locks.stage1_ready = true;
        locks_for_stage = &live_stage_locks;
    }

    const bool stage1_unlocked = (locks_for_stage == nullptr) || locks_for_stage->stage1_ready;
    const bool stage2_unlocked = (locks_for_stage == nullptr) || locks_for_stage->stage2_ready;
    const bool stage3_unlocked = (locks_for_stage == nullptr) || locks_for_stage->stage3_ready;
    const bool stage4_unlocked = (locks_for_stage == nullptr) || locks_for_stage->stage4_ready;
    const bool stage5_unlocked = (locks_for_stage == nullptr) || locks_for_stage->stage5_ready;

    // Per-round lock status report (restored): reflect snapshot gate statuses for all stages
    if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                   "[STAGE-LOCKS] frame_index=%d s1=%d s2=%d s3=%d s4=%d s5=%d\n",
                   frame_index_int,
                   stage1_unlocked ? 1 : 0,
                   stage2_unlocked ? 1 : 0,
                   stage3_unlocked ? 1 : 0,
                   stage4_unlocked ? 1 : 0,
                   stage5_unlocked ? 1 : 0);
    }

    if (stage1_unlocked) {
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                "[STAGE1-START] frame_index=%d frames=%d slot_count=%d final_delivery=%d\n",
                frame_index_int,
                frames,
                slot_count,
                final_delivery ? 1 : 0);
        }
        for (int slot = 0; slot < slot_count; ++slot) {
            if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                    "[STAGE1-PROCESS] frame_index=%d slot=%d\n",
                    frame_index_int,
                    slot);
            }
            size_t data_idx = base_index + (size_t)slot;
            auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            const bool warmup_was_complete = slot_state.warmup_complete;
            auto &forward_stage_real = slot_state.forward_stage_real;
            auto &forward_stage_imag = slot_state.forward_stage_imag;
            double *slot_spectral_real = slot_state.forward_real.empty() ? NULL : slot_state.forward_real.data();
            double *slot_spectral_imag = slot_state.forward_imag.empty() ? NULL : slot_state.forward_imag.data();
            auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];

            lane.frame_ready = false;

            if (slot_spectral_real == NULL || slot_spectral_imag == NULL) {
                continue;
            }

            const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
            double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice);
            double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice);
            const int scratch_bins = fftdiv_spectral_scratch_bins(state);

            if (!lane.active) {
                continue;
            }

            size_t frames_emitted = 0;
            if (lane.enable_pcm_in) {
                const size_t hop = (size_t)slot_count;
                double *pcm_cursor = lane_pcm_batch.data() + (size_t)slot * (size_t)frames;
                if (audio_base != NULL && frames > 0) {
                    const size_t backlog_growth = (size_t)frames;
                    const size_t reserve_target = slot_state.pcm_backlog.size() + backlog_growth;
                    if (reserve_target > slot_state.pcm_backlog.capacity()) {
                        try {
                            slot_state.pcm_backlog.reserve(reserve_target);
                        } catch (...) {
                            // Ignore reserve failures; push_back will retry allocations as needed.
                        }
                    }
                    for (int frame_cursor = 0; frame_cursor < frames; ++frame_cursor) {
                        const size_t source_frame = slot_state.input_frame_cursor + (size_t)frame_cursor;
                        const size_t audio_index = source_frame * hop + (size_t)slot;
                        double input_sample = 0.0;
                        if (audio_index < (size_t)slot_count * (size_t)frames) {
                            input_sample = audio_base[audio_index];
                        }
                        pcm_cursor[frame_cursor] = input_sample;
                        slot_state.pcm_backlog.push_back(input_sample);
                    }
                    slot_state.input_frame_cursor += (size_t)frames;
                    slot_state.total_ingested_samples += backlog_growth;
                }
                // Append tail zeros immediately after input PCM, before calculating pending_samples
                if (final_delivery && !slot_state.final_flag_observed) {
                    slot_state.final_flag_observed = true;
                    const size_t tail_frames = fftdiv_delay_frames_for_signal_length(
                        slot_state.total_ingested_samples,
                        final_tail_plan.window_size,
                        final_tail_plan.backend_hop,
                        final_tail_plan.working_hop,
                        final_tail_plan.window_size,
                        final_tail_plan.istft_tail_frames);
                    if (tail_frames > 0U) {
                        const bool already_enqueued = slot_state.zero_tail_enqueued;
                        fftdiv_stage1_append_zero_tail(state, slot_state, tail_frames);
                        if (!already_enqueued && slot_state.zero_tail_enqueued) {
                            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
                                       "[STAGE1-ZERO-TAIL] slot=%d tail_frames=%zu total_ingested=%zu\n",
                                       slot,
                                       tail_frames,
                                       slot_state.total_ingested_samples);
                        }
                    }
                }
                const size_t pending_samples = (slot_state.pcm_backlog.size() > slot_state.pcm_consumed_samples)
                    ? (slot_state.pcm_backlog.size() - slot_state.pcm_consumed_samples)
                    : 0U;
                if (slot_state.forward_handle != NULL && pending_samples > 0U) {
                    size_t stage_capacity_frames = slot_state.forward_frame_capacity;
                    if (stage_capacity_frames == 0U) {
                        stage_capacity_frames = state->u.fftdiv.stream_max_fft_frames;
                    }
                    if (stage_capacity_frames == 0U) {
                        stage_capacity_frames = 1U;
                    }
                    // Ensure capacity is at least large enough for pending PCM with W=window_size, H=1
                    // Max frames from N samples: (N - W) + 1
                    const size_t min_capacity = (pending_samples >= (size_t)window_size) 
                        ? (pending_samples - (size_t)window_size + 1U)
                        : 1U;
                    if (stage_capacity_frames < min_capacity) {
                        stage_capacity_frames = min_capacity;
                    }
                    const size_t stage_capacity = stage_capacity_frames * (size_t)window_size;
                    if (slot_state.forward_stage_real.size() != stage_capacity) {
                        try {
                            slot_state.forward_stage_real.assign(stage_capacity, 0.0);
                        } catch (...) {
                            slot_state.forward_stage_real.clear();
                        }
                    }
                    if (slot_state.forward_stage_imag.size() != stage_capacity) {
                        try {
                            slot_state.forward_stage_imag.assign(stage_capacity, 0.0);
                        } catch (...) {
                            slot_state.forward_stage_imag.clear();
                        }
                    }
                    if (slot_state.forward_real.size() != slot_state.forward_ring_capacity_frames * (size_t)window_size) {
                        try {
                            slot_state.forward_real.assign(
                                slot_state.forward_ring_capacity_frames * (size_t)window_size,
                                0.0);
                        } catch (...) {
                            slot_state.forward_real.clear();
                        }
                    }
                    if (slot_state.forward_imag.size() != slot_state.forward_ring_capacity_frames * (size_t)window_size) {
                        try {
                            slot_state.forward_imag.assign(
                                slot_state.forward_ring_capacity_frames * (size_t)window_size,
                                0.0);
                        } catch (...) {
                            slot_state.forward_imag.clear();
                        }
                    }
                    slot_state.forward_frame_capacity = stage_capacity_frames;
                    slot_spectral_real = slot_state.forward_stage_real.empty() ? NULL : slot_state.forward_stage_real.data();
                    slot_spectral_imag = slot_state.forward_stage_imag.empty() ? NULL : slot_state.forward_stage_imag.data();
                    const double *pcm_source = slot_state.pcm_backlog.empty()
                        ? NULL
                        : slot_state.pcm_backlog.data() + slot_state.pcm_consumed_samples;
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                               "[STAGE1-INPUT] slot=%d\n"
                               "  PCM calling code put into command: pcm_backlog.size()=%zu\n"
                               "  PCM stage 1 actually put into backend: pcm_consumed_samples=%zu\n"
                               "  Samples pending to push to backend: pending_samples=%zu\n"
                               "  Forward handle: %p\n",
                               slot,
                               slot_state.pcm_backlog.size(),
                               slot_state.pcm_consumed_samples,
                               pending_samples,
                               (void*)slot_state.forward_handle);
                    if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE) && pending_samples > 0U && pcm_source != NULL) {
                        const size_t pcm_log_count = (pending_samples < (size_t)window_size)
                            ? pending_samples
                            : (size_t)window_size;
                        const int pcm_log_cap = 8;
                        const size_t to_log = (pcm_log_count < (size_t)pcm_log_cap)
                            ? pcm_log_count
                            : (size_t)pcm_log_cap;
                        for (int sample_idx = 0; sample_idx < to_log; ++sample_idx) {
                            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                       "[STAGE1-PCM] slot=%d frame=%d sample=%d value=%.6f\n",
                                       slot,
                                       frame_index_int,
                                       sample_idx,
                                       pcm_source[sample_idx]);
                        }
                    }
#if FFTDIV_TRACE_ENABLED
                    const int pcm_trace_limit = 16;
                    const size_t pcm_values_to_log = (pending_samples < (size_t)pcm_trace_limit)
                        ? pending_samples
                        : (size_t)pcm_trace_limit;
                    FFTDIV_TRACE(
                        "[fftdiv] pcm-block slot=%d frame=%d samples=%zu log=%zu",
                        slot,
                        frame_index_int,
                        pending_samples,
                        pcm_values_to_log);
                    for (size_t frame_cursor = 0; frame_cursor < pcm_values_to_log; ++frame_cursor) {
                        FFTDIV_TRACE(
                            "[fftdiv] pcm[%d]=%.12f",
                            (int)frame_cursor,
                            pcm_source ? pcm_source[frame_cursor] : 0.0);
                    }
                    size_t inverse_pending_before = (slot_state.inverse_handle != NULL)
                        ? amp_fft_backend_stream_pending_pcm(slot_state.inverse_handle)
                        : 0U;
                    FFTDIV_TRACE(
                        "[fftdiv] forward-pre slot=%d frame=%d warmup=%d queue=%zu pending_pcm=%zu capacity=%zu",
                        slot,
                        frame_index_int,
                        slot_state.warmup_complete ? 1 : 0,
                        (size_t)slot_state.inverse_queue.size(),
                        inverse_pending_before,
                        stage_capacity_frames);
#endif
                    if (slot_spectral_real != NULL && slot_spectral_imag != NULL && pcm_source != NULL) {
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                                   "[STAGE1-FFT-PREP] slot=%d input_samples=%zu stage_capacity=%zu\n",
                                   slot,
                                   pending_samples,
                                   stage_capacity_frames);
                        frames_emitted = amp_fft_backend_stream_push(
                            slot_state.forward_handle,
                            pcm_source,
                            pending_samples,
                            window_size,
                            slot_spectral_real,
                            slot_spectral_imag,
                            stage_capacity_frames,
                            flush_mode);
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                                   "[STAGE1-FFT] slot=%d\n"
                                   "  Spectral data removed from backend: frames_emitted=%zu\n"
                                   "  FFT waiting in ring for Stage 2: forward_ring_filled=%zu\n",
                                   slot,
                                   frames_emitted,
                                   slot_state.forward_ring_filled);
                        if (frames_emitted > 0) {
                            fftdiv_ring_append_frames(
                                state,
                                slot_state,
                                slot_spectral_real,
                                slot_spectral_imag,
                                frames_emitted,
                                window_size);
                            const double *ring_real = fftdiv_ring_frame_real(slot_state, 0U, window_size);
                            const double *ring_imag = fftdiv_ring_frame_imag(slot_state, 0U, window_size);
                            if (ring_real != NULL && ring_imag != NULL) {
                                slot_spectral_real = const_cast<double *>(ring_real);
                                slot_spectral_imag = const_cast<double *>(ring_imag);
                            }
                            slot_state.forward_frames_ready = slot_state.forward_ring_filled;
                            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                                       "[STAGE1-READY] slot=%d\n"
                                       "  Spectral data made ready for Stage 2: forward_frames_ready=%zu\n",
                                       slot,
                                       slot_state.forward_frames_ready);
                        }
#if FFTDIV_TRACE_ENABLED
                        size_t inverse_pending_after = (slot_state.inverse_handle != NULL)
                            ? amp_fft_backend_stream_pending_pcm(slot_state.inverse_handle)
                            : 0U;
                        FFTDIV_TRACE(
                            "[fftdiv] forward-post slot=%d frame=%d emitted=%zu warmup=%d queue=%zu pending_pcm=%zu",
                            slot,
                            frame_index_int,
                            frames_emitted,
                            slot_state.warmup_complete ? 1 : 0,
                            (size_t)slot_state.inverse_queue.size(),
                            inverse_pending_after);
                        if (slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                            const size_t spectrum_frames_to_log = (frames_emitted < 4U) ? frames_emitted : 4U;
                            const int spectrum_bins_to_log = (window_size < 16) ? window_size : 16;
                            for (size_t emitted_index = 0; emitted_index < spectrum_frames_to_log; ++emitted_index) {
                                const size_t base = emitted_index * (size_t)window_size;
                                FFTDIV_TRACE(
                                    "[fftdiv] spectrum slot=%d frame=%d emitted_idx=%zu bins=%d",
                                    slot,
                                    frame_index_int,
                                    emitted_index,
                                    spectrum_bins_to_log);
                                for (int bin = 0; bin < spectrum_bins_to_log; ++bin) {
                                    const size_t idx = base + (size_t)bin;
                                    FFTDIV_TRACE(
                                        "[fftdiv] spec[%zu]=%.12f%+.12fi",
                                        idx,
                                        slot_spectral_real[idx],
                                        slot_spectral_imag[idx]);
                                }
                            }
                        }
#endif
                    } else {
                        frames_emitted = 0U;
                        const size_t ring_capacity = slot_state.forward_ring_capacity_frames * (size_t)window_size;
                        slot_state.forward_real.assign(ring_capacity, 0.0);
                        slot_state.forward_imag.assign(ring_capacity, 0.0);
                        slot_state.forward_frame_capacity = stage_capacity_frames;
                        slot_spectral_real = slot_state.forward_real.data();
                        slot_spectral_imag = slot_state.forward_imag.data();
                    }
                    slot_state.pcm_consumed_samples = slot_state.pcm_backlog.size();
                    fftdiv_trim_pcm_backlog(state, slot_state.pcm_backlog, slot_state.pcm_consumed_samples, (size_t)window_size);
                }
            }
            lane_frames_emitted[(size_t)slot] = frames_emitted;
            slot_state.forward_frames_ready = slot_state.forward_ring_filled;
            if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                           "[STAGE1-RING] slot=%d frames_ready=%zu read=%zu write=%zu filled=%zu capacity=%zu\n",
                           slot,
                           (size_t)slot_state.forward_frames_ready,
                           (size_t)slot_state.forward_ring_read,
                           (size_t)slot_state.forward_ring_write,
                           (size_t)slot_state.forward_ring_filled,
                           (size_t)slot_state.forward_ring_capacity_frames);
            }
            if (frames_emitted > 0) {
                slot_state.warmup_complete = true;
                made_progress = true;
            }
            if (!slot_state.warmup_complete) {
#if FFTDIV_TRACE_ENABLED
                FFTDIV_TRACE(
                    "[fftdiv] warmup slot=%d frame=%d emitted=%zu staged=%f",
                    slot,
                    frame_index_int,
                    frames_emitted,
                    slot_state.last_pcm_output);
#endif
                continue;
            }
            int spectral_ready = (frames_emitted > 0) ? 1 : 0;
            const int reset_spectral_buffer = 0;
            const int reset_scratch_slice = 0;

            if (scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                if (frames_emitted > 0) {
                    const int scratch_copy_bins = (scratch_bins < window_size) ? scratch_bins : window_size;
                    memcpy(scratch_real, slot_spectral_real, (size_t)scratch_copy_bins * sizeof(double));
                    memcpy(scratch_imag, slot_spectral_imag, (size_t)scratch_copy_bins * sizeof(double));
                    if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_copy_bins < scratch_bins) {
                        memset(scratch_real + scratch_copy_bins, 0, (size_t)(scratch_bins - scratch_copy_bins) * sizeof(double));
                        memset(scratch_imag + scratch_copy_bins, 0, (size_t)(scratch_bins - scratch_copy_bins) * sizeof(double));
                    }
                }
            }

            if (lane.enable_spectral_in) {
                spectral_ready |= stage_ingest_spectrum_input(
                    slot_spectral_real,
                    slot_spectral_imag,
                    window_size,
                    spectral_input_real_view,
                    spectral_input_imag_view,
                    slot,
                    frame_index_int,
                    reset_spectral_buffer,
                    reset_scratch_slice,
                    scratch_real,
                    scratch_imag,
                    scratch_bins);
            }

            if (!spectral_ready) {
#if FFTDIV_TRACE_ENABLED
                FFTDIV_TRACE(
                    "[fftdiv] spectral-wait slot=%d frame=%d queue=%zu warmup=%d",
                    slot,
                    frame_index_int,
                    (size_t)slot_state.inverse_queue.size(),
                    slot_state.warmup_complete ? 1 : 0);
#endif
                continue;
            }
        }
    } else {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                   "[STAGE1-SKIP] snapshot locked, deferring PCM -> FFT ingest\n");
    }
    double effective_sample_rate = state->u.fftdiv.sample_rate_hint;
    if (sample_rate > 0.0) {
        effective_sample_rate = sample_rate;
        state->u.fftdiv.sample_rate_hint = sample_rate;
    }
    const double hop_seconds = (effective_sample_rate > 0.0)
        ? ((double)wheel_hop) / effective_sample_rate
        : 0.0;
    const double timeline_seconds = state->u.fftdiv.timeline_seconds;

    auto &exec_snapshot = state->u.fftdiv.execute_snapshot;
    exec_snapshot.batches = batches;
    exec_snapshot.channels = input_channels;
    exec_snapshot.frames = frames;
    exec_snapshot.slot_count = slot_count;
    exec_snapshot.requested_sample_rate = sample_rate;
    exec_snapshot.effective_sample_rate = effective_sample_rate;
    exec_snapshot.window_size = window_size;
    exec_snapshot.working_tensor_lanes = state->u.fftdiv.working_tensor_lanes;
    exec_snapshot.working_tensor_freq_bins = tensor_freq_bins;
    exec_snapshot.working_tensor_time_slices = tensor_time_slices;
    exec_snapshot.wheel_length = wheel_length;
    exec_snapshot.wheel_head = wheel_head;
    exec_snapshot.wheel_tail = wheel_tail;
    exec_snapshot.wheel_filled = wheel_filled;
    exec_snapshot.wheel_hop = wheel_hop;
    exec_snapshot.scratch_time_cursor = scratch_time_cursor;
    exec_snapshot.scratch_time_slices = scratch_time_slices;
    exec_snapshot.hop_seconds = hop_seconds;
    exec_snapshot.timeline_seconds = timeline_seconds;

    size_t processed_passes = 0U;
    int wheel_head_cursor = wheel_head;
    int wheel_tail_cursor = wheel_tail;
    int wheel_filled_cursor = wheel_filled;
    int scratch_cursor = scratch_time_cursor;
    double timeline_cursor = timeline_seconds;

    std::vector<FftDivFilledSlice*> filled_slices;
    filled_slices.reserve((wheel_length > 0) ? (size_t)wheel_length : 1U);

    std::vector<size_t> lane_frames_remaining;
    std::vector<size_t> lane_frame_offsets;
    if (slot_count > 0) {
        lane_frames_remaining.resize((size_t)slot_count, 0U);
        lane_frame_offsets.resize((size_t)slot_count, 0U);
        for (int slot = 0; slot < slot_count; ++slot) {
            const auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            lane_frames_remaining[(size_t)slot] = slot_state.forward_ring_filled;
        }
    }

    const int active_span_limit = fftdiv_window_effective_span(state, wheel_length);

    if (stage2_unlocked) {
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                "[STAGE2-START] frame_index=%d wheel_head=%d wheel_tail=%d wheel_filled=%d wheel_len=%d active_span=%d\n",
                frame_index_int,
                wheel_head_cursor,
                wheel_tail_cursor,
                wheel_filled_cursor,
                wheel_length,
                active_span_limit);
        }
        while (true) {
            if (wheel_length > 0 && active_span_limit > 0 && wheel_filled_cursor >= active_span_limit) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                           "[STAGE2-BACKPRESSURE] waiting head=%d tail=%d filled=%d span=%d len=%d\n",
                           wheel_head_cursor,
                           wheel_tail_cursor,
                           wheel_filled_cursor,
                           active_span_limit,
                           wheel_length);
                break;
            }
            int ready_lane_count_iter = 0;

        for (int slot = 0; slot < slot_count; ++slot) {
            if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                           "[STAGE2-PROCESS] slot=%d frame=%lld\n",
                           slot,
                           (long long)(frame_counter + (int64_t)processed_passes));
            }
            auto &slot_state_iter = state->u.fftdiv.stream_slots[(size_t)slot];
            auto &lane_iter = state->u.fftdiv.lane_plan[(size_t)slot];

            if (!lane_iter.active) {
                lane_iter.frame_ready = false;
            } else {
                const size_t remaining_frames = (size_t)slot < lane_frames_remaining.size()
                    ? lane_frames_remaining[(size_t)slot]
                    : (size_t)slot_state_iter.forward_ring_filled;
                const bool ready_now = remaining_frames > 0U;
                lane_iter.frame_ready = ready_now;
                if (ready_now) {
                    ready_lane_count_iter += 1;
                }
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                               "[STAGE2-LANE] slot=%d active=%d ring_filled=%zu remaining=%zu frame_ready=%d\n",
                               slot,
                               lane_iter.active ? 1 : 0,
                               (size_t)slot_state_iter.forward_ring_filled,
                               remaining_frames,
                               ready_now ? 1 : 0);
                }
            }

        }

        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                       "[STAGE2-BARRIER] ready=%d active=%d satisfied=%d\n",
                       ready_lane_count_iter,
                       active_lane_count,
                       (ready_lane_count_iter == active_lane_count) ? 1 : 0);
        }

        const bool barrier_satisfied = (active_lane_count == 0)
            ? true
            : (ready_lane_count_iter == active_lane_count);

#if FFTDIV_TRACE_ENABLED
        FFTDIV_TRACE(
            "[fftdiv] barrier frame=%lld satisfied=%d ready=%d active=%d",
            (long long)(frame_counter + (int64_t)processed_passes),
            barrier_satisfied ? 1 : 0,
            ready_lane_count_iter,
            active_lane_count);
#endif

        if (!barrier_satisfied || active_lane_count == 0) {
            break;
        }

        const int tensor_slice_iter = wheel_head_cursor;
        const int scratch_slice_iter = scratch_cursor;
        const int64_t frame_index_iter = frame_counter + (int64_t)processed_passes;

        if (wheel_length > 0) {
            for (int slot = 0; slot < slot_count; ++slot) {
                auto &slot_state_snapshot = state->u.fftdiv.stream_slots[(size_t)slot];
                slot_state_snapshot.wheel_filled_pre_reserve = wheel_filled_cursor;
            }
            fftdiv_window_reserve_leading_edge(
                state,
                -1,
                (int)frame_index_iter,
                wheel_length,
                &wheel_tail_cursor,
                &wheel_filled_cursor);
        }

        bool iteration_updated = false;
        int view_filled_override = wheel_filled_cursor;

        if (working_tensor != NULL && tensor_freq_bins > 0) {
            for (int slot = 0; slot < slot_count; ++slot) {
                auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
                const size_t frame_offset = (size_t)slot < lane_frame_offsets.size()
                    ? lane_frame_offsets[(size_t)slot]
                    : 0U;
                const double *ring_real = fftdiv_ring_frame_real(slot_state, frame_offset, window_size);
                const double *ring_imag = fftdiv_ring_frame_imag(slot_state, frame_offset, window_size);
                double *slot_spectral_real = (ring_real != NULL)
                    ? const_cast<double *>(ring_real)
                    : slot_state.forward_real.data();
                double *slot_spectral_imag = (ring_imag != NULL)
                    ? const_cast<double *>(ring_imag)
                    : slot_state.forward_imag.data();
                auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
                if (!lane.active || !lane.frame_ready || slot_spectral_real == NULL || slot_spectral_imag == NULL) {
                    continue;
                }

                /* Stage 2 (Spectral In): summarize a few nonzero bins from the ring frame */
                {
                    int hits = 0;
                    const int max_hits = 8;
                    for (int f = 0; f < window_size && hits < max_hits; ++f) {
                        const double re = slot_spectral_real[f];
                        const double im = slot_spectral_imag[f];
                        if (re != 0.0 || im != 0.0) {
                            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                       "[STAGE2-SPEC-IN] slot=%d slice=%d bin=%d re=%.6g im=%.6g\n",
                                       slot, tensor_slice_iter, f, re, im);
                            ++hits;
                        }
                    }
                    if (hits == 0) {
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                   "[STAGE2-SPEC-IN] slot=%d slice=%d all-zero\n",
                                   slot, tensor_slice_iter);
                    }
                }

                const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
                double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice_iter);
                double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice_iter);
                const int scratch_bins = fftdiv_spectral_scratch_bins(state);

                /* Prefer ring frame as source unless operator wrote to scratch */
                const double *commit_real = slot_spectral_real;
                const double *commit_imag = slot_spectral_imag;
                int commit_bins = window_size;
                if (!state->u.fftdiv.operator_steps.empty() &&
                    scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    commit_real = scratch_real;
                    commit_imag = scratch_imag;
                    commit_bins = scratch_bins;
                }

                fftdiv_copy_spectrum_to_working(
                    working_tensor,
                    tensor_page,
                    tensor_lane,
                    tensor_slice_iter,
                    tensor_freq_bins,
                    commit_real,
                    commit_imag,
                    commit_bins);
                /* TODO: introduce interpolation when tensor_freq_bins != commit_bins. */
                if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                }

                iteration_updated = true;

            }

            if (iteration_updated) {
                if (wheel_filled_cursor >= wheel_length) {
                    view_filled_override = wheel_length;
                } else {
                    int hop_for_span = (wheel_hop > 0) ? wheel_hop : 1;
                    view_filled_override = wheel_filled_cursor + hop_for_span;
                    if (view_filled_override > wheel_length) {
                        view_filled_override = wheel_length;
                    }
                }
            }
        }

        const size_t next_slice_index = filled_slices.size();
        FftDivFilledSlice slice;
        slice.tensor_slice = tensor_slice_iter;
        slice.scratch_slice = scratch_slice_iter;
        slice.view_filled_override = view_filled_override;
        slice.wheel_head = wheel_head_cursor;
        slice.wheel_tail = wheel_tail_cursor;
        slice.frame_index = frame_index_iter;
        slice.pcm_sample_index = (int64_t)base_index + frame_index_iter * (int64_t)slot_count;
        slice.slice_index = next_slice_index;
        slice.timeline_seconds = timeline_cursor;
        slice.working_tensor_updated = iteration_updated;
        slice.lanes.resize(static_cast<size_t>(slot_count));
        slice.lane_frame_offsets.resize(static_cast<size_t>(slot_count), 0U);

        for (int slot = 0; slot < slot_count; ++slot) {
            auto &lane_meta = state->u.fftdiv.lane_plan[(size_t)slot];
            auto &lane_snapshot = slice.lanes[(size_t)slot];
            lane_snapshot.frame_ready = lane_meta.frame_ready;
            if ((size_t)slot < slice.lane_frame_offsets.size()) {
                const size_t frame_offset = (size_t)slot < lane_frame_offsets.size()
                    ? lane_frame_offsets[(size_t)slot]
                    : 0U;
                slice.lane_frame_offsets[(size_t)slot] = frame_offset;
            }

            lane_meta.frame_ready = false;

            if (lane_snapshot.frame_ready && (size_t)slot < lane_frame_offsets.size()) {
                if (lane_frames_remaining[(size_t)slot] > 0U) {
                    lane_frames_remaining[(size_t)slot] -= 1U;
                }
                lane_frame_offsets[(size_t)slot] += 1U;
                // Stage 2: Consume from forward_ring when ingesting into wheel
                auto &slot_state_consume = state->u.fftdiv.stream_slots[(size_t)slot];
                if (slot_state_consume.forward_ring_filled > 0U && slot_state_consume.forward_ring_capacity_frames > 0U) {
                    slot_state_consume.forward_ring_read =
                        (slot_state_consume.forward_ring_read + 1U) % slot_state_consume.forward_ring_capacity_frames;
                    slot_state_consume.forward_ring_filled -= 1U;
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE2-CONSUME] slot=%d read=%zu filled=%zu remaining=%zu\n",
                               slot,
                               (size_t)slot_state_consume.forward_ring_read,
                               (size_t)slot_state_consume.forward_ring_filled,
                               lane_frames_remaining[(size_t)slot]);
                }
            }
        }

        if (wheel_length > 0) {
            auto &meta = state->u.fftdiv.wheel_slice_metadata;
            if (meta.size() < (size_t)wheel_length) {
                meta.resize((size_t)wheel_length);
            }
            FftDivFilledSlice &slot_meta = meta[(size_t)tensor_slice_iter];
            slot_meta = std::move(slice);
            slot_meta.valid = true;
            slot_meta.stage4_emitted = false;
            filled_slices.push_back(&slot_meta);
        } else {
            filled_slices.push_back(nullptr);
        }

            processed_passes += 1U;
            timeline_cursor += hop_seconds;

            if (wheel_length > 0) {
                /* Spectral emission index must advance by 1 per output frame,
                   independent of any working-hop used by Stage 3 operators. */
                wheel_head_cursor = (wheel_head_cursor + 1) % wheel_length;
                if (wheel_filled_cursor < wheel_length) {
                    wheel_filled_cursor += 1;
                }
            }

            if (scratch_time_slices > 0) {
                scratch_cursor += 1;
                if (scratch_cursor >= scratch_time_slices) {
                    scratch_cursor = 0;
                }
            }
        }
    } else {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                   "[STAGE2-SKIP] snapshot locked, deferring wheel ingestion\n");
    }

    if (!stage3_unlocked) {
        metrics_window_span = 0;
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE3-SKIP] snapshot locked, deferring operator execution\n");
        }
    } else {
        // Stage3: iterate wheel metadata in FIFO order from tail, stopping after one hop span
        auto &meta = state->u.fftdiv.wheel_slice_metadata;
        const int stop_count = (wheel_hop > 0) ? wheel_hop : 1;
        int visited = 0;
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE3-START] FIFO walk tail=%d hop=%d len=%d stop_count=%d\n",
                       wheel_tail_cursor, wheel_hop, wheel_length, stop_count);
        }
        for (int i = 0; i < stop_count && wheel_length > 0; ++i) {
            const int idx = (wheel_tail_cursor + i) % wheel_length;
            if ((size_t)idx >= meta.size()) {
                break;
            }
            FftDivFilledSlice *slice_ptr = meta[idx].valid ? &meta[idx] : nullptr;
            visited += 1;
            if (slice_ptr == nullptr) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE3-SKIP] idx=%d null entry\n",
                               idx);
                }
                metrics_window_span = 0;
                continue;
            }
            if (slice_ptr->stage4_emitted) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE3-SKIP] idx=%d already emitted\n",
                               idx);
                }
                continue;
            }
            if (!slice_ptr->working_tensor_updated) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE3-SKIP] idx=%d tensor not updated (tensor_slice=%d)\n",
                               idx,
                               slice_ptr->tensor_slice);
                }
                metrics_window_span = 0;
                continue;
            }
            auto &slice = *slice_ptr;

            FftDivActiveWindowView active_window_view = fftdiv_build_active_window_view_internal(
                state,
                working_tensor,
                tensor_page,
                slice.tensor_slice,
                slice.view_filled_override);

            if (active_window_view.valid()) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    const int max_hits_per_slice = state->u.fftdiv.log_slice_bin_cap;
                    if (max_hits_per_slice > 0) {
                        const int tspan = active_window_view.window_span;
                        const int lanes = active_window_view.lanes;
                        const int bins = active_window_view.freq_bins;
                        const size_t lane_stride = active_window_view.lane_stride();
                        const size_t freq_stride = active_window_view.freq_stride();
                        for (int t = 0; t < tspan; ++t) {
                            std::complex<double> *tbase = active_window_view.time_slice(t);
                            if (tbase == nullptr) {
                                continue;
                            }
                            int hits = 0;
                            for (int lane = 0; lane < lanes && hits < max_hits_per_slice; ++lane) {
                                for (int f = 0; f < bins && hits < max_hits_per_slice; ++f) {
                                    std::complex<double> *ptr = tbase + (size_t)lane * lane_stride + (size_t)f * freq_stride;
                                    const double re = ptr->real();
                                    const double im = ptr->imag();
                                    if (re != 0.0 || im != 0.0) {
                                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                                   "[STAGE3-OP] t=%d lane=%d bin=%d re=%.6g im=%.6g\n",
                                                   t, lane, f, re, im);
                                        ++hits;
                                    }
                                }
                            }
                            if (hits == 0) {
                                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                           "[STAGE3-OP] t=%d no-nonzero\n", t);
                            }
                        }
                    }
                }
                FftDivOperatorContext operator_context;
                operator_context.window = active_window_view;
                operator_context.lanes = operator_lane_bindings.empty()
                    ? nullptr
                    : operator_lane_bindings.data();
                operator_context.lane_count = operator_lane_bindings.size();
                operator_context.hop = wheel_hop;
                operator_context.window_size = window_size;
                operator_context.wheel_length = wheel_length;
                operator_context.wheel_head = slice.wheel_head;
                operator_context.wheel_tail = slice.wheel_tail;
                operator_context.frame_index = slice.frame_index;
                operator_context.pcm_sample_index = slice.pcm_sample_index;
                operator_context.slice_index = slice.slice_index;
                operator_context.pcm_sample_stride = static_cast<size_t>(slot_count);
                operator_context.sample_rate = effective_sample_rate;
                operator_context.timeline_seconds = slice.timeline_seconds;
                operator_context.hop_seconds = hop_seconds;
                fftdiv_run_operator_stack(state, operator_context);
                metrics_window_span = active_window_view.window_span;

                const int release_tensor_slice = fftdiv_last_hop_slice_from_read_cursor(
                    slice.wheel_tail,
                    wheel_length,
                    wheel_hop,
                    slice.tensor_slice);

                for (int slot = 0; slot < slot_count; ++slot) {
                    auto &lane_meta = state->u.fftdiv.lane_plan[(size_t)slot];
                    auto &lane_snapshot = slice.lanes[(size_t)slot];
                    if (!lane_meta.active || !lane_snapshot.frame_ready) {
                        lane_snapshot.spectral_real.clear();
                        lane_snapshot.spectral_imag.clear();
                        continue;
                    }
                    const int tensor_lane = (lane_meta.tensor_lane >= 0) ? lane_meta.tensor_lane : slot;
                    try {
                        lane_snapshot.spectral_real.assign((size_t)window_size, 0.0);
                        lane_snapshot.spectral_imag.assign((size_t)window_size, 0.0);
                    } catch (...) {
                        lane_snapshot.spectral_real.clear();
                        lane_snapshot.spectral_imag.clear();
                        continue;
                    }
                    fftdiv_copy_working_to_slot_buffer(
                        working_tensor,
                        tensor_page,
                        tensor_lane,
                        release_tensor_slice,
                        tensor_freq_bins,
                        lane_snapshot.spectral_real.data(),
                        lane_snapshot.spectral_imag.data(),
                        window_size);
                }
            } else {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE3-SKIP] idx=%d tensor_slice=%d invalid active window\n",
                               idx,
                               slice.tensor_slice);
                }
                metrics_window_span = 0;
            }
        }
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE3-END] visited=%d\n",
                       visited);
        }
    }

    if (stage4_unlocked) {
        // Stage4: iterate wheel metadata in FIFO order from tail, stopping after one hop span
        auto &meta = state->u.fftdiv.wheel_slice_metadata;
        const int stop_count = (wheel_hop > 0) ? wheel_hop : 1;
        int visited = 0;
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE4-START] FIFO walk tail=%d hop=%d len=%d stop_count=%d\n",
                       wheel_tail_cursor, wheel_hop, wheel_length, stop_count);
        }
        std::vector<size_t> lane_frames_consumed(static_cast<size_t>(slot_count), 0U);
        for (int i = 0; i < stop_count && wheel_length > 0; ++i) {
            const int idx = (wheel_tail_cursor + i) % wheel_length;
            if ((size_t)idx >= meta.size()) {
                break;
            }
            FftDivFilledSlice *slice_ptr = meta[idx].valid ? &meta[idx] : nullptr;
            visited += 1;
            if (slice_ptr == nullptr) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE4-SKIP] idx=%d null entry\n",
                               idx);
                }
                continue;
            }
            auto &slice = *slice_ptr;
            if (slice.stage4_emitted) {
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE4-SKIP] idx=%d already emitted\n",
                               idx);
                }
                continue;
            }
            if (!slice.working_tensor_updated) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                           "[STAGE4-SKIP] idx=%d tensor slice=%d pending operator update\n",
                           idx, slice.tensor_slice);
                continue;
            }
            /* Debug: correlate Stage2 slice index with Stage3 window span and offsets */
            if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                           "[STAGE3-CONTEXT] tensor_slice=%d view_filled=%d wheel(head=%d tail=%d len=%d hop=%d) frame=%lld\n",
                           slice.tensor_slice,
                           slice.view_filled_override,
                           slice.wheel_head,
                           slice.wheel_tail,
                           wheel_length,
                           wheel_hop,
                           (long long)slice.frame_index);
                if (!slice.lane_frame_offsets.empty()) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE3-CONTEXT] lane0 assigned_offset=%zu\n",
                               slice.lane_frame_offsets[0]);
                }
            }
            const int tensor_slice_iter = slice.tensor_slice;
            const int scratch_slice_iter = slice.scratch_slice;
            const int64_t frame_index_iter = slice.frame_index;
            const size_t base_output_index = (slice.pcm_sample_index >= 0)
                ? (size_t)slice.pcm_sample_index
                : 0U;

            for (int slot = 0; slot < slot_count; ++slot) {
                size_t data_idx = base_output_index + (size_t)slot;
                if (data_idx >= (size_t)slot_count * (size_t)frames) {
                    break;
                }
                auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
                const size_t assigned_offset = (slice.lane_frame_offsets.size() > (size_t)slot)
                    ? slice.lane_frame_offsets[(size_t)slot]
                    : 0U;
                const size_t consumed_so_far = (lane_frames_consumed.size() > (size_t)slot)
                    ? lane_frames_consumed[(size_t)slot]
                    : 0U;
                const size_t relative_offset = (assigned_offset >= consumed_so_far)
                    ? (assigned_offset - consumed_so_far)
                    : 0U;
                auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
                auto &lane_snapshot = slice.lanes[(size_t)slot];
                const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
                const double *released_real = lane_snapshot.spectral_real.empty()
                    ? nullptr
                    : lane_snapshot.spectral_real.data();
                const double *released_imag = lane_snapshot.spectral_imag.empty()
                    ? nullptr
                    : lane_snapshot.spectral_imag.data();

                if (!(lane_snapshot.frame_ready && lane.active)) {
                    if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                   "[STAGE4-SKIP] idx=%d slot=%d frame_ready=%d lane_active=%d\n",
                                   idx,
                                   slot,
                                   lane_snapshot.frame_ready ? 1 : 0,
                                   lane.active ? 1 : 0);
                    }
                }
                if (!(released_real != nullptr && released_imag != nullptr)) {
                    if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                   "[STAGE4-SKIP] idx=%d slot=%d missing spectral buffers\n",
                                   idx,
                                   slot);
                    }
                }
                if (lane_snapshot.frame_ready && lane.active && released_real != nullptr && released_imag != nullptr) {
                    if (lane.enable_spectral_out) {
                        /* Emit spectral row aligned to PCM time: write at (frame + window_size - 1) */
                        const int emit_frame_index = (int)slice.frame_index + ((window_size > 0) ? (window_size - 1) : 0);
                        stage_emit_spectral(
                            state,
                            const_cast<double *>(released_real),
                            const_cast<double *>(released_imag),
                            window_size,
                            slot,
                            emit_frame_index,
                            working_tensor,
                            tensor_page,
                            tensor_lane,
                            tensor_slice_iter,
                            tensor_freq_bins);
                        /* Stage 4 (Spectral Out): we just emitted to spectral mailbox for this frame */
                        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                                   "[STAGE4-SPEC-OUT] slot=%d frame=%lld emit_row=%d bins=%d\n",
                                   slot, (long long)slice.frame_index, emit_frame_index, window_size);
                    }

                    if (lane.enable_pcm_out) {
                        // Stage 3 enqueues spectrum for deferred inverse in Stage 5
                        if (released_real != nullptr && released_imag != nullptr) {
                            decltype(slot_state.pending_spectra)::value_type pending;
                            try {
                                pending.real.assign(released_real, released_real + (size_t)window_size);
                                pending.imag.assign(released_imag, released_imag + (size_t)window_size);
                                slot_state.pending_spectra.push_back(std::move(pending));
                            } catch (...) {
                                // Allocation failure: drop this frame; Stage 5 will still drain what exists
                            }
                        }
                    }
                }
            }
            slice.stage4_emitted = true;
            if (wheel_length > 0) {
                fftdiv_window_release_tail(
                    state,
                    wheel_length,
                    &wheel_tail_cursor,
                    &wheel_filled_cursor,
                    "stage4-inline");
                if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                               "[STAGE4-TAIL] idx=%d new_tail=%d filled=%d\n",
                               idx,
                               wheel_tail_cursor,
                               wheel_filled_cursor);
                }
            }
        }
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE4-END] visited=%d\n",
                       visited);
        }
        // Write back cursors to global state so Stage 2 sees the updated capacity on next pass
        wheel_tail = wheel_tail_cursor;
        wheel_filled = wheel_filled_cursor;
    } else {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                   "[STAGE4-SKIP] snapshot locked, deferring spectral emission\n");
    }

        wheel_head = wheel_head_cursor;
        wheel_tail = wheel_tail_cursor;
        wheel_filled = wheel_filled_cursor;
        scratch_time_cursor = scratch_cursor;
        state->u.fftdiv.wheel_frame_counter += (int64_t)processed_passes;
        state->u.fftdiv.timeline_seconds = timeline_cursor;
    #endif /* defined(__cplusplus) */

    #if defined(__cplusplus)
    // Stage 5: Owns spectral -> PCM conversion (ISTFT) plus overlap-add retention; logs must only mention PCM work
    
    size_t buffer_offset = 0;
    for (int slot = 0; slot < slot_count; ++slot) {
        auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
        // First, run inverse on spectra accumulated by Stage 3
        while (!slot_state.pending_spectra.empty() && slot_state.inverse_handle != NULL && !slot_state.inverse_scratch.empty()) {
            auto &pending = slot_state.pending_spectra.front();
            const size_t produced_pcm = amp_fft_backend_stream_push_spectrum(
                slot_state.inverse_handle,
                pending.real.empty() ? NULL : pending.real.data(),
                pending.imag.empty() ? NULL : pending.imag.data(),
                1,
                window_size,
                slot_state.inverse_scratch.data(),
                slot_state.inverse_scratch.size(),
                flush_mode);
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                       "[STAGE5-PCM-ISTFT] slot=%d produced_pcm=%zu\n",
                       slot,
                       produced_pcm);
            if (produced_pcm > 0) {
                made_progress = true;
            }
            for (size_t i = 0; i < produced_pcm && i < slot_state.inverse_scratch.size(); ++i) {
                slot_state.inverse_queue.push_back(slot_state.inverse_scratch[i]);
            }
            slot_state.pending_spectra.pop_front();
        }
        while (!slot_state.inverse_queue.empty() && buffer_offset < total_samples) {
            buffer[buffer_offset] = slot_state.inverse_queue.front();
            slot_state.inverse_queue.pop_front();
                  FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                      "[STAGE5-PCM-QUEUE] slot=%d buffer_idx=%zu value=%.6g queue_remaining=%zu\n",
                       slot,
                       buffer_offset,
                       buffer[buffer_offset],
                       slot_state.inverse_queue.size());
            buffer_offset++;
            pcm_written_samples++;
        }
    }

    if (wheel_length > 0) {
        fftdiv_window_release_tail(
            state,
            wheel_length,
            &wheel_tail,
            &wheel_filled,
            "stage5-post");
    }

#endif

#if !defined(__cplusplus)
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = 0.0;
            if (audio_base != NULL) {
                sample = audio_base[data_idx];
            }
            buffer[data_idx] = sample;
        }
#endif /* !defined(__cplusplus) */
    state->u.fftdiv.wheel_length = wheel_length;
    state->u.fftdiv.wheel_head = wheel_head;
    state->u.fftdiv.wheel_tail = wheel_tail;
    state->u.fftdiv.wheel_filled_slices = wheel_filled;
    state->u.fftdiv.wheel_hop = wheel_hop;
#if defined(__cplusplus)
    state->u.fftdiv.spectral_scratch.time_cursor = scratch_time_cursor;
#endif
    state->u.fftdiv.execute_snapshot.wheel_head = wheel_head;
    state->u.fftdiv.execute_snapshot.wheel_tail = wheel_tail;
    state->u.fftdiv.execute_snapshot.wheel_filled = wheel_filled;
    state->u.fftdiv.execute_snapshot.wheel_length = wheel_length;
    state->u.fftdiv.execute_snapshot.wheel_hop = wheel_hop;
    state->u.fftdiv.execute_snapshot.scratch_time_cursor = scratch_time_cursor;
    state->u.fftdiv.execute_snapshot.timeline_seconds = state->u.fftdiv.timeline_seconds;

    // Push buffer to mailbox instead of direct return
    if (metrics != NULL) {
        const size_t derived_latency = fftdiv_declared_latency_frames(state);
        metrics->measured_delay_frames = static_cast<uint32_t>(derived_latency);
        metrics->accumulated_heat = 0.0f;
        metrics->processing_time_seconds = 0.0;
        metrics->logging_time_seconds = 0.0;
        metrics->total_time_seconds = 0.0;
        metrics->thread_cpu_time_seconds = 0.0;
        metrics->reserved[0] = (double)state->u.fftdiv.wheel_active_window_span;
        metrics->reserved[1] = (double)metrics_window_span;
        metrics->reserved[2] = (double)wheel_filled;
        metrics->reserved[3] = (double)wheel_length;
        metrics->reserved[4] = (double)window_size;
        metrics->reserved[5] = (double)default_algorithm;
    }
    
    // Push one mailbox entry per frame (only push what was actually written)
    size_t frames_to_push = pcm_written_samples;
    if (pcm_written_samples == 0 || pcm_written_samples > 1000) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[STAGE5-PCM-DISPATCH] pcm_written=%zu frames_to_push=%zu ANOMALY\n",
                   pcm_written_samples, frames_to_push);
    }
    for (size_t i = 0; i < frames_to_push; ++i) {
        double *frame_buffer = (double *)malloc(input_channels * sizeof(double));
        if (frame_buffer == NULL) {
            free(buffer);
            return -1;
        }
        for (int ch = 0; ch < input_channels; ++ch) {
            frame_buffer[ch] = buffer[i * input_channels + ch];
        }
        AmpMailboxEntry *entry = amp_mailbox_entry_create(
            frame_buffer,
            input_channels,
            1,  // One frame per entry
            0,
            metrics,
            NULL,
            NULL
        );
        if (entry != NULL) {
            amp_node_mailbox_push(state, entry);
        } else {
            free(frame_buffer);
            free(buffer);
            return -1;
        }
    }
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
               "[STAGE5-PCM-EMIT] pushed %zu individual frame entries\n",
               frames_to_push);
    // Per-round pipeline instrumentation: PCM emitted and queue depths across stages
    {
        int unemitted_slices = 0;
        for (const auto &slice : state->u.fftdiv.wheel_slice_metadata) {
            if (slice.valid && !slice.stage4_emitted) {
                unemitted_slices += 1;
            }
        }
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                   "[PIPELINE-ROUND] pcm_out=%zu wheel_filled=%d unemitted=%d\n",
                   frames_to_push,
                   state->u.fftdiv.wheel_filled_slices,
                   unemitted_slices);
        for (size_t s = 0; s < state->u.fftdiv.stream_slots.size(); ++s) {
            const auto &slot = state->u.fftdiv.stream_slots[s];
            const size_t pending_pcm = (slot.pcm_backlog.size() > slot.pcm_consumed_samples)
                ? (slot.pcm_backlog.size() - slot.pcm_consumed_samples)
                : 0U;
            const size_t ring_filled = slot.forward_ring_filled;
            const size_t spectra_pending = slot.pending_spectra.size();
            const size_t pcm_queue = slot.inverse_queue.size();
            size_t s34_unemitted = 0;
            for (const auto &slice : state->u.fftdiv.wheel_slice_metadata) {
                if (!slice.valid || slice.stage4_emitted) continue;
                if (slice.lanes.size() > s) {
                    const auto &lane_snap = slice.lanes[s];
                    if (lane_snap.frame_ready) {
                        s34_unemitted += 1;
                    }
                }
            }
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                       "[PIPELINE-SLOT] idx=%zu s1_pcm=%zu s1_ring=%zu s34_unem=%zu s5_spec=%zu s5_pcmq=%zu\n",
                       s,
                       pending_pcm,
                       ring_filled,
                       s34_unemitted,
                       spectra_pending,
                       pcm_queue);
        }
    }
    free(buffer);
    
    *out_buffer = NULL;
    *out_channels = 0;
    return AMP_E_PENDING;
}

#if defined(__cplusplus)
static bool fftdiv_stage_snapshot_any_ready(const FftDivStageLockSnapshot &snapshot) {
    return snapshot.stage1_ready
        || snapshot.stage2_ready
        || snapshot.stage3_ready
        || snapshot.stage4_ready
        || snapshot.stage5_ready;
}

static FftDivStageLockSnapshot fftdiv_snapshot_stage_locks(const node_state_t *state) {
    FftDivStageLockSnapshot snapshot = {0};
    if (state == nullptr) {
        return snapshot;
    }

    const int window_size = (state->u.fftdiv.window_size > 0) ? state->u.fftdiv.window_size : 0;
    const auto &slots = state->u.fftdiv.stream_slots;

    // Stage 1 (PCM -> FFT): requires either new PCM input or existing FFT frames to deliver,
    // and only blocks when there is no downstream capacity.
    for (const auto &slot : slots) {
        const bool ring_has_capacity = (slot.forward_ring_capacity_frames == 0U)
            || (slot.forward_ring_filled < slot.forward_ring_capacity_frames);
        const size_t pending_samples = (slot.pcm_backlog.size() > slot.pcm_consumed_samples)
            ? (slot.pcm_backlog.size() - slot.pcm_consumed_samples)
            : 0U;
        const bool has_fft_pending = slot.forward_ring_filled > 0U;
        const bool has_input = (pending_samples > 0U) || has_fft_pending;
        if (slot.forward_handle != nullptr && has_input && ring_has_capacity) {
            snapshot.stage1_ready = true;
            break;
        }
    }

    // Stage 2 (working window ingestion) : requires wheel capacity and every active
    // lane presenting at least one spectrum frame.
    int wheel_length = state->u.fftdiv.wheel_length;
    if (wheel_length <= 0) {
        wheel_length = state->u.fftdiv.working_tensor_time_slices;
    }
    if (wheel_length < 0) {
        wheel_length = 0;
    }
    const int active_span_limit = fftdiv_window_effective_span(state, wheel_length);
    int wheel_filled_for_gate = state->u.fftdiv.wheel_filled_slices;
#if defined(__cplusplus)
    bool have_prereserve_snapshot = false;
    for (const auto &slot : slots) {
        if (slot.wheel_filled_pre_reserve >= 0) {
            if (!have_prereserve_snapshot || slot.wheel_filled_pre_reserve < wheel_filled_for_gate) {
                wheel_filled_for_gate = slot.wheel_filled_pre_reserve;
            }
            have_prereserve_snapshot = true;
        }
    }
#endif
    const bool wheel_has_capacity = (wheel_length <= 0)
        || (active_span_limit <= 0)
        || (wheel_filled_for_gate < active_span_limit);

    bool lanes_have_frames = true;
    int active_lane_count = 0;
    const size_t lane_plan_count = state->u.fftdiv.lane_plan.size();
    for (size_t idx = 0; idx < lane_plan_count && idx < slots.size(); ++idx) {
        const auto &lane = state->u.fftdiv.lane_plan[idx];
        if (!lane.active) {
            continue;
        }
        active_lane_count += 1;
        if (slots[idx].forward_ring_filled == 0U) {
            lanes_have_frames = false;
            break;
        }
    }
    // Previous logic treated zero active lanes as "ready" which kept the worker loop
    // reporting stage2 readiness after full drain. Require at least one active lane.
    snapshot.stage2_ready = wheel_has_capacity && (active_lane_count > 0) && lanes_have_frames;

    // Stage 3/4 (operator and spectral emission) : ready while slices exist that
    // have not yet been emitted downstream.
    for (const auto &slice : state->u.fftdiv.wheel_slice_metadata) {
        if (slice.valid && !slice.stage4_emitted) {
            snapshot.stage3_ready = true;
            snapshot.stage4_ready = true;
            break;
        }
    }

    // Stage 5 (ISTFT/output): ready when either spectra await inversion or PCM is queued.
    for (const auto &slot : slots) {
        if (!slot.pending_spectra.empty()) {
            snapshot.stage5_ready = true;
            break;
        }
        if (!slot.inverse_queue.empty()) {
            snapshot.stage5_ready = true;
            break;
        }
        if (slot.inverse_handle != nullptr
            && amp_fft_backend_stream_pending_pcm(slot.inverse_handle) > 0U) {
            snapshot.stage5_ready = true;
            break;
        }
    }

    return snapshot;
}

static bool fftdiv_pipeline_can_advance(const node_state_t *state) {
    return fftdiv_stage_snapshot_any_ready(fftdiv_snapshot_stage_locks(state));
}

static bool fftdiv_stage1_zero_insertion_active(const node_state_t *state) {
    if (state == nullptr) {
        return false;
    }
    bool zero_tail_enqueued = false;
    for (const auto &slot : state->u.fftdiv.stream_slots) {
        if (slot.zero_tail_enqueued) {
            zero_tail_enqueued = true;
            break;
        }
    }
    if (!zero_tail_enqueued) {
        return false;
    }
    const FftDivStageLockSnapshot snapshot = fftdiv_snapshot_stage_locks(state);
    return fftdiv_stage_snapshot_any_ready(snapshot);
}


static void fftdiv_flush_with_zeroes(node_state_t *state) {
    if (state == nullptr) {
        FFTDIV_LOG(nullptr, FFTDIV_LOG_LEVEL_SUMMARY,
                   "[FFT-LATENCY-FLUSH] skip state=null reason=null-state\n");
        return;
    }
    auto &fftdiv = state->u.fftdiv;
    const EdgeRunnerNodeDescriptor *descriptor = fftdiv.last_descriptor;
    if (descriptor == nullptr) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
                   "[FFT-LATENCY-FLUSH] skip state=%p reason=no-descriptor slots=%zu\n",
                   (void *)state,
                   static_cast<size_t>(fftdiv.stream_slots.size()));
        return;
    }
    if (fftdiv.stream_slots.empty()) {
        FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
                   "[FFT-LATENCY-FLUSH] skip state=%p reason=no-slots descriptor=%s\n",
                   (void *)state,
                   descriptor->name != nullptr ? descriptor->name : "<unnamed>");
        return;
    }
    const FftDivStageLockSnapshot stage_snapshot = fftdiv_snapshot_stage_locks(state);
    const bool pipeline_active = fftdiv_stage_snapshot_any_ready(stage_snapshot);
    int batches = fftdiv.last_batches > 0 ? fftdiv.last_batches : 1;
    int channels = fftdiv.last_channels > 0 ? fftdiv.last_channels : 1;
    const FftDivTailPlan tail_plan = fftdiv_calculate_tail_plan(state, pipeline_active);
    const size_t frame0_delay = tail_plan.pcm_frames;
    int frames = (frame0_delay > 0U)
        ? static_cast<int>(frame0_delay)
        : 1;
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[FFT-LATENCY-FLUSH-COMPUTE] W_fft=%d H_fft=%d W_work=%d H_work=%d L_istft=%d frame0_delay=%zu inject_pcm=%d\n",
               tail_plan.window_size,
               tail_plan.backend_hop,
               tail_plan.working_span,
               tail_plan.working_hop,
               tail_plan.istft_tail_frames,
               frame0_delay,
               frames);
    int slot_count = fftdiv.last_slot_count > 0 ? fftdiv.last_slot_count : batches * channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }
    const double sample_rate = (fftdiv.last_sample_rate > 0.0)
        ? fftdiv.last_sample_rate
        : fftdiv.sample_rate_hint;

    std::vector<double> zeros(
        static_cast<size_t>(slot_count) * static_cast<size_t>(frames),
        0.0
    );

    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[FFT-ANALYTIC-FLUSH] start state=%p slots=%zu frames_pcm=%d pending=%s\n",
               (void *)state,
               static_cast<size_t>(fftdiv.stream_slots.size()),
               frames,
               pipeline_active ? "yes" : "no");

    EdgeRunnerAudioView audio{};
    audio.has_audio = 1;
    audio.batches = static_cast<uint32_t>(batches);
    audio.channels = static_cast<uint32_t>(channels);
    audio.frames = frames;
    audio.data = zeros.data();

    EdgeRunnerParamSet params{};
    EdgeRunnerNodeInputs flush_inputs{};
    flush_inputs.audio = audio;
    flush_inputs.params = params;
    flush_inputs.taps.outputs.items = nullptr;
    flush_inputs.taps.outputs.count = 0;
    flush_inputs.taps.status.items = nullptr;
    flush_inputs.taps.status.count = 0;

    // Analytic zero-injection: push zeros as normal input, no flush enum needed
    double *tmp_buf = nullptr;
    int tmp_ch = 0;
    const size_t runtime_pcm_block = frames > 0 ? static_cast<size_t>(frames) : 1U;

    if (tmp_buf != nullptr) {
        free(tmp_buf);
    }
    
}

static int fftdiv_wait_for_completion(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    node_state_t *state,
    double **out_buffer,
    int *out_channels,
    AmpNodeMetrics *metrics
) {
    if (state == nullptr || out_buffer == nullptr || out_channels == nullptr) {
        return -1;
    }
    *out_buffer = nullptr;
    *out_channels = 0;
    const auto stage_snapshot = fftdiv_snapshot_stage_locks(state);
    if (!fftdiv_stage_snapshot_any_ready(stage_snapshot)) {
        return AMP_E_PENDING;
    }

    auto &fftdiv = state->u.fftdiv;
    const EdgeRunnerNodeDescriptor *active_descriptor = descriptor;
    if (active_descriptor == nullptr) {
        active_descriptor = fftdiv.last_descriptor;
    }
    if (active_descriptor == nullptr) {
        return -1;
    }

    int effective_batches = (batches > 0) ? batches : fftdiv.last_batches;
    if (effective_batches <= 0) {
        effective_batches = 1;
    }
    int effective_channels = (channels > 0) ? channels : fftdiv.last_channels;
    if (effective_channels <= 0) {
        effective_channels = 1;
    }
    int effective_frames = (frames > 0) ? frames : fftdiv.last_frames;
    if (effective_frames <= 0) {
        effective_frames = fftdiv.window_size > 0 ? fftdiv.window_size : 1;
    }
    int slot_count = effective_batches * effective_channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }

    const double effective_sample_rate = (sample_rate > 0.0)
        ? sample_rate
        : (fftdiv.last_sample_rate > 0.0 ? fftdiv.last_sample_rate : fftdiv.sample_rate_hint);

    EdgeRunnerAudioView audio{};
    audio.has_audio = 0;
    audio.batches = static_cast<uint32_t>(effective_batches);
    audio.channels = static_cast<uint32_t>(effective_channels);
    audio.frames = static_cast<uint32_t>(effective_frames);
    audio.data = nullptr;

    EdgeRunnerParamSet params{};
    EdgeRunnerNodeInputs flush_inputs{};
    flush_inputs.audio = audio;
    if (inputs != nullptr) {
        flush_inputs.params = inputs->params;
        flush_inputs.taps = inputs->taps;
    } else {
        flush_inputs.params = params;
        flush_inputs.taps.outputs.items = nullptr;
        flush_inputs.taps.outputs.count = 0;
        flush_inputs.taps.status.items = nullptr;
        flush_inputs.taps.status.count = 0;
    }

    // Single call as normal input - no flush enum needed
    double *buffer = nullptr;
    int produced_channels = 0;
    const size_t runtime_pcm_block = effective_frames > 0 ? static_cast<size_t>(effective_frames) : 1U;
    *out_buffer = buffer;
    *out_channels = produced_channels;
    return -1;
}

static void fftdiv_worker_publish_entry(
    node_state_t *state,
    double *buffer,
    int channels,
    int frames,
    int status,
    const AmpNodeMetrics *metrics
) {
    if (state == nullptr) {
        if (buffer != nullptr) {
            amp_free(buffer);
        }
        return;
    }
    size_t frame_count = frames > 0 ? static_cast<size_t>(frames) : 0U;
    AmpMailboxEntry *entry = amp_mailbox_entry_create(
        buffer,
        channels,
        frame_count,
        status,
        metrics,
        nullptr,
        nullptr
    );
    if (entry == nullptr) {
        if (buffer != nullptr) {
            amp_free(buffer);
        }
        return;
    }
    amp_node_mailbox_push(state, entry);
}

static bool fftdiv_worker_process_command(
    node_state_t *state,
    const std::shared_ptr<FftDivWorkerCommand> &command,
    const FftDivStageLockSnapshot *stage_locks
) {
    if (!state || !command) {
        return true;
    }
    AmpNodeMetrics local_metrics{};
    AmpNodeMetrics *metrics_ptr = command->want_metrics ? &command->metrics_storage : &local_metrics;
    if (command->want_metrics) {
        command->metrics_storage = AmpNodeMetrics{};
    } else {
        local_metrics = AmpNodeMetrics{};
        metrics_ptr = &local_metrics;
    }

    double *buffer = nullptr;
    int out_channels = 0;
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[STAGE0-WORKER] Processing command frames=%d\n",
               command->task.frames);
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[STAGE0-WORKER] input.has_audio=0x%08x frames=%u data=%p\n",
               (unsigned)(command->task.inputs && command->task.inputs->audio.has_audio ? command->task.inputs->audio.has_audio : 0U),
               (unsigned)(command->task.inputs ? command->task.inputs->audio.frames : 0U),
               (void*)(command->task.inputs ? command->task.inputs->audio.data : nullptr));
    const size_t runtime_pcm_block = command->task.frames > 0
        ? static_cast<size_t>(command->task.frames)
        : 1U;
    const FftDivStageLockSnapshot *locks_to_use = stage_locks;
    const EdgeRunnerNodeInputs *inputs_arg = command->task.inputs;
    EdgeRunnerNodeInputs inputs_copy{};
    if (command->input_consumed_once && inputs_arg != nullptr) {
        inputs_copy = *inputs_arg;
        // Suppress audio on reuse but preserve frames (and thus the FINAL flag if present).
        inputs_copy.audio.has_audio = 0U;
        inputs_copy.audio.data = nullptr;
        // Keep frames unchanged so Stage 1 sees zero-tail plan correctly.
        inputs_arg = &inputs_copy;
    }
    // Always keep frames unchanged; suppressing audio via has_audio/data masking is sufficient.
    int frames_arg = command->task.frames;
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_SUMMARY,
               "[STAGE0-WORKER] input(has_audio=0x%08x, frames=%u, data=%p) consumed_once=%d\n",
               (unsigned)(inputs_arg && inputs_arg->audio.has_audio ? inputs_arg->audio.has_audio : 0U),
               (unsigned)(inputs_arg ? inputs_arg->audio.frames : 0U),
               (void*)(inputs_arg ? inputs_arg->audio.data : nullptr),
               command->input_consumed_once ? 1 : 0);
    int status = fftdiv_execute_block(
        command->task.descriptor,
        inputs_arg,
        command->task.batches,
        command->task.channels,
        frames_arg,
        runtime_pcm_block,
        locks_to_use,
        command->task.sample_rate,
        &buffer,
        &out_channels,
        state,
        command->want_metrics ? metrics_ptr : nullptr,
        command->task.flush_mode
    );
    FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
               "[STAGE5-PCM-MAILBOX] status=%d buffer=%p channels=%d\n",
               status,
               (void*)buffer,
               out_channels);

    if (command->task.flush_mode == AMP_FFT_STREAM_FLUSH_NONE) {
        state->u.fftdiv.last_descriptor = command->task.descriptor;
        state->u.fftdiv.last_batches = command->task.batches;
        state->u.fftdiv.last_channels = command->task.channels;
        state->u.fftdiv.last_frames = command->task.frames;
        state->u.fftdiv.last_slot_count = command->task.slot_count;
        state->u.fftdiv.last_sample_rate = command->task.sample_rate;
    }

    bool emit_entry = (buffer != nullptr) || (status != AMP_E_PENDING);
    if (emit_entry) {
        int frames = command->expected_frames > 0 ? command->expected_frames : command->task.frames;
        if (frames <= 0) {
            frames = state->u.fftdiv.last_frames > 0
                ? state->u.fftdiv.last_frames
                : state->u.fftdiv.window_size;
            if (frames <= 0) {
                frames = 1;
            }
        }
        const AmpNodeMetrics *metrics_to_send = command->want_metrics ? metrics_ptr : nullptr;
        if (!command->want_metrics) {
            command->metrics_storage = *metrics_ptr;
            metrics_to_send = &command->metrics_storage;
        }
        fftdiv_worker_publish_entry(state, buffer, out_channels, frames, status, metrics_to_send);
    } else if (buffer != nullptr) {
        fftdiv_worker_publish_entry(state, buffer, out_channels, command->expected_frames, status, command->want_metrics ? metrics_ptr : nullptr);
    }
    // After first execution of this command, suppress input on subsequent passes
    // within the worker loop to avoid re-including the same buffer.
    if (!command->input_consumed_once) {
        command->input_consumed_once = true;
    }
    return status != AMP_E_PENDING;
}

static bool fftdiv_worker_drain_pending_outputs(node_state_t *state) {
    if (state == nullptr) {
        return false;
    }
    bool did_work = false;
    while (true) {
        const FftDivStageLockSnapshot snapshot = fftdiv_snapshot_stage_locks(state);
        if (!fftdiv_stage_snapshot_any_ready(snapshot)) {
            break;
        }

        auto &fftdiv = state->u.fftdiv;
        const EdgeRunnerNodeDescriptor *descriptor = fftdiv.last_descriptor;
        if (descriptor == nullptr) {
            break;
        }

        int batches = (fftdiv.last_batches > 0) ? fftdiv.last_batches : 1;
        int channels = (fftdiv.last_channels > 0) ? fftdiv.last_channels : 1;
        int frames = (fftdiv.last_frames > 0) ? fftdiv.last_frames
                     : ((fftdiv.window_size > 0) ? fftdiv.window_size : 1);
        if (frames <= 0) {
            frames = 1;
        }

        EdgeRunnerAudioView audio{};
        audio.has_audio = 0; // No new PCM; advance existing pipeline only
        audio.batches = (uint32_t)batches;
        audio.channels = (uint32_t)channels;
        audio.frames = (uint32_t)frames;
        audio.data = nullptr;

        EdgeRunnerParamSet params{};
        EdgeRunnerNodeInputs inputs{};
        inputs.audio = audio;
        inputs.params = params;
        inputs.taps.outputs.items = nullptr;
        inputs.taps.outputs.count = 0;
        inputs.taps.status.items = nullptr;
        inputs.taps.status.count = 0;

        double *buffer = nullptr;
        int out_channels = 0;
        const double sample_rate = (fftdiv.last_sample_rate > 0.0)
            ? fftdiv.last_sample_rate
            : fftdiv.sample_rate_hint;

        // Advance one scheduler tick across all currently-ready stages.
        int status = fftdiv_execute_block(
            descriptor,
            &inputs,
            batches,
            channels,
            frames,
            (size_t)frames,
            &snapshot,
            sample_rate,
            &buffer,
            &out_channels,
            state,
            nullptr,
            AMP_FFT_STREAM_FLUSH_NONE
        );

        // Publish any produced output or terminal status
        if (buffer != nullptr || status != AMP_E_PENDING) {
            fftdiv_worker_publish_entry(state, buffer, out_channels, frames, status, nullptr);
        }

        did_work = true;
        if (status != 0 && status != AMP_E_PENDING) {
            break;
        }
    }
    return did_work;
}

static void fftdiv_worker_main(node_state_t *state) {
    if (state == nullptr) {
        return;
    }
    // The FFT division worker is intended to be a "live" stage scheduler that
    // keeps all five pipeline stages moving as long as there is outstanding
    // work anywhere in the internal queues.  Each pass through the loop pulls a
    // pending command (if any) and then calls `fftdiv_worker_drain_pending_outputs`
    // to move any outstanding spectra/PCM through Stage 5 without synthesizing
    // new input.  Only when neither step performed work do we yield back to the
    // condition variable.  This keeps the worker advancing previously-submitted
    // input even if the runtime stops issuing new `amp_run_node_v2` calls,
    // satisfying the always-running worker requirement without spurious flushes.
    auto &worker = state->u.fftdiv.worker;
    auto should_wake = [&worker, state]() {
        if (worker.stop_requested) {
            return true;
        }
        if (worker.active_command) {
            return true;
        }
        if (!worker.pending_commands.empty()) {
            return true;
        }
        return fftdiv_pipeline_can_advance(state);
    };

    std::unique_lock<std::mutex> lock(worker.mutex);

    auto process_ready_work = [&](std::unique_lock<std::mutex> &lock_guard) -> bool {
        std::shared_ptr<FftDivWorkerCommand> command = worker.active_command;
        if (!command && !worker.pending_commands.empty()) {
            command = worker.pending_commands.front();
            worker.pending_commands.pop_front();
            worker.active_command = command;
        }
        const FftDivStageLockSnapshot *locks_for_command = nullptr;
        if (worker.cached_stage_locks_valid) {
            locks_for_command = &worker.cached_stage_locks;
        }

        lock_guard.unlock();

        bool did_work = false;

        bool command_finished = false;
        if (command) {
            command_finished = fftdiv_worker_process_command(state, command, locks_for_command);
            did_work = true;
        }
        if (fftdiv_worker_drain_pending_outputs(state)) {
            did_work = true;
        }

        FftDivStageLockSnapshot post_snapshot = fftdiv_snapshot_stage_locks(state);
        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_DETAIL)) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                       "[STAGE-LOCKS] s1=%d s2=%d s3=%d s4=%d s5=%d\n",
                       post_snapshot.stage1_ready ? 1 : 0,
                       post_snapshot.stage2_ready ? 1 : 0,
                       post_snapshot.stage3_ready ? 1 : 0,
                       post_snapshot.stage4_ready ? 1 : 0,
                       post_snapshot.stage5_ready ? 1 : 0);
        }
        if (fftdiv_stage_snapshot_any_ready(post_snapshot)) {
            did_work = true;
        }

        lock_guard.lock();
        worker.cached_stage_locks = post_snapshot;
        worker.cached_stage_locks_valid = true;
        if (command_finished) {
            if (fftdiv_stage1_zero_insertion_active(state)) {
                // A zero-tail drain is still keeping at least one stage busy;
                // hold the command open until every stage lock reports idle.
            } else {
                // Reset command state for reuse in next streaming chunk.
                // This allows append_inputs to bind new audio data.
                command->inputs_bound = false;
                command->input_consumed_once = false;
                worker.shared_command = command;
                worker.active_command.reset();
            }
        }
        return did_work;
    };

    while (true) {
        while (!worker.stop_requested) {
            bool did_work = process_ready_work(lock);
            if (!did_work) {
                if (fftdiv_pipeline_can_advance(state)) {
                    lock.unlock();
                    std::this_thread::yield();
                    lock.lock();
                    continue;
                }
                break;
            }
        }

        if (worker.stop_requested) {
            bool flush_on_stop = worker.flush_on_stop;
            lock.unlock();
            if (flush_on_stop) {
                fftdiv_worker_drain_pending_outputs(state);
            }
            lock.lock();
            break;
        }

        worker.cv_request.wait(lock, should_wake);
        if (worker.stop_requested) {
            continue;
        }

        bool did_work = process_ready_work(lock);
        if (!did_work && fftdiv_pipeline_can_advance(state)) {
            std::this_thread::yield();
        }
    }

    worker.active_command.reset();
    worker.thread_started = false;
}

static int fftdiv_start_worker(node_state_t *state) {
    if (state == nullptr) {
        return -1;
    }
    auto &worker = state->u.fftdiv.worker;
    std::lock_guard<std::mutex> lock(worker.mutex);
    if (worker.thread_started) {
        return 0;
    }
    worker.cached_stage_locks = FftDivStageLockSnapshot{0, 0, 0, 0, 0};
    worker.cached_stage_locks_valid = false;
    worker.stop_requested = false;
    worker.flush_on_stop = true;
    try {
        worker.thread = std::thread([state]() {
            fftdiv_worker_main(state);
        });
        worker.thread_started = true;
        worker.cv_request.notify_one();
    } catch (...) {
        worker.thread_started = false;
        worker.stop_requested = false;
        return -1;
    }
    return 0;
}

static std::shared_ptr<FftDivWorkerCommand> fftdiv_worker_ensure_shared_command(
    node_state_t *state,
    const EdgeRunnerNodeDescriptor *descriptor,
    int batches,
    int channels,
    int frames,
    int slot_count,
    double sample_rate,
    bool want_metrics
) {
    if (state == nullptr) {
        return nullptr;
    }
    auto &worker = state->u.fftdiv.worker;
    std::shared_ptr<FftDivWorkerCommand> command;
    if (worker.shared_command) {
        command = worker.shared_command;
        worker.shared_command.reset();
    } else {
        command = std::make_shared<FftDivWorkerCommand>();
    }
    command->task.descriptor = descriptor;
    command->task.batches = batches;
    command->task.channels = channels;
    command->task.frames = frames;
    command->task.slot_count = slot_count;
    command->task.sample_rate = sample_rate;
    command->task.flush_mode = AMP_FFT_STREAM_FLUSH_NONE;
    command->task.final_delivery = false;
    command->want_metrics = want_metrics;
    return command;
}

static void fftdiv_stop_worker(node_state_t *state, bool flush) {
    if (state == nullptr) {
        return;
    }
    auto &worker = state->u.fftdiv.worker;
    std::thread join_thread;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.flush_on_stop = flush;
        if (!worker.thread_started) {
            worker.stop_requested = false;
            worker.active_command.reset();
            worker.pending_commands.clear();
            worker.shared_command.reset();
            worker.mailbox_expected_entries = 0;
            worker.mailbox_popped_entries = 0;
            worker.append_invocations = 0;
            worker.cached_stage_locks = FftDivStageLockSnapshot{0, 0, 0, 0, 0};
            worker.cached_stage_locks_valid = false;
            return;
        }
        worker.stop_requested = true;
        worker.cv_request.notify_all();
        join_thread = std::move(worker.thread);
    }
    if (join_thread.joinable()) {
        join_thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.thread_started = false;
        worker.stop_requested = false;
        worker.active_command.reset();
        worker.pending_commands.clear();
        worker.shared_command.reset();
        worker.mailbox_expected_entries = 0;
        worker.mailbox_popped_entries = 0;
        worker.append_invocations = 0;
        worker.cached_stage_locks = FftDivStageLockSnapshot{0, 0, 0, 0, 0};
        worker.cached_stage_locks_valid = false;
    }
}

#endif

static int run_fft_division_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics
) {
#if !defined(__cplusplus)
#error "fft division worker requires C++ compilation"
#endif
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    *out_buffer = NULL;
    *out_channels = 0;

    if (fftdiv_start_worker(state) != 0) {
        return -1;
    }

    int effective_batches = (batches > 0) ? batches : 1;
    int effective_channels = channels;
    if (effective_channels <= 0) {
        effective_channels = (inputs->audio.channels > 0U)
            ? (int)inputs->audio.channels
            : 1;
    }
    int slot_count = effective_batches * effective_channels;
    if (slot_count <= 0) {
        slot_count = 1;
    }

    auto &worker = state->u.fftdiv.worker;
    std::shared_ptr<FftDivWorkerCommand> command;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        command = fftdiv_worker_ensure_shared_command(
            state,
            descriptor,
            effective_batches,
            effective_channels,
            frames,
            slot_count,
            sample_rate,
            metrics != NULL
        );
        if (!command) {
            return -1;
        }
        if (!command->append_inputs(inputs, effective_batches, effective_channels, frames, slot_count)) {
            worker.shared_command = command;
            return -1;
        }
        worker.pending_commands.push_back(command);
        worker.append_invocations++;
        size_t expected = command->expected_frames > 0
            ? static_cast<size_t>(command->expected_frames)
            : 1U;
        worker.mailbox_expected_entries += expected;
    }
    worker.cv_request.notify_one();

    AmpMailboxEntry *entry = amp_node_mailbox_pop(state);
    if (entry == nullptr) {
        return AMP_E_PENDING;
    }

    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.mailbox_popped_entries++;
    }

    if (out_channels != NULL) {
        *out_channels = entry->channels > 0 ? entry->channels : effective_channels;
    }
    if (out_buffer != NULL) {
        *out_buffer = entry->buffer;
    }
    if (metrics != NULL) {
        *metrics = entry->metrics;
    }
    int status = entry->status;
    amp_mailbox_entry_release(entry);
    return status;
}

static int run_fft_division_node_backward(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics
) {
    (void)descriptor;
    (void)inputs;
    (void)batches;
    (void)channels;
    (void)frames;
    (void)sample_rate;
    (void)out_buffer;
    (void)out_channels;
    (void)state;
    (void)metrics;
    return AMP_E_UNSUPPORTED;
}
