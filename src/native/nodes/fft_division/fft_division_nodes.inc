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
#define FFTDIV_TRACE_ENABLED 1
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
    memcpy(batch_ptr, src, copy * sizeof(double));
    if (copy < channels) {
        size_t remaining = channels - copy;
        memset(batch_ptr + copy, 0, remaining * sizeof(double));
    }
}

static void zero_spectral_buffer(double *spectral_real, double *spectral_imag, int window_size) {
    if (spectral_real != NULL) {
        memset(spectral_real, 0, (size_t)window_size * sizeof(double));
    }
    if (spectral_imag != NULL) {
        memset(spectral_imag, 0, (size_t)window_size * sizeof(double));
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
    if (slot.forward_imag.size() < capacity_frames * (std::size_t)window_size) {
        fftdiv_ring_ensure_capacity(slot.forward_imag, capacity_frames * (std::size_t)window_size);
    }
    std::size_t write = slot.forward_ring_write % capacity_frames;
    std::size_t read = slot.forward_ring_read % capacity_frames;
    std::size_t filled = slot.forward_ring_filled;
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
        const std::size_t write_frames = std::min(remaining, capacity_frames - write);
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
    double *spectral_real,
    double *spectral_imag,
    int window_size,
    const EdgeRunnerTapBuffer *spectral_real_tap,
    const EdgeRunnerTapBuffer *spectral_imag_tap,
    int tap_slot,
    int frame_index,
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_lane,
    int tensor_slice,
    int tensor_freq_bins
) {
    int have_real_target = (spectral_real_tap != NULL && spectral_real_tap->data != NULL);
    int have_imag_target = (spectral_imag_tap != NULL && spectral_imag_tap->data != NULL);
    if (!have_real_target && !have_imag_target) {
        return;
    }
    if (tensor != NULL && tensor_freq_bins > 0) {
        const int copy_bins = tensor_freq_bins < window_size ? tensor_freq_bins : window_size;
        for (int bin = 0; bin < copy_bins; ++bin) {
            const std::complex<double> value = (*tensor)(tensor_page, tensor_lane, bin, tensor_slice);
            spectral_real[bin] = value.real();
            spectral_imag[bin] = value.imag();
        }
        for (int bin = copy_bins; bin < window_size; ++bin) {
            spectral_real[bin] = 0.0;
            spectral_imag[bin] = 0.0;
        }
    }
    tap_buffer_write_row(spectral_real_tap, tap_slot, frame_index, spectral_real, window_size);
    tap_buffer_write_row(spectral_imag_tap, tap_slot, frame_index, spectral_imag, window_size);
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

    const bool spectral_out_available =
        (spectral_real_tap != nullptr && spectral_real_tap->data != nullptr) ||
        (spectral_imag_tap != nullptr && spectral_imag_tap->data != nullptr);

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
        const bool has_spectral_in = spect_real_in || spect_imag_in;

        const bool enable_pcm_in = has_pcm_feed || has_spectral_in;
        const bool enable_pcm_out = enable_pcm_in || has_spectral_in;

        lane.enable_pcm_in = enable_pcm_in;
        lane.enable_pcm_out = enable_pcm_out;
        lane.enable_spectral_in = has_spectral_in;
        lane.enable_spectral_out = spectral_out_available;
        lane.active = (lane.enable_pcm_in || lane.enable_spectral_in) &&
            (lane.enable_pcm_out || lane.enable_spectral_out);
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
    (void)sample_rate;
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
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

    const size_t runtime_pcm_block = frames > 0 ? (size_t)frames : 1U;
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

    int working_freq_bins = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_frequency_bins",
        (default_freq_bins > 0) ? default_freq_bins : window_size
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
    int working_hop = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_hop",
        default_hop > 0 ? default_hop : (window_size > 1 ? window_size / 2 : 1)
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
    if (audio_base != NULL) {
        memcpy(buffer, audio_base, total_samples * sizeof(double));
    } else {
        memset(buffer, 0, total_samples * sizeof(double));
    }
    const EdgeRunnerParamView *spectral_input_real_view = find_param(inputs, "spectral_input_real");
    const EdgeRunnerParamView *spectral_input_imag_view = find_param(inputs, "spectral_input_imag");
    const EdgeRunnerTapBuffer *spectral_real_tap = find_tap_buffer(&inputs->taps, "spectral_real");
    const EdgeRunnerTapBuffer *spectral_imag_tap = find_tap_buffer(&inputs->taps, "spectral_imag");
    if (spectral_real_tap == NULL || spectral_imag_tap == NULL) {
        free(buffer);
        return AMP_E_UNSUPPORTED;
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

    int frame_index_int = 0;
    size_t base_index = 0;
    int metrics_window_span = 0;
#if defined(__cplusplus)
        fftdiv_prepare_operator_frame(state);
        const int tensor_slice = wheel_head;
    const int scratch_slice = scratch_time_cursor;
        int ready_lane_count = 0;
    int working_tensor_updated = 0;

    std::vector<size_t> lane_frames_emitted(static_cast<size_t>(slot_count), 0U);
    std::vector<double> lane_last_sample(static_cast<size_t>(slot_count), 0.0);
    std::vector<double> lane_pcm_batch(static_cast<size_t>(slot_count) * static_cast<size_t>(frames), 0.0);

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            const bool warmup_was_complete = slot_state.warmup_complete;
            auto &forward_stage_real = slot_state.forward_stage_real;
            auto &forward_stage_imag = slot_state.forward_stage_imag;
            double *slot_spectral_real = slot_state.forward_real.empty() ? NULL : slot_state.forward_real.data();
            double *slot_spectral_imag = slot_state.forward_imag.empty() ? NULL : slot_state.forward_imag.data();
            auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];

            lane.frame_ready = false;
            lane.staged_pcm_value = 0.0;
            lane.staged_pcm_valid = false;

            if (slot_spectral_real == NULL || slot_spectral_imag == NULL) {
                double passthrough = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
                lane.staged_pcm_value = passthrough;
                lane.staged_pcm_valid = true;
                continue;
            }

            const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
            double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice);
            double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice);
            const int scratch_bins = fftdiv_spectral_scratch_bins(state);

            if (!lane.active) {
                if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                }
                slot_state.inverse_queue.clear();
                lane.staged_pcm_value = 0.0;
                lane.staged_pcm_valid = true;
                continue;
            }

            size_t frames_emitted = 0;
            double last_sample = 0.0;
            if (lane.enable_pcm_in && audio_base != NULL) {
                const size_t hop = (size_t)slot_count;
                double *pcm_cursor = lane_pcm_batch.data() + (size_t)slot * (size_t)frames;
                for (int frame_cursor = 0; frame_cursor < frames; ++frame_cursor) {
                    const size_t audio_index = (size_t)frame_cursor * hop + (size_t)slot;
                    if (audio_index < (size_t)slot_count * (size_t)frames) {
                        last_sample = audio_base[audio_index];
                    } else {
                        last_sample = 0.0;
                    }
                    pcm_cursor[frame_cursor] = last_sample;
                }
                if (slot_state.forward_handle != NULL && frames > 0) {
                    size_t stage_capacity_frames = slot_state.forward_frame_capacity;
                    if (stage_capacity_frames == 0U) {
                        stage_capacity_frames = state->u.fftdiv.stream_max_fft_frames;
                    }
                    if (stage_capacity_frames == 0U) {
                        stage_capacity_frames = 1U;
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
#if FFTDIV_TRACE_ENABLED
                    const int pcm_trace_limit = 16;
                    const int pcm_values_to_log = (frames < pcm_trace_limit) ? frames : pcm_trace_limit;
                    FFTDIV_TRACE(
                        "[fftdiv] pcm-block slot=%d frame=%d frames=%d log=%d",
                        slot,
                        frame_index_int,
                        frames,
                        pcm_values_to_log);
                    for (int frame_cursor = 0; frame_cursor < pcm_values_to_log; ++frame_cursor) {
                        FFTDIV_TRACE(
                            "[fftdiv] pcm[%d]=%.12f",
                            frame_cursor,
                            pcm_cursor[frame_cursor]);
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
                        frame_capacity);
#endif
                    if (slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                        frames_emitted = amp_fft_backend_stream_push(
                            slot_state.forward_handle,
                            pcm_cursor,
                            (size_t)frames,
                            window_size,
                            slot_spectral_real,
                            slot_spectral_imag,
                            frame_capacity,
                            AMP_FFT_STREAM_FLUSH_NONE);
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
                }
            }
            lane_frames_emitted[(size_t)slot] = frames_emitted;
            slot_state.forward_frames_ready = slot_state.forward_ring_filled;
            lane_last_sample[(size_t)slot] = last_sample;
            if (frames_emitted > 0) {
                slot_state.warmup_complete = true;
                if (!warmup_was_complete) {
                    slot_state.drop_first_inverse_sample = true;
                }
            }
            if (!slot_state.warmup_complete) {
                double staged = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
#if FFTDIV_TRACE_ENABLED
                FFTDIV_TRACE(
                    "[fftdiv] warmup slot=%d frame=%d emitted=%zu staged=%f",
                    slot,
                    frame_index_int,
                    frames_emitted,
                    staged);
#endif
                lane.staged_pcm_value = staged;
                lane.staged_pcm_valid = true;
                continue;
            }
            if (frames_emitted == 0) {
#if FFTDIV_TRACE_ENABLED
                FFTDIV_TRACE(
                    "[fftdiv] zero-spectrum slot=%d frame=%d queue=%zu",
                    slot,
                    frame_index_int,
                    (size_t)slot_state.inverse_queue.size());
#endif
                zero_spectral_buffer(slot_spectral_real, slot_spectral_imag, window_size);
                if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                }
            }

            int spectral_ready = (frames_emitted > 0) ? 1 : 0;
            const int reset_spectral_buffer = spectral_ready ? 0 : 1;
            const int reset_scratch_slice = state->u.fftdiv.preserve_tensor_on_ingest ? 0 : reset_spectral_buffer;

            if (scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                if (frames_emitted > 0) {
                    const int scratch_copy_bins = (scratch_bins < window_size) ? scratch_bins : window_size;
                    memcpy(scratch_real, slot_spectral_real, (size_t)scratch_copy_bins * sizeof(double));
                    memcpy(scratch_imag, slot_spectral_imag, (size_t)scratch_copy_bins * sizeof(double));
                    if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_copy_bins < scratch_bins) {
                        memset(scratch_real + scratch_copy_bins, 0, (size_t)(scratch_bins - scratch_copy_bins) * sizeof(double));
                        memset(scratch_imag + scratch_copy_bins, 0, (size_t)(scratch_bins - scratch_copy_bins) * sizeof(double));
                    }
                } else if (reset_spectral_buffer && !state->u.fftdiv.preserve_tensor_on_ingest) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
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
                if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                }
                if (!lane.enable_pcm_out) {
                    slot_state.inverse_queue.clear();
                    lane.staged_pcm_value = 0.0;
                } else if (!slot_state.inverse_queue.empty()) {
                    lane.staged_pcm_value = slot_state.inverse_queue.front();
                } else {
                    lane.staged_pcm_value = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
                }
                lane.staged_pcm_valid = true;
                continue;
            }

            lane.frame_ready = (slot_state.forward_ring_filled > 0U);
            if (lane.frame_ready) {
                ready_lane_count += 1;
            }

            if (!lane.enable_pcm_out) {
                lane.staged_pcm_value = 0.0;
            } else if (!slot_state.inverse_queue.empty()) {
#if FFTDIV_TRACE_ENABLED
                size_t inverse_pending_post = (slot_state.inverse_handle != NULL)
                    ? amp_fft_backend_stream_pending_pcm(slot_state.inverse_handle)
                    : 0U;
                FFTDIV_TRACE(
                    "[fftdiv] emit slot=%d frame=%d value=%f queue_before=%zu pending_pcm=%zu",
                    slot,
                    frame_index_int,
                    slot_state.inverse_queue.front(),
                    (size_t)slot_state.inverse_queue.size(),
                    inverse_pending_post);
#endif
                lane.staged_pcm_value = slot_state.inverse_queue.front();
            } else {
                lane.staged_pcm_value = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
            }
            lane.staged_pcm_valid = true;
        }

    const bool barrier_satisfied = (active_lane_count == 0) ? true : (ready_lane_count == active_lane_count);
#if FFTDIV_TRACE_ENABLED
    FFTDIV_TRACE(
        "[fftdiv] barrier frame=%d satisfied=%d ready=%d active=%d",
        frame_index_int,
        barrier_satisfied ? 1 : 0,
        ready_lane_count,
        active_lane_count);
#endif

#if defined(__cplusplus)
    int view_filled_override = wheel_filled;
    FftDivActiveWindowView active_window_view;
    double effective_sample_rate = state->u.fftdiv.sample_rate_hint;
    if (sample_rate > 0.0) {
        effective_sample_rate = sample_rate;
        state->u.fftdiv.sample_rate_hint = sample_rate;
    }
    const double hop_seconds = (effective_sample_rate > 0.0)
        ? ((double)wheel_hop) / effective_sample_rate
        : 0.0;
    const int64_t frame_counter = state->u.fftdiv.wheel_frame_counter;
    frame_index_int = (int)frame_counter;
    const double timeline_seconds = state->u.fftdiv.timeline_seconds;
#endif
    if (barrier_satisfied && working_tensor != NULL && tensor_freq_bins > 0) {
            for (int slot = 0; slot < slot_count; ++slot) {
                auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
                double *slot_spectral_real = slot_state.forward_real.data();
                double *slot_spectral_imag = slot_state.forward_imag.data();
                auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
                if (!lane.active || !lane.frame_ready || slot_spectral_real == NULL || slot_spectral_imag == NULL) {
                    continue;
                }

                const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
                double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice);
                double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice);
                const int scratch_bins = fftdiv_spectral_scratch_bins(state);

                const double *commit_real = scratch_real;
                const double *commit_imag = scratch_imag;
                int commit_bins = scratch_bins;
                if (commit_real == NULL || commit_imag == NULL || commit_bins <= 0) {
                    commit_real = slot_spectral_real;
                    commit_imag = slot_spectral_imag;
                    commit_bins = window_size;
                }

                fftdiv_copy_spectrum_to_working(
                    working_tensor,
                    tensor_page,
                    tensor_lane,
                    tensor_slice,
                    tensor_freq_bins,
                    commit_real,
                    commit_imag,
                    commit_bins);
                /* TODO: introduce interpolation when tensor_freq_bins != commit_bins. */
                if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                    memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                    memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                }

                working_tensor_updated = 1;
            }

            if (working_tensor_updated) {
                if (wheel_filled >= wheel_length) {
                    view_filled_override = wheel_length;
                } else {
                    int hop_for_span = (wheel_hop > 0) ? wheel_hop : 1;
                    view_filled_override = wheel_filled + hop_for_span;
                    if (view_filled_override > wheel_length) {
                        view_filled_override = wheel_length;
                    }
                }
            }

            active_window_view = fftdiv_build_active_window_view_internal(
                state,
                working_tensor,
                tensor_page,
                tensor_slice,
                view_filled_override);

            if (active_window_view.valid()) {
                FftDivOperatorContext operator_context;
                operator_context.window = active_window_view;
                operator_context.lanes = operator_lane_bindings.empty()
                    ? nullptr
                    : operator_lane_bindings.data();
                operator_context.lane_count = operator_lane_bindings.size();
                operator_context.hop = wheel_hop;
                operator_context.window_size = window_size;
                operator_context.wheel_length = wheel_length;
                operator_context.wheel_head = wheel_head;
                operator_context.wheel_tail = wheel_tail;
                operator_context.frame_index = frame_counter;
                operator_context.sample_rate = effective_sample_rate;
                operator_context.timeline_seconds = timeline_seconds;
                operator_context.hop_seconds = hop_seconds;
                fftdiv_run_operator_stack(state, operator_context);
                metrics_window_span = active_window_view.window_span;
            } else {
                metrics_window_span = 0;
            }
        } else {
            metrics_window_span = 0;
        }

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            const double *ring_real = fftdiv_ring_frame_real(slot_state, 0U, window_size);
            const double *ring_imag = fftdiv_ring_frame_imag(slot_state, 0U, window_size);
            double *slot_spectral_real = (ring_real != NULL)
                ? const_cast<double *>(ring_real)
                : slot_state.forward_real.data();
            double *slot_spectral_imag = (ring_imag != NULL)
                ? const_cast<double *>(ring_imag)
                : slot_state.forward_imag.data();
            auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
            const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
            double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice);
            double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice);
            const int scratch_bins = fftdiv_spectral_scratch_bins(state);

            double output_value = lane.staged_pcm_valid ? lane.staged_pcm_value : 0.0;

            if (barrier_satisfied && lane.active && lane.frame_ready && slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                if ((lane.enable_pcm_out || lane.enable_spectral_out) &&
                    working_tensor != NULL &&
                    tensor_freq_bins > 0) {
                    fftdiv_copy_working_to_slot_buffer(
                        working_tensor,
                        tensor_page,
                        tensor_lane,
                        tensor_slice,
                        tensor_freq_bins,
                        slot_spectral_real,
                        slot_spectral_imag,
                        window_size);
                }

                if (lane.enable_spectral_out) {
                    stage_emit_spectral(
                        slot_spectral_real,
                        slot_spectral_imag,
                        window_size,
                        spectral_real_tap,
                        spectral_imag_tap,
                        slot,
                        frame_index_int,
                        working_tensor,
                        tensor_page,
                        tensor_lane,
                        tensor_slice,
                        tensor_freq_bins);
                }

                if (lane.enable_pcm_out && slot_state.inverse_handle != NULL && !slot_state.inverse_scratch.empty()) {
#if FFTDIV_TRACE_ENABLED
                    FFTDIV_TRACE(
                        "[fftdiv] inverse-pre slot=%d frame=%d queue_before=%zu scratch=%zu",
                        slot,
                        frame_index_int,
                        (size_t)slot_state.inverse_queue.size(),
                        slot_state.inverse_scratch.size());
#endif
                    const size_t produced_pcm = amp_fft_backend_stream_push_spectrum(
                        slot_state.inverse_handle,
                        slot_spectral_real,
                        slot_spectral_imag,
                        1,
                        window_size,
                        slot_state.inverse_scratch.data(),
                        slot_state.inverse_scratch.size(),
                        AMP_FFT_STREAM_FLUSH_NONE);
                    if (slot_state.drop_first_inverse_sample && produced_pcm > 0 && !slot_state.inverse_scratch.empty()) {
                        double passthrough = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
                        slot_state.inverse_scratch[0] = passthrough;
                        slot_state.drop_first_inverse_sample = false;
                    }
#if FFTDIV_TRACE_ENABLED
                    FFTDIV_TRACE(
                        "[fftdiv] inverse-produce slot=%d frame=%d produced=%zu",
                        slot,
                        frame_index_int,
                        produced_pcm);
#endif
                    for (size_t i = 0; i < produced_pcm && i < slot_state.inverse_scratch.size(); ++i) {
                        slot_state.inverse_queue.push_back(slot_state.inverse_scratch[i]);
                    }
#if FFTDIV_TRACE_ENABLED
                    FFTDIV_TRACE(
                        "[fftdiv] inverse-post slot=%d frame=%d queue_after=%zu",
                        slot,
                        frame_index_int,
                        (size_t)slot_state.inverse_queue.size());
#endif
                } else if (!lane.enable_pcm_out) {
                    slot_state.inverse_queue.clear();
                }

                if (lane.enable_pcm_out && !slot_state.inverse_queue.empty()) {
                    output_value = slot_state.inverse_queue.front();
                    slot_state.inverse_queue.pop_front();
                } else if (!lane.enable_pcm_out) {
                    output_value = 0.0;
                } else {
                    output_value = (audio_base != NULL) ? audio_base[data_idx] : 0.0;
                }
                if (slot_state.forward_ring_filled > 0U && slot_state.forward_ring_capacity_frames > 0U) {
                    slot_state.forward_ring_read =
                        (slot_state.forward_ring_read + 1U) % slot_state.forward_ring_capacity_frames;
                    slot_state.forward_ring_filled -= 1U;
                }
            } else {
                if (!barrier_satisfied && lane.frame_ready) {
                    if (slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                        zero_spectral_buffer(slot_spectral_real, slot_spectral_imag, window_size);
                    }
                    if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                        memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                        memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                    }
                }
                if (!lane.enable_pcm_out) {
                    slot_state.inverse_queue.clear();
                }
            }

            lane.frame_ready = false;
            lane.staged_pcm_valid = false;
            buffer[data_idx] = output_value;
#if FFTDIV_TRACE_ENABLED
            FFTDIV_TRACE(
                "[fftdiv] buffer slot=%d frame=%d value=%f queue_after=%zu",
                slot,
                frame_index_int,
                output_value,
                (size_t)slot_state.inverse_queue.size());
#endif
        }
#else
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = 0.0;
            if (audio_base != NULL) {
                sample = audio_base[data_idx];
            }
            buffer[data_idx] = sample;
        }
#endif

#if defined(__cplusplus)
        if (working_tensor_updated) {
            const bool wheel_was_full = (wheel_filled >= wheel_length);
            if (wheel_length > 0) {
                wheel_head = (wheel_head + wheel_hop) % wheel_length;
                if (!wheel_was_full) {
                    wheel_filled += wheel_hop;
                    if (wheel_filled > wheel_length) {
                        wheel_filled = wheel_length;
                    }
                } else {
                    wheel_filled = wheel_length;
                    wheel_tail = (wheel_tail + wheel_hop) % wheel_length;
                }
            }
            if (scratch_time_slices > 0) {
                scratch_time_cursor += 1;
                if (scratch_time_cursor >= scratch_time_slices) {
                    scratch_time_cursor = 0;
                }
            }
            state->u.fftdiv.wheel_frame_counter += 1;
            state->u.fftdiv.timeline_seconds += hop_seconds;
        }
#endif

    state->u.fftdiv.wheel_length = wheel_length;
    state->u.fftdiv.wheel_head = wheel_head;
    state->u.fftdiv.wheel_tail = wheel_tail;
    state->u.fftdiv.wheel_filled_slices = wheel_filled;
    state->u.fftdiv.wheel_hop = wheel_hop;
#if defined(__cplusplus)
    state->u.fftdiv.spectral_scratch.time_cursor = scratch_time_cursor;
#endif

    *out_buffer = buffer;
    *out_channels = input_channels;

    if (metrics != NULL) {
        metrics->measured_delay_frames = (uint32_t)((window_size > 0) ? (window_size - 1) : 0);
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

    return 0;
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
