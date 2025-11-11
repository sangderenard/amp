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
static double *fftdiv_spectral_scratch_real_ptr(node_state_t *state, int cache_page, int lane, int tensor_slice) {
    if (state == NULL) {
        return NULL;
    }
    auto &scratch = state->u.fftdiv.spectral_scratch;
    if (cache_page < 0 || lane < 0 || tensor_slice < 0) {
        return NULL;
    }
    if (cache_page >= scratch.cache_pages || lane >= scratch.lanes || tensor_slice >= scratch.time_slices) {
        return NULL;
    }
    if (scratch.freq_bins <= 0) {
        return NULL;
    }
    const size_t offset = ((((size_t)cache_page * (size_t)scratch.lanes) + (size_t)lane) *
        (size_t)scratch.time_slices + (size_t)tensor_slice) * (size_t)scratch.freq_bins;
    if (offset >= scratch.real.size()) {
        return NULL;
    }
    return scratch.real.data() + offset;
}

static double *fftdiv_spectral_scratch_imag_ptr(node_state_t *state, int cache_page, int lane, int tensor_slice) {
    if (state == NULL) {
        return NULL;
    }
    auto &scratch = state->u.fftdiv.spectral_scratch;
    if (cache_page < 0 || lane < 0 || tensor_slice < 0) {
        return NULL;
    }
    if (cache_page >= scratch.cache_pages || lane >= scratch.lanes || tensor_slice >= scratch.time_slices) {
        return NULL;
    }
    if (scratch.freq_bins <= 0) {
        return NULL;
    }
    const size_t offset = ((((size_t)cache_page * (size_t)scratch.lanes) + (size_t)lane) *
        (size_t)scratch.time_slices + (size_t)tensor_slice) * (size_t)scratch.freq_bins;
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

static void stage_run_operator_stack(
    node_state_t *state,
    double *spectral_real,
    double *spectral_imag,
    int window_size,
    FftWorkingTensor *tensor,
    int tensor_page,
    int tensor_lane,
    int tensor_slice,
    int tensor_freq_bins
) {
    (void)spectral_real;
    (void)spectral_imag;
    (void)window_size;
    (void)tensor;
    (void)tensor_page;
    (void)tensor_lane;
    (void)tensor_slice;
    (void)tensor_freq_bins;
#if defined(__cplusplus)
    if (state == nullptr) {
        return;
    }
    (void)state;
    /* TODO: execute operator_steps using operator_arena tensors. */
#else
    (void)state;
    /* Operators mutate the spectral buffer in-place. Intentionally left empty for now. */
#endif
}

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

    int window_size = json_get_int(descriptor->params_json, descriptor->params_len, "window_size", 8);
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
        window_size
    );
    if (working_freq_bins <= 0) {
        working_freq_bins = window_size;
    }
    int working_time_slices = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_duration_frames",
        frames > 0 ? frames : 1
    );
    if (working_time_slices <= 0) {
        working_time_slices = 1;
    }
    int working_cache_pages = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "working_ft_cache_pages",
        1
    );
    if (working_cache_pages <= 0) {
        working_cache_pages = 1;
    }
    if (ensure_fft_working_tensor(state, working_cache_pages, slot_count, working_freq_bins, working_time_slices) != 0) {
        return -1;
    }

    if (ensure_fft_spectral_scratch(state, working_cache_pages, slot_count, window_size, working_time_slices) != 0) {
        return -1;
    }

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
    if (spectral_real_tap == NULL || spectral_imag_tap == NULL) {
        free(buffer);
        return AMP_E_UNSUPPORTED;
    }

    FftWorkingTensor *working_tensor = state->u.fftdiv.working_tensor;
    int tensor_cache_cursor = state->u.fftdiv.working_tensor_cache_cursor;
    int tensor_time_cursor = state->u.fftdiv.working_tensor_time_cursor;
    const int tensor_time_slices = state->u.fftdiv.working_tensor_time_slices > 0
        ? state->u.fftdiv.working_tensor_time_slices
        : 1;
    const int tensor_freq_bins = state->u.fftdiv.working_tensor_freq_bins > 0
        ? state->u.fftdiv.working_tensor_freq_bins
        : window_size;
    const int tensor_cache_pages = state->u.fftdiv.working_tensor_cache_pages > 0
        ? state->u.fftdiv.working_tensor_cache_pages
        : 1;
    if (tensor_cache_cursor < 0 || tensor_cache_cursor >= tensor_cache_pages) {
        tensor_cache_cursor = 0;
    }
    int scratch_cache_cursor = state->u.fftdiv.spectral_scratch.cache_cursor;
    int scratch_time_cursor = state->u.fftdiv.spectral_scratch.time_cursor;
    const int scratch_cache_pages = state->u.fftdiv.spectral_scratch.cache_pages > 0
        ? state->u.fftdiv.spectral_scratch.cache_pages
        : 1;
    const int scratch_time_slices = state->u.fftdiv.spectral_scratch.time_slices > 0
        ? state->u.fftdiv.spectral_scratch.time_slices
        : 1;
    if (scratch_cache_cursor < 0 || scratch_cache_cursor >= scratch_cache_pages) {
        scratch_cache_cursor = 0;
    }
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

    const int frame_index = 0;
    size_t base_index = 0;
#if defined(__cplusplus)
        fftdiv_prepare_operator_frame(state);
        const int tensor_slice = tensor_time_cursor;
    const int scratch_slice = scratch_time_cursor;
        int ready_lane_count = 0;
    int working_tensor_updated = 0;

    std::vector<size_t> lane_frames_emitted(static_cast<size_t>(slot_count), 0U);
    std::vector<double> lane_last_sample(static_cast<size_t>(slot_count), 0.0);
    std::vector<double> lane_pcm_batch(static_cast<size_t>(slot_count) * static_cast<size_t>(frames), 0.0);

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            double *slot_spectral_real = slot_state.forward_real.data();
            double *slot_spectral_imag = slot_state.forward_imag.data();
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
            double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, scratch_cache_cursor, tensor_lane, scratch_slice);
            double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, scratch_cache_cursor, tensor_lane, scratch_slice);
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
                    frames_emitted = amp_fft_backend_stream_push(
                        slot_state.forward_handle,
                        pcm_cursor,
                        (size_t)frames,
                        window_size,
                        slot_spectral_real,
                        slot_spectral_imag,
                        1,
                        AMP_FFT_STREAM_FLUSH_NONE);
                }
            }
            lane_frames_emitted[(size_t)slot] = frames_emitted;
            lane_last_sample[(size_t)slot] = last_sample;
            if (frames_emitted > 0) {
                slot_state.warmup_complete = true;
            }
            if (!slot_state.warmup_complete) {
                lane.staged_pcm_value = last_sample;
                lane.staged_pcm_valid = true;
                continue;
            }
            if (frames_emitted == 0) {
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
                    frame_index,
                    reset_spectral_buffer,
                    reset_scratch_slice,
                    scratch_real,
                    scratch_imag,
                    scratch_bins);
            }

            if (!spectral_ready) {
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
                    lane.staged_pcm_value = 0.0;
                }
                lane.staged_pcm_valid = true;
                continue;
            }

            lane.frame_ready = true;
            ready_lane_count += 1;

            if (!lane.enable_pcm_out) {
                lane.staged_pcm_value = 0.0;
            } else if (!slot_state.inverse_queue.empty()) {
                lane.staged_pcm_value = slot_state.inverse_queue.front();
            } else {
                lane.staged_pcm_value = 0.0;
            }
            lane.staged_pcm_valid = true;
        }

        const bool barrier_satisfied =
            (active_lane_count == 0) ? true : (ready_lane_count == active_lane_count);

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            auto &slot_state = state->u.fftdiv.stream_slots[(size_t)slot];
            double *slot_spectral_real = slot_state.forward_real.data();
            double *slot_spectral_imag = slot_state.forward_imag.data();
            auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
            const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
            double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, scratch_cache_cursor, tensor_lane, scratch_slice);
            double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, scratch_cache_cursor, tensor_lane, scratch_slice);
            const int scratch_bins = fftdiv_spectral_scratch_bins(state);

            double output_value = lane.staged_pcm_valid ? lane.staged_pcm_value : 0.0;

            if (barrier_satisfied && lane.active && lane.frame_ready && slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                if (working_tensor != NULL && tensor_freq_bins > 0) {
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
                        tensor_cache_cursor,
                        tensor_lane,
                        tensor_slice,
                        tensor_freq_bins,
                        commit_real,
                        commit_imag,
                        commit_bins);
                    if (!state->u.fftdiv.preserve_tensor_on_ingest && scratch_real != NULL && scratch_imag != NULL && scratch_bins > 0) {
                        memset(scratch_real, 0, (size_t)scratch_bins * sizeof(double));
                        memset(scratch_imag, 0, (size_t)scratch_bins * sizeof(double));
                    }
                    working_tensor_updated = 1;
                }
                stage_run_operator_stack(
                    state,
                    slot_spectral_real,
                    slot_spectral_imag,
                    window_size,
                    working_tensor,
                    tensor_cache_cursor,
                    tensor_lane,
                    tensor_slice,
                    tensor_freq_bins);

                if ((lane.enable_pcm_out || lane.enable_spectral_out) &&
                    working_tensor != NULL &&
                    slot_spectral_real != NULL &&
                    slot_spectral_imag != NULL) {
                    fftdiv_copy_working_to_slot_buffer(
                        working_tensor,
                        tensor_cache_cursor,
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
                        frame_index,
                        working_tensor,
                        tensor_cache_cursor,
                        tensor_lane,
                        tensor_slice,
                        tensor_freq_bins);
                }

                if (lane.enable_pcm_out && slot_state.inverse_handle != NULL && !slot_state.inverse_scratch.empty()) {
                    const size_t produced_pcm = amp_fft_backend_stream_push_spectrum(
                        slot_state.inverse_handle,
                        slot_spectral_real,
                        slot_spectral_imag,
                        1,
                        window_size,
                        slot_state.inverse_scratch.data(),
                        slot_state.inverse_scratch.size(),
                        AMP_FFT_STREAM_FLUSH_NONE);
                    for (size_t i = 0; i < produced_pcm && i < slot_state.inverse_scratch.size(); ++i) {
                        slot_state.inverse_queue.push_back(slot_state.inverse_scratch[i]);
                    }
                } else if (!lane.enable_pcm_out) {
                    slot_state.inverse_queue.clear();
                }

                if (lane.enable_pcm_out && !slot_state.inverse_queue.empty()) {
                    output_value = slot_state.inverse_queue.front();
                    slot_state.inverse_queue.pop_front();
                } else if (!lane.enable_pcm_out) {
                    output_value = 0.0;
                } else {
                    output_value = 0.0;
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
            if (tensor_cache_pages > 0) {
                tensor_cache_cursor += 1;
                if (tensor_cache_cursor >= tensor_cache_pages) {
                    tensor_cache_cursor = 0;
                }
            }
            if (tensor_time_slices > 0) {
                tensor_time_cursor += 1;
                if (tensor_time_cursor >= tensor_time_slices) {
                    tensor_time_cursor = 0;
                }
            }
            if (scratch_cache_pages > 0) {
                scratch_cache_cursor += 1;
                if (scratch_cache_cursor >= scratch_cache_pages) {
                    scratch_cache_cursor = 0;
                }
            }
            if (scratch_time_slices > 0) {
                scratch_time_cursor += 1;
                if (scratch_time_cursor >= scratch_time_slices) {
                    scratch_time_cursor = 0;
                }
            }
        }
#endif

    state->u.fftdiv.working_tensor_time_cursor = tensor_time_cursor;
    state->u.fftdiv.working_tensor_cache_cursor = tensor_cache_cursor;
#if defined(__cplusplus)
    state->u.fftdiv.spectral_scratch.cache_cursor = scratch_cache_cursor;
    state->u.fftdiv.spectral_scratch.time_cursor = scratch_time_cursor;
#endif

    *out_buffer = buffer;
    *out_channels = input_channels;

    if (metrics != NULL) {
        metrics->accumulated_heat = 0.0f;
        metrics->reserved[0] = 0.0f;
        metrics->reserved[1] = 0.0f;
        metrics->reserved[2] = 0.0f;
        metrics->reserved[3] = 0.0f;
        metrics->reserved[4] = (float)window_size;
        metrics->reserved[5] = (float)default_algorithm;
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
