
        if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (flush_mode < AMP_FFT_STREAM_FLUSH_NONE || flush_mode > AMP_FFT_STREAM_FLUSH_FINAL) {
        flush_mode = AMP_FFT_STREAM_FLUSH_NONE;
    }
    const bool flush_request = (flush_mode != AMP_FFT_STREAM_FLUSH_NONE);
    if (frames <= 0 || batches <= 0 || channels <= 0 || slot_count <= 0) {
        return -1;
    }
    const int input_channels = channels;

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
                        const size_t audio_index = (size_t)frame_cursor * hop + (size_t)slot;
                        double input_sample = 0.0;
                        if (audio_index < (size_t)slot_count * (size_t)frames) {
                            input_sample = audio_base[audio_index];
                        }
                        pcm_cursor[frame_cursor] = input_sample;
                        slot_state.pcm_backlog.push_back(input_sample);
                    }
                    slot_state.total_ingested_samples += backlog_growth;
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
                               "[STAGE1-INPUT] slot=%d pending_samples=%zu handle=%p\n",
                               slot,
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
                                   "[STAGE1-FFT] slot=%d frames_emitted=%zu\n", slot, frames_emitted);
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
                    slot_state.pcm_consumed_samples = slot_state.pcm_backlog.size();
                    fftdiv_trim_pcm_backlog(state, slot_state.pcm_backlog, slot_state.pcm_consumed_samples, (size_t)window_size);
                    if (final_delivery) {
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

    while (true) {
        if (wheel_length > 0 && active_span_limit > 0 && wheel_filled_cursor >= active_span_limit) {
            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_DETAIL,
                       "[STAGE3-BACKPRESSURE] waiting head=%d tail=%d filled=%d span=%d len=%d\n",
                       wheel_head_cursor,
                       wheel_tail_cursor,
                       wheel_filled_cursor,
                       active_span_limit,
                       wheel_length);
            break;
        }
        int ready_lane_count_iter = 0;

        for (int slot = 0; slot < slot_count; ++slot) {
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
            }

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

    if (filled_slices.empty()) {
        metrics_window_span = 0;
    } else {
        for (const auto *slice_ptr : filled_slices) {
            if (slice_ptr == nullptr || !slice_ptr->working_tensor_updated) {
                metrics_window_span = 0;
                continue;
            }
            const auto &slice = *slice_ptr;

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
            } else {
                metrics_window_span = 0;
            }
        }

        std::vector<size_t> lane_frames_consumed(static_cast<size_t>(slot_count), 0U);
        for (auto *slice_ptr : filled_slices) {
            if (slice_ptr == nullptr) {
                continue;
            }
            auto &slice = *slice_ptr;
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
                const double *ring_real = fftdiv_ring_frame_real(slot_state, relative_offset, window_size);
                const double *ring_imag = fftdiv_ring_frame_imag(slot_state, relative_offset, window_size);
                double *slot_spectral_real = (ring_real != NULL)
                    ? const_cast<double *>(ring_real)
                    : slot_state.forward_real.data();
                double *slot_spectral_imag = (ring_imag != NULL)
                    ? const_cast<double *>(ring_imag)
                    : slot_state.forward_imag.data();
                auto &lane = state->u.fftdiv.lane_plan[(size_t)slot];
                const auto &lane_snapshot = slice.lanes[(size_t)slot];
                const int tensor_lane = (lane.tensor_lane >= 0) ? lane.tensor_lane : slot;
                double *scratch_real = fftdiv_spectral_scratch_real_ptr(state, tensor_lane, scratch_slice_iter);
                double *scratch_imag = fftdiv_spectral_scratch_imag_ptr(state, tensor_lane, scratch_slice_iter);
                const int scratch_bins = fftdiv_spectral_scratch_bins(state);

                if (lane_snapshot.frame_ready && lane.active && slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                    if ((lane.enable_pcm_out || lane.enable_spectral_out) &&
                        working_tensor != NULL &&
                        tensor_freq_bins > 0) {
                        fftdiv_copy_working_to_slot_buffer(
                            working_tensor,
                            tensor_page,
                            tensor_lane,
                            tensor_slice_iter,
                            tensor_freq_bins,
                            slot_spectral_real,
                            slot_spectral_imag,
                            window_size);
                    }

                    if (lane.enable_spectral_out) {
                        /* Emit spectral row aligned to PCM time: write at (frame + window_size - 1) */
                        const int emit_frame_index = (int)slice.frame_index + ((window_size > 0) ? (window_size - 1) : 0);
                        stage_emit_spectral(
                            state,
                            slot_spectral_real,
                            slot_spectral_imag,
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
                        if (slot_spectral_real != NULL && slot_spectral_imag != NULL) {
                            decltype(slot_state.pending_spectra)::value_type pending;
                            try {
                                pending.real.assign(slot_spectral_real, slot_spectral_real + (size_t)window_size);
                                pending.imag.assign(slot_spectral_imag, slot_spectral_imag + (size_t)window_size);
                                slot_state.pending_spectra.push_back(std::move(pending));
                            } catch (...) {
                                // Allocation failure: drop this frame; Stage 5 will still drain what exists
                            }
                        }
                    }

                    if (slot_state.forward_ring_filled > 0U && slot_state.forward_ring_capacity_frames > 0U) {
                        slot_state.forward_ring_read =
                            (slot_state.forward_ring_read + 1U) % slot_state.forward_ring_capacity_frames;
                        slot_state.forward_ring_filled -= 1U;
                        if (lane_frames_consumed.size() > (size_t)slot) {
                            lane_frames_consumed[(size_t)slot] += 1U;
                        }
                        if (fftdiv_should_log(state, FFTDIV_LOG_LEVEL_TRACE)) {
                            const size_t consumed_frames = (lane_frames_consumed.size() > (size_t)slot)
                                ? lane_frames_consumed[(size_t)slot]
                                : 0U;
                            FFTDIV_LOG(state, FFTDIV_LOG_LEVEL_TRACE,
                                       "[STAGE1-CONSUME] slot=%d read=%zu filled=%zu consumed=%zu assigned_offset=%zu relative_offset=%zu\n",
                                       slot,
                                       (size_t)slot_state.forward_ring_read,
                                       (size_t)slot_state.forward_ring_filled,
                                       consumed_frames,
                                       assigned_offset,
                                       relative_offset);
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
            }
        }

        wheel_head = wheel_head_cursor;
        wheel_tail = wheel_tail_cursor;
        wheel_filled = wheel_filled_cursor;
        scratch_time_cursor = scratch_cursor;
        state->u.fftdiv.wheel_frame_counter += (int64_t)processed_passes;
        state->u.fftdiv.timeline_seconds = timeline_cursor;
    }
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
    free(buffer);
    
    *out_buffer = NULL;
    *out_channels = 0;
    return AMP_E_PENDING;
