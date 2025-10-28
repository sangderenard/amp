static int run_resampler_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state,
    AmpNodeMetrics *metrics
) {
    if (descriptor == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }

    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }

    double ema_alpha = json_get_double(descriptor->params_json, descriptor->params_len, "sample_rate_ema_alpha", 0.0);
    if (ema_alpha < 0.0) {
        ema_alpha = 0.0;
    } else if (ema_alpha > 1.0) {
        ema_alpha = 1.0;
    }
    double fixed_sample_rate = json_get_double(descriptor->params_json, descriptor->params_len, "sample_rate_hz", 0.0);
    int free_rate = json_get_bool(descriptor->params_json, descriptor->params_len, "sample_rate_free", 1) ? 1 : 0;
    double window_val = json_get_double(descriptor->params_json, descriptor->params_len, "sample_rate_window", 0.0);
    if (window_val < 0.0) {
        window_val = 0.0;
    }
    uint32_t window_size = (uint32_t)(window_val + 0.5);
    double init_value = json_get_double(descriptor->params_json, descriptor->params_len, "initial_value", 0.0);

    int audio_batches = 0;
    int audio_channels = 0;
    int audio_frames = 0;
    const double *input_data = NULL;
    int has_audio = 0;

    if (inputs != NULL) {
        const EdgeRunnerAudioView *audio = &inputs->audio;
        if (audio->has_audio && audio->data != NULL && audio->frames > 0 && audio->channels > 0) {
            has_audio = 1;
            input_data = audio->data;
            audio_batches = (int)audio->batches;
            audio_channels = (int)audio->channels;
            audio_frames = (int)audio->frames;
        }
    }

    if (frames <= 0) {
        frames = (audio_frames > 0) ? audio_frames : 1;
    }
    if (has_audio && audio_frames > 0 && frames != audio_frames) {
        frames = audio_frames;
    }
    if (batches <= 0) {
        batches = has_audio && audio_batches > 0 ? audio_batches : 1;
    }

    int use_channels = has_audio && audio_channels > 0
        ? audio_channels
        : (state->u.resampler.channels > 0 ? (int)state->u.resampler.channels : 1);

    uint32_t logical_batches = batches > 0 ? (uint32_t)batches : 1U;
    uint32_t logical_channels = use_channels > 0 ? (uint32_t)use_channels : 1U;

    size_t total = (size_t)logical_batches * (size_t)logical_channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }

    size_t state_span = (size_t)logical_batches * (size_t)logical_channels;
    if (state->u.resampler.last_values == NULL
        || state->u.resampler.channels != logical_channels
        || state->u.resampler.batches != logical_batches) {
        free(state->u.resampler.last_values);
        state->u.resampler.last_values = (double *)malloc(state_span * sizeof(double));
        if (state->u.resampler.last_values == NULL) {
            free(buffer);
            return -1;
        }
        for (size_t i = 0; i < state_span; ++i) {
            state->u.resampler.last_values[i] = init_value;
        }
        state->u.resampler.channels = logical_channels;
        state->u.resampler.batches = logical_batches;
    }

    if (state->u.resampler.window_size != window_size) {
        free(state->u.resampler.rate_window);
        state->u.resampler.rate_window = NULL;
        state->u.resampler.window_size = window_size;
        state->u.resampler.window_index = 0;
        state->u.resampler.window_count = 0;
        state->u.resampler.window_sum = 0.0;
        if (window_size > 0U) {
            state->u.resampler.rate_window = (double *)calloc((size_t)window_size, sizeof(double));
            if (state->u.resampler.rate_window == NULL) {
                free(buffer);
                return -1;
            }
        }
    }

    state->u.resampler.ema_alpha = ema_alpha;
    state->u.resampler.fixed_sample_rate = fixed_sample_rate > 0.0 ? fixed_sample_rate : 0.0;
    state->u.resampler.free_rate = free_rate;

    size_t batch_stride = (size_t)logical_channels * (size_t)frames;
    size_t channel_stride = (size_t)frames;

    size_t input_batch_stride = 0;
    size_t input_channel_stride = (size_t)frames;
    uint32_t available_batches = has_audio ? (audio_batches > 0 ? (uint32_t)audio_batches : 1U) : 0U;
    uint32_t available_channels = has_audio ? (audio_channels > 0 ? (uint32_t)audio_channels : 1U) : 0U;
    if (has_audio) {
        input_batch_stride = (size_t)available_channels * (size_t)frames;
    }

    for (uint32_t b = 0; b < logical_batches; ++b) {
        size_t state_base = (size_t)b * (size_t)state->u.resampler.channels;
        size_t out_batch_offset = (size_t)b * batch_stride;
        uint32_t input_batch = (available_batches > 0U && b < available_batches)
            ? b
            : (available_batches > 0U ? (available_batches - 1U) : 0U);
        for (uint32_t c = 0; c < logical_channels; ++c) {
            double last = state->u.resampler.last_values[state_base + (size_t)c];
            double *out_channel = buffer + out_batch_offset + (size_t)c * channel_stride;
            if (has_audio) {
                uint32_t input_channel = (available_channels > 0U && c < available_channels)
                    ? c
                    : (available_channels > 0U ? (available_channels - 1U) : 0U);
                const double *in_channel = input_data
                    + (size_t)input_batch * input_batch_stride
                    + (size_t)input_channel * input_channel_stride;
                for (int f = 0; f < frames; ++f) {
                    double value = in_channel[f];
                    double delta = value - last;
                    out_channel[f] = delta;
                    last = value;
                }
                state->u.resampler.last_values[state_base + (size_t)c] = last;
            } else {
                for (int f = 0; f < frames; ++f) {
                    out_channel[f] = 0.0;
                }
            }
        }
    }

    double instant_rate = frames > 0 ? sample_rate / (double)frames : sample_rate;
    double reported_rate = instant_rate;

    if (!free_rate && state->u.resampler.fixed_sample_rate > 0.0) {
        reported_rate = state->u.resampler.fixed_sample_rate;
        state->u.resampler.last_rate = reported_rate;
    } else {
        double smoothed = instant_rate;
        if (state->u.resampler.window_size > 0U && state->u.resampler.rate_window != NULL) {
            if (state->u.resampler.window_count == state->u.resampler.window_size) {
                size_t idx = (size_t)state->u.resampler.window_index;
                state->u.resampler.window_sum -= state->u.resampler.rate_window[idx];
            } else if (state->u.resampler.window_count < state->u.resampler.window_size) {
                state->u.resampler.window_count += 1U;
            }
            size_t write_index = (size_t)state->u.resampler.window_index;
            state->u.resampler.rate_window[write_index] = smoothed;
            state->u.resampler.window_sum += smoothed;
            state->u.resampler.window_index = (state->u.resampler.window_index + 1U) % state->u.resampler.window_size;
            if (state->u.resampler.window_count > 0U) {
                smoothed = state->u.resampler.window_sum / (double)state->u.resampler.window_count;
            }
        } else if (ema_alpha > 0.0 && ema_alpha < 1.0) {
            double previous = state->u.resampler.last_rate;
            if (previous <= 0.0) {
                previous = smoothed;
            }
            smoothed = ema_alpha * smoothed + (1.0 - ema_alpha) * previous;
        }
        if (smoothed <= 0.0) {
            smoothed = instant_rate;
        }
        state->u.resampler.last_rate = smoothed;
        reported_rate = smoothed;
    }

    if (metrics != NULL) {
        metrics->reserved[0] = reported_rate;
        metrics->reserved[1] = instant_rate;
        metrics->reserved[2] = (double)frames;
        metrics->reserved[3] = (double)state->u.resampler.free_rate;
        metrics->reserved[4] = state->u.resampler.last_rate;
        metrics->reserved[5] = state->u.resampler.fixed_sample_rate;
    }

    *out_buffer = buffer;
    *out_channels = (int)logical_channels;
    return 0;
}

