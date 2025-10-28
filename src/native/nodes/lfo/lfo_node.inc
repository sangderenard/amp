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
    if (inputs->audio.batches > 0) {
        B = (int)inputs->audio.batches;
    }
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
    amp_last_alloc_count = total;
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

