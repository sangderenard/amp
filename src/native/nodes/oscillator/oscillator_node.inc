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
    if (descriptor == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
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
    int mode = parse_osc_mode(descriptor->params_json, descriptor->params_len, state->u.osc.mode);
    if (mode < OSC_MODE_POLYBLEP || mode > OSC_MODE_OP_AMP) {
        mode = OSC_MODE_POLYBLEP;
    }

    double integration_leak = json_get_double(descriptor->params_json, descriptor->params_len, "integration_leak", 0.9995);
    if (integration_leak < 0.0) integration_leak = 0.0;
    if (integration_leak > 0.999999) integration_leak = 0.999999;
    double integration_gain = json_get_double(descriptor->params_json, descriptor->params_len, "integration_gain", 1.0);
    double integration_clamp = json_get_double(descriptor->params_json, descriptor->params_len, "integration_clamp", 1.2);
    if (integration_clamp <= 0.0) integration_clamp = 1.2;

    double base_slew_rate = json_get_double(descriptor->params_json, descriptor->params_len, "slew_rate", 12000.0);
    if (base_slew_rate < 0.0) base_slew_rate = 0.0;
    double slew_clamp = json_get_double(descriptor->params_json, descriptor->params_len, "slew_clamp", 1.2);
    if (slew_clamp <= 0.0) slew_clamp = 1.2;

    const EdgeRunnerParamView *freq_view = find_param(inputs, "freq");
    if (freq_view == NULL) {
        freq_view = find_param(inputs, "frequency");
    }
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amp");
    if (amp_view == NULL) {
        amp_view = find_param(inputs, "amplitude");
    }
    const EdgeRunnerParamView *pan_view = find_param(inputs, "pan");
    const EdgeRunnerParamView *reset_view = accept_reset ? find_param(inputs, "reset") : NULL;
    const EdgeRunnerParamView *phase_offset_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *frame_delay_view = find_param(inputs, "frame_delay");
    const EdgeRunnerParamView *slew_view = find_param(inputs, "slew");

    const double *tap_data = NULL;
    int tap_batches = 0;
    int tap_frames = 0;
    int tap_channels = 0;
    size_t tap_batch_stride = 0;
    size_t tap_channel_stride = 0;

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = freq_view != NULL ? freq_view : amp_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }
    if (inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        tap_data = inputs->audio.data;
        tap_batches = inputs->audio.batches > 0 ? (int)inputs->audio.batches : B;
        tap_frames = inputs->audio.frames > 0 ? (int)inputs->audio.frames : F;
        tap_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 0;
        if (tap_batches > 0) {
            B = tap_batches;
        }
        if (tap_frames > 0) {
            F = tap_frames;
        }
        if (tap_frames > 0) {
            tap_channel_stride = (size_t)tap_frames;
            tap_batch_stride = (size_t)tap_channels * (size_t)tap_frames;
        }
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;

    double *owned_freq = NULL;
    double *owned_amp = NULL;
    double *owned_pan = NULL;
    double *owned_reset = NULL;
    double *owned_phase_offset = NULL;
    double *owned_frame_delay = NULL;
    double *owned_slew = NULL;

    const double *freq = ensure_param_plane(freq_view, B, F, 0.0, &owned_freq);
    const double *amp = ensure_param_plane(amp_view, B, F, 1.0, &owned_amp);
    const double *pan = ensure_param_plane(pan_view, B, F, 0.0, &owned_pan);
    const double *reset = ensure_param_plane(reset_view, B, F, 0.0, &owned_reset);
    const double *phase_offset = ensure_param_plane(phase_offset_view, B, F, 0.0, &owned_phase_offset);
    const double *frame_delay = ensure_param_plane(frame_delay_view, B, F, 0.0, &owned_frame_delay);
    const double *slew_curve = ensure_param_plane(slew_view, B, F, -1.0, &owned_slew);

    bool has_tap_freq = tap_data != NULL && tap_channels > 0 && tap_frames > 0 && tap_batches > 0;

    if ((mode != OSC_MODE_OP_AMP && freq == NULL) || amp == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    if (state->u.osc.phase == NULL || state->u.osc.batches != B) {
        free(state->u.osc.phase);
        state->u.osc.phase = (double *)calloc((size_t)B, sizeof(double));
    }
    if (state->u.osc.phase_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.phase_buffer);
        state->u.osc.phase_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.wave_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.wave_buffer);
        state->u.osc.wave_buffer = (double *)malloc(total * sizeof(double));
    }
    if (state->u.osc.dphi_buffer == NULL || state->u.osc.batches != B) {
        free(state->u.osc.dphi_buffer);
        state->u.osc.dphi_buffer = (double *)malloc(total * sizeof(double));
    }
    if (mode != OSC_MODE_OP_AMP && strcmp(wave_buf, "triangle") == 0) {
        if (state->u.osc.tri_state == NULL || state->u.osc.batches != B) {
            free(state->u.osc.tri_state);
            state->u.osc.tri_state = (double *)calloc((size_t)B, sizeof(double));
        }
    }
    if (mode == OSC_MODE_INTEGRATOR) {
        if (state->u.osc.integrator_state == NULL || state->u.osc.batches != B) {
            free(state->u.osc.integrator_state);
            state->u.osc.integrator_state = (double *)calloc((size_t)B, sizeof(double));
        }
    }

    int driver_channels = (has_tap_freq && tap_channels > 1) ? tap_channels - 1 : 0;
    if (mode == OSC_MODE_OP_AMP) {
        if (state->u.osc.op_amp_state == NULL || state->u.osc.batches != B || state->u.osc.driver_channels != driver_channels) {
            free(state->u.osc.op_amp_state);
            state->u.osc.op_amp_state = (double *)calloc((size_t)B, sizeof(double));
            state->u.osc.driver_channels = driver_channels;
        }
    }

    state->u.osc.batches = B;
    state->u.osc.mode = mode;

    if (state->u.osc.phase == NULL || state->u.osc.phase_buffer == NULL || state->u.osc.wave_buffer == NULL || state->u.osc.dphi_buffer == NULL) {
        free(owned_freq);
        free(owned_amp);
        free(owned_pan);
        free(owned_reset);
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
        return -1;
    }

    for (int b = 0; b < B; ++b) {
        size_t batch_base = (size_t)b * (size_t)F;
        int tap_batch_index = has_tap_freq ? (b < tap_batches ? b : tap_batches - 1) : 0;
        const double *tap_base = has_tap_freq ? tap_data + (size_t)tap_batch_index * tap_batch_stride : NULL;
        for (int f = 0; f < F; ++f) {
            size_t idx = batch_base + (size_t)f;
            double hz = (freq != NULL) ? freq[idx] : 0.0;
            if (tap_base != NULL) {
                int tap_frame_index = f;
                if (tap_frame_index >= tap_frames) {
                    tap_frame_index = tap_frames - 1;
                } else if (tap_frame_index < 0) {
                    tap_frame_index = 0;
                }
                hz = tap_base[tap_frame_index];
            }
            state->u.osc.dphi_buffer[idx] = hz / sample_rate;
        }
    }

    const double *reset_ptr = accept_reset ? reset : NULL;
    phase_advance(state->u.osc.dphi_buffer, state->u.osc.phase_buffer, B, F, state->u.osc.phase, reset_ptr);

    if ((phase_offset != NULL && owned_phase_offset != NULL) || (frame_delay != NULL && owned_frame_delay != NULL)) {
        for (int b = 0; b < B; ++b) {
            size_t base = (size_t)b * (size_t)F;
            for (int f = 0; f < F; ++f) {
                size_t idx = base + (size_t)f;
                double ph = state->u.osc.phase_buffer[idx];
                if (phase_offset != NULL) {
                    ph += phase_offset[idx];
                }
                if (frame_delay != NULL) {
                    ph += state->u.osc.dphi_buffer[idx] * frame_delay[idx];
                }
                ph = ph - floor(ph);
                state->u.osc.phase_buffer[idx] = ph;
            }
        }
    }

    if (mode != OSC_MODE_OP_AMP) {
        if (strcmp(wave_buf, "saw") == 0) {
            osc_saw_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F);
        } else if (strcmp(wave_buf, "square") == 0) {
            osc_square_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, 0.5, state->u.osc.wave_buffer, B, F);
        } else if (strcmp(wave_buf, "triangle") == 0) {
            if (state->u.osc.tri_state == NULL) {
                state->u.osc.tri_state = (double *)calloc((size_t)B, sizeof(double));
            }
            osc_triangle_blep_c(state->u.osc.phase_buffer, state->u.osc.dphi_buffer, state->u.osc.wave_buffer, B, F, state->u.osc.tri_state);
        } else {
            for (int b = 0; b < B; ++b) {
                size_t base = (size_t)b * (size_t)F;
                for (int f = 0; f < F; ++f) {
                    size_t idx = base + (size_t)f;
                    state->u.osc.wave_buffer[idx] = sin(state->u.osc.phase_buffer[idx] * 2.0 * M_PI);
                }
            }
        }
    } else {
        double per_sample_default = base_slew_rate > 0.0 ? base_slew_rate / sample_rate : 0.0;
        for (int b = 0; b < B; ++b) {
            double op_state = state->u.osc.op_amp_state != NULL ? state->u.osc.op_amp_state[b] : 0.0;
            size_t base_wave = (size_t)b * (size_t)F;
            int tap_batch_index = has_tap_freq ? (b < tap_batches ? b : tap_batches - 1) : 0;
            const double *tap_base = has_tap_freq ? tap_data + (size_t)tap_batch_index * tap_batch_stride : NULL;
            const double *driver_base = (tap_base != NULL && driver_channels > 0) ? tap_base + tap_channel_stride : NULL;
            const double *slew_base = slew_curve != NULL ? slew_curve + base_wave : NULL;
            for (int f = 0; f < F; ++f) {
                double target = 0.0;
                if (driver_base != NULL) {
                    int tap_frame_index = f;
                    if (tap_frame_index >= tap_frames) {
                        tap_frame_index = tap_frames - 1;
                    } else if (tap_frame_index < 0) {
                        tap_frame_index = 0;
                    }
                    double sum = 0.0;
                    for (int ch = 0; ch < driver_channels; ++ch) {
                        sum += driver_base[(size_t)ch * tap_channel_stride + (size_t)tap_frame_index];
                    }
                    target = sum / (double)driver_channels;
                }
                double per_sample_slew = per_sample_default;
                if (slew_base != NULL) {
                    double candidate = slew_base[f];
                    if (candidate >= 0.0) {
                        per_sample_slew = candidate / sample_rate;
                    }
                }
                double delta = target - op_state;
                if (per_sample_slew > 0.0) {
                    if (delta > per_sample_slew) delta = per_sample_slew;
                    else if (delta < -per_sample_slew) delta = -per_sample_slew;
                }
                op_state += delta;
                if (op_state > slew_clamp) op_state = slew_clamp;
                else if (op_state < -slew_clamp) op_state = -slew_clamp;
                state->u.osc.wave_buffer[base_wave + (size_t)f] = op_state;
            }
            if (state->u.osc.op_amp_state != NULL) {
                state->u.osc.op_amp_state[b] = op_state;
            }
        }
    }

    if (mode == OSC_MODE_INTEGRATOR && state->u.osc.integrator_state != NULL) {
        for (int b = 0; b < B; ++b) {
            double accum = state->u.osc.integrator_state[b];
            size_t base = (size_t)b * (size_t)F;
            for (int f = 0; f < F; ++f) {
                size_t idx = base + (size_t)f;
                accum = accum * integration_leak + integration_gain * state->u.osc.wave_buffer[idx];
                if (accum > integration_clamp) accum = integration_clamp;
                else if (accum < -integration_clamp) accum = -integration_clamp;
                state->u.osc.wave_buffer[idx] = accum;
            }
            state->u.osc.integrator_state[b] = accum;
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
        free(owned_phase_offset);
        free(owned_frame_delay);
        free(owned_slew);
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
    free(owned_phase_offset);
    free(owned_frame_delay);
    free(owned_slew);

    state->u.osc.channels = 1;
    state->u.osc.stereo = stereo;

    *out_buffer = buffer;
    *out_channels = stereo ? 2 : 1;
    return 0;
}

