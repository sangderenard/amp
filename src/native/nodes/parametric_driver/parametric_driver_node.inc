static int run_parametric_driver_node(
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

    int mode = parse_driver_mode(descriptor->params_json, descriptor->params_len, state->u.driver.mode);
    double harmonics_buffer[32];
    int harmonic_count = 0;

    if (mode == DRIVER_MODE_PIEZO) {
        harmonics_buffer[0] = 1.0;
        harmonics_buffer[1] = 0.35;
        harmonics_buffer[2] = 0.12;
        harmonic_count = 3;
    } else if (mode == DRIVER_MODE_CUSTOM) {
        char harmonic_csv[256];
        if (json_copy_string(descriptor->params_json, descriptor->params_len, "harmonics", harmonic_csv, sizeof(harmonic_csv))) {
            harmonic_count = parse_csv_doubles(harmonic_csv, harmonics_buffer, 32);
        }
        if (harmonic_count <= 0) {
            harmonics_buffer[0] = 1.0;
            harmonic_count = 1;
        }
    } else {
        harmonics_buffer[0] = 1.0;
        harmonic_count = 1;
    }

    if (state->u.driver.harmonics == NULL || state->u.driver.harmonic_count != harmonic_count || state->u.driver.mode != mode) {
        free(state->u.driver.harmonics);
        state->u.driver.harmonics = (double *)malloc((size_t)harmonic_count * sizeof(double));
        if (state->u.driver.harmonics == NULL) {
            state->u.driver.harmonic_count = 0;
            return -1;
        }
        for (int i = 0; i < harmonic_count; ++i) {
            state->u.driver.harmonics[i] = harmonics_buffer[i];
        }
        state->u.driver.harmonic_count = harmonic_count;
    }
    state->u.driver.mode = mode;

    const EdgeRunnerParamView *amp_view = find_param(inputs, "amplitude");
    if (amp_view == NULL) {
        amp_view = find_param(inputs, "amp");
    }
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    if (phase_view == NULL) {
        phase_view = find_param(inputs, "phase");
    }
    const EdgeRunnerParamView *render_view = find_param(inputs, "render_mode");

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = amp_view != NULL ? amp_view : phase_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }

    const double *tap_data = NULL;
    int tap_batches = 0;
    int tap_frames = 0;
    int tap_channels = 0;
    size_t tap_batch_stride = 0;
    size_t tap_channel_stride = 0;
    if (inputs != NULL && inputs->audio.has_audio && inputs->audio.data != NULL) {
        tap_data = inputs->audio.data;
        tap_batches = inputs->audio.batches > 0 ? (int)inputs->audio.batches : B;
        tap_frames = inputs->audio.frames > 0 ? (int)inputs->audio.frames : F;
        tap_channels = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 0;
        if (tap_frames > 0) {
            tap_channel_stride = (size_t)tap_frames;
            tap_batch_stride = (size_t)tap_channels * (size_t)tap_frames;
        }
    }
    bool has_tap_freq = tap_data != NULL && tap_channels > 0 && tap_frames > 0 && tap_batches > 0;
    if (!has_tap_freq) {
        return -1;
    }
    if (tap_batches > 0) {
        B = tap_batches;
    }
    if (tap_frames > 0) {
        F = tap_frames;
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }

    double *owned_amp = NULL;
    double *owned_phase = NULL;
    double *owned_render = NULL;

    const double *amp = ensure_param_plane(amp_view, B, F, 1.0, &owned_amp);
    const double *phase_offset = ensure_param_plane(phase_view, B, F, 0.0, &owned_phase);
    const double *render_mode = ensure_param_plane(render_view, B, F, 0.0, &owned_render);
    if (amp == NULL) {
        free(owned_amp);
        free(owned_phase);
        free(owned_render);
        return -1;
    }

    if (state->u.driver.phase == NULL || state->u.driver.batches != B) {
        free(state->u.driver.phase);
        state->u.driver.phase = (double *)calloc((size_t)B, sizeof(double));
        state->u.driver.batches = B;
    }
    if (state->u.driver.phase == NULL || state->u.driver.harmonics == NULL) {
        free(owned_amp);
        free(owned_phase);
        free(owned_render);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        free(owned_amp);
        free(owned_phase);
        free(owned_render);
        return -1;
    }

    int aux_channels = tap_channels > 1 ? tap_channels - 1 : 0;
    size_t tap_aux_stride = aux_channels > 0 ? tap_channel_stride : 0;

    for (int b = 0; b < B; ++b) {
        double phase = state->u.driver.phase[b];
        size_t base = (size_t)b * (size_t)F;
        int tap_batch_index = (b < tap_batches ? b : tap_batches - 1);
        const double *tap_base = tap_data + (size_t)tap_batch_index * tap_batch_stride;
        const double *tap_aux_base = aux_channels > 0 ? tap_base + tap_aux_stride : NULL;
        for (int f = 0; f < F; ++f) {
            size_t idx = base + (size_t)f;
            int tap_frame_index = f;
            if (tap_frame_index >= tap_frames) {
                tap_frame_index = tap_frames - 1;
            } else if (tap_frame_index < 0) {
                tap_frame_index = 0;
            }
            double hz = tap_base[tap_frame_index];
            if (hz < 0.0) hz = 0.0;
            double advance = hz / sample_rate;
            phase += advance;
            phase = phase - floor(phase);
            double ph = phase;
            if (phase_offset != NULL) {
                ph += phase_offset[idx];
                ph = ph - floor(ph);
            }
            double sample = 0.0;
            if (mode == DRIVER_MODE_QUARTZ && state->u.driver.harmonic_count == 1) {
                sample = sin(2.0 * M_PI * ph);
            } else {
                for (int h = 0; h < state->u.driver.harmonic_count; ++h) {
                    double coeff = state->u.driver.harmonics[h];
                    if (coeff == 0.0) {
                        continue;
                    }
                    double harmonic_phase = 2.0 * M_PI * ph * (double)(h + 1);
                    sample += coeff * sin(harmonic_phase);
                }
            }
            double blend = render_mode != NULL ? render_mode[idx] : 0.0;
            if (blend < 0.0) {
                blend = 0.0;
            } else if (blend > 1.0) {
                blend = 1.0;
            }
            double stream_val = 0.0;
            if (tap_aux_base != NULL) {
                double sum = 0.0;
                for (int ch = 0; ch < aux_channels; ++ch) {
                    sum += tap_aux_base[(size_t)ch * tap_channel_stride + (size_t)tap_frame_index];
                }
                stream_val = sum / (double)aux_channels;
            }
            double combined = (1.0 - blend) * sample + blend * stream_val;
            buffer[idx] = combined * amp[idx];
        }
        state->u.driver.phase[b] = phase - floor(phase);
    }

    free(owned_amp);
    free(owned_phase);
    free(owned_render);

    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

