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

static int ensure_pitch_shift_state(node_state_t *state, int window_size, int hop_size, int resynthesis_hop) {
    if (state == NULL) {
        return -1;
    }
    if (window_size <= 0) {
        window_size = PITCH_SHIFT_DEFAULT_WINDOW;
    }
    if (!is_power_of_two_int(window_size)) {
        int next_power = 1;
        while (next_power < window_size && next_power < 131072) {
            next_power <<= 1;
        }
        window_size = next_power;
    }
    if (window_size <= 0) {
        window_size = PITCH_SHIFT_DEFAULT_WINDOW;
    }
    if (hop_size <= 0) {
        hop_size = PITCH_SHIFT_DEFAULT_HOP;
    }
    if (resynthesis_hop <= 0) {
        resynthesis_hop = PITCH_SHIFT_DEFAULT_RESYNTH_HOP;
    }

    int needs_resize = (state->u.pitch_shift.analysis_window == NULL)
        || (state->u.pitch_shift.window_size != window_size);
    if (needs_resize) {
        double *analysis = (double *)malloc((size_t)window_size * sizeof(double));
        double *synthesis = (double *)malloc((size_t)window_size * sizeof(double));
        double *prev_phase = (double *)malloc((size_t)window_size * sizeof(double));
        double *phase_accum = (double *)malloc((size_t)window_size * sizeof(double));
        if (analysis == NULL || synthesis == NULL || prev_phase == NULL || phase_accum == NULL) {
            free(analysis);
            free(synthesis);
            free(prev_phase);
            free(phase_accum);
            return -1;
        }
        free(state->u.pitch_shift.analysis_window);
        free(state->u.pitch_shift.synthesis_window);
        free(state->u.pitch_shift.prev_phase);
        free(state->u.pitch_shift.phase_accum);
        state->u.pitch_shift.analysis_window = analysis;
        state->u.pitch_shift.synthesis_window = synthesis;
        state->u.pitch_shift.prev_phase = prev_phase;
        state->u.pitch_shift.phase_accum = phase_accum;
        state->u.pitch_shift.window_size = window_size;
    }

    if (state->u.pitch_shift.analysis_window == NULL || state->u.pitch_shift.synthesis_window == NULL
        || state->u.pitch_shift.prev_phase == NULL || state->u.pitch_shift.phase_accum == NULL) {
        return -1;
    }

    state->u.pitch_shift.window_size = window_size;
    state->u.pitch_shift.hop_size = hop_size;
    state->u.pitch_shift.resynthesis_hop = resynthesis_hop;

    fill_window_weights(state->u.pitch_shift.analysis_window, window_size, FFT_WINDOW_HANN);
    fill_window_weights(state->u.pitch_shift.synthesis_window, window_size, FFT_WINDOW_HANN);
    memset(state->u.pitch_shift.prev_phase, 0, (size_t)window_size * sizeof(double));
    memset(state->u.pitch_shift.phase_accum, 0, (size_t)window_size * sizeof(double));

    return 0;
}

static double wrap_phase(double value) {
    while (value > M_PI) {
        value -= 2.0 * M_PI;
    }
    while (value < -M_PI) {
        value += 2.0 * M_PI;
    }
    return value;
}

static int pitch_shift_process(
    const double *input,
    int frames,
    double ratio,
    node_state_t *state,
    double *output
) {
    if (input == NULL || output == NULL || state == NULL) {
        return -1;
    }
    int window_size = state->u.pitch_shift.window_size;
    int hop_size = state->u.pitch_shift.hop_size;
    int resynthesis_hop = state->u.pitch_shift.resynthesis_hop;
    if (window_size <= 0 || hop_size <= 0) {
        return -1;
    }
    if (frames <= 0) {
        return 0;
    }

    if (ratio < 0.125) {
        ratio = 0.125;
    } else if (ratio > 8.0) {
        ratio = 8.0;
    }
    double time_scale = 1.0 / ratio;
    if (time_scale <= 0.0) {
        time_scale = 1.0;
    }
    double synthesis_hop = (double)resynthesis_hop * time_scale;
    if (synthesis_hop < 1e-6) {
        synthesis_hop = 1.0;
    }

    int padded_len = frames + window_size;
    int frame_count = 1;
    if (hop_size > 0) {
        frame_count = (frames + hop_size - 1) / hop_size + 1;
    }
    if (frame_count < 1) {
        frame_count = 1;
    }

    double *analysis_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *analysis_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *synth_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *synth_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *ifft_real = (double *)malloc((size_t)window_size * sizeof(double));
    double *ifft_imag = (double *)malloc((size_t)window_size * sizeof(double));
    double *frame_buffer = (double *)malloc((size_t)window_size * sizeof(double));
    double *input_padded = (double *)calloc((size_t)padded_len, sizeof(double));
    if (analysis_real == NULL || analysis_imag == NULL || synth_real == NULL || synth_imag == NULL
        || ifft_real == NULL || ifft_imag == NULL || frame_buffer == NULL || input_padded == NULL) {
        free(analysis_real);
        free(analysis_imag);
        free(synth_real);
        free(synth_imag);
        free(ifft_real);
        free(ifft_imag);
        free(frame_buffer);
        free(input_padded);
        return -1;
    }

    for (int i = 0; i < frames; ++i) {
        input_padded[i] = input[i];
    }

    double last_position = synthesis_hop * (double)(frame_count - 1);
    int stretched_len = (int)ceil(last_position + (double)window_size + 2.0);
    if (stretched_len < window_size) {
        stretched_len = window_size;
    }
    double *stretched = (double *)calloc((size_t)stretched_len, sizeof(double));
    if (stretched == NULL) {
        free(analysis_real);
        free(analysis_imag);
        free(synth_real);
        free(synth_imag);
        free(ifft_real);
        free(ifft_imag);
        free(frame_buffer);
        free(input_padded);
        return -1;
    }

    memset(state->u.pitch_shift.prev_phase, 0, (size_t)window_size * sizeof(double));
    memset(state->u.pitch_shift.phase_accum, 0, (size_t)window_size * sizeof(double));

    double out_pos = 0.0;
    for (int frame = 0; frame < frame_count; ++frame) {
        int start = frame * hop_size;
        for (int i = 0; i < window_size; ++i) {
            int idx = start + i;
            double sample = 0.0;
            if (idx >= 0 && idx < padded_len) {
                sample = input_padded[idx];
            }
            frame_buffer[i] = sample * state->u.pitch_shift.analysis_window[i];
        }

        fft_backend_transform(frame_buffer, NULL, analysis_real, analysis_imag, window_size, 0);

        for (int bin = 0; bin < window_size; ++bin) {
            double real = analysis_real[bin];
            double imag = analysis_imag[bin];
            double magnitude = sqrt(real * real + imag * imag);
            double phase = atan2(imag, real);
            double omega = 2.0 * M_PI * (double)bin * (double)hop_size / (double)window_size;
            double delta = wrap_phase(phase - state->u.pitch_shift.prev_phase[bin] - omega);
            double true_phase = omega + delta;
            state->u.pitch_shift.phase_accum[bin] += true_phase * time_scale;
            state->u.pitch_shift.prev_phase[bin] = phase;
            synth_real[bin] = magnitude * cos(state->u.pitch_shift.phase_accum[bin]);
            synth_imag[bin] = magnitude * sin(state->u.pitch_shift.phase_accum[bin]);
        }

        fft_backend_transform(synth_real, synth_imag, ifft_real, ifft_imag, window_size, 1);

        int pos_floor = (int)floor(out_pos);
        double pos_frac = out_pos - (double)pos_floor;
        for (int i = 0; i < window_size; ++i) {
            double sample = ifft_real[i] * state->u.pitch_shift.synthesis_window[i];
            int idx0 = pos_floor + i;
            int idx1 = idx0 + 1;
            if (idx0 >= 0 && idx0 < stretched_len) {
                stretched[idx0] += sample * (1.0 - pos_frac);
            }
            if (idx1 >= 0 && idx1 < stretched_len) {
                stretched[idx1] += sample * pos_frac;
            }
        }
        out_pos += synthesis_hop;
    }

    double input_energy = 0.0;
    for (int i = 0; i < frames; ++i) {
        input_energy += input[i] * input[i];
    }

    if (frames == 1) {
        output[0] = stretched_len > 0 ? stretched[0] : 0.0;
    } else {
        double scale = (double)(stretched_len - 1) / (double)(frames - 1);
        for (int i = 0; i < frames; ++i) {
            double pos = (double)i * scale;
            int base = (int)floor(pos);
            double frac = pos - (double)base;
            if (base < 0) {
                base = 0;
                frac = 0.0;
            }
            if (base >= stretched_len - 1) {
                base = stretched_len - 1;
                frac = 0.0;
            }
            double sample_a = stretched[base];
            double sample_b = stretched_len > base + 1 ? stretched[base + 1] : stretched[base];
            output[i] = sample_a + (sample_b - sample_a) * frac;
        }
    }

    double output_energy = 0.0;
    for (int i = 0; i < frames; ++i) {
        output_energy += output[i] * output[i];
    }
    if (output_energy > 1e-12 && input_energy > 1e-12) {
        double gain = sqrt(input_energy / output_energy);
        for (int i = 0; i < frames; ++i) {
            output[i] *= gain;
        }
    }

    free(analysis_real);
    free(analysis_imag);
    free(synth_real);
    free(synth_imag);
    free(ifft_real);
    free(ifft_imag);
    free(frame_buffer);
    free(input_padded);
    free(stretched);
    return 0;
}

