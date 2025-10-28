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
    double epsilon_default = json_get_double(descriptor->params_json, descriptor->params_len, "stabilizer", 1e-9);
    double epsilon_json = json_get_double(descriptor->params_json, descriptor->params_len, "epsilon", epsilon_default);
    if (epsilon_json <= 0.0) {
        epsilon_json = 1e-9;
    }
    int max_batch_windows = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "max_batch_windows",
        32
    );
    if (max_batch_windows <= 0) {
        max_batch_windows = 1;
    }
    int enable_remainder = json_get_bool(
        descriptor->params_json,
        descriptor->params_len,
        "enable_remainder",
        1
    );

    int default_algorithm = parse_algorithm_string(descriptor->params_json, descriptor->params_len, FFT_ALGORITHM_EIGEN);
    default_algorithm = clamp_algorithm_kind(default_algorithm);
    int default_window_kind = parse_window_string(descriptor->params_json, descriptor->params_len, FFT_WINDOW_HANN);
    default_window_kind = clamp_window_kind(default_window_kind);

    if (ensure_fft_state_buffers(state, slot_count, window_size, max_batch_windows) != 0) {
        return -1;
    }
    state->u.fftdiv.batches = batches;
    state->u.fftdiv.channels = input_channels;
    state->u.fftdiv.algorithm = default_algorithm;
    state->u.fftdiv.epsilon = epsilon_json;
    state->u.fftdiv.enable_remainder = enable_remainder ? 1 : 0;
    state->u.fftdiv.max_batch_windows = max_batch_windows;
    if (state->u.fftdiv.window_kind != default_window_kind) {
        fill_window_weights(state->u.fftdiv.window, window_size, default_window_kind);
        state->u.fftdiv.window_kind = default_window_kind;
    }

    const EdgeRunnerParamView *divisor_view = find_param(inputs, "divisor");
    const EdgeRunnerParamView *divisor_imag_view = find_param(inputs, "divisor_imag");
    const EdgeRunnerParamView *algorithm_view = find_param(inputs, "algorithm_selector");
    const EdgeRunnerParamView *window_view = find_param(inputs, "window_selector");
    const EdgeRunnerParamView *stabilizer_view = find_param(inputs, "stabilizer");
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *lower_view = find_param(inputs, "lower_band");
    const EdgeRunnerParamView *upper_view = find_param(inputs, "upper_band");
    const EdgeRunnerParamView *filter_view = find_param(inputs, "filter_intensity");
    const EdgeRunnerParamView *carrier_views[FFT_DYNAMIC_CARRIER_LIMIT];
    uint32_t carrier_view_count = collect_dynamic_carrier_views(inputs, carrier_views, FFT_DYNAMIC_CARRIER_LIMIT);

    size_t total_samples = (size_t)slot_count * (size_t)frames;
    double *buffer = (double *)malloc(total_samples * sizeof(double));
    amp_last_alloc_count = total_samples;
    if (buffer == NULL) {
        return -1;
    }

    const double *audio_base = (inputs->audio.has_audio && inputs->audio.data != NULL) ? inputs->audio.data : NULL;
    size_t divisor_total = param_total_count(divisor_view);
    size_t divisor_imag_total = param_total_count(divisor_imag_view);
    size_t algorithm_total = param_total_count(algorithm_view);
    size_t window_total = param_total_count(window_view);
    size_t stabilizer_total = param_total_count(stabilizer_view);
    size_t phase_total = param_total_count(phase_view);
    size_t lower_total = param_total_count(lower_view);
    size_t upper_total = param_total_count(upper_view);
    size_t filter_total = param_total_count(filter_view);

    fft_dynamic_carrier_summary_t carrier_summary = summarize_dynamic_carriers(inputs);
    double *window_weights = state->u.fftdiv.window;
    int max_batch_capacity = state->u.fftdiv.batch_capacity > 0 ? state->u.fftdiv.batch_capacity : 1;
    int max_batch_runtime = state->u.fftdiv.max_batch_windows;
    if (max_batch_runtime <= 0 || max_batch_runtime > max_batch_capacity) {
        max_batch_runtime = max_batch_capacity;
    }
    int batch_count = 0;
    int batch_algorithm_kind = -1;
    const fft_algorithm_class_t *batch_algorithm_class = NULL;

    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        size_t base_index = (size_t)frame_index * (size_t)slot_count;
        double epsilon_frame = epsilon_json;
        if (stabilizer_total > 0U) {
            double candidate = read_param_value(stabilizer_view, base_index, epsilon_frame);
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon_frame = candidate;
            }
        }
        if (epsilon_frame < 1e-12) {
            epsilon_frame = 1e-12;
        }
        int algorithm_kind = default_algorithm;
        if (algorithm_total > 0U) {
            double raw = read_param_value(algorithm_view, base_index, (double)algorithm_kind);
            algorithm_kind = clamp_algorithm_kind(round_to_int(raw));
        }
        const fft_algorithm_class_t *algorithm_class = select_fft_algorithm(algorithm_kind);
        if (algorithm_class == NULL) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_hook && !amp_fft_backend_has_hook()) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_power_of_two && !is_power_of_two_int(window_size)) {
            algorithm_class = select_fft_algorithm(FFT_ALGORITHM_DFT);
            if (algorithm_class == NULL) {
                free(buffer);
                return AMP_E_UNSUPPORTED;
            }
        }
        algorithm_kind = algorithm_class->kind;
        int window_kind = default_window_kind;
        if (window_total > 0U) {
            double raw_w = read_param_value(window_view, base_index, (double)window_kind);
            window_kind = clamp_window_kind(round_to_int(raw_w));
        }
        if (state->u.fftdiv.window_kind != window_kind) {
            fill_window_weights(state->u.fftdiv.window, window_size, window_kind);
            state->u.fftdiv.window_kind = window_kind;
        }
        state->u.fftdiv.algorithm = algorithm_kind;
        state->u.fftdiv.epsilon = epsilon_frame;
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
        } else if (algorithm_class->supports_dynamic_carriers) {
            state->u.fftdiv.dynamic_carrier_band_count = (int)carrier_summary.band_count;
            state->u.fftdiv.dynamic_carrier_last_sum = carrier_summary.last_sum;
            state->u.fftdiv.dynamic_k_active = (int)carrier_summary.band_count;
        } else {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
        }

        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            if (batch_count > 0 && batch_algorithm_class != NULL) {
                fftdiv_flush_batch(state, batch_algorithm_class, window_size, batch_count, buffer);
                batch_count = 0;
                batch_algorithm_kind = -1;
                batch_algorithm_class = NULL;
            }
        }

        int filled = state->u.fftdiv.filled;
        double phase_last = state->u.fftdiv.last_phase;
        double lower_last = state->u.fftdiv.last_lower;
        double upper_last = state->u.fftdiv.last_upper;
        double filter_last = state->u.fftdiv.last_filter;
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = 0.0;
            if (audio_base != NULL) {
                sample = audio_base[data_idx];
            }
            double divisor_sample = divisor_total > 0U ? read_param_value(divisor_view, base_index + (size_t)slot, 1.0) : 1.0;
            double divisor_imag_sample = divisor_imag_total > 0U ? read_param_value(divisor_imag_view, base_index + (size_t)slot, 0.0) : 0.0;
            double phase_sample = phase_total > 0U ? read_param_value(phase_view, base_index + (size_t)slot, phase_last) : phase_last;
            double lower_sample = lower_total > 0U ? read_param_value(lower_view, base_index + (size_t)slot, lower_last) : lower_last;
            double upper_sample = upper_total > 0U ? read_param_value(upper_view, base_index + (size_t)slot, upper_last) : upper_last;
            double filter_sample = filter_total > 0U ? read_param_value(filter_view, base_index + (size_t)slot, filter_last) : filter_last;
            size_t offset = (size_t)slot * (size_t)window_size;
            double *input_ring = state->u.fftdiv.input_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            if (filled < window_size) {
                input_ring[filled] = sample;
                divisor_ring[filled] = divisor_sample;
                divisor_imag_ring[filled] = divisor_imag_sample;
                phase_ring[filled] = phase_sample;
                lower_ring[filled] = lower_sample;
                upper_ring[filled] = upper_sample;
                filter_ring[filled] = filter_sample;
            } else {
                if (window_size > 1) {
                    memmove(input_ring, input_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_ring, divisor_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_imag_ring, divisor_imag_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(phase_ring, phase_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(lower_ring, lower_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(upper_ring, upper_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(filter_ring, filter_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                }
                input_ring[window_size - 1] = sample;
                divisor_ring[window_size - 1] = divisor_sample;
                divisor_imag_ring[window_size - 1] = divisor_imag_sample;
                phase_ring[window_size - 1] = phase_sample;
                lower_ring[window_size - 1] = lower_sample;
                upper_ring[window_size - 1] = upper_sample;
                filter_ring[window_size - 1] = filter_sample;
            }
            phase_last = phase_sample;
            lower_last = lower_sample;
            upper_last = upper_sample;
            filter_last = filter_sample;
        }
        state->u.fftdiv.last_phase = phase_last;
        state->u.fftdiv.last_lower = lower_last;
        state->u.fftdiv.last_upper = upper_last;
        state->u.fftdiv.last_filter = filter_last;
        if (filled < window_size) {
            filled += 1;
            state->u.fftdiv.filled = filled;
        }
        if (state->u.fftdiv.filled < window_size) {
            if (batch_count > 0 && batch_algorithm_class != NULL) {
                fftdiv_flush_batch(state, batch_algorithm_class, window_size, batch_count, buffer);
                batch_count = 0;
                batch_algorithm_kind = -1;
                batch_algorithm_class = NULL;
            }
            for (int slot = 0; slot < slot_count; ++slot) {
                double divisor_sample = divisor_total > 0U ? read_param_value(divisor_view, base_index + (size_t)slot, 1.0) : 1.0;
                double safe_div = fabs(divisor_sample) < epsilon_frame ? (divisor_sample >= 0.0 ? epsilon_frame : -epsilon_frame) : divisor_sample;
                double sample = audio_base != NULL ? audio_base[base_index + (size_t)slot] : 0.0;
                buffer[base_index + (size_t)slot] = sample / safe_div;
            }
            continue;
        }
        int dynamic_active_frame = 0;
        double dynamic_sum_frame = 0.0;
        state->u.fftdiv.dynamic_k_active = 0;
        state->u.fftdiv.remainder_energy = 0.0;
        for (int slot = 0; slot < slot_count; ++slot) {
            size_t offset = (size_t)slot * (size_t)window_size;
            double *input_ring = state->u.fftdiv.input_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
            double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
            double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_clamped = clamp_unit_double(lower_mod);
            double upper_clamped = clamp_unit_double(upper_mod);
            if (upper_clamped < lower_clamped) {
                double tmp_bounds = lower_clamped;
                lower_clamped = upper_clamped;
                upper_clamped = tmp_bounds;
            }
            double intensity_clamped = clamp_unit_double(filter_mod);
            double cos_phase = cos(phase_mod);
            double sin_phase = sin(phase_mod);

            if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
                double carrier_fnorm[FFT_DYNAMIC_CARRIER_LIMIT];
                int carrier_indices_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double theta0[FFT_DYNAMIC_CARRIER_LIMIT];
                double step_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double step_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                double b_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double b_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double div_re_acc[FFT_DYNAMIC_CARRIER_LIMIT];
                double div_im_acc[FFT_DYNAMIC_CARRIER_LIMIT];
                double last_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double last_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double gram_real[FFT_DYNAMIC_CARRIER_LIMIT * FFT_DYNAMIC_CARRIER_LIMIT];
                double gram_imag[FFT_DYNAMIC_CARRIER_LIMIT * FFT_DYNAMIC_CARRIER_LIMIT];
                double amp_re[FFT_DYNAMIC_CARRIER_LIMIT];
                double amp_im[FFT_DYNAMIC_CARRIER_LIMIT];
                double ph_re_buffer[FFT_DYNAMIC_CARRIER_LIMIT];
                double ph_im_buffer[FFT_DYNAMIC_CARRIER_LIMIT];
                int active_count = 0;
                double carrier_sum_slot = 0.0;
                for (uint32_t idx = 0; idx < carrier_view_count && idx < FFT_DYNAMIC_CARRIER_LIMIT; ++idx) {
                    const EdgeRunnerParamView *carrier_view = carrier_views[idx];
                    if (carrier_view == NULL) {
                        continue;
                    }
                    size_t total = param_total_count(carrier_view);
                    if (total == 0U) {
                        continue;
                    }
                    double raw_value = read_param_value(
                        carrier_view,
                        base_index + (size_t)slot,
                        0.0
                    );
                    carrier_sum_slot += raw_value;
                    double normalized = raw_value;
                    if (sample_rate > 0.0 && fabs(normalized) > 1.0) {
                        normalized = raw_value / sample_rate;
                    }
                    normalized = clamp_unit_double(normalized);
                    carrier_fnorm[active_count] = normalized;
                    carrier_indices_local[active_count] = (int)idx;
                    active_count += 1;
                }
                double *remainder_ring = state->u.fftdiv.result_buffer + offset;
                for (int i = 0; i < window_size; ++i) {
                    state->u.fftdiv.div_fft_real[offset + (size_t)i] = 0.0;
                    state->u.fftdiv.div_fft_imag[offset + (size_t)i] = 0.0;
                    remainder_ring[i] = 0.0;
                }
                if (active_count > 0) {
                    size_t phase_offset = (size_t)slot * (size_t)FFT_DYNAMIC_CARRIER_LIMIT;
                    double inv_window = window_size > 0 ? 1.0 / (double)window_size : 1.0;
                    int has_divisor = (divisor_total > 0U) || (divisor_imag_total > 0U);
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta = state->u.fftdiv.dynamic_phase[phase_index];
                        theta0[k] = theta;
                        double fnorm = carrier_fnorm[k];
                        double angle = 2.0 * M_PI * fnorm;
                        double step_re = cos(angle);
                        double step_im = sin(angle);
                        step_re_local[k] = step_re;
                        step_im_local[k] = step_im;
                        state->u.fftdiv.dynamic_step_re[phase_index] = step_re;
                        state->u.fftdiv.dynamic_step_im[phase_index] = step_im;
                        ph_re_buffer[k] = cos(theta);
                        ph_im_buffer[k] = sin(theta);
                        b_re[k] = 0.0;
                        b_im[k] = 0.0;
                        div_re_acc[k] = 0.0;
                        div_im_acc[k] = 0.0;
                        last_re[k] = ph_re_buffer[k];
                        last_im[k] = ph_im_buffer[k];
                        for (int j = 0; j < active_count; ++j) {
                            gram_real[k * active_count + j] = 0.0;
                            gram_imag[k * active_count + j] = 0.0;
                        }
                    }
                    double phi_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    double phi_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    for (int n = 0; n < window_size; ++n) {
                        double w = window_weights != NULL ? window_weights[n] : 1.0;
                        double xw = input_ring[n] * w;
                        for (int k = 0; k < active_count; ++k) {
                            double ph_re = ph_re_buffer[k];
                            double ph_im = ph_im_buffer[k];
                            double wr = ph_re * w;
                            double wi = ph_im * w;
                            phi_re_local[k] = wr;
                            phi_im_local[k] = wi;
                            b_re[k] += xw * wr;
                            b_im[k] -= xw * wi;
                            if (has_divisor) {
                                double dw_re = divisor_ring[n] * w;
                                double dw_im = divisor_imag_ring[n] * w;
                                div_re_acc[k] += dw_re * ph_re + dw_im * ph_im;
                                div_im_acc[k] += -dw_re * ph_im + dw_im * ph_re;
                            }
                            if (n == window_size - 1) {
                                last_re[k] = ph_re;
                                last_im[k] = ph_im;
                            }
                        }
                        for (int i_local = 0; i_local < active_count; ++i_local) {
                            for (int j_local = i_local; j_local < active_count; ++j_local) {
                                double a_re = phi_re_local[i_local];
                                double a_im = phi_im_local[i_local];
                                double b_re_local = phi_re_local[j_local];
                                double b_im_local = phi_im_local[j_local];
                                double re = a_re * b_re_local + a_im * b_im_local;
                                double im = a_re * b_im_local - a_im * b_re_local;
                                gram_real[i_local * active_count + j_local] += re;
                                gram_imag[i_local * active_count + j_local] += im;
                                if (i_local != j_local) {
                                    gram_real[j_local * active_count + i_local] += re;
                                    gram_imag[j_local * active_count + i_local] -= im;
                                }
                            }
                        }
                        for (int k = 0; k < active_count; ++k) {
                            double next_re = ph_re_buffer[k] * step_re_local[k] - ph_im_buffer[k] * step_im_local[k];
                            double next_im = ph_re_buffer[k] * step_im_local[k] + ph_im_buffer[k] * step_re_local[k];
                            ph_re_buffer[k] = next_re;
                            ph_im_buffer[k] = next_im;
                        }
                    }
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta_next = theta0[k] + 2.0 * M_PI * carrier_fnorm[k] * (double)window_size;
                        state->u.fftdiv.dynamic_phase[phase_index] = wrap_phase_two_pi(theta_next);
                        if (!has_divisor) {
                            div_re_acc[k] = 1.0;
                            div_im_acc[k] = 0.0;
                        }
                    }
                    double coeff_re_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    double coeff_im_local[FFT_DYNAMIC_CARRIER_LIMIT];
                    if (state->u.fftdiv.enable_remainder) {
                        int dim = active_count * 2;
                        double normal_matrix[FFT_DYNAMIC_CARRIER_LIMIT * 2 * FFT_DYNAMIC_CARRIER_LIMIT * 2];
                        double rhs_vec[FFT_DYNAMIC_CARRIER_LIMIT * 2];
                        for (int i_local = 0; i_local < dim * dim; ++i_local) {
                            normal_matrix[i_local] = 0.0;
                        }
                        for (int i_local = 0; i_local < dim; ++i_local) {
                            rhs_vec[i_local] = 0.0;
                        }
                        double trace = 0.0;
                        for (int k = 0; k < active_count; ++k) {
                            trace += gram_real[k * active_count + k];
                        }
                        double lambda = 0.0;
                        if (active_count > 0 && trace > 0.0) {
                            lambda = 1e-8 * (trace / (double)active_count);
                        }
                        for (int i_local = 0; i_local < active_count; ++i_local) {
                            for (int j_local = 0; j_local < active_count; ++j_local) {
                                double real_part = gram_real[i_local * active_count + j_local];
                                double imag_part = gram_imag[i_local * active_count + j_local];
                                if (i_local == j_local) {
                                    real_part += lambda;
                                }
                                normal_matrix[i_local * dim + j_local] = real_part;
                                normal_matrix[i_local * dim + (j_local + active_count)] = -imag_part;
                                normal_matrix[(i_local + active_count) * dim + j_local] = imag_part;
                                normal_matrix[(i_local + active_count) * dim + (j_local + active_count)] = real_part;
                            }
                            rhs_vec[i_local] = b_re[i_local];
                            rhs_vec[i_local + active_count] = b_im[i_local];
                        }
                        if (solve_linear_system(normal_matrix, rhs_vec, dim) == 0) {
                            for (int k = 0; k < active_count; ++k) {
                                coeff_re_local[k] = rhs_vec[k];
                                coeff_im_local[k] = rhs_vec[k + active_count];
                            }
                        } else {
                            for (int k = 0; k < active_count; ++k) {
                                coeff_re_local[k] = b_re[k];
                                coeff_im_local[k] = b_im[k];
                            }
                        }
                    } else {
                        for (int k = 0; k < active_count; ++k) {
                            coeff_re_local[k] = b_re[k];
                            coeff_im_local[k] = b_im[k];
                        }
                    }
                    double sample_dynamic = 0.0;
                    for (int k = 0; k < active_count; ++k) {
                        double coeff_re = coeff_re_local[k];
                        double coeff_im = coeff_im_local[k];
                        double div_re = div_re_acc[k];
                        double div_im = div_im_acc[k];
                        if (has_divisor) {
                            double denom = div_re * div_re + div_im * div_im;
                            if (denom < epsilon_frame) {
                                denom = epsilon_frame;
                            }
                            double real_tmp = (coeff_re * div_re + coeff_im * div_im) / denom;
                            double imag_tmp = (coeff_im * div_re - coeff_re * div_im) / denom;
                            coeff_re = real_tmp;
                            coeff_im = imag_tmp;
                        }
                        double ratio = carrier_fnorm[k];
                        if (ratio < 0.0) {
                            ratio = 0.0;
                        } else if (ratio > 1.0) {
                            ratio = 1.0;
                        }
                        double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                        double gated_re = coeff_re * gain;
                        double gated_im = coeff_im * gain;
                        double rotated_re = gated_re * cos_phase - gated_im * sin_phase;
                        double rotated_im = gated_re * sin_phase + gated_im * cos_phase;
                        amp_re[k] = rotated_re;
                        amp_im[k] = rotated_im;
                        sample_dynamic += (rotated_re * last_re[k] - rotated_im * last_im[k]) * inv_window;
                    }
                    if (state->u.fftdiv.enable_remainder) {
                        double *modeled_ring = state->u.fftdiv.work_real;
                        for (int i = 0; i < window_size; ++i) {
                            modeled_ring[i] = 0.0;
                        }
                        for (int k = 0; k < active_count; ++k) {
                            ph_re_buffer[k] = cos(theta0[k]);
                            ph_im_buffer[k] = sin(theta0[k]);
                        }
                        double remainder_energy_local = 0.0;
                        for (int n = 0; n < window_size; ++n) {
                            double sum_val = 0.0;
                            for (int k = 0; k < active_count; ++k) {
                                sum_val += amp_re[k] * ph_re_buffer[k] - amp_im[k] * ph_im_buffer[k];
                            }
                            modeled_ring[n] = sum_val * inv_window;
                            double residual_val = input_ring[n] - modeled_ring[n];
                            remainder_ring[n] = residual_val;
                            remainder_energy_local += residual_val * residual_val;
                            for (int k = 0; k < active_count; ++k) {
                                double next_re = ph_re_buffer[k] * step_re_local[k] - ph_im_buffer[k] * step_im_local[k];
                                double next_im = ph_re_buffer[k] * step_im_local[k] + ph_im_buffer[k] * step_re_local[k];
                                ph_re_buffer[k] = next_re;
                                ph_im_buffer[k] = next_im;
                            }
                        }
                        state->u.fftdiv.remainder_energy += remainder_energy_local;
                    } else {
                        size_t tail_index = (size_t)(window_size > 0 ? window_size - 1 : 0);
                        remainder_ring[tail_index] = sample_dynamic;
                    }
                    buffer[base_index + (size_t)slot] = sample_dynamic;
                    if (active_count > dynamic_active_frame) {
                        dynamic_active_frame = active_count;
                    }
                    dynamic_sum_frame += carrier_sum_slot;
                    continue;
                }
                buffer[base_index + (size_t)slot] = 0.0;
                continue;
            }

            if (batch_count > 0 && algorithm_kind != batch_algorithm_kind) {
                if (batch_algorithm_class != NULL) {
                    fftdiv_flush_batch(state, batch_algorithm_class, window_size, batch_count, buffer);
                }
                batch_count = 0;
                batch_algorithm_kind = -1;
                batch_algorithm_class = NULL;
            }
            if (batch_count == 0) {
                batch_algorithm_kind = algorithm_kind;
                batch_algorithm_class = algorithm_class;
            }
            if (batch_count >= max_batch_runtime) {
                if (batch_algorithm_class != NULL) {
                    fftdiv_flush_batch(state, batch_algorithm_class, window_size, batch_count, buffer);
                }
                batch_count = 0;
                batch_algorithm_kind = -1;
                batch_algorithm_class = NULL;
            }

            size_t batch_offset = (size_t)batch_count * (size_t)window_size;
            double *batched_real = state->u.fftdiv.batch_input_real + batch_offset;
            double *batched_imag = state->u.fftdiv.batch_input_imag + batch_offset;
            double *batched_div_real = state->u.fftdiv.batch_div_real + batch_offset;
            double *batched_div_imag = state->u.fftdiv.batch_div_imag + batch_offset;

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                batched_real[i] = input_ring[i] * w;
                batched_imag[i] = 0.0;
                batched_div_real[i] = divisor_ring[i] * w;
                batched_div_imag[i] = divisor_imag_ring[i] * w;
            }

            state->u.fftdiv.batch_slot_map[batch_count] = slot;
            state->u.fftdiv.batch_slot_offset[batch_count] = offset;
            state->u.fftdiv.batch_output_index[batch_count] = base_index + (size_t)slot;
            state->u.fftdiv.batch_phase[batch_count] = phase_mod;
            state->u.fftdiv.batch_lower[batch_count] = lower_mod;
            state->u.fftdiv.batch_upper[batch_count] = upper_mod;
            state->u.fftdiv.batch_filter[batch_count] = filter_mod;
            state->u.fftdiv.batch_epsilon[batch_count] = epsilon_frame;

            batch_count += 1;
            if (batch_count >= max_batch_runtime) {
                if (batch_algorithm_class != NULL) {
                    fftdiv_flush_batch(state, batch_algorithm_class, window_size, batch_count, buffer);
                }
                batch_count = 0;
                batch_algorithm_kind = -1;
                batch_algorithm_class = NULL;
            }
        }
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            state->u.fftdiv.dynamic_carrier_band_count = dynamic_active_frame;
            state->u.fftdiv.dynamic_carrier_last_sum = dynamic_sum_frame;
            state->u.fftdiv.dynamic_k_active = dynamic_active_frame;
        }
        state->u.fftdiv.position = window_size - 1;
    }

    *out_buffer = buffer;
    *out_channels = input_channels;
    if (metrics != NULL) {
        metrics->measured_delay_frames = (uint32_t)(window_size > 0 ? window_size - 1 : 0);
        double complexity = 0.0;
        if (state->u.fftdiv.algorithm == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            complexity = (double)slot_count * (double)state->u.fftdiv.dynamic_k_active * (double)window_size * 8.0;
        } else if (
            state->u.fftdiv.algorithm == FFT_ALGORITHM_EIGEN
            || state->u.fftdiv.algorithm == FFT_ALGORITHM_HOOK
        ) {
            double stages = log((double)window_size) / log(2.0);
            if (stages < 1.0) {
                stages = 1.0;
            }
            complexity = (double)slot_count * (double)window_size * (stages * 6.0 + 4.0);
        } else {
            complexity = (double)slot_count * (double)window_size * (double)window_size * 2.0;
        }
        float heat = (float)(complexity / 1000.0);
        if (heat < 0.001f) {
            heat = 0.001f;
        }
        metrics->accumulated_heat = heat;
        state->u.fftdiv.total_heat += (double)heat;
        metrics->reserved[0] = state->u.fftdiv.last_phase;
        metrics->reserved[1] = state->u.fftdiv.last_lower;
        metrics->reserved[2] = state->u.fftdiv.last_upper;
        metrics->reserved[3] = state->u.fftdiv.last_filter;
        metrics->reserved[4] = (double)window_size;
        metrics->reserved[5] = (double)state->u.fftdiv.algorithm;
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
    double epsilon_default = json_get_double(descriptor->params_json, descriptor->params_len, "stabilizer", 1e-9);
    double epsilon_json = json_get_double(descriptor->params_json, descriptor->params_len, "epsilon", epsilon_default);
    if (epsilon_json <= 0.0) {
        epsilon_json = 1e-9;
    }
    int max_batch_windows = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "max_batch_windows",
        32
    );
    if (max_batch_windows <= 0) {
        max_batch_windows = 1;
    }

    int default_algorithm = parse_algorithm_string(descriptor->params_json, descriptor->params_len, FFT_ALGORITHM_EIGEN);
    default_algorithm = clamp_algorithm_kind(default_algorithm);
    int default_window_kind = parse_window_string(descriptor->params_json, descriptor->params_len, FFT_WINDOW_HANN);
    default_window_kind = clamp_window_kind(default_window_kind);

    if (ensure_fft_state_buffers(state, slot_count, window_size, max_batch_windows) != 0) {
        return -1;
    }
    state->u.fftdiv.max_batch_windows = max_batch_windows;
    if (state->u.fftdiv.window_kind != default_window_kind) {
        fill_window_weights(state->u.fftdiv.window, window_size, default_window_kind);
        state->u.fftdiv.window_kind = default_window_kind;
    }

    const EdgeRunnerParamView *divisor_view = find_param(inputs, "divisor");
    const EdgeRunnerParamView *divisor_imag_view = find_param(inputs, "divisor_imag");
    const EdgeRunnerParamView *stabilizer_view = find_param(inputs, "stabilizer");
    const EdgeRunnerParamView *algorithm_view = find_param(inputs, "algorithm_selector");
    const EdgeRunnerParamView *window_view = find_param(inputs, "window_selector");
    const EdgeRunnerParamView *phase_view = find_param(inputs, "phase_offset");
    const EdgeRunnerParamView *lower_view = find_param(inputs, "lower_band");
    const EdgeRunnerParamView *upper_view = find_param(inputs, "upper_band");
    const EdgeRunnerParamView *filter_view = find_param(inputs, "filter_intensity");

    size_t divisor_total = param_total_count(divisor_view);
    size_t divisor_imag_total = param_total_count(divisor_imag_view);
    size_t stabilizer_total = param_total_count(stabilizer_view);
    size_t algorithm_total = param_total_count(algorithm_view);
    size_t window_total = param_total_count(window_view);
    size_t phase_total = param_total_count(phase_view);
    size_t lower_total = param_total_count(lower_view);
    size_t upper_total = param_total_count(upper_view);
    size_t filter_total = param_total_count(filter_view);

    const EdgeRunnerParamView *carrier_views[FFT_DYNAMIC_CARRIER_LIMIT];
    uint32_t carrier_view_count = collect_dynamic_carrier_views(inputs, carrier_views, FFT_DYNAMIC_CARRIER_LIMIT);

    size_t total_samples = (size_t)slot_count * (size_t)frames;
    double *buffer = (double *)malloc(total_samples * sizeof(double));
    amp_last_alloc_count = total_samples;
    if (buffer == NULL) {
        return -1;
    }

    const double *audio_base = (inputs->audio.has_audio && inputs->audio.data != NULL) ? inputs->audio.data : NULL;
    int recomb_filled = state->u.fftdiv.recomb_filled;
    double phase_last_state = state->u.fftdiv.last_phase;
    double lower_last_state = state->u.fftdiv.last_lower;
    double upper_last_state = state->u.fftdiv.last_upper;
    double filter_last_state = state->u.fftdiv.last_filter;

    fft_dynamic_carrier_summary_t carrier_summary = summarize_dynamic_carriers(inputs);

    for (int frame_index = 0; frame_index < frames; ++frame_index) {
        size_t base_index = (size_t)frame_index * (size_t)slot_count;
        double epsilon_frame = epsilon_json;
        if (stabilizer_total > 0U) {
            double candidate = read_param_value(stabilizer_view, base_index, epsilon_frame);
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon_frame = candidate;
            }
        }
        if (epsilon_frame < 1e-12) {
            epsilon_frame = 1e-12;
        }
        int algorithm_kind = default_algorithm;
        if (algorithm_total > 0U) {
            double raw = read_param_value(algorithm_view, base_index, (double)algorithm_kind);
            algorithm_kind = clamp_algorithm_kind(round_to_int(raw));
        }
        const fft_algorithm_class_t *algorithm_class = select_fft_algorithm(algorithm_kind);
        if (algorithm_class == NULL) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_hook && !amp_fft_backend_has_hook()) {
            free(buffer);
            return AMP_E_UNSUPPORTED;
        }
        if (algorithm_class->requires_power_of_two && !is_power_of_two_int(window_size)) {
            algorithm_class = select_fft_algorithm(FFT_ALGORITHM_DFT);
            if (algorithm_class == NULL) {
                free(buffer);
                return AMP_E_UNSUPPORTED;
            }
        }
        algorithm_kind = algorithm_class->kind;
        if (algorithm_kind == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            double *window_weights = state->u.fftdiv.window;
            double inv_window = window_size > 0 ? 1.0 / (double)window_size : 1.0;
            int dynamic_active_frame = 0;
            double dynamic_sum_frame = 0.0;
            state->u.fftdiv.dynamic_k_active = 0;
            state->u.fftdiv.algorithm = algorithm_kind;
            state->u.fftdiv.epsilon = epsilon_frame;
            for (int slot = 0; slot < slot_count; ++slot) {
                size_t offset = (size_t)slot * (size_t)window_size;
                double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
                double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
                double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;
                double *phase_ring = state->u.fftdiv.phase_buffer + offset;
                double *lower_ring = state->u.fftdiv.lower_buffer + offset;
                double *upper_ring = state->u.fftdiv.upper_buffer + offset;
                double *filter_ring = state->u.fftdiv.filter_buffer + offset;
                double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
                double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
                double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
                double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
                double lower_clamped = clamp_unit_double(lower_mod);
                double upper_clamped = clamp_unit_double(upper_mod);
                if (upper_clamped < lower_clamped) {
                    double tmp_bounds = lower_clamped;
                    lower_clamped = upper_clamped;
                    upper_clamped = tmp_bounds;
                }
                double intensity_clamped = clamp_unit_double(filter_mod);
                double cos_phase = cos(phase_mod);
                double sin_phase = sin(phase_mod);
                int has_divisor = (divisor_total > 0U) || (divisor_imag_total > 0U);
                double carrier_fnorm[FFT_DYNAMIC_CARRIER_LIMIT];
                int carrier_indices_local[FFT_DYNAMIC_CARRIER_LIMIT];
                int active_count = 0;
                double carrier_sum_slot = 0.0;
                for (uint32_t idx = 0; idx < carrier_view_count && idx < FFT_DYNAMIC_CARRIER_LIMIT; ++idx) {
                    const EdgeRunnerParamView *carrier_view = carrier_views[idx];
                    if (carrier_view == NULL) {
                        continue;
                    }
                    size_t total = param_total_count(carrier_view);
                    if (total == 0U) {
                        continue;
                    }
                    double raw_value = read_param_value(
                        carrier_view,
                        base_index + (size_t)slot,
                        0.0
                    );
                    carrier_sum_slot += raw_value;
                    double normalized = raw_value;
                    if (sample_rate > 0.0 && fabs(normalized) > 1.0) {
                        normalized = raw_value / sample_rate;
                    }
                    normalized = clamp_unit_double(normalized);
                    carrier_fnorm[active_count] = normalized;
                    carrier_indices_local[active_count] = (int)idx;
                    active_count += 1;
                }
                if (active_count > 0) {
                    size_t phase_offset = (size_t)slot * (size_t)FFT_DYNAMIC_CARRIER_LIMIT;
                    double *phasor_re = state->u.fftdiv.work_real;
                    double *phasor_im = state->u.fftdiv.work_imag;
                    double sample_dynamic = 0.0;
                    for (int k = 0; k < active_count; ++k) {
                        size_t carrier_index = (size_t)carrier_indices_local[k];
                        size_t phase_index = phase_offset + carrier_index;
                        double theta = state->u.fftdiv.dynamic_phase[phase_index];
                        double fnorm = carrier_fnorm[k];
                        double angle = 2.0 * M_PI * fnorm;
                        double step_re = cos(angle);
                        double step_im = sin(angle);
                        state->u.fftdiv.dynamic_step_re[phase_index] = step_re;
                        state->u.fftdiv.dynamic_step_im[phase_index] = step_im;
                        double ph_re = cos(theta);
                        double ph_im = sin(theta);
                        double div_re_acc = 0.0;
                        double div_im_acc = 0.0;
                        double last_re = ph_re;
                        double last_im = ph_im;
                        for (int n = 0; n < window_size; ++n) {
                            double w = window_weights != NULL ? window_weights[n] : 1.0;
                            phasor_re[n] = ph_re;
                            phasor_im[n] = ph_im;
                            if (has_divisor) {
                                double dw_re = divisor_ring[n] * w;
                                double dw_im = divisor_imag_ring[n] * w;
                                div_re_acc += dw_re * ph_re + dw_im * ph_im;
                                div_im_acc += -dw_re * ph_im + dw_im * ph_re;
                            }
                            if (n == window_size - 1) {
                                last_re = ph_re;
                                last_im = ph_im;
                            }
                            double next_re = ph_re * step_re - ph_im * step_im;
                            double next_im = ph_re * step_im + ph_im * step_re;
                            ph_re = next_re;
                            ph_im = next_im;
                        }
                        double theta_next = state->u.fftdiv.dynamic_phase[phase_index] + angle * (double)window_size;
                        state->u.fftdiv.dynamic_phase[phase_index] = wrap_phase_two_pi(theta_next);
                        if (!has_divisor) {
                            div_re_acc = 1.0;
                            div_im_acc = 0.0;
                        }
                        double ratio = clamp_unit_double(fnorm);
                        double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                        if (gain < 1e-6) {
                            gain = 1e-6;
                        }
                        double rotated_re = last_re * cos_phase - last_im * sin_phase;
                        double rotated_im = last_re * sin_phase + last_im * cos_phase;
                        double scale_re = rotated_re * gain;
                        double scale_im = rotated_im * gain;
                        if (has_divisor) {
                            double denom = div_re_acc * div_re_acc + div_im_acc * div_im_acc;
                            if (denom < epsilon_frame) {
                                denom = epsilon_frame;
                            }
                            double tmp_re = (scale_re * div_re_acc + scale_im * div_im_acc) / denom;
                            double tmp_im = (scale_im * div_re_acc - scale_re * div_im_acc) / denom;
                            scale_re = tmp_re;
                            scale_im = tmp_im;
                        }
                        for (int n = 0; n < window_size; ++n) {
                            double w = window_weights != NULL ? window_weights[n] : 1.0;
                            double coeff = inv_window * w * (scale_re * phasor_re[n] + scale_im * phasor_im[n]);
                            sample_dynamic += coeff * recomb_ring[n];
                        }
                    }
                    buffer[base_index + (size_t)slot] = sample_dynamic;
                    if (active_count > dynamic_active_frame) {
                        dynamic_active_frame = active_count;
                    }
                    dynamic_sum_frame += carrier_sum_slot;
                    continue;
                }
                buffer[base_index + (size_t)slot] = 0.0;
            }
            state->u.fftdiv.dynamic_carrier_band_count = dynamic_active_frame;
            state->u.fftdiv.dynamic_carrier_last_sum = dynamic_sum_frame;
            state->u.fftdiv.dynamic_k_active = dynamic_active_frame;
            state->u.fftdiv.position = window_size > 0 ? window_size - 1 : 0;
            continue;
        }
        int window_kind = default_window_kind;
        if (window_total > 0U) {
            double raw_w = read_param_value(window_view, base_index, (double)window_kind);
            window_kind = clamp_window_kind(round_to_int(raw_w));
        }
        if (state->u.fftdiv.window_kind != window_kind) {
            fill_window_weights(state->u.fftdiv.window, window_size, window_kind);
            state->u.fftdiv.window_kind = window_kind;
        }
        state->u.fftdiv.algorithm = algorithm_kind;
        state->u.fftdiv.epsilon = epsilon_frame;
        if (algorithm_class->supports_dynamic_carriers) {
            state->u.fftdiv.dynamic_carrier_band_count = (int)carrier_summary.band_count;
            state->u.fftdiv.dynamic_carrier_last_sum = carrier_summary.last_sum;
        } else {
            state->u.fftdiv.dynamic_carrier_band_count = 0;
            state->u.fftdiv.dynamic_carrier_last_sum = 0.0;
        }

        double phase_last = phase_last_state;
        double lower_last = lower_last_state;
        double upper_last = upper_last_state;
        double filter_last = filter_last_state;

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t data_idx = base_index + (size_t)slot;
            double sample = audio_base != NULL ? audio_base[data_idx] : 0.0;
            size_t offset = (size_t)slot * (size_t)window_size;
            double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
            double *divisor_imag_ring = state->u.fftdiv.divisor_imag_buffer + offset;

            size_t tail_index = (size_t)(window_size > 0 ? window_size - 1 : 0);
            double phase_default = (recomb_filled < window_size) ? phase_ring[recomb_filled] : phase_ring[tail_index];
            double lower_default = (recomb_filled < window_size) ? lower_ring[recomb_filled] : lower_ring[tail_index];
            double upper_default = (recomb_filled < window_size) ? upper_ring[recomb_filled] : upper_ring[tail_index];
            double filter_default = (recomb_filled < window_size) ? filter_ring[recomb_filled] : filter_ring[tail_index];
            double divisor_default = (recomb_filled < window_size) ? divisor_ring[recomb_filled] : divisor_ring[tail_index];
            double divisor_imag_default = (recomb_filled < window_size) ? divisor_imag_ring[recomb_filled] : divisor_imag_ring[tail_index];

            double phase_sample = phase_total > 0U
                ? read_param_value(phase_view, base_index + (size_t)slot, phase_default)
                : phase_default;
            double lower_sample = lower_total > 0U
                ? read_param_value(lower_view, base_index + (size_t)slot, lower_default)
                : lower_default;
            double upper_sample = upper_total > 0U
                ? read_param_value(upper_view, base_index + (size_t)slot, upper_default)
                : upper_default;
            double filter_sample = filter_total > 0U
                ? read_param_value(filter_view, base_index + (size_t)slot, filter_default)
                : filter_default;
            double divisor_sample = divisor_total > 0U
                ? read_param_value(divisor_view, base_index + (size_t)slot, divisor_default)
                : divisor_default;
            double divisor_imag_sample = divisor_imag_total > 0U
                ? read_param_value(divisor_imag_view, base_index + (size_t)slot, divisor_imag_default)
                : divisor_imag_default;

            if (recomb_filled < window_size) {
                recomb_ring[recomb_filled] = sample;
                phase_ring[recomb_filled] = phase_sample;
                lower_ring[recomb_filled] = lower_sample;
                upper_ring[recomb_filled] = upper_sample;
                filter_ring[recomb_filled] = filter_sample;
                divisor_ring[recomb_filled] = divisor_sample;
                divisor_imag_ring[recomb_filled] = divisor_imag_sample;
            } else {
                if (window_size > 1) {
                    memmove(recomb_ring, recomb_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(phase_ring, phase_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(lower_ring, lower_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(upper_ring, upper_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(filter_ring, filter_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_ring, divisor_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                    memmove(divisor_imag_ring, divisor_imag_ring + 1, (size_t)(window_size - 1) * sizeof(double));
                }
                recomb_ring[window_size - 1] = sample;
                phase_ring[window_size - 1] = phase_sample;
                lower_ring[window_size - 1] = lower_sample;
                upper_ring[window_size - 1] = upper_sample;
                filter_ring[window_size - 1] = filter_sample;
                divisor_ring[window_size - 1] = divisor_sample;
                divisor_imag_ring[window_size - 1] = divisor_imag_sample;
            }

            phase_last = phase_sample;
            lower_last = lower_sample;
            upper_last = upper_sample;
            filter_last = filter_sample;
        }

        phase_last_state = phase_last;
        lower_last_state = lower_last;
        upper_last_state = upper_last;
        filter_last_state = filter_last;
        state->u.fftdiv.last_phase = phase_last;
        state->u.fftdiv.last_lower = lower_last;
        state->u.fftdiv.last_upper = upper_last;
        state->u.fftdiv.last_filter = filter_last;
        if (recomb_filled < window_size) {
            recomb_filled += 1;
            if (recomb_filled > window_size) {
                recomb_filled = window_size;
            }
            for (int slot = 0; slot < slot_count; ++slot) {
                size_t data_idx = base_index + (size_t)slot;
                size_t offset = (size_t)slot * (size_t)window_size;
                double *divisor_ring = state->u.fftdiv.divisor_buffer + offset;
                double divisor_sample = divisor_ring[recomb_filled > 0 ? recomb_filled - 1 : 0];
                if (divisor_total > 0U) {
                    divisor_sample = read_param_value(divisor_view, base_index + (size_t)slot, divisor_sample);
                }
                double safe_div = fabs(divisor_sample) < epsilon_frame
                    ? (divisor_sample >= 0.0 ? epsilon_frame : -epsilon_frame)
                    : divisor_sample;
                double quotient_sample = audio_base != NULL ? audio_base[data_idx] : 0.0;
                buffer[data_idx] = quotient_sample * safe_div;
            }
            continue;
        }

        double *work_real = state->u.fftdiv.work_real;
        double *work_imag = state->u.fftdiv.work_imag;
        double *div_real = state->u.fftdiv.div_real;
        double *div_imag = state->u.fftdiv.div_imag;
        double *ifft_real = state->u.fftdiv.ifft_real;
        double *ifft_imag = state->u.fftdiv.ifft_imag;
        double *window_weights = state->u.fftdiv.window;

        for (int slot = 0; slot < slot_count; ++slot) {
            size_t offset = (size_t)slot * (size_t)window_size;
            double *recomb_ring = state->u.fftdiv.recomb_buffer + offset;
            double *phase_ring = state->u.fftdiv.phase_buffer + offset;
            double *lower_ring = state->u.fftdiv.lower_buffer + offset;
            double *upper_ring = state->u.fftdiv.upper_buffer + offset;
            double *filter_ring = state->u.fftdiv.filter_buffer + offset;
            double phase_mod = phase_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_mod = lower_ring[window_size > 0 ? window_size - 1 : 0];
            double upper_mod = upper_ring[window_size > 0 ? window_size - 1 : 0];
            double filter_mod = filter_ring[window_size > 0 ? window_size - 1 : 0];
            double lower_clamped = clamp_unit_double(lower_mod);
            double upper_clamped = clamp_unit_double(upper_mod);
            if (upper_clamped < lower_clamped) {
                double tmp_bounds = lower_clamped;
                lower_clamped = upper_clamped;
                upper_clamped = tmp_bounds;
            }
            double intensity_clamped = clamp_unit_double(filter_mod);
            double cos_phase = cos(phase_mod);
            double sin_phase = sin(phase_mod);

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                div_real[i] = state->u.fftdiv.divisor_buffer[offset + (size_t)i] * w;
                div_imag[i] = state->u.fftdiv.divisor_imag_buffer[offset + (size_t)i] * w;
            }
            algorithm_class->forward(div_real, div_imag, div_real, div_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                state->u.fftdiv.div_fft_real[offset + (size_t)i] = div_real[i];
                state->u.fftdiv.div_fft_imag[offset + (size_t)i] = div_imag[i];
            }

            for (int i = 0; i < window_size; ++i) {
                double w = window_weights != NULL ? window_weights[i] : 1.0;
                work_real[i] = recomb_ring[i] * w;
                work_imag[i] = 0.0;
            }
            algorithm_class->forward(work_real, work_imag, work_real, work_imag, window_size);
            for (int i = 0; i < window_size; ++i) {
                double ratio = window_size > 1 ? (double)i / (double)(window_size - 1) : 0.0;
                double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                if (gain < 1e-6) {
                    gain = 1e-6;
                }
                double rotated_real = work_real[i];
                double rotated_imag = work_imag[i];
                double inv_real = rotated_real * cos_phase + rotated_imag * sin_phase;
                double inv_imag = -rotated_real * sin_phase + rotated_imag * cos_phase;
                inv_real /= gain;
                inv_imag /= gain;
                double c = state->u.fftdiv.div_fft_real[offset + (size_t)i];
                double d = state->u.fftdiv.div_fft_imag[offset + (size_t)i];
                double out_real_freq = inv_real * c - inv_imag * d;
                double out_imag_freq = inv_real * d + inv_imag * c;
                work_real[i] = out_real_freq;
                work_imag[i] = out_imag_freq;
            }
            algorithm_class->inverse(work_real, work_imag, ifft_real, ifft_imag, window_size);
            buffer[base_index + (size_t)slot] = ifft_real[window_size - 1];
        }
    }

    state->u.fftdiv.recomb_filled = recomb_filled;
    *out_buffer = buffer;
    *out_channels = input_channels;
    if (metrics != NULL) {
        metrics->measured_delay_frames = (uint32_t)(window_size > 0 ? window_size - 1 : 0);
        double complexity = 0.0;
        if (state->u.fftdiv.algorithm == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            complexity = (double)slot_count * (double)state->u.fftdiv.dynamic_k_active * (double)window_size * 8.0;
        } else if (
            state->u.fftdiv.algorithm == FFT_ALGORITHM_EIGEN
            || state->u.fftdiv.algorithm == FFT_ALGORITHM_HOOK
        ) {
            double stages = log((double)window_size) / log(2.0);
            if (stages < 1.0) {
                stages = 1.0;
            }
            complexity = (double)slot_count * (double)window_size * (stages * 6.0 + 4.0);
        } else {
            complexity = (double)slot_count * (double)window_size * (double)window_size * 2.0;
        }
        float heat = (float)(complexity / 1000.0);
        if (heat < 0.001f) {
            heat = 0.001f;
        }
        metrics->accumulated_heat = heat;
        state->u.fftdiv.total_heat += (double)heat;
        metrics->reserved[0] = state->u.fftdiv.last_phase;
        metrics->reserved[1] = state->u.fftdiv.last_lower;
        metrics->reserved[2] = state->u.fftdiv.last_upper;
        metrics->reserved[3] = state->u.fftdiv.last_filter;
        metrics->reserved[4] = (double)window_size;
        metrics->reserved[5] = (double)state->u.fftdiv.algorithm;
    }
    return 0;
}

