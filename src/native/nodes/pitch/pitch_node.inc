static int run_pitch_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    (void)sample_rate;
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    char grid_csv[1024];
    double grid_values[256];
    double grid_sorted_vals[256];
    double grid_ext[257];
    int grid_count = 0;
    if (json_copy_string(descriptor->params_json, descriptor->params_len, "grid_cents", grid_csv, sizeof(grid_csv))) {
        grid_count = parse_csv_doubles(grid_csv, grid_values, 256);
    }
    if (grid_count <= 0) {
        for (int i = 0; i < 12; ++i) {
            grid_values[i] = (double)i * 100.0;
        }
        grid_count = 12;
    }
    int grid_size = build_sorted_grid(grid_values, grid_count, grid_sorted_vals, grid_ext);
    int is_free_mode = json_get_bool(descriptor->params_json, descriptor->params_len, "is_free_mode", 0);
    char variant_buf[32];
    if (!json_copy_string(descriptor->params_json, descriptor->params_len, "free_variant", variant_buf, sizeof(variant_buf))) {
        strcpy(variant_buf, "continuous");
    }
    double span_default = json_get_double(descriptor->params_json, descriptor->params_len, "span_default", 2.0);
    int slew_enabled = json_get_bool(descriptor->params_json, descriptor->params_len, "slew", 1);
    const EdgeRunnerParamView *input_view = find_param(inputs, "input");
    const EdgeRunnerParamView *root_view = find_param(inputs, "root_midi");
    const EdgeRunnerParamView *span_view = find_param(inputs, "span_oct");
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    if (input_view != NULL && input_view->batches > 0) {
        B = (int)input_view->batches;
    }
    if (input_view != NULL && input_view->frames > 0) {
        F = (int)input_view->frames;
    }
    if (B <= 0) B = 1;
    if (F <= 0) F = 1;
    double *owned_input = NULL;
    double *owned_root = NULL;
    double *owned_span = NULL;
    const double *ctrl = ensure_param_plane(input_view, B, F, 0.0, &owned_input);
    const double *root = ensure_param_plane(root_view, B, F, 60.0, &owned_root);
    const double *span = ensure_param_plane(span_view, B, F, span_default, &owned_span);
    if (ctrl == NULL || root == NULL || span == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    size_t total = (size_t)B * (size_t)F;
    double *freq_target = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (freq_target == NULL) {
        free(owned_input);
        free(owned_root);
        free(owned_span);
        return -1;
    }
    using RowArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowArrayXXd> ctrl_map(ctrl, B, F);
    Eigen::Map<const RowArrayXXd> span_map(span, B, F);
    Eigen::Map<const RowArrayXXd> root_map(root, B, F);
    RowArrayXXd ctrl_scaled = ctrl_map * span_map;
    RowArrayXXd root_freq_array = ((root_map - 69.0) / 12.0 * M_LN2).exp() * 440.0;
    Eigen::Map<RowArrayXXd> freq_target_map(freq_target, B, F);
    int variant_weighted = strcmp(variant_buf, "weighted") == 0;
    int variant_stepped = strcmp(variant_buf, "stepped") == 0;
    if (is_free_mode && !variant_weighted && !variant_stepped) {
        RowArrayXXd cents = ctrl_scaled * 1200.0;
        freq_target_map = root_freq_array * ((cents / 1200.0 * M_LN2).exp());
    } else {
        for (int b = 0; b < B; ++b) {
            for (int f = 0; f < F; ++f) {
                double ctrl_scaled_val = ctrl_scaled(b, f);
                double cents = 0.0;
                if (is_free_mode) {
                    if (variant_weighted) {
                        double u = ctrl_scaled_val * (double)grid_size;
                        cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                    } else if (variant_stepped) {
                        double u = round(ctrl_scaled_val * (double)grid_size);
                        cents = grid_warp_inverse_value(u, grid_sorted_vals, grid_ext, grid_size);
                    } else {
                        cents = ctrl_scaled_val * 1200.0;
                    }
                } else {
                    double cents_unq = ctrl_scaled_val * 1200.0;
                    double u = grid_warp_forward_value(cents_unq, grid_sorted_vals, grid_ext, grid_size);
                    double u_round = round(u);
                    cents = grid_warp_inverse_value(u_round, grid_sorted_vals, grid_ext, grid_size);
                }
                freq_target_map(b, f) = root_freq_array(b, f) * exp((cents / 1200.0) * M_LN2);
            }
        }
    }
    free(owned_input);
    free(owned_root);
    free(owned_span);
    double *output = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (output == NULL) {
        free(freq_target);
        return -1;
    }
    if (slew_enabled) {
        if (state->u.pitch.last_freq == NULL || state->u.pitch.batches != B) {
            free(state->u.pitch.last_freq);
            state->u.pitch.last_freq = (double *)calloc((size_t)B, sizeof(double));
            state->u.pitch.batches = B;
        }
        if (state->u.pitch.last_freq == NULL) {
            free(freq_target);
            free(output);
            return -1;
        }
        Eigen::ArrayXd t_values;
        if (F > 1) {
            double t_last = (double)(F - 1) / (double)F;
            t_values = Eigen::ArrayXd::LinSpaced(F, 0.0, t_last);
        } else {
            t_values = Eigen::ArrayXd::Zero(1);
        }
        Eigen::ArrayXd hermite = 3.0 * t_values.square() - 2.0 * t_values.cube();
        for (int b = 0; b < B; ++b) {
            double y0 = state->u.pitch.last_freq[b];
            double y1 = freq_target_map(b, F - 1);
            Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>> out_row(output + (size_t)b * (size_t)F, F);
            out_row = hermite * (y1 - y0) + y0;
            state->u.pitch.last_freq[b] = y1;
        }
    } else {
        memcpy(output, freq_target, total * sizeof(double));
    }
    free(freq_target);
    *out_buffer = output;
    *out_channels = 1;
    return 0;
}

