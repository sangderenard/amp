static int run_oscillator_pitch_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }

    double min_freq = json_get_double(descriptor->params_json, descriptor->params_len, "min_freq", 0.0);
    double default_slew = json_get_double(descriptor->params_json, descriptor->params_len, "default_slew", 0.0);
    if (default_slew < 0.0) {
        default_slew = 0.0;
    }

    const EdgeRunnerParamView *pitch_view = find_param(inputs, "pitch_hz");
    const EdgeRunnerParamView *root_view = find_param(inputs, "root_hz");
    const EdgeRunnerParamView *offset_view = find_param(inputs, "offset_cents");
    const EdgeRunnerParamView *add_view = find_param(inputs, "add_hz");
    const EdgeRunnerParamView *slew_view = find_param(inputs, "slew_hz_per_s");

    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *shape_source = pitch_view != NULL ? pitch_view : root_view;
    if (shape_source != NULL) {
        if (shape_source->batches > 0) {
            B = (int)shape_source->batches;
        }
        if (shape_source->frames > 0) {
            F = (int)shape_source->frames;
        }
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }

    double *owned_pitch = NULL;
    double *owned_root = NULL;
    double *owned_offset = NULL;
    double *owned_add = NULL;
    double *owned_slew = NULL;

    int has_direct = (pitch_view != NULL && pitch_view->data != NULL);
    const double *pitch_curve = ensure_param_plane(pitch_view, B, F, 0.0, &owned_pitch);
    const double *root_curve = NULL;
    const double *offset_curve = NULL;
    if (!has_direct) {
        root_curve = ensure_param_plane(root_view, B, F, min_freq > 0.0 ? min_freq : 0.0, &owned_root);
        offset_curve = ensure_param_plane(offset_view, B, F, 0.0, &owned_offset);
        if (root_curve == NULL || offset_curve == NULL) {
            free(owned_pitch);
            free(owned_root);
            free(owned_offset);
            return -1;
        }
    }
    const double *add_curve = ensure_param_plane(add_view, B, F, 0.0, &owned_add);
    const double *slew_curve = ensure_param_plane(slew_view, B, F, default_slew, &owned_slew);

    if (pitch_curve == NULL && !has_direct) {
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    size_t total = (size_t)B * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    if (state->u.osc_pitch.last_value == NULL || state->u.osc_pitch.batches != B) {
        free(state->u.osc_pitch.last_value);
        state->u.osc_pitch.last_value = (double *)calloc((size_t)B, sizeof(double));
        state->u.osc_pitch.batches = B;
    }
    if (state->u.osc_pitch.last_value == NULL) {
        free(buffer);
        free(owned_pitch);
        free(owned_root);
        free(owned_offset);
        free(owned_add);
        free(owned_slew);
        return -1;
    }

    using RowArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowArrayXXd target_array(B, F);
    if (has_direct && pitch_curve != NULL) {
        Eigen::Map<const RowArrayXXd> pitch_map(pitch_curve, B, F);
        target_array = pitch_map;
    } else if (!has_direct && root_curve != NULL && offset_curve != NULL) {
        Eigen::Map<const RowArrayXXd> root_map(root_curve, B, F);
        Eigen::Map<const RowArrayXXd> offset_map(offset_curve, B, F);
        target_array = root_map * ((offset_map / 1200.0 * M_LN2).exp());
    } else {
        target_array.setZero();
    }
    if (add_curve != NULL) {
        Eigen::Map<const RowArrayXXd> add_map(add_curve, B, F);
        target_array += add_map;
    }
    target_array = target_array.cwiseMax(min_freq);
    double default_limit = default_slew > 0.0 ? default_slew / sample_rate : 0.0;
    RowArrayXXd limit_array = RowArrayXXd::Constant(B, F, default_limit);
    if (slew_curve != NULL) {
        Eigen::Map<const RowArrayXXd> slew_map(slew_curve, B, F);
        limit_array = slew_map.unaryExpr([sample_rate](double per_sec) {
            if (per_sec <= 0.0) {
                return 0.0;
            }
            return per_sec / sample_rate;
        });
    }
    for (int b = 0; b < B; ++b) {
        double current = state->u.osc_pitch.last_value[b];
        size_t base = (size_t)b * (size_t)F;
        auto target_row = target_array.row(b);
        auto limit_row = limit_array.row(b);
        for (int f = 0; f < F; ++f) {
            size_t idx = base + (size_t)f;
            double target = target_row(f);
            double limit = limit_row(f);
            if (limit > 0.0) {
                double delta = target - current;
                if (delta > limit) {
                    delta = limit;
                } else if (delta < -limit) {
                    delta = -limit;
                }
                current += delta;
            } else {
                current = target;
            }
            if (current < min_freq) {
                current = min_freq;
            }
            buffer[idx] = current;
        }
        state->u.osc_pitch.last_value[b] = current;
    }

    state->u.osc_pitch.batches = B;

    free(owned_pitch);
    free(owned_root);
    free(owned_offset);
    free(owned_add);
    free(owned_slew);

    *out_buffer = buffer;
    *out_channels = 1;
    return 0;
}

static double alpha_lp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    return 1.0 - exp(-2.0 * M_PI * fc / sr);
}

static double alpha_hp(double fc, double sr) {
    if (fc < 1.0) {
        fc = 1.0;
    }
    double rc = 1.0 / (2.0 * M_PI * fc);
    return rc / (rc + 1.0 / sr);
}

