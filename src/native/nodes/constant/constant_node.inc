static int run_constant_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    double value = json_get_double(descriptor->params_json, descriptor->params_len, "value", 0.0);
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    for (size_t i = 0; i < total; ++i) {
        buffer[i] = value;
    }
    if (state != NULL) {
        state->u.constant.value = value;
        state->u.constant.channels = channels;
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

static double render_lfo_wave(const char *wave, double phase) {
    if (wave != NULL) {
        if (strcmp(wave, "square") == 0) {
            return phase < 0.5 ? 1.0 : -1.0;
        }
        if (strcmp(wave, "saw") == 0) {
            double t = phase - floor(phase);
            return 2.0 * t - 1.0;
        }
        if (strcmp(wave, "triangle") == 0) {
            double t = phase - floor(phase);
            return 2.0 * fabs(2.0 * t - 1.0) - 1.0;
        }
    }
    return sin(phase * 2.0 * M_PI);
}

