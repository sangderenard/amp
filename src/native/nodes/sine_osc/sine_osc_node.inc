static int run_sine_osc_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    double initial_phase = json_get_double(descriptor->params_json, descriptor->params_len, "phase", 0.0);
    double normalized_phase = initial_phase - floor(initial_phase);
    if (state != NULL) {
        if (state->u.sine.phase == NULL || state->u.sine.batches != batches || state->u.sine.channels != channels) {
            free(state->u.sine.phase);
            state->u.sine.phase = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.sine.phase == NULL) {
                free(buffer);
                state->u.sine.batches = 0;
                state->u.sine.channels = 0;
                return -1;
            }
            state->u.sine.batches = batches;
            state->u.sine.channels = channels;
            state->u.sine.base_phase = normalized_phase;
            for (size_t idx = 0; idx < (size_t)batches * (size_t)channels; ++idx) {
                state->u.sine.phase[idx] = normalized_phase;
            }
        }
    }
    double *phase = state != NULL ? state->u.sine.phase : NULL;
    double base_freq = json_get_double(descriptor->params_json, descriptor->params_len, "frequency", 440.0);
    double base_amp = json_get_double(descriptor->params_json, descriptor->params_len, "amplitude", 0.5);
    const EdgeRunnerParamView *freq_view = find_param(inputs, "frequency");
    const EdgeRunnerParamView *amp_view = find_param(inputs, "amplitude");
    const double *freq_data = freq_view != NULL ? freq_view->data : NULL;
    const double *amp_data = amp_view != NULL ? amp_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t bc = (size_t)b * (size_t)channels + (size_t)c;
            double phase_acc = phase != NULL ? phase[bc] : normalized_phase;
            for (int f = 0; f < frames; ++f) {
                size_t idx = bc * (size_t)frames + (size_t)f;
                double freq = freq_data != NULL ? freq_data[idx] : base_freq;
                double amp = amp_data != NULL ? amp_data[idx] : base_amp;
                double step = freq / sample_rate;
                phase_acc += step;
                phase_acc -= floor(phase_acc);
                buffer[idx] = sin(phase_acc * 2.0 * M_PI) * amp;
            }
            if (phase != NULL) {
                phase[bc] = phase_acc;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

