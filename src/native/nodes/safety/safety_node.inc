static int run_safety_node(
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
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", (int)inputs->audio.channels);
    if (channels <= 0) {
        channels = (int)inputs->audio.channels;
    }
    if (channels <= 0) {
        channels = 1;
    }
    double alpha = json_get_double(descriptor->params_json, descriptor->params_len, "dc_alpha", 0.995);
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (state != NULL) {
        if (state->u.safety.state == NULL || state->u.safety.batches != batches || state->u.safety.channels != channels) {
            free(state->u.safety.state);
            state->u.safety.state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (state->u.safety.state == NULL) {
                free(buffer);
                state->u.safety.batches = 0;
                state->u.safety.channels = 0;
                return -1;
            }
            state->u.safety.batches = batches;
            state->u.safety.channels = channels;
        }
        state->u.safety.alpha = alpha;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL) {
        memset(buffer, 0, total * sizeof(double));
    } else {
        double *dc_state = NULL;
        if (state != NULL) {
            dc_state = state->u.safety.state;
        }
        if (dc_state == NULL) {
            dc_state = (double *)calloc((size_t)batches * (size_t)channels, sizeof(double));
            if (dc_state == NULL) {
                free(buffer);
                return -1;
            }
            if (state != NULL) {
                state->u.safety.state = dc_state;
                state->u.safety.batches = batches;
                state->u.safety.channels = channels;
            }
        }
        dc_block(inputs->audio.data, buffer, batches, channels, frames, alpha, dc_state);
        for (size_t i = 0; i < total; ++i) {
            double v = buffer[i];
            if (v > 1.0) {
                buffer[i] = 1.0;
            } else if (v < -1.0) {
                buffer[i] = -1.0;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

