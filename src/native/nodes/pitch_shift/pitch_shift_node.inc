static int run_pitch_shift_node(
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
    if (descriptor == NULL || inputs == NULL || out_buffer == NULL || out_channels == NULL || state == NULL) {
        return -1;
    }

    int window_size = json_get_int(descriptor->params_json, descriptor->params_len, "window_size", PITCH_SHIFT_DEFAULT_WINDOW);
    int hop_size = json_get_int(descriptor->params_json, descriptor->params_len, "hop_size", PITCH_SHIFT_DEFAULT_HOP);
    int resynthesis_hop = json_get_int(
        descriptor->params_json,
        descriptor->params_len,
        "resynthesis_hop",
        PITCH_SHIFT_DEFAULT_RESYNTH_HOP
    );

    if (ensure_pitch_shift_state(state, window_size, hop_size, resynthesis_hop) != 0) {
        return -1;
    }

    int B = batches > 0 ? batches : 1;
    int channels = (int)inputs->audio.channels;
    if (channels <= 0) {
        channels = 1;
    }
    int input_frames = frames > 0 ? frames : 1;
    if (inputs->audio.frames > 0) {
        input_frames = (int)inputs->audio.frames;
    }
    if (input_frames <= 0) {
        input_frames = 1;
    }

    size_t total = (size_t)B * (size_t)channels * (size_t)input_frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }

    const EdgeRunnerParamView *ratio_view = find_param(inputs, "ratio");
    const EdgeRunnerParamView *semitone_view = find_param(inputs, "semitones");
    double ratio_default = json_get_double(descriptor->params_json, descriptor->params_len, "ratio", 1.0);
    double semitone_default = json_get_double(descriptor->params_json, descriptor->params_len, "semitones", 0.0);
    double base_ratio = ratio_default;
    if (semitone_default != 0.0) {
        base_ratio *= pow(2.0, semitone_default / 12.0);
    }
    if (base_ratio <= 0.0) {
        base_ratio = 1.0;
    }

    double *owned_ratio = NULL;
    double *owned_semitone = NULL;
    const double *ratio_plane = NULL;
    const double *semitone_plane = NULL;
    if (ratio_view != NULL) {
        ratio_plane = ensure_param_plane(ratio_view, B, input_frames, base_ratio, &owned_ratio);
    }
    if (semitone_view != NULL) {
        semitone_plane = ensure_param_plane(semitone_view, B, input_frames, 0.0, &owned_semitone);
    }

    if (!inputs->audio.has_audio || inputs->audio.data == NULL) {
        memset(buffer, 0, total * sizeof(double));
        *out_buffer = buffer;
        *out_channels = channels;
        free(owned_ratio);
        free(owned_semitone);
        return 0;
    }

    const double *audio = inputs->audio.data;
    for (int b = 0; b < B; ++b) {
        double ratio_value = base_ratio;
        size_t idx_base = (size_t)b * (size_t)input_frames;
        if (semitone_plane != NULL) {
            size_t idx = idx_base + (size_t)(input_frames > 0 ? input_frames - 1 : 0);
            ratio_value = pow(2.0, semitone_plane[idx] / 12.0);
        } else if (ratio_plane != NULL) {
            size_t idx = idx_base + (size_t)(input_frames > 0 ? input_frames - 1 : 0);
            ratio_value = ratio_plane[idx];
        }
        if (ratio_value <= 0.0) {
            ratio_value = base_ratio;
        }
        for (int c = 0; c < channels; ++c) {
            size_t base = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)input_frames;
            const double *in_ptr = audio + base;
            double *out_ptr = buffer + base;
            if (pitch_shift_process(in_ptr, input_frames, ratio_value, state, out_ptr) != 0) {
                free(buffer);
                free(owned_ratio);
                free(owned_semitone);
                return -1;
            }
        }
    }

    free(owned_ratio);
    free(owned_semitone);
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

