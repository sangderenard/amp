static int run_envelope_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    node_state_t *state
) {
    if (out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    int B = batches > 0 ? batches : 1;
    int F = frames > 0 ? frames : 1;
    const EdgeRunnerParamView *trigger_view = find_param(inputs, "trigger");
    const EdgeRunnerParamView *gate_view = find_param(inputs, "gate");
    const EdgeRunnerParamView *drone_view = find_param(inputs, "drone");
    const EdgeRunnerParamView *velocity_view = find_param(inputs, "velocity");
    const EdgeRunnerParamView *send_reset_view = find_param(inputs, "send_reset");
    if (trigger_view != NULL && trigger_view->batches > 0) {
        B = (int)trigger_view->batches;
    }
    if (trigger_view != NULL && trigger_view->frames > 0) {
        F = (int)trigger_view->frames;
    }
    if (B <= 0) {
        B = 1;
    }
    if (F <= 0) {
        F = 1;
    }
    double attack_ms = json_get_double(descriptor->params_json, descriptor->params_len, "attack_ms", 12.0);
    double hold_ms = json_get_double(descriptor->params_json, descriptor->params_len, "hold_ms", 8.0);
    double decay_ms = json_get_double(descriptor->params_json, descriptor->params_len, "decay_ms", 90.0);
    double sustain_level = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_level", 0.7);
    double sustain_ms = json_get_double(descriptor->params_json, descriptor->params_len, "sustain_ms", 0.0);
    double release_ms = json_get_double(descriptor->params_json, descriptor->params_len, "release_ms", 220.0);
    int send_resets_default = json_get_bool(descriptor->params_json, descriptor->params_len, "send_resets", 1);
    int atk_frames = (int)lrint((attack_ms / 1000.0) * sample_rate);
    int hold_frames = (int)lrint((hold_ms / 1000.0) * sample_rate);
    int dec_frames = (int)lrint((decay_ms / 1000.0) * sample_rate);
    int sus_frames = (int)lrint((sustain_ms / 1000.0) * sample_rate);
    int rel_frames = (int)lrint((release_ms / 1000.0) * sample_rate);
    if (atk_frames < 0) atk_frames = 0;
    if (hold_frames < 0) hold_frames = 0;
    if (dec_frames < 0) dec_frames = 0;
    if (sus_frames < 0) sus_frames = 0;
    if (rel_frames < 0) rel_frames = 0;
    double *owned_trigger = NULL;
    double *owned_gate = NULL;
    double *owned_drone = NULL;
    double *owned_velocity = NULL;
    const double *trigger = ensure_param_plane(trigger_view, B, F, 0.0, &owned_trigger);
    const double *gate = ensure_param_plane(gate_view, B, F, 0.0, &owned_gate);
    const double *drone = ensure_param_plane(drone_view, B, F, 0.0, &owned_drone);
    const double *velocity = ensure_param_plane(velocity_view, B, F, 1.0, &owned_velocity);
    if (trigger == NULL || gate == NULL || drone == NULL || velocity == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double send_reset_value = read_scalar_param(send_reset_view, (double)send_resets_default);
    int send_reset_flag = send_reset_value >= 0.5 ? 1 : 0;
    size_t total = (size_t)B * (size_t)F * 2;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    double *amp_plane = buffer;
    double *reset_plane = buffer + (size_t)B * (size_t)F;
    if (state == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.batches != B) {
        free(state->u.envelope.stage);
        free(state->u.envelope.value);
        free(state->u.envelope.timer);
        free(state->u.envelope.velocity);
        free(state->u.envelope.activations);
        free(state->u.envelope.release_start);
        state->u.envelope.stage = (int *)calloc((size_t)B, sizeof(int));
        state->u.envelope.value = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.timer = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.velocity = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.activations = (int64_t *)calloc((size_t)B, sizeof(int64_t));
        state->u.envelope.release_start = (double *)calloc((size_t)B, sizeof(double));
        state->u.envelope.batches = B;
    }
    if (state->u.envelope.stage == NULL || state->u.envelope.value == NULL || state->u.envelope.timer == NULL) {
        free(buffer);
        free(owned_trigger);
        free(owned_gate);
        free(owned_drone);
        free(owned_velocity);
        return -1;
    }
    envelope_process(
        trigger,
        gate,
        drone,
        velocity,
        B,
        F,
        atk_frames,
        hold_frames,
        dec_frames,
        sus_frames,
        rel_frames,
        sustain_level,
        send_reset_flag,
        state->u.envelope.stage,
        state->u.envelope.value,
        state->u.envelope.timer,
        state->u.envelope.velocity,
        state->u.envelope.activations,
        state->u.envelope.release_start,
        amp_plane,
        reset_plane
    );
    free(owned_trigger);
    free(owned_gate);
    free(owned_drone);
    free(owned_velocity);
    *out_buffer = buffer;
    *out_channels = 2;
    return 0;
}

