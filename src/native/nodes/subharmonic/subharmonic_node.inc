static int run_subharm_node(
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
    if (inputs == NULL || !inputs->audio.has_audio || inputs->audio.data == NULL) {
        return -1;
    }
    int B = inputs->audio.batches > 0 ? (int)inputs->audio.batches : (batches > 0 ? batches : 1);
    int C = inputs->audio.channels > 0 ? (int)inputs->audio.channels : 1;
    int F = inputs->audio.frames > 0 ? (int)inputs->audio.frames : (frames > 0 ? frames : 1);
    if (sample_rate <= 0.0) {
        sample_rate = 48000.0;
    }
    const double *audio = inputs->audio.data;
    double band_lo = json_get_double(descriptor->params_json, descriptor->params_len, "band_lo", 70.0);
    double band_hi = json_get_double(descriptor->params_json, descriptor->params_len, "band_hi", 160.0);
    double mix = json_get_double(descriptor->params_json, descriptor->params_len, "mix", 0.5);
    double drive = json_get_double(descriptor->params_json, descriptor->params_len, "drive", 1.0);
    double out_hp = json_get_double(descriptor->params_json, descriptor->params_len, "out_hp", 25.0);
    int use_div4 = json_get_bool(descriptor->params_json, descriptor->params_len, "use_div4", 0);
    double a_hp_in = alpha_hp(band_lo, sample_rate);
    double a_lp_in = alpha_lp(band_hi, sample_rate);
    double a_sub2 = alpha_lp(fmax(band_hi / 3.0, 30.0), sample_rate);
    double a_sub4 = use_div4 ? alpha_lp(fmax(band_hi / 5.0, 20.0), sample_rate) : 0.0;
    double a_env_attack = alpha_lp(100.0, sample_rate);
    double a_env_release = alpha_lp(5.0, sample_rate);
    double a_hp_out = alpha_hp(out_hp, sample_rate);
    size_t total = (size_t)B * (size_t)C * (size_t)F;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    int need_resize = state->u.subharm.batches != B || state->u.subharm.channels != C || state->u.subharm.use_div4 != use_div4;
    if (need_resize) {
        free(state->u.subharm.hp_y);
        free(state->u.subharm.lp_y);
        free(state->u.subharm.prev);
        free(state->u.subharm.sign);
        free(state->u.subharm.ff2);
        free(state->u.subharm.ff4);
        free(state->u.subharm.ff4_count);
        free(state->u.subharm.sub2_lp);
        free(state->u.subharm.sub4_lp);
        free(state->u.subharm.env);
        free(state->u.subharm.hp_out_y);
        free(state->u.subharm.hp_out_x);
        state->u.subharm.hp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.lp_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.prev = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.sign = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.ff2 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
        state->u.subharm.sub2_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.env = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_y = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        state->u.subharm.hp_out_x = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        if (use_div4) {
            state->u.subharm.ff4 = (int8_t *)calloc((size_t)B * (size_t)C, sizeof(int8_t));
            state->u.subharm.ff4_count = (int32_t *)calloc((size_t)B * (size_t)C, sizeof(int32_t));
            state->u.subharm.sub4_lp = (double *)calloc((size_t)B * (size_t)C, sizeof(double));
        } else {
            free(state->u.subharm.ff4);
            free(state->u.subharm.ff4_count);
            free(state->u.subharm.sub4_lp);
            state->u.subharm.ff4 = NULL;
            state->u.subharm.ff4_count = NULL;
            state->u.subharm.sub4_lp = NULL;
        }
        state->u.subharm.batches = B;
        state->u.subharm.channels = C;
        state->u.subharm.use_div4 = use_div4;
    }
    if (state->u.subharm.hp_y == NULL || state->u.subharm.lp_y == NULL || state->u.subharm.prev == NULL || state->u.subharm.sign == NULL || state->u.subharm.ff2 == NULL || state->u.subharm.sub2_lp == NULL || state->u.subharm.env == NULL || state->u.subharm.hp_out_y == NULL || state->u.subharm.hp_out_x == NULL) {
        free(buffer);
        return -1;
    }
    subharmonic_process(
        audio,
        buffer,
        B,
        C,
        F,
        a_hp_in,
        a_lp_in,
        a_sub2,
        use_div4,
        a_sub4,
        a_env_attack,
        a_env_release,
        a_hp_out,
        drive,
        mix,
        state->u.subharm.hp_y,
        state->u.subharm.lp_y,
        state->u.subharm.prev,
        state->u.subharm.sign,
        state->u.subharm.ff2,
        state->u.subharm.ff4,
        state->u.subharm.ff4_count,
        state->u.subharm.sub2_lp,
        state->u.subharm.sub4_lp,
        state->u.subharm.env,
        state->u.subharm.hp_out_y,
        state->u.subharm.hp_out_x
    );
    *out_buffer = buffer;
    *out_channels = C;
    return 0;
}

