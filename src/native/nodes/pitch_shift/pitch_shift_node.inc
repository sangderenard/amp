// Pitch shift node intentionally disabled; runtime never calls this stub.
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
    (void)descriptor;
    (void)inputs;
    (void)batches;
    (void)frames;
    (void)sample_rate;
    (void)out_buffer;
    (void)out_channels;
    (void)state;
    return -1;
}

