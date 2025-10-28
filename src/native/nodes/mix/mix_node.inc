static int run_mix_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int frames,
    double **out_buffer,
    int *out_channels
) {
    if (batches <= 0) {
        batches = 1;
    }
    if (frames <= 0) {
        frames = 1;
    }
    int target_channels = json_get_int(descriptor->params_json, descriptor->params_len, "channels", 1);
    if (target_channels <= 0) {
        target_channels = 1;
    }
    size_t total = (size_t)batches * (size_t)target_channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    if (buffer == NULL) {
        return -1;
    }
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || inputs->audio.channels == 0) {
        memset(buffer, 0, total * sizeof(double));
        *out_buffer = buffer;
        *out_channels = target_channels;
        return 0;
    }
    int in_channels = (int)inputs->audio.channels;
    const double *audio = inputs->audio.data;
    for (int b = 0; b < batches; ++b) {
        for (int f = 0; f < frames; ++f) {
            double sum = 0.0;
            for (int c = 0; c < in_channels; ++c) {
                size_t idx = ((size_t)b * (size_t)in_channels + (size_t)c) * (size_t)frames + (size_t)f;
                sum += audio[idx];
            }
            for (int oc = 0; oc < target_channels; ++oc) {
                size_t out_idx = ((size_t)b * (size_t)target_channels + (size_t)oc) * (size_t)frames + (size_t)f;
                buffer[out_idx] = sum;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = target_channels;
    return 0;
}
