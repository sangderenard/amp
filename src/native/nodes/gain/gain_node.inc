static int run_gain_node(
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
    int channels = (int)inputs->audio.channels;
    if (!inputs->audio.has_audio || inputs->audio.data == NULL || channels <= 0) {
        channels = channels > 0 ? channels : 1;
        size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
        double *buffer = (double *)calloc(total, sizeof(double));
        if (buffer == NULL) {
            return -1;
        }
        *out_buffer = buffer;
        *out_channels = channels;
        return 0;
    }
    if (channels <= 0) {
        channels = 1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    double *buffer = (double *)malloc(total * sizeof(double));
    amp_last_alloc_count = total;
    if (buffer == NULL) {
        return -1;
    }
    const double *audio = inputs->audio.data;
    const EdgeRunnerParamView *gain_view = find_param(inputs, "gain");
    const double *gain = (gain_view != NULL) ? gain_view->data : NULL;
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            size_t base = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)frames;
            for (int f = 0; f < frames; ++f) {
                double sample = audio[base + (size_t)f];
                double g = gain != NULL ? gain[base + (size_t)f] : 1.0;
                buffer[base + (size_t)f] = sample * g;
            }
        }
    }
    *out_buffer = buffer;
    *out_channels = channels;
    return 0;
}

