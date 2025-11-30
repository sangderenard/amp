#include "include/amp_native.h"

static int run_sampler_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
) {
    (void)inputs;
    (void)sample_rate;
    (void)history;
    (void)mode;
    (void)metrics;
    if (descriptor == NULL || out_buffer == NULL || out_channels == NULL) {
        return -1;
    }
    const char *node_name = descriptor->name != NULL ? descriptor->name : NULL;
    if (node_name == NULL) {
        return -1;
    }
    const double *samples = NULL;
    size_t total_frames = 0;
    uint32_t available_channels = 0;
    size_t read_pos = 0;
    int rc = amp_sampler_peek(node_name, &samples, &total_frames, &available_channels, &read_pos);
    if (rc != 0 || samples == NULL || total_frames == 0 || available_channels == 0) {
        // No staged buffer available: return silence
        size_t out_total = (size_t)(batches > 0 ? batches : 1) * (size_t)(frames > 0 ? frames : 1);
        double *buf = (double *)calloc(out_total, sizeof(double));
        if (buf == NULL) return -1;
        *out_buffer = buf;
        *out_channels = 1;
        return 0;
    }
    size_t want_frames = (size_t)(frames > 0 ? frames : 1);
    size_t available = (total_frames > read_pos) ? (total_frames - read_pos) : 0;
    size_t deliver = want_frames;
    if (deliver > available) deliver = available;
    if (deliver == 0) {
        // no more data -> produce silence
        size_t out_total = (size_t)(batches > 0 ? batches : 1) * (size_t)(frames > 0 ? frames : 1);
        double *buf = (double *)calloc(out_total, sizeof(double));
        if (buf == NULL) return -1;
        *out_buffer = buf;
        *out_channels = 1;
        return 0;
    }
    size_t out_batches = (size_t)(batches > 0 ? batches : 1);
    size_t out_channels_u = (size_t)(available_channels > 0 ? available_channels : 1);
    size_t total_out_samples = out_batches * out_channels_u * deliver;
    double *buf = (double *)malloc(total_out_samples * sizeof(double));
    if (buf == NULL) return -1;
    // layout: per-batch contiguous frames; we assume single batch or copy frame-major
    for (size_t f = 0; f < deliver; ++f) {
        for (size_t ch = 0; ch < out_channels_u; ++ch) {
            size_t src_idx = (read_pos + f) * out_channels_u + ch;
            size_t dst_idx = f * out_channels_u + ch;
            buf[dst_idx] = samples[src_idx];
        }
    }
    // If frames requested > deliver, pad with zeros up to frames
    if (deliver < want_frames) {
        size_t padded_total = out_batches * out_channels_u * (size_t)want_frames;
        double *buf2 = (double *)calloc(padded_total, sizeof(double));
        if (buf2 == NULL) {
            free(buf);
            return -1;
        }
        // copy existing
        for (size_t i = 0; i < total_out_samples; ++i) buf2[i] = buf[i];
        free(buf);
        buf = buf2;
    }
    *out_buffer = buf;
    *out_channels = (int)out_channels_u;
    // advance read cursor
    amp_sampler_advance(node_name, deliver);
    return 0;
}
