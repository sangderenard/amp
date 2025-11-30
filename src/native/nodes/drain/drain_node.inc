#include "include/amp_native.h"
#include "mailbox.h"

static int run_drain_node(
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
    (void)sample_rate;
    // This node acts as a consumer/forwarder: it writes incoming audio
    // slices into the node mailbox so external harnesses can pop them.

    if (inputs == NULL || out_buffer == NULL || out_channels == NULL) {
        return -1;
    }

    const EdgeRunnerAudioView *audio = &inputs->audio;
    if ((audio->has_audio & EDGE_RUNNER_AUDIO_FLAG_HAS_DATA) == 0 || audio->data == NULL || audio->frames == 0) {
        // produce silence so runtime can continue; no mailbox push
        size_t out_total = (size_t)(batches > 0 ? batches : 1) * (size_t)(frames > 0 ? frames : 1);
        double *buf = (double *)calloc(out_total, sizeof(double));
        if (buf == NULL) return -1;
        *out_buffer = buf;
        *out_channels = 1;
        return 0;
    }

    uint32_t channels = audio->channels > 0 ? audio->channels : 1;
    size_t src_frames = audio->frames > 0 ? (size_t)audio->frames : (size_t)(frames > 0 ? frames : 1);
    size_t total_samples = src_frames * (size_t)channels;

    /* Allocate an output buffer that will be returned to the runtime so it
       is written into the sink ring. This makes the Drain node bridge into
       the streamer's sink path by default (so the dump queue will be
       populated). Also create a separate copy for the node mailbox so
       mailbox ownership/release is independent. */
    double *out_buf = (double *)malloc(total_samples * sizeof(double));
    if (out_buf == NULL) {
        return -1;
    }
    memcpy(out_buf, audio->data, total_samples * sizeof(double));

    double *mail_buf = (double *)malloc(total_samples * sizeof(double));
    if (mail_buf == NULL) {
        free(out_buf);
        return -1;
    }
    memcpy(mail_buf, audio->data, total_samples * sizeof(double));

    /* Create mailbox entry that owns its copy (context + release) so
       releasing the mailbox entry won't free the buffer returned to the
       runtime. */
    AmpMailboxEntry *entry = amp_mailbox_entry_create(mail_buf, (int)channels, src_frames, 0, NULL, mail_buf, (AmpMailboxReleaseFn)free);
    if (entry == NULL) {
        free(mail_buf);
        free(out_buf);
        return -1;
    }
    amp_node_mailbox_push((void*)state, entry);

    /* Return the captured audio as the node output so the runtime will
       write it into the sink ring and the streamer will surface it to
       host-visible dumps. */
    *out_buffer = out_buf;
    *out_channels = (int)channels;
    return 0;
}
