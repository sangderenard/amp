// Small wrapper implementing a KPN streaming session that exposes
// AmpGraphStreamer functionality via a C API.
// This file mirrors demo_kpn_native.cpp usage and offers a NULL-blob
// convenience path that constructs a demo runtime when blobs are omitted.

#include "amp_native.h"
#include "amp_descriptor_builder.h"
#include <cstring>
#include <memory>

struct KpnStreamSession {
    AmpGraphRuntime *runtime = nullptr;
    AmpGraphStreamer *streamer = nullptr;
    // Other bookkeeping can be added as needed.
};

extern "C" {

KpnStreamSession *amp_kpn_session_create_from_blobs(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len,
    int frames_hint,
    double sample_rate,
    uint32_t ring_frames,
    uint32_t block_frames
){
    KpnStreamSession *s = new KpnStreamSession();

    // If the caller provided compiled blobs, attempt to build a runtime from them.
    if (descriptor_blob && descriptor_len > 0) {
        s->runtime = amp_graph_runtime_create(descriptor_blob, descriptor_len, plan_blob, plan_len);
    }

    // If no runtime was created, fall back to building the demo descriptor (mirror demo_kpn_native).
    AmpDescriptorBuffer descriptor{};
    AmpDescriptorBuilder descriptor_builder{};
    bool used_descriptor_builder = false;
    if (!s->runtime) {
        amp_descriptor_buffer_init(&descriptor);
        if (amp_descriptor_builder_init(&descriptor_builder, &descriptor) == 0) {
            // Build the same minimal demo graph as demo_kpn_native
            // Build minimal Sampler -> FFT graph for headless streaming tests.
            // The SamplerNode will read data staged by the host under the node name "sampler".
            const char *sampler_json = "{\"channels\":1,\"declared_delay\":0,\"supports_v2\":true}";
            const char *fft_json = "{\"algorithm\":\"radix2\",\"declared_delay\":511,\"enable_remainder\":true,\"oversample_ratio\":1,\"supports_v2\":true,\"window_size\":512,\"hop_size\":256}";

            const char *fft_inputs[] = { "sampler" };

            // Sampler node: provides PCM to the graph from host-staged buffers
            amp_descriptor_builder_append_node(&descriptor_builder, "sampler", "SamplerNode", nullptr, 0U, sampler_json, nullptr, 0U);
            // FFT node: consumes the sampler output and will produce spectral dumps
            amp_descriptor_builder_append_node(&descriptor_builder, "fft", "FFTDivisionNode", fft_inputs, 1U, fft_json, nullptr, 0U);
            // Drain node: consumes FFT audio output and forwards PCM into the node mailbox
            const char *drain_inputs[] = { "fft" };
            const char *drain_json = "{}";
            amp_descriptor_builder_append_node(&descriptor_builder, "drain", "DrainNode", drain_inputs, 1U, drain_json, nullptr, 0U);
            if (amp_descriptor_builder_finalize(&descriptor_builder) == 0) {
                used_descriptor_builder = true;
                s->runtime = amp_graph_runtime_create(descriptor.data, descriptor.size, nullptr, 0U);
            }
        }
    }

    if (!s->runtime) {
        // Unable to create runtime
        if (used_descriptor_builder) {
            amp_descriptor_buffer_destroy(&descriptor);
        }
        delete s;
        return nullptr;
    }

    // Configure runtime if we used the descriptor builder (follow demo defaults)
    if (used_descriptor_builder) {
        amp_graph_runtime_configure(s->runtime, 1U, static_cast<uint32_t>(frames_hint));
        amp_graph_runtime_set_dsp_sample_rate(s->runtime, sample_rate);
        amp_graph_runtime_set_scheduler_mode(s->runtime, AMP_SCHEDULER_ORDERED);
        AmpGraphSchedulerParams scheduler_params{};
        scheduler_params.early_bias = 0.5;
        scheduler_params.late_bias = 0.5;
        scheduler_params.saturation_bias = 1.0;
        amp_graph_runtime_set_scheduler_params(s->runtime, &scheduler_params);
    }

    // Create streamer wrapping the runtime.
    s->streamer = amp_graph_streamer_create(s->runtime, nullptr, 0U, frames_hint, sample_rate, ring_frames, block_frames);
    if (!s->streamer) {
        amp_graph_runtime_destroy(s->runtime);
        delete s;
        return nullptr;
    }

    return s;
}

int amp_kpn_session_start(KpnStreamSession *session) {
    if (!session || !session->streamer) return -1;
    return amp_graph_streamer_start(session->streamer);
}

void amp_kpn_session_stop(KpnStreamSession *session) {
    if (!session || !session->streamer) return;
    amp_graph_streamer_stop(session->streamer);
}

void amp_kpn_session_destroy(KpnStreamSession *session) {
    if (!session) return;
    if (session->streamer) {
        amp_graph_streamer_destroy(session->streamer);
        session->streamer = nullptr;
    }
    if (session->runtime) {
        amp_graph_runtime_destroy(session->runtime);
        session->runtime = nullptr;
    }
    delete session;
}

int amp_kpn_session_available(KpnStreamSession *session, uint64_t *out_frames) {
    if (!session || !session->streamer || !out_frames) return -1;
    uint64_t avail = 0;
    amp_graph_streamer_available(session->streamer, &avail);
    *out_frames = avail;
    return 0;
}

int amp_kpn_session_read(
    KpnStreamSession *session,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
){
    if (!session || !session->streamer || !destination) return -1;
    // amp_graph_streamer_read reads PCM frames from the streamer ring.
    uint64_t seq = 0;
    uint32_t frames = 0;
    uint32_t channels = 0;
        int rc = amp_graph_streamer_read(session->streamer, destination, max_frames, &frames, &channels, &seq);
        if (out_frames) *out_frames = frames;
        if (out_channels) *out_channels = channels;
        if (out_sequence) *out_sequence = seq;
        if (rc == 0) return 0;
        if (rc == 1) return 1;
        return rc;
}

int amp_kpn_session_dump_count(KpnStreamSession *session, uint32_t *out_count) {
    if (!session || !session->streamer || !out_count) return -1;
    uint32_t c = 0;
    amp_graph_streamer_dump_count(session->streamer, &c);
    *out_count = c;
    return 0;
}

int amp_kpn_session_pop_dump(
    KpnStreamSession *session,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
){
    if (!session || !session->streamer || !destination) return -1;
    uint64_t seq = 0;
    uint32_t frames = 0;
    uint32_t channels = 0;
        int rc = amp_graph_streamer_pop_dump(session->streamer, destination, max_frames, &frames, &channels, &seq);
        if (out_frames) *out_frames = frames;
        if (out_channels) *out_channels = channels;
        if (out_sequence) *out_sequence = seq;
        if (rc == 0) return 0;
        if (rc == 1) return 1;
        return rc;
}

int amp_kpn_session_status(KpnStreamSession *session, uint64_t *out_produced_frames, uint64_t *out_consumed_frames) {
    if (!session || !session->streamer) return -1;
    amp_graph_streamer_status(session->streamer, out_produced_frames, out_consumed_frames);
    return 0;
}

int amp_kpn_session_debug_snapshot(
    KpnStreamSession *session,
    AmpGraphNodeDebugEntry *node_entries,
    uint32_t node_capacity,
    AmpGraphDebugSnapshot *snapshot
){
    if (!session || !session->runtime || !session->streamer || !snapshot) return -1;
    // node_entries may be NULL if caller only wants the snapshot
    int rc = amp_graph_runtime_debug_snapshot(session->runtime, session->streamer, node_entries, node_capacity, snapshot);
    return rc;
}

} // extern C
