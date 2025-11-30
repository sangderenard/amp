// Minimal KPN harness: build a graph containing a single FFTDivisionNode
// and run it via the repository KPN runtime. No GUI — minimal behaviour
// to exercise the runtime harness.

#include "amp_native.h"
#include "amp_descriptor_builder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    const uint32_t batches = 1U;
    const double sample_rate = 48000.0;
    // Default frames to produce (can be adjusted in later iterations)
    const uint32_t frames = 4096U;

    AmpDescriptorBuffer descriptor;
    amp_descriptor_buffer_init(&descriptor);
    AmpDescriptorBuilder builder{};
    if (amp_descriptor_builder_init(&builder, &descriptor) != 0) {
        std::fprintf(stderr, "[kpn_run_fft_division] failed to initialise descriptor builder\n");
        return 1;
    }

    // Add a single FFTDivisionNode with empty params (node defaults expected).
    const char params_json[] = "{}";

    if (amp_descriptor_builder_append_node(
            &builder,
            "fft",
            "FFTDivisionNode",
            nullptr,
            0U,
            params_json,
            nullptr,
            0U
        ) != 0) {
        std::fprintf(stderr, "[kpn_run_fft_division] failed to append FFT node\n");
        amp_descriptor_buffer_destroy(&descriptor);
        return 2;
    }

    if (amp_descriptor_builder_finalize(&builder) != 0) {
        std::fprintf(stderr, "[kpn_run_fft_division] failed to finalise descriptor\n");
        amp_descriptor_buffer_destroy(&descriptor);
        return 3;
    }

    AmpGraphRuntime *runtime = amp_graph_runtime_create(descriptor.data, descriptor.size, nullptr, 0U);
    if (runtime == nullptr) {
        std::fprintf(stderr, "[kpn_run_fft_division] amp_graph_runtime_create failed\n");
        amp_descriptor_buffer_destroy(&descriptor);
        return 4;
    }

    if (amp_graph_runtime_configure(runtime, batches, frames) != 0) {
        std::fprintf(stderr, "[kpn_run_fft_division] amp_graph_runtime_configure failed\n");
        amp_graph_runtime_destroy(runtime);
        amp_descriptor_buffer_destroy(&descriptor);
        return 5;
    }
    amp_graph_runtime_set_dsp_sample_rate(runtime, sample_rate);

    // Create streamer to execute the graph and collect outputs.
    AmpGraphStreamer *streamer = amp_graph_streamer_create(
        runtime,
        nullptr,
        0U,
        static_cast<int>(frames),
        sample_rate,
        frames,
        512U
    );
    if (streamer == nullptr) {
        std::fprintf(stderr, "[kpn_run_fft_division] failed to create graph streamer\n");
        amp_graph_runtime_destroy(runtime);
        amp_descriptor_buffer_destroy(&descriptor);
        return 6;
    }

    if (amp_graph_streamer_start(streamer) != 0) {
        std::fprintf(stderr, "[kpn_run_fft_division] failed to start streamer\n");
        amp_graph_streamer_destroy(streamer);
        amp_graph_runtime_destroy(runtime);
        amp_descriptor_buffer_destroy(&descriptor);
        return 7;
    }

    AmpGraphStreamerCompletionContract contract{};
    contract.target_produced_frames = frames;
    contract.target_consumed_frames = frames;
    contract.require_ring_drain = 1;
    contract.require_dump_drain = 1;
    contract.idle_timeout_millis = 500U;
    contract.total_timeout_millis = 10000U;

    AmpGraphStreamerCompletionState state{};
    AmpGraphStreamerCompletionVerdict verdict{};

    // Wait for the streamer to satisfy the completion contract.
    while (true) {
        int status = amp_graph_streamer_evaluate_completion(streamer, &contract, &state, &verdict);
        if (status != 0) {
            std::fprintf(stderr, "[kpn_run_fft_division] streamer reported error status %d\n", status);
            break;
        }
        if (verdict.timed_out) {
            std::fprintf(stderr, "[kpn_run_fft_division] streamer timed out\n");
            break;
        }
        if (verdict.contract_satisfied) {
            std::fprintf(stderr, "[kpn_run_fft_division] contract satisfied: produced=%llu consumed=%llu\n",
                         static_cast<unsigned long long>(state.produced_frames),
                         static_cast<unsigned long long>(state.consumed_frames));
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop and clean up
    amp_graph_streamer_stop(streamer);
    amp_graph_streamer_destroy(streamer);
    amp_graph_runtime_destroy(runtime);
    amp_descriptor_buffer_destroy(&descriptor);

    return 0;
}
