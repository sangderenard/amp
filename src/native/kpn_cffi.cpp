extern "C" {
#include "amp_native.h"
}

#include "tests/fft_division_test_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

using amp::tests::fft_division_shared::BuildPcmTapDescriptor;
using amp::tests::fft_division_shared::BuildSpectralTapDescriptor;
using amp::tests::fft_division_shared::InstantiateTapBuffer;
using amp::tests::fft_division_shared::PopulateLegacyPcmFromMailbox;
using amp::tests::fft_division_shared::PopulateLegacySpectrumFromMailbox;

static int parse_window_size_from_params(const char *params_json) {
    if (params_json == nullptr) return 4;
    const char *key = "\"window_size\"";
    const char *p = std::strstr(params_json, key);
    if (!p) return 4;
    p += std::strlen(key);
    // find ':'
    const char *colon = std::strchr(p, ':');
    if (!colon) return 4;
    const char *num = colon + 1;
    while (*num && (*num == ' ' || *num == '\t')) ++num;
    long val = std::strtol(num, nullptr, 10);
    if (val <= 0) return 4;
    return static_cast<int>(val);
}

// Exposed C entrypoint for running FFTDivisionNode from an in-memory buffer.
// - samples: pointer to double PCM samples (mono)
// - frames: sample count
// - params_json: optional node params JSON (may be nullptr)
// - dump_prefix: optional prefix to write dump files (may be nullptr)
// - chunk_frames: if >0, feed samples in chunks of this size (streaming); if <=0, single-shot
extern "C" {
AMP_CAPI int kpn_run_fft_division_from_buffer(
    const double *samples,
    size_t frames,
    const char *params_json,
    const char *dump_prefix,
    int chunk_frames
) {
    if (samples == nullptr || frames == 0) return -1;

    const char *effective_params = (params_json != nullptr) ? params_json : "{}";
    const int window_size = parse_window_size_from_params(effective_params);
    const uint32_t hop = static_cast<uint32_t>(std::max(1, window_size / 2));

    // Prepare tap descriptors and legacy buffers
    const size_t total_frames = frames;
    auto spectral_desc = BuildSpectralTapDescriptor(static_cast<uint32_t>(window_size), hop, total_frames);
    auto spectral_real_desc = spectral_desc; spectral_real_desc.name = "spectral_0"; spectral_real_desc.buffer_class = "spectrum_real";
    auto spectral_imag_desc = spectral_desc; spectral_imag_desc.name = "spectral_0"; spectral_imag_desc.buffer_class = "spectrum_imag";
    auto pcm_desc = BuildPcmTapDescriptor(static_cast<uint32_t>(window_size), 1U, total_frames);

    const size_t spectral_frames = spectral_desc.shape.frames;
    const size_t spectral_values = spectral_frames * static_cast<size_t>(window_size);

    std::vector<double> spectral_real(spectral_values, 0.0);
    std::vector<double> spectral_imag(spectral_values, 0.0);
    std::vector<double> pcm_out(total_frames, 0.0);

    EdgeRunnerTapBuffer tap_buffers[3];
    tap_buffers[0] = InstantiateTapBuffer(spectral_real_desc, spectral_real.data());
    tap_buffers[1] = InstantiateTapBuffer(spectral_imag_desc, spectral_imag.data());
    tap_buffers[2] = InstantiateTapBuffer(pcm_desc, pcm_out.data());

    EdgeRunnerNodeDescriptor descriptor{};
    descriptor.name = "fft_division_node";
    descriptor.name_len = std::strlen(descriptor.name);
    descriptor.type_name = "FFTDivisionNode";
    descriptor.type_len = std::strlen(descriptor.type_name);
    descriptor.params_json = effective_params;
    descriptor.params_len = std::strlen(effective_params);

    EdgeRunnerParamSet params{}; params.count = 0; params.items = nullptr;

    EdgeRunnerTapBufferSet tap_set{}; tap_set.items = tap_buffers; tap_set.count = 3;
    EdgeRunnerTapStatusSet status_set{}; status_set.items = nullptr; status_set.count = 0;
    EdgeRunnerTapContext tap_context{}; tap_context.outputs = tap_set; tap_context.status = status_set;

    void *state = nullptr;
    double *out_buffer = nullptr;
    int out_channels = 0;
    AmpNodeMetrics metrics{};

    if (chunk_frames > 0) {
        // streaming mode
        size_t processed = 0;
        int chunk_index = 0;
        while (processed < total_frames) {
            const size_t n = std::min(static_cast<size_t>(chunk_frames), total_frames - processed);
            EdgeRunnerAudioView audio{};
            audio.has_audio = EDGE_RUNNER_AUDIO_FLAG_HAS_DATA;
            audio.batches = 1U; audio.channels = 1U; audio.frames = static_cast<uint32_t>(n);
            audio.data = const_cast<double *>(samples + processed);
            if (processed + n >= total_frames) audio.has_audio |= EDGE_RUNNER_AUDIO_FLAG_FINAL;

            EdgeRunnerNodeInputs inputs{};
            inputs.audio = audio; inputs.params = params; inputs.taps = tap_context;

            int rc = amp_run_node_v2(&descriptor, &inputs, 1, 1, static_cast<int>(n), 48000.0, &out_buffer, &out_channels, &state, nullptr, AMP_EXECUTION_MODE_FORWARD, &metrics);
            if (rc != 0 && rc != AMP_E_PENDING) {
                if (out_buffer) { amp_free(out_buffer); out_buffer = nullptr; }
                if (state) { amp_release_state(state); state = nullptr; }
                return rc;
            }
            if (out_buffer) { amp_free(out_buffer); out_buffer = nullptr; }
            processed += n; ++chunk_index;
        }
        // After streaming, block until tap caches are ready
        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[2], tap_buffers[2].tap_name, 0);
        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[0], tap_buffers[0].tap_name, 0);
        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[1], tap_buffers[1].tap_name, 0);
    } else {
        // single-shot
        EdgeRunnerAudioView audio = {};
        audio.has_audio = EDGE_RUNNER_AUDIO_FLAG_HAS_DATA | EDGE_RUNNER_AUDIO_FLAG_FINAL;
        audio.batches = 1U; audio.channels = 1U; audio.frames = static_cast<uint32_t>(total_frames);
        audio.data = const_cast<double *>(samples);
        EdgeRunnerNodeInputs inputs{};
        inputs.audio = audio; inputs.params = params; inputs.taps = tap_context;

        int rc = amp_run_node_v2(&descriptor, &inputs, 1, 1, static_cast<int>(total_frames), 48000.0, &out_buffer, &out_channels, &state, nullptr, AMP_EXECUTION_MODE_FORWARD, &metrics);
        if (rc != 0) {
            if (out_buffer) { amp_free(out_buffer); out_buffer = nullptr; }
            if (state) { amp_release_state(state); state = nullptr; }
            return rc;
        }
        if (out_buffer) { amp_free(out_buffer); out_buffer = nullptr; }

        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[2], tap_buffers[2].tap_name, 0);
        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[0], tap_buffers[0].tap_name, 0);
        (void)amp_tap_cache_block_until_ready(state, &tap_buffers[1], tap_buffers[1].tap_name, 0);
    }

    // Populate legacy buffers from mailbox chains
    (void)PopulateLegacyPcmFromMailbox(tap_buffers[2], pcm_out.data(), pcm_out.size());
    (void)PopulateLegacySpectrumFromMailbox(tap_buffers[0], tap_buffers[1], spectral_real.data(), spectral_imag.data(), spectral_real.size());

    // Optionally write dumps
    if (dump_prefix != nullptr && dump_prefix[0] != '\0') {
        try {
            std::string prefix(dump_prefix);
            std::ofstream f1(prefix + "_first_pcm.txt");
            f1 << pcm_out.size() << "\n";
            for (size_t i = 0; i < pcm_out.size(); ++i) f1 << std::setprecision(12) << pcm_out[i] << "\n";
            f1.close();

            std::ofstream fr(prefix + "_first_spec_real.txt");
            fr << spectral_real.size() << "\n";
            for (size_t i = 0; i < spectral_real.size(); ++i) fr << std::setprecision(12) << spectral_real[i] << "\n";
            fr.close();

            std::ofstream fi(prefix + "_first_spec_imag.txt");
            fi << spectral_imag.size() << "\n";
            for (size_t i = 0; i < spectral_imag.size(); ++i) fi << std::setprecision(12) << spectral_imag[i] << "\n";
            fi.close();
        } catch (...) {
            // ignore file write errors
        }
    }

    if (state) {
        amp_release_state(state);
        state = nullptr;
    }

    return 0;
}
} // extern "C"
