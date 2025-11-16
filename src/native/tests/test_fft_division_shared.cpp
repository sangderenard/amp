#ifndef AMP_TEST_FFT_DIVISION_SHARED_CPP
#define AMP_TEST_FFT_DIVISION_SHARED_CPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdlib>

extern "C" {
#include "amp_fft_backend.h"
#include "amp_native.h"
#include "mailbox.h"
}

namespace amp::tests::fft_division_shared {
namespace {

constexpr double kSampleRate = 48000.0;
void emit_diagnostic(const char *fmt, ...);

struct TestConfig {
    int window_size;
    int frames;
    double tolerance;
    size_t streaming_frames;
    size_t streaming_chunk;
};

TestConfig g_config{4, 8, 1e-4, 4096, 64};
bool g_failed = false;

void apply_window_scaling(TestConfig &config) {
    if (config.window_size < 2) {
        config.window_size = 2;
    }
    const bool is_power_of_two = (config.window_size & (config.window_size - 1)) == 0;
    if (!is_power_of_two) {
        // Clamp to next power of two to keep backend expectations intact.
        int pow2 = 1;
        while (pow2 < config.window_size) {
            pow2 <<= 1;
        }
        config.window_size = pow2;
    }
    config.frames = config.window_size * 2;
    const size_t window = static_cast<size_t>(config.window_size);
    // Keep streaming runs small but still large enough to flush FFT latency reliably.
    constexpr size_t kStreamingChunkMultiplier = 16U;
    constexpr size_t kStreamingPasses = 2U;
    config.streaming_chunk = window * kStreamingChunkMultiplier;
    if (config.streaming_chunk == 0U) {
        config.streaming_chunk = window;
    }
    config.streaming_frames = config.streaming_chunk * kStreamingPasses;
}

void configure_from_args(int argc, char **argv) {
    int window_power = 2;  // 2^2 = 4 default window size.
    double tolerance = g_config.tolerance;

    if (argc > 1 && argv[1] != nullptr) {
        char *end = nullptr;
        long parsed = std::strtol(argv[1], &end, 10);
        if (end != argv[1] && parsed >= 1 && parsed <= 16) {
            window_power = static_cast<int>(parsed);
        } else {
            emit_diagnostic("invalid window power '%s', keeping default", argv[1]);
        }
    }

    if (argc > 2 && argv[2] != nullptr) {
        char *end = nullptr;
        double parsed = std::strtod(argv[2], &end);
        if (end != argv[2] && parsed > 0.0) {
            tolerance = parsed;
        } else {
            emit_diagnostic("invalid tolerance '%s', keeping default", argv[2]);
        }
    }

    g_config.window_size = 1 << window_power;
    g_config.tolerance = tolerance;
    apply_window_scaling(g_config);

    emit_diagnostic(
        "config: window_size=%d frames=%d tolerance=%g streaming_chunk=%zu streaming_frames=%zu",
        g_config.window_size,
        g_config.frames,
        g_config.tolerance,
        g_config.streaming_chunk,
        g_config.streaming_frames
    );
}

void emit_diagnostic(const char *fmt, ...) {
    std::fprintf(stderr, "[fft_division_node][diag] ");

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
}

void record_failure(const char *fmt, ...) {
    g_failed = true;
    std::fprintf(stderr, "[fft_division_node] ");

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
}

bool nearly_equal(double a, double b, double tol = -1.0) {
    const double effective_tol = (tol >= 0.0) ? tol : g_config.tolerance;
    return std::fabs(a - b) <= effective_tol;
}

int wait_for_completion(
    const EdgeRunnerNodeDescriptor &descriptor,
    const EdgeRunnerNodeInputs &inputs,
    int batches,
    int channels,
    int expected_frames,
    double sample_rate,
    void **state,
    double **out_buffer,
    int *out_channels,
    AmpNodeMetrics *metrics
) {
    std::fprintf(
        stderr,
        "[FFT-TEST] wait_for_completion descriptor=%s expected=%d batches=%d channels=%d\n",
        descriptor.name != nullptr ? descriptor.name : "<unnamed>",
        expected_frames,
        batches,
        channels
    );
    // Call amp_wait_node_completion with expected frame count
    // It will poll internally until it accumulates the expected number of frames
    return amp_wait_node_completion(
        &descriptor,
        &inputs,
        batches,
        channels,
        expected_frames,
        sample_rate,
        state,
        out_buffer,
        out_channels,
        metrics
    );
}

void log_vector_segment(
    const char *label,
    const double *actual,
    const double *expected,
    size_t count,
    size_t center
) {
    const size_t window = 4;
    const size_t start = (center > window) ? (center - window) : 0;
    const size_t end = std::min(count, center + window + 1);
    for (size_t i = start; i < end; ++i) {
        const double diff = actual[i] - expected[i];
        emit_diagnostic(
            "%s[%04zu] actual=% .12f expected=% .12f diff=% .12f",
            label,
            i,
            actual[i],
            expected[i],
            diff
        );
    }
}

struct RunResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    size_t pcm_frames_committed{0};
    size_t spectral_rows_committed{0};
    AmpNodeMetrics metrics{};
};

struct StreamingRunResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    std::vector<AmpNodeMetrics> metrics_per_call;
    size_t call_count{0};
    bool state_allocated{false};
    size_t pcm_frames_committed{0};
    size_t spectral_rows_committed{0};
};

struct SimulationResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
};

struct TapDescriptor {
    std::string name;
    std::string buffer_class;
    EdgeRunnerTensorShape shape{};
    uint32_t hop_size{1U};

    size_t ValueCount() const {
        const size_t batches = std::max<uint32_t>(1U, shape.batches);
        const size_t channels = std::max<uint32_t>(1U, shape.channels);
        const size_t frames = std::max<uint32_t>(1U, shape.frames);
        return batches * channels * frames;
    }
};

TapDescriptor BuildPcmTapDescriptor(
    uint32_t window_size,
    uint32_t hop_count,
    size_t total_frames,
    uint32_t channels = 1U
) {
    TapDescriptor descriptor{};
    descriptor.name = "pcm";
    descriptor.buffer_class = "pcm";
    descriptor.hop_size = hop_count > 0U ? hop_count : 1U;
    descriptor.shape.batches = 1U;
    descriptor.shape.channels = std::max<uint32_t>(1U, channels);
    descriptor.shape.frames = static_cast<uint32_t>(total_frames);
    (void)window_size;  // window size does not influence PCM layout yet but remains for symmetry.
    return descriptor;
}

static uint32_t ComputeFrameCount(size_t total_frames, uint32_t hop_count) {
    if (hop_count == 0U) {
        hop_count = 1U;
    }
    if (total_frames == 0U) {
        return 0U;
    }
    const size_t frames = (total_frames + static_cast<size_t>(hop_count) - 1U) /
        static_cast<size_t>(hop_count);
    return static_cast<uint32_t>(frames);
}

TapDescriptor BuildSpectralTapDescriptor(
    uint32_t window_size,
    uint32_t hop_count,
    size_t total_frames,
    uint32_t spectral_lanes = 1U
) {
    TapDescriptor descriptor{};
    descriptor.name = "spectral";
    descriptor.buffer_class = "spectrum";
    descriptor.hop_size = hop_count > 0U ? hop_count : 1U;
    descriptor.shape.batches = std::max<uint32_t>(1U, spectral_lanes);
    descriptor.shape.channels = std::max<uint32_t>(1U, window_size);
    descriptor.shape.frames = ComputeFrameCount(total_frames, descriptor.hop_size);
    return descriptor;
}

EdgeRunnerTapBuffer make_tap_buffer(
    const char *name,
    const char *buffer_class,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames,
    double *data
) {
    EdgeRunnerTapBuffer tap{};
    tap.tap_name = name;
    tap.buffer_class = buffer_class;
    tap.shape.batches = batches;
    tap.shape.channels = channels;
    tap.shape.frames = frames;
    const size_t computed_stride = static_cast<size_t>(
        std::max<uint32_t>(1U, batches) * std::max<uint32_t>(1U, channels));
    tap.frame_stride = computed_stride;
    tap.data = data;
    return tap;
}

EdgeRunnerTapBuffer InstantiateTapBuffer(const TapDescriptor &descriptor, double *data) {
    return make_tap_buffer(
        descriptor.name.c_str(),
        descriptor.buffer_class.c_str(),
        descriptor.shape.batches,
        descriptor.shape.channels,
        descriptor.shape.frames,
        data
    );
}

static size_t SafeDim(uint32_t value) {
    return (value > 0U) ? static_cast<size_t>(value) : 1U;
}

static size_t ComputeFrameStride(const EdgeRunnerTapBuffer &buffer) {
    if (buffer.frame_stride > 0U) {
        return buffer.frame_stride;
    }
    return SafeDim(buffer.shape.batches) * SafeDim(buffer.shape.channels);
}

std::vector<float> DecodeTapTensor(const EdgeRunnerTapBuffer &buffer) {
    const size_t frames = SafeDim(buffer.shape.frames);
    const size_t batches = SafeDim(buffer.shape.batches);
    const size_t channels = SafeDim(buffer.shape.channels);
    const size_t total = frames * batches * channels;
    std::vector<float> decoded(total, 0.0f);
    if (buffer.data == nullptr || total == 0U) {
        return decoded;
    }
    const size_t frame_stride = ComputeFrameStride(buffer);
    for (size_t frame = 0; frame < frames; ++frame) {
        const double *frame_ptr = buffer.data + frame * frame_stride;
        for (size_t batch = 0; batch < batches; ++batch) {
            const double *batch_ptr = frame_ptr + batch * channels;
            const size_t base = frame * batches * channels + batch * channels;
            for (size_t channel = 0; channel < channels; ++channel) {
                decoded[base + channel] = static_cast<float>(batch_ptr[channel]);
            }
        }
    }
    return decoded;
}

static std::string DeriveTapName(const EdgeRunnerTapBuffer &buffer, size_t ordinal) {
    if (buffer.tap_name != nullptr && buffer.tap_name[0] != '\0') {
        return std::string(buffer.tap_name);
    }
    char generated[32];
    std::snprintf(generated, sizeof(generated), "tap_%zu", ordinal);
    return std::string(generated);
}

std::unordered_map<std::string, std::vector<float>> DecodeTapBuffers(const EdgeRunnerTapBufferSet &set) {
    std::unordered_map<std::string, std::vector<float>> decoded;
    if (set.items == nullptr || set.count == 0U) {
        return decoded;
    }
    for (uint32_t i = 0; i < set.count; ++i) {
        const EdgeRunnerTapBuffer &buffer = set.items[i];
        decoded.emplace(DeriveTapName(buffer, i), DecodeTapTensor(buffer));
    }
    return decoded;
}

void run_shared_helper_unit_tests() {
    const uint32_t window_size = 8U;
    const uint32_t hop_count = 4U;
    const size_t total_frames = 32U;
    TapDescriptor pcm_descriptor = BuildPcmTapDescriptor(window_size, hop_count, total_frames, 2U);
    TapDescriptor spectral_descriptor = BuildSpectralTapDescriptor(window_size, hop_count, total_frames, 3U);

    std::vector<double> pcm_storage(pcm_descriptor.ValueCount(), 0.0);
    std::vector<double> spectral_storage(spectral_descriptor.ValueCount(), 0.0);
    for (size_t i = 0; i < pcm_storage.size(); ++i) {
        pcm_storage[i] = 0.25 * static_cast<double>(i + 1);
    }
    for (size_t i = 0; i < spectral_storage.size(); ++i) {
        spectral_storage[i] = 0.125 * static_cast<double>(i + 1);
    }

    EdgeRunnerTapBuffer pcm_buffer = InstantiateTapBuffer(pcm_descriptor, pcm_storage.data());
    EdgeRunnerTapBuffer spectral_buffer = InstantiateTapBuffer(spectral_descriptor, spectral_storage.data());

    const auto pcm_decoded = DecodeTapTensor(pcm_buffer);
    const auto spectral_decoded = DecodeTapTensor(spectral_buffer);
    if (pcm_decoded.size() != pcm_storage.size()) {
        record_failure(
            "pcm helper decode size mismatch got=%zu expected=%zu",
            pcm_decoded.size(),
            pcm_storage.size()
        );
        return;
    }
    if (spectral_decoded.size() != spectral_storage.size()) {
        record_failure(
            "spectral helper decode size mismatch got=%zu expected=%zu",
            spectral_decoded.size(),
            spectral_storage.size()
        );
        return;
    }

    constexpr double kFloatTolerance = 1e-6;
    for (size_t i = 0; i < pcm_storage.size(); ++i) {
        const double expected = pcm_storage[i];
        const double actual = pcm_decoded[i];
        if (std::fabs(actual - expected) > kFloatTolerance) {
            record_failure(
                "pcm helper decode mismatch index=%zu got=%g expected=%g",
                i,
                actual,
                expected
            );
            return;
        }
    }
    for (size_t i = 0; i < spectral_storage.size(); ++i) {
        const double expected = spectral_storage[i];
        const double actual = spectral_decoded[i];
        if (std::fabs(actual - expected) > kFloatTolerance) {
            record_failure(
                "spectral helper decode mismatch index=%zu got=%g expected=%g",
                i,
                actual,
                expected
            );
            return;
        }
    }

    std::array<EdgeRunnerTapBuffer, 2> buffers{spectral_buffer, pcm_buffer};
    EdgeRunnerTapBufferSet set{};
    set.items = buffers.data();
    set.count = static_cast<uint32_t>(buffers.size());
    const auto decoded_map = DecodeTapBuffers(set);
    auto pcm_it = decoded_map.find(pcm_descriptor.name);
    auto spectral_it = decoded_map.find(spectral_descriptor.name);
    if (pcm_it == decoded_map.end()) {
        record_failure("decoded_map missing pcm entry");
        return;
    }
    if (spectral_it == decoded_map.end()) {
        record_failure("decoded_map missing spectral entry");
        return;
    }
    if (pcm_it->second != pcm_decoded) {
        record_failure("decoded_map pcm values mismatch");
        return;
    }
    if (spectral_it->second != spectral_decoded) {
        record_failure("decoded_map spectral values mismatch");
        return;
    }
}

void write_tap_row(
    const EdgeRunnerTapBuffer &buffer,
    int batch,
    int frame_index,
    const double *src,
    int value_count
) {
    if (buffer.data == nullptr || src == nullptr || batch < 0 || frame_index < 0 || value_count <= 0) {
        return;
    }
    const uint32_t batches = buffer.shape.batches > 0U ? buffer.shape.batches : 1U;
    const uint32_t frames = buffer.shape.frames > 0U ? buffer.shape.frames : 0U;
    const uint32_t channels = buffer.shape.channels > 0U ? buffer.shape.channels : 1U;
    if (static_cast<uint32_t>(batch) >= batches) {
        return;
    }
    if (frames > 0U && static_cast<uint32_t>(frame_index) >= frames) {
        return;
    }
    size_t stride = buffer.frame_stride > 0U
        ? buffer.frame_stride
        : static_cast<size_t>(batches) * channels;
    if (stride == 0U) {
        stride = static_cast<size_t>(channels);
    }
    double *frame_ptr = buffer.data + static_cast<size_t>(frame_index) * stride;
    double *batch_ptr = frame_ptr + static_cast<size_t>(batch) * channels;
    size_t copy = static_cast<size_t>(value_count);
    if (copy > channels) {
        copy = channels;
    }
    std::memcpy(batch_ptr, src, copy * sizeof(double));
    if (copy < channels) {
        std::fill(batch_ptr + copy, batch_ptr + channels, 0.0);
    }
}

size_t drain_spectral_mailbox_rows(
    void *state,
    EdgeRunnerTapBuffer &spectral_real_tap,
    EdgeRunnerTapBuffer &spectral_imag_tap,
    std::vector<uint8_t> &spectral_row_written,
    uint32_t frame_count
) {
    if (state == nullptr) {
        return 0U;
    }

    const size_t tap_frames = spectral_real_tap.shape.frames > 0U
        ? spectral_real_tap.shape.frames
        : frame_count;
    size_t spectral_rows_captured = 0U;
    AmpSpectralMailboxEntry *entry = nullptr;
    while ((entry = amp_node_spectral_mailbox_pop(state)) != nullptr) {
        const int slot = entry->slot;
        const int frame_index = entry->frame_index;
        const int window_size = entry->window_size;
        const int latency = (window_size > 0) ? (window_size - 1) : 0;  // remove declared FFT delay so indices start at zero
        const int aligned_frame_index = frame_index - latency;
        const bool slot_in_range = slot >= 0 && static_cast<uint32_t>(slot) < spectral_real_tap.shape.batches;
        if (slot_in_range && aligned_frame_index >= 0 && static_cast<uint32_t>(aligned_frame_index) < tap_frames) {
            const size_t row_index = static_cast<size_t>(slot) * frame_count + static_cast<size_t>(aligned_frame_index);
            if (row_index < spectral_row_written.size() && spectral_row_written[row_index] == 0U) {
                write_tap_row(spectral_real_tap, slot, aligned_frame_index, entry->spectral_real, entry->window_size);
                write_tap_row(spectral_imag_tap, slot, aligned_frame_index, entry->spectral_imag, entry->window_size);
                spectral_row_written[row_index] = 1U;
                spectral_rows_captured += 1U;
            }
        }
        amp_spectral_mailbox_entry_release(entry);
    }
    return spectral_rows_captured;
}

std::string build_params_json() {
    char buffer[256];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "{\"window_size\":%d,\"algorithm\":\"fft\",\"window\":\"hann\",\"supports_v2\":true,"
        "\"declared_delay\":%d,\"oversample_ratio\":1,\"epsilon\":1e-9,\"max_batch_windows\":1,"
        "\"backend_hop\":1}",
        g_config.window_size,
        g_config.window_size - 1
    );
    return std::string(buffer);
}

EdgeRunnerNodeDescriptor build_descriptor(std::string &params_json) {
    params_json = build_params_json();

    EdgeRunnerNodeDescriptor descriptor{};
    descriptor.name = "fft_division_node";
    descriptor.name_len = std::strlen(descriptor.name);
    descriptor.type_name = "FFTDivisionNode";
    descriptor.type_len = std::strlen(descriptor.type_name);
    descriptor.params_json = params_json.c_str();
    descriptor.params_len = params_json.size();
    return descriptor;
}

EdgeRunnerAudioView build_audio_view_span(const double *data, size_t frames) {
    EdgeRunnerAudioView audio{};
    audio.has_audio = 1U;
    audio.batches = 1U;
    audio.channels = 1U;
    audio.frames = static_cast<uint32_t>(frames);
    audio.data = data;
    return audio;
}

EdgeRunnerAudioView build_audio_view(const std::vector<double> &signal) {
    if (signal.size() != static_cast<size_t>(g_config.frames)) {
        record_failure(
            "signal length %zu does not match expected frame count %d",
            signal.size(),
            g_config.frames
        );
    }
    return build_audio_view_span(signal.data(), signal.size());
}

RunResult run_fft_node_once(const std::vector<double> &signal) {
    RunResult result;
    result.pcm.assign(signal.size(), 0.0);
    result.spectral_real.assign(signal.size() * g_config.window_size, 0.0);
    result.spectral_imag.assign(signal.size() * g_config.window_size, 0.0);
    const uint32_t frame_count = static_cast<uint32_t>(signal.size());
    std::array<EdgeRunnerTapBuffer, 3> tap_buffers{};
    tap_buffers[0] = make_tap_buffer(
        "spectral_real",
        "spectrum",
        1U,
        g_config.window_size,
        frame_count,
        result.spectral_real.data());
    tap_buffers[1] = make_tap_buffer(
        "spectral_imag",
        "spectrum",
        1U,
        g_config.window_size,
        frame_count,
        result.spectral_imag.data());
    tap_buffers[2] = make_tap_buffer(
        "pcm",
        "pcm",
        1U,
        1U,
        frame_count,
        result.pcm.data());

    const size_t spectral_lane_count = tap_buffers[0].shape.batches > 0U
        ? tap_buffers[0].shape.batches
        : 1U;
    std::vector<uint8_t> spectral_row_written(spectral_lane_count * frame_count, 0U);
    size_t spectral_rows_captured = 0;

    std::string params_json;
    EdgeRunnerNodeDescriptor descriptor = build_descriptor(params_json);

    EdgeRunnerAudioView audio = build_audio_view(signal);

    EdgeRunnerParamSet params{};
    params.count = 0U;
    params.items = nullptr;

    EdgeRunnerTapBufferSet tap_set{};
    tap_set.items = tap_buffers.data();
    tap_set.count = static_cast<uint32_t>(tap_buffers.size());

    EdgeRunnerTapStatusSet status_set{};
    status_set.items = nullptr;
    status_set.count = 0U;

    EdgeRunnerTapContext tap_context{};
    tap_context.outputs = tap_set;
    tap_context.status = status_set;

    EdgeRunnerTapBuffer &spectral_real_tap = tap_buffers[0];
    EdgeRunnerTapBuffer &spectral_imag_tap = tap_buffers[1];
    EdgeRunnerTapBuffer &pcm_tap = tap_buffers[2];

    EdgeRunnerNodeInputs inputs{};
    inputs.audio = audio;
    inputs.params = params;
    inputs.taps = tap_context;

    double *out_buffer = nullptr;
    int out_channels = 0;
    void *state = nullptr;
    AmpNodeMetrics metrics{};

    int rc = amp_run_node_v2(
        &descriptor,
        &inputs,
        1,
        1,
        static_cast<int>(signal.size()),
        kSampleRate,
        &out_buffer,
        &out_channels,
        &state,
        nullptr,
        AMP_EXECUTION_MODE_FORWARD,
        &metrics
    );

    std::fprintf(
        stderr,
        "[FFT-TEST] run_once amp_run_node_v2 rc=%d buffer=%p channels=%d\n",
        rc,
        static_cast<void *>(out_buffer),
        out_channels
    );

    if (rc == AMP_E_PENDING) {
        std::fprintf(
            stderr,
            "[FFT-TEST] run_once pending -> wait expected=%zu\n",
            signal.size()
        );
        rc = wait_for_completion(
            descriptor,
            inputs,
            1,
            1,
            static_cast<int>(signal.size()),
            kSampleRate,
            &state,
            &out_buffer,
            &out_channels,
            &metrics
        );
    }

    if (rc != 0) {
        record_failure(
            "amp_run_node_v2 forward failed rc=%d",
            rc
        );
    }

    spectral_rows_captured += drain_spectral_mailbox_rows(
        state,
        spectral_real_tap,
        spectral_imag_tap,
        spectral_row_written,
        frame_count
    );

    // Copy PCM output from out_buffer into the PCM tap buffer
    size_t pcm_frames_captured = 0;
    if (out_buffer != nullptr) {
        const size_t frames_to_copy = std::min(result.pcm.size(), signal.size());
        for (size_t frame = 0; frame < frames_to_copy; ++frame) {
            write_tap_row(pcm_tap, 0, static_cast<int>(frame), out_buffer + frame, 1);
            pcm_frames_captured += 1;
        }
        amp_free(out_buffer);
        out_buffer = nullptr;
    }

    result.spectral_rows_committed = spectral_rows_captured;
    result.pcm_frames_committed = pcm_frames_captured;

    emit_diagnostic("run_once spectral rows committed=%zu pcm frames committed=%zu",
                    result.spectral_rows_committed,
                    result.pcm_frames_committed);

    if (state != nullptr) {
        amp_release_state(state);
        state = nullptr;
    }

    result.metrics = metrics;
    return result;
}

StreamingRunResult run_fft_node_streaming(const std::vector<double> &signal, size_t chunk_frames) {
    StreamingRunResult result;
    const size_t total_frames = signal.size();
    result.pcm.assign(total_frames, 0.0);
    result.spectral_real.assign(total_frames * g_config.window_size, 0.0);
    result.spectral_imag.assign(total_frames * g_config.window_size, 0.0);

    if (chunk_frames == 0) {
        record_failure("chunk size must be greater than zero");
        return result;
    }

    std::string params_json;
    EdgeRunnerNodeDescriptor descriptor = build_descriptor(params_json);

    EdgeRunnerParamSet params{};
    params.count = 0U;
    params.items = nullptr;

    std::array<EdgeRunnerTapBuffer, 3> tap_buffers{};
    tap_buffers[0] = make_tap_buffer(
        "spectral_real",
        "spectrum",
        1U,
        g_config.window_size,
        static_cast<uint32_t>(total_frames),
        result.spectral_real.data());
    tap_buffers[1] = make_tap_buffer(
        "spectral_imag",
        "spectrum",
        1U,
        g_config.window_size,
        static_cast<uint32_t>(total_frames),
        result.spectral_imag.data());
    tap_buffers[2] = make_tap_buffer(
        "pcm",
        "pcm",
        1U,
        1U,
        static_cast<uint32_t>(total_frames),
        result.pcm.data());

    const size_t spectral_lane_count = tap_buffers[0].shape.batches > 0U
        ? tap_buffers[0].shape.batches
        : 1U;
    std::vector<uint8_t> spectral_row_written(spectral_lane_count * total_frames, 0U);
    size_t spectral_rows_captured = 0U;
    size_t pcm_frames_captured = 0U;

    EdgeRunnerTapBufferSet tap_set{};
    tap_set.items = tap_buffers.data();
    tap_set.count = static_cast<uint32_t>(tap_buffers.size());

    EdgeRunnerTapStatusSet status_set{};
    status_set.items = nullptr;
    status_set.count = 0U;

    EdgeRunnerTapContext tap_context{};
    tap_context.outputs = tap_set;
    tap_context.status = status_set;

    void *state = nullptr;
    bool saw_state = false;

    EdgeRunnerNodeInputs inputs{};
    inputs.params = params;
    inputs.taps = tap_context;

    for (size_t offset = 0; offset < total_frames; offset += chunk_frames) {
        const size_t frames = std::min(chunk_frames, total_frames - offset);

        EdgeRunnerAudioView audio = build_audio_view_span(signal.data() + offset, frames);
        inputs.audio = audio;

        double *out_buffer = nullptr;
        int out_channels = 0;
        AmpNodeMetrics metrics{};

        int rc = amp_run_node_v2(
            &descriptor,
            &inputs,
            1,
            1,
            static_cast<int>(frames),
            kSampleRate,
            &out_buffer,
            &out_channels,
            &state,
            nullptr,
            AMP_EXECUTION_MODE_FORWARD,
            &metrics
        );

        if (rc == AMP_E_PENDING) {
            std::fprintf(
                stderr,
                "[FFT-TEST] streaming pending offset=%zu frames=%zu\n",
                offset,
                frames
            );
            rc = wait_for_completion(
                descriptor,
                inputs,
                1,
                1,
                static_cast<int>(frames),
                kSampleRate,
                &state,
                &out_buffer,
                &out_channels,
                &metrics
            );
        }

        if (rc != 0 || out_buffer == nullptr || out_channels != 1) {
            record_failure(
                "streaming call %zu failed rc=%d buffer=%p channels=%d",
                offset / chunk_frames,
                rc,
                static_cast<void *>(out_buffer),
                out_channels
            );
        } else {
            std::copy(out_buffer, out_buffer + frames, result.pcm.begin() + offset);
            pcm_frames_captured += frames;
        }

        if (out_buffer != nullptr) {
            amp_free(out_buffer);
            out_buffer = nullptr;
        }

        if (state == nullptr) {
            record_failure("streaming call %zu did not return persistent state", offset / chunk_frames);
        } else {
            saw_state = true;
            result.state_allocated = true;
        }

        result.metrics_per_call.push_back(metrics);
        result.call_count += 1;
    }

    spectral_rows_captured += drain_spectral_mailbox_rows(
        state,
        tap_buffers[0],
        tap_buffers[1],
        spectral_row_written,
        static_cast<uint32_t>(total_frames)
    );

    result.spectral_rows_committed = spectral_rows_captured;
    result.pcm_frames_committed = pcm_frames_captured;

    if (state != nullptr) {
        amp_release_state(state);
        state = nullptr;
    }

    if (!saw_state) {
        record_failure("streaming run never produced node state");
    }

    return result;
}

void verify_close(const char *label, const double *actual, const double *expected, size_t count, double tolerance) {
    constexpr size_t kMaxLoggedMismatches = 16;
    size_t mismatch_count = 0;
    size_t first_index = 0;
    double first_actual = 0.0;
    double first_expected = 0.0;
    double first_diff = 0.0;

    for (size_t i = 0; i < count; ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            if (mismatch_count == 0) {
                first_index = i;
                first_actual = actual[i];
                first_expected = expected[i];
                first_diff = diff;
            }

            if (mismatch_count < kMaxLoggedMismatches) {
                emit_diagnostic(
                    "%s mismatch index=%zu tolerance=%g",
                    label,
                    i,
                    tolerance
                );
                log_vector_segment(label, actual, expected, count, i);
            }
            ++mismatch_count;
        }
    }

    if (mismatch_count > 0) {
        if (mismatch_count > kMaxLoggedMismatches) {
            emit_diagnostic(
                "%s additional mismatches suppressed=%zu",
                label,
                mismatch_count - kMaxLoggedMismatches
            );
        }
        record_failure(
            "%s mismatch: %zu samples exceeded tolerance (first index %zu got %.12f expected %.12f diff %.12f)",
            label,
            mismatch_count,
            first_index,
            first_actual,
            first_expected,
            first_diff
        );
    }
}

void require_identity(const std::vector<double> &input, const std::vector<double> &output, const char *label) {
    if (input.size() != output.size()) {
        record_failure(
            "%s size mismatch input=%zu output=%zu",
            label,
            input.size(),
            output.size()
        );
        return;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        if (!nearly_equal(input[i], output[i])) {
            emit_diagnostic(
                "%s mismatch frame=%zu",
                label,
                i
            );
            log_vector_segment(label, output.data(), input.data(), input.size(), i);
            record_failure(
                "%s mismatch at frame %zu got %.12f expected %.12f",
                label,
                i,
                output[i],
                input[i]
            );
            return;
        }
    }
}

void require_equal(const std::vector<double> &a, const std::vector<double> &b, const char *label) {
    if (a.size() != b.size()) {
        record_failure(
            "%s size mismatch first=%zu second=%zu",
            label,
            a.size(),
            b.size()
        );
        return;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!nearly_equal(a[i], b[i])) {
            emit_diagnostic(
                "%s mismatch index=%zu",
                label,
                i
            );
            log_vector_segment(label, b.data(), a.data(), a.size(), i);
            record_failure(
                "%s mismatch at index %zu got %.12f expected %.12f",
                label,
                i,
                b[i],
                a[i]
            );
            return;
        }
    }
}

void verify_metrics(const AmpNodeMetrics &metrics, const char *label) {
    if (metrics.measured_delay_frames != static_cast<uint32_t>(g_config.window_size - 1)) {
        record_failure(
            "%s unexpected delay %u (expected %u)",
            label,
            metrics.measured_delay_frames,
            static_cast<uint32_t>(g_config.window_size - 1)
        );
    }

    if (metrics.accumulated_heat < 0.0f) {
        record_failure("%s accumulated_heat negative", label);
    }

    if (metrics.processing_time_seconds < 0.0 || metrics.logging_time_seconds < 0.0 ||
        metrics.total_time_seconds < 0.0 || metrics.thread_cpu_time_seconds < 0.0) {
        record_failure("%s negative timing metric", label);
    }

    if (metrics.total_time_seconds + 1e-12 < metrics.processing_time_seconds) {
        record_failure("%s total time < processing time", label);
    }
}

SimulationResult simulate_stream_identity(const std::vector<double> &signal, int window_kind, int window_size) {
    SimulationResult result;
    result.pcm.assign(signal.size(), 0.0);
    result.spectral_real.assign(signal.size() * window_size, 0.0);
    result.spectral_imag.assign(signal.size() * window_size, 0.0);

    void *forward = amp_fft_backend_stream_create(window_size, window_size, 1, window_kind);
    void *inverse = amp_fft_backend_stream_create_inverse(window_size, window_size, 1, window_kind);
    if (forward == nullptr || inverse == nullptr) {
        record_failure(
            "amp_fft_backend_stream_create failed (forward=%p inverse=%p)",
            forward,
            inverse
        );
        if (forward != nullptr) {
            amp_fft_backend_stream_destroy(forward);
        }
        if (inverse != nullptr) {
            amp_fft_backend_stream_destroy(inverse);
        }
        return result;
    }

    const size_t tail_frames = (window_size > 0) ? static_cast<size_t>(window_size - 1) : 0U;
    const size_t padded_frames = signal.size() + tail_frames;
    std::vector<double> padded_signal = signal;
    padded_signal.resize(padded_frames, 0.0);

    const size_t stage_capacity_frames = padded_frames > 0 ? padded_frames : 1U;
    std::vector<double> spectral_stage_real(stage_capacity_frames * window_size, 0.0);
    std::vector<double> spectral_stage_imag(stage_capacity_frames * window_size, 0.0);
    std::vector<double> inverse_scratch(window_size, 0.0);
    std::vector<double> produced_pcm;
    produced_pcm.reserve(padded_frames + static_cast<size_t>(window_size));

    size_t spectral_frames_emitted = 0;
    auto push_and_capture = [&](const double *pcm, size_t samples, int flush_mode) -> size_t {
        if (stage_capacity_frames <= spectral_frames_emitted) {
            return 0U;
        }
        double *real_dst = spectral_stage_real.data() + spectral_frames_emitted * window_size;
        double *imag_dst = spectral_stage_imag.data() + spectral_frames_emitted * window_size;
        const size_t max_frames = stage_capacity_frames - spectral_frames_emitted;
        const size_t produced = amp_fft_backend_stream_push(
            forward,
            pcm,
            samples,
            window_size,
            real_dst,
            imag_dst,
            max_frames,
            flush_mode
        );
        spectral_frames_emitted += produced;
        return produced;
    };

    if (!padded_signal.empty()) {
        push_and_capture(padded_signal.data(), padded_signal.size(), AMP_FFT_STREAM_FLUSH_NONE);
    }

    // Drain any ready frames and then issue repeated final flushes until nothing remains.
    for (int flush_iteration = 0; flush_iteration < 8; ++flush_iteration) {
        if (push_and_capture(nullptr, 0, AMP_FFT_STREAM_FLUSH_PARTIAL) == 0U) {
            break;
        }
    }
    for (int flush_iteration = 0; flush_iteration < 8; ++flush_iteration) {
        if (push_and_capture(nullptr, 0, AMP_FFT_STREAM_FLUSH_FINAL) == 0U) {
            break;
        }
    }

    const size_t frames_to_copy = std::min(signal.size(), spectral_frames_emitted);
    for (size_t frame = 0; frame < frames_to_copy; ++frame) {
        const double *src_real = spectral_stage_real.data() + frame * window_size;
        const double *src_imag = spectral_stage_imag.data() + frame * window_size;
        double *dst_real = result.spectral_real.data() + frame * window_size;
        double *dst_imag = result.spectral_imag.data() + frame * window_size;
        std::copy(src_real, src_real + window_size, dst_real);
        std::copy(src_imag, src_imag + window_size, dst_imag);
    }

    if (spectral_frames_emitted > 0) {
        size_t produced = amp_fft_backend_stream_push_spectrum(
            inverse,
            spectral_stage_real.data(),
            spectral_stage_imag.data(),
            spectral_frames_emitted,
            window_size,
            inverse_scratch.data(),
            inverse_scratch.size(),
            AMP_FFT_STREAM_FLUSH_NONE
        );
        for (size_t i = 0; i < produced; ++i) {
            produced_pcm.push_back(inverse_scratch[i]);
        }
    }

    int flush_iterations = 0;
    while (amp_fft_backend_stream_pending_pcm(inverse) > 0 && flush_iterations < 8) {
        const int flush_mode = (flush_iterations + 1 < 8)
            ? AMP_FFT_STREAM_FLUSH_PARTIAL
            : AMP_FFT_STREAM_FLUSH_FINAL;
        const size_t drained = amp_fft_backend_stream_push_spectrum(
            inverse,
            nullptr,
            nullptr,
            0,
            window_size,
            inverse_scratch.data(),
            inverse_scratch.size(),
            flush_mode
        );
        if (drained == 0) {
            flush_iterations++;
            continue;
        }
        for (size_t i = 0; i < drained; ++i) {
            produced_pcm.push_back(inverse_scratch[i]);
        }
        flush_iterations++;
    }

    const size_t copy_count = std::min(result.pcm.size(), produced_pcm.size());
    if (copy_count > 0) {
        std::copy(produced_pcm.begin(), produced_pcm.begin() + copy_count, result.pcm.begin());
    }

    amp_fft_backend_stream_destroy(forward);
    amp_fft_backend_stream_destroy(inverse);
    return result;
}

void require_backward_unsupported(const std::vector<double> &signal, const std::vector<double> &forward_pcm) {
    std::string params_json;
    EdgeRunnerNodeDescriptor descriptor = build_descriptor(params_json);

    EdgeRunnerAudioView gradient_audio = build_audio_view(signal);
    gradient_audio.data = forward_pcm.data();

    EdgeRunnerParamSet params{};
    params.count = 0U;
    params.items = nullptr;

    EdgeRunnerTapBufferSet tap_set{};
    tap_set.items = nullptr;
    tap_set.count = 0U;

    EdgeRunnerTapStatusSet status_set{};
    status_set.items = nullptr;
    status_set.count = 0U;

    EdgeRunnerTapContext tap_context{};
    tap_context.outputs = tap_set;
    tap_context.status = status_set;

    EdgeRunnerNodeInputs inputs{};
    inputs.audio = gradient_audio;
    inputs.params = params;
    inputs.taps = tap_context;

    double *out_buffer = nullptr;
    int out_channels = 0;
    void *state = nullptr;
    AmpNodeMetrics metrics{};

    int rc = amp_run_node_v2(
        &descriptor,
        &inputs,
        1,
        1,
        static_cast<int>(signal.size()),
        kSampleRate,
        &out_buffer,
        &out_channels,
        &state,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &metrics
    );

    if (rc == AMP_E_PENDING) {
        rc = wait_for_completion(
            descriptor,
            inputs,
            1,
            1,
            static_cast<int>(signal.size()),
            kSampleRate,
            &state,
            &out_buffer,
            &out_channels,
            &metrics
        );
    }

    if (out_buffer != nullptr) {
        amp_free(out_buffer);
        out_buffer = nullptr;
    }

    if (state != nullptr) {
        amp_release_state(state);
        state = nullptr;
    }

    if (rc != AMP_E_UNSUPPORTED) {
        record_failure("backward execution returned %d (expected AMP_E_UNSUPPORTED)", rc);
    }
}

}  // namespace

int RunShared(int argc, char **argv) {
    configure_from_args(argc, argv);
    run_shared_helper_unit_tests();
    // Generate a test signal by starting with an arbitrary waveform,
    // then pre-conditioning it through a forward+inverse FFT roundtrip.
    // This ensures the signal is "well-behaved" and can be perfectly
    // reconstructed, avoiding boundary artifacts from DC offsets or
    // non-zero endpoints.
    // Well-behaved original: starts and ends at zero (half-sine across 8 frames)
    std::vector<double> raw_signal(static_cast<size_t>(g_config.frames), 0.0);
    if (g_config.frames >= 2) {
        const double pi = std::acos(-1.0);
        for (int i = 0; i < g_config.frames; ++i) {
            const double angle = pi * static_cast<double>(i) / static_cast<double>(g_config.frames - 1);
            raw_signal[static_cast<size_t>(i)] = std::sin(angle);
        }
    }

    // Pre-condition the signal: run it through FFT roundtrip once
    SimulationResult preconditioned = simulate_stream_identity(raw_signal, AMP_FFT_WINDOW_HANN, g_config.window_size);

    // Recite original and cleaned signals for full visibility
    emit_diagnostic("original_raw_signal (frames=%zu)", raw_signal.size());
    for (size_t i = 0; i < raw_signal.size(); ++i) {
        emit_diagnostic("raw[%04zu]= % .12f", i, raw_signal[i]);
    }
    emit_diagnostic("cleaned_signal (frames=%zu)", preconditioned.pcm.size());
    for (size_t i = 0; i < preconditioned.pcm.size(); ++i) {
        emit_diagnostic("cleaned[%04zu]= % .12f", i, preconditioned.pcm[i]);
    }

    // Hard fail immediately if lengths are not exactly as expected
    if (raw_signal.size() != static_cast<size_t>(g_config.frames)) {
        record_failure("raw_signal length %zu != expected %d", raw_signal.size(), g_config.frames);
        return 1;
    }
    if (preconditioned.pcm.size() != static_cast<size_t>(g_config.frames)) {
        record_failure("cleaned_signal length %zu != expected %d", preconditioned.pcm.size(), g_config.frames);
        return 1;
    }

    // Use the pre-conditioned PCM as the actual test signal (identity-cleaned)
    const std::vector<double> signal = preconditioned.pcm;

    // Derive expectations from the identity-cleaned signal:
    // - PCM expectation is the cleaned signal itself (identity target)
    // - Spectral expectations come from a forward/inverse simulated pass on the cleaned signal
    const std::vector<double> expected_pcm = signal;
    SimulationResult expected_spec = simulate_stream_identity(signal, AMP_FFT_WINDOW_HANN, g_config.window_size);
    if (expected_spec.spectral_real.size() != signal.size() * static_cast<size_t>(g_config.window_size) ||
        expected_spec.spectral_imag.size() != signal.size() * static_cast<size_t>(g_config.window_size)) {
        record_failure(
            "expected_spec spectral lengths mismatch real=%zu imag=%zu expected=%zu",
            expected_spec.spectral_real.size(),
            expected_spec.spectral_imag.size(),
            signal.size() * static_cast<size_t>(g_config.window_size)
        );
        return 1;
    }

    RunResult first = run_fft_node_once(signal);
    RunResult second = run_fft_node_once(signal);

    const size_t expected_pcm_frames = expected_pcm.size();
    const size_t pcm_frames_to_check = std::min(expected_pcm_frames, first.pcm_frames_committed);
    if (pcm_frames_to_check == 0) {
        record_failure("pcm_frames_to_check is zero");
    } else {
        verify_close("pcm_vs_expected", first.pcm.data(), expected_pcm.data(), pcm_frames_to_check, g_config.tolerance);
    }

    const size_t expected_spectral_frames = expected_spec.spectral_real.size() /
        static_cast<size_t>(g_config.window_size);
    const size_t spectral_frames_to_check = std::min(first.spectral_rows_committed, expected_spectral_frames);
    const size_t spectral_values_to_check = spectral_frames_to_check *
        static_cast<size_t>(g_config.window_size);
    if (spectral_frames_to_check == 0) {
        record_failure("spectral_frames_to_check is zero");
    } else {
        const double *actual_real_ptr = first.spectral_real.data();
        const double *actual_imag_ptr = first.spectral_imag.data();
        const double *expected_real_ptr = expected_spec.spectral_real.data();
        const double *expected_imag_ptr = expected_spec.spectral_imag.data();
        verify_close(
            "spectral_real_vs_expected",
            actual_real_ptr,
            expected_real_ptr,
            spectral_values_to_check,
            g_config.tolerance
        );
        verify_close(
            "spectral_imag_vs_expected",
            actual_imag_ptr,
            expected_imag_ptr,
            spectral_values_to_check,
            g_config.tolerance
        );
        if (first.spectral_rows_committed != expected_spectral_frames) {
            emit_diagnostic(
                "spectral rows committed %zu != expected %zu",
                first.spectral_rows_committed,
                expected_spectral_frames
            );
        }
    }

    if (first.pcm_frames_committed != second.pcm_frames_committed) {
        record_failure(
            "pcm_frames_committed mismatch first=%zu second=%zu",
            first.pcm_frames_committed,
            second.pcm_frames_committed
        );
    }

    if (first.spectral_rows_committed != second.spectral_rows_committed) {
        record_failure(
            "spectral_rows_committed mismatch first=%zu second=%zu",
            first.spectral_rows_committed,
            second.spectral_rows_committed
        );
    }

    require_identity(signal, first.pcm, "forward_identity_first");
    require_identity(signal, second.pcm, "forward_identity_second");
    require_equal(first.pcm, second.pcm, "pcm_repeat_stability");
    require_equal(first.spectral_real, second.spectral_real, "spectral_real_repeat_stability");
    require_equal(first.spectral_imag, second.spectral_imag, "spectral_imag_repeat_stability");

    verify_metrics(first.metrics, "forward_metrics");
    verify_metrics(second.metrics, "repeat_metrics");

    const bool forward_failed = g_failed;
    if (!forward_failed) {
        std::vector<double> raw_streaming_signal(g_config.streaming_frames, 0.0);
        for (size_t i = 0; i < raw_streaming_signal.size(); ++i) {
            double t = static_cast<double>(i);
            raw_streaming_signal[i] = std::sin(0.005 * t) * std::cos(0.013 * t);
        }

        SimulationResult streaming_preconditioned = simulate_stream_identity(
            raw_streaming_signal,
            AMP_FFT_WINDOW_HANN,
            g_config.window_size
        );
        if (streaming_preconditioned.pcm.size() != raw_streaming_signal.size()) {
            record_failure(
                "streaming_preconditioned length %zu != expected %zu",
                streaming_preconditioned.pcm.size(),
                raw_streaming_signal.size()
            );
        }
        const std::vector<double> streaming_signal = streaming_preconditioned.pcm;

        SimulationResult streaming_expected = simulate_stream_identity(
            streaming_signal,
            AMP_FFT_WINDOW_HANN,
            g_config.window_size
        );
        StreamingRunResult streaming_result = run_fft_node_streaming(streaming_signal, g_config.streaming_chunk);

        verify_close(
            "streaming_pcm_vs_expected",
            streaming_result.pcm.data(),
            streaming_expected.pcm.data(),
            streaming_expected.pcm.size(),
            g_config.tolerance
        );
        verify_close(
            "streaming_spectral_real_vs_expected",
            streaming_result.spectral_real.data(),
            streaming_expected.spectral_real.data(),
            streaming_expected.spectral_real.size(),
            g_config.tolerance
        );
        verify_close(
            "streaming_spectral_imag_vs_expected",
            streaming_result.spectral_imag.data(),
            streaming_expected.spectral_imag.data(),
            streaming_expected.spectral_imag.size(),
            g_config.tolerance
        );

        if (streaming_result.call_count != (g_config.streaming_frames + g_config.streaming_chunk - 1) / g_config.streaming_chunk) {
            record_failure(
                "streaming call count mismatch got %zu expected %zu",
                streaming_result.call_count,
                (g_config.streaming_frames + g_config.streaming_chunk - 1) / g_config.streaming_chunk
            );
        }

        if (!streaming_result.state_allocated) {
            record_failure("streaming_result did not retain node state");
        }

        if (streaming_result.metrics_per_call.size() != streaming_result.call_count) {
            record_failure(
                "metrics_per_call size %zu does not match call count %zu",
                streaming_result.metrics_per_call.size(),
                streaming_result.call_count
            );
        }

        for (size_t i = 0; i < streaming_result.metrics_per_call.size(); ++i) {
            verify_metrics(streaming_result.metrics_per_call[i], "streaming_metrics");
        }
    } else {
        emit_diagnostic("skipping streaming checks because forward regression failed");
    }

    require_backward_unsupported(signal, first.pcm);

    if (g_failed) {
        std::printf("test_fft_division_node: FAIL\n");
        return 1;
    }

    std::printf("test_fft_division_node: PASS\n");
    return 0;
}

}  // namespace amp::tests::fft_division_shared

#if defined(AMP_TEST_FFT_DIVISION_SHARED_STANDALONE)
int main(int argc, char **argv) {
    return amp::tests::fft_division_shared::RunShared(argc, argv);
}
#endif  // AMP_TEST_FFT_DIVISION_SHARED_STANDALONE

#endif  // AMP_TEST_FFT_DIVISION_SHARED_CPP

