#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

extern "C" {
#include "amp_fft_backend.h"
#include "amp_native.h"
}

namespace {

constexpr int kFrames = 8;
constexpr int kWindowSize = 4;
constexpr double kSampleRate = 48000.0;
constexpr double kTolerance = 1e-9;
constexpr size_t kStreamingFrames = 4096;
constexpr size_t kStreamingChunk = 64;

bool g_failed = false;

void record_failure(const char *fmt, ...) {
    g_failed = true;
    std::fprintf(stderr, "[fft_division_node] ");

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
}

bool nearly_equal(double a, double b, double tol = kTolerance) {
    return std::fabs(a - b) <= tol;
}

struct RunResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    AmpNodeMetrics metrics{};
};

struct StreamingRunResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    std::vector<AmpNodeMetrics> metrics_per_call;
    size_t call_count{0};
    bool state_allocated{false};
};

struct SimulationResult {
    std::vector<double> pcm;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
};

std::string build_params_json() {
    char buffer[256];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "{\"window_size\":%d,\"algorithm\":\"fft\",\"window\":\"hann\",\"supports_v2\":true,"
        "\"declared_delay\":%d,\"oversample_ratio\":1,\"epsilon\":1e-9,\"max_batch_windows\":1,"
        "\"backend_hop\":1}",
        kWindowSize,
        kWindowSize - 1
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
    if (signal.size() != static_cast<size_t>(kFrames)) {
        record_failure(
            "signal length %zu does not match expected frame count %d",
            signal.size(),
            kFrames
        );
    }
    return build_audio_view_span(signal.data(), signal.size());
}

RunResult run_fft_node_once(const std::vector<double> &signal) {
    RunResult result;
    result.pcm.assign(signal.size(), 0.0);
    result.spectral_real.assign(signal.size() * kWindowSize, 0.0);
    result.spectral_imag.assign(signal.size() * kWindowSize, 0.0);

    std::string params_json;
    EdgeRunnerNodeDescriptor descriptor = build_descriptor(params_json);

    EdgeRunnerParamSet params{};
    params.count = 0U;
    params.items = nullptr;

    std::vector<uint8_t> frame_filled(signal.size(), 0U);
    void *state = nullptr;
    size_t consumed_frames = 0U;
    size_t produced_frames = 0U;
    AmpNodeMetrics last_metrics{};
    int delay_frames = kWindowSize - 1;
    bool delay_initialized = false;

    const size_t total_frames = signal.size();
    const size_t max_iterations = total_frames * 8U;
    size_t iteration = 0U;

    while (produced_frames < total_frames && iteration < max_iterations) {
        iteration += 1U;

        EdgeRunnerAudioView audio{};
        if (consumed_frames < total_frames) {
            audio = build_audio_view_span(signal.data() + consumed_frames, 1U);
            consumed_frames += static_cast<size_t>(audio.frames);
        } else {
            audio.has_audio = 1U;
            audio.batches = 1U;
            audio.channels = 1U;
            audio.frames = 1U;
            audio.data = signal.empty() ? nullptr : signal.data() + (total_frames - 1);
        }
        const size_t consumed_before = (consumed_frames > 0U) ? (consumed_frames - static_cast<size_t>(audio.frames)) : 0U;

        std::array<double, kWindowSize> spectral_real_frame{};
        std::array<double, kWindowSize> spectral_imag_frame{};

        EdgeRunnerTapBuffer tap_buffers[2]{};
        tap_buffers[0].tap_name = "spectral_real";
        tap_buffers[0].buffer_class = nullptr;
        tap_buffers[0].shape.batches = 1U;
        tap_buffers[0].shape.channels = kWindowSize;
        tap_buffers[0].shape.frames = audio.frames;
        tap_buffers[0].frame_stride = kWindowSize;
        tap_buffers[0].data = spectral_real_frame.data();

        tap_buffers[1].tap_name = "spectral_imag";
        tap_buffers[1].buffer_class = nullptr;
        tap_buffers[1].shape = tap_buffers[0].shape;
        tap_buffers[1].frame_stride = kWindowSize;
        tap_buffers[1].data = spectral_imag_frame.data();

        EdgeRunnerTapBufferSet tap_set{};
        tap_set.items = tap_buffers;
        tap_set.count = 2U;

        EdgeRunnerTapStatusSet status_set{};
        status_set.items = nullptr;
        status_set.count = 0U;

        EdgeRunnerTapContext tap_context{};
        tap_context.outputs = tap_set;
        tap_context.status = status_set;

        EdgeRunnerNodeInputs inputs{};
        inputs.audio = audio;
        inputs.params = params;
        inputs.taps = tap_context;

        double *out_buffer = nullptr;
        int out_channels = 0;
        AmpNodeMetrics metrics{};

        const int rc = amp_run_node_v2(
            &descriptor,
            &inputs,
            1,
            1,
            static_cast<int>(audio.frames),
            kSampleRate,
            &out_buffer,
            &out_channels,
            &state,
            nullptr,
            AMP_EXECUTION_MODE_FORWARD,
            &metrics
        );

        if (rc == AMP_E_PENDING) {
            if (out_buffer != nullptr) {
                amp_free(out_buffer);
            }
            continue;
        }

        if (rc != 0 || out_buffer == nullptr || out_channels != 1) {
            record_failure(
                "amp_run_node_v2 forward failed rc=%d buffer=%p channels=%d",
                rc,
                static_cast<void *>(out_buffer),
                out_channels
            );
        } else {
            if (!delay_initialized) {
                delay_frames = static_cast<int>(metrics.measured_delay_frames);
                delay_initialized = true;
            }

            const size_t frames_emitted = static_cast<size_t>(audio.frames);
            for (size_t frame = 0; frame < frames_emitted; ++frame) {
                const int64_t input_index = static_cast<int64_t>(consumed_before)
                    + static_cast<int64_t>(frame)
                    - static_cast<int64_t>(delay_frames);
                if (input_index < 0) {
                    continue;
                }
                if (input_index >= static_cast<int64_t>(signal.size())) {
                    continue;
                }

                const size_t target_index = static_cast<size_t>(input_index);
                if (frame_filled[target_index]) {
                    record_failure(
                        "amp_run_node_v2 produced duplicate output for frame %zu",
                        target_index
                    );
                    continue;
                }

                result.pcm[target_index] = out_buffer[frame];

                double *dest_real = result.spectral_real.data() + target_index * kWindowSize;
                double *dest_imag = result.spectral_imag.data() + target_index * kWindowSize;
                const double *src_real = spectral_real_frame.data() + frame * kWindowSize;
                const double *src_imag = spectral_imag_frame.data() + frame * kWindowSize;
                std::copy(src_real, src_real + kWindowSize, dest_real);
                std::copy(src_imag, src_imag + kWindowSize, dest_imag);

                frame_filled[target_index] = 1U;
                produced_frames += 1U;
            }

            last_metrics = metrics;
        }

        if (out_buffer != nullptr) {
            amp_free(out_buffer);
        }
    }

    if (produced_frames < total_frames) {
        record_failure(
            "amp_run_node_v2 forward produced %zu of %zu frames after %zu iterations",
            produced_frames,
            total_frames,
            iteration
        );
    }

    if (state != nullptr) {
        amp_release_state(state);
        state = nullptr;
    }

    for (size_t i = 0; i < frame_filled.size(); ++i) {
        if (frame_filled[i] == 0U) {
            record_failure("amp_run_node_v2 forward left frame %zu pending", i);
        }
    }

    result.metrics = last_metrics;
    return result;
}

StreamingRunResult run_fft_node_streaming(const std::vector<double> &signal, size_t chunk_frames) {
    StreamingRunResult result;
    const size_t total_frames = signal.size();
    result.pcm.assign(total_frames, 0.0);
    result.spectral_real.assign(total_frames * kWindowSize, 0.0);
    result.spectral_imag.assign(total_frames * kWindowSize, 0.0);

    if (chunk_frames == 0) {
        record_failure("chunk size must be greater than zero");
        return result;
    }

    std::string params_json;
    EdgeRunnerNodeDescriptor descriptor = build_descriptor(params_json);

    EdgeRunnerParamSet params{};
    params.count = 0U;
    params.items = nullptr;

    void *state = nullptr;
    bool saw_state = false;
    size_t produced_frames = 0U;
    size_t consumed_frames = 0U;
    std::vector<uint8_t> frame_filled(total_frames, 0U);
    int delay_frames = kWindowSize - 1;
    bool delay_initialized = false;
    const size_t max_iterations = (total_frames / chunk_frames + 1U) * 8U;
    size_t iteration = 0U;

    size_t offset = 0U;
    while (produced_frames < total_frames && iteration < max_iterations) {
        const size_t call_index = iteration;
        iteration += 1U;

        const size_t frames_available = (offset < total_frames)
            ? std::min(chunk_frames, total_frames - offset)
            : chunk_frames;
        const size_t frames = (frames_available > 0U) ? frames_available : chunk_frames;

        EdgeRunnerAudioView audio{};
        if (offset < total_frames) {
            audio = build_audio_view_span(signal.data() + offset, frames);
            offset += frames;
        } else {
            audio.has_audio = 1U;
            audio.batches = 1U;
            audio.channels = 1U;
            audio.frames = static_cast<uint32_t>(frames);
            audio.data = signal.empty() ? nullptr : signal.data() + (total_frames - frames);
        }
        const size_t consumed_before = consumed_frames;
        consumed_frames += frames;

        std::vector<double> spectral_real_chunk(frames * kWindowSize, 0.0);
        std::vector<double> spectral_imag_chunk(frames * kWindowSize, 0.0);

        EdgeRunnerTapBuffer tap_buffers[2]{};
        tap_buffers[0].tap_name = "spectral_real";
        tap_buffers[0].buffer_class = nullptr;
        tap_buffers[0].shape.batches = 1U;
        tap_buffers[0].shape.channels = kWindowSize;
        tap_buffers[0].shape.frames = static_cast<uint32_t>(frames);
        tap_buffers[0].frame_stride = kWindowSize;
        tap_buffers[0].data = spectral_real_chunk.data();

        tap_buffers[1].tap_name = "spectral_imag";
        tap_buffers[1].buffer_class = nullptr;
        tap_buffers[1].shape = tap_buffers[0].shape;
        tap_buffers[1].frame_stride = kWindowSize;
        tap_buffers[1].data = spectral_imag_chunk.data();

        EdgeRunnerTapBufferSet tap_set{};
        tap_set.items = tap_buffers;
        tap_set.count = 2U;

        EdgeRunnerTapStatusSet status_set{};
        status_set.items = nullptr;
        status_set.count = 0U;

        EdgeRunnerTapContext tap_context{};
        tap_context.outputs = tap_set;
        tap_context.status = status_set;

        EdgeRunnerNodeInputs inputs{};
        inputs.audio = audio;
        inputs.params = params;
        inputs.taps = tap_context;

        double *out_buffer = nullptr;
        int out_channels = 0;
        AmpNodeMetrics metrics{};

        const int rc = amp_run_node_v2(
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

        result.metrics_per_call.push_back(metrics);
        result.call_count += 1;

        if (rc == AMP_E_PENDING) {
            if (out_buffer != nullptr) {
                amp_free(out_buffer);
            }
        } else if (rc != 0 || out_buffer == nullptr || out_channels != 1) {
            record_failure(
                "streaming call %zu failed rc=%d buffer=%p channels=%d",
                call_index,
                rc,
                static_cast<void *>(out_buffer),
                out_channels
            );
            if (out_buffer != nullptr) {
                amp_free(out_buffer);
            }
        } else {
            if (!delay_initialized) {
                delay_frames = static_cast<int>(metrics.measured_delay_frames);
                delay_initialized = true;
            }

            const size_t frames_emitted = frames;

            for (size_t frame = 0; frame < frames_emitted; ++frame) {
                const int64_t input_index = static_cast<int64_t>(consumed_before)
                    + static_cast<int64_t>(frame)
                    - static_cast<int64_t>(delay_frames);
                if (input_index < 0) {
                    continue;
                }
                if (input_index >= static_cast<int64_t>(total_frames)) {
                    continue;
                }
                const size_t target_index = static_cast<size_t>(input_index);
                if (frame_filled[target_index]) {
                    record_failure(
                        "streaming call %zu produced duplicate output for frame %zu",
                        call_index,
                        target_index
                    );
                    continue;
                }

                result.pcm[target_index] = out_buffer[frame];

                double *dest_real = result.spectral_real.data() + target_index * kWindowSize;
                double *dest_imag = result.spectral_imag.data() + target_index * kWindowSize;
                const double *src_real = spectral_real_chunk.data() + frame * kWindowSize;
                const double *src_imag = spectral_imag_chunk.data() + frame * kWindowSize;
                std::copy(src_real, src_real + kWindowSize, dest_real);
                std::copy(src_imag, src_imag + kWindowSize, dest_imag);

                frame_filled[target_index] = 1U;
                produced_frames += 1U;
            }

            if (out_buffer != nullptr) {
                amp_free(out_buffer);
            }
        }

        if (state == nullptr) {
            record_failure("streaming call %zu did not return persistent state", call_index);
        } else {
            saw_state = true;
            result.state_allocated = true;
        }

    }

    if (produced_frames < total_frames) {
        record_failure(
            "streaming run produced %zu of %zu frames after %zu iterations",
            produced_frames,
            total_frames,
            iteration
        );
    }

    if (state != nullptr) {
        amp_release_state(state);
        state = nullptr;
    }

    if (!saw_state) {
        record_failure("streaming run never produced node state");
    }

    for (size_t i = 0; i < frame_filled.size(); ++i) {
        if (frame_filled[i] == 0U) {
            record_failure("streaming run left frame %zu pending", i);
        }
    }

    return result;
}

void verify_close(const char *label, const double *actual, const double *expected, size_t count, double tolerance) {
    for (size_t i = 0; i < count; ++i) {
        const double diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            record_failure(
                "%s mismatch at index %zu: got %.12f expected %.12f diff %.12f",
                label,
                i,
                actual[i],
                expected[i],
                diff
            );
            return;
        }
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
    if (metrics.measured_delay_frames != static_cast<uint32_t>(kWindowSize - 1)) {
        record_failure(
            "%s unexpected delay %u (expected %u)",
            label,
            metrics.measured_delay_frames,
            static_cast<uint32_t>(kWindowSize - 1)
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

SimulationResult simulate_stream_identity(const std::vector<double> &signal, int window_kind) {
    SimulationResult result;
    result.pcm.assign(signal.size(), 0.0);
    result.spectral_real.assign(signal.size() * kWindowSize, 0.0);
    result.spectral_imag.assign(signal.size() * kWindowSize, 0.0);

    void *forward = amp_fft_backend_stream_create(kWindowSize, kWindowSize, 1, window_kind);
    void *inverse = amp_fft_backend_stream_create_inverse(kWindowSize, kWindowSize, 1, window_kind);
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

    std::vector<double> spectral_real(kWindowSize, 0.0);
    std::vector<double> spectral_imag(kWindowSize, 0.0);
    std::vector<double> inverse_scratch(kWindowSize, 0.0);
    std::deque<double> pending_pcm;
    bool warmup_complete = false;

    for (size_t frame = 0; frame < signal.size(); ++frame) {
        double sample = signal[frame];
        size_t frames_emitted = amp_fft_backend_stream_push(
            forward,
            &sample,
            1,
            kWindowSize,
            spectral_real.data(),
            spectral_imag.data(),
            1,
            AMP_FFT_STREAM_FLUSH_NONE
        );

        if (frames_emitted > 0) {
            warmup_complete = true;
        } else {
            std::fill(spectral_real.begin(), spectral_real.end(), 0.0);
            std::fill(spectral_imag.begin(), spectral_imag.end(), 0.0);
        }

        double *real_slot = result.spectral_real.data() + frame * kWindowSize;
        double *imag_slot = result.spectral_imag.data() + frame * kWindowSize;
        std::copy(spectral_real.begin(), spectral_real.end(), real_slot);
        std::copy(spectral_imag.begin(), spectral_imag.end(), imag_slot);

        if (frames_emitted > 0) {
            const size_t produced = amp_fft_backend_stream_push_spectrum(
                inverse,
                spectral_real.data(),
                spectral_imag.data(),
                frames_emitted,
                kWindowSize,
                inverse_scratch.data(),
                inverse_scratch.size(),
                AMP_FFT_STREAM_FLUSH_NONE
            );
            for (size_t i = 0; i < produced; ++i) {
                pending_pcm.push_back(inverse_scratch[i]);
            }
        }

        double pcm_value = sample;
        if (warmup_complete && !pending_pcm.empty()) {
            pcm_value = pending_pcm.front();
            pending_pcm.pop_front();
        }
        result.pcm[frame] = pcm_value;
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

    const int rc = amp_run_node_v2(
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

int main() {
    const std::vector<double> signal{
        1.0,
        -0.5,
        0.25,
        -0.125,
        0.0625,
        -0.03125,
        0.015625,
        -0.0078125
    };

    SimulationResult expected = simulate_stream_identity(signal, AMP_FFT_WINDOW_HANN);

    RunResult first = run_fft_node_once(signal);
    RunResult second = run_fft_node_once(signal);

    std::vector<double> streaming_signal(kStreamingFrames, 0.0);
    for (size_t i = 0; i < streaming_signal.size(); ++i) {
        double t = static_cast<double>(i);
        streaming_signal[i] = std::sin(0.005 * t) * std::cos(0.013 * t);
    }

    SimulationResult streaming_expected = simulate_stream_identity(streaming_signal, AMP_FFT_WINDOW_HANN);
    StreamingRunResult streaming_result = run_fft_node_streaming(streaming_signal, kStreamingChunk);

    verify_close("pcm_vs_expected", first.pcm.data(), expected.pcm.data(), expected.pcm.size(), 1e-9);
    verify_close(
        "spectral_real_vs_expected",
        first.spectral_real.data(),
        expected.spectral_real.data(),
        expected.spectral_real.size(),
        1e-9
    );
    verify_close(
        "spectral_imag_vs_expected",
        first.spectral_imag.data(),
        expected.spectral_imag.data(),
        expected.spectral_imag.size(),
        1e-9
    );

    require_identity(signal, first.pcm, "forward_identity_first");
    require_identity(signal, second.pcm, "forward_identity_second");
    require_equal(first.pcm, second.pcm, "pcm_repeat_stability");
    require_equal(first.spectral_real, second.spectral_real, "spectral_real_repeat_stability");
    require_equal(first.spectral_imag, second.spectral_imag, "spectral_imag_repeat_stability");

    verify_metrics(first.metrics, "forward_metrics");
    verify_metrics(second.metrics, "repeat_metrics");

    verify_close(
        "streaming_pcm_vs_expected",
        streaming_result.pcm.data(),
        streaming_expected.pcm.data(),
        streaming_expected.pcm.size(),
        1e-9
    );
    verify_close(
        "streaming_spectral_real_vs_expected",
        streaming_result.spectral_real.data(),
        streaming_expected.spectral_real.data(),
        streaming_expected.spectral_real.size(),
        1e-8
    );
    verify_close(
        "streaming_spectral_imag_vs_expected",
        streaming_result.spectral_imag.data(),
        streaming_expected.spectral_imag.data(),
        streaming_expected.spectral_imag.size(),
        1e-8
    );

    const size_t expected_streaming_calls = (kStreamingFrames + kStreamingChunk - 1) / kStreamingChunk;
    if (streaming_result.call_count < expected_streaming_calls) {
        record_failure(
            "streaming call count %zu below minimum %zu",
            streaming_result.call_count,
            expected_streaming_calls
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

    require_backward_unsupported(signal, first.pcm);

    if (g_failed) {
        std::printf("test_fft_division_node: FAIL\n");
        return 1;
    }

    std::printf("test_fft_division_node: PASS\n");
    return 0;
}

