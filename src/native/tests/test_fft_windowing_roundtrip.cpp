#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../include/amp_fft_backend.h"

namespace {

constexpr double kTolerance = 1e-4;

struct WindowScenario {
    int n;
    int window;
    int hop;
    int window_kind;
    const char *name;
};

std::vector<double> make_signal(std::size_t total, std::size_t guard) {
    std::vector<double> pcm(total, 0.0);
    const std::size_t payload = (total > 2 * guard) ? (total - 2 * guard) : 0U;
    for (std::size_t i = 0; i < payload; ++i) {
        const std::size_t idx = guard + i;
        const double t = static_cast<double>(i);
        pcm[idx] = std::sin(0.17 * t) + 0.35 * std::cos(0.07 * t) + 0.1 * std::sin(0.013 * t * t);
    }
    return pcm;
}

bool compare_trimmed(const std::vector<double> &input,
                     const std::vector<double> &reconstructed,
                     std::size_t guard,
                     std::size_t payload) {
    if (input.size() <= 2 * guard || reconstructed.size() <= 2 * guard) {
        std::cerr << "insufficient samples after trimming guard" << std::endl;
        return false;
    }
    const std::size_t start = guard;
    const std::size_t input_stop = input.size() - guard;
    const std::size_t recon_stop = reconstructed.size() > guard
        ? (reconstructed.size() - guard)
        : guard;
    const std::size_t stop = std::min(input_stop, recon_stop);
    if (stop <= start || (stop - start) < payload) {
        std::cerr << "insufficient overlap for comparison" << std::endl;
        return false;
    }
    double max_err = 0.0;
    for (std::size_t i = 0; i < payload; ++i) {
        const double expected = input[start + i];
        const double actual = reconstructed[start + i];
        const double diff = std::fabs(actual - expected);
        max_err = std::max(max_err, diff);
        if (diff > kTolerance) {
            std::cerr << "sample mismatch at i=" << i
                      << " expected=" << expected
                      << " actual=" << actual
                      << " diff=" << diff << std::endl;
            return false;
        }
    }
    std::cout << "max error after trimming guard=" << max_err << std::endl;
    return true;
}

bool run_roundtrip(const WindowScenario &scenario) {
    void *forward = amp_fft_backend_stream_create(scenario.n, scenario.window, scenario.hop, scenario.window_kind);
    if (forward == nullptr) {
        std::cerr << "failed to create forward stream for scenario " << scenario.name << std::endl;
        return false;
    }
    void *inverse = amp_fft_backend_stream_create_inverse(scenario.n, scenario.window, scenario.hop, scenario.window_kind);
    if (inverse == nullptr) {
        std::cerr << "failed to create inverse stream for scenario " << scenario.name << std::endl;
        amp_fft_backend_stream_destroy(forward);
        return false;
    }

    const std::size_t guard = static_cast<std::size_t>(scenario.window);
    const std::size_t payload = 64U;
    const std::size_t total_samples = payload + 2 * guard;
    std::vector<double> input = make_signal(total_samples, guard);

    const std::array<std::size_t, 4> chunk_pattern{3, 5, 7, static_cast<std::size_t>(scenario.window)};
    const std::size_t max_frame_block = static_cast<std::size_t>(scenario.window) + 8U;
    const std::size_t frame_width = static_cast<std::size_t>(scenario.n);
    std::vector<double> spectral_real(max_frame_block * frame_width, 0.0);
    std::vector<double> spectral_imag(max_frame_block * frame_width, 0.0);
    std::vector<double> pcm_block(max_frame_block * static_cast<std::size_t>(scenario.window), 0.0);
    std::vector<double> reconstructed;
    reconstructed.reserve(total_samples + scenario.window * 2);

    std::size_t processed = 0U;
    std::size_t chunk_index = 0U;
    while (processed < input.size()) {
        const std::size_t remaining = input.size() - processed;
        const std::size_t chunk = std::min(chunk_pattern[chunk_index % chunk_pattern.size()], remaining);
        const int flush_mode = (processed + chunk >= input.size())
            ? AMP_FFT_STREAM_FLUSH_FINAL
            : AMP_FFT_STREAM_FLUSH_NONE;
        const std::size_t frames = amp_fft_backend_stream_push(
            forward,
            input.data() + processed,
            chunk,
            scenario.n,
            spectral_real.data(),
            spectral_imag.data(),
            max_frame_block,
            flush_mode);
        processed += chunk;
        chunk_index += 1U;
        if (frames > 0) {
            const std::size_t produced_pcm = amp_fft_backend_stream_push_spectrum(
                inverse,
                spectral_real.data(),
                spectral_imag.data(),
                frames,
                scenario.n,
                pcm_block.data(),
                pcm_block.size(),
                AMP_FFT_STREAM_FLUSH_NONE);
            reconstructed.insert(reconstructed.end(), pcm_block.begin(), pcm_block.begin() + produced_pcm);
        }
    }

    const std::size_t drained = amp_fft_backend_stream_push_spectrum(
        inverse,
        nullptr,
        nullptr,
        0,
        scenario.n,
        pcm_block.data(),
        pcm_block.size(),
        AMP_FFT_STREAM_FLUSH_FINAL);
    reconstructed.insert(reconstructed.end(), pcm_block.begin(), pcm_block.begin() + drained);

    amp_fft_backend_stream_destroy(forward);
    amp_fft_backend_stream_destroy(inverse);

    if (!compare_trimmed(input, reconstructed, guard, payload)) {
        std::cerr << "round-trip mismatch for scenario " << scenario.name << std::endl;
        return false;
    }

    std::cout << "scenario " << scenario.name << " passed with "
              << reconstructed.size() << " reconstructed samples" << std::endl;
    return true;
}

} // namespace

int main() {
    const std::vector<WindowScenario> scenarios{
        {4, 4, 2, AMP_FFT_WINDOW_RECT, "rect_w4_h2"},
        {8, 8, 4, AMP_FFT_WINDOW_HANN, "hann_w8_h4"},
        {16, 16, 8, AMP_FFT_WINDOW_HAMMING, "hamming_w16_h8"},
        {32, 32, 16, AMP_FFT_WINDOW_BLACKMAN, "blackman_w32_h16"}
    };

    bool all_passed = true;
    for (const auto &scenario : scenarios) {
        std::cout << "Running scenario: " << scenario.name << std::endl;
        const bool scenario_passed = run_roundtrip(scenario);
        all_passed = all_passed && scenario_passed;
    }

    if (!all_passed) {
        std::cerr << "windowing round-trip test failed" << std::endl;
        return 1;
    }

    std::cout << "All windowing scenarios passed" << std::endl;
    return 0;
}
