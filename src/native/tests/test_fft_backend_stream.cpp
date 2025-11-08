#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "mock_fftfree.hpp"

#define AMP_NATIVE_USE_FFTFREE 1
#include "../fft_backend.cpp"

namespace {

bool nearly_equal(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

bool compare_float_vector(const std::vector<float> &floats, const std::vector<double> &doubles) {
    if (floats.size() != doubles.size()) {
        return false;
    }
    for (std::size_t i = 0; i < floats.size(); ++i) {
        if (!nearly_equal(static_cast<double>(floats[i]), static_cast<double>(static_cast<float>(doubles[i])))) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace mock_fftfree;

    reset_all();

    const int n = 4;
    const int window = 4;
    const int hop = 2;

    void *stream = amp_fft_backend_stream_create(n, window, hop, AMP_FFT_WINDOW_RECT);
    if (stream == nullptr) {
        std::cerr << "failed to create streaming context" << std::endl;
        return 1;
    }

    if (init_full_call_count() != 1) {
        std::cerr << "expected single fft_init_full_v2 call for stream creation" << std::endl;
        return 1;
    }

    ContextState created_state = describe_handle(stream);
    if (created_state.n != n || created_state.window != window || created_state.hop != hop ||
        created_state.stft_mode != 2 || !created_state.apply_windows) {
        std::cerr << "stream context parameters not forwarded correctly" << std::endl;
        return 1;
    }

    std::vector<double> chunk1{1.0, -2.0, 3.0, -4.0};
    std::vector<double> out_real1(static_cast<std::size_t>(n), 0.0);
    std::vector<double> out_imag1(static_cast<std::size_t>(n), 0.0);

    std::size_t frames1 = amp_fft_backend_stream_push(
        stream,
        chunk1.data(),
        chunk1.size(),
        n,
        out_real1.data(),
        out_imag1.data(),
        1,
        AMP_FFT_STREAM_FLUSH_NONE);

    if (frames1 != 1) {
        std::cerr << "expected one frame from initial push" << std::endl;
        return 1;
    }

    if (stream_calls().size() != 1) {
        std::cerr << "stream push not logged" << std::endl;
        return 1;
    }

    const auto first_call = stream_calls().front();
    if (first_call.samples != chunk1.size() || first_call.max_frames != 1 || first_call.produced != frames1) {
        std::cerr << "unexpected streaming call metadata" << std::endl;
        return 1;
    }
    if (!compare_float_vector(first_call.pcm, chunk1)) {
        std::cerr << "streaming PCM not converted to float" << std::endl;
        return 1;
    }

    for (std::size_t i = 0; i < out_real1.size(); ++i) {
        if (!nearly_equal(out_real1[i], chunk1[i]) || !nearly_equal(out_imag1[i], -chunk1[i])) {
            std::cerr << "streaming output mismatch after first push at index " << i << std::endl;
            return 1;
        }
    }

    ContextState after_first_push = describe_handle(stream);
    if (after_first_push.pending_samples != static_cast<std::size_t>(hop)) {
        std::cerr << "unexpected pending sample count after first push" << std::endl;
        return 1;
    }

    clear_call_history();

    std::vector<double> chunk2{5.0, -6.0};
    std::vector<double> out_real2(static_cast<std::size_t>(n), 0.0);
    std::vector<double> out_imag2(static_cast<std::size_t>(n), 0.0);

    std::size_t frames2 = amp_fft_backend_stream_push(
        stream,
        chunk2.data(),
        chunk2.size(),
        n,
        out_real2.data(),
        out_imag2.data(),
        1,
        AMP_FFT_STREAM_FLUSH_NONE);

    if (frames2 != 1) {
        std::cerr << "expected overlapped frame after second push" << std::endl;
        return 1;
    }

    if (stream_calls().size() != 1) {
        std::cerr << "second streaming push not logged" << std::endl;
        return 1;
    }

    const auto second_call = stream_calls().front();
    if (second_call.samples != chunk2.size() || second_call.produced != frames2) {
        std::cerr << "unexpected metadata for second streaming push" << std::endl;
        return 1;
    }
    if (!compare_float_vector(second_call.pcm, chunk2)) {
        std::cerr << "second streaming PCM not converted to float" << std::endl;
        return 1;
    }

    const std::vector<double> expected_second_frame{3.0, -4.0, 5.0, -6.0};
    for (std::size_t i = 0; i < out_real2.size(); ++i) {
        if (!nearly_equal(out_real2[i], expected_second_frame[i]) ||
            !nearly_equal(out_imag2[i], -expected_second_frame[i])) {
            std::cerr << "streaming output mismatch after second push at index " << i << std::endl;
            return 1;
        }
    }

    ContextState after_second_push = describe_handle(stream);
    if (after_second_push.pending_samples != chunk2.size()) {
        std::cerr << "unexpected pending sample count after second push" << std::endl;
        return 1;
    }

    amp_fft_backend_stream_destroy(stream);
    if (free_call_count() != 1) {
        std::cerr << "stream destroy did not forward to fft_free" << std::endl;
        return 1;
    }

    return 0;
}
