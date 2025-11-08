#include <algorithm>
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
    const int batch = 2;
    const std::size_t total = static_cast<std::size_t>(n * batch);

    std::vector<double> in_real{1.0, -2.5, 3.25, -4.0, 0.5, 0.25, -0.75, 2.0};
    std::vector<double> in_imag{0.25, -0.5, 0.75, -1.0, 1.5, -1.25, 0.5, -0.25};
    std::vector<double> out_real(total, 0.0);
    std::vector<double> out_imag(total, 0.0);

    if (amp_fft_backend_transform_many(in_real.data(), nullptr, out_real.data(), out_imag.data(), n, batch, 0) != 1) {
        std::cerr << "forward transform failed" << std::endl;
        return 1;
    }

    if (batch_calls().size() != 1) {
        std::cerr << "expected single fft_execute_batched call for real input" << std::endl;
        return 1;
    }

    const auto forward_call = batch_calls().front();
    if (forward_call.samples != total || forward_call.frames != static_cast<std::size_t>(batch)) {
        std::cerr << "unexpected sample/frame counts for fft_execute_batched" << std::endl;
        return 1;
    }

    if (!compare_float_vector(forward_call.pcm, in_real)) {
        std::cerr << "real input not converted to float as expected" << std::endl;
        return 1;
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!nearly_equal(out_real[i], in_real[i]) || !nearly_equal(out_imag[i], 0.0)) {
            std::cerr << "forward output mismatch at index " << i << std::endl;
            return 1;
        }
    }

    const int forward_inits = init_full_call_count();
    if (forward_inits != 1) {
        std::cerr << "expected single fft_init_full_v2 invocation for forward context" << std::endl;
        return 1;
    }

    clear_call_history();
    if (amp_fft_backend_transform_many(in_real.data(), nullptr, out_real.data(), out_imag.data(), n, batch, 0) != 1) {
        std::cerr << "second forward transform failed" << std::endl;
        return 1;
    }
    if (init_full_call_count() != forward_inits) {
        std::cerr << "context cache missed for repeated forward transform" << std::endl;
        return 1;
    }
    if (batch_calls().size() != 1 || !compare_float_vector(batch_calls().front().pcm, in_real)) {
        std::cerr << "unexpected logging for repeated forward transform" << std::endl;
        return 1;
    }

    clear_call_history();
    if (amp_fft_backend_transform_many(in_real.data(), in_imag.data(), out_real.data(), out_imag.data(), n, batch, 0) != 1) {
        std::cerr << "complex forward transform failed" << std::endl;
        return 1;
    }
    if (batch_calls().size() != 2) {
        std::cerr << "expected two fft_execute_batched calls for complex input" << std::endl;
        return 1;
    }
    if (!compare_float_vector(batch_calls()[0].pcm, in_real) || !compare_float_vector(batch_calls()[1].pcm, in_imag)) {
        std::cerr << "complex input buffers not forwarded correctly" << std::endl;
        return 1;
    }
    for (std::size_t i = 0; i < total; ++i) {
        if (!nearly_equal(out_real[i], in_real[i]) || !nearly_equal(out_imag[i], in_imag[i])) {
            std::cerr << "complex output mismatch at index " << i << std::endl;
            return 1;
        }
    }

    clear_call_history();
    std::vector<double> freq_real{0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0};
    std::vector<double> freq_imag{0.75, -0.25, 0.5, -1.5, 1.25, -0.75, 0.0, 0.5};
    if (amp_fft_backend_transform_many(freq_real.data(), freq_imag.data(), out_real.data(), out_imag.data(), n, batch, 1) != 1) {
        std::cerr << "inverse transform failed" << std::endl;
        return 1;
    }

    if (complex_calls().size() != 1) {
        std::cerr << "expected single fft_execute_complex_batched call for inverse transform" << std::endl;
        return 1;
    }
    const auto inverse_call = complex_calls().front();
    if (inverse_call.frames != static_cast<std::size_t>(batch) || inverse_call.real.size() != total) {
        std::cerr << "unexpected frame/sample counts for inverse call" << std::endl;
        return 1;
    }
    if (!compare_float_vector(inverse_call.real, freq_real) || !compare_float_vector(inverse_call.imag, freq_imag)) {
        std::cerr << "inverse input buffers not forwarded correctly" << std::endl;
        return 1;
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!nearly_equal(out_real[i], freq_real[i]) || !nearly_equal(out_imag[i], 0.0)) {
            std::cerr << "inverse output mismatch at index " << i << std::endl;
            return 1;
        }
    }

    if (init_full_call_count() != forward_inits + 1) {
        std::cerr << "expected separate cached context for inverse transform" << std::endl;
        return 1;
    }

    return 0;
}
