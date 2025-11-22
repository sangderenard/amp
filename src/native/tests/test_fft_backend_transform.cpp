#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#if __has_include("fftfree/fft_cffi.hpp")
#include "fftfree/fft_cffi.hpp"
#elif __has_include(<fft_cffi.hpp>)
#include <fft_cffi.hpp>
#else
#error "fft_cffi.hpp header not found; ensure fftfree is available"
#endif

#define AMP_NATIVE_USE_FFTFREE 1
#include "../fft_backend.cpp"

namespace {

constexpr double kEpsilon = 1e-5;

bool nearly_equal(double a, double b, double eps = kEpsilon) {
    return std::fabs(a - b) <= eps;
}

bool compare_buffers(const std::vector<double> &lhs, const std::vector<double> &rhs, double eps = kEpsilon) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs[i], rhs[i], eps)) {
            return false;
        }
    }
    return true;
}

class FftfreeContext {
public:
    FftfreeContext(int n, bool inverse, int window, int hop, bool apply_windows, int analysis_window, int synthesis_window)
        : n_(n) {
        if (n_ <= 0) {
            return;
        }
        if (window <= 0) {
            window = n_;
        }
        if (hop <= 0) {
            hop = window;
        }
        handle_ = fft_init_full_v2(
            static_cast<std::size_t>(n_),
            0,
            1,
            inverse ? 1 : 0,
            FFT_KERNEL_COOLEYTUKEY,
            0,
            nullptr,
            0,
            2,
            window,
            hop,
            kStftModeBatched,
            FFT_TRANSFORM_C2C,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            apply_windows ? 1 : 0,
            0,
            analysis_window,
            0.0f,
            0.0f,
            synthesis_window,
            0.0f,
            0.0f,
            FFT_WINDOW_NORM_NONE,
            FFT_COLA_OFF);
    }

    FftfreeContext(FftfreeContext &&other) noexcept : handle_(other.handle_), n_(other.n_) {
        other.handle_ = nullptr;
        other.n_ = 0;
    }

    FftfreeContext &operator=(FftfreeContext &&other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            n_ = other.n_;
            other.handle_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    FftfreeContext(const FftfreeContext &) = delete;
    FftfreeContext &operator=(const FftfreeContext &) = delete;

    ~FftfreeContext() {
        reset();
    }

    static FftfreeContext MakeTransform(int n, bool inverse) {
        return FftfreeContext(n, inverse, n, n, false, FFT_WINDOW_RECT, FFT_WINDOW_RECT);
    }

    int n() const { return n_; }
    void *get() const { return handle_; }
    bool valid() const { return handle_ != nullptr; }

private:
    void reset() {
        if (handle_ != nullptr) {
            fft_free(handle_);
            handle_ = nullptr;
        }
        n_ = 0;
    }

    void *handle_{nullptr};
    int n_{0};
};

struct ComplexFrames {
    std::vector<double> real;
    std::vector<double> imag;
};

ComplexFrames run_fftfree_forward(FftfreeContext &ctx, const std::vector<double> &real, const std::vector<double> *imag) {
    if (!ctx.valid()) {
        throw std::runtime_error("fftfree forward context not initialised");
    }
    const int n = ctx.n();
    const std::size_t frame_len = static_cast<std::size_t>(n);
    if (real.size() % frame_len != 0) {
        throw std::runtime_error("real buffer is not aligned to frame length");
    }
    const std::size_t frames = real.size() / frame_len;
    const std::size_t total = real.size();

    std::vector<float> pcm(total, 0.0f);
    std::vector<float> spec_real(total, 0.0f);
    std::vector<float> spec_imag(total, 0.0f);
    std::vector<float> spec_mag(total, 0.0f);

    auto run_single = [&](const std::vector<double> &src, std::vector<double> &dst_real, std::vector<double> &dst_imag) {
        std::copy(src.begin(), src.end(), pcm.begin());
        const std::size_t produced = fft_execute_batched(
            ctx.get(),
            pcm.data(),
            total,
            spec_real.data(),
            spec_imag.data(),
            spec_mag.data(),
            2,
            0,
            frames);
        if (produced != frames) {
            throw std::runtime_error("fft_execute_batched did not produce frames");
        }
        dst_real.resize(total, 0.0);
        dst_imag.resize(total, 0.0);
        for (std::size_t idx = 0; idx < total; ++idx) {
            dst_real[idx] = static_cast<double>(spec_real[idx]);
            dst_imag[idx] = static_cast<double>(spec_imag[idx]);
        }
    };

    ComplexFrames result;
    run_single(real, result.real, result.imag);
    if (imag != nullptr) {
        std::vector<double> imag_real;
        std::vector<double> imag_imag;
        run_single(*imag, imag_real, imag_imag);
        for (std::size_t idx = 0; idx < total; ++idx) {
            const double a_real = result.real[idx];
            const double a_imag = result.imag[idx];
            const double b_real = imag_real[idx];
            const double b_imag = imag_imag[idx];
            result.real[idx] = a_real - b_imag;
            result.imag[idx] = a_imag + b_real;
        }
    }
    return result;
}

std::vector<double> run_fftfree_inverse(FftfreeContext &ctx, const std::vector<double> &real, const std::vector<double> &imag) {
    if (!ctx.valid()) {
        throw std::runtime_error("fftfree inverse context not initialised");
    }
    if (real.size() != imag.size()) {
        throw std::runtime_error("inverse buffers differ in size");
    }
    const int n = ctx.n();
    const std::size_t frame_len = static_cast<std::size_t>(n);
    if (real.size() % frame_len != 0) {
        throw std::runtime_error("complex buffer is not aligned to frame length");
    }
    const std::size_t frames = real.size() / frame_len;

    std::vector<float> input_real(real.size(), 0.0f);
    std::vector<float> input_imag(imag.size(), 0.0f);
    for (std::size_t idx = 0; idx < real.size(); ++idx) {
        input_real[idx] = static_cast<float>(real[idx]);
        input_imag[idx] = static_cast<float>(imag[idx]);
    }

    std::vector<float> pcm_out(real.size(), 0.0f);
    const std::size_t produced = fft_execute_complex_batched(
        ctx.get(),
        input_real.data(),
        input_imag.data(),
        frames,
        pcm_out.data(),
        2,
        0,
        frames);
    if (produced != frames) {
        throw std::runtime_error("fft_execute_complex_batched did not produce frames");
    }

    std::vector<double> result(real.size(), 0.0);
    for (std::size_t idx = 0; idx < real.size(); ++idx) {
        result[idx] = static_cast<double>(pcm_out[idx]);
    }
    return result;
}

}  // namespace

int main() {
    const int n = 8;
    const int batch = 3;
    const std::size_t total = static_cast<std::size_t>(n * batch);

    std::vector<double> in_real{
        0.5, -1.0, 1.5, -2.0, 0.25, -0.25, 0.75, -0.75,
        1.25, -1.25, 2.25, -2.25, 0.5, -0.5, 1.0, -1.0,
        0.33, -0.44, 0.55, -0.66, 0.77, -0.88, 0.99, -1.1};
    std::vector<double> in_imag{
        -0.25, 0.25, -0.5, 0.5, -0.75, 0.1, -0.2, 0.3,
        0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, -1.1,
        -0.15, 0.2, -0.25, 0.3, -0.35, 0.4, -0.45, 0.5};
    std::vector<double> out_real(total, 0.0);
    std::vector<double> out_imag(total, 0.0);

    auto forward_ctx = FftfreeContext::MakeTransform(n, false);
    if (!forward_ctx.valid()) {
        std::cerr << "failed to create fftfree forward context" << std::endl;
        return 1;
    }

    if (amp_fft_backend_transform_many(in_real.data(), nullptr, out_real.data(), out_imag.data(), n, batch, 0) != 1) {
        std::cerr << "forward transform failed" << std::endl;
        return 1;
    }
    try {
        auto expected_forward_real = run_fftfree_forward(forward_ctx, in_real, nullptr);
        if (!compare_buffers(out_real, expected_forward_real.real) || !compare_buffers(out_imag, expected_forward_real.imag)) {
            std::cerr << "amp forward transform does not match fftfree output (real input)" << std::endl;
            return 1;
        }
    } catch (const std::exception &ex) {
        std::cerr << "fftfree forward reference failed: " << ex.what() << std::endl;
        return 1;
    }

    if (amp_fft_backend_transform_many(in_real.data(), in_imag.data(), out_real.data(), out_imag.data(), n, batch, 0) != 1) {
        std::cerr << "complex forward transform failed" << std::endl;
        return 1;
    }
    ComplexFrames expected_complex;
    try {
        expected_complex = run_fftfree_forward(forward_ctx, in_real, &in_imag);
    } catch (const std::exception &ex) {
        std::cerr << "fftfree complex forward reference failed: " << ex.what() << std::endl;
        return 1;
    }
    if (!compare_buffers(out_real, expected_complex.real) || !compare_buffers(out_imag, expected_complex.imag)) {
        std::cerr << "complex forward output mismatch vs fftfree" << std::endl;
        return 1;
    }

    auto inverse_ctx = FftfreeContext::MakeTransform(n, true);
    if (!inverse_ctx.valid()) {
        std::cerr << "failed to create fftfree inverse context" << std::endl;
        return 1;
    }

    if (amp_fft_backend_transform_many(expected_complex.real.data(), expected_complex.imag.data(), out_real.data(), out_imag.data(), n, batch, 1) != 1) {
        std::cerr << "inverse transform failed" << std::endl;
        return 1;
    }

    std::vector<double> expected_inverse;
    try {
        expected_inverse = run_fftfree_inverse(inverse_ctx, expected_complex.real, expected_complex.imag);
    } catch (const std::exception &ex) {
        std::cerr << "fftfree inverse reference failed: " << ex.what() << std::endl;
        return 1;
    }

    if (!compare_buffers(out_real, expected_inverse)) {
        std::cerr << "inverse PCM mismatch vs fftfree" << std::endl;
        return 1;
    }
    for (double value : out_imag) {
        if (!nearly_equal(value, 0.0, 1e-7)) {
            std::cerr << "inverse output contains non-zero imaginary component" << std::endl;
            return 1;
        }
    }

    return 0;
}
