#include "amp_fft_backend.h"

#include "fftfree/fft_cffi.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef AMP_NATIVE_USE_FFTFREE
#error "AMP_NATIVE_USE_FFTFREE must be defined; fftfree backend is mandatory"
#endif

namespace {

struct fft_handle_deleter {
    void operator()(void *handle) const noexcept {
        if (handle != nullptr) {
            fft_free(handle);
        }
    }
};

struct fft_context {
    int n{0};
    bool inverse{false};
    std::unique_ptr<void, fft_handle_deleter> handle;
};

using context_map = std::unordered_map<std::uint64_t, fft_context>;

std::uint64_t make_key(int n, bool inverse) {
    return (static_cast<std::uint64_t>(n) << 1) | (inverse ? 1ULL : 0ULL);
}

fft_context *get_context(int n, bool inverse) {
    static std::mutex mutex;
    static context_map contexts;

    if (n <= 0) {
        return nullptr;
    }

    const std::uint64_t key = make_key(n, inverse);
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = contexts.find(key);
        if (it != contexts.end()) {
            return &it->second;
        }
    }

    void *handle = fft_init_full_v2(
        static_cast<std::size_t>(n),
        0,   /* threads: auto */
        1,   /* lanes */
        inverse ? 1 : 0,
        FFT_KERNEL_COOLEYTUKEY,
        0,   /* radix */
        nullptr,
        0,
        2,   /* pad_mode = never */
        n,
        n,
        0,   /* stft_mode */
        FFT_TRANSFORM_C2C,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,   /* silent_crash_reports */
        0,   /* apply_windows */
        0,   /* apply_ola */
        FFT_WINDOW_RECT,
        0.0f,
        0.0f,
        FFT_WINDOW_RECT,
        0.0f,
        0.0f,
        FFT_WINDOW_NORM_NONE,
        FFT_COLA_OFF);

    fft_context created;
    created.n = n;
    created.inverse = inverse;
    created.handle.reset(handle);

    std::lock_guard<std::mutex> lock(mutex);
    auto result = contexts.emplace(key, std::move(created));
    if (!result.first->second.handle) {
        contexts.erase(result.first);
        return nullptr;
    }
    return &result.first->second;
}

void compute_dft_single(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    bool inverse
) {
    if (n <= 0 || out_real == nullptr || out_imag == nullptr) {
        return;
    }
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const double sign = inverse ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (int t = 0; t < n; ++t) {
            const double real = in_real != nullptr ? in_real[t] : 0.0;
            const double imag = in_imag != nullptr ? in_imag[t] : 0.0;
            const double angle = sign * kTwoPi * static_cast<double>(k) * static_cast<double>(t) / static_cast<double>(n);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            sum_real += real * c - imag * s;
            sum_imag += real * s + imag * c;
        }
        if (inverse) {
            sum_real /= static_cast<double>(n);
            sum_imag /= static_cast<double>(n);
        }
        out_real[k] = sum_real;
        out_imag[k] = sum_imag;
    }
}

void fallback_transform_many(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    bool inverse
) {
    const std::size_t frame_len = static_cast<std::size_t>(n);
    for (int b = 0; b < batch; ++b) {
        const std::size_t offset = frame_len * static_cast<std::size_t>(b);
        compute_dft_single(
            in_real ? in_real + offset : nullptr,
            in_imag ? in_imag + offset : nullptr,
            out_real + offset,
            out_imag + offset,
            n,
            inverse);
    }
}

int run_fftfree_many(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    bool inverse
) {
    auto *ctx = get_context(n, inverse);
    if (ctx == nullptr || !ctx->handle) {
        fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
        return 1;
    }

    const std::size_t frame_len = static_cast<std::size_t>(ctx->n);
    const std::size_t frames = static_cast<std::size_t>(batch);
    const std::size_t total = frame_len * frames;

    std::vector<float> buffer_real(total, 0.0f);
    std::vector<float> buffer_imag;
    if (in_imag != nullptr) {
        buffer_imag.resize(total, 0.0f);
    }
    for (std::size_t idx = 0; idx < total; ++idx) {
        buffer_real[idx] = static_cast<float>(in_real ? in_real[idx] : 0.0);
        if (!buffer_imag.empty()) {
            buffer_imag[idx] = static_cast<float>(in_imag[idx]);
        }
    }

    std::vector<float> out_real_f(total, 0.0f);
    std::vector<float> out_imag_f(total, 0.0f);

    const float *imag_in_ptr = buffer_imag.empty() ? nullptr : buffer_imag.data();
    const std::size_t produced = fft_execute_c2c_batched(
        ctx->handle.get(),
        buffer_real.data(),
        imag_in_ptr,
        out_real_f.data(),
        out_imag_f.data(),
        frames);

    if (produced != frames) {
        fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
        return 1;
    }

    for (std::size_t idx = 0; idx < total; ++idx) {
        out_real[idx] = static_cast<double>(out_real_f[idx]);
        if (out_imag != nullptr) {
            out_imag[idx] = static_cast<double>(out_imag_f[idx]);
        }
    }

    return 1;
}

} // namespace

extern "C" {

int amp_fft_backend_transform_many(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    int inverse
) {
    if (n <= 0 || batch <= 0 || out_real == nullptr || out_imag == nullptr || in_real == nullptr) {
        return 0;
    }
    return run_fftfree_many(in_real, in_imag, out_real, out_imag, n, batch, inverse != 0);
}

void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    (void)amp_fft_backend_transform_many(in_real, in_imag, out_real, out_imag, n, 1, inverse);
}

int amp_fft_backend_has_hook(void) {
    return 0;
}

} // extern "C"
