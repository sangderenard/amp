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

    if (!inverse) {
        if (in_real == nullptr) {
            fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
            return 1;
        }

        std::vector<float> pcm(total, 0.0f);
        std::vector<float> spec_real(total, 0.0f);
        std::vector<float> spec_imag(total, 0.0f);
        std::vector<float> spec_mag(total, 0.0f);

        auto run_forward = [&](const double *src) -> bool {
            if (src == nullptr) {
                return false;
            }
            for (std::size_t i = 0; i < total; ++i) {
                pcm[i] = static_cast<float>(src[i]);
            }
            const std::size_t produced = fft_execute_batched(
                ctx->handle.get(),
                pcm.data(),
                total,
                spec_real.data(),
                spec_imag.data(),
                spec_mag.data(),
                2,   /* pad_mode = never */
                0,   /* enable_backup */
                frames);
            return produced == frames;
        };

        if (!run_forward(in_real)) {
            fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
            return 1;
        }

        for (std::size_t idx = 0; idx < total; ++idx) {
            out_real[idx] = static_cast<double>(spec_real[idx]);
            out_imag[idx] = static_cast<double>(spec_imag[idx]);
        }

        if (in_imag != nullptr) {
            if (!run_forward(in_imag)) {
                fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
                return 1;
            }
            for (std::size_t idx = 0; idx < total; ++idx) {
                const double a_real = out_real[idx];
                const double a_imag = out_imag[idx];
                const double b_real = static_cast<double>(spec_real[idx]);
                const double b_imag = static_cast<double>(spec_imag[idx]);
                out_real[idx] = a_real - b_imag;
                out_imag[idx] = a_imag + b_real;
            }
        }

        return 1;
    }

    std::vector<float> input_real(total, 0.0f);
    std::vector<float> input_imag(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        input_real[idx] = static_cast<float>(in_real ? in_real[idx] : 0.0);
        input_imag[idx] = static_cast<float>(in_imag ? in_imag[idx] : 0.0);
    }

    std::vector<float> pcm_out(total, 0.0f);
    const std::size_t produced = fft_execute_complex_batched(
        ctx->handle.get(),
        input_real.data(),
        input_imag.data(),
        frames,
        pcm_out.data(),
        2,   /* pad_mode = never */
        0,   /* enable_backup */
        frames);

    if (produced != frames) {
        fallback_transform_many(in_real, in_imag, out_real, out_imag, n, batch, inverse);
        return 1;
    }

    for (std::size_t idx = 0; idx < total; ++idx) {
        out_real[idx] = static_cast<double>(pcm_out[idx]);
        out_imag[idx] = 0.0;
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
