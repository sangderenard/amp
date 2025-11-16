#include "amp_fft_backend.h"

#if __has_include("fftfree/fft_cffi.hpp")
#include "fftfree/fft_cffi.hpp"
#elif __has_include(<fft_cffi.hpp>)
#include <fft_cffi.hpp>
#else
#error "fft_cffi.hpp header not found; ensure fftfree is available"
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

#ifndef AMP_NATIVE_USE_FFTFREE
#error "AMP_NATIVE_USE_FFTFREE must be defined; fftfree backend is mandatory"
#endif

namespace {

constexpr int kStftModeLegacy = 0;
constexpr int kStftModeBatched = 1;
constexpr int kStftModeStreaming = 2;

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
        kStftModeLegacy,
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
            return produced != 0;
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

    if (produced == 0) {
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

AMP_CAPI int amp_fft_backend_transform_many(
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

AMP_CAPI int amp_fft_backend_transform_many_ex(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    int inverse,
    int window,
    int hop,
    int apply_windows,
    int analysis_window_kind,
    int synthesis_window_kind
) {
    if (n <= 0 || batch <= 0 || out_real == nullptr || out_imag == nullptr || in_real == nullptr) {
        return 0;
    }

    if (window <= 0) {
        window = n;
    }
    if (hop <= 0) {
        hop = window;
    }
    if (analysis_window_kind < 0) {
        analysis_window_kind = FFT_WINDOW_RECT;
    }
    if (synthesis_window_kind < 0) {
        synthesis_window_kind = analysis_window_kind;
    }

    // Create a temporary fftfree context configured with STFT/window params.
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
        1,   /* silent_crash_reports */
        apply_windows ? 1 : 0,
        0,   /* apply_ola */
        analysis_window_kind,
        0.0f,
        0.0f,
        synthesis_window_kind,
        0.0f,
        0.0f,
        FFT_WINDOW_NORM_NONE,
        FFT_COLA_OFF);

    if (handle == nullptr) {
        // Fallback to safe implementation
        return run_fftfree_many(in_real, in_imag, out_real, out_imag, n, batch, inverse != 0);
    }

    const std::size_t frame_len = static_cast<std::size_t>(n);
    const std::size_t frames = static_cast<std::size_t>(batch);
    const std::size_t total = frame_len * frames;

    if (!inverse) {
        // Forward: caller supplies PCM-like concatenation of frames.
        std::vector<float> pcm(total, 0.0f);
        std::vector<float> spec_real(total, 0.0f);
        std::vector<float> spec_imag(total, 0.0f);
        std::vector<float> spec_mag(total, 0.0f);
        for (std::size_t i = 0; i < total; ++i) {
            pcm[i] = static_cast<float>(in_real[i]);
        }
        const std::size_t produced = fft_execute_batched(
            handle,
            pcm.data(),
            total,
            spec_real.data(),
            spec_imag.data(),
            spec_mag.data(),
            2,   /* pad_mode = never */
            0,   /* enable_backup */
            frames);
        if (produced == 0) {
            fft_free(handle);
            return run_fftfree_many(in_real, in_imag, out_real, out_imag, n, batch, inverse != 0);
        }
        for (std::size_t idx = 0; idx < total; ++idx) {
            out_real[idx] = static_cast<double>(spec_real[idx]);
            out_imag[idx] = static_cast<double>(spec_imag[idx]);
        }
        if (in_imag != nullptr) {
            // If a separate imaginary input is provided, run it too and combine
            std::vector<float> pcm_b(total, 0.0f);
            for (std::size_t i = 0; i < total; ++i) {
                pcm_b[i] = static_cast<float>(in_imag[i]);
            }
            const std::size_t produced_b = fft_execute_batched(
                handle,
                pcm_b.data(),
                total,
                spec_real.data(),
                spec_imag.data(),
                spec_mag.data(),
                2,
                0,
                frames);
            if (produced_b == 0) {
                fft_free(handle);
                return run_fftfree_many(in_real, in_imag, out_real, out_imag, n, batch, inverse != 0);
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
        fft_free(handle);
        return 1;
    }

    // Inverse: accept complex frames and produce PCM
    std::vector<float> input_real(total, 0.0f);
    std::vector<float> input_imag(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        input_real[idx] = static_cast<float>(in_real ? in_real[idx] : 0.0);
        input_imag[idx] = static_cast<float>(in_imag ? in_imag[idx] : 0.0);
    }

    std::vector<float> pcm_out(total, 0.0f);
    const std::size_t produced = fft_execute_complex_batched(
        handle,
        input_real.data(),
        input_imag.data(),
        frames,
        pcm_out.data(),
        2,   /* pad_mode = never */
        0,   /* enable_backup */
        frames);

    if (produced == 0) {
        fft_free(handle);
        return run_fftfree_many(in_real, in_imag, out_real, out_imag, n, batch, inverse != 0);
    }

    for (std::size_t idx = 0; idx < total; ++idx) {
        out_real[idx] = static_cast<double>(pcm_out[idx]);
        out_imag[idx] = 0.0;
    }
    fft_free(handle);
    return 1;
}

AMP_CAPI void *amp_fft_backend_stream_create(
    int n,
    int window,
    int hop,
    int analysis_window_kind
) {
    if (n <= 0) {
        return nullptr;
    }
    if (window <= 0) {
        window = n;
    }
    if (hop <= 0) {
        hop = 1;
    }
    if (analysis_window_kind < 0) {
        analysis_window_kind = FFT_WINDOW_RECT;
    }
    return fft_init_full_v2(
        static_cast<std::size_t>(n),
        0,
        1,
        0,
        FFT_KERNEL_COOLEYTUKEY,
        0,
        nullptr,
        0,
        2,
        window,
        hop,
        kStftModeStreaming,
        FFT_TRANSFORM_C2C,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1, /* apply_windows */
        0, /* apply_ola */
        analysis_window_kind,
        0.0f,
        0.0f,
        analysis_window_kind,
        0.0f,
        0.0f,
        FFT_WINDOW_NORM_NONE,
        FFT_COLA_OFF);
}

AMP_CAPI void amp_fft_backend_stream_destroy(void *handle) {
    if (handle != nullptr) {
        fft_free(handle);
    }
}

AMP_CAPI size_t amp_fft_backend_stream_push(
    void *handle,
    const double *pcm,
    size_t samples,
    int n,
    double *out_real,
    double *out_imag,
    size_t max_frames,
    int flush_mode
) {
    if (handle == nullptr || pcm == nullptr || samples == 0 || n <= 0) {
        return 0;
    }
    std::vector<float> pcm_f(samples, 0.0f);
    for (size_t i = 0; i < samples; ++i) {
        pcm_f[i] = static_cast<float>(pcm[i]);
    }
    std::vector<float> out_real_f;
    std::vector<float> out_imag_f;
    std::vector<float> out_mag_f;
    if (out_real != nullptr || out_imag != nullptr) {
        const size_t capacity = max_frames * static_cast<size_t>(n);
        out_real_f.resize(capacity, 0.0f);
        out_imag_f.resize(capacity, 0.0f);
        out_mag_f.resize(capacity, 0.0f);
    }
    const size_t frames = fft_stream_push_pcm(
        handle,
        pcm_f.data(),
        samples,
        out_real_f.empty() ? nullptr : out_real_f.data(),
        out_imag_f.empty() ? nullptr : out_imag_f.data(),
        out_mag_f.empty() ? nullptr : out_mag_f.data(),
        max_frames,
        flush_mode);
    if (frames == 0 || out_real_f.empty()) {
        return frames;
    }
    const size_t frame_len = static_cast<size_t>(n);
    const size_t copy_len = frames * frame_len;
    for (size_t i = 0; i < copy_len; ++i) {
        if (out_real != nullptr) {
            out_real[i] = static_cast<double>(out_real_f[i]);
        }
        if (out_imag != nullptr) {
            out_imag[i] = static_cast<double>(out_imag_f[i]);
        }
    }
    return frames;
}

AMP_CAPI void *amp_fft_backend_stream_create_inverse(
    int n,
    int window,
    int hop,
    int synthesis_window_kind
) {
    if (n <= 0) {
        return nullptr;
    }
    if (window <= 0) {
        window = n;
    }
    if (hop <= 0) {
        hop = 1;
    }
    if (synthesis_window_kind < 0) {
        synthesis_window_kind = FFT_WINDOW_RECT;
    }
    return fft_init_full_v2(
        static_cast<std::size_t>(n),
        0,
        1,
        1,
        FFT_KERNEL_COOLEYTUKEY,
        0,
        nullptr,
        0,
        2,
        window,
        hop,
        kStftModeStreaming,
        FFT_TRANSFORM_C2C,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,  // apply_ola = 0: handle OLA manually in streaming code
        synthesis_window_kind,
        0.0f,
        0.0f,
        synthesis_window_kind,
        0.0f,
        0.0f,
        FFT_WINDOW_NORM_NONE,
        FFT_COLA_NORMALIZE);
}

AMP_CAPI size_t amp_fft_backend_stream_push_spectrum(
    void *handle,
    const double *in_real,
    const double *in_imag,
    size_t frames,
    int n,
    double *out_pcm,
    size_t max_samples,
    int flush_mode
) {
    if (handle == nullptr) {
        return 0;
    }
    if (frames > 0 && (in_real == nullptr || n <= 0)) {
        return 0;
    }
    if (n <= 0) {
        return 0;
    }

    std::vector<float> real_f;
    std::vector<float> imag_f;
    if (frames > 0) {
        const size_t total = frames * static_cast<size_t>(n);
        real_f.resize(total, 0.0f);
        imag_f.resize(total, 0.0f);
        for (size_t i = 0; i < total; ++i) {
            real_f[i] = static_cast<float>(in_real[i]);
            if (in_imag != nullptr) {
                imag_f[i] = static_cast<float>(in_imag[i]);
            }
        }
    }

    std::vector<float> pcm_f;
    float *pcm_ptr = nullptr;
    if (out_pcm != nullptr && max_samples > 0) {
        pcm_f.resize(max_samples, 0.0f);
        pcm_ptr = pcm_f.data();
    }

    const size_t produced = fft_stream_push_spectrum(
        handle,
        frames > 0 ? real_f.data() : nullptr,
        frames > 0 && !imag_f.empty() ? imag_f.data() : nullptr,
        frames,
        pcm_ptr,
        max_samples,
        flush_mode);

    if (pcm_ptr != nullptr && produced > 0) {
        const size_t copy = std::min(produced, max_samples);
        for (size_t i = 0; i < copy; ++i) {
            out_pcm[i] = static_cast<double>(pcm_f[i]);
        }
    }

    return produced;
}

AMP_CAPI size_t amp_fft_backend_stream_pending_pcm(void *handle) {
    if (handle == nullptr) {
        return 0;
    }
    return fft_stream_pending_pcm(handle);
}

AMP_CAPI void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    (void)amp_fft_backend_transform_many(in_real, in_imag, out_real, out_imag, n, 1, inverse);
}

AMP_CAPI int amp_fft_backend_has_hook(void) {
    return 0;
}

} // extern "C"
