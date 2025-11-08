#ifndef MOCK_FFTFREE_FFT_CFFI_HPP
#define MOCK_FFTFREE_FFT_CFFI_HPP

#include <cstddef>
#include <cstdint>

extern "C" {

enum {
    FFT_KERNEL_COOLEYTUKEY = 0,
};

enum {
    FFT_TRANSFORM_C2C = 0,
};

enum {
    FFT_WINDOW_RECT = 0,
};

enum {
    FFT_WINDOW_NORM_NONE = 0,
};

enum {
    FFT_COLA_OFF = 0,
};

enum {
    FFT_STREAM_FLUSH_NONE = 0,
    FFT_STREAM_FLUSH_PARTIAL = 1,
    FFT_STREAM_FLUSH_FINAL = 2,
};

void *fft_init_full_v2(
    std::size_t n,
    int threads,
    int lanes,
    int inverse,
    int kernel,
    int radix,
    const void *plan,
    std::size_t plan_bytes,
    int pad_mode,
    int window,
    int hop,
    int stft_mode,
    int transform,
    int phase_mode,
    int reorder,
    int real_policy,
    int freq_policy,
    int time_policy,
    int reserved0,
    int reserved1,
    int silent_crash_reports,
    int apply_windows,
    int apply_ola,
    int analysis_window,
    float analysis_alpha,
    float analysis_beta,
    int synthesis_window,
    float synthesis_alpha,
    float synthesis_beta,
    int window_norm,
    int cola_mode
);

void fft_free(void *handle);

std::size_t fft_execute_batched(
    void *handle,
    const float *pcm,
    std::size_t samples,
    float *spec_real,
    float *spec_imag,
    float *spec_mag,
    int pad_mode,
    int enable_backup,
    std::size_t frames
);

std::size_t fft_execute_complex_batched(
    void *handle,
    const float *input_real,
    const float *input_imag,
    std::size_t frames,
    float *pcm_out,
    int pad_mode,
    int enable_backup,
    std::size_t lanes
);

void *fft_init_ex(
    int n,
    int threads,
    int lanes,
    int inverse,
    int kernel,
    int radix,
    const void *plan,
    std::size_t plan_bytes,
    int pad_mode,
    int window,
    int hop,
    int stft_mode
);

void fft_stream_reset(void *handle);

std::size_t fft_stream_push_pcm(
    void *handle,
    const float *pcm,
    std::size_t samples,
    float *out_real,
    float *out_imag,
    float *out_mag,
    std::size_t max_frames,
    int flush_mode
);

std::size_t fft_stream_pending_frames(void *handle);

std::size_t fft_stream_backlog_samples(void *handle);

}

#endif  // MOCK_FFTFREE_FFT_CFFI_HPP
