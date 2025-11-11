#ifndef AMP_FFT_BACKEND_H
#define AMP_FFT_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "amp_native.h"

AMP_CAPI void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
);

AMP_CAPI int amp_fft_backend_transform_many(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    int inverse
);

/* Extended variant: pass STFT framing/window instructions. This is optional
   and callers may use the legacy amp_fft_backend_transform_many if they don't
   need STFT framing. When provided, `window` and `hop` describe the analysis
   window and hop parameters to be used by the backend's batched helpers; if
   `stft_mode` is non-zero the backend will initialize an STFT-capable
   context. Returns 1 on success, 0 on error. */
/* Window kind constants mirror fftfree's FFT_WINDOW_* values. */
#define AMP_FFT_WINDOW_RECT 0
#define AMP_FFT_WINDOW_HANN 1
#define AMP_FFT_WINDOW_HAMMING 2
#define AMP_FFT_WINDOW_BLACKMAN 3
#define AMP_FFT_WINDOW_TUKEY 4
#define AMP_FFT_WINDOW_KAISER 5

#define AMP_FFT_STREAM_FLUSH_NONE 0
#define AMP_FFT_STREAM_FLUSH_PARTIAL 1
#define AMP_FFT_STREAM_FLUSH_FINAL 2

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
    int stft_mode,
    int apply_windows,
    int analysis_window_kind,
    int synthesis_window_kind
);

AMP_CAPI void *amp_fft_backend_stream_create(
    int n,
    int window,
    int hop,
    int analysis_window_kind
);

AMP_CAPI void amp_fft_backend_stream_destroy(void *handle);

AMP_CAPI size_t amp_fft_backend_stream_push(
    void *handle,
    const double *pcm,
    size_t samples,
    int n,
    double *out_real,
    double *out_imag,
    size_t max_frames,
    int flush_mode
);

AMP_CAPI void *amp_fft_backend_stream_create_inverse(
    int n,
    int window,
    int hop,
    int synthesis_window_kind
);

AMP_CAPI size_t amp_fft_backend_stream_push_spectrum(
    void *handle,
    const double *in_real,
    const double *in_imag,
    size_t frames,
    int n,
    double *out_pcm,
    size_t max_samples,
    int flush_mode
);

AMP_CAPI size_t amp_fft_backend_stream_pending_pcm(void *handle);

AMP_CAPI int amp_fft_backend_has_hook(void);

#ifdef __cplusplus
}
#endif

#endif /* AMP_FFT_BACKEND_H */
