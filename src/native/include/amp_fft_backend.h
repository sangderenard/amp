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
    int stft_mode
);

AMP_CAPI int amp_fft_backend_has_hook(void);

#ifdef __cplusplus
}
#endif

#endif /* AMP_FFT_BACKEND_H */
