#ifndef AMP_FFT_BACKEND_H
#define AMP_FFT_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
);

int amp_fft_backend_transform_many(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int batch,
    int inverse
);

int amp_fft_backend_has_hook(void);

#ifdef __cplusplus
}
#endif

#endif /* AMP_FFT_BACKEND_H */
