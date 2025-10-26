#ifndef AMP_FFT_BACKEND_H
#define AMP_FFT_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * FFT backend contract for native nodes.
 *
 * - Transform hooks receive contiguous real/imag arrays of length `n`.
 * - Hook implementations may reuse the provided output buffers, but must
 *   populate them fully before returning.
 * - Inputs may alias outputs; hooks must account for in-place execution.
 * - The `inverse` flag is 0 for forward transforms and non-zero for inverse.
 */
typedef void (*amp_fft_transform_hook)(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse,
    void *user_data
);

void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
);

void amp_fft_backend_register_hook(amp_fft_transform_hook hook, void *user_data);
void amp_fft_backend_clear_hook(void);
int amp_fft_backend_has_hook(void);

#ifdef __cplusplus
}
#endif

#endif /* AMP_FFT_BACKEND_H */
