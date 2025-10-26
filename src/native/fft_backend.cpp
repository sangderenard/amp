#include "amp_fft_backend.h"

#include <complex>
#include <mutex>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace {
struct amp_fft_hook_state {
    amp_fft_transform_hook fn{nullptr};
    void *user{nullptr};
};

amp_fft_hook_state g_hook_state;
std::mutex g_hook_mutex;

void eigen_fft_execute(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    if (n <= 0 || out_real == nullptr || out_imag == nullptr) {
        return;
    }
    using ComplexVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    ComplexVector input(n);
    for (int i = 0; i < n; ++i) {
        double real_value = (in_real != nullptr) ? in_real[i] : 0.0;
        double imag_value = (in_imag != nullptr) ? in_imag[i] : 0.0;
        input[i] = std::complex<double>(real_value, imag_value);
    }
    ComplexVector output(n);
    Eigen::FFT<double> fft;
    if (inverse != 0) {
        fft.inv(output, input);
    } else {
        fft.fwd(output, input);
    }
    for (int i = 0; i < n; ++i) {
        out_real[i] = output[i].real();
        out_imag[i] = output[i].imag();
    }
}

} // namespace

extern "C" {

void amp_fft_backend_transform(
    const double *in_real,
    const double *in_imag,
    double *out_real,
    double *out_imag,
    int n,
    int inverse
) {
    amp_fft_hook_state hook_snapshot;
    {
        std::lock_guard<std::mutex> guard(g_hook_mutex);
        hook_snapshot = g_hook_state;
    }
    if (hook_snapshot.fn != nullptr) {
        hook_snapshot.fn(in_real, in_imag, out_real, out_imag, n, inverse, hook_snapshot.user);
        return;
    }
    eigen_fft_execute(in_real, in_imag, out_real, out_imag, n, inverse);
}

void amp_fft_backend_register_hook(amp_fft_transform_hook hook, void *user_data) {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    g_hook_state.fn = hook;
    g_hook_state.user = user_data;
}

void amp_fft_backend_clear_hook(void) {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    g_hook_state.fn = nullptr;
    g_hook_state.user = nullptr;
}

int amp_fft_backend_has_hook(void) {
    std::lock_guard<std::mutex> guard(g_hook_mutex);
    return g_hook_state.fn != nullptr;
}

} // extern "C"
