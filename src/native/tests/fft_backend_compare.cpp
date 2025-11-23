// Simple diagnostic: compare Eigen FFT vs amp_fft_backend_transform (fftfree)
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <unsupported/Eigen/FFT>

#if __has_include("fftfree/fft_cffi.hpp")
#include "fftfree/fft_cffi.hpp"
#elif __has_include(<fft_cffi.hpp>)
#include <fft_cffi.hpp>
#else
#error "fft_cffi.hpp header not found; ensure fftfree is available"
#endif
extern "C" {
#include "amp_fft_backend.h"
}

static void compute_fft_eigen(std::vector<double> &real, std::vector<double> &imag, int inverse) {
    const int n = static_cast<int>(real.size());
    using ComplexVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    ComplexVector input(n);
    for (int i = 0; i < n; ++i) input[i] = std::complex<double>(real[i], imag[i]);
    ComplexVector output(n);
    Eigen::FFT<double> fft;
    if (inverse) fft.inv(output, input);
    else fft.fwd(output, input);
    for (int i = 0; i < n; ++i) {
        real[i] = output[i].real();
        imag[i] = output[i].imag();
    }
}

static void print_vec(const char *label, const std::vector<double> &v) {
    std::cout << label << ": ";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "\n";
}

int main() {
    const int window_size = 8;
    std::vector<double> signal(window_size);
    std::vector<double> divisor(window_size, 1.5);
    // fill a simple test signal
    for (int i = 0; i < window_size; ++i) signal[i] = pow(-0.5, i);

    // prepare eigen copies
    std::vector<double> r1 = signal;
    std::vector<double> i1(window_size, 0.0);
    std::vector<double> r2 = divisor;
    std::vector<double> i2(window_size, 0.0);

    compute_fft_eigen(r1, i1, 0);
    compute_fft_eigen(r2, i2, 0);

    print_vec("Eigen signal real", r1);
    print_vec("Eigen signal imag", i1);
    print_vec("Eigen divisor real", r2);
    print_vec("Eigen divisor imag", i2);

    // call backend (fftfree) via amp_fft_backend_transform_many
    std::vector<double> out_sig_r(window_size, 0.0);
    std::vector<double> out_sig_i(window_size, 0.0);
    std::vector<double> out_div_r(window_size, 0.0);
    std::vector<double> out_div_i(window_size, 0.0);

    // single-batch
    amp_fft_backend_transform(signal.data(), nullptr, out_sig_r.data(), out_sig_i.data(), window_size, 0);
    amp_fft_backend_transform(divisor.data(), nullptr, out_div_r.data(), out_div_i.data(), window_size, 0);

    print_vec("Backend signal real", out_sig_r);
    print_vec("Backend signal imag", out_sig_i);
    print_vec("Backend divisor real", out_div_r);
    print_vec("Backend divisor imag", out_div_i);

    // show differences
    std::cout << "Diffs (signal real): ";
    for (int i = 0; i < window_size; ++i) std::cout << fabs(r1[i] - out_sig_r[i]) << (i + 1 < window_size ? ", " : "\n");
    std::cout << "Diffs (signal imag): ";
    for (int i = 0; i < window_size; ++i) std::cout << fabs(i1[i] - out_sig_i[i]) << (i + 1 < window_size ? ", " : "\n");

    // --- Streaming STFT exercise ---------------------------------
    void* stream_ctx = fft_init_ex(
        window_size,   // n
        0,             // threads (auto)
        1,             // lanes
        0,             // inverse
        FFT_KERNEL_COOLEYTUKEY,
        0,             // radix auto
        nullptr,
        0,
        2,             // pad_mode = never
        window_size,
        window_size);

    if (!stream_ctx) {
        std::cerr << "Failed to initialize streaming FFT context" << std::endl;
        return 1;
    }

    fft_stream_reset(stream_ctx);

    std::vector<float> signal_f(signal.size(), 0.0f);
    for (size_t idx = 0; idx < signal.size(); ++idx) {
        signal_f[idx] = static_cast<float>(signal[idx]);
    }

    const size_t half = static_cast<size_t>(window_size / 2);
    size_t frames_written = fft_stream_push_pcm(
        stream_ctx,
        signal_f.data(),
        half,
        nullptr,
        nullptr,
        nullptr,
        0,
        FFT_STREAM_FLUSH_NONE);
    std::cout << "Streaming frames after first chunk: " << frames_written
              << ", pending=" << fft_stream_pending_frames(stream_ctx)
              << ", backlog=" << fft_stream_backlog_samples(stream_ctx) << "\n";

    std::vector<float> stream_real(window_size, 0.0f);
    std::vector<float> stream_imag(window_size, 0.0f);
    std::vector<float> stream_mag(window_size, 0.0f);

    frames_written = fft_stream_push_pcm(
        stream_ctx,
        signal_f.data() + half,
        signal_f.size() - half,
        stream_real.data(),
        stream_imag.data(),
        stream_mag.data(),
        1,
        FFT_STREAM_FLUSH_NONE);

    std::cout << "Streaming frames after second chunk: " << frames_written
              << ", pending=" << fft_stream_pending_frames(stream_ctx)
              << ", backlog=" << fft_stream_backlog_samples(stream_ctx) << "\n";

    std::vector<double> stream_r(stream_real.begin(), stream_real.end());
    std::vector<double> stream_i(stream_imag.begin(), stream_imag.end());
    print_vec("Streaming signal real", stream_r);
    print_vec("Streaming signal imag", stream_i);

    std::cout << "Diffs (stream vs backend real): ";
    for (int i = 0; i < window_size; ++i) {
        std::cout << fabs(out_sig_r[i] - static_cast<double>(stream_real[i]))
                  << (i + 1 < window_size ? ", " : "\n");
    }
    std::cout << "Diffs (stream vs backend imag): ";
    for (int i = 0; i < window_size; ++i) {
        std::cout << fabs(out_sig_i[i] - static_cast<double>(stream_imag[i]))
                  << (i + 1 < window_size ? ", " : "\n");
    }

    fft_free(stream_ctx);

    return 0;
}
