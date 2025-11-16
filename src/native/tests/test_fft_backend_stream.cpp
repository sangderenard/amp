#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#if __has_include("fftfree/fft_cffi.hpp")
#include "fftfree/fft_cffi.hpp"
#elif __has_include(<fft_cffi.hpp>)
#include <fft_cffi.hpp>
#else
#error "fft_cffi.hpp header not found; ensure fftfree is available"
#endif

#define AMP_NATIVE_USE_FFTFREE 1
#include "../fft_backend.cpp"

namespace {

constexpr double kEpsilon = 1e-5;

bool nearly_equal(double a, double b, double eps = kEpsilon) {
    return std::fabs(a - b) <= eps;
}

bool compare_buffers(const std::vector<double> &lhs, const std::vector<double> &rhs, double eps = kEpsilon) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t idx = 0; idx < lhs.size(); ++idx) {
        if (!nearly_equal(lhs[idx], rhs[idx], eps)) {
            return false;
        }
    }
    return true;
}

class StreamingReference {
public:
    StreamingReference(int n, int window, int hop, int window_kind)
        : n_(n) {
        if (n_ <= 0) {
            return;
        }
        if (window <= 0) {
            window = n_;
        }
        if (hop <= 0) {
            hop = 1;
        }
        if constexpr (kFftInitSupportsStftMode) {
            handle_ = fft_init_full_v2(
                static_cast<std::size_t>(n_),
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
                1,
                0,
                window_kind,
                0.0f,
                0.0f,
                window_kind,
                0.0f,
                0.0f,
                FFT_WINDOW_NORM_NONE,
                FFT_COLA_OFF);
        } else {
            handle_ = fft_init_full_v2(
                static_cast<std::size_t>(n_),
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
                0,
                window_kind,
                0.0f,
                0.0f,
                window_kind,
                0.0f,
                0.0f,
                FFT_WINDOW_NORM_NONE,
                FFT_COLA_OFF);
        }
    }

    StreamingReference(StreamingReference &&other) noexcept : handle_(other.handle_), n_(other.n_) {
        other.handle_ = nullptr;
        other.n_ = 0;
    }

    StreamingReference &operator=(StreamingReference &&other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            n_ = other.n_;
            other.handle_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    StreamingReference(const StreamingReference &) = delete;
    StreamingReference &operator=(const StreamingReference &) = delete;

    ~StreamingReference() {
        reset();
    }

    bool valid() const { return handle_ != nullptr; }
    int n() const { return n_; }

    std::size_t push(
        const std::vector<double> &pcm,
        std::size_t max_frames,
        int flush_mode,
        std::vector<double> &out_real,
        std::vector<double> &out_imag
    ) {
        if (!valid() || max_frames == 0) {
            out_real.clear();
            out_imag.clear();
            return 0;
        }
        std::vector<float> pcm_f(pcm.size(), 0.0f);
        for (std::size_t i = 0; i < pcm.size(); ++i) {
            pcm_f[i] = static_cast<float>(pcm[i]);
        }
        const std::size_t capacity = max_frames * static_cast<std::size_t>(n_);
        std::vector<float> real_f(capacity, 0.0f);
        std::vector<float> imag_f(capacity, 0.0f);
        std::vector<float> mag_f(capacity, 0.0f);
        const std::size_t produced = fft_stream_push_pcm(
            handle_,
            pcm_f.data(),
            pcm_f.size(),
            real_f.data(),
            imag_f.data(),
            mag_f.data(),
            max_frames,
            flush_mode);
        const std::size_t copy_len = produced * static_cast<std::size_t>(n_);
        out_real.assign(real_f.begin(), real_f.begin() + copy_len);
        out_imag.assign(imag_f.begin(), imag_f.begin() + copy_len);
        return produced;
    }

private:
    void reset() {
        if (handle_ != nullptr) {
            fft_free(handle_);
            handle_ = nullptr;
        }
        n_ = 0;
    }

    void *handle_{nullptr};
    int n_{0};
};

}  // namespace

int main() {
    const int n = 8;
    const int window = 4;
    const int hop = 2;

    void *stream = amp_fft_backend_stream_create(n, window, hop, AMP_FFT_WINDOW_RECT);
    if (stream == nullptr) {
        std::cerr << "failed to create streaming context" << std::endl;
        return 1;
    }

    StreamingReference reference(n, window, hop, FFT_WINDOW_RECT);
    if (!reference.valid()) {
        std::cerr << "failed to create fftfree streaming reference" << std::endl;
        amp_fft_backend_stream_destroy(stream);
        return 1;
    }

    auto process_chunk = [&](const std::vector<double> &chunk) -> bool {
        std::vector<double> amp_real(static_cast<std::size_t>(n), 0.0);
        std::vector<double> amp_imag(static_cast<std::size_t>(n), 0.0);
        std::vector<double> ref_real;
        std::vector<double> ref_imag;
        const std::size_t produced_amp = amp_fft_backend_stream_push(
            stream,
            chunk.data(),
            chunk.size(),
            n,
            amp_real.data(),
            amp_imag.data(),
            1,
            AMP_FFT_STREAM_FLUSH_NONE);
        const std::size_t produced_ref = reference.push(
            chunk,
            1,
            FFT_STREAM_FLUSH_NONE,
            ref_real,
            ref_imag);
        if (produced_amp != produced_ref) {
            std::cerr << "produced frame count mismatch (amp=" << produced_amp << ", fftfree=" << produced_ref << ")" << std::endl;
            return false;
        }
        const std::size_t copy_len = produced_amp * static_cast<std::size_t>(n);
        amp_real.resize(copy_len);
        amp_imag.resize(copy_len);
        if (!compare_buffers(amp_real, ref_real) || !compare_buffers(amp_imag, ref_imag)) {
            std::cerr << "stream output mismatch vs fftfree reference" << std::endl;
            return false;
        }
        return true;
    };

    const std::vector<double> chunk1{1.0, -2.0, 3.0, -4.0};
    const std::vector<double> chunk2{5.0, -6.0};

    if (!process_chunk(chunk1)) {
        amp_fft_backend_stream_destroy(stream);
        return 1;
    }
    if (!process_chunk(chunk2)) {
        amp_fft_backend_stream_destroy(stream);
        return 1;
    }

    amp_fft_backend_stream_destroy(stream);

    // Inverse streaming round-trip with non-overlapping framing for clarity.
    {
        const int inv_n = 4;
        const int inv_window = 4;
        const int inv_hop = 4;
        void *forward = amp_fft_backend_stream_create(inv_n, inv_window, inv_hop, AMP_FFT_WINDOW_RECT);
        if (forward == nullptr) {
            std::cerr << "failed to create forward stream for inverse test" << std::endl;
            return 1;
        }
        void *inverse = amp_fft_backend_stream_create_inverse(inv_n, inv_window, inv_hop, AMP_FFT_WINDOW_RECT);
        if (inverse == nullptr) {
            std::cerr << "failed to create inverse stream" << std::endl;
            amp_fft_backend_stream_destroy(forward);
            return 1;
        }

        const std::vector<double> pcm_in{0.5, -1.0, 0.25, 0.75, -0.5, 1.0, -0.25, -0.75};
        const std::size_t max_frames = 4;
        std::vector<double> f_real(max_frames * static_cast<std::size_t>(inv_n), 0.0);
        std::vector<double> f_imag(max_frames * static_cast<std::size_t>(inv_n), 0.0);

        const std::size_t produced_frames = amp_fft_backend_stream_push(
            forward,
            pcm_in.data(),
            pcm_in.size(),
            inv_n,
            f_real.data(),
            f_imag.data(),
            max_frames,
            AMP_FFT_STREAM_FLUSH_FINAL);
        if (produced_frames == 0) {
            std::cerr << "forward stream produced no frames for inverse test" << std::endl;
            amp_fft_backend_stream_destroy(forward);
            amp_fft_backend_stream_destroy(inverse);
            return 1;
        }

        f_real.resize(produced_frames * static_cast<std::size_t>(inv_n));
        f_imag.resize(produced_frames * static_cast<std::size_t>(inv_n));

        std::vector<double> reconstructed;
        reconstructed.reserve(pcm_in.size() + inv_n);
        std::vector<double> pcm_scratch(static_cast<std::size_t>(inv_n), 0.0);

        for (std::size_t frame = 0; frame < produced_frames; ++frame) {
            const std::size_t offset = frame * static_cast<std::size_t>(inv_n);
            std::fill(pcm_scratch.begin(), pcm_scratch.end(), 0.0);
            const std::size_t produced = amp_fft_backend_stream_push_spectrum(
                inverse,
                f_real.data() + offset,
                f_imag.data() + offset,
                1,
                inv_n,
                pcm_scratch.data(),
                pcm_scratch.size(),
                AMP_FFT_STREAM_FLUSH_NONE);
            reconstructed.insert(reconstructed.end(), pcm_scratch.begin(), pcm_scratch.begin() + produced);
        }

        std::vector<double> tail(static_cast<std::size_t>(inv_n), 0.0);
        const std::size_t drained = amp_fft_backend_stream_push_spectrum(
            inverse,
            nullptr,
            nullptr,
            0,
            inv_n,
            tail.data(),
            tail.size(),
            AMP_FFT_STREAM_FLUSH_FINAL);
        reconstructed.insert(reconstructed.end(), tail.begin(), tail.begin() + drained);

        const std::size_t compare_len = std::min(reconstructed.size(), pcm_in.size());
        bool inverse_ok = compare_len == pcm_in.size();
        for (std::size_t i = 0; i < compare_len && inverse_ok; ++i) {
            if (std::fabs(reconstructed[i] - pcm_in[i]) > 1e-6) {
                inverse_ok = false;
            }
        }
        if (!inverse_ok) {
            std::cerr << "inverse streaming round-trip mismatch" << std::endl;
            amp_fft_backend_stream_destroy(forward);
            amp_fft_backend_stream_destroy(inverse);
            return 1;
        }

        amp_fft_backend_stream_destroy(forward);
        amp_fft_backend_stream_destroy(inverse);
    }

    return 0;
}
