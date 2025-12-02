#ifndef AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_
#define AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>


extern "C" {
#include "amp_native.h"
#include "amp_fft_backend.h"
}

#ifdef __cplusplus
#include "amp_native_mailbox_chain.hpp"
#include "nodes/fft_division/fft_division_mailbox_helpers.h"
#endif

namespace amp::tests::fft_division_shared {

struct TapDescriptor {
    std::string name;
    std::string buffer_class;
    EdgeRunnerTensorShape shape{};
    uint32_t hop_size{1U};

    size_t ValueCount() const {
        const size_t batches = std::max<uint32_t>(1U, shape.batches);
        const size_t channels = std::max<uint32_t>(1U, shape.channels);
        const size_t frames = std::max<uint32_t>(1U, shape.frames);
        return batches * channels * frames;
    }
};

inline uint32_t ComputeFrameCount(size_t total_frames, uint32_t hop_count) {
    if (hop_count == 0U) {
        hop_count = 1U;
    }
    if (total_frames == 0U) {
        return 0U;
    }
    const size_t frames = (total_frames + static_cast<size_t>(hop_count) - 1U) /
        static_cast<size_t>(hop_count);
    return static_cast<uint32_t>(frames);
}

inline TapDescriptor BuildPcmTapDescriptor(
    uint32_t window_size,
    uint32_t hop_count,
    size_t total_frames,
    uint32_t channels = 1U
) {
    TapDescriptor descriptor{};
    descriptor.name = "pcm_0";
    descriptor.buffer_class = "pcm";
    descriptor.hop_size = hop_count > 0U ? hop_count : 1U;
    descriptor.shape.batches = 1U;
    descriptor.shape.channels = std::max<uint32_t>(1U, channels);
    descriptor.shape.frames = static_cast<uint32_t>(total_frames);
    (void)window_size;
    return descriptor;
}

inline TapDescriptor BuildSpectralTapDescriptor(
    uint32_t window_size,
    uint32_t hop_count,
    size_t total_frames,
    uint32_t spectral_lanes = 1U
) {
    TapDescriptor descriptor{};
    descriptor.name = "spectral_0";
    descriptor.buffer_class = "spectrum";
    descriptor.hop_size = hop_count > 0U ? hop_count : 1U;
    descriptor.shape.batches = std::max<uint32_t>(1U, spectral_lanes);
    descriptor.shape.channels = std::max<uint32_t>(1U, window_size);
    descriptor.shape.frames = ComputeFrameCount(total_frames, descriptor.hop_size);
    return descriptor;
}


#ifdef __cplusplus
// Overload for persistent mailbox chain and optional legacy data buffer
inline EdgeRunnerTapBuffer InstantiateTapBuffer(
    const TapDescriptor &descriptor,
    double *data,
    amp::tests::fft_division_shared::PersistentMailboxNode* mailbox_head = nullptr
) {
    EdgeRunnerTapBuffer tap{};
    tap.tap_name = descriptor.name.c_str();
    tap.buffer_class = descriptor.buffer_class.c_str();
    tap.shape = descriptor.shape;
    const size_t computed_stride = static_cast<size_t>(
        std::max<uint32_t>(1U, descriptor.shape.batches) *
        std::max<uint32_t>(1U, descriptor.shape.channels));
    tap.frame_stride = computed_stride;
    tap.data = data;
    tap.mailbox_head = mailbox_head;
    return tap;
}

// Helper to build a persistent mailbox chain from a vector of data pointers
inline amp::tests::fft_division_shared::PersistentMailboxNode* BuildPersistentMailboxChain(const std::vector<void*>& chunks, const std::vector<size_t>& sizes) {
    using Node = amp::tests::fft_division_shared::PersistentMailboxNode;
    if (chunks.empty() || chunks.size() != sizes.size()) return nullptr;
    Node* head = new Node(chunks[0], sizes[0]);
    Node* current = head;
    for (size_t i = 1; i < chunks.size(); ++i) {
        current->next = new Node(chunks[i], sizes[i]);
        current = current->next;
    }
    return head;
}

#if 0
// mailbox helpers moved to node header; included above.
#endif
#endif

namespace detail {

inline size_t SafeDim(uint32_t value) {
    return (value > 0U) ? static_cast<size_t>(value) : 1U;
}

inline size_t ComputeFrameStride(const EdgeRunnerTapBuffer &buffer) {
    if (buffer.frame_stride > 0U) {
        return buffer.frame_stride;
    }
    return SafeDim(buffer.shape.batches) * SafeDim(buffer.shape.channels);
}

inline std::string DeriveTapName(const EdgeRunnerTapBuffer &buffer, size_t ordinal) {
    if (buffer.tap_name != nullptr && buffer.tap_name[0] != '\0') {
        return std::string(buffer.tap_name);
    }
    char generated[32];
    std::snprintf(generated, sizeof(generated), "tap_%zu", ordinal);
    return std::string(generated);
}

}  // namespace detail

inline std::vector<double> DecodeTapTensor(const EdgeRunnerTapBuffer &buffer) {
    const size_t frames = detail::SafeDim(buffer.shape.frames);
    const size_t batches = detail::SafeDim(buffer.shape.batches);
    const size_t channels = detail::SafeDim(buffer.shape.channels);
    const size_t total = frames * batches * channels;
    std::vector<double> decoded(total, 0.0);
    if (buffer.data == nullptr || total == 0U) {
        return decoded;
    }
    const size_t frame_stride = detail::ComputeFrameStride(buffer);
    for (size_t frame = 0; frame < frames; ++frame) {
        const double *frame_ptr = buffer.data + frame * frame_stride;
        for (size_t batch = 0; batch < batches; ++batch) {
            const double *batch_ptr = frame_ptr + batch * channels;
            const size_t base = frame * batches * channels + batch * channels;
            for (size_t channel = 0; channel < channels; ++channel) {
                decoded[base + channel] = batch_ptr[channel];
            }
        }
    }
    return decoded;
}

inline std::unordered_map<std::string, std::vector<double>> DecodeTapBuffers(const EdgeRunnerTapBufferSet &set) {
    std::unordered_map<std::string, std::vector<double>> decoded;
    if (set.items == nullptr || set.count == 0U) {
        return decoded;
    }
    for (uint32_t i = 0; i < set.count; ++i) {
        const EdgeRunnerTapBuffer &buffer = set.items[i];
        decoded.emplace(detail::DeriveTapName(buffer, i), DecodeTapTensor(buffer));
    }
    return decoded;
}

}  // namespace amp::tests::fft_division_shared

// FFT identity helpers for cleaning signals and spectral data
namespace amp::tests::fft_identity {

// Forward FFT: PCM -> spectral (real, imag)
inline std::pair<std::vector<double>, std::vector<double>> forward_fft(
    const std::vector<double>& pcm, int window_size, int hop, int window_kind = AMP_FFT_WINDOW_HANN) {
    std::pair<std::vector<double>, std::vector<double>> result;
    if (pcm.empty() || window_size <= 0 || hop <= 0) {
        return result;
    }

    void* forward = amp_fft_backend_stream_create(window_size, window_size, hop, window_kind);
    if (forward == nullptr) {
        return result;
    }

    const size_t tail_frames = 0; // no tail for simple identity
    const size_t total_input = pcm.size() + tail_frames;
    std::vector<double> spectral_real(total_input * window_size, 0.0);
    std::vector<double> spectral_imag(total_input * window_size, 0.0);

    // Push the PCM + zero tail
    std::vector<double> input = pcm;
    size_t spectral_emitted = 0;
    const size_t produced = amp_fft_backend_stream_push(
        forward, input.data(), input.size(), window_size,
        spectral_real.data(), spectral_imag.data(), spectral_real.size(), AMP_FFT_STREAM_FLUSH_NONE);
    spectral_emitted += produced;

    // Drain any remaining
    for (int i = 0; i < 8; ++i) {
        const size_t p = amp_fft_backend_stream_push(
            forward, nullptr, 0, window_size,
            spectral_real.data() + spectral_emitted * window_size,
            spectral_imag.data() + spectral_emitted * window_size,
            spectral_real.size() - spectral_emitted * window_size, AMP_FFT_STREAM_FLUSH_PARTIAL);
        if (p == 0) break;
        spectral_emitted += p;
    }
    for (int i = 0; i < 8; ++i) {
        const size_t p = amp_fft_backend_stream_push(
            forward, nullptr, 0, window_size,
            spectral_real.data() + spectral_emitted * window_size,
            spectral_imag.data() + spectral_emitted * window_size,
            spectral_real.size() - spectral_emitted * window_size, AMP_FFT_STREAM_FLUSH_FINAL);
        if (p == 0) break;
        spectral_emitted += p;
    }

    amp_fft_backend_stream_destroy(forward);

    result.first.assign(spectral_real.begin(), spectral_real.begin() + spectral_emitted * window_size);
    result.second.assign(spectral_imag.begin(), spectral_imag.begin() + spectral_emitted * window_size);
    return result;
}

// Reverse FFT: spectral (real, imag) -> PCM
inline std::vector<double> reverse_fft(
    const std::vector<double>& spectral_real, const std::vector<double>& spectral_imag,
    int window_size, int hop, int window_kind = AMP_FFT_WINDOW_HANN) {
    std::vector<double> result;
    if (spectral_real.empty() || spectral_imag.empty() || spectral_real.size() != spectral_imag.size() ||
        window_size <= 0 || hop <= 0) {
        return result;
    }

    void* inverse = amp_fft_backend_stream_create_inverse(window_size, window_size, hop, window_kind);
    if (inverse == nullptr) {
        return result;
    }

    const size_t spectral_frames = spectral_real.size() / window_size;
    std::vector<double> pcm_scratch(window_size, 0.0);
    std::vector<double> produced_pcm;

    // Push all spectral frames
    const size_t produced = amp_fft_backend_stream_push_spectrum(
        inverse, spectral_real.data(), spectral_imag.data(), spectral_frames,
        window_size, pcm_scratch.data(), pcm_scratch.size(), AMP_FFT_STREAM_FLUSH_NONE);
    for (size_t i = 0; i < produced; ++i) {
        produced_pcm.push_back(pcm_scratch[i]);
    }

    // Drain
    for (int i = 0; i < 8; ++i) {
        const size_t p = amp_fft_backend_stream_push_spectrum(
            inverse, nullptr, nullptr, 0, window_size,
            pcm_scratch.data(), pcm_scratch.size(), AMP_FFT_STREAM_FLUSH_PARTIAL);
        if (p == 0) break;
        for (size_t j = 0; j < p; ++j) {
            produced_pcm.push_back(pcm_scratch[j]);
        }
    }
    for (int i = 0; i < 8; ++i) {
        const size_t p = amp_fft_backend_stream_push_spectrum(
            inverse, nullptr, nullptr, 0, window_size,
            pcm_scratch.data(), pcm_scratch.size(), AMP_FFT_STREAM_FLUSH_FINAL);
        if (p == 0) break;
        for (size_t j = 0; j < p; ++j) {
            produced_pcm.push_back(pcm_scratch[j]);
        }
    }

    amp_fft_backend_stream_destroy(inverse);
    return produced_pcm;
}

// Clean PCM: identity roundtrip (forward then reverse)
inline std::vector<double> clean_pcm(
    const std::vector<double>& pcm, int window_size, int hop, int window_kind = AMP_FFT_WINDOW_HANN) {
    auto [real, imag] = forward_fft(pcm, window_size, hop, window_kind);
    std::vector<double> cleaned = reverse_fft(real, imag, window_size, hop, window_kind);
    cleaned.resize(pcm.size()); // truncate to input size
    return cleaned;
}

// Clean spectral: reverse then forward
inline std::pair<std::vector<double>, std::vector<double>> clean_spectral(
    const std::vector<double>& spectral_real, const std::vector<double>& spectral_imag,
    int window_size, int hop, int window_kind = AMP_FFT_WINDOW_HANN) {
    std::vector<double> pcm = reverse_fft(spectral_real, spectral_imag, window_size, hop, window_kind);
    return forward_fft(pcm, window_size, hop, window_kind);
}

}  // namespace amp::tests::fft_identity

#endif  // AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_
