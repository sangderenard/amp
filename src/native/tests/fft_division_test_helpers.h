#ifndef AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_
#define AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "amp_native.h"
}

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
    descriptor.name = "pcm";
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
    descriptor.name = "spectral";
    descriptor.buffer_class = "spectrum";
    descriptor.hop_size = hop_count > 0U ? hop_count : 1U;
    descriptor.shape.batches = std::max<uint32_t>(1U, spectral_lanes);
    descriptor.shape.channels = std::max<uint32_t>(1U, window_size);
    descriptor.shape.frames = ComputeFrameCount(total_frames, descriptor.hop_size);
    return descriptor;
}

inline EdgeRunnerTapBuffer InstantiateTapBuffer(const TapDescriptor &descriptor, double *data) {
    EdgeRunnerTapBuffer tap{};
    tap.tap_name = descriptor.name.c_str();
    tap.buffer_class = descriptor.buffer_class.c_str();
    tap.shape = descriptor.shape;
    const size_t computed_stride = static_cast<size_t>(
        std::max<uint32_t>(1U, descriptor.shape.batches) *
        std::max<uint32_t>(1U, descriptor.shape.channels));
    tap.frame_stride = computed_stride;
    tap.data = data;
    return tap;
}

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

inline std::vector<float> DecodeTapTensor(const EdgeRunnerTapBuffer &buffer) {
    const size_t frames = detail::SafeDim(buffer.shape.frames);
    const size_t batches = detail::SafeDim(buffer.shape.batches);
    const size_t channels = detail::SafeDim(buffer.shape.channels);
    const size_t total = frames * batches * channels;
    std::vector<float> decoded(total, 0.0f);
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
                decoded[base + channel] = static_cast<float>(batch_ptr[channel]);
            }
        }
    }
    return decoded;
}

inline std::unordered_map<std::string, std::vector<float>> DecodeTapBuffers(const EdgeRunnerTapBufferSet &set) {
    std::unordered_map<std::string, std::vector<float>> decoded;
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

#endif  // AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_
