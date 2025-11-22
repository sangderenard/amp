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


extern "C" {
#include "amp_native.h"
}

#ifdef __cplusplus
#include "amp_native_mailbox_chain.hpp"
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

struct TapMailboxReadResult {
    size_t frames_committed{0};
    size_t values_written{0};
    // Indicates whether the helper copied values into the legacy buffer (true)
    // or simply aliased an existing region (false). At present, mailbox nodes
    // are discrete so reads are performed via copy into the legacy tap buffer.
    bool copied_from_mailbox{false};
    // Communicates whether the caller is now holding a direct reference to the
    // legacy tap buffer supplied to the node. When copying from a mailbox chain
    // this remains false; callers can set it when choosing to retain/alias.
    bool aliased_legacy_buffer{false};
};

inline TapMailboxReadResult PopulateLegacyPcmFromMailbox(
    const EdgeRunnerTapBuffer &pcm_buffer,
    double *legacy_pcm,
    size_t legacy_capacity_frames
) {
    using MailboxNode = amp::tests::fft_division_shared::PersistentMailboxNode;
    using MailboxChain = amp::tests::fft_division_shared::EdgeRunnerTapMailboxChain;

    TapMailboxReadResult result{};
    if (legacy_pcm == nullptr || legacy_capacity_frames == 0U) {
        return result;
    }

    // Communicate that the caller supplied the legacy tap buffer directly.
    // This distinguishes copy-from-mailbox from alias/reference semantics.
    result.aliased_legacy_buffer = (legacy_pcm == pcm_buffer.data);

    MailboxNode *node = MailboxChain::get_head(const_cast<EdgeRunnerTapBuffer &>(pcm_buffer));

    // Normalize frame indices so the first observed frame maps to legacy index 0.
    int base_frame = std::numeric_limits<int>::max();
    for (MailboxNode *cursor = node; cursor != nullptr; cursor = cursor->next) {
        if (cursor->node_kind == MailboxNode::NodeKind::PCM && cursor->frame_index >= 0) {
            base_frame = std::min(base_frame, cursor->frame_index);
        }
    }
    if (base_frame == std::numeric_limits<int>::max()) {
        return result;
    }

    while (node != nullptr) {
        if (node->node_kind == MailboxNode::NodeKind::PCM && node->frame_index >= 0) {
            const size_t frame = static_cast<size_t>(
                static_cast<int>(node->frame_index) - base_frame);
            if (frame < legacy_capacity_frames) {
                legacy_pcm[frame] = node->pcm_sample;
                result.frames_committed = std::max(result.frames_committed, frame + 1);
                ++result.values_written;
            }
        }
        node = node->next;
    }

    result.copied_from_mailbox = (result.values_written > 0U);
    return result;
}

inline TapMailboxReadResult PopulateLegacySpectrumFromMailbox(
    const EdgeRunnerTapBuffer &real_buffer,
    const EdgeRunnerTapBuffer &imag_buffer,
    double *legacy_real,
    double *legacy_imag,
    size_t legacy_value_count
) {
    using MailboxNode = amp::tests::fft_division_shared::PersistentMailboxNode;
    using MailboxChain = amp::tests::fft_division_shared::EdgeRunnerTapMailboxChain;

    TapMailboxReadResult result{};
    if (legacy_real == nullptr || legacy_imag == nullptr || legacy_value_count == 0U) {
        return result;
    }

    // Communicate that the caller supplied the legacy spectral tap buffers directly.
    result.aliased_legacy_buffer = (
        legacy_real == real_buffer.data && legacy_imag == imag_buffer.data
    );

    const size_t window = real_buffer.shape.channels > 0U
        ? static_cast<size_t>(real_buffer.shape.channels)
        : 1U;
    const size_t capacity_frames = window > 0U ? legacy_value_count / window : 0U;

    // Normalize frame indices to start at 0 in the legacy buffer.
    int base_frame = std::numeric_limits<int>::max();
    MailboxNode *scan = MailboxChain::get_head(const_cast<EdgeRunnerTapBuffer &>(real_buffer));
    while (scan != nullptr) {
        if (scan->node_kind == MailboxNode::NodeKind::SPECTRAL && scan->frame_index >= 0) {
            base_frame = std::min(base_frame, scan->frame_index);
        }
        scan = scan->next;
    }
    if (base_frame == std::numeric_limits<int>::max()) {
        return result;
    }

    MailboxNode *node = MailboxChain::get_head(const_cast<EdgeRunnerTapBuffer &>(real_buffer));
    while (node != nullptr) {
        if (node->node_kind == MailboxNode::NodeKind::SPECTRAL && node->frame_index >= 0) {
            const size_t frame = static_cast<size_t>(
                static_cast<int>(node->frame_index) - base_frame);
            const size_t bin = (node->slot >= 0 && node->slot < static_cast<int>(window))
                ? static_cast<size_t>(node->slot)
                : 0U;
            if (frame < capacity_frames && bin < window) {
                const size_t base = frame * window + bin;
                if (base < legacy_value_count) {
                    legacy_real[base] = node->spectral_real;
                    legacy_imag[base] = node->spectral_imag;
                    result.frames_committed = std::max(result.frames_committed, frame + 1);
                    ++result.values_written;
                }
            }
        }
        node = node->next;
    }

    result.copied_from_mailbox = (result.values_written > 0U);
    return result;
}
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

#endif  // AMP_NATIVE_TESTS_FFT_DIVISION_TEST_HELPERS_H_
