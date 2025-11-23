// Helper routines for copying mailbox chain nodes into legacy tap buffers.
// Moved from test helpers into node-related code so node internals can reuse them.
#ifndef AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_
#define AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_

#include <cstddef>
#include <cstdio>
#include <limits>

extern "C" {
#include "amp_native.h"
}

#ifdef __cplusplus
#include "amp_native_mailbox_chain.hpp"

namespace amp::tests::fft_division_shared {

// Result from attempting to populate a legacy tap buffer from a mailbox chain.
struct TapMailboxReadResult {
    size_t frames_committed{0};
    size_t values_written{0};
    bool copied_from_mailbox{false};
    bool aliased_legacy_buffer{false};
};

inline TapMailboxReadResult PopulateLegacyPcmFromMailbox(
    const EdgeRunnerTapBuffer &pcm_buffer,
    double *legacy_pcm,
    size_t legacy_capacity_frames
) {
    using MailboxNode = PersistentMailboxNode;
    using MailboxChain = EdgeRunnerTapMailboxChain;

    TapMailboxReadResult result{};
    if (legacy_pcm == nullptr || legacy_capacity_frames == 0U) {
        return result;
    }

    result.aliased_legacy_buffer = (legacy_pcm == pcm_buffer.data);

    // If a staged cache exists on the tap, prefer copying from that cache
    // rather than traversing the mailbox chain. The staged cache is the
    // consumer-visible buffer produced by the blocking helper and reflects
    // the packed PCM samples the mailbox provided.
    if (pcm_buffer.cache_data != nullptr) {
        size_t copy_frames = std::min(legacy_capacity_frames, pcm_buffer.cache_buffer_len);
        for (size_t i = 0; i < copy_frames; ++i) {
            legacy_pcm[i] = pcm_buffer.cache_data[i];
        }
        result.frames_committed = copy_frames;
        result.values_written = copy_frames;
        result.copied_from_mailbox = (result.values_written > 0U);
        fprintf(stderr, "[MAILBOX-COPY] pcm copied_from_cache frames_committed=%zu values_written=%zu aliased=%d\n",
                result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stderr);
        return result;
    }

    MailboxNode *node = MailboxChain::get_head(const_cast<EdgeRunnerTapBuffer &>(pcm_buffer));

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
    if (result.copied_from_mailbox) {
        fprintf(stderr, "[MAILBOX-COPY] pcm copied frames_committed=%zu values_written=%zu aliased=%d\n",
                result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stderr);
    } else {
        fprintf(stderr, "[MAILBOX-COPY] pcm nothing_copied\n");
        fflush(stderr);
    }
    return result;
}

inline TapMailboxReadResult PopulateLegacySpectrumFromMailbox(
    const EdgeRunnerTapBuffer &real_buffer,
    const EdgeRunnerTapBuffer &imag_buffer,
    double *legacy_real,
    double *legacy_imag,
    size_t legacy_value_count
) {
    using MailboxNode = PersistentMailboxNode;
    using MailboxChain = EdgeRunnerTapMailboxChain;

    TapMailboxReadResult result{};
    if (legacy_real == nullptr || legacy_imag == nullptr || legacy_value_count == 0U) {
        return result;
    }

    result.aliased_legacy_buffer = (
        legacy_real == real_buffer.data && legacy_imag == imag_buffer.data
    );

    const size_t window = real_buffer.shape.channels > 0U
        ? static_cast<size_t>(real_buffer.shape.channels)
        : 1U;
    const size_t capacity_frames = window > 0U ? legacy_value_count / window : 0U;

    // If staged caches exist for both real and imag taps, prefer copying
    // directly from those caches into the legacy arrays. This ensures
    // the verification path uses the same consumer-visible buffer that
    // `amp_tap_cache_block_until_ready` prepared.
    if (real_buffer.cache_data != nullptr && imag_buffer.cache_data != nullptr) {
        size_t real_len = real_buffer.cache_buffer_len;
        size_t imag_len = imag_buffer.cache_buffer_len;
        size_t copy_values = legacy_value_count;
        if (real_len < copy_values) copy_values = real_len;
        if (imag_len < copy_values) copy_values = imag_len;
        if (copy_values > 0) {
            // Detect whether the staged caches contain interleaved (real,imag)
            // pairs rather than contiguous per-component arrays. Some
            // tap-fill paths write interleaved pairs into each staged buffer
            // (one pair per mailbox node). In that case both buffers will
            // contain identical interleaved data and we must de-interleave
            // into the legacy real/imag arrays. Heuristic: when both cache
            // lengths match, are even, and the first two doubles of the
            // real and imag caches are equal, assume interleaved layout.
            bool handled = false;
            if (real_len == imag_len && (copy_values % 2) == 0 && real_len >= 2) {
                double r0 = real_buffer.cache_data[0];
                double r1 = real_buffer.cache_data[1];
                double i0 = imag_buffer.cache_data[0];
                double i1 = imag_buffer.cache_data[1];
                if (r0 == i0 && r1 == i1) {
                    // Treat cache as interleaved pairs: [r0,i0,r1,i1,...]
                    const size_t pairs = copy_values / 2;
                    for (size_t p = 0; p < pairs; ++p) {
                        const size_t src_idx = p * 2;
                        legacy_real[p] = real_buffer.cache_data[src_idx];
                        legacy_imag[p] = real_buffer.cache_data[src_idx + 1];
                    }
                    result.values_written = pairs * 2;
                    result.frames_committed = (window > 0) ? pairs / window : pairs;
                    result.copied_from_mailbox = (result.values_written > 0U);
                    fprintf(stderr, "[MAILBOX-COPY] spectral deinterleaved_from_cache frames_committed=%zu values_written=%zu aliased=%d\n",
                            result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
                    fflush(stderr);
                    return result;
                }
            }

            // Fallback: copy contiguous value arrays into legacy buffers
            for (size_t i = 0; i < copy_values; ++i) {
                legacy_real[i] = real_buffer.cache_data[i];
                legacy_imag[i] = imag_buffer.cache_data[i];
            }
            result.values_written = copy_values;
            result.frames_committed = (window > 0) ? (copy_values / window) : 0;
            result.copied_from_mailbox = (result.values_written > 0U);
            fprintf(stderr, "[MAILBOX-COPY] spectral copied_from_cache frames_committed=%zu values_written=%zu aliased=%d\n",
                    result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
            fflush(stderr);
            return result;
        }
    }

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
        if (node->node_kind == MailboxNode::NodeKind::SPECTRAL &&
            node->frame_index >= 0 &&
            node->spectral_real != nullptr &&
            node->spectral_imag != nullptr) {

            const size_t frame = static_cast<size_t>(
                static_cast<int>(node->frame_index) - base_frame);
            const size_t bins = (node->window_size > 0)
                ? static_cast<size_t>(node->window_size)
                : window;

            if (frame < capacity_frames) {
                const size_t bins_to_copy = std::min(window, bins);
                const size_t base = frame * window;
                for (size_t bin = 0; bin < bins_to_copy; ++bin) {
                    const size_t idx = base + bin;
                    if (idx < legacy_value_count) {
                        legacy_real[idx] = node->spectral_real[bin];
                        legacy_imag[idx] = node->spectral_imag[bin];
                        result.values_written += 1U;
                    }
                }
                result.frames_committed = std::max(result.frames_committed, frame + 1);
            }
        }
        node = node->next;
    }

    result.copied_from_mailbox = (result.values_written > 0U);
    if (result.copied_from_mailbox) {
        fprintf(stderr, "[MAILBOX-COPY] spectral copied frames_committed=%zu values_written=%zu aliased=%d\n",
                result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stderr);
    } else {
        fprintf(stderr, "[MAILBOX-COPY] spectral nothing_copied\n");
        fflush(stderr);
    }
    return result;
}

}  // namespace amp::tests::fft_division_shared

#endif  // __cplusplus

#endif  // AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_
