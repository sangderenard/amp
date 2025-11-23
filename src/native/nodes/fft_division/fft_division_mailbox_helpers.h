// Helper routines for copying mailbox chain nodes into legacy tap buffers.
// Moved from test helpers into node-related code so node internals can reuse them.
#ifndef AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_
#define AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_

#include <algorithm>
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
        const size_t cache_window = real_buffer.shape.channels > 0U
            ? static_cast<size_t>(real_buffer.shape.channels)
            : window;
        const size_t cache_frames_hint = real_buffer.cache_frames > 0U
            ? static_cast<size_t>(real_buffer.cache_frames)
            : 0U;
        const size_t cache_frames_from_len = (cache_window > 0U)
            ? (real_buffer.cache_buffer_len / cache_window)
            : 0U;
        const size_t cache_frames = std::max(cache_frames_hint, cache_frames_from_len);
        const size_t cache_values = cache_frames * cache_window;
        if (cache_values > 0) {
            const size_t copy_values = std::min(legacy_value_count, cache_values);
            for (size_t i = 0; i < copy_values; ++i) {
                legacy_real[i] = real_buffer.cache_data[i];
                legacy_imag[i] = imag_buffer.cache_data[i];
            }
            result.values_written = copy_values;
            result.frames_committed = (cache_window > 0U) ? (copy_values / cache_window) : 0U;
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
            if (frame < capacity_frames) {
                const size_t idx = frame * window;
                const size_t bins_to_copy = std::min(
                    window,
                    static_cast<size_t>(node->window_size > 0 ? node->window_size : 1)
                );
                const auto &real_bins = node->spectral_real_bins;
                const auto &imag_bins = node->spectral_imag_bins;
                if (idx + bins_to_copy <= legacy_value_count) {
                    for (size_t bin = 0; bin < bins_to_copy; ++bin) {
                        legacy_real[idx + bin] = bin < real_bins.size() ? real_bins[bin] : 0.0;
                        legacy_imag[idx + bin] = bin < imag_bins.size() ? imag_bins[bin] : 0.0;
                    }
                    result.values_written += bins_to_copy;
                    result.frames_committed = std::max(result.frames_committed, frame + 1);
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
