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

/* Avoid conflicts with Windows `min`/`max` macros when this header is
    included into translation units that pull in Windows headers. Push/pop
    the macro state so we don't disturb other code. */
#if defined(_MSC_VER)
#pragma push_macro("max")
#pragma push_macro("min")
#undef max
#undef min
#endif

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
        fprintf(stdout, "[MAILBOX-COPY] pcm copied_from_cache frames_committed=%zu values_written=%zu aliased=%d\n",
            result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stdout);
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
        fprintf(stdout, "[MAILBOX-COPY] pcm copied frames_committed=%zu values_written=%zu aliased=%d\n",
            result.frames_committed, result.values_written, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stdout);
    } else {
        fprintf(stdout, "[MAILBOX-COPY] pcm nothing_copied\n");
        fflush(stdout);
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
    // `result.values_written` will represent the number of complex bins written
    // (one complex bin == real+imag). Keep a separate scalar counter for
    // diagnostic prints when needed.
    size_t values_written_complex = 0U;
    size_t values_written_scalars = 0U;
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

            // If both taps expose the same staged cache pointer, the data is
            // stored interleaved as [r0,i0,r1,i1,...]. Handle that case by
            // deinterleaving from the single buffer. This avoids false
            // positives when comparing values across aliased pointers.
            if (real_buffer.cache_data == imag_buffer.cache_data) {
                const double *cache = real_buffer.cache_data;
                const size_t total_scalars = real_buffer.cache_buffer_len;
                const size_t pairs = std::min(copy_values, total_scalars / 2U);
                for (size_t p = 0; p < pairs; ++p) {
                    const size_t src_idx = p * 2U;
                    legacy_real[p] = cache[src_idx];
                    legacy_imag[p] = cache[src_idx + 1U];
                }
                // `pairs` counts complex bins (pairs of scalars). Report complex
                // bin counts in `result.values_written` and keep scalar totals
                // for diagnostics.
                result.values_written = pairs; // complex bins
                values_written_complex = pairs;
                values_written_scalars = pairs * 2U;
                result.frames_committed = (cache_window > 0U) ? (pairs / cache_window) : pairs;
                result.copied_from_mailbox = (result.values_written > 0U);
                fprintf(stdout, "[MAILBOX-COPY] spectral deinterleaved_from_alias_cache window=%zu cache_frames=%zu cache_values=%zu frames_committed=%zu values_written(scalars)=%zu values_written(complex)=%zu aliased=%d\n",
                    cache_window, cache_frames, cache_values, result.frames_committed, result.values_written, values_written_complex,
                    result.aliased_legacy_buffer ? 1 : 0);
                fflush(stdout);
                return result;
            }

            // Non-aliased caches: copy per-component from each buffer into
            // legacy arrays.
            for (size_t i = 0; i < copy_values; ++i) {
                legacy_real[i] = real_buffer.cache_data[i];
                legacy_imag[i] = imag_buffer.cache_data[i];
            }
                // `copy_values` counts per-component values copied into real (and imag).
                // So `copy_values` equals the number of complex bins written.
                result.values_written = copy_values; // complex bins
                values_written_complex = copy_values;
                values_written_scalars = copy_values * 2U;
                // frames_committed computed from complex bins per row (copy_values / window)
                result.frames_committed = (cache_window > 0U) ? (copy_values / cache_window) : 0U;
                result.copied_from_mailbox = (result.values_written > 0U);
                fprintf(stdout, "[MAILBOX-COPY] spectral copied_from_cache window=%zu cache_frames=%zu cache_values=%zu frames_committed=%zu values_written(scalars)=%zu values_written(complex)=%zu aliased=%d\n",
                    cache_window, cache_frames, cache_values, result.frames_committed, result.values_written, values_written_complex, result.aliased_legacy_buffer ? 1 : 0);
            fflush(stdout);
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
            node->frame_index >= 0) {

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
                    // we wrote `bins_to_copy` complex bins (each bin contributes
                    // one value in real and one in imag). Count complex bins.
                    result.values_written += bins_to_copy;
                    values_written_complex += bins_to_copy;
                    values_written_scalars += (bins_to_copy * 2U);
                    result.frames_committed = std::max(result.frames_committed, frame + 1);
                }
                result.frames_committed = std::max(result.frames_committed, frame + 1);
            }
        } // i hurt so much.
        node = node->next;
    }
    // Normalize/interpret frames_committed from scalar counts across real+imag
    if (values_written_complex > 0U) {
        // compute frames from complex-bin counts (bins per row = window)
        size_t computed_frames = 0U;
        if (window > 0U) {
            computed_frames = values_written_complex / window; // complex bins per row
        }
        size_t interpreted = result.frames_committed;
        if (computed_frames > interpreted) interpreted = computed_frames;
        if (interpreted > capacity_frames) interpreted = capacity_frames;
        result.frames_committed = interpreted;
        result.copied_from_mailbox = true;
        fprintf(stdout, "[MAILBOX-COPY] spectral copied frames_committed=%zu values_written(scalars)=%zu values_written(complex)=%zu aliased=%d\n",
            result.frames_committed, values_written_scalars, values_written_complex, result.aliased_legacy_buffer ? 1 : 0);
        fflush(stdout);
    } else {
        fprintf(stdout, "[MAILBOX-COPY] spectral nothing_copied\n");
        fflush(stdout);
    }
    return result;
}

}  // namespace amp::tests::fft_division_shared

#if defined(_MSC_VER)
#pragma pop_macro("min")
#pragma pop_macro("max")
#endif
#endif  // __cplusplus

#endif  // AMP_NATIVE_NODES_FFT_DIVISION_MAILBOX_HELPERS_H_
