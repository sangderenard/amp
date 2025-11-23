#include "amp_mailbox_capi.h"
#ifdef __cplusplus

#include "amp_native_mailbox_chain.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include "nodes/fft_division/fft_division_types.h"

static std::mutex g_states_mtx;
static std::unordered_map<void*, struct MailboxStateWrapper*> g_states;
// Map of tap -> allocated buffer for buffers the mailbox allocated on behalf
// of blocking readers. These buffers are owned by the mailbox until the
// tap is marked READ, at which point mailbox will free them. If the tap
// is marked ACCEPTED the caller is assumed to take ownership and mailbox
// will not free the buffer.
static std::unordered_map<EdgeRunnerTapBuffer*, double*> g_owned_cache_buffers;

using namespace amp::tests::fft_division_shared;

struct MailboxStateWrapper {
    std::map<std::string, MailboxChainHead> spectral_mailbox_chains;
    MailboxChainHead pcm_mailbox_chain;
    std::map<std::string, FftDivTapMailboxCursor> tap_mailbox_cursors;
    std::mutex mtx;
    std::condition_variable cv;
    std::unordered_map<std::string, size_t> spectral_counts;
    size_t pcm_count{0};
    // Per-state PCM read cursor used by blocking readers so they can advance
    // independently of other consumers. When null the reader will observe
    // the chain head as the start.
    PersistentMailboxNode* pcm_read_cursor{nullptr};
    // instrumentation counters
    size_t total_spectral_appends{0};
    size_t total_pcm_appends{0};
};

static size_t expected_nodes_for_tap(const EdgeRunnerTapBuffer* tap) {
    if (!tap) return 0;
    const bool is_pcm = (tap->buffer_class && std::strcmp(tap->buffer_class, "pcm") == 0);
    // `frames` represents the consumer-visible number of rows the tap
    // expects. For PCM each node contributes one scalar sample; for
    // spectral taps each mailbox node represents a full spectral row
    // (window_size values) and thus counts as a single node. The
    // mailbox predicate compares node counts, so the expected value
    // must be in units of nodes (rows), not raw sample values.
    const size_t frames = tap->cache_frames > 0 ? tap->cache_frames : (tap->shape.frames > 0 ? tap->shape.frames : 1U);
    const size_t channels = tap->cache_channels > 0 ? tap->cache_channels : (tap->shape.channels > 0 ? tap->shape.channels : 1U);
    const size_t expected = frames; // number of mailbox nodes (rows) required
    return expected > 0 ? expected : 1U;
}

static size_t tap_chain_length_locked(
    MailboxStateWrapper* w,
    const EdgeRunnerTapBuffer* tap,
    const char* tap_name,
    std::string* resolved_name_out,
    MailboxChainHead** chain_out) {
    if (!w) return 0;
    const bool is_pcm = tap && tap->buffer_class && std::strcmp(tap->buffer_class, "pcm") == 0;
    if (is_pcm) {
        if (chain_out) *chain_out = &w->pcm_mailbox_chain;
        // If a per-state read cursor exists, report the number of nodes
        // available from that cursor to the tail. Otherwise return the
        // total pcm_count.
        if (w->pcm_read_cursor) {
            return EdgeRunnerTapMailboxChain::count_nodes(w->pcm_read_cursor);
        }
        return w->pcm_count;
    }
    std::string resolved_name;
    if (tap && tap->tap_name) {
        resolved_name = std::string(tap->tap_name);
    }
    if (resolved_name.empty() && tap_name) {
        resolved_name = std::string(tap_name);
    }
    if (resolved_name.empty()) {
        resolved_name = "spectral";
    }
    if (resolved_name_out) {
        *resolved_name_out = resolved_name;
    }
    auto it = w->spectral_counts.find(resolved_name);
    if (it != w->spectral_counts.end()) {
        if (chain_out) {
            *chain_out = &w->spectral_mailbox_chains[resolved_name];
        }
        return it->second;
    }
    auto it_chain = w->spectral_mailbox_chains.find(resolved_name);
    if (it_chain == w->spectral_mailbox_chains.end()) {
        if (chain_out) *chain_out = nullptr;
        return 0;
    }
    if (chain_out) {
        *chain_out = &it_chain->second;
    }
    return EdgeRunnerTapMailboxChain::count_nodes(it_chain->second.head);
}

static MailboxStateWrapper* state_for(void* key) {
    std::lock_guard<std::mutex> lock(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    MailboxStateWrapper* w = new MailboxStateWrapper();
    g_states[key] = w;
    return w;
}

AmpMailboxNode amp_mailbox_create_spectral_node(const double* real, const double* imag, int slot, int frame_index, int window_size) {
    auto* node = new PersistentMailboxNode(real, imag, slot, frame_index, window_size);
    return reinterpret_cast<AmpMailboxNode>(node);
}

void amp_mailbox_append_node_to_tap(EdgeRunnerTapBuffer* tap_buf, AmpMailboxNode node) {
    if (!tap_buf || !node) return;
    // If an external legacy buffer was provided via `tap_buf->data` but not
    // yet staged in the tap cache, treat that pointer as the staged cache
    // and initialize cache metadata based on the tap shape/frame_stride.
    if (tap_buf->data != nullptr && tap_buf->cache_data == nullptr) {
        size_t stride = tap_buf->frame_stride;
        if (stride == 0) {
            size_t batches = static_cast<size_t>(std::max<uint32_t>(1U, tap_buf->shape.batches));
            size_t channels = static_cast<size_t>(std::max<uint32_t>(1U, tap_buf->shape.channels));
            stride = batches * channels;
        }
        size_t frames = static_cast<size_t>(std::max<uint32_t>(1U, tap_buf->shape.frames));
        size_t buf_len = stride * frames;
        // Stage the provided pointer as the tap cache (no allocation/free).
        amp_tap_cache_stage(tap_buf, tap_buf->data, buf_len, tap_buf->shape.batches, tap_buf->shape.channels, tap_buf->shape.frames);
    }
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    // If the tap already has a head, append to the tail by walking the list.
    PersistentMailboxNode* head = reinterpret_cast<PersistentMailboxNode*>(tap_buf->mailbox_head);
    if (!head) {
        tap_buf->mailbox_head = n;
    } else {
        PersistentMailboxNode* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = n;
    }
}

// Tap cache helpers -------------------------------------------------------
// These helpers stage an externally-provided buffer on the tap, allow the
// mailbox code or runtime to pack data into that buffer from the persistent
// mailbox chain, and expose simple state transitions (accepted/read).

extern "C" AMP_CAPI int amp_tap_cache_stage(EdgeRunnerTapBuffer* tap, double* buffer, size_t buffer_len, uint32_t batches, uint32_t channels, uint32_t frames) {
    if (!tap) return -1;
    tap->cache_data = buffer;
    tap->cache_buffer_len = buffer_len;
    tap->cache_batches = batches;
    tap->cache_channels = channels;
    tap->cache_frames = frames;
    if (tap->cache_frames == 0 && frames > 0) tap->cache_frames = frames;
    if (tap->frame_stride == 0) {
        size_t stride = static_cast<size_t>(std::max<uint32_t>(1U, tap->shape.batches)) * static_cast<size_t>(std::max<uint32_t>(1U, tap->shape.channels));
        tap->frame_stride = stride;
    }
    tap->cache_state = 0; // empty/staged but not yet filled
    return 0;
}

// Copy up to the capacity of the staged cache from the mailbox chain into
// the staged buffer. Returns number of doubles written, or negative on error.
extern "C" AMP_CAPI int amp_tap_cache_fill_from_chain(EdgeRunnerTapBuffer* tap) {
    if (!tap) return -1;
    if (!tap->cache_data) return -2; // no buffer staged
    if (!tap->mailbox_head) return 0; // nothing to copy

    PersistentMailboxNode* cur = reinterpret_cast<PersistentMailboxNode*>(tap->mailbox_head);
    size_t written = 0;
    size_t frames_written = 0;
    size_t capacity = tap->cache_buffer_len;
    const size_t window = tap->shape.channels > 0 ? static_cast<size_t>(tap->shape.channels) : 1U;
    const char *tname = tap->tap_name;
    const char *bclass = tap->buffer_class;
    const bool prefer_imag = (tname != nullptr && std::strstr(tname, "imag") != nullptr) ||
        (bclass != nullptr && std::strstr(bclass, "imag") != nullptr);
    const bool prefer_real = (tname != nullptr && std::strstr(tname, "real") != nullptr) ||
        (bclass != nullptr && std::strstr(bclass, "real") != nullptr);
    // Pack each mailbox node contiguously: spectral nodes contribute full window bins,
    // PCM nodes contribute a single sample.
    while (cur && written < capacity) {
        if (cur->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) {
            const size_t node_window = static_cast<size_t>(cur->window_size > 0 ? cur->window_size : 1);
            const size_t bins_to_copy = std::min(window, node_window);
            const auto &real_bins = cur->spectral_real_bins;
            const auto &imag_bins = cur->spectral_imag_bins;

            if (prefer_imag) {
                if (written + bins_to_copy > capacity) break;
                for (size_t bin = 0; bin < bins_to_copy; ++bin) {
                    const double imag_val = bin < imag_bins.size() ? imag_bins[bin] : 0.0;
                    tap->cache_data[written++] = imag_val;
                }
            } else if (prefer_real) {
                if (written + bins_to_copy > capacity) break;
                for (size_t bin = 0; bin < bins_to_copy; ++bin) {
                    const double real_val = bin < real_bins.size() ? real_bins[bin] : 0.0;
                    tap->cache_data[written++] = real_val;
                }
            } else {
                const size_t required = bins_to_copy * 2U;
                if (written + required > capacity) break;
                for (size_t bin = 0; bin < bins_to_copy; ++bin) {
                    const double real_val = bin < real_bins.size() ? real_bins[bin] : 0.0;
                    const double imag_val = bin < imag_bins.size() ? imag_bins[bin] : 0.0;
                    tap->cache_data[written++] = real_val;
                    tap->cache_data[written++] = imag_val;
                }
            }
            ++frames_written;
        } else if (cur->node_kind == PersistentMailboxNode::NodeKind::PCM) {
            if (written + 1 <= capacity) {
                tap->cache_data[written++] = cur->pcm_sample;
            } else {
                break;
            }
        } else {
            // unknown node kind; stop to avoid corrupt writes
            break;
        }
        cur = cur->next;
    }
    tap->cache_state = (written > 0) ? 1 /*staged/filled*/ : 0;
    if (frames_written > 0) {
        const uint32_t frames_written_u32 = static_cast<uint32_t>(frames_written);
        tap->cache_frames = (tap->cache_frames > 0)
            ? std::max<uint32_t>(tap->cache_frames, frames_written_u32)
            : frames_written_u32;
    }
    // Report whether this buffer was mailbox-allocated (owned) so callers
    // can see ownership transitions in logs.
    bool owned = false;
    {
        std::lock_guard<std::mutex> lock(g_states_mtx);
        owned = g_owned_cache_buffers.find(tap) != g_owned_cache_buffers.end();
    }
    auto _now = std::chrono::steady_clock::now().time_since_epoch();
    long long _ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_now).count());
    fprintf(stderr, "[TAP-FILL] t=%lldms tap=%p cache=%p owned=%d written=%zu capacity=%zu from_head=%p\n",
            _ms, reinterpret_cast<void*>(tap), reinterpret_cast<void*>(tap->cache_data), owned ? 1 : 0, written, capacity, reinterpret_cast<void*>(tap->mailbox_head));
    return static_cast<int>(written);
}

extern "C" AMP_CAPI int amp_tap_cache_block_until_ready(
    void* state,
    EdgeRunnerTapBuffer* tap,
    const char* tap_name,
    uint64_t timeout_ms) {
    if (!tap) return -1;
    MailboxStateWrapper* w = state_for(state);
    const size_t expected = expected_nodes_for_tap(tap);
    if (expected == 0) return 0;

    auto _start_time = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(w->mtx);
    auto now_ms_since_start = [&]() -> long long {
        auto _now = std::chrono::steady_clock::now();
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_now - _start_time).count());
    };

    auto predicate = [&]() {
        return tap_chain_length_locked(w, tap, tap_name, nullptr, nullptr) >= expected;
    };
    bool ready = false;
    // Implement explicit wait loops so we can log the available length each check.
    if (timeout_ms == 0) {
        while (!predicate()) {
            size_t available_now = tap_chain_length_locked(w, tap, tap_name, nullptr, nullptr);
            long long _ms = now_ms_since_start();
            fprintf(stderr, "[TAP-WAIT-CHECK] t=%lldms tap=%p tap_name='%s' available=%zu expected=%zu\n",
                    _ms, reinterpret_cast<void*>(tap), tap_name ? tap_name : "(null)", available_now, expected);
            w->cv.wait(lock);
        }
        ready = true;
    } else {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!predicate()) {
            size_t available_now = tap_chain_length_locked(w, tap, tap_name, nullptr, nullptr);
            long long _ms = now_ms_since_start();
            fprintf(stderr, "[TAP-WAIT-CHECK] t=%lldms tap=%p tap_name='%s' available=%zu expected=%zu deadline_ms=%llu\n",
                    _ms, reinterpret_cast<void*>(tap), tap_name ? tap_name : "(null)", available_now, expected,
                    static_cast<unsigned long long>(timeout_ms));
            if (w->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                // timed out
                break;
            }
        }
        ready = predicate();
    }

    std::string resolved_name;
    MailboxChainHead* chain = nullptr;
    const size_t available = tap_chain_length_locked(w, tap, tap_name, &resolved_name, &chain);
    if (chain != nullptr) {
        // For spectral chains the chain pointer directly provides the head.
        tap->mailbox_head = chain->head;
    } else if (tap->buffer_class && std::strcmp(tap->buffer_class, "pcm") == 0) {
        // For PCM taps prefer the per-state read cursor if set so readers
        // observe a stateful window into the chain rather than the absolute
        // head which represents the entire global history.
        if (w->pcm_read_cursor) {
            tap->mailbox_head = w->pcm_read_cursor;
        } else {
            tap->mailbox_head = w->pcm_mailbox_chain.head;
        }
    }

    // Log initial tap/cache state right after resolving the mailbox head.
    bool initially_owned = false;
    {
        std::lock_guard<std::mutex> lock(g_states_mtx);
        initially_owned = g_owned_cache_buffers.find(tap) != g_owned_cache_buffers.end();
    }
    auto _now_init = std::chrono::steady_clock::now();
    long long _init_ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_now_init - _start_time).count());
    fprintf(stderr, "[TAP-BLOCK-INIT] t=%lldms tap=%p tap_name='%s' mailbox_head=%p cache=%p owned=%d expected=%zu available=%zu ready=%d\n",
            _init_ms, reinterpret_cast<void*>(tap), tap_name ? tap_name : "(null)", reinterpret_cast<void*>(tap->mailbox_head), reinterpret_cast<void*>(tap->cache_data), initially_owned ? 1 : 0, expected, available, ready ? 1 : 0);

    // If a staged legacy buffer exists on the tap, attempt to fill it from
    // the mailbox chain and wait briefly until that staged buffer contains
    // at least `expected` entries. This closes the race where the predicate
    // (which counts nodes) may be satisfied while the consumer-visible
    // staged cache is not yet populated.
    int filled = 0;

    // If no staged legacy buffer exists for this tap, allocate one now so
    // the blocking reader can be provided a consumer-visible buffer. The
    // allocation is registered in `g_owned_cache_buffers` and will be
    // freed when the cache is marked read/accepted.
    if (tap->cache_data == nullptr) {
        size_t stride = tap->frame_stride;
        if (stride == 0) {
            size_t batches = static_cast<size_t>(std::max<uint32_t>(1U, tap->shape.batches));
            size_t channels = static_cast<size_t>(std::max<uint32_t>(1U, tap->shape.channels));
            stride = batches * channels;
        }
        size_t frames = static_cast<size_t>(std::max<uint32_t>(1U, tap->shape.frames));
        size_t buf_len = stride * frames;
        if (buf_len > 0) {
            double *buf = nullptr;
            try {
                buf = new double[buf_len]();
            } catch (...) {
                buf = nullptr;
            }
            if (buf != nullptr) {
                tap->cache_data = buf;
                tap->cache_buffer_len = buf_len;
                tap->cache_batches = tap->shape.batches;
                tap->cache_channels = tap->shape.channels;
                tap->cache_frames = tap->shape.frames > 0 ? tap->shape.frames : static_cast<uint32_t>(frames);
                tap->cache_state = 0; // staged but empty
                g_owned_cache_buffers[tap] = buf;
                auto _now_alloc = std::chrono::steady_clock::now();
                long long _alloc_ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_now_alloc - _start_time).count());
                fprintf(stderr, "[TAP-ALLOC] t=%lldms tap_name='%s' tap=%p buf=%p len=%zu mailbox_owned=1\n",
                        _alloc_ms, tap_name ? tap_name : "(null)", reinterpret_cast<void*>(tap), reinterpret_cast<void*>(buf), buf_len);
            }
        }
    }

    if (tap->cache_data != nullptr) {
        // Try an immediate fill first.
        filled = amp_tap_cache_fill_from_chain(tap);
        // If we didn't fill enough, wait a short time and retry until
        // either filled >= expected or the caller's timeout window elapses.
        if (filled < static_cast<int>(expected) && ready) {
            // Give a small grace period where we re-check the condition.
            const auto fill_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
            int _retry_count = 0;
            while (std::chrono::steady_clock::now() < fill_deadline) {
                // Wait on the mailbox condition variable for small increments.
                w->cv.wait_for(lock, std::chrono::milliseconds(5));
                ++_retry_count;
                filled = amp_tap_cache_fill_from_chain(tap);
                auto _now_retry = std::chrono::steady_clock::now();
                long long _retry_ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_now_retry - _start_time).count());
                fprintf(stderr, "[TAP-FILL-RETRY] t=%lldms tap=%p tap_name='%s' retry=%d filled=%d expected=%zu\n",
                        _retry_ms, reinterpret_cast<void*>(tap), tap_name ? tap_name : "(null)", _retry_count, filled, expected);
                if (filled >= static_cast<int>(expected)) break;
            }
        }
    }

    auto _end_time = std::chrono::steady_clock::now();
    long long _elapsed_ms = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(_end_time - _start_time).count());
    fprintf(stderr, "[TAP-BLOCK] t=%lldms tap_name='%s' resolved='%s' expected=%zu available=%zu ready=%d filled=%d elapsed_ms=%lld\n",
            _elapsed_ms, tap_name ? tap_name : "(null)", resolved_name.c_str(), expected, available, ready ? 1 : 0, filled, _elapsed_ms);

    // Consider the tap ready only if the original predicate succeeded and,
    // when a staged buffer exists, we filled at least `expected` entries.
    const bool effective_ready = ready && (tap->cache_data == nullptr || filled >= static_cast<int>(expected));
    return effective_ready ? static_cast<int>(available) : 0;
}

extern "C" AMP_CAPI void amp_tap_cache_mark_accepted(EdgeRunnerTapBuffer* tap) {
    if (!tap) return;
    tap->cache_state = 2; // accepted by downstream (ownership transferred)
    // If we allocated this buffer on behalf of a blocking reader, free it
    // now that downstream has accepted it to avoid leaking memory. This
    // keeps allocation lifecycle local to the mailbox helpers.
    {
        // Do NOT free buffers here. The test harness or downstream code
        // that requested/owns the buffer is responsible for freeing it.
        // Retain any bookkeeping so ownership can be observed externally.
        std::lock_guard<std::mutex> lock(g_states_mtx);
    }
}

extern "C" AMP_CAPI void amp_tap_cache_mark_read(EdgeRunnerTapBuffer* tap) {
    if (!tap) return;
    tap->cache_state = 3; // read: buffer may be reused/destroyed in-place
    // Free any buffer we allocated for this tap when it is marked read.
    {
        std::lock_guard<std::mutex> lock(g_states_mtx);
        auto it = g_owned_cache_buffers.find(tap);
        if (it != g_owned_cache_buffers.end()) {
            double *ptr = it->second;
            delete [] ptr;
            g_owned_cache_buffers.erase(it);
            tap->cache_data = nullptr;
            tap->cache_buffer_len = 0;
            tap->cache_batches = 0;
            tap->cache_channels = 0;
            tap->cache_frames = 0;
            fprintf(stderr, "[TAP-FREE] read free tap=%p\n", reinterpret_cast<void*>(tap));
        }
    }
}

extern "C" AMP_CAPI double* amp_tap_cache_get_buffer(EdgeRunnerTapBuffer* tap) {
    if (!tap) return nullptr;
    return tap->cache_data;
}

extern "C" AMP_CAPI int amp_tap_cache_state(EdgeRunnerTapBuffer* tap) {
    if (!tap) return -1;
    return tap->cache_state;
}

void amp_mailbox_attach_spectral_node(void* state, const char* tap_name, AmpMailboxNode node) {
    if (!tap_name) return;
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    auto &chain = w->spectral_mailbox_chains[std::string(tap_name)];
    if (!chain.head) {
        chain.head = n;
        chain.tail = n;
    } else {
        chain.tail->next = n;
        chain.tail = n;
    }
    // No legacy cloning: append only to the explicit tap name provided.
    // Legacy compatibility (publishing clones into a canonical
    // "spectral_real" chain) has been removed so consumers should
    // subscribe to per-lane taps named `spectral_<N>`.
    // initialize cursor only if it doesn't already exist. We don't want to
    // advance or overwrite an existing read cursor when appending new nodes.
    auto &cursor = w->tap_mailbox_cursors[std::string(tap_name)];
    if (cursor.read_cursor == nullptr) {
        cursor.read_cursor = reinterpret_cast<const AmpSpectralMailboxEntry*>(n);
        cursor.last_frame_index = n ? n->frame_index : -1;
    }
    // update instrumentation
    w->spectral_counts[std::string(tap_name)] = EdgeRunnerTapMailboxChain::count_nodes(chain.head);
    ++w->total_spectral_appends;
    fprintf(stderr, "[MAILBOX-APPEND] spectral tap='%s' frame=%d appended total=%zu total_appends=%zu\n",
            tap_name ? tap_name : "(null)", n ? n->frame_index : -1,
            w->spectral_counts[std::string(tap_name)], w->total_spectral_appends);
    w->cv.notify_all();
}

AmpMailboxNode amp_mailbox_create_pcm_node(double value, int frame_index) {
    auto* node = new PersistentMailboxNode(value, frame_index);
    return reinterpret_cast<AmpMailboxNode>(node);
}

void amp_mailbox_append_pcm_node(void* state, AmpMailboxNode node) {
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    auto &chain = w->pcm_mailbox_chain;
    if (!chain.head) {
        chain.head = n;
        chain.tail = n;
    } else {
        chain.tail->next = n;
        chain.tail = n;
    }
    // update instrumentation
    w->pcm_count = EdgeRunnerTapMailboxChain::count_nodes(chain.head);
    ++w->total_pcm_appends;
    // Initialize per-state read cursor on first append if not present so
    // blocking readers that rely on a cursor will start at the chain head.
    if (w->pcm_read_cursor == nullptr) {
        w->pcm_read_cursor = chain.head;
    }
    fprintf(stderr, "[MAILBOX-APPEND] pcm frame=%d appended pcm_count=%zu total_pcm_appends=%zu\n",
            n ? n->frame_index : -1, w->pcm_count, w->total_pcm_appends);
    w->cv.notify_all();
    // Also notify the node worker (if any) so long-running workers that
    // wait on internal predicates can re-evaluate when mailbox entries
    // have been appended. This bridges the mailbox condition variable and
    // the node worker condition used by some node implementations.
    amp_fftdiv_notify_worker(state);
}

extern "C" AMP_CAPI int amp_mailbox_advance_pcm_cursor(void* state, const char* tap_name, size_t count) {
    MailboxStateWrapper* w = state_for(state);
    if (!w) return 0;
    std::lock_guard<std::mutex> lock(w->mtx);
    // Ensure cursor is initialized to chain head if not yet set
    if (w->pcm_read_cursor == nullptr) {
        w->pcm_read_cursor = w->pcm_mailbox_chain.head;
    }
    PersistentMailboxNode* cur = w->pcm_read_cursor;
    size_t advanced = 0;
    while (cur && advanced < count) {
        cur = cur->next;
        ++advanced;
    }
    w->pcm_read_cursor = cur;
    // Compute remaining available nodes from new cursor position
    size_t available = 0;
    if (w->pcm_read_cursor) {
        available = EdgeRunnerTapMailboxChain::count_nodes(w->pcm_read_cursor);
    }
    // Notify potential waiters that a cursor moved (not strictly necessary)
    fprintf(stderr, "[MAILBOX-CURSOR] tap='%s' advanced=%zu available=%zu new_cursor=%p\n",
            tap_name ? tap_name : "(pcm)", advanced, available, reinterpret_cast<void*>(w->pcm_read_cursor));
    w->cv.notify_all();
    // Wake the fft-division worker (if present) so it can re-evaluate
    // pipeline predicates that may depend on mailbox consumption.
    amp_fftdiv_notify_worker(state);
    return static_cast<int>(available);
}

size_t amp_mailbox_spectral_chain_length(void* state, const char* tap_name) {
    if (!tap_name) return 0;
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    auto it = w->spectral_counts.find(std::string(tap_name));
    if (it != w->spectral_counts.end()) return it->second;
    auto it2 = w->spectral_mailbox_chains.find(std::string(tap_name));
    if (it2 == w->spectral_mailbox_chains.end()) return 0;
    return EdgeRunnerTapMailboxChain::count_nodes(it2->second.head);
}

size_t amp_mailbox_pcm_chain_length(void* state) {
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    return w->pcm_count;
}

void amp_mailbox_log_stats(void* state) {
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    fprintf(stderr, "[MAILBOX-STATS] PCM chain length=%zu\n", w->pcm_count);
    fprintf(stderr, "[MAILBOX-STATS] total_pcm_appends=%zu total_spectral_appends=%zu\n", w->total_pcm_appends, w->total_spectral_appends);
    for (const auto &kv : w->spectral_mailbox_chains) {
        const std::string &tap = kv.first;
        size_t cnt = EdgeRunnerTapMailboxChain::count_nodes(kv.second.head);
        int last_frame = kv.second.tail ? kv.second.tail->frame_index : -1;
        fprintf(stderr, "[MAILBOX-STATS] spectral tap='%s' len=%zu tail_frame=%d\n", tap.c_str(), cnt, last_frame);
    }
}

AmpMailboxNode amp_mailbox_get_spectral_head(void* state, const char* tap_name) {
    if (!tap_name) return nullptr;
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    auto it = w->spectral_mailbox_chains.find(std::string(tap_name));
    if (it == w->spectral_mailbox_chains.end()) return nullptr;
    return reinterpret_cast<AmpMailboxNode>(it->second.head);
}

AmpMailboxNode amp_mailbox_get_pcm_head(void* state) {
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    return reinterpret_cast<AmpMailboxNode>(w->pcm_mailbox_chain.head);
}

AmpMailboxNode amp_mailbox_node_next(AmpMailboxNode node) {
    if (!node) return nullptr;
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    return reinterpret_cast<AmpMailboxNode>(n->next);
}

int amp_mailbox_node_is_spectral(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0;
    return n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL ? 1 : 0;
}

double amp_mailbox_node_spectral_value(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0.0;
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) {
        return n->spectral_real_bins.empty() ? 0.0 : n->spectral_real_bins.front();
    }
    return 0.0;
}

void amp_mailbox_set_tap_cursor(void* state, const char* tap_name, AmpMailboxNode node) {
    if (!tap_name) return;
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    auto &cursor = w->tap_mailbox_cursors[std::string(tap_name)];
    if (node) {
        PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
        cursor.read_cursor = reinterpret_cast<const AmpSpectralMailboxEntry*>(n);
        cursor.last_frame_index = n ? n->frame_index : -1;
    } else {
        cursor.read_cursor = nullptr;
        cursor.last_frame_index = -1;
    }
}

AmpMailboxNode amp_mailbox_get_tap_cursor(void* state, const char* tap_name) {
    if (!tap_name) return nullptr;
    MailboxStateWrapper* w = state_for(state);
    std::lock_guard<std::mutex> lock(w->mtx);
    auto it = w->tap_mailbox_cursors.find(std::string(tap_name));
    if (it == w->tap_mailbox_cursors.end()) return nullptr;
    const FftDivTapMailboxCursor &cursor = it->second;
    return reinterpret_cast<AmpMailboxNode>(const_cast<AmpSpectralMailboxEntry*>(cursor.read_cursor));
}

double amp_mailbox_node_real(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0.0;
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) {
        return n->spectral_real_bins.empty() ? 0.0 : n->spectral_real_bins.front();
    }
    return 0.0;
}
double amp_mailbox_node_imag(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0.0;
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) {
        return n->spectral_imag_bins.empty() ? 0.0 : n->spectral_imag_bins.front();
    }
    return 0.0;
}
int amp_mailbox_node_window_size(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    return n ? n->window_size : 0;
}
int amp_mailbox_node_frame_index(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    return n ? n->frame_index : 0;
}

int amp_mailbox_node_is_pcm(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0;
    return n->node_kind == PersistentMailboxNode::NodeKind::PCM ? 1 : 0;
}

double amp_mailbox_node_pcm_sample(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return 0.0;
    if (n->node_kind == PersistentMailboxNode::NodeKind::PCM) return n->pcm_sample;
    return 0.0;
}

#endif
