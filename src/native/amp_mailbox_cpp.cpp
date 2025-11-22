#include "amp_mailbox_capi.h"
#ifdef __cplusplus

#include "amp_native_mailbox_chain.hpp"
#include <map>
#include <string>
#include <mutex>
#include <unordered_map>
#include "nodes/fft_division/fft_division_types.h"

static std::mutex g_states_mtx;
static std::unordered_map<void*, struct MailboxStateWrapper*> g_states;

using namespace amp::tests::fft_division_shared;

struct MailboxStateWrapper {
    std::map<std::string, MailboxChainHead> spectral_mailbox_chains;
    MailboxChainHead pcm_mailbox_chain;
    std::map<std::string, FftDivTapMailboxCursor> tap_mailbox_cursors;
    std::mutex mtx;
    std::unordered_map<std::string, size_t> spectral_counts;
    size_t pcm_count{0};
    // instrumentation counters
    size_t total_spectral_appends{0};
    size_t total_pcm_appends{0};
};

static MailboxStateWrapper* state_for(void* key) {
    std::lock_guard<std::mutex> lock(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    MailboxStateWrapper* w = new MailboxStateWrapper();
    g_states[key] = w;
    return w;
}

AmpMailboxNode amp_mailbox_create_spectral_node(const double* real, const double* imag, int slot, int frame_index) {
    auto* node = new PersistentMailboxNode(real, imag, slot, frame_index);
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
    size_t capacity = tap->cache_buffer_len;
    // Simple packing: write scalar values in sequence. For spectral nodes
    // write real then imag; for PCM nodes write single value.
    while (cur && written < capacity) {
        if (cur->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) {
            if (written + 2 <= capacity) {
                tap->cache_data[written++] = cur->spectral_real;
                tap->cache_data[written++] = cur->spectral_imag;
            } else {
                break;
            }
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
    return static_cast<int>(written);
}

extern "C" AMP_CAPI void amp_tap_cache_mark_accepted(EdgeRunnerTapBuffer* tap) {
    if (!tap) return;
    tap->cache_state = 2; // accepted by downstream (ownership transferred)
}

extern "C" AMP_CAPI void amp_tap_cache_mark_read(EdgeRunnerTapBuffer* tap) {
    if (!tap) return;
    tap->cache_state = 3; // read: buffer may be reused/destroyed in-place
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
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) return n->spectral_real;
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

const double* amp_mailbox_node_real(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return nullptr;
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) return &n->spectral_real;
    return nullptr;
}
const double* amp_mailbox_node_imag(AmpMailboxNode node) {
    PersistentMailboxNode* n = reinterpret_cast<PersistentMailboxNode*>(node);
    if (!n) return nullptr;
    if (n->node_kind == PersistentMailboxNode::NodeKind::SPECTRAL) return &n->spectral_imag;
    return nullptr;
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
