
#ifndef AMP_NATIVE_MAILBOX_CHAIN_HPP
#define AMP_NATIVE_MAILBOX_CHAIN_HPP

#ifdef __cplusplus
#include "amp_native.h"
#include <cstddef>
#include <vector>

namespace amp { namespace tests { namespace fft_division_shared {

// Persistent, non-consuming mailbox node for tap output chain, now holds frame values directly
struct PersistentMailboxNode {
    PersistentMailboxNode* next = nullptr;
    // Node carries exactly one payload. Use `node_kind` to know which.
    std::vector<double> spectral_real_bins{};
    std::vector<double> spectral_imag_bins{};
    double spectral_real = 0.0;
    double spectral_imag = 0.0;
    double pcm_sample = 0.0;
    enum class NodeKind : int { SPECTRAL = 0, PCM = 1 };
    NodeKind node_kind = NodeKind::SPECTRAL;
    int slot = 0;
    int frame_index = 0;
    int window_size = 1;

    PersistentMailboxNode() = default;
    PersistentMailboxNode(const double* real, const double* imag, int slot_, int frame_idx, int window_size_)
        : next(nullptr), slot(slot_), frame_index(frame_idx) {
        node_kind = NodeKind::SPECTRAL;
        pcm_sample = 0.0;
        window_size = window_size_ > 0 ? window_size_ : 1;
        spectral_real_bins.assign(
            real ? real : nullptr,
            real ? real + static_cast<size_t>(window_size) : nullptr
        );
        spectral_imag_bins.assign(
            imag ? imag : nullptr,
            imag ? imag + static_cast<size_t>(window_size) : nullptr
        );
        if (!spectral_real_bins.empty()) spectral_real = spectral_real_bins[0];
        if (!spectral_imag_bins.empty()) spectral_imag = spectral_imag_bins[0];
    }
    PersistentMailboxNode(const void* data_ptr, size_t size)
        : next(nullptr), slot(0), frame_index(0), window_size(static_cast<int>(size)) {
        const double* d = reinterpret_cast<const double*>(data_ptr);
        spectral_real_bins.assign(
            d ? d : nullptr,
            d ? d + static_cast<size_t>(size) : nullptr
        );
        spectral_imag_bins.assign(spectral_real_bins.size(), 0.0);
        if (!spectral_real_bins.empty()) spectral_real = spectral_real_bins[0];
        if (!spectral_imag_bins.empty()) spectral_imag = spectral_imag_bins[0];
    }
    // PCM constructor - creates a node that carries a single PCM sample
    PersistentMailboxNode(double pcm, int frame_idx)
        : next(nullptr), spectral_real{0.0}, spectral_imag{0.0}, pcm_sample(pcm), slot(0), frame_index(frame_idx), window_size(1), node_kind(NodeKind::PCM) {}
};

// Simple head/tail struct for mailbox chains
struct MailboxChainHead {
    PersistentMailboxNode* head = nullptr;
    PersistentMailboxNode* tail = nullptr;
};

class EdgeRunnerTapMailboxChain {
public:
    static const PersistentMailboxNode* get_head(const EdgeRunnerTapBuffer& buf) {
        return reinterpret_cast<const PersistentMailboxNode*>(buf.mailbox_head);
    }
    static PersistentMailboxNode* get_head(EdgeRunnerTapBuffer& buf) {
        return reinterpret_cast<PersistentMailboxNode*>(buf.mailbox_head);
    }
    static void set_head(EdgeRunnerTapBuffer& buf, PersistentMailboxNode* head) {
        buf.mailbox_head = head;
    }
    static PersistentMailboxNode* get_node_by_index(PersistentMailboxNode* head, size_t index) {
        PersistentMailboxNode* node = head;
        for (size_t i = 0; node && i < index; ++i) {
            node = node->next;
        }
        return node;
    }
    static const PersistentMailboxNode* get_node_by_index(const PersistentMailboxNode* head, size_t index) {
        const PersistentMailboxNode* node = head;
        for (size_t i = 0; node && i < index; ++i) {
            node = node->next;
        }
        return node;
    }
    static size_t count_nodes(const PersistentMailboxNode* head) {
        size_t count = 0;
        for (const PersistentMailboxNode* node = head; node; node = node->next) {
            ++count;
        }
        return count;
    }
};

}}} // namespace amp::tests::fft_division_shared
#endif // __cplusplus

#endif // AMP_NATIVE_MAILBOX_CHAIN_HPP
