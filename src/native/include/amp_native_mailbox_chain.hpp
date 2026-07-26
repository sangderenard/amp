
#ifndef AMP_NATIVE_MAILBOX_CHAIN_HPP
#define AMP_NATIVE_MAILBOX_CHAIN_HPP

#ifdef __cplusplus
#include "amp_native.h"
#include <cstddef>
#include <vector>

namespace amp { namespace tests { namespace fft_division_shared {

// Persistent, non-consuming mailbox node for tap output chain, now holds frame values directly
struct PersistentMailboxNode {
    enum class FifoValueKind : int { FIFO_DOUBLE = 0, FIFO_I64 = 1, FIFO_PTR = 2 };
    union FifoValue {
        constexpr FifoValue() : as_ptr(nullptr) {}
        double as_double;
        int64_t as_i64;
        void* as_ptr;
    };
    PersistentMailboxNode* next = nullptr;
    // Node carries exactly one payload. Use `node_kind` to know which.
    std::vector<double> spectral_real_bins{};
    std::vector<double> spectral_imag_bins{};
    double spectral_real = 0.0;
    double spectral_imag = 0.0;
    enum class NodeKind : int { SPECTRAL = 0, PCM = 1 };
    FifoValue fifo_value{};
    FifoValueKind fifo_value_kind = FifoValueKind::FIFO_DOUBLE;
    NodeKind node_kind = NodeKind::SPECTRAL;
    int slot = 0;
    int frame_index = 0;
    int window_size = 1;

    PersistentMailboxNode() = default;
    void set_fifo_value(FifoValueKind kind, double as_double, int64_t as_i64, void* as_ptr) {
        fifo_value_kind = kind;
        switch (kind) {
            case FifoValueKind::FIFO_DOUBLE:
                fifo_value.as_double = as_double;
                break;
            case FifoValueKind::FIFO_I64:
                fifo_value.as_i64 = as_i64;
                break;
            case FifoValueKind::FIFO_PTR:
                fifo_value.as_ptr = as_ptr;
                break;
        }
    }
    PersistentMailboxNode(const double* real, const double* imag, int slot_, int frame_idx, int window_size_)
        : next(nullptr), slot(slot_), frame_index(frame_idx) {
        node_kind = NodeKind::SPECTRAL;
        fifo_value_kind = FifoValueKind::FIFO_DOUBLE;
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
        fifo_value_kind = FifoValueKind::FIFO_DOUBLE;
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
    PersistentMailboxNode(double pcm, int frame_idx, FifoValueKind kind = FifoValueKind::FIFO_DOUBLE)
        : next(nullptr), spectral_real{0.0}, spectral_imag{0.0}, slot(0), frame_index(frame_idx), window_size(1), node_kind(NodeKind::PCM) {
        set_fifo_value(kind, pcm, 0, nullptr);
    }
    PersistentMailboxNode(int64_t value, int frame_idx, FifoValueKind kind = FifoValueKind::FIFO_I64)
        : next(nullptr), spectral_real{0.0}, spectral_imag{0.0}, slot(0), frame_index(frame_idx), window_size(1), node_kind(NodeKind::PCM) {
        set_fifo_value(kind, 0.0, value, nullptr);
    }
    PersistentMailboxNode(void* ptr, int frame_idx, FifoValueKind kind = FifoValueKind::FIFO_PTR)
        : next(nullptr), spectral_real{0.0}, spectral_imag{0.0}, slot(0), frame_index(frame_idx), window_size(1), node_kind(NodeKind::PCM) {
        set_fifo_value(kind, 0.0, 0, ptr);
    }
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

inline PersistentMailboxNode::FifoValueKind ToMailboxFifoValueKind(AmpFifoValueKind kind) {
    switch (kind) {
        case AMP_FIFO_VALUE_I64:
            return PersistentMailboxNode::FifoValueKind::FIFO_I64;
        case AMP_FIFO_VALUE_PTR:
            return PersistentMailboxNode::FifoValueKind::FIFO_PTR;
        case AMP_FIFO_VALUE_DOUBLE:
        default:
            return PersistentMailboxNode::FifoValueKind::FIFO_DOUBLE;
    }
}

inline AmpFifoValueKind ToAmpFifoValueKind(PersistentMailboxNode::FifoValueKind kind) {
    switch (kind) {
        case PersistentMailboxNode::FifoValueKind::FIFO_I64:
            return AMP_FIFO_VALUE_I64;
        case PersistentMailboxNode::FifoValueKind::FIFO_PTR:
            return AMP_FIFO_VALUE_PTR;
        case PersistentMailboxNode::FifoValueKind::FIFO_DOUBLE:
        default:
            return AMP_FIFO_VALUE_DOUBLE;
    }
}

}}} // namespace amp::tests::fft_division_shared
#endif // __cplusplus

#endif // AMP_NATIVE_MAILBOX_CHAIN_HPP
