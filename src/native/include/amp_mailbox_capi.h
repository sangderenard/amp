#ifndef AMP_MAILBOX_CAPI_H
#define AMP_MAILBOX_CAPI_H

#include "amp_native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque mailbox node pointer
typedef void* AmpMailboxNode;

// Create/append spectral node to a tap buffer
AmpMailboxNode amp_mailbox_create_spectral_node(const double* real, const double* imag, int slot, int frame_index);
void amp_mailbox_append_node_to_tap(EdgeRunnerTapBuffer* tap_buf, AmpMailboxNode node);

// Attach spectral node to per-state tap chain and update cursor
void amp_mailbox_attach_spectral_node(void* state, const char* tap_name, AmpMailboxNode node);

// Create/append PCM node to per-state PCM chain
AmpMailboxNode amp_mailbox_create_pcm_node(double value, int frame_index);
void amp_mailbox_append_pcm_node(void* state, AmpMailboxNode node);

// Cursor helpers
void amp_mailbox_set_tap_cursor(void* state, const char* tap_name, AmpMailboxNode node);
AmpMailboxNode amp_mailbox_get_tap_cursor(void* state, const char* tap_name);

// Introspection / instrumentation
// Return the number of nodes currently attached to the spectral chain for `tap_name`.
size_t amp_mailbox_spectral_chain_length(void* state, const char* tap_name);
// Return the number of nodes currently in the PCM mailbox chain.
size_t amp_mailbox_pcm_chain_length(void* state);
// Log current mailbox stats for debugging (writes to stderr).
void amp_mailbox_log_stats(void* state);

// Accessors to obtain the head pointer for mailbox chains so taps can read.
// Returns an opaque `AmpMailboxNode` pointing to the first persistent node
// in the spectral chain for `tap_name`, or NULL if none.
AmpMailboxNode amp_mailbox_get_spectral_head(void* state, const char* tap_name);
// Returns an opaque `AmpMailboxNode` pointing to the first persistent PCM node
// in the per-state PCM chain, or NULL if none.
AmpMailboxNode amp_mailbox_get_pcm_head(void* state);

// Node accessors
const double* amp_mailbox_node_real(AmpMailboxNode node);
const double* amp_mailbox_node_imag(AmpMailboxNode node);
int amp_mailbox_node_window_size(AmpMailboxNode node);
int amp_mailbox_node_frame_index(AmpMailboxNode node);
// PCM helpers
int amp_mailbox_node_is_pcm(AmpMailboxNode node);
double amp_mailbox_node_pcm_sample(AmpMailboxNode node);
// Iterator helper: get the next node in the chain (or NULL)
AmpMailboxNode amp_mailbox_node_next(AmpMailboxNode node);
// Returns 1 if node is spectral (non-PCM)
int amp_mailbox_node_is_spectral(AmpMailboxNode node);
// Get the spectral real component value (first element) for single-item spectral nodes
double amp_mailbox_node_spectral_value(AmpMailboxNode node);

#ifdef __cplusplus
}
#endif

#endif // AMP_MAILBOX_CAPI_H
