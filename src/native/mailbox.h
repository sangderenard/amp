#ifndef AMP_MAILBOX_H
#define AMP_MAILBOX_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "amp_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AmpMailboxReleaseFn)(void *);

typedef struct AmpMailboxEntry {
    struct AmpMailboxEntry *next;
    double *buffer;
    int channels;
    size_t frames;
    int status;
    AmpNodeMetrics metrics;
    void *context;
    AmpMailboxReleaseFn context_release;
} AmpMailboxEntry;

typedef struct AmpSpectralMailboxEntry {
    struct AmpSpectralMailboxEntry *next;
    double *spectral_real;
    double *spectral_imag;
    int slot;
    int frame_index;
    int window_size;
    int ref_count;
} AmpSpectralMailboxEntry;

typedef struct AmpMailbox {
    AmpMailboxEntry *head;
    AmpMailboxEntry *tail;
} AmpMailbox;

typedef struct AmpSpectralMailbox {
    AmpSpectralMailboxEntry *head;
    AmpSpectralMailboxEntry *tail;
} AmpSpectralMailbox;

static inline void amp_mailbox_init(AmpMailbox *mailbox) {
    if (mailbox != NULL) {
        mailbox->head = NULL;
        mailbox->tail = NULL;
    }
}

static inline AmpMailboxEntry *amp_mailbox_pop(AmpMailbox *mailbox) {
    if (mailbox == NULL || mailbox->head == NULL) {
        return NULL;
    }
    AmpMailboxEntry *entry = mailbox->head;
    mailbox->head = entry->next;
    if (mailbox->head == NULL) {
        mailbox->tail = NULL;
    }
    entry->next = NULL;
    return entry;
}

static inline void amp_mailbox_push(AmpMailbox *mailbox, AmpMailboxEntry *entry) {
    if (mailbox == NULL || entry == NULL) {
        return;
    }
    entry->next = NULL;
    if (mailbox->tail != NULL) {
        mailbox->tail->next = entry;
    } else {
        mailbox->head = entry;
    }
    mailbox->tail = entry;
}

static inline void amp_mailbox_unshift(AmpMailbox *mailbox, AmpMailboxEntry *entry) {
    if (mailbox == NULL || entry == NULL) {
        return;
    }
    entry->next = mailbox->head;
    mailbox->head = entry;
    if (mailbox->tail == NULL) {
        mailbox->tail = entry;
    }
}

static inline AmpMailboxEntry *amp_mailbox_peek(const AmpMailbox *mailbox) {
    return (mailbox != NULL) ? mailbox->head : NULL;
}

static inline AmpMailboxEntry *amp_mailbox_entry_create(
    double *buffer,
    int channels,
    size_t frames,
    int status,
    const AmpNodeMetrics *metrics,
    void *context,
    AmpMailboxReleaseFn release_fn
) {
    AmpMailboxEntry *entry = (AmpMailboxEntry *)calloc(1, sizeof(AmpMailboxEntry));
    if (entry == NULL) {
        return NULL;
    }
    if (metrics != NULL) {
        entry->metrics = *metrics;
    } else {
        memset(&entry->metrics, 0, sizeof(entry->metrics));
    }
    entry->buffer = buffer;
    entry->channels = channels;
    entry->frames = frames;
    entry->status = status;
    entry->context = context;
    entry->context_release = release_fn;
    entry->next = NULL;
    return entry;
}

static inline void amp_mailbox_entry_release(AmpMailboxEntry *entry) {
    if (entry == NULL) {
        return;
    }
    if (entry->context_release != NULL) {
        entry->context_release(entry->context);
    }
    free(entry);
}

static inline void amp_spectral_mailbox_init(AmpSpectralMailbox *mailbox) {
    if (mailbox != NULL) {
        mailbox->head = NULL;
        mailbox->tail = NULL;
    }
}

static inline AmpSpectralMailboxEntry *amp_spectral_mailbox_pop(AmpSpectralMailbox *mailbox) {
    if (mailbox == NULL || mailbox->head == NULL) {
        return NULL;
    }
    AmpSpectralMailboxEntry *entry = mailbox->head;
    mailbox->head = entry->next;
    if (mailbox->head == NULL) {
        mailbox->tail = NULL;
    }
    entry->next = NULL;
    return entry;
}

static inline void amp_spectral_mailbox_push(AmpSpectralMailbox *mailbox, AmpSpectralMailboxEntry *entry) {
    if (mailbox == NULL || entry == NULL) {
        return;
    }
    entry->next = NULL;
    if (mailbox->tail != NULL) {
        mailbox->tail->next = entry;
    } else {
        mailbox->head = entry;
    }
    mailbox->tail = entry;
}

static inline AmpSpectralMailboxEntry *amp_spectral_mailbox_entry_create(
    int slot,
    int frame_index,
    int window_size,
    const double *spectral_real,
    const double *spectral_imag
) {
    AmpSpectralMailboxEntry *entry = (AmpSpectralMailboxEntry *)calloc(1, sizeof(AmpSpectralMailboxEntry));
    if (entry == NULL) {
        return NULL;
    }
    entry->spectral_real = (double *)malloc((size_t)window_size * sizeof(double));
    entry->spectral_imag = (double *)malloc((size_t)window_size * sizeof(double));
    if (entry->spectral_real == NULL || entry->spectral_imag == NULL) {
        free(entry->spectral_real);
        free(entry->spectral_imag);
        free(entry);
        return NULL;
    }
    memcpy(entry->spectral_real, spectral_real, (size_t)window_size * sizeof(double));
    memcpy(entry->spectral_imag, spectral_imag, (size_t)window_size * sizeof(double));
    entry->slot = slot;
    entry->frame_index = frame_index;
    entry->window_size = window_size;
    entry->ref_count = 1;
    entry->next = NULL;
    return entry;
}

static inline void amp_spectral_mailbox_entry_release(AmpSpectralMailboxEntry *entry) {
    if (entry == NULL) {
        return;
    }
    entry->ref_count--;
    if (entry->ref_count <= 0) {
        free(entry->spectral_real);
        free(entry->spectral_imag);
        free(entry);
    }
}

AMP_CAPI AmpMailboxEntry *amp_node_mailbox_pop(void *state);
AMP_CAPI void amp_node_mailbox_push(void *state, AmpMailboxEntry *entry);
AMP_CAPI void amp_node_mailbox_clear(void *state);
AMP_CAPI AmpMailboxEntry *amp_node_mailbox_head(void *state);
AMP_CAPI AmpMailboxEntry *amp_node_mailbox_tail(void *state);
AMP_CAPI AmpSpectralMailboxEntry *amp_node_spectral_mailbox_pop(void *state);
AMP_CAPI void amp_node_spectral_mailbox_push(void *state, AmpSpectralMailboxEntry *entry);
AMP_CAPI void amp_node_spectral_mailbox_clear(void *state);

#ifdef __cplusplus
}
#endif

#endif  // AMP_MAILBOX_H
