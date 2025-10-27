#ifndef AMP_DESCRIPTOR_BUILDER_H
#define AMP_DESCRIPTOR_BUILDER_H

#include <stddef.h>
#include <stdint.h>

#include "amp_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AmpDescriptorBuffer {
    uint8_t *data;
    size_t size;
    size_t capacity;
} AmpDescriptorBuffer;

typedef struct AmpDescriptorParam {
    const char *name;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    const double *values;
    size_t value_count;
} AmpDescriptorParam;

typedef struct AmpDescriptorBuilder {
    AmpDescriptorBuffer *buffer;
    size_t count_offset;
    uint32_t node_count;
    int finalized;
} AmpDescriptorBuilder;

AMP_CAPI void amp_descriptor_buffer_init(AmpDescriptorBuffer *buffer);
AMP_CAPI void amp_descriptor_buffer_destroy(AmpDescriptorBuffer *buffer);

AMP_CAPI int amp_descriptor_builder_init(AmpDescriptorBuilder *builder, AmpDescriptorBuffer *buffer);
AMP_CAPI int amp_descriptor_builder_append_node(
    AmpDescriptorBuilder *builder,
    const char *name,
    const char *type_name,
    const char *const *audio_inputs,
    uint32_t audio_input_count,
    const char *params_json,
    const AmpDescriptorParam *params,
    uint32_t param_count
);
AMP_CAPI int amp_descriptor_builder_finalize(AmpDescriptorBuilder *builder);

#ifdef __cplusplus
}
#endif

#endif /* AMP_DESCRIPTOR_BUILDER_H */
