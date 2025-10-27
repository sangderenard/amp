#include "amp_descriptor_builder.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int amp_descriptor_buffer_reserve(AmpDescriptorBuffer *buffer, size_t additional) {
    if (buffer == NULL) {
        return -1;
    }
    if (additional == 0U) {
        return 0;
    }
    size_t required = buffer->size + additional;
    if (required <= buffer->capacity) {
        return 0;
    }
    size_t new_capacity = buffer->capacity ? buffer->capacity : 1024U;
    while (new_capacity < required) {
        size_t next = new_capacity << 1U;
        if (next <= new_capacity) {
            new_capacity = required;
            break;
        }
        new_capacity = next;
    }
    uint8_t *next_block = (uint8_t *)realloc(buffer->data, new_capacity);
    if (next_block == NULL) {
        return -1;
    }
    buffer->data = next_block;
    buffer->capacity = new_capacity;
    return 0;
}

static int amp_descriptor_buffer_append_bytes(AmpDescriptorBuffer *buffer, const void *data, size_t length) {
    if (buffer == NULL || (length > 0U && data == NULL)) {
        return -1;
    }
    if (length == 0U) {
        return 0;
    }
    if (amp_descriptor_buffer_reserve(buffer, length) != 0) {
        return -1;
    }
    memcpy(buffer->data + buffer->size, data, length);
    buffer->size += length;
    return 0;
}

static int amp_descriptor_buffer_append_u32(AmpDescriptorBuffer *buffer, uint32_t value) {
    uint8_t bytes[4];
    bytes[0] = (uint8_t)(value & 0xFFU);
    bytes[1] = (uint8_t)((value >> 8) & 0xFFU);
    bytes[2] = (uint8_t)((value >> 16) & 0xFFU);
    bytes[3] = (uint8_t)((value >> 24) & 0xFFU);
    return amp_descriptor_buffer_append_bytes(buffer, bytes, sizeof(bytes));
}

static int amp_descriptor_buffer_append_u64(AmpDescriptorBuffer *buffer, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = (uint8_t)((value >> (i * 8)) & 0xFFU);
    }
    return amp_descriptor_buffer_append_bytes(buffer, bytes, sizeof(bytes));
}

static int amp_descriptor_buffer_write_u32_at(AmpDescriptorBuffer *buffer, size_t offset, uint32_t value) {
    if (buffer == NULL || offset + 4U > buffer->size) {
        return -1;
    }
    buffer->data[offset + 0] = (uint8_t)(value & 0xFFU);
    buffer->data[offset + 1] = (uint8_t)((value >> 8) & 0xFFU);
    buffer->data[offset + 2] = (uint8_t)((value >> 16) & 0xFFU);
    buffer->data[offset + 3] = (uint8_t)((value >> 24) & 0xFFU);
    return 0;
}

void amp_descriptor_buffer_init(AmpDescriptorBuffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    buffer->data = NULL;
    buffer->size = 0U;
    buffer->capacity = 0U;
}

void amp_descriptor_buffer_destroy(AmpDescriptorBuffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    buffer->data = NULL;
    buffer->size = 0U;
    buffer->capacity = 0U;
}

int amp_descriptor_builder_init(AmpDescriptorBuilder *builder, AmpDescriptorBuffer *buffer) {
    if (builder == NULL || buffer == NULL) {
        return -1;
    }
    builder->buffer = buffer;
    builder->node_count = 0U;
    builder->finalized = 0;
    builder->count_offset = buffer->size;
    if (amp_descriptor_buffer_append_u32(buffer, 0U) != 0) {
        builder->buffer = NULL;
        builder->count_offset = 0U;
        return -1;
    }
    return 0;
}

static int amp_descriptor_append_string(AmpDescriptorBuffer *buffer, const char *str, uint32_t length) {
    if (length == 0U) {
        return 0;
    }
    if (str == NULL) {
        return -1;
    }
    return amp_descriptor_buffer_append_bytes(buffer, str, (size_t)length);
}

static int amp_descriptor_append_param(
    AmpDescriptorBuffer *buffer,
    const AmpDescriptorParam *param
) {
    if (param == NULL || buffer == NULL) {
        return -1;
    }
    if (param->name == NULL) {
        return -1;
    }
    size_t name_len = strlen(param->name);
    uint64_t blob_len = (param->values && param->value_count > 0U)
        ? (uint64_t)(param->value_count * sizeof(double))
        : 0U;

    if (amp_descriptor_buffer_append_u32(buffer, (uint32_t)name_len) != 0) return -1;
    if (amp_descriptor_buffer_append_u32(buffer, param->batches) != 0) return -1;
    if (amp_descriptor_buffer_append_u32(buffer, param->channels) != 0) return -1;
    if (amp_descriptor_buffer_append_u32(buffer, param->frames) != 0) return -1;
    if (amp_descriptor_buffer_append_u64(buffer, blob_len) != 0) return -1;
    if (amp_descriptor_append_string(buffer, param->name, (uint32_t)name_len) != 0) return -1;

    if (blob_len > 0U) {
        if (param->values == NULL) {
            return -1;
        }
        if (amp_descriptor_buffer_append_bytes(buffer, param->values, (size_t)blob_len) != 0) {
            return -1;
        }
    }
    return 0;
}

int amp_descriptor_builder_append_node(
    AmpDescriptorBuilder *builder,
    const char *name,
    const char *type_name,
    const char *const *audio_inputs,
    uint32_t audio_input_count,
    const char *params_json,
    const AmpDescriptorParam *params,
    uint32_t param_count
) {
    if (builder == NULL || builder->buffer == NULL || name == NULL || type_name == NULL) {
        return -1;
    }
    if (builder->finalized) {
        return -1;
    }
    if (audio_input_count > 0U && audio_inputs == NULL) {
        return -1;
    }
    AmpDescriptorBuffer *buffer = builder->buffer;
    size_t start_size = buffer->size;

    size_t name_len = strlen(name);
    size_t type_len = strlen(type_name);
    const char *json = params_json ? params_json : "";
    size_t json_len = strlen(json);

    if (amp_descriptor_buffer_append_u32(buffer, 0U) != 0) goto error; /* type_id */
    if (amp_descriptor_buffer_append_u32(buffer, (uint32_t)name_len) != 0) goto error;
    if (amp_descriptor_buffer_append_u32(buffer, (uint32_t)type_len) != 0) goto error;
    if (amp_descriptor_buffer_append_u32(buffer, audio_input_count) != 0) goto error;
    if (amp_descriptor_buffer_append_u32(buffer, 0U) != 0) goto error; /* mod_count */
    if (amp_descriptor_buffer_append_u32(buffer, param_count) != 0) goto error;
    if (amp_descriptor_buffer_append_u32(buffer, 0U) != 0) goto error; /* shape_count */
    if (amp_descriptor_buffer_append_u32(buffer, (uint32_t)json_len) != 0) goto error;

    if (amp_descriptor_append_string(buffer, name, (uint32_t)name_len) != 0) goto error;
    if (amp_descriptor_append_string(buffer, type_name, (uint32_t)type_len) != 0) goto error;

    for (uint32_t i = 0; i < audio_input_count; ++i) {
        const char *source = audio_inputs[i];
        if (source == NULL) goto error;
        size_t source_len = strlen(source);
        if (amp_descriptor_buffer_append_u32(buffer, (uint32_t)source_len) != 0) goto error;
        if (amp_descriptor_append_string(buffer, source, (uint32_t)source_len) != 0) goto error;
    }

    for (uint32_t p = 0; p < param_count; ++p) {
        if (amp_descriptor_append_param(buffer, &params[p]) != 0) goto error;
    }

    if (amp_descriptor_append_string(buffer, json, (uint32_t)json_len) != 0) goto error;

    builder->node_count += 1U;
    if (amp_descriptor_buffer_write_u32_at(buffer, builder->count_offset, builder->node_count) != 0) {
        goto error;
    }

    return 0;

error:
    /* Roll back payload */
    buffer->size = start_size;
    return -1;
}

int amp_descriptor_builder_finalize(AmpDescriptorBuilder *builder) {
    if (builder == NULL || builder->buffer == NULL) {
        return -1;
    }
    if (builder->finalized) {
        return 0;
    }
    if (amp_descriptor_buffer_write_u32_at(builder->buffer, builder->count_offset, builder->node_count) != 0) {
        return -1;
    }
    builder->finalized = 1;
    return 0;
}
