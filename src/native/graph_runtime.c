#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define AMP_API __declspec(dllexport)
#else
#define AMP_API
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct EdgeRunnerControlHistory EdgeRunnerControlHistory;
typedef EdgeRunnerControlHistory AmpGraphControlHistory;
typedef struct EdgeRunnerParamView EdgeRunnerParamView;
typedef struct EdgeRunnerNodeInputs EdgeRunnerNodeInputs;
typedef struct EdgeRunnerNodeDescriptor EdgeRunnerNodeDescriptor;

extern EdgeRunnerControlHistory *amp_load_control_history(const uint8_t *blob, size_t blob_len, int frames_hint);
extern void amp_release_control_history(EdgeRunnerControlHistory *history);
extern int amp_run_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history
);
extern void amp_free(double *buffer);
extern void amp_release_state(void *state);

typedef struct {
    double *data;
    size_t capacity;
    int in_use;
} buffer_pool_entry_t;

typedef struct {
    buffer_pool_entry_t *entries;
    size_t count;
} buffer_pool_t;

typedef struct {
    char *name;
    char *source_name;
    uint32_t source_index;
    double scale;
    int mode;  /* 0 -> add, 1 -> mul */
    int channel;
} mod_connection_t;

typedef struct {
    char *name;
    double *data;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
} param_binding_t;

typedef struct {
    char *name;
    size_t name_len;
    char *type_name;
    size_t type_name_len;
    uint32_t type_id;
    char **audio_inputs;
    uint32_t audio_input_count;
    uint32_t *audio_indices;
    mod_connection_t *mod_connections;
    uint32_t mod_connection_count;
    char *params_json;
    size_t params_json_len;
    uint32_t channel_hint;
    param_binding_t *param_bindings;
    size_t param_binding_count;
    void *state;
    double *output;
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
    EdgeRunnerNodeDescriptor descriptor;
} runtime_node_t;

typedef struct AmpGraphRuntime {
    runtime_node_t *nodes;
    uint32_t node_count;
    uint32_t *execution_order;
    uint32_t sink_index;
    buffer_pool_t pool;
    uint32_t default_batches;
    uint32_t default_frames;
} AmpGraphRuntime;

static char *dup_string(const char *src, size_t length) {
    if (src == NULL || length == 0) {
        return NULL;
    }
    char *dest = (char *)malloc(length + 1);
    if (dest == NULL) {
        return NULL;
    }
    memcpy(dest, src, length);
    dest[length] = '\0';
    return dest;
}

static void buffer_pool_destroy(buffer_pool_t *pool) {
    if (pool == NULL) {
        return;
    }
    if (pool->entries != NULL) {
        for (size_t i = 0; i < pool->count; ++i) {
            free(pool->entries[i].data);
            pool->entries[i].data = NULL;
            pool->entries[i].capacity = 0;
            pool->entries[i].in_use = 0;
        }
        free(pool->entries);
        pool->entries = NULL;
        pool->count = 0;
    }
}

static double *buffer_pool_acquire(buffer_pool_t *pool, size_t elements) {
    if (pool == NULL || elements == 0) {
        return NULL;
    }
    for (size_t i = 0; i < pool->count; ++i) {
        buffer_pool_entry_t *entry = &pool->entries[i];
        if (!entry->in_use && entry->capacity >= elements) {
            entry->in_use = 1;
            return entry->data;
        }
    }
    size_t new_index = pool->count;
    buffer_pool_entry_t *new_entries = (buffer_pool_entry_t *)realloc(
        pool->entries,
        (pool->count + 1) * sizeof(buffer_pool_entry_t)
    );
    if (new_entries == NULL) {
        return NULL;
    }
    pool->entries = new_entries;
    buffer_pool_entry_t *entry = &pool->entries[new_index];
    entry->data = (double *)malloc(elements * sizeof(double));
    if (entry->data == NULL) {
        return NULL;
    }
    entry->capacity = elements;
    entry->in_use = 1;
    pool->count += 1;
    return entry->data;
}

static void buffer_pool_reset(buffer_pool_t *pool) {
    if (pool == NULL) {
        return;
    }
    for (size_t i = 0; i < pool->count; ++i) {
        pool->entries[i].in_use = 0;
    }
}

static void buffer_pool_release(buffer_pool_t *pool, double *data) {
    if (pool == NULL || data == NULL) {
        return;
    }
    for (size_t i = 0; i < pool->count; ++i) {
        buffer_pool_entry_t *entry = &pool->entries[i];
        if (entry->data == data) {
            entry->in_use = 0;
            return;
        }
    }
}

static void destroy_runtime_node(runtime_node_t *node) {
    if (node == NULL) {
        return;
    }
    if (node->output != NULL) {
        amp_free(node->output);
        node->output = NULL;
    }
    if (node->state != NULL) {
        amp_release_state(node->state);
        node->state = NULL;
    }
    if (node->audio_inputs != NULL) {
        for (uint32_t i = 0; i < node->audio_input_count; ++i) {
            free(node->audio_inputs[i]);
        }
        free(node->audio_inputs);
    }
    free(node->audio_indices);
    if (node->mod_connections != NULL) {
        for (uint32_t i = 0; i < node->mod_connection_count; ++i) {
            free(node->mod_connections[i].name);
            free(node->mod_connections[i].source_name);
        }
        free(node->mod_connections);
    }
    if (node->param_bindings != NULL) {
        for (size_t i = 0; i < node->param_binding_count; ++i) {
            free(node->param_bindings[i].name);
            free(node->param_bindings[i].data);
        }
        free(node->param_bindings);
    }
    free(node->name);
    free(node->type_name);
    free(node->params_json);
}

static void release_runtime(AmpGraphRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    if (runtime->nodes != NULL) {
        for (uint32_t i = 0; i < runtime->node_count; ++i) {
            destroy_runtime_node(&runtime->nodes[i]);
        }
        free(runtime->nodes);
    }
    free(runtime->execution_order);
    buffer_pool_destroy(&runtime->pool);
    free(runtime);
}

static int read_u32(const uint8_t **cursor, size_t *remaining, uint32_t *out) {
    if (cursor == NULL || remaining == NULL || out == NULL) {
        return 0;
    }
    if (*remaining < 4) {
        return 0;
    }
    const uint8_t *ptr = *cursor;
    *out = (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) | ((uint32_t)ptr[2] << 16) | ((uint32_t)ptr[3] << 24);
    *cursor += 4;
    *remaining -= 4;
    return 1;
}

static int read_i32(const uint8_t **cursor, size_t *remaining, int32_t *out) {
    uint32_t value = 0;
    if (!read_u32(cursor, remaining, &value)) {
        return 0;
    }
    *out = (int32_t)value;
    return 1;
}

static int read_f32(const uint8_t **cursor, size_t *remaining, float *out) {
    if (cursor == NULL || remaining == NULL || out == NULL) {
        return 0;
    }
    if (*remaining < 4) {
        return 0;
    }
    uint32_t bits = 0;
    memcpy(&bits, *cursor, sizeof(uint32_t));
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(float));
    *cursor += 4;
    *remaining -= 4;
    *out = value;
    return 1;
}

static int json_find_key(const char *json, size_t json_len, const char *key, const char **value) {
    if (json == NULL || key == NULL || value == NULL) {
        return 0;
    }
    size_t key_len = strlen(key);
    const char *cursor = json;
    const char *end = json + json_len;
    while (cursor < end) {
        const char *found = strstr(cursor, key);
        if (found == NULL || found >= end) {
            break;
        }
        const char *colon = strchr(found + key_len, ':');
        if (colon == NULL || colon >= end) {
            break;
        }
        cursor = colon + 1;
        while (cursor < end && isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (cursor < end) {
            *value = cursor;
            return 1;
        }
    }
    return 0;
}

static double json_get_double(const char *json, size_t json_len, const char *key, double default_value) {
    const char *cursor = NULL;
    if (!json_find_key(json, json_len, key, &cursor)) {
        return default_value;
    }
    char *endptr = NULL;
    double value = strtod(cursor, &endptr);
    if (endptr == cursor) {
        return default_value;
    }
    return value;
}

static int json_get_int(const char *json, size_t json_len, const char *key, int default_value) {
    double value = json_get_double(json, json_len, key, (double)default_value);
    return (int)lrint(value);
}

static int32_t find_node_index(AmpGraphRuntime *runtime, const char *name) {
    if (runtime == NULL || name == NULL) {
        return -1;
    }
    for (uint32_t i = 0; i < runtime->node_count; ++i) {
        if (runtime->nodes[i].name != NULL && strcmp(runtime->nodes[i].name, name) == 0) {
            return (int32_t)i;
        }
    }
    return -1;
}

static runtime_node_t *find_node_by_name(AmpGraphRuntime *runtime, const char *name) {
    int32_t idx = find_node_index(runtime, name);
    if (idx < 0) {
        return NULL;
    }
    return &runtime->nodes[idx];
}

static int resolve_node_references(AmpGraphRuntime *runtime) {
    if (runtime == NULL) {
        return 0;
    }
    for (uint32_t i = 0; i < runtime->node_count; ++i) {
        runtime_node_t *node = &runtime->nodes[i];
        if (node->audio_input_count > 0U) {
            node->audio_indices = (uint32_t *)calloc(node->audio_input_count, sizeof(uint32_t));
            if (node->audio_indices == NULL) {
                return 0;
            }
            for (uint32_t j = 0; j < node->audio_input_count; ++j) {
                int32_t idx = find_node_index(runtime, node->audio_inputs[j]);
                if (idx < 0) {
                    return 0;
                }
                node->audio_indices[j] = (uint32_t)idx;
            }
        }
        for (uint32_t m = 0; m < node->mod_connection_count; ++m) {
            mod_connection_t *conn = &node->mod_connections[m];
            int32_t idx = find_node_index(runtime, conn->source_name);
            if (idx < 0) {
                return 0;
            }
            conn->source_index = (uint32_t)idx;
        }
        node->channel_hint = (uint32_t)json_get_int(node->params_json, node->params_json_len, "channels", 1);
        if (node->channel_hint == 0U) {
            node->channel_hint = 1U;
        }
        node->descriptor.name = node->name;
        node->descriptor.name_len = node->name_len;
        node->descriptor.type_name = node->type_name;
        node->descriptor.type_len = node->type_name_len;
        node->descriptor.params_json = node->params_json;
        node->descriptor.params_len = node->params_json_len;
    }
    return 1;
}

static int parse_node_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t blob_len) {
    if (runtime == NULL || blob == NULL) {
        return 0;
    }
    const uint8_t *cursor = blob;
    size_t remaining = blob_len;
    uint32_t node_count = 0;
    if (!read_u32(&cursor, &remaining, &node_count)) {
        return 0;
    }
    runtime->node_count = node_count;
    runtime->nodes = (runtime_node_t *)calloc(node_count, sizeof(runtime_node_t));
    if (runtime->nodes == NULL) {
        return 0;
    }
    for (uint32_t i = 0; i < node_count; ++i) {
        runtime_node_t *node = &runtime->nodes[i];
        uint32_t header[8];
        for (size_t h = 0; h < 8; ++h) {
            if (!read_u32(&cursor, &remaining, &header[h])) {
                return 0;
            }
        }
        uint32_t type_id = header[0];
        uint32_t name_len = header[1];
        uint32_t type_len = header[2];
        uint32_t audio_count = header[3];
        uint32_t mod_count = header[4];
        uint32_t param_buffer_count = header[5];
        uint32_t buffer_shape_count = header[6];
        uint32_t params_json_len = header[7];
        if (remaining < name_len + type_len) {
            return 0;
        }
        node->type_id = type_id;
        node->name = dup_string((const char *)cursor, name_len);
        node->name_len = name_len;
        cursor += name_len;
        remaining -= name_len;
        node->type_name = dup_string((const char *)cursor, type_len);
        node->type_name_len = type_len;
        cursor += type_len;
        remaining -= type_len;
        node->audio_input_count = audio_count;
        if (audio_count > 0U) {
            node->audio_inputs = (char **)calloc(audio_count, sizeof(char *));
            if (node->audio_inputs == NULL) {
                return 0;
            }
            for (uint32_t a = 0; a < audio_count; ++a) {
                uint32_t src_len = 0;
                if (!read_u32(&cursor, &remaining, &src_len)) {
                    return 0;
                }
                if (remaining < src_len) {
                    return 0;
                }
                node->audio_inputs[a] = dup_string((const char *)cursor, src_len);
                cursor += src_len;
                remaining -= src_len;
            }
        }
        node->mod_connection_count = mod_count;
        if (mod_count > 0U) {
            node->mod_connections = (mod_connection_t *)calloc(mod_count, sizeof(mod_connection_t));
            if (node->mod_connections == NULL) {
                return 0;
            }
            for (uint32_t m = 0; m < mod_count; ++m) {
                uint32_t source_len = 0;
                uint32_t param_len = 0;
                uint32_t mode_code = 0;
                float scale = 0.0f;
                int32_t channel = -1;
                if (!read_u32(&cursor, &remaining, &source_len) ||
                    !read_u32(&cursor, &remaining, &param_len) ||
                    !read_u32(&cursor, &remaining, &mode_code) ||
                    !read_f32(&cursor, &remaining, &scale) ||
                    !read_i32(&cursor, &remaining, &channel)) {
                    return 0;
                }
                if (remaining < source_len + param_len) {
                    return 0;
                }
                mod_connection_t *conn = &node->mod_connections[m];
                conn->source_name = dup_string((const char *)cursor, source_len);
                cursor += source_len;
                remaining -= source_len;
                conn->name = dup_string((const char *)cursor, param_len);
                cursor += param_len;
                remaining -= param_len;
                conn->scale = (double)scale;
                conn->mode = (mode_code == 1U) ? 1 : 0;
                conn->channel = (int)channel;
                conn->source_index = 0U;
            }
        }
        for (uint32_t p = 0; p < param_buffer_count; ++p) {
            uint32_t name_size = 0;
            if (!read_u32(&cursor, &remaining, &name_size)) {
                return 0;
            }
            int32_t dims[3];
            int32_t byte_len = 0;
            if (!read_i32(&cursor, &remaining, &dims[0]) ||
                !read_i32(&cursor, &remaining, &dims[1]) ||
                !read_i32(&cursor, &remaining, &dims[2]) ||
                !read_i32(&cursor, &remaining, &byte_len)) {
                return 0;
            }
            if (remaining < name_size + (size_t)byte_len) {
                return 0;
            }
            cursor += name_size + (size_t)byte_len;
            remaining -= name_size + (size_t)byte_len;
        }
        size_t shapes_bytes = (size_t)buffer_shape_count * 3U * sizeof(uint32_t);
        if (remaining < shapes_bytes) {
            return 0;
        }
        cursor += shapes_bytes;
        remaining -= shapes_bytes;
        if (remaining < params_json_len) {
            return 0;
        }
        node->params_json = dup_string((const char *)cursor, params_json_len);
        node->params_json_len = params_json_len;
        cursor += params_json_len;
        remaining -= params_json_len;
    }
    return resolve_node_references(runtime);
}

static int parse_plan_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t blob_len) {
    if (runtime == NULL || blob == NULL) {
        return 0;
    }
    if (blob_len < 12) {
        return 0;
    }
    if (memcmp(blob, "AMPL", 4) != 0) {
        return 0;
    }
    const uint8_t *cursor = blob + 4;
    size_t remaining = blob_len - 4;
    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!read_u32(&cursor, &remaining, &version) || !read_u32(&cursor, &remaining, &node_count)) {
        return 0;
    }
    if (node_count != runtime->node_count) {
        return 0;
    }
    runtime->execution_order = (uint32_t *)calloc(node_count, sizeof(uint32_t));
    if (runtime->execution_order == NULL) {
        return 0;
    }
    for (uint32_t i = 0; i < node_count; ++i) {
        uint32_t function_id = 0;
        uint32_t name_len = 0;
        uint32_t audio_offset = 0;
        uint32_t audio_span = 0;
        uint32_t param_count = 0;
        if (!read_u32(&cursor, &remaining, &function_id) ||
            !read_u32(&cursor, &remaining, &name_len) ||
            !read_u32(&cursor, &remaining, &audio_offset) ||
            !read_u32(&cursor, &remaining, &audio_span) ||
            !read_u32(&cursor, &remaining, &param_count)) {
            return 0;
        }
        if (remaining < name_len) {
            return 0;
        }
        char *name = dup_string((const char *)cursor, name_len);
        cursor += name_len;
        remaining -= name_len;
        for (uint32_t p = 0; p < param_count; ++p) {
            uint32_t param_name_len = 0;
            uint32_t offset = 0;
            uint32_t span = 0;
            if (!read_u32(&cursor, &remaining, &param_name_len) ||
                !read_u32(&cursor, &remaining, &offset) ||
                !read_u32(&cursor, &remaining, &span)) {
                free(name);
                return 0;
            }
            if (remaining < param_name_len) {
                free(name);
                return 0;
            }
            cursor += param_name_len;
            remaining -= param_name_len;
        }
        int32_t idx = find_node_index(runtime, name);
        free(name);
        if (idx < 0 || function_id >= node_count) {
            return 0;
        }
        runtime->execution_order[function_id] = (uint32_t)idx;
    }
    runtime->sink_index = runtime->node_count > 0U ? runtime->execution_order[runtime->node_count - 1U] : 0U;
    (void)version;
    return 1;
}

static param_binding_t *find_param_binding(runtime_node_t *node, const char *name) {
    if (node == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < node->param_binding_count; ++i) {
        if (node->param_bindings[i].name != NULL && strcmp(node->param_bindings[i].name, name) == 0) {
            return &node->param_bindings[i];
        }
    }
    return NULL;
}

static double *copy_param_binding(buffer_pool_t *pool, const param_binding_t *binding) {
    if (pool == NULL || binding == NULL) {
        return NULL;
    }
    size_t total = (size_t)binding->batches * (size_t)binding->channels * (size_t)binding->frames;
    if (total == 0) {
        return NULL;
    }
    double *dest = buffer_pool_acquire(pool, total);
    if (dest == NULL) {
        return NULL;
    }
    memcpy(dest, binding->data, total * sizeof(double));
    return dest;
}

static int apply_mod_connections(
    AmpGraphRuntime *runtime,
    runtime_node_t *node,
    const char *param_name,
    double *buffer,
    size_t total,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
) {
    if (runtime == NULL || node == NULL || param_name == NULL || buffer == NULL) {
        return 0;
    }
    (void)total;
    for (uint32_t i = 0; i < node->mod_connection_count; ++i) {
        mod_connection_t *conn = &node->mod_connections[i];
        if (conn->name == NULL || strcmp(conn->name, param_name) != 0) {
            continue;
        }
        if (conn->source_index >= runtime->node_count) {
            return 0;
        }
        runtime_node_t *source = &runtime->nodes[conn->source_index];
        if (source->output == NULL) {
            return 0;
        }
        uint32_t src_batches = source->batches;
        uint32_t src_channels = source->channels;
        uint32_t src_frames = source->frames;
        if (src_batches != batches || src_frames != frames) {
            return 0;
        }
        const double *src = source->output;
        if (conn->channel >= 0) {
            if ((uint32_t)conn->channel >= src_channels) {
                return 0;
            }
            for (uint32_t b = 0; b < batches; ++b) {
                for (uint32_t c = 0; c < channels; ++c) {
                    for (uint32_t f = 0; f < frames; ++f) {
                        size_t dst_idx = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)frames + (size_t)f;
                        size_t src_idx = ((size_t)b * (size_t)src_channels + (size_t)conn->channel) * (size_t)frames + (size_t)f;
                        if (conn->mode == 0) {
                            buffer[dst_idx] += conn->scale * src[src_idx];
                        } else {
                            buffer[dst_idx] *= 1.0 + conn->scale * src[src_idx];
                        }
                    }
                }
            }
        } else {
            if (src_channels != channels && src_channels != 1U) {
                return 0;
            }
            for (uint32_t b = 0; b < batches; ++b) {
                for (uint32_t c = 0; c < channels; ++c) {
                    uint32_t src_c = src_channels == 1U ? 0U : c;
                    for (uint32_t f = 0; f < frames; ++f) {
                        size_t dst_idx = ((size_t)b * (size_t)channels + (size_t)c) * (size_t)frames + (size_t)f;
                        size_t src_idx = ((size_t)b * (size_t)src_channels + (size_t)src_c) * (size_t)frames + (size_t)f;
                        if (conn->mode == 0) {
                            buffer[dst_idx] += conn->scale * src[src_idx];
                        } else {
                            buffer[dst_idx] *= 1.0 + conn->scale * src[src_idx];
                        }
                    }
                }
            }
        }
    }
    return 1;
}

static int prepare_param_buffer(
    AmpGraphRuntime *runtime,
    runtime_node_t *node,
    const char *param_name,
    buffer_pool_t *pool,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames,
    double default_value,
    double **out_buffer
) {
    if (out_buffer == NULL) {
        return 0;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    if (total == 0) {
        return 0;
    }
    param_binding_t *binding = find_param_binding(node, param_name);
    double *buffer = NULL;
    if (binding != NULL) {
        if (binding->batches != batches || binding->channels != channels || binding->frames != frames) {
            return 0;
        }
        buffer = copy_param_binding(pool, binding);
    } else {
        buffer = buffer_pool_acquire(pool, total);
        if (buffer != NULL) {
            for (size_t i = 0; i < total; ++i) {
                buffer[i] = default_value;
            }
        }
    }
    if (buffer == NULL) {
        return 0;
    }
    if (!apply_mod_connections(runtime, node, param_name, buffer, total, batches, channels, frames)) {
        buffer_pool_release(pool, buffer);
        return 0;
    }
    *out_buffer = buffer;
    return 1;
}

static int collect_param_names(runtime_node_t *node, const char ***names_out, size_t *count_out) {
    if (node == NULL || names_out == NULL || count_out == NULL) {
        return 0;
    }
    size_t capacity = node->param_binding_count + node->mod_connection_count + 4U;
    if (capacity == 0) {
        *names_out = NULL;
        *count_out = 0;
        return 1;
    }
    const char **names = (const char **)calloc(capacity, sizeof(const char *));
    if (names == NULL) {
        return 0;
    }
    size_t count = 0;
    for (size_t i = 0; i < node->param_binding_count; ++i) {
        const char *name = node->param_bindings[i].name;
        if (name == NULL) {
            continue;
        }
        int seen = 0;
        for (size_t j = 0; j < count; ++j) {
            if (strcmp(names[j], name) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            names[count++] = name;
        }
    }
    for (uint32_t i = 0; i < node->mod_connection_count; ++i) {
        const char *name = node->mod_connections[i].name;
        if (name == NULL) {
            continue;
        }
        int seen = 0;
        for (size_t j = 0; j < count; ++j) {
            if (strcmp(names[j], name) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            if (count >= capacity) {
                const char **new_names = (const char **)realloc(names, (capacity + 4U) * sizeof(const char *));
                if (new_names == NULL) {
                    free(names);
                    return 0;
                }
                names = new_names;
                capacity += 4U;
            }
            names[count++] = name;
        }
    }
    *names_out = names;
    *count_out = count;
    return 1;
}

static int build_param_views(
    AmpGraphRuntime *runtime,
    runtime_node_t *node,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames,
    EdgeRunnerParamView **views_out,
    uint32_t *count_out,
    double ***owned_buffers_out,
    uint32_t *owned_count_out
) {
    if (runtime == NULL || node == NULL || views_out == NULL || count_out == NULL || owned_buffers_out == NULL || owned_count_out == NULL) {
        return 0;
    }
    const char **names = NULL;
    size_t name_count = 0;
    if (!collect_param_names(node, &names, &name_count)) {
        return 0;
    }
    if (name_count == 0) {
        *views_out = NULL;
        *count_out = 0;
        *owned_buffers_out = NULL;
        *owned_count_out = 0;
        free(names);
        return 1;
    }
    EdgeRunnerParamView *views = (EdgeRunnerParamView *)calloc(name_count, sizeof(EdgeRunnerParamView));
    if (views == NULL) {
        free(names);
        return 0;
    }
    double **owned = (double **)calloc(name_count, sizeof(double *));
    if (owned == NULL) {
        free(views);
        free(names);
        return 0;
    }
    uint32_t owned_count = 0;
    for (size_t i = 0; i < name_count; ++i) {
        const char *param_name = names[i];
        double default_value = json_get_double(node->params_json, node->params_json_len, param_name, 0.0);
        double *buffer = NULL;
        if (!prepare_param_buffer(runtime, node, param_name, &runtime->pool, batches, channels, frames, default_value, &buffer)) {
            free(owned);
            free(views);
            free(names);
            return 0;
        }
        views[i].name = param_name;
        views[i].batches = batches;
        views[i].channels = channels;
        views[i].frames = frames;
        views[i].data = buffer;
        owned[owned_count++] = buffer;
    }
    free(names);
    *views_out = views;
    *count_out = (uint32_t)name_count;
    *owned_buffers_out = owned;
    *owned_count_out = owned_count;
    return 1;
}

static void release_param_views(buffer_pool_t *pool, EdgeRunnerParamView *views, uint32_t count, double **owned, uint32_t owned_count) {
    (void)count;
    if (owned != NULL) {
        for (uint32_t i = 0; i < owned_count; ++i) {
            buffer_pool_release(pool, owned[i]);
        }
        free(owned);
    }
    free(views);
}

static double *merge_audio_inputs(
    AmpGraphRuntime *runtime,
    runtime_node_t *node,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames,
    double ***scratch_buffers,
    uint32_t *scratch_count
) {
    if (runtime == NULL || node == NULL || out_batches == NULL || out_channels == NULL || out_frames == NULL || scratch_buffers == NULL || scratch_count == NULL) {
        return NULL;
    }
    *scratch_buffers = NULL;
    *scratch_count = 0;
    if (node->audio_input_count == 0U) {
        *out_batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
        *out_channels = node->channel_hint > 0U ? node->channel_hint : 1U;
        *out_frames = runtime->default_frames > 0U ? runtime->default_frames : 1U;
        return NULL;
    }
    double **owned = NULL;
    uint32_t owned_count = 0;
    uint32_t batches = 0;
    uint32_t frames = 0;
    uint32_t total_channels = 0;
    for (uint32_t i = 0; i < node->audio_input_count; ++i) {
        uint32_t src_idx = node->audio_indices[i];
        if (src_idx >= runtime->node_count) {
            free(owned);
            return NULL;
        }
        runtime_node_t *source = &runtime->nodes[src_idx];
        if (source->output == NULL) {
            free(owned);
            return NULL;
        }
        if (i == 0) {
            batches = source->batches;
            frames = source->frames;
        } else {
            if (source->batches != batches || source->frames != frames) {
                free(owned);
                return NULL;
            }
        }
        total_channels += source->channels;
    }
    if (total_channels == 0U) {
        free(owned);
        return NULL;
    }
    if (node->audio_input_count == 1U) {
        runtime_node_t *source = &runtime->nodes[node->audio_indices[0]];
        *out_batches = source->batches;
        *out_channels = source->channels;
        *out_frames = source->frames;
        return source->output;
    }
    size_t total = (size_t)batches * (size_t)total_channels * (size_t)frames;
    double *buffer = buffer_pool_acquire(&runtime->pool, total);
    if (buffer == NULL) {
        free(owned);
        return NULL;
    }
    size_t offset = 0;
    for (uint32_t i = 0; i < node->audio_input_count; ++i) {
        runtime_node_t *source = &runtime->nodes[node->audio_indices[i]];
        size_t block = (size_t)source->batches * (size_t)source->channels * (size_t)source->frames;
        memcpy(buffer + offset, source->output, block * sizeof(double));
        offset += block;
    }
    owned = (double **)calloc(1, sizeof(double *));
    if (owned == NULL) {
        buffer_pool_release(&runtime->pool, buffer);
        return NULL;
    }
    owned[0] = buffer;
    *scratch_buffers = owned;
    *scratch_count = 1U;
    *out_batches = batches;
    *out_channels = total_channels;
    *out_frames = frames;
    return buffer;
}

AMP_API AmpGraphRuntime *amp_graph_runtime_create(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    AmpGraphRuntime *runtime = (AmpGraphRuntime *)calloc(1, sizeof(AmpGraphRuntime));
    if (runtime == NULL) {
        return NULL;
    }
    runtime->default_batches = 1U;
    runtime->default_frames = 0U;
    if (!parse_node_blob(runtime, descriptor_blob, descriptor_len)) {
        release_runtime(runtime);
        return NULL;
    }
    if (!parse_plan_blob(runtime, plan_blob, plan_len)) {
        release_runtime(runtime);
        return NULL;
    }
    return runtime;
}

AMP_API void amp_graph_runtime_destroy(AmpGraphRuntime *runtime) {
    release_runtime(runtime);
}

AMP_API int amp_graph_runtime_configure(AmpGraphRuntime *runtime, uint32_t batches, uint32_t frames) {
    if (runtime == NULL) {
        return -1;
    }
    runtime->default_batches = batches > 0U ? batches : 1U;
    runtime->default_frames = frames;
    return 0;
}

AMP_API void amp_graph_runtime_clear_params(AmpGraphRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    for (uint32_t i = 0; i < runtime->node_count; ++i) {
        runtime_node_t *node = &runtime->nodes[i];
        if (node->param_bindings != NULL) {
            for (size_t j = 0; j < node->param_binding_count; ++j) {
                free(node->param_bindings[j].name);
                free(node->param_bindings[j].data);
            }
            free(node->param_bindings);
            node->param_bindings = NULL;
            node->param_binding_count = 0;
        }
    }
}

AMP_API int amp_graph_runtime_set_param(
    AmpGraphRuntime *runtime,
    const char *node_name,
    const char *param_name,
    const double *data,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
) {
    if (runtime == NULL || node_name == NULL || param_name == NULL || data == NULL) {
        return -1;
    }
    runtime_node_t *node = find_node_by_name(runtime, node_name);
    if (node == NULL) {
        return -1;
    }
    size_t total = (size_t)batches * (size_t)channels * (size_t)frames;
    if (total == 0) {
        return -1;
    }
    double *copy = (double *)malloc(total * sizeof(double));
    if (copy == NULL) {
        return -1;
    }
    memcpy(copy, data, total * sizeof(double));
    param_binding_t *binding = find_param_binding(node, param_name);
    if (binding != NULL) {
        free(binding->data);
        binding->data = copy;
        binding->batches = batches;
        binding->channels = channels;
        binding->frames = frames;
        return 0;
    }
    param_binding_t *new_bindings = (param_binding_t *)realloc(
        node->param_bindings,
        (node->param_binding_count + 1U) * sizeof(param_binding_t)
    );
    if (new_bindings == NULL) {
        free(copy);
        return -1;
    }
    node->param_bindings = new_bindings;
    binding = &node->param_bindings[node->param_binding_count];
    binding->name = dup_string(param_name, strlen(param_name));
    if (binding->name == NULL) {
        free(copy);
        return -1;
    }
    binding->data = copy;
    binding->batches = batches;
    binding->channels = channels;
    binding->frames = frames;
    node->param_binding_count += 1U;
    return 0;
}

AMP_API int amp_graph_runtime_execute(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    if (runtime == NULL || out_buffer == NULL || out_batches == NULL || out_channels == NULL || out_frames == NULL) {
        return -1;
    }
    EdgeRunnerControlHistory *history = NULL;
    if (control_blob != NULL && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (history == NULL) {
            return -1;
        }
    }
    for (uint32_t i = 0; i < runtime->node_count; ++i) {
        runtime_node_t *node = &runtime->nodes[i];
        if (node->output != NULL) {
            amp_free(node->output);
            node->output = NULL;
        }
        node->batches = 0;
        node->channels = 0;
        node->frames = 0;
    }
    buffer_pool_reset(&runtime->pool);
    int status = 0;
    for (uint32_t order = 0; order < runtime->node_count; ++order) {
        uint32_t node_idx = runtime->execution_order[order];
        runtime_node_t *node = &runtime->nodes[node_idx];
        uint32_t batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
        uint32_t frames = runtime->default_frames > 0U ? runtime->default_frames : 1U;
        uint32_t input_channels = node->channel_hint > 0U ? node->channel_hint : 1U;
        double **audio_owned = NULL;
        uint32_t audio_owned_count = 0;
        double *audio_data = merge_audio_inputs(runtime, node, &batches, &input_channels, &frames, &audio_owned, &audio_owned_count);
        EdgeRunnerParamView *param_views = NULL;
        uint32_t param_count = 0;
        double **owned_buffers = NULL;
        uint32_t owned_count = 0;
        if (!build_param_views(runtime, node, batches, input_channels, frames, &param_views, &param_count, &owned_buffers, &owned_count)) {
            status = -1;
            if (audio_owned != NULL) {
                for (uint32_t i = 0; i < audio_owned_count; ++i) {
                    buffer_pool_release(&runtime->pool, audio_owned[i]);
                }
                free(audio_owned);
            }
            break;
        }
        EdgeRunnerNodeInputs inputs;
        memset(&inputs, 0, sizeof(inputs));
        if (audio_data != NULL) {
            inputs.audio.has_audio = 1;
            inputs.audio.batches = batches;
            inputs.audio.channels = input_channels;
            inputs.audio.frames = frames;
            inputs.audio.data = audio_data;
        } else {
            inputs.audio.has_audio = 0;
            inputs.audio.batches = batches;
            inputs.audio.channels = 0;
            inputs.audio.frames = frames;
            inputs.audio.data = NULL;
        }
        if (param_count > 0) {
            inputs.params.count = param_count;
            inputs.params.items = param_views;
        } else {
            inputs.params.count = 0;
            inputs.params.items = NULL;
        }
        double *out_ptr = NULL;
        int out_ch = 0;
        void *state = node->state;
        status = amp_run_node(
            &node->descriptor,
            &inputs,
            (int)batches,
            (int)input_channels,
            (int)frames,
            sample_rate,
            &out_ptr,
            &out_ch,
            &state,
            history
        );
        if (status != 0) {
            release_param_views(&runtime->pool, param_views, param_count, owned_buffers, owned_count);
            if (audio_owned != NULL) {
                for (uint32_t i = 0; i < audio_owned_count; ++i) {
                    buffer_pool_release(&runtime->pool, audio_owned[i]);
                }
                free(audio_owned);
            }
            break;
        }
        if (node->state != state) {
            if (node->state != NULL) {
                amp_release_state(node->state);
            }
            node->state = state;
        }
        node->output = out_ptr;
        node->batches = batches;
        node->channels = (uint32_t)out_ch;
        node->frames = frames;
        release_param_views(&runtime->pool, param_views, param_count, owned_buffers, owned_count);
        if (audio_owned != NULL) {
            for (uint32_t i = 0; i < audio_owned_count; ++i) {
                buffer_pool_release(&runtime->pool, audio_owned[i]);
            }
            free(audio_owned);
        }
    }
    if (history != NULL) {
        amp_release_control_history(history);
    }
    if (status != 0) {
        return status;
    }
    runtime_node_t *sink = &runtime->nodes[runtime->sink_index];
    if (sink->output == NULL) {
        return -1;
    }
    *out_buffer = sink->output;
    *out_batches = sink->batches;
    *out_channels = sink->channels;
    *out_frames = sink->frames;
    return 0;
}

AMP_API void amp_graph_runtime_buffer_free(double *buffer) {
    if (buffer != NULL) {
        amp_free(buffer);
    }
}

AMP_API AmpGraphControlHistory *amp_graph_history_load(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
) {
    return amp_load_control_history(blob, blob_len, frames_hint);
}

AMP_API void amp_graph_history_destroy(AmpGraphControlHistory *history) {
    amp_release_control_history(history);
}
