#include "kpn_graph_converter.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void amp_kpn_zero_graph(AmpKpnGraph *graph) {
    if (graph == NULL) {
        return;
    }
    graph->nodes = NULL;
    graph->node_count = 0;
    graph->edges = NULL;
    graph->edge_count = 0;
}

static char *amp_kpn_strdup(const char *text) {
    if (text == NULL) {
        return NULL;
    }
    size_t length = strlen(text);
    char *copy = (char *)malloc(length + 1U);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, text, length);
    copy[length] = '\0';
    return copy;
}

static void amp_kpn_write_error(char *buffer, size_t buffer_len, const char *message) {
    if (buffer == NULL || buffer_len == 0U) {
        return;
    }
    if (message == NULL) {
        buffer[0] = '\0';
        return;
    }
    (void)snprintf(buffer, buffer_len, "%s", message);
}

static bool amp_kpn_copy_taps(
    AmpKpnTapDefault **target,
    size_t count,
    const AmpTapSpec *specs,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (count == 0U) {
        *target = NULL;
        return true;
    }
    AmpKpnTapDefault *taps = (AmpKpnTapDefault *)calloc(count, sizeof(AmpKpnTapDefault));
    if (taps == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate tap array");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const AmpTapSpec *spec = &specs[i];
        if (spec->name == NULL || spec->name[0] == '\0') {
            amp_kpn_write_error(error_buffer, error_buffer_len, "tap specification missing name");
            for (size_t j = 0; j < i; ++j) {
                free(taps[j].name);
                free(taps[j].group);
            }
            free(taps);
            return false;
        }
        taps[i].name = amp_kpn_strdup(spec->name);
        taps[i].group = amp_kpn_strdup(spec->group);
        if ((spec->group != NULL && taps[i].group == NULL) || taps[i].name == NULL) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap metadata");
            for (size_t j = 0; j <= i; ++j) {
                free(taps[j].name);
                free(taps[j].group);
            }
            free(taps);
            return false;
        }
        taps[i].shape = spec->shape;
        taps[i].capacity_frames = spec->capacity_frames;
        taps[i].delivery_mode = spec->delivery_mode;
        taps[i].simultaneous_availability = spec->simultaneous_availability;
        taps[i].release_policy = spec->release_policy;
    }
    *target = taps;
    return true;
}

static bool amp_kpn_copy_subchannels(
    AmpKpnTapSubchannel **target,
    size_t count,
    const AmpTapSubchannelSpec *specs,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (count == 0U) {
        *target = NULL;
        return true;
    }
    AmpKpnTapSubchannel *subs = (AmpKpnTapSubchannel *)calloc(count, sizeof(AmpKpnTapSubchannel));
    if (subs == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate tap subchannel array");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const AmpTapSubchannelSpec *spec = &specs[i];
        subs[i].name = amp_kpn_strdup(spec->name);
        if (spec->name != NULL && subs[i].name == NULL) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy subchannel name");
            for (size_t j = 0; j <= i; ++j) {
                free(subs[j].name);
            }
            free(subs);
            return false;
        }
        subs[i].enabled = spec->enabled;
        subs[i].stride_elems = spec->stride_elems;
    }
    *target = subs;
    return true;
}

static bool amp_kpn_copy_string_array(
    char ***target,
    size_t *out_count,
    const char **source,
    size_t count,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (count == 0U) {
        *target = NULL;
        *out_count = 0U;
        return true;
    }
    char **copies = (char **)calloc(count, sizeof(char *));
    if (copies == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate string array");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (source[i] != NULL) {
            copies[i] = amp_kpn_strdup(source[i]);
            if (copies[i] == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy string entry");
                for (size_t j = 0; j < i; ++j) {
                    free(copies[j]);
                }
                free(copies);
                return false;
            }
        } else {
            copies[i] = NULL;
        }
    }
    *target = copies;
    *out_count = count;
    return true;
}

static bool amp_kpn_copy_channels(
    AmpKpnTapChannel **target,
    size_t count,
    const AmpTapChannelSpec *specs,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (count == 0U) {
        *target = NULL;
        return true;
    }
    AmpKpnTapChannel *channels = (AmpKpnTapChannel *)calloc(count, sizeof(AmpKpnTapChannel));
    if (channels == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate tap channel array");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const AmpTapChannelSpec *spec = &specs[i];
        if (spec->name != NULL) {
            channels[i].name = amp_kpn_strdup(spec->name);
            if (channels[i].name == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy channel name");
                amp_kpn_free_channels(channels, count);
                return false;
            }
        }
        channels[i].enabled = spec->enabled;
        channels[i].count = spec->count;
        channels[i].unit_shape = spec->unit_shape;
        channels[i].active_index_count = spec->active_index_count;
        if (spec->active_index_count > 0U && spec->active_indices != NULL) {
            size_t bytes = spec->active_index_count * sizeof(uint32_t);
            channels[i].active_indices = (uint32_t *)malloc(bytes);
            if (channels[i].active_indices == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy channel indices");
                amp_kpn_free_channels(channels, count);
                return false;
            }
            memcpy(channels[i].active_indices, spec->active_indices, bytes);
        }
        if (spec->active_mask_b64 != NULL) {
            channels[i].active_mask_b64 = amp_kpn_strdup(spec->active_mask_b64);
            if (channels[i].active_mask_b64 == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy channel mask");
                amp_kpn_free_channels(channels, count);
                return false;
            }
        }
        channels[i].subchannel_count = spec->subchannel_count;
        if (!amp_kpn_copy_subchannels(&channels[i].subchannels, spec->subchannel_count, spec->subchannels, error_buffer, error_buffer_len)) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy channel subchannels");
            amp_kpn_free_channels(channels, count);
            return false;
        }
    }
    *target = channels;
    return true;
}

static void amp_kpn_free_subchannels(AmpKpnTapSubchannel *subs, size_t count) {
    if (subs == NULL) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free(subs[i].name);
    }
    free(subs);
}

static void amp_kpn_free_channels(AmpKpnTapChannel *channels, size_t count) {
    if (channels == NULL) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free(channels[i].name);
        free(channels[i].active_indices);
        free(channels[i].active_mask_b64);
        amp_kpn_free_subchannels(channels[i].subchannels, channels[i].subchannel_count);
    }
    free(channels);
}

static void amp_kpn_free_group(AmpKpnTapGroup *group) {
    if (group == NULL) {
        return;
    }
    free(group->name);
    free(group->dtype);
    free(group->buffer.layout);
    free(group->delivery.mode);
    free(group->delivery.multicast.ack_policy);
    free(group->delivery.fifo_pc.release_policy);
    if (group->indexing.major_order != NULL) {
        for (size_t i = 0; i < group->indexing.major_order_count; ++i) {
            free(group->indexing.major_order[i]);
        }
        free(group->indexing.major_order);
    }
    free(group->indexing.band_indexing);
    amp_kpn_free_channels(group->channels, group->channel_count);
}

static void amp_kpn_free_groups(AmpKpnTapGroup *groups, size_t count) {
    if (groups == NULL) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        amp_kpn_free_group(&groups[i]);
    }
    free(groups);
}

static bool amp_kpn_copy_groups(
    AmpKpnTapGroup **target,
    size_t count,
    const AmpTapGroupSpec *specs,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (count == 0U) {
        *target = NULL;
        return true;
    }
    AmpKpnTapGroup *groups = (AmpKpnTapGroup *)calloc(count, sizeof(AmpKpnTapGroup));
    if (groups == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate tap group array");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        const AmpTapGroupSpec *spec = &specs[i];
        if (spec->name != NULL) {
            groups[i].name = amp_kpn_strdup(spec->name);
            if (groups[i].name == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap group name");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].unit_shape = spec->unit_shape;
        if (spec->dtype != NULL) {
            groups[i].dtype = amp_kpn_strdup(spec->dtype);
            if (groups[i].dtype == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap group dtype");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].buffer.prealloc_units = spec->buffer.prealloc_units;
        if (spec->buffer.layout != NULL) {
            groups[i].buffer.layout = amp_kpn_strdup(spec->buffer.layout);
            if (groups[i].buffer.layout == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap buffer layout");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].delivery.full_percent_target = spec->delivery.full_percent_target;
        groups[i].delivery.low_watermark_percent = spec->delivery.low_watermark_percent;
        if (spec->delivery.mode != NULL) {
            groups[i].delivery.mode = amp_kpn_strdup(spec->delivery.mode);
            if (groups[i].delivery.mode == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap delivery mode");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].delivery.multicast.quorum_k = spec->delivery.multicast.quorum_k;
        if (spec->delivery.multicast.ack_policy != NULL) {
            groups[i].delivery.multicast.ack_policy = amp_kpn_strdup(spec->delivery.multicast.ack_policy);
            if (groups[i].delivery.multicast.ack_policy == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy multicast ack policy");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        if (spec->delivery.fifo_pc.release_policy != NULL) {
            groups[i].delivery.fifo_pc.release_policy = amp_kpn_strdup(spec->delivery.fifo_pc.release_policy);
            if (groups[i].delivery.fifo_pc.release_policy == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy fifo release policy");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].indexing.major_order = NULL;
        groups[i].indexing.major_order_count = 0U;
        if (!amp_kpn_copy_string_array(
                &groups[i].indexing.major_order,
                &groups[i].indexing.major_order_count,
                spec->indexing.major_order,
                spec->indexing.major_order_count,
                error_buffer,
                error_buffer_len
            )) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy indexing order");
            amp_kpn_free_groups(groups, count);
            return false;
        }
        if (spec->indexing.band_indexing != NULL) {
            groups[i].indexing.band_indexing = amp_kpn_strdup(spec->indexing.band_indexing);
            if (groups[i].indexing.band_indexing == NULL) {
                amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy band indexing policy");
                amp_kpn_free_groups(groups, count);
                return false;
            }
        }
        groups[i].channel_count = spec->channel_count;
        if (!amp_kpn_copy_channels(&groups[i].channels, spec->channel_count, spec->channels, error_buffer, error_buffer_len)) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy tap group channels");
            amp_kpn_free_groups(groups, count);
            return false;
        }
    }
    *target = groups;
    return true;
}

static bool amp_kpn_node_name_exists(const AmpKpnGraph *graph, const char *name) {
    for (size_t i = 0; i < graph->node_count; ++i) {
        const AmpKpnNode *node = &graph->nodes[i];
        if (node->name != NULL && name != NULL && strcmp(node->name, name) == 0) {
            return true;
        }
    }
    return false;
}

static int amp_kpn_find_node_index(const AmpKpnGraph *graph, const char *name) {
    if (graph == NULL || name == NULL) {
        return -1;
    }
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].name != NULL && strcmp(graph->nodes[i].name, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int amp_kpn_find_input_index(const AmpKpnNode *node, const char *tap_name) {
    if (node == NULL || tap_name == NULL) {
        return -1;
    }
    for (size_t i = 0; i < node->input_count; ++i) {
        const AmpKpnTapDefault *tap = &node->inputs[i];
        if (tap->name != NULL && strcmp(tap->name, tap_name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int amp_kpn_find_output_index(const AmpKpnNode *node, const char *tap_name) {
    if (node == NULL || tap_name == NULL) {
        return -1;
    }
    for (size_t i = 0; i < node->output_count; ++i) {
        const AmpKpnTapDefault *tap = &node->outputs[i];
        if (tap->name != NULL && strcmp(tap->name, tap_name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

int amp_kpn_graph_convert(
    const AmpNodeSpec *nodes,
    size_t node_count,
    const AmpEdgeSpec *edges,
    size_t edge_count,
    AmpKpnGraph *out_graph,
    char *error_buffer,
    size_t error_buffer_len
) {
    if (out_graph == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "output graph pointer is null");
        return -1;
    }
    amp_kpn_zero_graph(out_graph);

    if (node_count == 0U) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "graph must contain at least one node");
        return -1;
    }
    if (nodes == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "node specification array is null");
        return -1;
    }

    AmpKpnGraph graph;
    amp_kpn_zero_graph(&graph);

    graph.nodes = (AmpKpnNode *)calloc(node_count, sizeof(AmpKpnNode));
    if (graph.nodes == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate graph nodes");
        return -1;
    }
    graph.node_count = node_count;

    for (size_t i = 0; i < node_count; ++i) {
        const AmpNodeSpec *spec = &nodes[i];
        AmpKpnNode *node = &graph.nodes[i];
        if (spec->name == NULL || spec->name[0] == '\0') {
            amp_kpn_write_error(error_buffer, error_buffer_len, "node specification missing name");
            amp_kpn_graph_free(&graph);
            return -1;
        }
        if (amp_kpn_node_name_exists(&graph, spec->name)) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "duplicate node name detected");
            amp_kpn_graph_free(&graph);
            return -1;
        }
        node->name = amp_kpn_strdup(spec->name);
        if (node->name == NULL) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "failed to copy node name");
            amp_kpn_graph_free(&graph);
            return -1;
        }
        if (!amp_kpn_copy_taps(&node->inputs, spec->input_count, spec->inputs, error_buffer, error_buffer_len)) {
            amp_kpn_graph_free(&graph);
            return -1;
        }
        if (!amp_kpn_copy_taps(&node->outputs, spec->output_count, spec->outputs, error_buffer, error_buffer_len)) {
            amp_kpn_graph_free(&graph);
            return -1;
        }
        if (!amp_kpn_copy_groups(&node->tap_groups, spec->tap_group_count, spec->tap_groups, error_buffer, error_buffer_len)) {
            amp_kpn_graph_free(&graph);
            return -1;
        }
        node->input_count = spec->input_count;
        node->output_count = spec->output_count;
        node->tap_group_count = spec->tap_group_count;
    }

    graph.edges = (AmpKpnEdge *)calloc(edge_count, sizeof(AmpKpnEdge));
    if (edge_count > 0U && graph.edges == NULL) {
        amp_kpn_write_error(error_buffer, error_buffer_len, "failed to allocate graph edges");
        amp_kpn_graph_free(&graph);
        return -1;
    }
    graph.edge_count = edge_count;

    for (size_t i = 0; i < edge_count; ++i) {
        const AmpEdgeSpec *spec = &edges[i];
        AmpKpnEdge *edge = &graph.edges[i];
        int producer_index = amp_kpn_find_node_index(&graph, spec->producer_node);
        int consumer_index = amp_kpn_find_node_index(&graph, spec->consumer_node);
        if (producer_index < 0 || consumer_index < 0) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "edge references unknown node");
            amp_kpn_graph_free(&graph);
            return -1;
        }
        AmpKpnNode *producer = &graph.nodes[(size_t)producer_index];
        AmpKpnNode *consumer = &graph.nodes[(size_t)consumer_index];
        int output_index = amp_kpn_find_output_index(producer, spec->producer_tap);
        int input_index = amp_kpn_find_input_index(consumer, spec->consumer_tap);
        if (output_index < 0 || input_index < 0) {
            amp_kpn_write_error(error_buffer, error_buffer_len, "edge references unknown tap");
            amp_kpn_graph_free(&graph);
            return -1;
        }
        edge->producer = producer;
        edge->producer_output_index = (size_t)output_index;
        edge->producer_output = &producer->outputs[(size_t)output_index];
        edge->consumer = consumer;
        edge->consumer_input_index = (size_t)input_index;
        edge->consumer_input = &consumer->inputs[(size_t)input_index];
    }

    *out_graph = graph;
    amp_kpn_write_error(error_buffer, error_buffer_len, NULL);
    return 0;
}

void amp_kpn_graph_free(AmpKpnGraph *graph) {
    if (graph == NULL) {
        return;
    }
    if (graph->nodes != NULL) {
        for (size_t i = 0; i < graph->node_count; ++i) {
            AmpKpnNode *node = &graph->nodes[i];
            free(node->name);
            if (node->inputs != NULL) {
                for (size_t j = 0; j < node->input_count; ++j) {
                    free(node->inputs[j].name);
                    free(node->inputs[j].group);
                }
                free(node->inputs);
            }
            if (node->outputs != NULL) {
                for (size_t j = 0; j < node->output_count; ++j) {
                    free(node->outputs[j].name);
                    free(node->outputs[j].group);
                }
                free(node->outputs);
            }
            amp_kpn_free_groups(node->tap_groups, node->tap_group_count);
        }
        free(graph->nodes);
    }
    free(graph->edges);
    amp_kpn_zero_graph(graph);
}

const AmpKpnNode *amp_kpn_graph_find_node(const AmpKpnGraph *graph, const char *name) {
    if (graph == NULL || name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < graph->node_count; ++i) {
        if (graph->nodes[i].name != NULL && strcmp(graph->nodes[i].name, name) == 0) {
            return &graph->nodes[i];
        }
    }
    return NULL;
}

const AmpKpnTapDefault *amp_kpn_node_find_output(const AmpKpnNode *node, const char *tap_name) {
    if (node == NULL || tap_name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < node->output_count; ++i) {
        const AmpKpnTapDefault *tap = &node->outputs[i];
        if (tap->name != NULL && strcmp(tap->name, tap_name) == 0) {
            return tap;
        }
    }
    return NULL;
}

const AmpKpnTapDefault *amp_kpn_node_find_input(const AmpKpnNode *node, const char *tap_name) {
    if (node == NULL || tap_name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < node->input_count; ++i) {
        const AmpKpnTapDefault *tap = &node->inputs[i];
        if (tap->name != NULL && strcmp(tap->name, tap_name) == 0) {
            return tap;
        }
    }
    return NULL;
}
