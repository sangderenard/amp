#ifndef AMP_KPN_GRAPH_CONVERTER_H
#define AMP_KPN_GRAPH_CONVERTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum AmpTapDeliveryMode {
    AMP_TAP_DELIVERY_FIFO = 0,
    AMP_TAP_DELIVERY_MULTICAST = 1
} AmpTapDeliveryMode;

typedef enum AmpTapReleasePolicy {
    AMP_TAP_RELEASE_ALL = 0,
    AMP_TAP_RELEASE_PRIMARY = 1
} AmpTapReleasePolicy;

typedef struct AmpTapShape {
    uint32_t batches;
    uint32_t channels;
    uint32_t frames;
} AmpTapShape;

typedef struct AmpTapSpec {
    const char *name;
    const char *group;
    AmpTapShape shape;
    uint32_t capacity_frames;
    AmpTapDeliveryMode delivery_mode;
    int simultaneous_availability;
    AmpTapReleasePolicy release_policy;
} AmpTapSpec;

typedef struct AmpTapSubchannelSpec {
    const char *name;
    int enabled;
    uint32_t stride_elems;
} AmpTapSubchannelSpec;

typedef struct AmpTapChannelSpec {
    const char *name;
    int enabled;
    uint32_t count;
    const uint32_t *active_indices;
    size_t active_index_count;
    const char *active_mask_b64;
    AmpTapShape unit_shape;
    const AmpTapSubchannelSpec *subchannels;
    size_t subchannel_count;
} AmpTapChannelSpec;

typedef struct AmpTapBufferPolicy {
    uint32_t prealloc_units;
    const char *layout;
} AmpTapBufferPolicy;

typedef struct AmpTapDeliveryMulticastPolicy {
    const char *ack_policy;
    uint32_t quorum_k;
} AmpTapDeliveryMulticastPolicy;

typedef struct AmpTapDeliveryFifoPolicy {
    const char *release_policy;
} AmpTapDeliveryFifoPolicy;

typedef struct AmpTapDeliveryPolicy {
    const char *mode;
    uint32_t full_percent_target;
    uint32_t low_watermark_percent;
    AmpTapDeliveryMulticastPolicy multicast;
    AmpTapDeliveryFifoPolicy fifo_pc;
} AmpTapDeliveryPolicy;

typedef struct AmpTapIndexingPolicy {
    const char **major_order;
    size_t major_order_count;
    const char *band_indexing;
} AmpTapIndexingPolicy;

typedef struct AmpTapGroupSpec {
    const char *name;
    AmpTapShape unit_shape;
    const char *dtype;
    AmpTapBufferPolicy buffer;
    AmpTapDeliveryPolicy delivery;
    AmpTapIndexingPolicy indexing;
    const AmpTapChannelSpec *channels;
    size_t channel_count;
} AmpTapGroupSpec;

typedef struct AmpNodeSpec {
    const char *name;
    const AmpTapSpec *inputs;
    size_t input_count;
    const AmpTapSpec *outputs;
    size_t output_count;
    const AmpTapGroupSpec *tap_groups;
    size_t tap_group_count;
} AmpNodeSpec;

typedef struct AmpEdgeSpec {
    const char *producer_node;
    const char *producer_tap;
    const char *consumer_node;
    const char *consumer_tap;
} AmpEdgeSpec;

typedef struct AmpKpnTapDefault {
    char *name;
    char *group;
    AmpTapShape shape;
    uint32_t capacity_frames;
    AmpTapDeliveryMode delivery_mode;
    int simultaneous_availability;
    AmpTapReleasePolicy release_policy;
} AmpKpnTapDefault;

typedef struct AmpKpnTapSubchannel {
    char *name;
    int enabled;
    uint32_t stride_elems;
} AmpKpnTapSubchannel;

typedef struct AmpKpnTapChannel {
    char *name;
    int enabled;
    uint32_t count;
    uint32_t *active_indices;
    size_t active_index_count;
    char *active_mask_b64;
    AmpTapShape unit_shape;
    AmpKpnTapSubchannel *subchannels;
    size_t subchannel_count;
} AmpKpnTapChannel;

typedef struct AmpKpnTapBufferPolicy {
    uint32_t prealloc_units;
    char *layout;
} AmpKpnTapBufferPolicy;

typedef struct AmpKpnTapDeliveryMulticastPolicy {
    char *ack_policy;
    uint32_t quorum_k;
} AmpKpnTapDeliveryMulticastPolicy;

typedef struct AmpKpnTapDeliveryFifoPolicy {
    char *release_policy;
} AmpKpnTapDeliveryFifoPolicy;

typedef struct AmpKpnTapDeliveryPolicy {
    char *mode;
    uint32_t full_percent_target;
    uint32_t low_watermark_percent;
    AmpKpnTapDeliveryMulticastPolicy multicast;
    AmpKpnTapDeliveryFifoPolicy fifo_pc;
} AmpKpnTapDeliveryPolicy;

typedef struct AmpKpnTapIndexingPolicy {
    char **major_order;
    size_t major_order_count;
    char *band_indexing;
} AmpKpnTapIndexingPolicy;

typedef struct AmpKpnTapGroup {
    char *name;
    AmpTapShape unit_shape;
    char *dtype;
    AmpKpnTapBufferPolicy buffer;
    AmpKpnTapDeliveryPolicy delivery;
    AmpKpnTapIndexingPolicy indexing;
    AmpKpnTapChannel *channels;
    size_t channel_count;
} AmpKpnTapGroup;

typedef struct AmpKpnNode {
    char *name;
    AmpKpnTapDefault *inputs;
    size_t input_count;
    AmpKpnTapDefault *outputs;
    size_t output_count;
    AmpKpnTapGroup *tap_groups;
    size_t tap_group_count;
} AmpKpnNode;

typedef struct AmpKpnEdge {
    AmpKpnNode *producer;
    size_t producer_output_index;
    AmpKpnTapDefault *producer_output;
    AmpKpnNode *consumer;
    size_t consumer_input_index;
    AmpKpnTapDefault *consumer_input;
} AmpKpnEdge;

typedef struct AmpKpnGraph {
    AmpKpnNode *nodes;
    size_t node_count;
    AmpKpnEdge *edges;
    size_t edge_count;
} AmpKpnGraph;

int amp_kpn_graph_convert(
    const AmpNodeSpec *nodes,
    size_t node_count,
    const AmpEdgeSpec *edges,
    size_t edge_count,
    AmpKpnGraph *out_graph,
    char *error_buffer,
    size_t error_buffer_len
);

void amp_kpn_graph_free(AmpKpnGraph *graph);

const AmpKpnNode *amp_kpn_graph_find_node(const AmpKpnGraph *graph, const char *name);
const AmpKpnTapDefault *amp_kpn_node_find_output(const AmpKpnNode *node, const char *tap_name);
const AmpKpnTapDefault *amp_kpn_node_find_input(const AmpKpnNode *node, const char *tap_name);

#ifdef __cplusplus
}
#endif

#endif /* AMP_KPN_GRAPH_CONVERTER_H */
