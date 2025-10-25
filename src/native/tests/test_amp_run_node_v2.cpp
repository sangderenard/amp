#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
#include "amp_native.h"
}

namespace {

void append_u32(std::vector<uint8_t> &buffer, uint32_t value) {
    buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFFU));
}

void append_string(std::vector<uint8_t> &buffer, const std::string &value) {
    buffer.insert(buffer.end(), value.begin(), value.end());
}

std::string build_params_json(
    double value,
    uint32_t declared_delay,
    uint32_t oversample_ratio,
    bool supports_v2
) {
    char json[256];
    std::snprintf(
        json,
        sizeof(json),
        "{\"declared_delay\":%u,\"oversample_ratio\":%u,\"supports_v2\":%s,\"value\":%.8f}",
        declared_delay,
        oversample_ratio,
        supports_v2 ? "true" : "false",
        value
    );
    return std::string(json);
}

std::vector<uint8_t> build_constant_descriptor(
    const std::string &name,
    double value,
    uint32_t declared_delay,
    uint32_t oversample_ratio,
    bool supports_v2
) {
    std::vector<uint8_t> blob;
    blob.reserve(128);
    append_u32(blob, 1U);  // node count

    const std::string type_name = "ConstantNode";
    const std::string params_json = build_params_json(value, declared_delay, oversample_ratio, supports_v2);

    append_u32(blob, 0U);  // type_id placeholder
    append_u32(blob, static_cast<uint32_t>(name.size()));
    append_u32(blob, static_cast<uint32_t>(type_name.size()));
    append_u32(blob, 0U);  // audio inputs
    append_u32(blob, 0U);  // mod connections
    append_u32(blob, 0U);  // params
    append_u32(blob, 0U);  // buffer shapes
    append_u32(blob, static_cast<uint32_t>(params_json.size()));

    append_string(blob, name);
    append_string(blob, type_name);
    append_string(blob, params_json);
    return blob;
}

std::vector<uint8_t> build_plan_blob(
    const std::string &name,
    uint32_t declared_delay,
    uint32_t oversample_ratio
) {
    std::vector<uint8_t> blob;
    blob.reserve(64);
    blob.insert(blob.end(), {'A', 'M', 'P', 'L'});
    append_u32(blob, 2U);  // version
    append_u32(blob, 1U);  // node count

    append_u32(blob, 0U);  // function id
    append_u32(blob, static_cast<uint32_t>(name.size()));
    append_u32(blob, 0U);  // audio offset
    append_u32(blob, 0U);  // audio span
    append_u32(blob, 0U);  // param count
    append_u32(blob, declared_delay);
    append_u32(blob, oversample_ratio);
    append_string(blob, name);
    return blob;
}

void run_basic_execution(AmpGraphRuntime *runtime, int frames) {
    assert(runtime != nullptr);

    double *out_buffer = nullptr;
    uint32_t out_batches = 0;
    uint32_t out_channels = 0;
    uint32_t out_frames = 0;
    int rc = amp_graph_runtime_execute(
        runtime,
        nullptr,
        0,
        frames,
        48000.0,
        &out_buffer,
        &out_batches,
        &out_channels,
        &out_frames
    );
    assert(rc == 0);
    assert(out_buffer != nullptr);
    assert(out_batches == 1U);
    assert(out_channels == 1U);
    assert(out_frames == static_cast<uint32_t>(frames));
    amp_graph_runtime_buffer_free(out_buffer);
}

}  // namespace

int main() {
    {
        const std::string node_name = "constant_v2";
        auto descriptor = build_constant_descriptor(node_name, 0.25, 16U, 2U, true);
        auto plan = build_plan_blob(node_name, 32U, 4U);
        AmpGraphRuntime *runtime = amp_graph_runtime_create(
            descriptor.data(),
            descriptor.size(),
            plan.data(),
            plan.size()
        );
        assert(runtime != nullptr);
        assert(amp_graph_runtime_configure(runtime, 1U, 4U) == 0);

        run_basic_execution(runtime, 4);

        AmpGraphNodeSummary summary{};
        int rc = amp_graph_runtime_describe_node(runtime, node_name.c_str(), &summary);
        assert(rc == 0);
        assert(summary.declared_delay_frames == 32U);     // plan overrides descriptor metadata
        assert(summary.oversample_ratio == 4U);
        assert(summary.supports_v2 == 1);
        assert(summary.has_metrics == 1);
        assert(summary.metrics.measured_delay_frames == 0U);
        assert(summary.metrics.accumulated_heat == 0.0f);
        assert(summary.total_heat_accumulated == 0.0);

        amp_graph_runtime_destroy(runtime);
    }

    {
        const std::string node_name = "constant_fallback";
        auto descriptor = build_constant_descriptor(node_name, 0.5, 0U, 1U, false);
        auto plan = build_plan_blob(node_name, 0U, 1U);
        AmpGraphRuntime *runtime = amp_graph_runtime_create(
            descriptor.data(),
            descriptor.size(),
            plan.data(),
            plan.size()
        );
        assert(runtime != nullptr);
        assert(amp_graph_runtime_configure(runtime, 1U, 2U) == 0);

        run_basic_execution(runtime, 2);

        AmpGraphNodeSummary summary{};
        int rc = amp_graph_runtime_describe_node(runtime, node_name.c_str(), &summary);
        assert(rc == 0);
        assert(summary.declared_delay_frames == 0U);
        assert(summary.oversample_ratio == 1U);
        assert(summary.supports_v2 == 0);
        assert(summary.has_metrics == 0);
        assert(summary.metrics.measured_delay_frames == 0U);
        assert(summary.total_heat_accumulated == 0.0);

        amp_graph_runtime_destroy(runtime);
    }

    return 0;
}
