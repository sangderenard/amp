// Native unit test validating thermo.heat parameter overrides

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "amp_native.h"
}

namespace {

static void append_u32(std::vector<uint8_t> &buffer, uint32_t value) {
    buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFFU));
}

static void append_u64(std::vector<uint8_t> &buffer, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFU));
    }
}

static void append_string(std::vector<uint8_t> &buffer, const std::string &value) {
    buffer.insert(buffer.end(), value.begin(), value.end());
}

static void append_doubles(std::vector<uint8_t> &buffer, const std::vector<double> &values) {
    for (double v : values) {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(double));
        append_u64(buffer, bits);
    }
}

struct ParamDescriptor {
    std::string name;
    uint32_t batches{1};
    uint32_t channels{1};
    uint32_t frames{1};
    std::vector<double> data;
};

static void append_node(
    std::vector<uint8_t> &buffer,
    const std::string &name,
    const std::string &type_name,
    const std::vector<std::string> &audio_inputs,
    const std::string &params_json,
    const std::vector<ParamDescriptor> &params
) {
    append_u32(buffer, 0U); // type_id (unused placeholder)
    append_u32(buffer, static_cast<uint32_t>(name.size()));
    append_u32(buffer, static_cast<uint32_t>(type_name.size()));
    append_u32(buffer, static_cast<uint32_t>(audio_inputs.size()));
    append_u32(buffer, 0U); // mod_count
    append_u32(buffer, static_cast<uint32_t>(params.size()));
    append_u32(buffer, 0U); // shape_count
    append_u32(buffer, static_cast<uint32_t>(params_json.size()));

    append_string(buffer, name);
    append_string(buffer, type_name);

    for (const std::string &src : audio_inputs) {
        append_u32(buffer, static_cast<uint32_t>(src.size()));
        append_string(buffer, src);
    }

    for (const ParamDescriptor &param : params) {
        uint64_t blob_len = static_cast<uint64_t>(param.data.size() * sizeof(double));
        append_u32(buffer, static_cast<uint32_t>(param.name.size()));
        append_u32(buffer, param.batches);
        append_u32(buffer, param.channels);
        append_u32(buffer, param.frames);
        append_u64(buffer, blob_len);
        append_string(buffer, param.name);
        append_doubles(buffer, param.data);
    }

    append_string(buffer, params_json);
}

} // namespace

int main() {
    constexpr uint32_t kBatches = 1;
    constexpr uint32_t kChannels = 1;
    constexpr uint32_t kFrames = 4;

    std::vector<uint8_t> descriptor;
    descriptor.reserve(512);
    append_u32(descriptor, 2U); // node count

    append_node(
        descriptor,
        "source",
        "ConstantNode",
        {},
        "{\"value\":0.5,\"channels\":1}",
        {}
    );

    append_node(
        descriptor,
        "gain",
        "GainNode",
        {"source"},
        "{\"thermo\":{\"heat_param\":\"thermo.heat\"}}",
        {
            ParamDescriptor{
                "gain",
                kBatches,
                kChannels,
                kFrames,
                std::vector<double>(kFrames, 1.0)
            },
            ParamDescriptor{
                "thermo.heat",
                kBatches,
                kChannels,
                kFrames,
                std::vector<double>(kFrames, 0.0)
            }
        }
    );

    AmpGraphRuntime *runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    if (runtime == nullptr) {
        std::fprintf(stderr, "test_thermo_param: failed to create runtime\n");
        return 2;
    }

    if (amp_graph_runtime_configure(runtime, kBatches, kFrames) != 0) {
        std::fprintf(stderr, "test_thermo_param: configure failed\n");
        amp_graph_runtime_destroy(runtime);
        return 3;
    }

    const double heat_bad[1] = {1.0};
    int rc = amp_graph_runtime_set_param(runtime, "gain", "thermo.heat", heat_bad, kBatches, kChannels, 1U);
    if (rc != -2) {
        std::fprintf(stderr, "test_thermo_param: expected shape mismatch rc=-2, got %d\n", rc);
        amp_graph_runtime_destroy(runtime);
        return 4;
    }

    const double heat_override[kFrames] = {0.1, 0.2, 0.3, 0.4};
    rc = amp_graph_runtime_set_param(runtime, "gain", "thermo.heat", heat_override, kBatches, kChannels, kFrames);
    if (rc != 0) {
        std::fprintf(stderr, "test_thermo_param: thermo override failed rc=%d\n", rc);
        amp_graph_runtime_destroy(runtime);
        return 5;
    }

    double *out_buf = nullptr;
    uint32_t out_batches = 0;
    uint32_t out_channels = 0;
    uint32_t out_frames = 0;
    rc = amp_graph_runtime_execute(runtime, nullptr, 0U, static_cast<int>(kFrames), 48000.0, &out_buf, &out_batches, &out_channels, &out_frames);
    if (rc != 0) {
        std::fprintf(stderr, "test_thermo_param: execute failed rc=%d\n", rc);
        amp_graph_runtime_destroy(runtime);
        return 6;
    }
    if (out_buf == nullptr) {
        std::fprintf(stderr, "test_thermo_param: execute returned null buffer\n");
        amp_graph_runtime_destroy(runtime);
        return 7;
    }
    if (out_batches != kBatches || out_channels != kChannels || out_frames != kFrames) {
        std::fprintf(
            stderr,
            "test_thermo_param: unexpected output shape %u x %u x %u\n",
            out_batches,
            out_channels,
            out_frames
        );
        amp_graph_runtime_buffer_free(out_buf);
        amp_graph_runtime_destroy(runtime);
        return 8;
    }

    for (uint32_t i = 0; i < kFrames; ++i) {
        double expected = 0.5; // constant value * gain of 1.0
        if (std::fabs(out_buf[i] - expected) > 1e-9) {
            std::fprintf(
                stderr,
                "test_thermo_param: output mismatch at frame %u got=%0.12f expected=%0.12f\n",
                i,
                out_buf[i],
                expected
            );
            amp_graph_runtime_buffer_free(out_buf);
            amp_graph_runtime_destroy(runtime);
            return 9;
        }
    }

    amp_graph_runtime_buffer_free(out_buf);
    amp_graph_runtime_destroy(runtime);
    std::printf("test_thermo_param: PASS (thermo.heat validated)\n");
    return 0;
}

