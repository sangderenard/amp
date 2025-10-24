// Native C++ unit test for the AMP graph runtime (KPN)
// Exercises a semi-complex network with deterministic expectations.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
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

static std::vector<double> simulate_gain(
    double input_value,
    const std::vector<double> &gain_values,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
) {
    size_t stride = static_cast<size_t>(batches) * static_cast<size_t>(channels);
    size_t total = stride * static_cast<size_t>(frames);
    std::vector<double> output(total, 0.0);
    for (uint32_t f = 0; f < frames; ++f) {
        size_t frame_base = static_cast<size_t>(f) * stride;
        for (uint32_t b = 0; b < batches; ++b) {
            for (uint32_t c = 0; c < channels; ++c) {
                size_t idx = frame_base + static_cast<size_t>(b) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                output[idx] = input_value * gain_values[idx];
            }
        }
    }
    return output;
}

static std::vector<double> elementwise_product(
    const std::vector<double> &lhs,
    const std::vector<double> &rhs
) {
    std::vector<double> result;
    result.resize(lhs.size(), 0.0);
    for (size_t i = 0; i < lhs.size() && i < rhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}


static bool verify_output(
    const char *label,
    const double *actual,
    const std::vector<double> &expected,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames,
    double tolerance
) {
    if (actual == nullptr) {
        std::fprintf(stderr, "%s: output buffer was null\n", label);
        return false;
    }
    size_t total = static_cast<size_t>(batches) * static_cast<size_t>(channels) * static_cast<size_t>(frames);
    if (expected.size() != total) {
        std::fprintf(stderr, "%s: expected vector size mismatch (%zu vs %zu)\n", label, expected.size(), total);
        return false;
    }
    bool ok = true;
    size_t stride = static_cast<size_t>(batches) * static_cast<size_t>(channels);
    for (uint32_t f = 0; f < frames && ok; ++f) {
        size_t frame_base = static_cast<size_t>(f) * stride;
        for (uint32_t b = 0; b < batches && ok; ++b) {
            for (uint32_t c = 0; c < channels; ++c) {
                size_t idx = frame_base + static_cast<size_t>(b) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                double got = actual[idx];
                double want = expected[idx];
                if (std::fabs(got - want) > tolerance) {
                    std::fprintf(
                        stderr,
                        "%s: mismatch at frame=%u b=%u c=%u idx=%zu got=%0.12f expected=%0.12f diff=%0.12f\n",
                        label,
                        f,
                        b,
                        c,
                        idx,
                        got,
                        want,
                        got - want
                    );
                    ok = false;
                    break;
                }
            }
        }
    }
    if (!ok) {
        std::fprintf(stderr, "%s: dumping output buffer\n", label);
        for (size_t i = 0; i < total; ++i) {
            std::fprintf(stderr, "  [%zu] = %0.12f\n", i, actual[i]);
        }
    }
    return ok;
}

} // namespace

int main() {
    constexpr uint32_t kBatches = 2;
    constexpr uint32_t kFrames = 4;
    constexpr uint32_t kChannels = 1;
    constexpr double carrier_value = 0.75; // carrier constant

    const std::vector<double> shape_a_defaults{
        1.0, 0.5,  // frame 0: batches 0,1
        1.1, 0.6,  // frame 1
        1.2, 0.7,  // frame 2
        1.3, 0.8   // frame 3
    };
    const std::vector<double> shape_b_defaults{
        1.5, 0.5,   // frame 0
        1.25, 0.25, // frame 1
        1.0, 0.0,   // frame 2
        0.75, -0.25 // frame 3
    };
    const std::vector<double> shape_c_defaults{
        1.0, 1.0, // frame 0
        1.0, 1.0, // frame 1
        1.0, 1.0, // frame 2
        1.0, 1.0  // frame 3
    };
    const std::vector<double> shape_c_override{
        1.5, 0.75,  // frame 0
        1.25, 0.9,  // frame 1
        1.0, 1.05,  // frame 2
        0.8, 1.2    // frame 3
    };

    assert(shape_a_defaults.size() == kBatches * kChannels * kFrames);
    assert(shape_b_defaults.size() == kBatches * kChannels * kFrames);
    assert(shape_c_defaults.size() == kBatches * kChannels * kFrames);
    assert(shape_c_override.size() == kBatches * kChannels * kFrames);

    std::vector<uint8_t> descriptor;
    descriptor.reserve(1024);
    append_u32(descriptor, 4U); // node count

    append_node(
        descriptor,
        "carrier",
        "ConstantNode",
        {},
        "{\"value\":0.75,\"channels\":1}",
        {}
    );

    append_node(
        descriptor,
        "shape_a",
        "GainNode",
        {"carrier"},
        "{}",
        {
            ParamDescriptor{
                "gain",
                kBatches,
                kChannels,
                kFrames,
                shape_a_defaults
            }
        }
    );

    append_node(
        descriptor,
        "shape_b",
        "GainNode",
        {"shape_a"},
        "{}",
        {
            ParamDescriptor{
                "gain",
                kBatches,
                kChannels,
                kFrames,
                shape_b_defaults
            }
        }
    );

    append_node(
        descriptor,
        "shape_c",
        "GainNode",
        {"shape_b"},
        "{}",
        {
            ParamDescriptor{
                "gain",
                kBatches,
                kChannels,
                kFrames,
                shape_c_defaults
            }
        }
    );

    AmpGraphRuntime *runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    if (runtime == nullptr) {
        std::fprintf(stderr, "Failed to create AmpGraphRuntime\n");
        return 2;
    }

    if (amp_graph_runtime_configure(runtime, kBatches, kFrames) != 0) {
        std::fprintf(stderr, "amp_graph_runtime_configure failed\n");
        amp_graph_runtime_destroy(runtime);
        return 3;
    }

    const double tolerance = 1e-9;
    auto stage_a = simulate_gain(carrier_value, shape_a_defaults, kBatches, kChannels, kFrames);
    auto stage_b = elementwise_product(stage_a, shape_b_defaults);
    auto stage_c_default = elementwise_product(stage_b, shape_c_defaults);
    auto stage_c_override = elementwise_product(stage_b, shape_c_override);
    const auto &expected_defaults = stage_c_default;
    const auto &expected_override = stage_c_override;

    auto run_and_check = [&](const char *label, const std::vector<double> &expected) -> bool {
        double *out_buf = nullptr;
        uint32_t out_batches = 0;
        uint32_t out_channels = 0;
        uint32_t out_frames = 0;
        int rc = amp_graph_runtime_execute(runtime, nullptr, 0U, static_cast<int>(kFrames), 48000.0, &out_buf, &out_batches, &out_channels, &out_frames);
        if (rc != 0) {
            std::fprintf(stderr, "%s: execute failed with rc=%d\n", label, rc);
            return false;
        }
        if (out_buf == nullptr) {
            std::fprintf(stderr, "%s: runtime returned null output buffer\n", label);
            return false;
        }
        if (out_batches != kBatches || out_channels != kChannels || out_frames != kFrames) {
            std::fprintf(
                stderr,
                "%s: unexpected output shape batches=%u channels=%u frames=%u\n",
                label,
                out_batches,
                out_channels,
                out_frames
            );
            return false;
        }
        bool ok = verify_output(label, out_buf, expected, kBatches, kChannels, kFrames, tolerance);
        amp_graph_runtime_buffer_free(out_buf);
        return ok;
    };

    if (!run_and_check("defaults", expected_defaults)) {
        amp_graph_runtime_destroy(runtime);
        return 4;
    }

    if (amp_graph_runtime_set_param(
            runtime,
            "shape_c",
            "gain",
            shape_c_override.data(),
            kBatches,
            kChannels,
            kFrames
        ) != 0) {
        std::fprintf(stderr, "amp_graph_runtime_set_param failed\n");
        amp_graph_runtime_destroy(runtime);
        return 5;
    }

    if (!run_and_check("override", expected_override)) {
        amp_graph_runtime_destroy(runtime);
        return 6;
    }

    amp_graph_runtime_clear_params(runtime);

    if (!run_and_check("defaults_after_clear", expected_defaults)) {
        amp_graph_runtime_destroy(runtime);
        return 7;
    }

    amp_graph_runtime_destroy(runtime);
    std::printf("kpn_unit_test: PASS (complex network validated default/override parameter flows)\n");
    return 0;
}

