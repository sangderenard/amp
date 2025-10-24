// Minimal C++ unit test for the graph runtime (KPN) using the ConstantNode
// This test is pure C++ and does not require Python.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>

extern "C" {
#include "amp_native.h"
}

// Helpers to append little-endian integers to a vector<uint8_t>
static void append_u32(std::vector<uint8_t> &v, uint32_t x) {
    v.push_back((uint8_t)(x & 0xFF));
    v.push_back((uint8_t)((x >> 8) & 0xFF));
    v.push_back((uint8_t)((x >> 16) & 0xFF));
    v.push_back((uint8_t)((x >> 24) & 0xFF));
}

static void append_u64(std::vector<uint8_t> &v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back((uint8_t)((x >> (8 * i)) & 0xFF));
}

int main() {
    // Build descriptor blob with four nodes creating a small pipeline:
    //  - c0: ConstantNode value=0.5 channels=1
    //  - c1: ConstantNode value=0.25 channels=1
    //  - gain: GainNode audio input ["c0"] with default param "gain" shaped per-batch [2.0, 1.0]
    //  - mix: MixNode audio inputs ["gain","c1"] with target channels=1
    // Expected sink output per-batch: batch0 -> (0.5 * 2.0) + 0.25 = 1.25; batch1 -> (0.5 * 1.0) + 0.25 = 0.75

    std::vector<uint8_t> desc;
    // node_count = 4
    append_u32(desc, 4);

    const std::string c0_name = "c0";
    const std::string c0_type = "ConstantNode";
    const std::string c0_params = "{\"value\":0.5,\"channels\":1}";
    append_u32(desc, 0); // type_id
    append_u32(desc, (uint32_t)c0_name.size());
    append_u32(desc, (uint32_t)c0_type.size());
    append_u32(desc, 0); // audio_count
    append_u32(desc, 0); // mod_count
    append_u32(desc, 0); // param_count
    append_u32(desc, 0); // shape_count
    append_u32(desc, (uint32_t)c0_params.size());
    desc.insert(desc.end(), c0_name.begin(), c0_name.end());
    desc.insert(desc.end(), c0_type.begin(), c0_type.end());
    desc.insert(desc.end(), c0_params.begin(), c0_params.end());

    const std::string c1_name = "c1";
    const std::string c1_type = "ConstantNode";
    const std::string c1_params = "{\"value\":0.25,\"channels\":1}";
    append_u32(desc, 0); // type_id
    append_u32(desc, (uint32_t)c1_name.size());
    append_u32(desc, (uint32_t)c1_type.size());
    append_u32(desc, 0); // audio_count
    append_u32(desc, 0); // mod_count
    append_u32(desc, 0); // param_count
    append_u32(desc, 0); // shape_count
    append_u32(desc, (uint32_t)c1_params.size());
    desc.insert(desc.end(), c1_name.begin(), c1_name.end());
    desc.insert(desc.end(), c1_type.begin(), c1_type.end());
    desc.insert(desc.end(), c1_params.begin(), c1_params.end());

    // gain node: takes audio from c0 and has one default param named "gain"
    const std::string gain_name = "gain";
    const std::string gain_type = "GainNode";
    const std::string gain_params_json = "{}";
    // We'll specify param_count = 1 and include a blob for param "gain"
    append_u32(desc, 0); // type_id
    append_u32(desc, (uint32_t)gain_name.size());
    append_u32(desc, (uint32_t)gain_type.size());
    append_u32(desc, 1); // audio_count (source: c0)
    append_u32(desc, 0); // mod_count
    append_u32(desc, 0); // param_count (we'll set 'gain' via amp_graph_runtime_set_param)
    append_u32(desc, 0); // shape_count
    append_u32(desc, (uint32_t)gain_params_json.size());
    desc.insert(desc.end(), gain_name.begin(), gain_name.end());
    desc.insert(desc.end(), gain_type.begin(), gain_type.end());
    // audio input name
    append_u32(desc, (uint32_t)c0_name.size());
    desc.insert(desc.end(), c0_name.begin(), c0_name.end());
    // include default param 'gain' with two entries (one per batch) both = 2.0
    // param entry for 'gain'
    const std::string param_name = "gain";
    append_u32(desc, (uint32_t)param_name.size());
    // param shape: batches, channels, frames
    append_u32(desc, 2); // batches
    append_u32(desc, 1); // channels
    append_u32(desc, 1); // frames
    // blob_len (bytes): 2 doubles
    append_u64(desc, (uint64_t)(2 * sizeof(double)));
    // param name string
    desc.insert(desc.end(), param_name.begin(), param_name.end());
    // param blob: two doubles [2.0, 2.0]
    double gain_blob_vals[2] = {2.0, 2.0};
    for (size_t i = 0; i < 2; ++i) {
        uint64_t bits;
        memcpy(&bits, &gain_blob_vals[i], sizeof(double));
        for (int b = 0; b < 8; ++b) desc.push_back((uint8_t)((bits >> (8 * b)) & 0xFF));
    }
    // append gain node params_json
    desc.insert(desc.end(), gain_params_json.begin(), gain_params_json.end());

    // mix node: takes audio from ["gain","c1"] and outputs single channel sum
    const std::string mix_name = "mix";
    const std::string mix_type = "MixNode";
    const std::string mix_params = "{\"channels\":1}";
    append_u32(desc, 0); // type_id
    append_u32(desc, (uint32_t)mix_name.size());
    append_u32(desc, (uint32_t)mix_type.size());
    append_u32(desc, 2); // audio_count
    append_u32(desc, 0); // mod_count
    append_u32(desc, 0); // param_count
    append_u32(desc, 0); // shape_count
    append_u32(desc, (uint32_t)mix_params.size());
    desc.insert(desc.end(), mix_name.begin(), mix_name.end());
    desc.insert(desc.end(), mix_type.begin(), mix_type.end());
    append_u32(desc, (uint32_t)gain_name.size());
    desc.insert(desc.end(), gain_name.begin(), gain_name.end());
    append_u32(desc, (uint32_t)c1_name.size());
    desc.insert(desc.end(), c1_name.begin(), c1_name.end());
    desc.insert(desc.end(), mix_params.begin(), mix_params.end());

    // plan blob NULL -> runtime will use default execution order (c0,c1,gain,mix)
    AmpGraphRuntime *runtime = amp_graph_runtime_create(desc.data(), desc.size(), nullptr, 0);
    if (!runtime) {
        std::fprintf(stderr, "Failed to create runtime\n");
        return 2;
    }

    // Configure runtime with batches=2 frames=4
    if (amp_graph_runtime_configure(runtime, 2, 4) != 0) {
        std::fprintf(stderr, "Failed to configure runtime\n");
        amp_graph_runtime_destroy(runtime);
        return 3;
    }

    double *out_buf = nullptr;
    uint32_t out_batches = 0, out_channels = 0, out_frames = 0;
    int rc = amp_graph_runtime_execute(runtime, nullptr, 0, 4, 48000.0, &out_buf, &out_batches, &out_channels, &out_frames);
    if (rc != 0) {
        std::fprintf(stderr, "Execute returned %d\n", rc);
        amp_graph_runtime_destroy(runtime);
        return 4;
    }
    if (!out_buf) {
        std::fprintf(stderr, "No output buffer returned\n");
        amp_graph_runtime_destroy(runtime);
        return 5;
    }

    // Validate output shape and contents
    if (out_batches != 2 || out_channels != 1 || out_frames != 4) {
        std::fprintf(stderr, "Unexpected output shape: batches=%u channels=%u frames=%u\n", out_batches, out_channels, out_frames);
        amp_graph_runtime_destroy(runtime);
        return 6;
    }

    size_t total = (size_t)out_batches * (size_t)out_channels * (size_t)out_frames;
    // Expected: for each batch b: (0.5 * gain[b]) + 0.25 ; gain values were [2.0, 1.0]
    double expected_by_batch[2] = { (0.5 * 2.0) + 0.25, (0.5 * 1.0) + 0.25 };
    bool ok = true;
    for (uint32_t b = 0; b < out_batches; ++b) {
        for (uint32_t c = 0; c < out_channels; ++c) {
            for (uint32_t f = 0; f < out_frames; ++f) {
                size_t idx = ((size_t)b * (size_t)out_channels + (size_t)c) * (size_t)out_frames + (size_t)f;
                double expected = expected_by_batch[b];
                if (out_buf[idx] != expected) {
                    ok = false;
                    std::fprintf(stderr, "Mismatch at b=%u c=%u f=%u idx=%zu: got %f expected %f\n", b, c, f, idx, out_buf[idx], expected);
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
    }
    if (!ok) {
        std::fprintf(stderr, "Dumping output buffer (%zu samples):\n", total);
        for (size_t i = 0; i < total; ++i) {
            std::fprintf(stderr, "  [%zu] = %f\n", i, out_buf[i]);
        }
        amp_graph_runtime_destroy(runtime);
        return 7;
    }

    std::printf("kpn_unit_test: PASS (multi-node pipeline produced expected per-batch outputs)\n");

    amp_graph_runtime_destroy(runtime);
    return 0;
}
