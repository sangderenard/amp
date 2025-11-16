#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "amp_native.h"
}

#include "fft_division_test_helpers.h"

namespace {

constexpr uint32_t kBatches = 1;
constexpr uint32_t kChannels = 1;
constexpr uint32_t kFrames = 8;
constexpr uint32_t kHopSize = 1;
constexpr int kWindowSize = 4;
constexpr double kTolerance = 1e-9;

using amp::tests::fft_division_shared::BuildPcmTapDescriptor;
using amp::tests::fft_division_shared::BuildSpectralTapDescriptor;
using amp::tests::fft_division_shared::TapDescriptor;

struct ParamDescriptor {
    std::string name;
    uint32_t batches{1};
    uint32_t channels{1};
    uint32_t frames{1};
    std::vector<double> data;
};

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

static void append_node(
    std::vector<uint8_t> &buffer,
    const std::string &name,
    const std::string &type_name,
    const std::vector<std::string> &audio_inputs,
    const std::string &params_json,
    const std::vector<ParamDescriptor> &params
) {
    append_u32(buffer, 0U);
    append_u32(buffer, static_cast<uint32_t>(name.size()));
    append_u32(buffer, static_cast<uint32_t>(type_name.size()));
    append_u32(buffer, static_cast<uint32_t>(audio_inputs.size()));
    append_u32(buffer, 0U);
    append_u32(buffer, static_cast<uint32_t>(params.size()));
    append_u32(buffer, 0U);
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

struct GraphDescriptor {
    std::vector<uint8_t> blob;
    TapDescriptor pcm_sink;
    TapDescriptor spectral_real_sink;
    TapDescriptor spectral_imag_sink;
};

static GraphDescriptor build_descriptor(const std::vector<double> &signal) {
    GraphDescriptor result{};
    result.blob.reserve(1024);
    append_u32(result.blob, 3U);

    append_node(
        result.blob,
        "carrier",
        "ConstantNode",
        {},
        "{\"value\":1.0,\"channels\":1}",
        {}
    );

    append_node(
        result.blob,
        "signal",
        "GainNode",
        {"carrier"},
        "{}",
        {ParamDescriptor{"gain", kBatches, kChannels, kFrames, signal}}
    );

    char params_buffer[256];
    std::snprintf(
        params_buffer,
        sizeof(params_buffer),
        "{\"window_size\":%d,\"algorithm\":\"fft\",\"window\":\"hann\",\"supports_v2\":true,"
        "\"declared_delay\":%d,\"oversample_ratio\":1,\"epsilon\":1e-9,\"max_batch_windows\":1,"
        "\"backend_hop\":1,\"enable_spectrum_taps\":true}",
        kWindowSize,
        kWindowSize - 1
    );

    append_node(
        result.blob,
        "fft_divider",
        "FFTDivisionNode",
        {"signal"},
        params_buffer,
        {}
    );

    result.pcm_sink = BuildPcmTapDescriptor(kWindowSize, kHopSize, signal.size(), kChannels);
    result.spectral_real_sink = BuildSpectralTapDescriptor(kWindowSize, kHopSize, signal.size(), kBatches);
    result.spectral_real_sink.name = "spectral_real";
    result.spectral_imag_sink = result.spectral_real_sink;
    result.spectral_imag_sink.name = "spectral_imag";

    return result;
}

static bool compare_frames(const std::vector<double> &reference, const std::vector<double> &candidate) {
    if (reference.size() != candidate.size()) {
        return false;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = std::fabs(reference[i] - candidate[i]);
        if (diff > kTolerance) {
            std::fprintf(
                stderr,
                "test_fft_division_kpn: frame %zu differs %.12f vs %.12f\n",
                i,
                candidate[i],
                reference[i]
            );
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const std::vector<double> signal{
        1.0,
        -0.5,
        0.25,
        -0.125,
        0.0625,
        -0.03125,
        0.015625,
        -0.0078125
    };

    GraphDescriptor descriptor = build_descriptor(signal);

    if (descriptor.pcm_sink.shape.channels != kChannels) {
        std::fprintf(
            stderr,
            "test_fft_division_kpn: pcm sink channel count mismatch %u vs %u\n",
            descriptor.pcm_sink.shape.channels,
            kChannels
        );
        return 2;
    }
    if (descriptor.spectral_real_sink.shape.channels != static_cast<uint32_t>(kWindowSize) ||
        descriptor.spectral_imag_sink.shape.channels != static_cast<uint32_t>(kWindowSize)) {
        std::fprintf(
            stderr,
            "test_fft_division_kpn: spectral sink channel count mismatch vs window size %d\n",
            kWindowSize
        );
        return 2;
    }

    AmpGraphRuntime *runtime = amp_graph_runtime_create(
        descriptor.blob.data(),
        descriptor.blob.size(),
        nullptr,
        0U
    );
    if (runtime == nullptr) {
        std::fprintf(stderr, "test_fft_division_kpn: runtime create failed\n");
        return 2;
    }

    if (amp_graph_runtime_configure(runtime, kBatches, kFrames) != 0) {
        std::fprintf(stderr, "test_fft_division_kpn: configure failed\n");
        amp_graph_runtime_destroy(runtime);
        return 3;
    }

    double *out_buffer = nullptr;
    uint32_t out_batches = 0;
    uint32_t out_channels = 0;
    uint32_t out_frames = 0;
    int exec_rc = amp_graph_runtime_execute(
        runtime,
        nullptr,
        0U,
        static_cast<int>(kFrames),
        48000.0,
        &out_buffer,
        &out_batches,
        &out_channels,
        &out_frames
    );
    if (exec_rc != 0 || out_buffer == nullptr) {
        std::fprintf(
            stderr,
            "test_fft_division_kpn: execute failed rc=%d buffer=%p\n",
            exec_rc,
            static_cast<void *>(out_buffer)
        );
        if (out_buffer != nullptr) {
            amp_graph_runtime_buffer_free(out_buffer);
        }
        amp_graph_runtime_destroy(runtime);
        return 4;
    }

    const size_t total = static_cast<size_t>(out_batches) * out_channels * out_frames;
    std::vector<double> candidate(out_buffer, out_buffer + total);
    amp_graph_runtime_buffer_free(out_buffer);
    amp_graph_runtime_destroy(runtime);

    if (out_batches != descriptor.pcm_sink.shape.batches ||
        out_channels != descriptor.pcm_sink.shape.channels ||
        out_frames != descriptor.pcm_sink.shape.frames) {
        std::fprintf(
            stderr,
            "test_fft_division_kpn: unexpected output shape %u x %u x %u\n",
            out_batches,
            out_channels,
            out_frames
        );
        return 5;
    }

    std::fprintf(stderr, "candidate: ");
    for (double v : candidate) {
        std::fprintf(stderr, "%.12f ", v);
    }
    std::fprintf(stderr, "\n");

    if (!compare_frames(signal, candidate)) {
        return 6;
    }

    return 0;
}
