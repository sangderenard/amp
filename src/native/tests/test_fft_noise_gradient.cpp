#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

extern "C" {
#include "amp_native.h"
}

namespace {

struct GradientData {
    std::vector<double> audio;
    std::vector<double> divisor;
    std::vector<double> divisor_imag;
    std::vector<double> phase;
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> filter;
    std::vector<double> stabilizer;
    uint32_t frames{0};
    uint32_t window_size{0};
    uint32_t oversample_ratio{1};
    double sample_rate{48000.0};
};

constexpr const char *kParamNames[] = {
    "divisor",
    "divisor_imag",
    "phase_offset",
    "lower_band",
    "upper_band",
    "filter_intensity",
    "stabilizer"
};

#pragma pack(push, 1)
struct ParamFileHeader {
    uint32_t frames;
    uint32_t window_size;
    double sample_rate;
    uint32_t oversample_ratio;
};
#pragma pack(pop)
static_assert(sizeof(ParamFileHeader) == 20, "ParamFileHeader must be packed to 20 bytes");

bool read_parameter_blob(const std::filesystem::path &path, GradientData &out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "Failed to open parameter blob '%s'\n", path.string().c_str());
        return false;
    }
    ParamFileHeader header{};
    stream.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!stream) {
        std::fprintf(stderr, "Parameter blob '%s' ended before header\n", path.string().c_str());
        return false;
    }
    if (header.frames == 0U || header.window_size == 0U) {
        std::fprintf(stderr, "Parameter blob header is invalid (frames=%u window_size=%u)\n", header.frames, header.window_size);
        return false;
    }
    out.frames = header.frames;
    out.window_size = header.window_size;
    out.sample_rate = header.sample_rate > 0.0 ? header.sample_rate : 48000.0;
    out.oversample_ratio = header.oversample_ratio > 0U ? header.oversample_ratio : 1U;
    const size_t total_values = static_cast<size_t>(out.frames);
    auto read_vector = [&](std::vector<double> &vec) -> bool {
        vec.resize(total_values);
        stream.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(total_values * sizeof(double)));
        return static_cast<size_t>(stream.gcount()) == total_values * sizeof(double);
    };
    if (!read_vector(out.audio) || !read_vector(out.divisor) || !read_vector(out.divisor_imag) ||
        !read_vector(out.phase) || !read_vector(out.lower) || !read_vector(out.upper) ||
        !read_vector(out.filter) || !read_vector(out.stabilizer)) {
        std::fprintf(stderr, "Parameter blob '%s' ended prematurely (expected %zu doubles per array)\n", path.string().c_str(), total_values);
        return false;
    }
    return true;
}

double smooth_random(double t, double magnitude, double offset) {
    double curve = 0.5 * (1.0 - std::cos(2.0 * M_PI * t));
    return offset + magnitude * curve;
}

void generate_default_gradient(uint32_t frames, uint32_t window_size, GradientData &out) {
    out.frames = frames;
    out.window_size = window_size;
    out.sample_rate = 48000.0;
    out.oversample_ratio = 4U;
    const size_t total = static_cast<size_t>(frames);
    out.audio.resize(total);
    out.divisor.assign(total, 1.0);
    out.divisor_imag.assign(total, 0.0);
    out.phase.resize(total);
    out.lower.resize(total);
    out.upper.resize(total);
    out.filter.resize(total);
    out.stabilizer.assign(total, 1.0e-9);

    std::mt19937_64 rng(0xC0FFEEu);
    std::uniform_real_distribution<double> noise(-1.0, 1.0);
    double prev = 0.0;
    for (uint32_t i = 0; i < frames; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(frames - 1U);
        double n = noise(rng);
        prev = 0.98 * prev + 0.02 * n;
        out.audio[i] = prev;
        out.phase[i] = 0.5 * M_PI + 0.5 * M_PI * std::sin(2.0 * M_PI * t);
        double lower = 0.05 + 0.4 * t;
        double span = smooth_random(t, 0.3, 0.15);
        double upper = std::min(0.95, lower + span);
        out.lower[i] = std::clamp(lower, 0.0, 1.0);
        out.upper[i] = std::clamp(upper, 0.0, 1.0);
        out.filter[i] = std::clamp(0.35 + 0.6 * std::sin(M_PI * t), 0.05, 0.95);
        out.divisor[i] = 0.75 + 0.25 * std::cos(4.0 * M_PI * t);
    }
}

static bool g_fft_backward_supported = true;

std::vector<double> run_fft_backward(const GradientData &data, const std::string &algorithm_override) {
    if (data.frames == 0U || data.window_size == 0U) {
        std::fprintf(stderr, "GradientData not initialised\n");
        return {};
    }
    const std::string name = "fft_noise";
    const std::string type_name = "FFTDivisionNode";
    std::string params_json = std::string("{\"window_size\":") +
        std::to_string(data.window_size) +
        ",\"stabilizer\":1e-9,\"epsilon\":1e-9,\"declared_delay\":" +
        std::to_string(data.window_size > 0 ? data.window_size - 1U : 0U) +
        ",\"oversample_ratio\":" + std::to_string(data.oversample_ratio) +
        ",\"supports_v2\":true";
    if (!algorithm_override.empty()) {
        params_json += ",\"algorithm\":\"" + algorithm_override + "\"";
    }
    params_json += "}";

    EdgeRunnerNodeDescriptor descriptor{};
    descriptor.name = name.c_str();
    descriptor.name_len = static_cast<uint32_t>(name.size());
    descriptor.type_name = type_name.c_str();
    descriptor.type_len = static_cast<uint32_t>(type_name.size());
    descriptor.params_json = params_json.c_str();
    descriptor.params_len = static_cast<uint32_t>(params_json.size());

    EdgeRunnerAudioView audio_view{};
    audio_view.has_audio = 1;
    audio_view.batches = 1;
    audio_view.channels = 1;
    audio_view.frames = data.frames;
    audio_view.data = data.audio.data();

    std::vector<EdgeRunnerParamView> param_views;
    param_views.reserve(std::size(kParamNames));
    auto make_view = [&](const char *name_ptr, const std::vector<double> &values) {
        EdgeRunnerParamView view{};
        view.name = name_ptr;
        view.batches = 1;
        view.channels = 1;
        view.frames = data.frames;
        view.data = values.data();
        param_views.push_back(view);
    };
    make_view(kParamNames[0], data.divisor);
    make_view(kParamNames[1], data.divisor_imag);
    make_view(kParamNames[2], data.phase);
    make_view(kParamNames[3], data.lower);
    make_view(kParamNames[4], data.upper);
    make_view(kParamNames[5], data.filter);
    make_view(kParamNames[6], data.stabilizer);

    EdgeRunnerParamSet param_set{};
    param_set.count = static_cast<uint32_t>(param_views.size());
    param_set.items = param_views.data();

    EdgeRunnerNodeInputs inputs{};
    inputs.audio = audio_view;
    inputs.params = param_set;

    void *state = nullptr;
    double *out_buffer = nullptr;
    int out_channels = 0;
    AmpNodeMetrics metrics{};

    int rc = amp_run_node_v2(
        &descriptor,
        &inputs,
        1,
        1,
        static_cast<int>(data.frames),
        data.sample_rate,
        &out_buffer,
        &out_channels,
        &state,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &metrics
    );
    if (rc != 0 || out_buffer == nullptr) {
        if (rc == AMP_E_UNSUPPORTED) {
            g_fft_backward_supported = false;
            std::fprintf(stderr, "amp_run_node_v2 returned AMP_E_UNSUPPORTED for FFTDivisionNode backward mode\n");
        } else {
            std::fprintf(stderr, "amp_run_node_v2 failed with rc=%d\n", rc);
        }
        if (out_buffer != nullptr) {
            amp_free(out_buffer);
        }
        if (state != nullptr) {
            amp_release_state(state);
        }
        return {};
    }

    std::vector<double> output(static_cast<size_t>(data.frames));
    const size_t copy_count = static_cast<size_t>(out_channels) * static_cast<size_t>(data.frames);
    for (size_t i = 0; i < copy_count && i < output.size(); ++i) {
        output[i] = out_buffer[i];
    }

    amp_free(out_buffer);
    if (state != nullptr) {
        amp_release_state(state);
    }

    std::printf(
        "fft_noise_gradient: frames=%u window=%u oversample=%u delay=%u heat=%.6f\n",
        data.frames,
        data.window_size,
        data.oversample_ratio,
        metrics.measured_delay_frames,
        metrics.accumulated_heat
    );
    return output;
}

bool write_wav_int16(const std::filesystem::path &path, const std::vector<double> &samples, double sample_rate) {
    if (samples.empty()) {
        std::fprintf(stderr, "No samples to write for '%s'\n", path.string().c_str());
        return false;
    }
    double peak = 0.0;
    for (double v : samples) {
        peak = std::max(peak, std::fabs(v));
    }
    const double normaliser = peak > 0.0 ? (32760.0 / peak) : 1.0;

    struct WavHeader {
        char riff[4];
        uint32_t chunk_size;
        char wave[4];
        char fmt_chunk_id[4];
        uint32_t fmt_chunk_size;
        uint16_t audio_format;
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample;
        char data_chunk_id[4];
        uint32_t data_size;
    } header{};

    std::memcpy(header.riff, "RIFF", 4);
    std::memcpy(header.wave, "WAVE", 4);
    std::memcpy(header.fmt_chunk_id, "fmt ", 4);
    std::memcpy(header.data_chunk_id, "data", 4);
    header.fmt_chunk_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = static_cast<uint32_t>(sample_rate);
    header.bits_per_sample = 16;
    header.block_align = static_cast<uint16_t>(header.num_channels * header.bits_per_sample / 8);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = static_cast<uint32_t>(samples.size() * header.block_align);
    header.chunk_size = 36 + header.data_size;

    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        double scaled = samples[i] * normaliser;
        scaled = std::clamp(scaled, -32760.0, 32760.0);
        pcm[i] = static_cast<int16_t>(std::lrint(scaled));
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "Failed to open '%s' for writing\n", path.string().c_str());
        return false;
    }
    stream.write(reinterpret_cast<const char *>(&header), sizeof(header));
    stream.write(reinterpret_cast<const char *>(pcm.data()), static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
    stream.flush();
    return static_cast<bool>(stream);
}

struct Arguments {
    std::filesystem::path param_blob;
    std::filesystem::path output_wav{"output.wav"};
    uint32_t frames{32768};
    uint32_t window_size{1024};
    std::string algorithm;
};

Arguments parse_arguments(int argc, char **argv) {
    Arguments args{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view current(argv[i]);
        if (current == "--params" && i + 1 < argc) {
            args.param_blob = std::filesystem::path(argv[++i]);
            continue;
        }
        if (current == "--output" && i + 1 < argc) {
            args.output_wav = std::filesystem::path(argv[++i]);
            continue;
        }
        if (current == "--frames" && i + 1 < argc) {
            args.frames = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (current == "--window" && i + 1 < argc) {
            args.window_size = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            continue;
        }
        if (current == "--algorithm" && i + 1 < argc) {
            args.algorithm = std::string(argv[++i]);
            continue;
        }
    }
    if (args.frames == 0U) {
        args.frames = 32768;
    }
    if (args.window_size == 0U) {
        args.window_size = 1024;
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    Arguments args = parse_arguments(argc, argv);
    GradientData gradient{};

    if (!args.param_blob.empty()) {
        if (!read_parameter_blob(args.param_blob, gradient)) {
            return 2;
        }
    } else {
        generate_default_gradient(args.frames, args.window_size, gradient);
    }

    auto samples = run_fft_backward(gradient, args.algorithm);
    if (samples.empty()) {
        if (!g_fft_backward_supported) {
            std::fprintf(stderr, "FFT gradient path is unsupported; skipping synthesis\n");
            return 0;
        }
        std::fprintf(stderr, "FFT gradient synthesis failed\n");
        return 3;
    }

    std::filesystem::path target = args.output_wav;
    try {
        std::filesystem::create_directories(target.parent_path());
    } catch (const std::exception &) {
        // ignore parent creation failures; writing will fail later if necessary
    }
    if (!write_wav_int16(target, samples, gradient.sample_rate)) {
        std::fprintf(stderr, "Failed to write '%s'\n", target.string().c_str());
        return 4;
    }
    std::printf("fft_noise_gradient: wrote %zu samples to %s\n", samples.size(), target.string().c_str());
    return 0;
}
