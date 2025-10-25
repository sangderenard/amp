#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

extern "C" {
#include "amp_native.h"
}

namespace {

double compute_energy(const std::vector<double> &data) {
    double energy = 0.0;
    for (double v : data) {
        energy += v * v;
    }
    return energy;
}

double estimate_frequency(const double *data, int frames, double sample_rate, int skip) {
    int last_crossing = -1;
    int crossings = 0;
    double period_sum = 0.0;
    for (int i = skip + 1; i < frames; ++i) {
        double prev = data[i - 1];
        double curr = data[i];
        if (prev <= 0.0 && curr > 0.0) {
            if (last_crossing >= 0) {
                period_sum += static_cast<double>(i - last_crossing);
                crossings += 1;
            }
            last_crossing = i;
        }
    }
    if (crossings == 0 || period_sum <= 0.0) {
        return 0.0;
    }
    double average_period = period_sum / static_cast<double>(crossings);
    if (average_period <= 0.0) {
        return 0.0;
    }
    return sample_rate / average_period;
}

void run_pitch_shift(
    double ratio,
    const std::vector<double> &audio,
    int frames,
    double sample_rate,
    std::vector<double> &output
) {
    std::string params_json = std::string("{\"ratio\":") + std::to_string(ratio) + "}";
    std::string name = "pitch_shift_test";
    std::string type_name = "PitchShiftNode";

    EdgeRunnerNodeDescriptor descriptor{};
    descriptor.name = name.c_str();
    descriptor.name_len = static_cast<size_t>(name.size());
    descriptor.type_name = type_name.c_str();
    descriptor.type_len = static_cast<size_t>(type_name.size());
    descriptor.params_json = params_json.c_str();
    descriptor.params_len = static_cast<size_t>(params_json.size());

    EdgeRunnerAudioView audio_view{};
    audio_view.has_audio = 1;
    audio_view.batches = 1;
    audio_view.channels = 1;
    audio_view.frames = static_cast<uint32_t>(frames);
    audio_view.data = audio.data();

    EdgeRunnerParamSet param_set{};
    param_set.count = 0;
    param_set.items = nullptr;

    EdgeRunnerNodeInputs inputs{};
    inputs.audio = audio_view;
    inputs.params = param_set;

    double *out_ptr = nullptr;
    int out_channels = 0;
    void *state = nullptr;
    AmpNodeMetrics metrics{};

    int rc = amp_run_node_v2(
        &descriptor,
        &inputs,
        1,
        1,
        frames,
        sample_rate,
        &out_ptr,
        &out_channels,
        &state,
        nullptr,
        AMP_EXECUTION_MODE_FORWARD,
        &metrics
    );
    assert(rc == 0);
    assert(out_ptr != nullptr);
    assert(out_channels == 1);

    output.assign(out_ptr, out_ptr + frames);

    amp_free(out_ptr);
    amp_release_state(state);
}

} // namespace

int main() {
    constexpr int kFrames = 4096;
    constexpr double kSampleRate = 48000.0;
    constexpr double kBaseFrequency = 440.0;

    std::vector<double> input(kFrames);
    for (int i = 0; i < kFrames; ++i) {
        double t = static_cast<double>(i) / kSampleRate;
        input[i] = std::sin(2.0 * M_PI * kBaseFrequency * t);
    }

    std::vector<double> identity_output;
    run_pitch_shift(1.0, input, kFrames, kSampleRate, identity_output);
    assert(identity_output.size() == static_cast<size_t>(kFrames));

    double input_energy = compute_energy(input);
    double identity_energy = compute_energy(identity_output);
    double energy_ratio = identity_energy / (input_energy + 1e-12);
    assert(energy_ratio > 0.85 && energy_ratio < 1.15);

    std::vector<double> octave_output;
    run_pitch_shift(2.0, input, kFrames, kSampleRate, octave_output);
    assert(octave_output.size() == static_cast<size_t>(kFrames));

    int skip = 1024;
    if (skip > kFrames / 2) {
        skip = kFrames / 2;
    }
    double measured = estimate_frequency(octave_output.data(), kFrames, kSampleRate, skip);
    double expected = kBaseFrequency * 2.0;
    assert(measured > 0.0);
    double rel_error = std::fabs(measured - expected) / expected;
    assert(rel_error < 0.05);

    std::printf("test_pitch_shift_node: PASS (energy and frequency checks OK)\n");
    return 0;
}
