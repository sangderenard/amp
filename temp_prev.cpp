#include <cstdarg>#include <cstdarg>#include <algorithm>#ifdef NDEBUG

#include <cstdio>

#include <cstring>#include <cstdio>

#include <vector>

#include <string>#include <cstring>#include <cmath>#undef NDEBUG

#include <cmath>

#include <vector>

extern "C" {

#include "amp_fft_backend.h"#include <cmath>#include <cstdarg>#endif

#include "amp_native.h"

}



namespace {extern "C" {#include <cstdint>



constexpr int kFrames = 8;#include "amp_fft_backend.h"

constexpr int kWindowSize = 4;

#include "amp_native.h"#include <cstdio>#include <algorithm>

bool g_failed = false;

}

void fail(const char *fmt, ...) {

    g_failed = true;#include <cstring>#include <cmath>

    std::fprintf(stderr, "[fft_division_node] ");

    va_list args;namespace {

    va_start(args, fmt);

    std::vfprintf(stderr, fmt, args);#include <deque>#include <cstdarg>

    va_end(args);

    std::fprintf(stderr, "\n");constexpr int kFrames = 8;

}

constexpr int kWindowSize = 4;#include <string>#include <cstdint>

bool nearly_equal(double a, double b, double tol = 1e-9) {

    return std::fabs(a - b) <= tol;

}

bool g_failed = false;#include <vector>#include <cstdio>

struct RunResult {

    std::vector<double> pcm;

    std::vector<double> spectral_real;

    std::vector<double> spectral_imag;void fail(const char *fmt, ...) {#include <cstring>

};

    g_failed = true;

RunResult run_fft_node_once(const std::vector<double> &signal) {

    RunResult result;    std::fprintf(stderr, "[fft_division_node] ");extern "C" {#include <string>

    result.pcm.assign(signal.size(), 0.0);

    result.spectral_real.assign(signal.size() * kWindowSize, 0.0);    va_list args;

    result.spectral_imag.assign(signal.size() * kWindowSize, 0.0);

    va_start(args, fmt);#include "amp_fft_backend.h"#include <vector>

    static const char *kName = "fft_division_node";

    static const char *kType = "FFTDivisionNode";    std::vfprintf(stderr, fmt, args);

    const std::string params_json =

        "{\"window_size\":" + std::to_string(kWindowSize) +    va_end(args);#include "amp_native.h"#include <complex>

        ",\"algorithm\":\"fft\",\"window\":\"hann\"}";

    std::fprintf(stderr, "\n");

    EdgeRunnerNodeDescriptor descriptor{};

    descriptor.name = kName;}}

    descriptor.name_len = std::strlen(kName);

    descriptor.type_name = kType;

    descriptor.type_len = std::strlen(kType);

    descriptor.params_json = params_json.c_str();bool nearly_equal(double a, double b, double tol = 1e-9) {#include <unsupported/Eigen/FFT>

    descriptor.params_len = params_json.size();

    return std::fabs(a - b) <= tol;

    EdgeRunnerAudioView audio{};

    audio.has_audio = 1U;}namespace {

    audio.batches = 1U;

    audio.channels = 1U;

    audio.frames = static_cast<uint32_t>(signal.size());

    audio.data = signal.data();struct RunResult {extern "C" {



    EdgeRunnerParamSet params{};    std::vector<double> pcm;

    params.count = 0U;

    params.items = nullptr;    std::vector<double> spectral_real;constexpr int kFrames = 8;#include "amp_native.h"



    EdgeRunnerTapBuffer tap_buffers[2]{};    std::vector<double> spectral_imag;

    tap_buffers[0].tap_name = "spectral_0";

    tap_buffers[0].shape.batches = 1U;};constexpr int kWindowSize = 4;}

    tap_buffers[0].shape.channels = kWindowSize;

    tap_buffers[0].shape.frames = audio.frames;

    tap_buffers[0].frame_stride = kWindowSize;

    tap_buffers[0].data = result.spectral_real.data();RunResult run_fft_node_once(const std::vector<double> &signal) {



    tap_buffers[1].tap_name = "spectral_0";    RunResult result;

    tap_buffers[1].shape = tap_buffers[0].shape;

    tap_buffers[1].frame_stride = kWindowSize;    result.pcm.assign(signal.size(), 0.0);bool g_test_failed = false;namespace {

    tap_buffers[1].data = result.spectral_imag.data();

    result.spectral_real.assign(signal.size() * kWindowSize, 0.0);

    EdgeRunnerTapBufferSet tap_set{};

    tap_set.items = tap_buffers;    result.spectral_imag.assign(signal.size() * kWindowSize, 0.0);

    tap_set.count = 2U;



    EdgeRunnerTapContext tap_context{};

    tap_context.outputs = tap_set;    static const char *kName = "fft_division_node";void record_failure(const char *fmt, ...) {constexpr int kFrames = 8;



    EdgeRunnerNodeInputs inputs{};    static const char *kType = "FFTDivisionNode";

    inputs.audio = audio;

    inputs.params = params;    const std::string params_json =    g_test_failed = true;constexpr int kWindowSize = 4;

    inputs.taps = tap_context;

        "{\"window_size\":" + std::to_string(kWindowSize) +

    AmpNodeMetrics metrics{};

    double *out_buffer = nullptr;        ",\"algorithm\":\"fft\",\"window\":\"hann\"}";    std::va_list args;

    int out_channels = 0;

    void *state = nullptr;



    int rc = amp_run_node_v2(    EdgeRunnerNodeDescriptor descriptor{};    va_start(args, fmt);constexpr int FFT_ALGORITHM_EIGEN = 0;

        &descriptor,

        &inputs,    descriptor.name = kName;

        1,

        1,    descriptor.name_len = std::strlen(kName);    std::vfprintf(stderr, fmt, args);constexpr int FFT_ALGORITHM_DFT = 1;

        static_cast<int>(signal.size()),

        48000.0,    descriptor.type_name = kType;

        &out_buffer,

        &out_channels,    descriptor.type_len = std::strlen(kType);    va_end(args);constexpr int FFT_ALGORITHM_DYNAMIC_OSCILLATORS = 2;

        &state,

        nullptr,    descriptor.params_json = params_json.c_str();

        AMP_EXECUTION_MODE_FORWARD,

        &metrics);    descriptor.params_len = params_json.size();    std::fprintf(stderr, "\n");constexpr int FFT_ALGORITHM_HOOK = 3;



    if (rc != 0 || out_buffer == nullptr || out_channels != 1) {

        fail("amp_run_node_v2 forward failed rc=%d buffer=%p channels=%d", rc, static_cast<void *>(out_buffer), out_channels);

    } else {    EdgeRunnerAudioView audio{};}

        result.pcm.assign(out_buffer, out_buffer + signal.size());

    }    audio.has_audio = 1U;



    if (out_buffer != nullptr) {    audio.batches = 1U;constexpr int FFT_WINDOW_RECTANGULAR = 0;

        amp_free(out_buffer);

    }    audio.channels = 1U;

    if (state != nullptr) {

        amp_release_state(state);    audio.frames = static_cast<uint32_t>(signal.size());struct SimulationResult {constexpr int FFT_WINDOW_HANN = 1;

    }

    audio.data = signal.data();

    return result;

}    std::vector<double> pcm;constexpr int FFT_WINDOW_HAMMING = 2;



void require_identity(const std::vector<double> &input, const std::vector<double> &output, const char *label) {    EdgeRunnerParamSet params{};

    if (input.size() != output.size()) {

        fail("%s size mismatch input=%zu output=%zu", label, input.size(), output.size());    params.count = 0U;    std::vector<double> spectral_real;

        return;

    }    params.items = nullptr;

    for (size_t i = 0; i < input.size(); ++i) {

        if (!nearly_equal(input[i], output[i])) {    std::vector<double> spectral_imag;bool g_test_failed = false;

            fail("%s mismatch at frame %zu got %.12f expected %.12f", label, i, output[i], input[i]);

            return;    EdgeRunnerTapBuffer tap_buffers[2]{};

        }

    }    tap_buffers[0].tap_name = "spectral_real";};

}

    tap_buffers[0].shape.batches = 1U;

void require_equal(const std::vector<double> &a, const std::vector<double> &b, const char *label) {

    if (a.size() != b.size()) {    tap_buffers[0].shape.channels = kWindowSize;void record_failure(const char *fmt, ...) {

        fail("%s size mismatch first=%zu second=%zu", label, a.size(), b.size());

        return;    tap_buffers[0].shape.frames = audio.frames;

    }

    for (size_t i = 0; i < a.size(); ++i) {    tap_buffers[0].frame_stride = kWindowSize;SimulationResult simulate_stream_identity(const std::vector<double> &signal, int window_kind) {    g_test_failed = true;

        if (!nearly_equal(a[i], b[i])) {

            fail("%s mismatch at index %zu got %.12f expected %.12f", label, i, b[i], a[i]);    tap_buffers[0].data = result.spectral_real.data();

            return;

        }        std::vector<double> pcm;

    }

}    tap_buffers[1].tap_name = "spectral_imag";



}  // namespace    tap_buffers[1].shape = tap_buffers[0].shape;    result.pcm.assign(signal.size(), 0.0);    va_list args;



int main() {    tap_buffers[1].frame_stride = kWindowSize;

    const std::vector<double> signal{1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125};

    tap_buffers[1].data = result.spectral_imag.data();    result.spectral_real.assign(signal.size() * kWindowSize, 0.0);    va_start(args, fmt);

    RunResult first = run_fft_node_once(signal);

    RunResult second = run_fft_node_once(signal);



    require_identity(signal, first.pcm, "forward_identity_first");    EdgeRunnerTapBufferSet tap_set{};    result.spectral_imag.assign(signal.size() * kWindowSize, 0.0);    std::vfprintf(stderr, fmt, args);

    require_identity(signal, second.pcm, "forward_identity_second");

    require_equal(first.pcm, second.pcm, "pcm_repeat_stability");    tap_set.items = tap_buffers;

    require_equal(first.spectral_real, second.spectral_real, "spectral_real_repeat_stability");

    require_equal(first.spectral_imag, second.spectral_imag, "spectral_imag_repeat_stability");    tap_set.count = 2U;    va_end(args);



    if (g_failed) {

        std::printf("test_fft_division_node: FAIL\n");

        return 1;    EdgeRunnerTapContext tap_context{};    void *forward = amp_fft_backend_stream_create(kWindowSize, kWindowSize, 1, window_kind);    std::fprintf(stderr, "\n");

    }

    std::printf("test_fft_division_node: PASS\n");    tap_context.outputs = tap_set;

    return 0;

}    void *inverse = amp_fft_backend_stream_create_inverse(kWindowSize, kWindowSize, 1, window_kind);}


    EdgeRunnerNodeInputs inputs{};

    inputs.audio = audio;    if (forward == nullptr || inverse == nullptr) {

    inputs.params = params;

    inputs.taps = tap_context;        record_failure("amp_fft_backend_stream_create failed (forward=%p inverse=%p)", forward, inverse);struct ParamDescriptor {



    AmpNodeMetrics metrics{};        if (forward != nullptr) {    std::string name;

    double *out_buffer = nullptr;

    int out_channels = 0;            amp_fft_backend_stream_destroy(forward);    uint32_t batches{1};

    void *state = nullptr;

        }    uint32_t channels{1};

    int rc = amp_run_node_v2(

        &descriptor,        if (inverse != nullptr) {    uint32_t frames{1};

        &inputs,

        1,            amp_fft_backend_stream_destroy(inverse);    std::vector<double> data;

        1,

        static_cast<int>(signal.size()),        }};

        48000.0,

        &out_buffer,        return result;

        &out_channels,

        &state,    }void append_u32(std::vector<uint8_t> &buffer, uint32_t value) {

        nullptr,

        AMP_EXECUTION_MODE_FORWARD,    buffer.push_back(static_cast<uint8_t>(value & 0xFFU));

        &metrics);

    std::vector<double> spectral_real(kWindowSize, 0.0);    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFFU));

    if (rc != 0 || out_buffer == nullptr || out_channels != 1) {

        fail("amp_run_node_v2 forward failed rc=%d buffer=%p channels=%d", rc, static_cast<void *>(out_buffer), out_channels);    std::vector<double> spectral_imag(kWindowSize, 0.0);    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFFU));

    } else {

        result.pcm.assign(out_buffer, out_buffer + signal.size());    std::vector<double> inverse_scratch(kWindowSize, 0.0);    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFFU));

    }

    std::deque<double> pending_pcm;}

    if (out_buffer != nullptr) {

        amp_free(out_buffer);    bool warmup_complete = false;

    }

    if (state != nullptr) {void append_u64(std::vector<uint8_t> &buffer, uint64_t value) {

        amp_release_state(state);

    }    for (size_t frame = 0; frame < signal.size(); ++frame) {    for (int i = 0; i < 8; ++i) {



    return result;        double sample = signal[frame];        buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFU));

}

        size_t frames_emitted = amp_fft_backend_stream_push(    }

void require_identity(const std::vector<double> &input, const std::vector<double> &output, const char *label) {

    if (input.size() != output.size()) {            forward,}

        fail("%s size mismatch input=%zu output=%zu", label, input.size(), output.size());

        return;            &sample,

    }

    for (size_t i = 0; i < input.size(); ++i) {            1,void append_string(std::vector<uint8_t> &buffer, const std::string &value) {

        if (!nearly_equal(input[i], output[i])) {

            fail("%s mismatch at frame %zu got %.12f expected %.12f", label, i, output[i], input[i]);            kWindowSize,    buffer.insert(buffer.end(), value.begin(), value.end());

            return;

        }            spectral_real.data(),}

    }

}            spectral_imag.data(),



void require_equal(const std::vector<double> &a, const std::vector<double> &b, const char *label) {            1,void append_doubles(std::vector<uint8_t> &buffer, const std::vector<double> &values) {

    if (a.size() != b.size()) {

        fail("%s size mismatch first=%zu second=%zu", label, a.size(), b.size());            AMP_FFT_STREAM_FLUSH_NONE);    for (double v : values) {

        return;

    }        uint64_t bits = 0;

    for (size_t i = 0; i < a.size(); ++i) {

        if (!nearly_equal(a[i], b[i])) {        if (frames_emitted > 0) {        std::memcpy(&bits, &v, sizeof(double));

            fail("%s mismatch at index %zu got %.12f expected %.12f", label, i, b[i], a[i]);

            return;            warmup_complete = true;        append_u64(buffer, bits);

        }

    }        }    }

}

}

}  // namespace

        if (!warmup_complete) {

int main() {

    const std::vector<double> signal{1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125};            result.pcm[frame] = sample;void append_node(



    RunResult first = run_fft_node_once(signal);            continue;    std::vector<uint8_t> &buffer,

    RunResult second = run_fft_node_once(signal);

        }    const std::string &name,

    require_identity(signal, first.pcm, "forward_identity_first");

    require_identity(signal, second.pcm, "forward_identity_second");    const std::string &type_name,

    require_equal(first.pcm, second.pcm, "pcm_repeat_stability");

    require_equal(first.spectral_real, second.spectral_real, "spectral_real_repeat_stability");        if (frames_emitted == 0) {    const std::vector<std::string> &audio_inputs,

    require_equal(first.spectral_imag, second.spectral_imag, "spectral_imag_repeat_stability");

            std::fill(spectral_real.begin(), spectral_real.end(), 0.0);    const std::string &params_json,

    if (g_failed) {

        std::printf("test_fft_division_node: FAIL\n");            std::fill(spectral_imag.begin(), spectral_imag.end(), 0.0);    const std::vector<ParamDescriptor> &params

        return 1;

    }        }) {

    std::printf("test_fft_division_node: PASS\n");

    return 0;    append_u32(buffer, 0U);

}

        double *real_slot = result.spectral_real.data() + frame * kWindowSize;    append_u32(buffer, static_cast<uint32_t>(name.size()));

        double *imag_slot = result.spectral_imag.data() + frame * kWindowSize;    append_u32(buffer, static_cast<uint32_t>(type_name.size()));

        std::copy(spectral_real.begin(), spectral_real.end(), real_slot);    append_u32(buffer, static_cast<uint32_t>(audio_inputs.size()));

        std::copy(spectral_imag.begin(), spectral_imag.end(), imag_slot);    append_u32(buffer, 0U);

    append_u32(buffer, static_cast<uint32_t>(params.size()));

        if (!inverse_scratch.empty()) {    append_u32(buffer, 0U);

            std::fill(inverse_scratch.begin(), inverse_scratch.end(), 0.0);    append_u32(buffer, static_cast<uint32_t>(params_json.size()));

        }

    append_string(buffer, name);

        const size_t produced = amp_fft_backend_stream_push_spectrum(    append_string(buffer, type_name);

            inverse,

            spectral_real.data(),    for (const std::string &src : audio_inputs) {

            spectral_imag.data(),        append_u32(buffer, static_cast<uint32_t>(src.size()));

            frames_emitted,        append_string(buffer, src);

            kWindowSize,    }

            inverse_scratch.data(),

            inverse_scratch.size(),    for (const ParamDescriptor &param : params) {

            AMP_FFT_STREAM_FLUSH_NONE);        uint64_t blob_len = static_cast<uint64_t>(param.data.size() * sizeof(double));

        for (size_t i = 0; i < produced && i < inverse_scratch.size(); ++i) {        append_u32(buffer, static_cast<uint32_t>(param.name.size()));

            pending_pcm.push_back(inverse_scratch[i]);        append_u32(buffer, param.batches);

        }        append_u32(buffer, param.channels);

        append_u32(buffer, param.frames);

        double pcm_value = 0.0;        append_u64(buffer, blob_len);

        if (!pending_pcm.empty()) {        append_string(buffer, param.name);

            pcm_value = pending_pcm.front();        append_doubles(buffer, param.data);

            pending_pcm.pop_front();    }

        }

        result.pcm[frame] = pcm_value;    append_string(buffer, params_json);

    }}



    amp_fft_backend_stream_destroy(forward);bool is_power_of_two(int value) {

    amp_fft_backend_stream_destroy(inverse);    if (value <= 0) {

    return result;        return false;

}    }

    return (value & (value - 1)) == 0;

void verify_close(const char *label, const double *actual, const double *expected, size_t count, double tolerance) {}

    for (size_t i = 0; i < count; ++i) {

        double diff = std::fabs(actual[i] - expected[i]);int clamp_algorithm_kind(int kind) {

        if (diff > tolerance) {    switch (kind) {

            record_failure(        case FFT_ALGORITHM_EIGEN:

                "%s mismatch at index %zu: got %.12f expected %.12f diff %.12f",        case FFT_ALGORITHM_DFT:

                label,        case FFT_ALGORITHM_DYNAMIC_OSCILLATORS:

                i,        case FFT_ALGORITHM_HOOK:

                actual[i],            return kind;

                expected[i],        default:

                diff);            break;

            break;    }

        }    return FFT_ALGORITHM_EIGEN;

    }}

}

int clamp_window_kind(int kind) {

}  // namespace    switch (kind) {

        case FFT_WINDOW_RECTANGULAR:

int main() {        case FFT_WINDOW_HANN:

    const std::vector<double> signal{        case FFT_WINDOW_HAMMING:

        1.0,            return kind;

        -0.5,        default:

        0.25,            break;

        -0.125,    }

        0.0625,    return FFT_WINDOW_RECTANGULAR;

        -0.03125,}

        0.015625,

        -0.0078125void compute_fft_eigen(std::vector<double> &real, std::vector<double> &imag, int inverse) {

    };    const int n = static_cast<int>(real.size());

    using ComplexVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

    SimulationResult expected = simulate_stream_identity(signal, AMP_FFT_WINDOW_HANN);    ComplexVector input(n);

    for (int i = 0; i < n; ++i) {

    std::vector<double> spectral_real_capture(expected.spectral_real.size(), 0.0);        input[i] = std::complex<double>(real[i], imag[i]);

    std::vector<double> spectral_imag_capture(expected.spectral_imag.size(), 0.0);    }

    ComplexVector output(n);

    EdgeRunnerTapBuffer tap_buffers[2]{};    Eigen::FFT<double> fft;

    tap_buffers[0].tap_name = "spectral_real";    fft.SetFlag(Eigen::FFT<double>::Unscaled);

    tap_buffers[0].shape.batches = 1U;    if (inverse != 0) {

    tap_buffers[0].shape.channels = kWindowSize;        fft.inv(output, input);

    tap_buffers[0].shape.frames = kFrames;        output /= static_cast<double>(n);

    tap_buffers[0].frame_stride = kWindowSize;    } else {

    tap_buffers[0].data = spectral_real_capture.data();        fft.fwd(output, input);

    }

    tap_buffers[1].tap_name = "spectral_imag";    for (int i = 0; i < n; ++i) {

    tap_buffers[1].shape.batches = 1U;        real[i] = output[i].real();

    tap_buffers[1].shape.channels = kWindowSize;        imag[i] = output[i].imag();

    tap_buffers[1].shape.frames = kFrames;    }

    tap_buffers[1].frame_stride = kWindowSize;}

    tap_buffers[1].data = spectral_imag_capture.data();

void compute_dft(std::vector<double> &real, std::vector<double> &imag, int inverse) {

    EdgeRunnerTapBufferSet tap_set{};    const int n = static_cast<int>(real.size());

    tap_set.items = tap_buffers;    std::vector<double> out_real(n, 0.0);

    tap_set.count = 2U;    std::vector<double> out_imag(n, 0.0);

    double sign = inverse != 0 ? 1.0 : -1.0;

    EdgeRunnerTapContext tap_context{};    for (int k = 0; k < n; ++k) {

    tap_context.outputs = tap_set;        double sum_real = 0.0;

        double sum_imag = 0.0;

    EdgeRunnerParamSet param_set{};        for (int t = 0; t < n; ++t) {

    param_set.count = 0U;            double angle = sign * 2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);

    param_set.items = nullptr;            double c = std::cos(angle);

            double s = std::sin(angle);

    EdgeRunnerAudioView audio{};            sum_real += real[t] * c - imag[t] * s;

    audio.has_audio = 1U;            sum_imag += real[t] * s + imag[t] * c;

    audio.batches = 1U;        }

    audio.channels = 1U;        if (inverse != 0) {

    audio.frames = kFrames;            sum_real /= static_cast<double>(n);

    audio.data = signal.data();            sum_imag /= static_cast<double>(n);

        }

    EdgeRunnerNodeInputs inputs{};        out_real[k] = sum_real;

    inputs.audio = audio;        out_imag[k] = sum_imag;

    inputs.params = param_set;    }

    inputs.taps = tap_context;    real.swap(out_real);

    imag.swap(out_imag);

    const char *params_json =}

        "{\"window_size\":4,\"algorithm\":\"fft\",\"window\":\"hann\",\"supports_v2\":true,"

        "\"declared_delay\":3,\"oversample_ratio\":1,\"epsilon\":1e-9,\"max_batch_windows\":1,"void fill_window(int window_kind, std::vector<double> &window) {

        "\"backend_mode\":0,\"backend_hop\":1}";    const int size = static_cast<int>(window.size());

    if (size <= 0) {

    EdgeRunnerNodeDescriptor descriptor{};        return;

    descriptor.name = "fft_divider_direct";    }

    descriptor.name_len = std::strlen(descriptor.name);    if (size == 1) {

    descriptor.type_name = "FFTDivisionNode";        window[0] = 1.0;

    descriptor.type_len = std::strlen(descriptor.type_name);        return;

    descriptor.params_json = params_json;    }

    descriptor.params_len = std::strlen(params_json);    for (int i = 0; i < size; ++i) {

        double phase = static_cast<double>(i) / static_cast<double>(size - 1);

    AmpNodeMetrics metrics{};        double value = 1.0;

    double *out_buffer = nullptr;        switch (window_kind) {

    int out_channels = 0;            case FFT_WINDOW_RECTANGULAR:

    void *state = nullptr;                value = 1.0;

                break;

    int rc = amp_run_node_v2(            case FFT_WINDOW_HANN:

        &descriptor,                value = 0.5 * (1.0 - std::cos(2.0 * M_PI * phase));

        &inputs,                break;

        1,            case FFT_WINDOW_HAMMING:

        1,                value = 0.54 - 0.46 * std::cos(2.0 * M_PI * phase);

        kFrames,                break;

        48000.0,            default:

        &out_buffer,                value = 1.0;

        &out_channels,                break;

        &state,        }

        nullptr,        window[i] = value;

        AMP_EXECUTION_MODE_FORWARD,    }

        &metrics);}

    if (rc != 0) {

        record_failure("amp_run_node_v2 forward failed rc=%d", rc);std::vector<double> simulate_fft_reference(

    }    const std::vector<double> &signal,

    if (out_buffer == nullptr) {    const std::vector<int> &algorithm_selector,

        record_failure("amp_run_node_v2 forward returned null buffer");    const std::vector<int> &window_selector,

    }    int window_size,

    if (out_channels != 1) {    double epsilon_default,

        record_failure("amp_run_node_v2 forward produced %d channels (expected 1)", out_channels);    int default_algorithm,

    }    int default_window_kind

    if (state == nullptr) {) {

        record_failure("amp_run_node_v2 forward returned null state");    const int frames = static_cast<int>(signal.size());

    }    std::vector<double> ring(window_size, 0.0);

    int filled = 0;

    if (out_buffer != nullptr) {    std::vector<double> window(window_size, 1.0);

        verify_close("pcm_forward", out_buffer, expected.pcm.data(), expected.pcm.size(), 1e-9);    int current_window_kind = -1;

        for (int frame = 0; frame < kWindowSize - 1 && frame < static_cast<int>(signal.size()); ++frame) {    std::vector<double> output(frames, 0.0);

            if (std::fabs(out_buffer[frame] - signal[frame]) > 1e-9) {

                record_failure(    for (int frame = 0; frame < frames; ++frame) {

                    "warmup frame %d mismatch: got %.12f expected %.12f",        double epsilon = epsilon_default;

                    frame,        if (epsilon < 1e-12) {

                    out_buffer[frame],            epsilon = 1e-12;

                    signal[frame]);        }

                break;        (void)epsilon;

            }

        }        double sample = signal[frame];

    }        if (filled < window_size) {

            ring[filled] = sample;

    verify_close(            filled += 1;

        "spectral_real",        } else {

        spectral_real_capture.data(),            if (window_size > 1) {

        expected.spectral_real.data(),                std::memmove(ring.data(), ring.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));

        expected.spectral_real.size(),            }

        1e-9);            ring[window_size - 1] = sample;

    verify_close(        }

        "spectral_imag",

        spectral_imag_capture.data(),        if (filled < window_size) {

        expected.spectral_imag.data(),            output[frame] = sample;

        expected.spectral_imag.size(),            continue;

        1e-9);        }



    double *backward_out = nullptr;        int algorithm = default_algorithm;

    int backward_channels = 0;        if (!algorithm_selector.empty() && frame < static_cast<int>(algorithm_selector.size())) {

    EdgeRunnerAudioView backward_audio = audio;            algorithm = clamp_algorithm_kind(algorithm_selector[frame]);

    backward_audio.data = out_buffer;        }

    EdgeRunnerNodeInputs backward_inputs = inputs;        if (algorithm == FFT_ALGORITHM_EIGEN && !is_power_of_two(window_size)) {

    backward_inputs.audio = backward_audio;            algorithm = FFT_ALGORITHM_DFT;

        }

    int backward_rc = amp_run_node_v2(

        &descriptor,        int window_kind = default_window_kind;

        &backward_inputs,        if (!window_selector.empty() && frame < static_cast<int>(window_selector.size())) {

        1,            window_kind = clamp_window_kind(window_selector[frame]);

        1,        }

        kFrames,        if (window_kind != current_window_kind) {

        48000.0,            fill_window(window_kind, window);

        &backward_out,            current_window_kind = window_kind;

        &backward_channels,        }

        &state,

        nullptr,        std::vector<double> real(window_size, 0.0);

        AMP_EXECUTION_MODE_BACKWARD,        std::vector<double> imag(window_size, 0.0);

        &metrics);        std::copy(ring.begin(), ring.end(), real.begin());

    if (backward_rc != AMP_E_UNSUPPORTED) {        for (int i = 0; i < window_size; ++i) {

        record_failure("amp_run_node_v2 backward returned %d (expected AMP_E_UNSUPPORTED)", backward_rc);            real[i] *= window[i];

    }        }

    if (backward_out != nullptr) {

        record_failure("amp_run_node_v2 backward produced unexpected buffer");        if (algorithm == FFT_ALGORITHM_EIGEN || algorithm == FFT_ALGORITHM_HOOK) {

        amp_free(backward_out);            compute_fft_eigen(real, imag, 0);

        backward_out = nullptr;            compute_fft_eigen(real, imag, 1);

    }        } else {

            compute_dft(real, imag, 0);

    if (out_buffer != nullptr) {            compute_dft(real, imag, 1);

        amp_free(out_buffer);        }

        out_buffer = nullptr;

    }        const int sample_index = window_size > 0 ? window_size - 1 : 0;

    if (state != nullptr) {        output[frame] = real[sample_index];

        amp_release_state(state);    }

        state = nullptr;

    }    return output;

}

    if (g_test_failed) {

        std::printf("test_fft_division_node: FAIL\n");std::vector<uint8_t> build_descriptor(

        return 1;    const ParamDescriptor &signal_param,

    }    int backend_mode

) {

    std::printf("test_fft_division_node: PASS\n");    std::vector<uint8_t> descriptor;

    return 0;    descriptor.reserve(1024);

}    append_u32(descriptor, 3U);


    append_node(descriptor, "carrier", "ConstantNode", {}, "{\"value\":1.0,\"channels\":1}", {});
    append_node(descriptor, "signal", "GainNode", {"carrier"}, "{}", {signal_param});

    char params_buffer[256];
    std::snprintf(
        params_buffer,
        sizeof(params_buffer),
        "{\"window_size\":%d,\"algorithm\":\"fft\",\"window\":\"hann\",\"supports_v2\":true,\"declared_delay\":%d,\"oversample_ratio\":1,\"epsilon\":1e-9,\"max_batch_windows\":1,\"backend_mode\":%d,\"backend_hop\":1}",
        kWindowSize,
        kWindowSize - 1,
        backend_mode
    );
    append_node(descriptor, "fft", "FFTDivisionNode", {"signal"}, params_buffer, {});
    return descriptor;
}

std::vector<double> run_backend(const std::vector<uint8_t> &descriptor, const char *label) {
    AmpGraphRuntime *runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    if (runtime == nullptr) {
        record_failure("amp_graph_runtime_create failed for %s", label);
        return {};
    }
    if (amp_graph_runtime_configure(runtime, 1U, static_cast<uint32_t>(kFrames)) != 0) {
        record_failure("amp_graph_runtime_configure failed for %s", label);
        amp_graph_runtime_destroy(runtime);
        return {};
    }

    double *out_buffer = nullptr;
    uint32_t out_batches = 0;
    uint32_t out_channels = 0;
    uint32_t out_frames = 0;
    int exec_rc = amp_graph_runtime_execute(
        runtime,
        nullptr,
        0U,
        kFrames,
        48000.0,
        &out_buffer,
        &out_batches,
        &out_channels,
        &out_frames
    );
    if (exec_rc != 0 || out_buffer == nullptr) {
        record_failure("amp_graph_runtime_execute failed for %s rc=%d buffer=%p", label, exec_rc, static_cast<void *>(out_buffer));
        if (out_buffer != nullptr) {
            amp_graph_runtime_buffer_free(out_buffer);
        }
        amp_graph_runtime_destroy(runtime);
        return {};
    }

    size_t total = static_cast<size_t>(out_batches) * out_channels * out_frames;
    std::vector<double> result(out_buffer, out_buffer + total);
    amp_graph_runtime_buffer_free(out_buffer);
    amp_graph_runtime_destroy(runtime);
    return result;
}

void compare_frames(const std::vector<double> &reference, const std::vector<double> &candidate, const char *label) {
    if (reference.size() != candidate.size()) {
        record_failure("%s length mismatch reference=%zu candidate=%zu", label, reference.size(), candidate.size());
        return;
    }
    constexpr double kTolerance = 1e-7;
    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = std::fabs(reference[i] - candidate[i]);
        if (diff > kTolerance) {
            record_failure("%s frame %zu differs %.12f vs %.12f", label, i, candidate[i], reference[i]);
            return;
        }
    }
}

}  // namespace

int main() {
    const std::vector<double> signal{
        1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125
    };
    const std::vector<int> algorithm_selector(kFrames, FFT_ALGORITHM_EIGEN);
    const std::vector<int> window_selector(kFrames, FFT_WINDOW_HANN);

    std::vector<double> expected = simulate_fft_reference(
        signal,
        algorithm_selector,
        window_selector,
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );

    ParamDescriptor signal_param{
        "gain",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        signal
    };

    std::vector<uint8_t> descriptor_mode0 = build_descriptor(signal_param, 0);
    std::vector<uint8_t> descriptor_mode1 = build_descriptor(signal_param, 1);
    std::vector<uint8_t> descriptor_mode2 = build_descriptor(signal_param, 2);

    std::vector<double> candidate0 = run_backend(descriptor_mode0, "backend_mode_amp_window");
    if (!candidate0.empty()) {
        compare_frames(expected, candidate0, "backend_mode_amp_window");
        std::vector<double> candidate0_repeat = run_backend(descriptor_mode0, "backend_mode_amp_window_repeat");
        if (!candidate0_repeat.empty()) {
            compare_frames(candidate0, candidate0_repeat, "backend_mode_amp_window_repeat_stability");
        }
    }
    std::vector<double> candidate1 = run_backend(descriptor_mode1, "backend_mode_fftfree_batched");
    if (!candidate1.empty()) {
        compare_frames(expected, candidate1, "backend_mode_fftfree_batched");
        std::vector<double> candidate1_repeat = run_backend(descriptor_mode1, "backend_mode_fftfree_batched_repeat");
        if (!candidate1_repeat.empty()) {
            compare_frames(candidate1, candidate1_repeat, "backend_mode_fftfree_batched_repeat_stability");
        }
    }
    std::vector<double> candidate2 = run_backend(descriptor_mode2, "backend_mode_fftfree_inline");
    if (!candidate2.empty()) {
        compare_frames(expected, candidate2, "backend_mode_fftfree_inline");
        std::vector<double> candidate2_repeat = run_backend(descriptor_mode2, "backend_mode_fftfree_inline_repeat");
        if (!candidate2_repeat.empty()) {
            compare_frames(candidate2, candidate2_repeat, "backend_mode_fftfree_inline_repeat_stability");
        }
    }

    return g_test_failed ? 1 : 0;
}
