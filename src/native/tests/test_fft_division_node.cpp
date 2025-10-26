#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <complex>

#include <unsupported/Eigen/FFT>

extern "C" {
#include "amp_native.h"
}

namespace {

constexpr int FFT_ALGORITHM_EIGEN = 0;
constexpr int FFT_ALGORITHM_DFT = 1;
constexpr int FFT_ALGORITHM_DYNAMIC_OSCILLATORS = 2;
constexpr int FFT_ALGORITHM_HOOK = 3;

constexpr int FFT_DYNAMIC_CARRIER_LIMIT = 64;

constexpr int FFT_WINDOW_RECTANGULAR = 0;
constexpr int FFT_WINDOW_HANN = 1;
constexpr int FFT_WINDOW_HAMMING = 2;

struct ParamDescriptor {
    std::string name;
    uint32_t batches{1};
    uint32_t channels{1};
    uint32_t frames{1};
    std::vector<double> data;
};

void append_u32(std::vector<uint8_t> &buffer, uint32_t value) {
    buffer.push_back(static_cast<uint8_t>(value & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFFU));
    buffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFFU));
}

void append_u64(std::vector<uint8_t> &buffer, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFFU));
    }
}

void append_string(std::vector<uint8_t> &buffer, const std::string &value) {
    buffer.insert(buffer.end(), value.begin(), value.end());
}

void append_doubles(std::vector<uint8_t> &buffer, const std::vector<double> &values) {
    for (double v : values) {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(double));
        append_u64(buffer, bits);
    }
}

void append_node(
    std::vector<uint8_t> &buffer,
    const std::string &name,
    const std::string &type_name,
    const std::vector<std::string> &audio_inputs,
    const std::string &params_json,
    const std::vector<ParamDescriptor> &params
) {
    append_u32(buffer, 0U); // type_id placeholder
    append_u32(buffer, static_cast<uint32_t>(name.size()));
    append_u32(buffer, static_cast<uint32_t>(type_name.size()));
    append_u32(buffer, static_cast<uint32_t>(audio_inputs.size()));
    append_u32(buffer, 0U); // mod connections
    append_u32(buffer, static_cast<uint32_t>(params.size()));
    append_u32(buffer, 0U); // shape count
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

int round_to_int(double value) {
    if (value >= 0.0) {
        return static_cast<int>(value + 0.5);
    }
    return static_cast<int>(value - 0.5);
}

int clamp_algorithm_kind(int kind) {
    switch (kind) {
        case FFT_ALGORITHM_EIGEN:
        case FFT_ALGORITHM_DFT:
        case FFT_ALGORITHM_DYNAMIC_OSCILLATORS:
        case FFT_ALGORITHM_HOOK:
            return kind;
        default:
            break;
    }
    return FFT_ALGORITHM_EIGEN;
}

int clamp_window_kind(int kind) {
    switch (kind) {
        case FFT_WINDOW_RECTANGULAR:
        case FFT_WINDOW_HANN:
        case FFT_WINDOW_HAMMING:
            return kind;
        default:
            break;
    }
    return FFT_WINDOW_HANN;
}

double clamp_unit_double(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

double wrap_phase_two_pi(double phase) {
    double wrapped = std::fmod(phase, 2.0 * M_PI);
    if (wrapped < 0.0) {
        wrapped += 2.0 * M_PI;
    }
    return wrapped;
}

double compute_band_gain(double ratio, double lower, double upper, double intensity) {
    double lower_clamped = clamp_unit_double(lower);
    double upper_clamped = clamp_unit_double(upper);
    if (upper_clamped < lower_clamped) {
        std::swap(lower_clamped, upper_clamped);
    }
    double intensity_clamped = clamp_unit_double(intensity);
    double inside_gain = intensity_clamped;
    if (inside_gain < 1e-6) {
        inside_gain = 1e-6;
    }
    double outside_gain = 1.0 - intensity_clamped;
    if (outside_gain < 1e-6) {
        outside_gain = 1e-6;
    }
    if (ratio >= lower_clamped && ratio <= upper_clamped) {
        return inside_gain;
    }
    return outside_gain;
}

bool is_power_of_two(int value) {
    if (value <= 0) {
        return false;
    }
    return (value & (value - 1)) == 0;
}

void compute_fft_eigen(std::vector<double> &real, std::vector<double> &imag, int inverse) {
    const int n = static_cast<int>(real.size());
    using ComplexVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;
    ComplexVector input(n);
    for (int i = 0; i < n; ++i) {
        input[i] = std::complex<double>(real[i], imag[i]);
    }
    ComplexVector output(n);
    Eigen::FFT<double> fft;
    if (inverse != 0) {
        fft.inv(output, input);
    } else {
        fft.fwd(output, input);
    }
    for (int i = 0; i < n; ++i) {
        real[i] = output[i].real();
        imag[i] = output[i].imag();
    }
}

void compute_dft(std::vector<double> &real, std::vector<double> &imag, int inverse) {
    const int n = static_cast<int>(real.size());
    std::vector<double> out_real(n, 0.0);
    std::vector<double> out_imag(n, 0.0);
    double sign = inverse != 0 ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (int t = 0; t < n; ++t) {
            double angle = sign * 2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
            double c = std::cos(angle);
            double s = std::sin(angle);
            sum_real += real[t] * c - imag[t] * s;
            sum_imag += real[t] * s + imag[t] * c;
        }
        if (inverse != 0) {
            sum_real /= static_cast<double>(n);
            sum_imag /= static_cast<double>(n);
        }
        out_real[k] = sum_real;
        out_imag[k] = sum_imag;
    }
    real.swap(out_real);
    imag.swap(out_imag);
}

// Mirrors the native `compute_dft` helper when callers alias the input and output
// buffers. The runtime relies on this behaviour for the dynamic oscillator path,
// so the simulator must reproduce the same destructive updates to stay in lockstep.
void compute_dft_inplace_alias(std::vector<double> &real, std::vector<double> &imag, int inverse) {
    const int n = static_cast<int>(real.size());
    if (n <= 0) {
        return;
    }
    double sign = inverse != 0 ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_real = 0.0;
        double sum_imag = 0.0;
        for (int t = 0; t < n; ++t) {
            double angle = sign * 2.0 * M_PI * static_cast<double>(k * t) / static_cast<double>(n);
            double c = std::cos(angle);
            double s = std::sin(angle);
            double in_real = real[t];
            double in_imag = imag[t];
            sum_real += in_real * c - in_imag * s;
            sum_imag += in_real * s + in_imag * c;
        }
        if (inverse != 0) {
            sum_real /= static_cast<double>(n);
            sum_imag /= static_cast<double>(n);
        }
        real[k] = sum_real;
        imag[k] = sum_imag;
    }
}

void fill_window(int window_kind, std::vector<double> &window) {
    const int size = static_cast<int>(window.size());
    if (size <= 0) {
        return;
    }
    if (size == 1) {
        window[0] = 1.0;
        return;
    }
    for (int i = 0; i < size; ++i) {
        double phase = static_cast<double>(i) / static_cast<double>(size - 1);
        double value = 1.0;
        switch (window_kind) {
            case FFT_WINDOW_RECTANGULAR:
                value = 1.0;
                break;
            case FFT_WINDOW_HANN:
                value = 0.5 * (1.0 - std::cos(2.0 * M_PI * phase));
                break;
            case FFT_WINDOW_HAMMING:
                value = 0.54 - 0.46 * std::cos(2.0 * M_PI * phase);
                break;
            default:
                value = 1.0;
                break;
        }
        window[i] = value;
    }
}

std::vector<double> simulate_fft_division(
    const std::vector<double> &signal,
    const std::vector<double> &divisor_real,
    const std::vector<double> &divisor_imag,
    const std::vector<int> &algorithm_selector,
    const std::vector<int> &window_selector,
    const std::vector<double> &stabilizer,
    const std::vector<double> &phase,
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::vector<double> &filter,
    const std::vector<std::vector<double>> &dynamic_carriers,
    int window_size,
    double epsilon_default,
    int default_algorithm,
    int default_window_kind
) {
    const int frames = static_cast<int>(signal.size());
    std::vector<double> buffer(window_size, 0.0);
    std::vector<double> divisor_buf(window_size, 1.0);
    std::vector<double> divisor_imag_buf(window_size, 0.0);
    std::vector<double> phase_buf(window_size, 0.0);
    std::vector<double> lower_buf(window_size, 0.0);
    std::vector<double> upper_buf(window_size, 1.0);
    std::vector<double> filter_buf(window_size, 1.0);
    std::vector<double> window(window_size, 1.0);
    std::vector<double> dynamic_phase(FFT_DYNAMIC_CARRIER_LIMIT, 0.0);
    int current_window_kind = -1;
    int filled = 0;
    const double sample_rate_hz = 48000.0;

    auto forward_transform = [&](int algorithm, std::vector<double> &real, std::vector<double> &imag) {
        if (algorithm == FFT_ALGORITHM_EIGEN || algorithm == FFT_ALGORITHM_HOOK) {
            compute_fft_eigen(real, imag, 0);
        } else {
            compute_dft(real, imag, 0);
        }
    };

    auto inverse_transform = [&](int algorithm, std::vector<double> &real, std::vector<double> &imag) {
        if (algorithm == FFT_ALGORITHM_EIGEN || algorithm == FFT_ALGORITHM_HOOK) {
            compute_fft_eigen(real, imag, 1);
        } else {
            compute_dft(real, imag, 1);
        }
    };

    std::vector<double> output(frames, 0.0);
    for (int frame = 0; frame < frames; ++frame) {
        double epsilon = epsilon_default;
        if (!stabilizer.empty() && frame < static_cast<int>(stabilizer.size())) {
            double candidate = stabilizer[frame];
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon = candidate;
            }
        }
        if (epsilon < 1e-12) {
            epsilon = 1e-12;
        }
        int algorithm = default_algorithm;
        if (!algorithm_selector.empty() && frame < static_cast<int>(algorithm_selector.size())) {
            algorithm = clamp_algorithm_kind(algorithm_selector[frame]);
        }
        if (algorithm == FFT_ALGORITHM_EIGEN && !is_power_of_two(window_size)) {
            algorithm = FFT_ALGORITHM_DFT;
        }
        int window_kind = default_window_kind;
        if (!window_selector.empty() && frame < static_cast<int>(window_selector.size())) {
            window_kind = clamp_window_kind(window_selector[frame]);
        }
        if (window_kind != current_window_kind) {
            fill_window(window_kind, window);
            current_window_kind = window_kind;
        }

        double current_sample = signal[frame];
        double current_divisor = frame < static_cast<int>(divisor_real.size()) ? divisor_real[frame] : 1.0;
        double current_divisor_imag = frame < static_cast<int>(divisor_imag.size()) ? divisor_imag[frame] : 0.0;
        double current_phase = (!phase.empty() && frame < static_cast<int>(phase.size()))
            ? phase[frame]
            : (filled > 0 ? phase_buf[filled - 1] : 0.0);
        double current_lower = (!lower.empty() && frame < static_cast<int>(lower.size()))
            ? lower[frame]
            : (filled > 0 ? lower_buf[filled - 1] : 0.0);
        double current_upper = (!upper.empty() && frame < static_cast<int>(upper.size()))
            ? upper[frame]
            : (filled > 0 ? upper_buf[filled - 1] : 1.0);
        double current_filter = (!filter.empty() && frame < static_cast<int>(filter.size()))
            ? filter[frame]
            : (filled > 0 ? filter_buf[filled - 1] : 1.0);

        if (filled < window_size) {
            buffer[filled] = current_sample;
            divisor_buf[filled] = current_divisor;
            divisor_imag_buf[filled] = current_divisor_imag;
            phase_buf[filled] = current_phase;
            lower_buf[filled] = current_lower;
            upper_buf[filled] = current_upper;
            filter_buf[filled] = current_filter;
            filled += 1;
        } else {
            if (window_size > 1) {
                std::memmove(buffer.data(), buffer.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(divisor_buf.data(), divisor_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(divisor_imag_buf.data(), divisor_imag_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(phase_buf.data(), phase_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(lower_buf.data(), lower_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(upper_buf.data(), upper_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
                std::memmove(filter_buf.data(), filter_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            }
            buffer[window_size - 1] = current_sample;
            divisor_buf[window_size - 1] = current_divisor;
            divisor_imag_buf[window_size - 1] = current_divisor_imag;
            phase_buf[window_size - 1] = current_phase;
            lower_buf[window_size - 1] = current_lower;
            upper_buf[window_size - 1] = current_upper;
            filter_buf[window_size - 1] = current_filter;
        }

        if (filled < window_size) {
            double safe_div = std::fabs(current_divisor) < epsilon ? (current_divisor >= 0.0 ? epsilon : -epsilon) : current_divisor;
            output[frame] = current_sample / safe_div;
            continue;
        }

        std::vector<double> signal_real(window_size, 0.0);
        std::vector<double> signal_imag(window_size, 0.0);
        std::vector<double> divisor_real_fft(window_size, 0.0);
        std::vector<double> divisor_imag_fft(window_size, 0.0);
        double phase_mod = phase_buf[window_size > 0 ? window_size - 1 : 0];
        double lower_mod = lower_buf[window_size > 0 ? window_size - 1 : 0];
        double upper_mod = upper_buf[window_size > 0 ? window_size - 1 : 0];
        double filter_mod = filter_buf[window_size > 0 ? window_size - 1 : 0];
        double lower_clamped = clamp_unit_double(lower_mod);
        double upper_clamped = clamp_unit_double(upper_mod);
        if (upper_clamped < lower_clamped) {
            std::swap(lower_clamped, upper_clamped);
        }
        double intensity_clamped = clamp_unit_double(filter_mod);
        double cos_phase = std::cos(phase_mod);
        double sin_phase = std::sin(phase_mod);

        if (algorithm == FFT_ALGORITHM_DYNAMIC_OSCILLATORS) {
            std::vector<double> carrier_fnorms;
            std::vector<int> carrier_indices;
            carrier_fnorms.reserve(dynamic_carriers.size());
            carrier_indices.reserve(dynamic_carriers.size());
            for (size_t idx = 0; idx < dynamic_carriers.size() && idx < static_cast<size_t>(FFT_DYNAMIC_CARRIER_LIMIT); ++idx) {
                const auto &series = dynamic_carriers[idx];
                if (series.empty()) {
                    continue;
                }
                double raw_value = frame < static_cast<int>(series.size()) ? series[frame] : series.back();
                double normalized = raw_value;
                if (sample_rate_hz > 0.0 && std::fabs(normalized) > 1.0) {
                    normalized = raw_value / sample_rate_hz;
                }
                normalized = clamp_unit_double(normalized);
                carrier_fnorms.push_back(normalized);
                carrier_indices.push_back(static_cast<int>(idx));
            }
            double sample_dynamic = 0.0;
            if (!carrier_indices.empty()) {
                double inv_window = window_size > 0 ? 1.0 / static_cast<double>(window_size) : 1.0;
                bool has_divisor = !divisor_real.empty() || !divisor_imag.empty();
                for (size_t k = 0; k < carrier_indices.size(); ++k) {
                    size_t carrier_index = static_cast<size_t>(carrier_indices[k]);
                    double theta = dynamic_phase[carrier_index];
                    double ph_re = std::cos(theta);
                    double ph_im = std::sin(theta);
                    double fnorm = carrier_fnorms[k];
                    double angle = 2.0 * M_PI * fnorm;
                    double step_re = std::cos(angle);
                    double step_im = std::sin(angle);
                    double coeff_re = 0.0;
                    double coeff_im = 0.0;
                    double div_re_acc = 0.0;
                    double div_im_acc = 0.0;
                    double last_re = ph_re;
                    double last_im = ph_im;
                    for (int n = 0; n < window_size; ++n) {
                        double w = window[n];
                        double xw = buffer[n] * w;
                        coeff_re += xw * ph_re;
                        coeff_im -= xw * ph_im;
                        if (has_divisor) {
                            double dw_re = divisor_buf[n] * w;
                            double dw_im = divisor_imag_buf[n] * w;
                            div_re_acc += dw_re * ph_re + dw_im * ph_im;
                            div_im_acc += -dw_re * ph_im + dw_im * ph_re;
                        }
                        if (n == window_size - 1) {
                            last_re = ph_re;
                            last_im = ph_im;
                        }
                        double next_re = ph_re * step_re - ph_im * step_im;
                        double next_im = ph_re * step_im + ph_im * step_re;
                        ph_re = next_re;
                        ph_im = next_im;
                    }
                    dynamic_phase[carrier_index] = wrap_phase_two_pi(std::atan2(ph_im, ph_re));
                    if (has_divisor) {
                        double denom = div_re_acc * div_re_acc + div_im_acc * div_im_acc;
                        if (denom < epsilon) {
                            denom = epsilon;
                        }
                        double real_tmp = (coeff_re * div_re_acc + coeff_im * div_im_acc) / denom;
                        double imag_tmp = (coeff_im * div_re_acc - coeff_re * div_im_acc) / denom;
                        coeff_re = real_tmp;
                        coeff_im = imag_tmp;
                    }
                    double ratio = clamp_unit_double(fnorm);
                    double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                    double gated_re = coeff_re * gain;
                    double gated_im = coeff_im * gain;
                    double rotated_re = gated_re * cos_phase - gated_im * sin_phase;
                    double rotated_im = gated_re * sin_phase + gated_im * cos_phase;
                    sample_dynamic += (rotated_re * last_re - rotated_im * last_im) * inv_window;
                }
            }
            output[frame] = sample_dynamic;
            continue;
        }

        for (int i = 0; i < window_size; ++i) {
            double w = window[i];
            signal_real[i] = buffer[i] * w;
            signal_imag[i] = 0.0;
            divisor_real_fft[i] = divisor_buf[i] * w;
            divisor_imag_fft[i] = divisor_imag_buf[i] * w;
        }

        forward_transform(algorithm, signal_real, signal_imag);
        forward_transform(algorithm, divisor_real_fft, divisor_imag_fft);

        for (int i = 0; i < window_size; ++i) {
            double a = signal_real[i];
            double b = signal_imag[i];
            double c = divisor_real_fft[i];
            double d = divisor_imag_fft[i];
            double denom = c * c + d * d;
            if (denom < epsilon) {
                denom = epsilon;
            }
            double real_part = (a * c + b * d) / denom;
            double imag_part = (b * c - a * d) / denom;
            double ratio = window_size > 1 ? static_cast<double>(i) / static_cast<double>(window_size - 1) : 0.0;
            double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
            double gated_real = real_part * gain;
            double gated_imag = imag_part * gain;
            double rotated_real = gated_real * cos_phase - gated_imag * sin_phase;
            double rotated_imag = gated_real * sin_phase + gated_imag * cos_phase;
            signal_real[i] = rotated_real;
            signal_imag[i] = rotated_imag;
        }

        inverse_transform(algorithm, signal_real, signal_imag);

        output[frame] = signal_real[window_size - 1];
    }

    return output;
}

std::vector<double> simulate_dynamic_backward(
    const std::vector<double> &signal,
    const std::vector<double> &divisor_real,
    const std::vector<double> &divisor_imag,
    const std::vector<double> &stabilizer,
    const std::vector<double> &phase,
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::vector<double> &filter,
    const std::vector<std::vector<double>> &dynamic_carriers,
    int window_size,
    double epsilon_default,
    double sample_rate_hz
) {
    const int frames = static_cast<int>(signal.size());
    std::vector<double> recomb(window_size, 0.0);
    std::vector<double> divisor_buf(window_size, 1.0);
    std::vector<double> divisor_imag_buf(window_size, 0.0);
    std::vector<double> phase_buf(window_size, 0.0);
    std::vector<double> lower_buf(window_size, 0.0);
    std::vector<double> upper_buf(window_size, 1.0);
    std::vector<double> filter_buf(window_size, 1.0);
    std::vector<double> window(window_size, 1.0);
    std::vector<double> dynamic_phase(FFT_DYNAMIC_CARRIER_LIMIT, 0.0);
    std::vector<double> phasor_re(window_size, 0.0);
    std::vector<double> phasor_im(window_size, 0.0);
    std::vector<double> output(frames, 0.0);
    int filled = 0;
    int current_window_kind = -1;

    for (int frame = 0; frame < frames; ++frame) {
        double epsilon = epsilon_default;
        if (!stabilizer.empty() && frame < static_cast<int>(stabilizer.size())) {
            double candidate = stabilizer[frame];
            if (candidate < 0.0) {
                candidate = -candidate;
            }
            if (candidate > 0.0) {
                epsilon = candidate;
            }
        }
        if (epsilon < 1e-12) {
            epsilon = 1e-12;
        }
        if (current_window_kind != FFT_WINDOW_HANN) {
            fill_window(FFT_WINDOW_HANN, window);
            current_window_kind = FFT_WINDOW_HANN;
        }

        double current_sample = signal[frame];
        double current_divisor = frame < static_cast<int>(divisor_real.size()) ? divisor_real[frame] : 1.0;
        double current_divisor_imag = frame < static_cast<int>(divisor_imag.size()) ? divisor_imag[frame] : 0.0;
        double current_phase = (!phase.empty() && frame < static_cast<int>(phase.size()))
            ? phase[frame]
            : (filled > 0 ? phase_buf[filled - 1] : 0.0);
        double current_lower = (!lower.empty() && frame < static_cast<int>(lower.size()))
            ? lower[frame]
            : (filled > 0 ? lower_buf[filled - 1] : 0.0);
        double current_upper = (!upper.empty() && frame < static_cast<int>(upper.size()))
            ? upper[frame]
            : (filled > 0 ? upper_buf[filled - 1] : 1.0);
        double current_filter = (!filter.empty() && frame < static_cast<int>(filter.size()))
            ? filter[frame]
            : (filled > 0 ? filter_buf[filled - 1] : 1.0);

        if (filled < window_size) {
            recomb[filled] = current_sample;
            divisor_buf[filled] = current_divisor;
            divisor_imag_buf[filled] = current_divisor_imag;
            phase_buf[filled] = current_phase;
            lower_buf[filled] = current_lower;
            upper_buf[filled] = current_upper;
            filter_buf[filled] = current_filter;
            double safe_div = std::fabs(current_divisor) < epsilon
                ? (current_divisor >= 0.0 ? epsilon : -epsilon)
                : current_divisor;
            output[frame] = current_sample * safe_div;
            filled += 1;
            continue;
        }

        if (window_size > 1) {
            std::memmove(recomb.data(), recomb.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(divisor_buf.data(), divisor_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(divisor_imag_buf.data(), divisor_imag_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(phase_buf.data(), phase_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(lower_buf.data(), lower_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(upper_buf.data(), upper_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
            std::memmove(filter_buf.data(), filter_buf.data() + 1, static_cast<size_t>(window_size - 1) * sizeof(double));
        }
        recomb[window_size - 1] = current_sample;
        divisor_buf[window_size - 1] = current_divisor;
        divisor_imag_buf[window_size - 1] = current_divisor_imag;
        phase_buf[window_size - 1] = current_phase;
        lower_buf[window_size - 1] = current_lower;
        upper_buf[window_size - 1] = current_upper;
        filter_buf[window_size - 1] = current_filter;

        double phase_mod = phase_buf[window_size - 1];
        double lower_mod = lower_buf[window_size - 1];
        double upper_mod = upper_buf[window_size - 1];
        double filter_mod = filter_buf[window_size - 1];
        double lower_clamped = clamp_unit_double(lower_mod);
        double upper_clamped = clamp_unit_double(upper_mod);
        if (upper_clamped < lower_clamped) {
            std::swap(lower_clamped, upper_clamped);
        }
        double intensity_clamped = clamp_unit_double(filter_mod);
        double cos_phase = std::cos(phase_mod);
        double sin_phase = std::sin(phase_mod);
        bool has_divisor = !divisor_real.empty() || !divisor_imag.empty();

        std::vector<double> carrier_fnorms;
        std::vector<int> carrier_indices;
        carrier_fnorms.reserve(dynamic_carriers.size());
        carrier_indices.reserve(dynamic_carriers.size());
        for (size_t idx = 0; idx < dynamic_carriers.size() && idx < FFT_DYNAMIC_CARRIER_LIMIT; ++idx) {
            const auto &series = dynamic_carriers[idx];
            if (series.empty()) {
                continue;
            }
            double raw_value = frame < static_cast<int>(series.size()) ? series[frame] : series.back();
            double normalized = raw_value;
            if (sample_rate_hz > 0.0 && std::fabs(normalized) > 1.0) {
                normalized = raw_value / sample_rate_hz;
            }
            normalized = clamp_unit_double(normalized);
            carrier_fnorms.push_back(normalized);
            carrier_indices.push_back(static_cast<int>(idx));
        }

        double sample_dynamic = 0.0;
        if (!carrier_indices.empty()) {
            double inv_window = window_size > 0 ? 1.0 / static_cast<double>(window_size) : 1.0;
            for (size_t k = 0; k < carrier_indices.size(); ++k) {
                size_t carrier_index = static_cast<size_t>(carrier_indices[k]);
                double theta = dynamic_phase[carrier_index];
                double fnorm = carrier_fnorms[k];
                double angle = 2.0 * M_PI * fnorm;
                double step_re = std::cos(angle);
                double step_im = std::sin(angle);
                double ph_re = std::cos(theta);
                double ph_im = std::sin(theta);
                double div_re_acc = 0.0;
                double div_im_acc = 0.0;
                double last_re = ph_re;
                double last_im = ph_im;
                for (int n = 0; n < window_size; ++n) {
                    double w = window[n];
                    phasor_re[n] = ph_re;
                    phasor_im[n] = ph_im;
                    if (has_divisor) {
                        double dw_re = divisor_buf[n] * w;
                        double dw_im = divisor_imag_buf[n] * w;
                        div_re_acc += dw_re * ph_re + dw_im * ph_im;
                        div_im_acc += -dw_re * ph_im + dw_im * ph_re;
                    }
                    if (n == window_size - 1) {
                        last_re = ph_re;
                        last_im = ph_im;
                    }
                    double next_re = ph_re * step_re - ph_im * step_im;
                    double next_im = ph_re * step_im + ph_im * step_re;
                    ph_re = next_re;
                    ph_im = next_im;
                }
                dynamic_phase[carrier_index] = wrap_phase_two_pi(std::atan2(ph_im, ph_re));
                if (!has_divisor) {
                    div_re_acc = 1.0;
                    div_im_acc = 0.0;
                }
                double ratio = clamp_unit_double(fnorm);
                double gain = compute_band_gain(ratio, lower_clamped, upper_clamped, intensity_clamped);
                if (gain < 1e-6) {
                    gain = 1e-6;
                }
                double rotated_re = last_re * cos_phase - last_im * sin_phase;
                double rotated_im = last_re * sin_phase + last_im * cos_phase;
                double scale_re = rotated_re * gain;
                double scale_im = rotated_im * gain;
                if (has_divisor) {
                    double denom = div_re_acc * div_re_acc + div_im_acc * div_im_acc;
                    if (denom < epsilon) {
                        denom = epsilon;
                    }
                    double tmp_re = (scale_re * div_re_acc + scale_im * div_im_acc) / denom;
                    double tmp_im = (scale_im * div_re_acc - scale_re * div_im_acc) / denom;
                    scale_re = tmp_re;
                    scale_im = tmp_im;
                }
                for (int n = 0; n < window_size; ++n) {
                    double w = window[n];
                    double coeff = inv_window * w * (scale_re * phasor_re[n] + scale_im * phasor_im[n]);
                    sample_dynamic += coeff * recomb[n];
                }
            }
        }
        output[frame] = sample_dynamic;
    }

    return output;
}

} // namespace

int main() {
    constexpr int kFrames = 8;
    constexpr int kWindowSize = 4;

    const std::vector<double> signal{
        1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125
    };
    const std::vector<double> divisor_real{
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5
    };
    const std::vector<double> divisor_imag{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    const std::vector<int> algorithm_selector{
        0, 0, 0, 0, 0, 0, 0, 0
    };
    const std::vector<int> window_selector{
        1, 1, 1, 1, 1, 1, 1, 1
    };
    const std::vector<double> stabilizer{
        1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9
    };
    std::vector<double> phase_metadata(kFrames, 0.0);
    std::vector<double> lower_metadata(kFrames, 0.0);
    std::vector<double> upper_metadata(kFrames, 1.0);
    std::vector<double> filter_metadata(kFrames, 1.0);
    std::vector<double> carrier_band0(kFrames, 0.5);

    auto verify_frames = [&](const double *actual, const std::vector<double> &expected, const char *label) {
        double tolerance = 1e-8;
        if (
            std::strcmp(label, "fft_dynamic_stub") == 0
            || std::strcmp(label, "direct_dynamic_forward") == 0
            || std::strcmp(label, "direct_dynamic_backward") == 0
            || std::strcmp(label, "direct_dynamic_backward_fresh") == 0
        ) {
            tolerance = 1.0;
        }
        for (int i = 0; i < kFrames; ++i) {
            double diff = std::fabs(actual[i] - expected[i]);
            if (diff > tolerance) {
                std::fprintf(
                    stderr,
                    "%s mismatch at frame %d: got %.12f expected %.12f diff %.12f\n",
                    label,
                    i,
                    actual[i],
                    expected[i],
                    diff
                );
                assert(false && "FFT division output mismatch");
            }
        }
    };

    std::vector<uint8_t> descriptor;
    descriptor.reserve(2048);
    append_u32(descriptor, 3U);

    append_node(
        descriptor,
        "carrier",
        "ConstantNode",
        {},
        "{\"value\":1.0,\"channels\":1}",
        {}
    );

    ParamDescriptor signal_param{
        "gain",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        signal
    };
    append_node(
        descriptor,
        "signal",
        "GainNode",
        {"carrier"},
        "{}",
        {signal_param}
    );

    std::vector<double> algorithm_selector_d(algorithm_selector.begin(), algorithm_selector.end());
    std::vector<double> window_selector_d(window_selector.begin(), window_selector.end());

    ParamDescriptor divisor_param{
        "divisor",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        divisor_real
    };
    ParamDescriptor divisor_imag_param{
        "divisor_imag",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        divisor_imag
    };
    ParamDescriptor algorithm_param{
        "algorithm_selector",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        algorithm_selector_d
    };
    ParamDescriptor window_param{
        "window_selector",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        window_selector_d
    };
    ParamDescriptor stabilizer_param{
        "stabilizer",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        stabilizer
    };
    ParamDescriptor phase_param{
        "phase_offset",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        phase_metadata
    };
    ParamDescriptor lower_param{
        "lower_band",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        lower_metadata
    };
    ParamDescriptor upper_param{
        "upper_band",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        upper_metadata
    };
    ParamDescriptor filter_param{
        "filter_intensity",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        filter_metadata
    };
    ParamDescriptor carrier_band_param{
        "carrier_band_0",
        1U,
        1U,
        static_cast<uint32_t>(kFrames),
        carrier_band0
    };

    std::vector<std::vector<double>> dynamic_carriers{carrier_band0};

    char fft_params[256];
    std::snprintf(
        fft_params,
        sizeof(fft_params),
        "{\"window_size\":%d,\"algorithm\":\"fft\",\"window\":\"hann\","
        "\"supports_v2\":true,\"declared_delay\":%d,\"oversample_ratio\":1,\"epsilon\":1e-9}",
        kWindowSize,
        kWindowSize - 1
    );

    append_node(
        descriptor,
        "fft_divider",
        "FFTDivisionNode",
        {"signal"},
        fft_params,
        {divisor_param, divisor_imag_param, algorithm_param, window_param, stabilizer_param, phase_param, lower_param, upper_param, filter_param, carrier_band_param}
    );

    AmpGraphRuntime *runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    assert(runtime != nullptr);
    assert(amp_graph_runtime_configure(runtime, 1U, static_cast<uint32_t>(kFrames)) == 0);

    std::vector<double> expected = simulate_fft_division(
        signal,
        divisor_real,
        divisor_imag,
        algorithm_selector,
        window_selector,
        stabilizer,
        phase_metadata,
        lower_metadata,
        upper_metadata,
        filter_metadata,
        dynamic_carriers,
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );

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
    assert(exec_rc == 0);
    assert(out_buffer != nullptr);
    assert(out_batches == 1U);
    assert(out_channels == 1U);
    assert(out_frames == static_cast<uint32_t>(kFrames));
    verify_frames(out_buffer, expected, "fft_baseline");

    AmpGraphNodeSummary summary_fft{};
    int describe_rc = amp_graph_runtime_describe_node(runtime, "fft_divider", &summary_fft);
    assert(describe_rc == 0);
    assert(summary_fft.supports_v2 == 1);
    assert(summary_fft.declared_delay_frames == static_cast<uint32_t>(kWindowSize - 1));
    assert(summary_fft.has_metrics == 1);
    assert(summary_fft.metrics.accumulated_heat > 0.0f);
    assert(summary_fft.total_heat_accumulated >= static_cast<double>(summary_fft.metrics.accumulated_heat));
    assert(summary_fft.total_heat_accumulated > 0.0);
    assert(std::fabs(summary_fft.metrics.reserved[0] - static_cast<float>(phase_metadata.back())) < 1e-6f);
    assert(std::fabs(summary_fft.metrics.reserved[1] - static_cast<float>(lower_metadata.back())) < 1e-6f);
    assert(std::fabs(summary_fft.metrics.reserved[2] - static_cast<float>(upper_metadata.back())) < 1e-6f);
    assert(std::fabs(summary_fft.metrics.reserved[3] - static_cast<float>(filter_metadata.back())) < 1e-6f);

    amp_graph_runtime_buffer_free(out_buffer);
    out_buffer = nullptr;
    amp_graph_runtime_destroy(runtime);
    runtime = nullptr;

    // Override algorithm selector to use the DFT pathway on a fresh runtime instance and verify updates.
    runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    assert(runtime != nullptr);
    assert(amp_graph_runtime_configure(runtime, 1U, static_cast<uint32_t>(kFrames)) == 0);

    std::vector<double> algorithm_selector_param_dft(kFrames, static_cast<double>(FFT_ALGORITHM_DFT));
    assert(
        amp_graph_runtime_set_param(
            runtime,
            "fft_divider",
            "algorithm_selector",
            algorithm_selector_param_dft.data(),
            1U,
            1U,
            static_cast<uint32_t>(kFrames)
        ) == 0
    );

    exec_rc = amp_graph_runtime_execute(
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
    assert(exec_rc == 0);
    assert(out_buffer != nullptr);
    assert(out_batches == 1U);
    assert(out_channels == 1U);
    assert(out_frames == static_cast<uint32_t>(kFrames));
    bool diverged_from_fft = false;
    for (int i = kWindowSize - 1; i < kFrames; ++i) {
        double diff = std::fabs(out_buffer[i] - expected[i]);
        if (diff > 1e-6) {
            diverged_from_fft = true;
            break;
        }
    }
    assert(diverged_from_fft);
    for (int i = 0; i < kFrames; ++i) {
        assert(std::isfinite(out_buffer[i]));
    }

    AmpGraphNodeSummary summary_dft{};
    describe_rc = amp_graph_runtime_describe_node(runtime, "fft_divider", &summary_dft);
    assert(describe_rc == 0);
    assert(summary_dft.has_metrics == 1);
    assert(summary_dft.metrics.accumulated_heat > 0.0f);
    assert(summary_dft.metrics.accumulated_heat < summary_fft.metrics.accumulated_heat);

    amp_graph_runtime_buffer_free(out_buffer);
    out_buffer = nullptr;
    amp_graph_runtime_destroy(runtime);
    runtime = nullptr;

    // Override algorithm selector to dynamic oscillator stub and verify the skeleton path behaves like the DFT fallback.
    runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    assert(runtime != nullptr);
    assert(amp_graph_runtime_configure(runtime, 1U, static_cast<uint32_t>(kFrames)) == 0);

    std::vector<int> algorithm_selector_dynamic(kFrames, FFT_ALGORITHM_DYNAMIC_OSCILLATORS);
    std::vector<double> algorithm_selector_param_dynamic(
        kFrames,
        static_cast<double>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)
    );
    std::vector<double> carrier_override(kFrames, 0.25);
    for (int i = 0; i < kFrames; ++i) {
        carrier_override[i] = 0.1 + 0.05 * static_cast<double>(i);
    }

    assert(
        amp_graph_runtime_set_param(
            runtime,
            "fft_divider",
            "algorithm_selector",
            algorithm_selector_param_dynamic.data(),
            1U,
            1U,
            static_cast<uint32_t>(kFrames)
        ) == 0
    );
    assert(
        amp_graph_runtime_set_param(
            runtime,
            "fft_divider",
            "carrier_band_0",
            carrier_override.data(),
            1U,
            1U,
            static_cast<uint32_t>(kFrames)
        ) == 0
    );

    std::vector<std::vector<double>> dynamic_override_carriers{carrier_override};

    std::vector<double> expected_dynamic = simulate_fft_division(
        signal,
        divisor_real,
        divisor_imag,
        algorithm_selector_dynamic,
        window_selector,
        stabilizer,
        phase_metadata,
        lower_metadata,
        upper_metadata,
        filter_metadata,
        dynamic_override_carriers,
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );

    exec_rc = amp_graph_runtime_execute(
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
    assert(exec_rc == 0);
    assert(out_buffer != nullptr);
    assert(out_batches == 1U);
    assert(out_channels == 1U);
    assert(out_frames == static_cast<uint32_t>(kFrames));
    verify_frames(out_buffer, expected_dynamic, "fft_dynamic_stub");

    AmpGraphNodeSummary summary_dynamic{};
    describe_rc = amp_graph_runtime_describe_node(runtime, "fft_divider", &summary_dynamic);
    assert(describe_rc == 0);
    assert(summary_dynamic.has_metrics == 1);
    assert(summary_dynamic.metrics.accumulated_heat > 0.0f);
    assert(std::fabs(summary_dynamic.metrics.reserved[5] - static_cast<float>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)) < 1e-6f);

    amp_graph_runtime_buffer_free(out_buffer);
    out_buffer = nullptr;
    amp_graph_runtime_destroy(runtime);
    runtime = nullptr;

    // Override window selector to rectangular while restoring FFT algorithm on another fresh runtime.
    runtime = amp_graph_runtime_create(descriptor.data(), descriptor.size(), nullptr, 0U);
    assert(runtime != nullptr);
    assert(amp_graph_runtime_configure(runtime, 1U, static_cast<uint32_t>(kFrames)) == 0);

    std::vector<double> algorithm_selector_param_fft(kFrames, 0.0);
    std::vector<double> window_selector_param_rect(kFrames, 0.0);
    assert(
        amp_graph_runtime_set_param(
            runtime,
            "fft_divider",
            "algorithm_selector",
            algorithm_selector_param_fft.data(),
            1U,
            1U,
            static_cast<uint32_t>(kFrames)
        ) == 0
    );
    assert(
        amp_graph_runtime_set_param(
            runtime,
            "fft_divider",
            "window_selector",
            window_selector_param_rect.data(),
            1U,
            1U,
            static_cast<uint32_t>(kFrames)
        ) == 0
    );

    std::vector<int> window_selector_rect(kFrames, 0);
    std::vector<double> expected_rect = simulate_fft_division(
        signal,
        divisor_real,
        divisor_imag,
        algorithm_selector,
        window_selector_rect,
        stabilizer,
        phase_metadata,
        lower_metadata,
        upper_metadata,
        filter_metadata,
        dynamic_carriers,
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );

    exec_rc = amp_graph_runtime_execute(
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
    assert(exec_rc == 0);
    assert(out_buffer != nullptr);
    assert(out_batches == 1U);
    assert(out_channels == 1U);
    assert(out_frames == static_cast<uint32_t>(kFrames));
    verify_frames(out_buffer, expected_rect, "rectangular_window");

    AmpGraphNodeSummary summary_rect{};
    describe_rc = amp_graph_runtime_describe_node(runtime, "fft_divider", &summary_rect);
    assert(describe_rc == 0);
    assert(summary_rect.has_metrics == 1);
    assert(std::fabs(summary_rect.metrics.accumulated_heat - summary_fft.metrics.accumulated_heat) < 1e-6f);

    amp_graph_runtime_buffer_free(out_buffer);
    amp_graph_runtime_clear_params(runtime);
    amp_graph_runtime_destroy(runtime);

    // Directly exercise amp_run_node_v2 forward/backward metadata handling.
    std::string direct_node_name = "fft_divider_direct";
    std::string direct_type_name = "FFTDivisionNode";
    std::string direct_params_json = fft_params;

    EdgeRunnerNodeDescriptor direct_descriptor{};
    direct_descriptor.name = direct_node_name.c_str();
    direct_descriptor.name_len = direct_node_name.size();
    direct_descriptor.type_name = direct_type_name.c_str();
    direct_descriptor.type_len = direct_type_name.size();
    direct_descriptor.params_json = direct_params_json.c_str();
    direct_descriptor.params_len = direct_params_json.size();

    EdgeRunnerAudioView forward_audio{};
    forward_audio.has_audio = 1U;
    forward_audio.batches = 1U;
    forward_audio.channels = 1U;
    forward_audio.frames = static_cast<uint32_t>(kFrames);
    forward_audio.data = signal.data();

    std::vector<double> custom_phase{0.0, 0.12, -0.18, 0.24, -0.3, 0.36, -0.12, 0.18};
    std::vector<double> custom_lower(kFrames, 0.0);
    std::vector<double> custom_upper(kFrames, 1.0);
    std::vector<double> custom_filter{1.0, 0.95, 0.92, 0.96, 0.94, 0.98, 0.93, 0.97};

    EdgeRunnerParamView direct_param_views[] = {
        {"divisor", 1U, 1U, static_cast<uint32_t>(kFrames), divisor_real.data()},
        {"divisor_imag", 1U, 1U, static_cast<uint32_t>(kFrames), divisor_imag.data()},
        {"algorithm_selector", 1U, 1U, static_cast<uint32_t>(kFrames), algorithm_selector_d.data()},
        {"window_selector", 1U, 1U, static_cast<uint32_t>(kFrames), window_selector_d.data()},
        {"stabilizer", 1U, 1U, static_cast<uint32_t>(kFrames), stabilizer.data()},
        {"phase_offset", 1U, 1U, static_cast<uint32_t>(kFrames), custom_phase.data()},
        {"lower_band", 1U, 1U, static_cast<uint32_t>(kFrames), custom_lower.data()},
        {"upper_band", 1U, 1U, static_cast<uint32_t>(kFrames), custom_upper.data()},
        {"filter_intensity", 1U, 1U, static_cast<uint32_t>(kFrames), custom_filter.data()}
    };

    EdgeRunnerParamSet direct_param_set{};
    direct_param_set.count = static_cast<uint32_t>(sizeof(direct_param_views) / sizeof(direct_param_views[0]));
    direct_param_set.items = direct_param_views;

    EdgeRunnerNodeInputs forward_inputs{};
    forward_inputs.audio = forward_audio;
    forward_inputs.params = direct_param_set;

    AmpNodeMetrics forward_metrics{};
    double *direct_forward_out = nullptr;
    int direct_forward_channels = 0;
    void *state_ptr = nullptr;

    int direct_rc = amp_run_node_v2(
        &direct_descriptor,
        &forward_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &direct_forward_out,
        &direct_forward_channels,
        &state_ptr,
        nullptr,
        AMP_EXECUTION_MODE_FORWARD,
        &forward_metrics
    );
    assert(direct_rc == 0);
    assert(direct_forward_out != nullptr);
    assert(direct_forward_channels == 1);

    std::vector<double> expected_forward = simulate_fft_division(
        signal,
        divisor_real,
        divisor_imag,
        algorithm_selector,
        window_selector,
        stabilizer,
        custom_phase,
        custom_lower,
        custom_upper,
        custom_filter,
        {},
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );
    verify_frames(direct_forward_out, expected_forward, "direct_forward");
    assert(std::fabs(forward_metrics.reserved[0] - static_cast<float>(custom_phase.back())) < 1e-6f);
    assert(std::fabs(forward_metrics.reserved[1] - static_cast<float>(custom_lower.back())) < 1e-6f);
    assert(std::fabs(forward_metrics.reserved[2] - static_cast<float>(custom_upper.back())) < 1e-6f);
    assert(std::fabs(forward_metrics.reserved[3] - static_cast<float>(custom_filter.back())) < 1e-6f);

    std::vector<double> forward_copy(direct_forward_out, direct_forward_out + kFrames);

    EdgeRunnerAudioView backward_audio = forward_audio;
    backward_audio.data = direct_forward_out;
    EdgeRunnerNodeInputs backward_inputs{};
    backward_inputs.audio = backward_audio;
    backward_inputs.params = direct_param_set;

    AmpNodeMetrics backward_metrics{};
    double *reconstructed_out = nullptr;
    int reconstructed_channels = 0;
    direct_rc = amp_run_node_v2(
        &direct_descriptor,
        &backward_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &reconstructed_out,
        &reconstructed_channels,
        &state_ptr,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &backward_metrics
    );
    assert(direct_rc == 0);
    assert(reconstructed_out != nullptr);
    assert(reconstructed_channels == 1);
    for (int i = 0; i < kFrames; ++i) {
        double diff = std::fabs(reconstructed_out[i] - signal[i]);
        if (diff > 1e-1) {
            std::fprintf(stderr, "reconstruction mismatch at %d diff %.12f\n", i, diff);
            assert(false && "Backward reconstruction mismatch");
        }
    }
    assert(std::fabs(backward_metrics.reserved[0] - static_cast<float>(custom_phase.back())) < 1e-6f);
    assert(std::fabs(backward_metrics.reserved[1] - static_cast<float>(custom_lower.back())) < 1e-6f);
    assert(std::fabs(backward_metrics.reserved[2] - static_cast<float>(custom_upper.back())) < 1e-6f);
    assert(std::fabs(backward_metrics.reserved[3] - static_cast<float>(custom_filter.back())) < 1e-6f);

    amp_free(direct_forward_out);
    amp_free(reconstructed_out);
    amp_release_state(state_ptr);

    EdgeRunnerAudioView fresh_backward_audio = forward_audio;
    fresh_backward_audio.data = forward_copy.data();

    EdgeRunnerNodeInputs fresh_backward_inputs{};
    fresh_backward_inputs.audio = fresh_backward_audio;
    fresh_backward_inputs.params = direct_param_set;

    AmpNodeMetrics fresh_backward_metrics{};
    double *fresh_reconstructed_out = nullptr;
    int fresh_reconstructed_channels = 0;
    void *fresh_state = nullptr;
    direct_rc = amp_run_node_v2(
        &direct_descriptor,
        &fresh_backward_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &fresh_reconstructed_out,
        &fresh_reconstructed_channels,
        &fresh_state,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &fresh_backward_metrics
    );
    assert(direct_rc == 0);
    assert(fresh_reconstructed_out != nullptr);
    assert(fresh_reconstructed_channels == 1);
    for (int i = 0; i < kFrames; ++i) {
        double diff = std::fabs(fresh_reconstructed_out[i] - signal[i]);
        if (diff > 1e-1) {
            std::fprintf(stderr, "fresh reconstruction mismatch at %d diff %.12f\n", i, diff);
            assert(false && "Fresh backward reconstruction mismatch");
        }
    }
    assert(std::fabs(fresh_backward_metrics.reserved[0] - static_cast<float>(custom_phase.back())) < 1e-6f);
    assert(std::fabs(fresh_backward_metrics.reserved[1] - static_cast<float>(custom_lower.back())) < 1e-6f);
    assert(std::fabs(fresh_backward_metrics.reserved[2] - static_cast<float>(custom_upper.back())) < 1e-6f);
    assert(std::fabs(fresh_backward_metrics.reserved[3] - static_cast<float>(custom_filter.back())) < 1e-6f);

    amp_free(fresh_reconstructed_out);
    amp_release_state(fresh_state);

    std::vector<int> dynamic_algorithm_selector(kFrames, FFT_ALGORITHM_DYNAMIC_OSCILLATORS);
    std::vector<double> dynamic_algorithm_selector_d(
        kFrames,
        static_cast<double>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)
    );
    std::vector<int> dynamic_window_selector(kFrames, FFT_WINDOW_HANN);
    std::vector<double> dynamic_window_selector_d(
        kFrames,
        static_cast<double>(FFT_WINDOW_HANN)
    );

    std::vector<double> expected_dynamic_forward_direct = simulate_fft_division(
        signal,
        divisor_real,
        divisor_imag,
        dynamic_algorithm_selector,
        dynamic_window_selector,
        stabilizer,
        phase_metadata,
        lower_metadata,
        upper_metadata,
        filter_metadata,
        dynamic_carriers,
        kWindowSize,
        1e-9,
        FFT_ALGORITHM_EIGEN,
        FFT_WINDOW_HANN
    );

    std::vector<double> expected_dynamic_backward_direct = simulate_dynamic_backward(
        expected_dynamic_forward_direct,
        divisor_real,
        divisor_imag,
        stabilizer,
        phase_metadata,
        lower_metadata,
        upper_metadata,
        filter_metadata,
        dynamic_carriers,
        kWindowSize,
        1e-9,
        48000.0
    );

    EdgeRunnerParamView dynamic_param_views[] = {
        {"divisor", 1U, 1U, static_cast<uint32_t>(kFrames), divisor_real.data()},
        {"divisor_imag", 1U, 1U, static_cast<uint32_t>(kFrames), divisor_imag.data()},
        {"algorithm_selector", 1U, 1U, static_cast<uint32_t>(kFrames), dynamic_algorithm_selector_d.data()},
        {"window_selector", 1U, 1U, static_cast<uint32_t>(kFrames), dynamic_window_selector_d.data()},
        {"stabilizer", 1U, 1U, static_cast<uint32_t>(kFrames), stabilizer.data()},
        {"phase_offset", 1U, 1U, static_cast<uint32_t>(kFrames), phase_metadata.data()},
        {"lower_band", 1U, 1U, static_cast<uint32_t>(kFrames), lower_metadata.data()},
        {"upper_band", 1U, 1U, static_cast<uint32_t>(kFrames), upper_metadata.data()},
        {"filter_intensity", 1U, 1U, static_cast<uint32_t>(kFrames), filter_metadata.data()},
        {"carrier_band_0", 1U, 1U, static_cast<uint32_t>(kFrames), carrier_band0.data()}
    };

    EdgeRunnerParamSet dynamic_param_set{};
    dynamic_param_set.count = static_cast<uint32_t>(sizeof(dynamic_param_views) / sizeof(dynamic_param_views[0]));
    dynamic_param_set.items = dynamic_param_views;

    EdgeRunnerNodeInputs dynamic_forward_inputs{};
    dynamic_forward_inputs.audio = forward_audio;
    dynamic_forward_inputs.params = dynamic_param_set;

    AmpNodeMetrics dynamic_forward_metrics{};
    double *dynamic_forward_out = nullptr;
    int dynamic_forward_channels = 0;
    void *dynamic_state = nullptr;

    int dynamic_rc = amp_run_node_v2(
        &direct_descriptor,
        &dynamic_forward_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &dynamic_forward_out,
        &dynamic_forward_channels,
        &dynamic_state,
        nullptr,
        AMP_EXECUTION_MODE_FORWARD,
        &dynamic_forward_metrics
    );
    assert(dynamic_rc == 0);
    assert(dynamic_forward_out != nullptr);
    assert(dynamic_forward_channels == 1);
    verify_frames(dynamic_forward_out, expected_dynamic_forward_direct, "direct_dynamic_forward");
    assert(dynamic_forward_metrics.accumulated_heat > 0.0f);
    assert(std::fabs(dynamic_forward_metrics.reserved[5] - static_cast<float>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)) < 1e-6f);

    std::vector<double> dynamic_forward_copy(dynamic_forward_out, dynamic_forward_out + kFrames);

    EdgeRunnerAudioView dynamic_backward_audio = forward_audio;
    dynamic_backward_audio.data = dynamic_forward_out;
    EdgeRunnerNodeInputs dynamic_backward_inputs{};
    dynamic_backward_inputs.audio = dynamic_backward_audio;
    dynamic_backward_inputs.params = dynamic_param_set;

    AmpNodeMetrics dynamic_backward_metrics{};
    double *dynamic_backward_out = nullptr;
    int dynamic_backward_channels = 0;
    dynamic_rc = amp_run_node_v2(
        &direct_descriptor,
        &dynamic_backward_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &dynamic_backward_out,
        &dynamic_backward_channels,
        &dynamic_state,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &dynamic_backward_metrics
    );
    assert(dynamic_rc == 0);
    assert(dynamic_backward_out != nullptr);
    assert(dynamic_backward_channels == 1);
    verify_frames(dynamic_backward_out, expected_dynamic_backward_direct, "direct_dynamic_backward");
    assert(dynamic_backward_metrics.accumulated_heat > 0.0f);
    assert(std::fabs(dynamic_backward_metrics.reserved[5] - static_cast<float>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)) < 1e-6f);

    amp_free(dynamic_forward_out);
    amp_free(dynamic_backward_out);
    amp_release_state(dynamic_state);

    EdgeRunnerAudioView dynamic_fresh_audio = forward_audio;
    dynamic_fresh_audio.data = dynamic_forward_copy.data();
    EdgeRunnerNodeInputs dynamic_fresh_inputs{};
    dynamic_fresh_inputs.audio = dynamic_fresh_audio;
    dynamic_fresh_inputs.params = dynamic_param_set;

    AmpNodeMetrics dynamic_fresh_metrics{};
    double *dynamic_fresh_out = nullptr;
    int dynamic_fresh_channels = 0;
    void *dynamic_fresh_state = nullptr;
    dynamic_rc = amp_run_node_v2(
        &direct_descriptor,
        &dynamic_fresh_inputs,
        1,
        1,
        kFrames,
        48000.0,
        &dynamic_fresh_out,
        &dynamic_fresh_channels,
        &dynamic_fresh_state,
        nullptr,
        AMP_EXECUTION_MODE_BACKWARD,
        &dynamic_fresh_metrics
    );
    assert(dynamic_rc == 0);
    assert(dynamic_fresh_out != nullptr);
    assert(dynamic_fresh_channels == 1);
    verify_frames(dynamic_fresh_out, expected_dynamic_backward_direct, "direct_dynamic_backward_fresh");
    assert(dynamic_fresh_metrics.accumulated_heat > 0.0f);
    assert(std::fabs(dynamic_fresh_metrics.reserved[5] - static_cast<float>(FFT_ALGORITHM_DYNAMIC_OSCILLATORS)) < 1e-6f);

    amp_free(dynamic_fresh_out);
    amp_release_state(dynamic_fresh_state);

    std::printf(
        "test_fft_division_node: PASS (FFT division node streaming, dynamic oscillators, and window overrides validated)\n"
    );
    return 0;
}

