#include "amp_native.h"
#include "amp_descriptor_builder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

static void amp_generate_driver_curves(
    double *freq,
    double *amp,
    double *render,
    uint32_t frames,
    double sample_rate
) {
    const double base_freq = 220.0;
    const double sweep_hz = 1.0;
    const double sweep_depth = 110.0;
    const double render_depth = 0.4;
    for (uint32_t i = 0; i < frames; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double slow = std::sin(2.0 * M_PI * sweep_hz * t);
        freq[i] = base_freq + sweep_depth * slow;
        if (freq[i] < 40.0) {
            freq[i] = 40.0;
        }
        amp[i] = 0.55 + 0.4 * slow;
        if (amp[i] < 0.1) {
            amp[i] = 0.1;
        }
        render[i] = 0.5 + render_depth * slow;
        if (render[i] < 0.0) {
            render[i] = 0.0;
        } else if (render[i] > 1.0) {
            render[i] = 1.0;
        }
    }
}

static void amp_generate_oscillator_curves(
    double *freq,
    double *amp,
    double *slew,
    uint32_t frames,
    double sample_rate,
    const double *driver_freq
) {
    (void)sample_rate;
    for (uint32_t i = 0; i < frames; ++i) {
        freq[i] = driver_freq != nullptr ? driver_freq[i] : 220.0;
        amp[i] = 0.4;
        slew[i] = 12000.0;
    }
}

struct StreamChunk {
    uint64_t sequence;
    uint32_t frames;
    uint32_t channels;
    std::vector<double> data;
};

static bool pop_all_dumps(AmpGraphStreamer *streamer, uint32_t max_frames, std::vector<StreamChunk> &chunks) {
    while (true) {
        uint32_t dump_count = 0;
        if (amp_graph_streamer_dump_count(streamer, &dump_count) != 0) {
            return false;
        }
        if (dump_count == 0U) {
            break;
        }

        StreamChunk chunk{};
        const uint32_t request = std::max<uint32_t>(max_frames, 1U);
        std::vector<double> buffer(static_cast<size_t>(request));
        uint32_t out_frames = 0;
        uint32_t out_channels = 0;
        uint64_t sequence = 0;
        int rc = amp_graph_streamer_pop_dump(
            streamer,
            buffer.data(),
            request,
            &out_frames,
            &out_channels,
            &sequence
        );
        if (rc == 1) {
            const size_t needed = static_cast<size_t>(out_frames ? out_frames : request) *
                static_cast<size_t>(std::max<uint32_t>(out_channels, 1U));
            buffer.assign(needed, 0.0);
            rc = amp_graph_streamer_pop_dump(
                streamer,
                buffer.data(),
                static_cast<uint32_t>(out_frames),
                &out_frames,
                &out_channels,
                &sequence
            );
        }
        if (rc != 0) {
            return false;
        }
        if (out_frames == 0U) {
            continue;
        }
        const size_t stride = static_cast<size_t>(std::max<uint32_t>(out_channels, 1U));
        buffer.resize(static_cast<size_t>(out_frames) * stride);
        chunk.sequence = sequence;
        chunk.frames = out_frames;
        chunk.channels = out_channels ? out_channels : 1U;
        chunk.data = std::move(buffer);
        chunks.emplace_back(std::move(chunk));
    }
    return true;
}

static bool read_ring_snapshot(AmpGraphStreamer *streamer, uint32_t max_frames, std::vector<StreamChunk> &chunks) {
    uint64_t available = 0;
    if (amp_graph_streamer_available(streamer, &available) != 0) {
        return false;
    }
    if (available == 0U) {
        return true;
    }
    uint32_t request = static_cast<uint32_t>(std::min<uint64_t>(available, max_frames));
    request = std::max<uint32_t>(request, 1U);
    std::vector<double> buffer(static_cast<size_t>(request));
    uint32_t out_frames = 0;
    uint32_t out_channels = 0;
    uint64_t sequence = 0;
    if (amp_graph_streamer_read(
            streamer,
            buffer.data(),
            request,
            &out_frames,
            &out_channels,
            &sequence
        ) != 0) {
        return false;
    }
    if (out_frames == 0U) {
        return true;
    }
    const size_t stride = static_cast<size_t>(std::max<uint32_t>(out_channels, 1U));
    buffer.resize(static_cast<size_t>(out_frames) * stride);
    StreamChunk chunk{};
    chunk.sequence = sequence;
    chunk.frames = out_frames;
    chunk.channels = out_channels ? out_channels : 1U;
    chunk.data = std::move(buffer);
    chunks.emplace_back(std::move(chunk));
    return true;
}

static std::vector<double> collect_pcm_from_streamer(
    AmpGraphStreamer *streamer,
    uint32_t total_frames,
    uint32_t &out_channels
) {
    std::vector<StreamChunk> chunks;
    if (!pop_all_dumps(streamer, total_frames, chunks)) {
        return {};
    }
    if (!read_ring_snapshot(streamer, total_frames, chunks)) {
        return {};
    }

    std::sort(chunks.begin(), chunks.end(), [](const StreamChunk &a, const StreamChunk &b) {
        return a.sequence < b.sequence;
    });

    out_channels = 1U;
    uint64_t gathered_frames = 0;
    for (const StreamChunk &chunk : chunks) {
        gathered_frames += chunk.frames;
        out_channels = std::max<uint32_t>(out_channels, chunk.channels);
    }

    std::vector<double> pcm;
    pcm.reserve(static_cast<size_t>(gathered_frames) * static_cast<size_t>(out_channels));
    for (const StreamChunk &chunk : chunks) {
        pcm.insert(pcm.end(), chunk.data.begin(), chunk.data.end());
    }

    const size_t expected = static_cast<size_t>(total_frames) * static_cast<size_t>(out_channels);
    if (pcm.size() < expected) {
        pcm.resize(expected, 0.0);
    } else if (pcm.size() > expected) {
        pcm.resize(expected);
    }

    return pcm;
}

static std::vector<double> collapse_to_mono(const std::vector<double> &interleaved, uint32_t frames, uint32_t channels) {
    if (frames == 0U) {
        return {};
    }
    if (channels <= 1U) {
        return std::vector<double>(interleaved.begin(), interleaved.begin() + static_cast<std::ptrdiff_t>(frames));
    }
    std::vector<double> mono(frames, 0.0);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        double accum = 0.0;
        for (uint32_t channel = 0; channel < channels; ++channel) {
            accum += interleaved[static_cast<size_t>(frame) * static_cast<size_t>(channels) + channel];
        }
        mono[frame] = accum / static_cast<double>(channels);
    }
    return mono;
}

static std::vector<uint8_t> compute_spectrogram_image(
    const std::vector<double> &pcm,
    double sample_rate,
    uint32_t window_size,
    uint32_t hop_size,
    uint32_t &out_width,
    uint32_t &out_height
) {
    (void)sample_rate;
    if (window_size == 0U || hop_size == 0U) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    std::vector<double> padded = pcm;
    if (padded.size() < window_size) {
        padded.resize(window_size, 0.0);
    }
    const size_t total_samples = padded.size();
    if (total_samples < window_size) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    const size_t segment_count = 1U + (total_samples - window_size) / hop_size;
    const size_t bins = window_size / 2U + 1U;

    std::vector<double> window(window_size);
    for (size_t n = 0; n < window_size; ++n) {
        window[n] = 0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(n) / static_cast<double>(window_size - 1U));
    }

    std::vector<double> log_spectra(segment_count * bins, 0.0);
    std::vector<double> tapered(window_size, 0.0);

    for (size_t segment = 0; segment < segment_count; ++segment) {
        size_t start = segment * hop_size;
        for (size_t n = 0; n < window_size; ++n) {
            tapered[n] = padded[start + n] * window[n];
        }
        for (size_t k = 0; k < bins; ++k) {
            double real = 0.0;
            double imag = 0.0;
            const double angular = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(window_size);
            for (size_t n = 0; n < window_size; ++n) {
                double angle = angular * static_cast<double>(n);
                double value = tapered[n];
                real += value * std::cos(angle);
                imag += value * std::sin(angle);
            }
            double magnitude = std::sqrt(real * real + imag * imag);
            double log_mag = std::log10(magnitude + 1e-6);
            log_spectra[segment * bins + k] = log_mag;
        }
    }

    double max_val = -std::numeric_limits<double>::infinity();
    double min_val = std::numeric_limits<double>::infinity();
    for (double value : log_spectra) {
        max_val = std::max(max_val, value);
        min_val = std::min(min_val, value);
    }
    if (!std::isfinite(max_val) || !std::isfinite(min_val) || max_val == min_val) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    for (double &value : log_spectra) {
        value = (value - min_val) / (max_val - min_val);
        value = std::clamp(value, 0.0, 1.0);
    }

    std::vector<uint8_t> image(segment_count * bins, 0U);
    for (size_t row = 0; row < bins; ++row) {
        for (size_t col = 0; col < segment_count; ++col) {
            size_t src_row = bins - 1U - row;
            double normalized = log_spectra[col * bins + src_row];
            double pixel = 1.0 - normalized;
            image[row * segment_count + col] = static_cast<uint8_t>(std::lround(pixel * 255.0));
        }
    }

    out_width = static_cast<uint32_t>(segment_count);
    out_height = static_cast<uint32_t>(bins);
    return image;
}

static uint32_t adler32(const uint8_t *data, size_t length) {
    const uint32_t MOD = 65521U;
    uint32_t a = 1U;
    uint32_t b = 0U;
    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % MOD;
        b = (b + a) % MOD;
    }
    return (b << 16) | a;
}

static uint32_t crc32_accumulate(uint32_t crc, const uint8_t *data, size_t length) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                if (value & 1U) {
                    value = (value >> 1) ^ 0xEDB88320U;
                } else {
                    value >>= 1;
                }
            }
            table[i] = value;
        }
        initialized = true;
    }

    for (size_t i = 0; i < length; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFFU];
    }
    return crc;
}

static void write_u32_be(std::ofstream &stream, uint32_t value) {
    uint8_t bytes[4];
    bytes[0] = static_cast<uint8_t>((value >> 24) & 0xFFU);
    bytes[1] = static_cast<uint8_t>((value >> 16) & 0xFFU);
    bytes[2] = static_cast<uint8_t>((value >> 8) & 0xFFU);
    bytes[3] = static_cast<uint8_t>(value & 0xFFU);
    stream.write(reinterpret_cast<const char *>(bytes), 4);
}

static bool write_png_signature(std::ofstream &stream) {
    static const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    stream.write(reinterpret_cast<const char *>(signature), sizeof(signature));
    return stream.good();
}

static bool write_png_chunk(std::ofstream &stream, const char type[4], const std::vector<uint8_t> &payload) {
    if (!stream.good()) {
        return false;
    }
    write_u32_be(stream, static_cast<uint32_t>(payload.size()));
    stream.write(type, 4);
    if (!payload.empty()) {
        stream.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    uint32_t crc = crc32_accumulate(0xFFFFFFFFU, reinterpret_cast<const uint8_t *>(type), 4U);
    if (!payload.empty()) {
        crc = crc32_accumulate(crc, payload.data(), payload.size());
    }
    crc ^= 0xFFFFFFFFU;
    write_u32_be(stream, crc);
    return stream.good();
}

static bool write_grayscale_png(const fs::path &path, const std::vector<uint8_t> &image, uint32_t width, uint32_t height) {
    if (width == 0U || height == 0U) {
        return false;
    }
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (image.size() < expected) {
        return false;
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    if (!write_png_signature(stream)) {
        return false;
    }

    std::vector<uint8_t> ihdr(13U, 0U);
    auto write_be = [](uint32_t value, std::vector<uint8_t> &buffer, size_t offset) {
        buffer[offset + 0] = static_cast<uint8_t>((value >> 24) & 0xFFU);
        buffer[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFFU);
        buffer[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFFU);
        buffer[offset + 3] = static_cast<uint8_t>(value & 0xFFU);
    };
    write_be(width, ihdr, 0);
    write_be(height, ihdr, 4);
    ihdr[8] = 8U;
    ihdr[9] = 0U;
    ihdr[10] = 0U;
    ihdr[11] = 0U;
    ihdr[12] = 0U;

    if (!write_png_chunk(stream, "IHDR", ihdr)) {
        return false;
    }

    std::vector<uint8_t> raw(static_cast<size_t>(height) * (static_cast<size_t>(width) + 1U), 0U);
    for (uint32_t row = 0; row < height; ++row) {
        uint8_t *dst = raw.data() + static_cast<size_t>(row) * (static_cast<size_t>(width) + 1U);
        dst[0] = 0U;
        const uint8_t *src = image.data() + static_cast<size_t>(row) * static_cast<size_t>(width);
        std::memcpy(dst + 1, src, static_cast<size_t>(width));
    }

    std::vector<uint8_t> idat;
    idat.reserve(raw.size() + raw.size() / 64U + 6U);
    idat.push_back(0x78);
    idat.push_back(0x01);

    size_t remaining = raw.size();
    size_t offset = 0;
    while (remaining > 0) {
        const size_t block_len = std::min<size_t>(remaining, 65535U);
        const uint16_t len = static_cast<uint16_t>(block_len);
        const uint16_t nlen = static_cast<uint16_t>(~len);
        const bool final_block = (remaining == block_len);

        idat.push_back(static_cast<uint8_t>(final_block ? 0x01 : 0x00));
        idat.push_back(static_cast<uint8_t>(len & 0xFFU));
        idat.push_back(static_cast<uint8_t>((len >> 8) & 0xFFU));
        idat.push_back(static_cast<uint8_t>(nlen & 0xFFU));
        idat.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFFU));

        idat.insert(idat.end(), raw.begin() + offset, raw.begin() + offset + block_len);

        offset += block_len;
        remaining -= block_len;
    }

    uint32_t adler = adler32(raw.data(), raw.size());
    idat.push_back(static_cast<uint8_t>((adler >> 24) & 0xFFU));
    idat.push_back(static_cast<uint8_t>((adler >> 16) & 0xFFU));
    idat.push_back(static_cast<uint8_t>((adler >> 8) & 0xFFU));
    idat.push_back(static_cast<uint8_t>(adler & 0xFFU));

    if (!write_png_chunk(stream, "IDAT", idat)) {
        return false;
    }

    std::vector<uint8_t> iend;
    if (!write_png_chunk(stream, "IEND", iend)) {
        return false;
    }

    return stream.good();
}

static void amp_write_wav16(
    const fs::path &path,
    const double *samples,
    uint32_t channels,
    uint32_t frames,
    double sample_rate
) {
    if (!samples || channels == 0U || frames == 0U) {
        return;
    }

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        return;
    }

    size_t total = static_cast<size_t>(channels) * static_cast<size_t>(frames);
    double peak = 0.0;
    for (size_t i = 0; i < total; ++i) {
        peak = std::max(peak, std::fabs(samples[i]));
    }
    double scale = peak > 0.0 ? 0.98 / peak : 1.0;

    std::vector<int16_t> pcm(total);
    for (size_t i = 0; i < total; ++i) {
        double value = samples[i] * scale;
        if (value > 1.0) {
            value = 1.0;
        } else if (value < -1.0) {
            value = -1.0;
        }
        pcm[i] = static_cast<int16_t>(std::lround(value * 32767.0));
    }

    const uint32_t bytes_per_sample = static_cast<uint32_t>(sizeof(int16_t));
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * static_cast<double>(channels) * bytes_per_sample);
    const uint16_t block_align = static_cast<uint16_t>(channels * bytes_per_sample);
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t riff_size = 4U + 8U + 16U + 8U + data_bytes;

    stream.write("RIFF", 4);
    stream.write(reinterpret_cast<const char *>(&riff_size), sizeof(riff_size));
    stream.write("WAVE", 4);
    stream.write("fmt ", 4);
    const uint32_t fmt_size = 16U;
    stream.write(reinterpret_cast<const char *>(&fmt_size), sizeof(fmt_size));
    const uint16_t audio_format = 1U;
    stream.write(reinterpret_cast<const char *>(&audio_format), sizeof(audio_format));
    const uint16_t wav_channels = static_cast<uint16_t>(channels);
    stream.write(reinterpret_cast<const char *>(&wav_channels), sizeof(wav_channels));
    const uint32_t wav_sr = static_cast<uint32_t>(sample_rate + 0.5);
    stream.write(reinterpret_cast<const char *>(&wav_sr), sizeof(wav_sr));
    stream.write(reinterpret_cast<const char *>(&byte_rate), sizeof(byte_rate));
    stream.write(reinterpret_cast<const char *>(&block_align), sizeof(block_align));
    const uint16_t bits_per_sample = 16U;
    stream.write(reinterpret_cast<const char *>(&bits_per_sample), sizeof(bits_per_sample));
    stream.write("data", 4);
    stream.write(reinterpret_cast<const char *>(&data_bytes), sizeof(data_bytes));
    stream.write(reinterpret_cast<const char *>(pcm.data()), static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
}

int main() {
    const uint32_t batches = 1U;
    const double sample_rate = 48000.0;
    const double duration_seconds = 2.0;
    const uint32_t frames = static_cast<uint32_t>(duration_seconds * sample_rate);

    AmpDescriptorBuffer descriptor;
    amp_descriptor_buffer_init(&descriptor);
    AmpDescriptorBuilder descriptor_builder{};
    if (amp_descriptor_builder_init(&descriptor_builder, &descriptor) != 0) {
        std::fprintf(stderr, "[demo] failed to initialise descriptor builder\n");
        return 1;
    }

    int exit_code = 0;
    AmpGraphRuntime *runtime = nullptr;
    AmpGraphStreamer *streamer = nullptr;
    bool streamer_started = false;
    uint64_t produced_frames = 0U;
    uint64_t consumed_frames = 0U;

    do {
        const char *const *pitch_inputs = nullptr;
        const char *const *driver_inputs = nullptr;
        static const char *const osc_inputs[] = {"driver"};
        static const char *const mix_inputs[] = {"osc"};

        static const char pitch_json[] =
            "{\"declared_delay\":0,\"default_slew\":0.0,\"min_freq\":0.0,\"oversample_ratio\":1,\"supports_v2\":true}";
        static const char driver_json[] =
            "{\"declared_delay\":0,\"mode\":\"piezo\",\"oversample_ratio\":1,\"supports_v2\":true}";
        static const char osc_json[] =
            "{\"accept_reset\":false,\"declared_delay\":0,\"integration_clamp\":1.2,"
            "\"integration_gain\":0.5,\"integration_leak\":0.997,\"mode\":\"op_amp\","\
            "\"oversample_ratio\":1,\"slew_clamp\":1.2,\"slew_rate\":12000.0,\"supports_v2\":true,\"wave\":\"saw\"}";
        static const char mix_json[] =
            "{\"channels\":1,\"declared_delay\":0,\"oversample_ratio\":1,\"supports_v2\":true}";

        if (amp_descriptor_builder_append_node(
                &descriptor_builder,
                "pitch",
                "OscillatorPitchNode",
                pitch_inputs,
                0U,
                pitch_json,
                nullptr,
                0U
            ) != 0) {
            std::fprintf(stderr, "[demo] failed to append pitch node\n");
            exit_code = 3;
            break;
        }
        if (amp_descriptor_builder_append_node(
                &descriptor_builder,
                "driver",
                "ParametricDriverNode",
                driver_inputs,
                0U,
                driver_json,
                nullptr,
                0U
            ) != 0) {
            std::fprintf(stderr, "[demo] failed to append driver node\n");
            exit_code = 3;
            break;
        }
        if (amp_descriptor_builder_append_node(
                &descriptor_builder,
                "osc",
                "OscNode",
                osc_inputs,
                1U,
                osc_json,
                nullptr,
                0U
            ) != 0) {
            std::fprintf(stderr, "[demo] failed to append oscillator node\n");
            exit_code = 4;
            break;
        }
        if (amp_descriptor_builder_append_node(
                &descriptor_builder,
                "mix",
                "MixNode",
                mix_inputs,
                1U,
                mix_json,
                nullptr,
                0U
            ) != 0) {
            std::fprintf(stderr, "[demo] failed to append mix node\n");
            exit_code = 5;
            break;
        }

        if (amp_descriptor_builder_finalize(&descriptor_builder) != 0) {
            std::fprintf(stderr, "[demo] failed to finalise descriptor\n");
            exit_code = 2;
            break;
        }

        runtime = amp_graph_runtime_create(descriptor.data, descriptor.size, nullptr, 0U);
        if (runtime == nullptr) {
            std::fprintf(stderr, "[demo] amp_graph_runtime_create failed\n");
            exit_code = 6;
            break;
        }

        if (amp_graph_runtime_configure(runtime, batches, frames) != 0) {
            std::fprintf(stderr, "[demo] amp_graph_runtime_configure failed\n");
            exit_code = 7;
            break;
        }
        amp_graph_runtime_set_dsp_sample_rate(runtime, sample_rate);
        amp_graph_runtime_set_scheduler_mode(runtime, AMP_SCHEDULER_LEARNED);
        AmpGraphSchedulerParams scheduler_params{};
        scheduler_params.early_bias = 0.5;
        scheduler_params.late_bias = 0.5;
        scheduler_params.saturation_bias = 1.0;
        amp_graph_runtime_set_scheduler_params(runtime, &scheduler_params);

        const size_t param_count = static_cast<size_t>(batches) * static_cast<size_t>(frames);
        std::vector<double> pitch_freq(param_count, 0.0);
        std::vector<double> driver_freq(param_count, 0.0);
        std::vector<double> driver_amp(param_count, 0.0);
        std::vector<double> driver_render(param_count, 0.0);
        std::vector<double> osc_freq(param_count, 0.0);
        std::vector<double> osc_amp(param_count, 0.0);
        std::vector<double> osc_slew(param_count, 0.0);

        amp_generate_driver_curves(driver_freq.data(), driver_amp.data(), driver_render.data(), frames, sample_rate);
        amp_generate_oscillator_curves(
            osc_freq.data(),
            osc_amp.data(),
            osc_slew.data(),
            frames,
            sample_rate,
            driver_freq.data()
        );

        if (amp_graph_runtime_set_param(runtime, "pitch", "freq", pitch_freq.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set pitch freq\n");
        }
        if (amp_graph_runtime_set_param(runtime, "driver", "frequency", driver_freq.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set driver frequency\n");
        }
        if (amp_graph_runtime_set_param(runtime, "driver", "amplitude", driver_amp.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set driver amplitude\n");
        }
        if (amp_graph_runtime_set_param(runtime, "driver", "render_mode", driver_render.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set driver render_mode\n");
        }
        if (amp_graph_runtime_set_param(runtime, "osc", "freq", osc_freq.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set oscillator freq\n");
        }
        if (amp_graph_runtime_set_param(runtime, "osc", "amp", osc_amp.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set oscillator amp\n");
        }
        if (amp_graph_runtime_set_param(runtime, "osc", "slew", osc_slew.data(), batches, 1U, frames) != 0) {
            std::fprintf(stderr, "[demo] failed to set oscillator slew\n");
        }

        streamer = amp_graph_streamer_create(
            runtime,
            nullptr,
            0U,
            static_cast<int>(frames),
            sample_rate,
            frames,
            512U
        );
        if (streamer == nullptr) {
            std::fprintf(stderr, "[demo] failed to create graph streamer\n");
            exit_code = 9;
            break;
        }

        if (amp_graph_streamer_start(streamer) != 0) {
            std::fprintf(stderr, "[demo] failed to start streamer\n");
            exit_code = 9;
            break;
        }
        streamer_started = true;

        while (produced_frames < frames) {
            int status = amp_graph_streamer_status(streamer, &produced_frames, &consumed_frames);
            if (status != 0) {
                std::fprintf(stderr, "[demo] streamer reported error status %d\n", status);
                exit_code = 9;
                break;
            }
            if (produced_frames >= frames) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (exit_code != 0) {
            break;
        }

        if (streamer_started) {
            amp_graph_streamer_stop(streamer);
            amp_graph_streamer_status(streamer, &produced_frames, &consumed_frames);
            streamer_started = false;
        }

        uint32_t output_channels = 1U;
        std::vector<double> pcm_interleaved = collect_pcm_from_streamer(streamer, frames, output_channels);
        if (pcm_interleaved.empty()) {
            std::fprintf(stderr, "[demo] streamer produced no PCM data\n");
            exit_code = 10;
            break;
        }

        std::vector<double> pcm_mono = collapse_to_mono(pcm_interleaved, frames, output_channels);

        fs::path output_dir("output/demo_kpn_native");
        std::error_code dir_error;
        fs::create_directories(output_dir, dir_error);

        const fs::path wav_path = output_dir / "output.wav";
        amp_write_wav16(wav_path, pcm_interleaved.data(), output_channels, frames, sample_rate);
        std::printf(
            "[demo] wrote %s (%u channels, %u frames)\n",
            wav_path.string().c_str(),
            output_channels,
            frames
        );

        const uint32_t window_size = 512U;
        const uint32_t hop_size = window_size / 4U;
        uint32_t image_width = 0U;
        uint32_t image_height = 0U;
        std::vector<uint8_t> spectrogram = compute_spectrogram_image(
            pcm_mono,
            sample_rate,
            window_size,
            hop_size,
            image_width,
            image_height
        );

        const fs::path png_path = output_dir / "spectrogram.png";
        if (!spectrogram.empty() && image_width > 0U && image_height > 0U) {
            if (write_grayscale_png(png_path, spectrogram, image_width, image_height)) {
                std::printf(
                    "[demo] wrote %s (%u x %u)\n",
                    png_path.string().c_str(),
                    image_width,
                    image_height
                );
            } else {
                std::fprintf(stderr, "[demo] failed to write spectrogram PNG\n");
            }
        } else {
            std::fprintf(stderr, "[demo] spectrogram image was empty\n");
        }
    } while (false);

    if (streamer_started && streamer != nullptr) {
        amp_graph_streamer_stop(streamer);
        amp_graph_streamer_status(streamer, &produced_frames, &consumed_frames);
    }
    if (streamer != nullptr) {
        amp_graph_streamer_destroy(streamer);
    }
    if (runtime != nullptr) {
        amp_graph_runtime_destroy(runtime);
    }
    amp_descriptor_buffer_destroy(&descriptor);

    return exit_code;
}
