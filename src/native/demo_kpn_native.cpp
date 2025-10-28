#include "amp_native.h"
#include "amp_descriptor_builder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

enum class StreamSource {
    Dump,
    Ring,
};

struct StreamChunk {
    uint64_t sequence;
    uint32_t frames;
    uint32_t channels;
    StreamSource source;
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
        chunk.source = StreamSource::Dump;
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
    chunk.source = StreamSource::Ring;
    chunk.data = std::move(buffer);
    chunks.emplace_back(std::move(chunk));
    return true;
}

struct TapCollection {
    std::vector<double> pcm;
    uint32_t pcm_channels{1U};
    uint32_t pcm_frames{0U};
    std::vector<double> spectral_power;
    uint32_t spectral_bins{0U};
    uint32_t spectral_columns{0U};
};

static bool collect_tap_streams(
    AmpGraphStreamer *streamer,
    uint32_t total_frames,
    TapCollection &collection
) {
    std::vector<StreamChunk> dump_chunks;
    if (!pop_all_dumps(streamer, total_frames, dump_chunks)) {
        return false;
    }

    std::vector<StreamChunk> pcm_chunks;
    std::vector<StreamChunk> spectral_chunks;
    pcm_chunks.reserve(dump_chunks.size());
    spectral_chunks.reserve(dump_chunks.size());

    // The dump queue carries tap payloads produced by FFTDivisionNode. PCM tap frames surface as
    // single-channel payloads while the spectral bins tap exposes a dense band × subchannel stride.
    // That separation lets us safely classify chunks by channel count without inspecting metadata.
    for (StreamChunk &chunk : dump_chunks) {
        const uint32_t channels = chunk.channels ? chunk.channels : 1U;
        if (channels == 1U) {
            pcm_chunks.emplace_back(std::move(chunk));
        } else {
            spectral_chunks.emplace_back(std::move(chunk));
        }
    }

    std::vector<StreamChunk> ring_chunks;
    if (!read_ring_snapshot(streamer, total_frames, ring_chunks)) {
        return false;
    }

    const bool queue_provided_pcm = !pcm_chunks.empty();
    if (!queue_provided_pcm) {
        // Preserve the legacy PCM ring fallback so we still emit audio if the tap queue is empty.
        // The ring is the same sink buffer MixNode exposes, so it is safe to fall back without
        // double-counting frames when the dedicated tap is healthy.
        for (StreamChunk &chunk : ring_chunks) {
            pcm_chunks.emplace_back(std::move(chunk));
        }
    }

    auto by_sequence = [](const StreamChunk &a, const StreamChunk &b) {
        return a.sequence < b.sequence;
    };
    std::sort(pcm_chunks.begin(), pcm_chunks.end(), by_sequence);
    std::sort(spectral_chunks.begin(), spectral_chunks.end(), by_sequence);

    uint64_t gathered_pcm_frames = 0;
    uint32_t pcm_channels = 1U;
    for (const StreamChunk &chunk : pcm_chunks) {
        gathered_pcm_frames += chunk.frames;
        pcm_channels = std::max<uint32_t>(pcm_channels, chunk.channels ? chunk.channels : 1U);
    }

    collection.pcm.clear();
    collection.pcm.reserve(static_cast<size_t>(gathered_pcm_frames) * static_cast<size_t>(pcm_channels));
    for (const StreamChunk &chunk : pcm_chunks) {
        collection.pcm.insert(collection.pcm.end(), chunk.data.begin(), chunk.data.end());
    }
    collection.pcm_channels = pcm_channels;
    collection.pcm_frames = total_frames;

    const size_t expected_pcm = static_cast<size_t>(collection.pcm_frames) * static_cast<size_t>(collection.pcm_channels);
    if (collection.pcm.size() < expected_pcm) {
        collection.pcm.resize(expected_pcm, 0.0);
    } else if (collection.pcm.size() > expected_pcm) {
        collection.pcm.resize(expected_pcm);
    }

    uint32_t spectral_channels = 0U;
    uint64_t spectral_frames = 0;
    for (const StreamChunk &chunk : spectral_chunks) {
        if (spectral_channels == 0U) {
            spectral_channels = chunk.channels;
        }
        spectral_channels = std::max<uint32_t>(spectral_channels, chunk.channels);
        spectral_frames += chunk.frames;
    }

    if (spectral_channels > 0U) {
        const uint32_t subchannel_stride = 3U;
        if (spectral_channels % subchannel_stride != 0U) {
            return false;
        }
        const uint32_t band_count = spectral_channels / subchannel_stride;
        collection.spectral_power.clear();
        collection.spectral_power.reserve(static_cast<size_t>(band_count) * static_cast<size_t>(spectral_frames));

        for (const StreamChunk &chunk : spectral_chunks) {
            const uint32_t channels = chunk.channels ? chunk.channels : spectral_channels;
            if (channels != spectral_channels) {
                continue;
            }
            const size_t stride = static_cast<size_t>(channels);
            for (uint32_t frame = 0; frame < chunk.frames; ++frame) {
                const size_t base = static_cast<size_t>(frame) * stride;
                for (uint32_t band = 0; band < band_count; ++band) {
                    const size_t power_index = base + static_cast<size_t>(band) * subchannel_stride + 2U;
                    if (power_index < chunk.data.size()) {
                        collection.spectral_power.push_back(chunk.data[power_index]);
                    }
                }
            }
        }

        collection.spectral_bins = band_count;
        collection.spectral_columns = static_cast<uint32_t>(spectral_frames);
    } else {
        collection.spectral_power.clear();
        collection.spectral_bins = 0U;
        collection.spectral_columns = 0U;
    }

    return true;
}

static std::vector<uint8_t> render_spectrogram_image(
    const std::vector<double> &power_columns,
    uint32_t band_count,
    uint32_t column_count,
    uint32_t &out_width,
    uint32_t &out_height
) {
    if (band_count == 0U || column_count == 0U) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    const size_t expected = static_cast<size_t>(band_count) * static_cast<size_t>(column_count);
    if (power_columns.size() < expected) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    std::vector<double> log_scaled(power_columns.begin(), power_columns.begin() + static_cast<std::ptrdiff_t>(expected));
    double max_val = -std::numeric_limits<double>::infinity();
    double min_val = std::numeric_limits<double>::infinity();
    for (double &value : log_scaled) {
        double log_mag = std::log10(std::max(value, 1e-12));
        value = log_mag;
        max_val = std::max(max_val, log_mag);
        min_val = std::min(min_val, log_mag);
    }

    if (!std::isfinite(max_val) || !std::isfinite(min_val) || max_val == min_val) {
        out_width = 0U;
        out_height = 0U;
        return {};
    }

    const double range = max_val - min_val;
    for (double &value : log_scaled) {
        value = std::clamp((value - min_val) / range, 0.0, 1.0);
    }

    std::vector<uint8_t> image(expected, 0U);
    for (uint32_t band = 0; band < band_count; ++band) {
        const uint32_t dest_row = band_count - 1U - band;
        for (uint32_t column = 0; column < column_count; ++column) {
            const size_t src_index = static_cast<size_t>(band) * column_count + column;
            const double normalized = log_scaled[src_index];
            const double pixel = 1.0 - normalized;
            image[static_cast<size_t>(dest_row) * column_count + column] = static_cast<uint8_t>(
                std::lround(std::clamp(pixel, 0.0, 1.0) * 255.0)
            );
        }
    }

    out_width = column_count;
    out_height = band_count;
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

static void print_final_overlay_snapshot(
    AmpGraphRuntime *runtime,
    AmpGraphStreamer *streamer,
    bool free_clock_mode,
    double free_clock_hz
) {
    if (!runtime || !streamer) {
        return;
    }

    std::vector<AmpGraphNodeDebugEntry> entries(32U);
    AmpGraphDebugSnapshot snapshot{};

    while (true) {
        int rc = amp_graph_runtime_debug_snapshot(
            runtime,
            streamer,
            entries.data(),
            static_cast<uint32_t>(entries.size()),
            &snapshot
        );
        if (rc < 0) {
            std::fprintf(stderr, "[demo] failed to capture final snapshot (rc=%d)\n", rc);
            return;
        }
        if (static_cast<uint32_t>(rc) > entries.size()) {
            entries.resize(static_cast<size_t>(rc));
            continue;
        }
        if (snapshot.node_count > entries.size()) {
            entries.resize(snapshot.node_count);
            continue;
        }
        entries.resize(snapshot.node_count);
        break;
    }

    if (free_clock_mode) {
        std::printf(
            "AMP KPN Final Stats | nodes:%u sink:%u | mode:%u | sr:%0.1f Hz | free:%0.2f Hz | ring:%u/%u | dumps:%u\n",
            snapshot.node_count,
            snapshot.sink_index,
            snapshot.scheduler_mode,
            snapshot.sample_rate,
            free_clock_hz,
            snapshot.ring_size,
            snapshot.ring_capacity,
            snapshot.dump_queue_depth
        );
    } else {
        std::printf(
            "AMP KPN Final Stats | nodes:%u sink:%u | mode:%u | sr:%0.1f Hz | ring:%u/%u | dumps:%u\n",
            snapshot.node_count,
            snapshot.sink_index,
            snapshot.scheduler_mode,
            snapshot.sample_rate,
            snapshot.ring_size,
            snapshot.ring_capacity,
            snapshot.dump_queue_depth
        );
    }

    std::printf(
        "Produced:%llu Consumed:%llu\n",
        static_cast<unsigned long long>(snapshot.produced_frames),
        static_cast<unsigned long long>(snapshot.consumed_frames)
    );
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf(
        "Node                             | Ring%%  | Used/Cap | Delay | Heat  | AvgProc(ms) | AvgTotal(ms) | Min/Pref | Max  | Calls | Frames | LastF | LastB | LastC\n"
    );

    for (uint32_t i = 0; i < snapshot.node_count; ++i) {
        const AmpGraphNodeDebugEntry &entry = entries[i];
        double percent = 0.0;
        if (entry.ring_capacity > 0U) {
            percent = (100.0 * static_cast<double>(entry.ring_size)) / static_cast<double>(entry.ring_capacity);
        }
        double proc_ms = entry.last_processing_time_seconds * 1000.0;
        double total_ms = entry.last_total_time_seconds * 1000.0;
        if (entry.debug_metrics_samples > 0ULL) {
            double sample_count = static_cast<double>(entry.debug_metrics_samples);
            proc_ms = (entry.debug_sum_processing_seconds / sample_count) * 1000.0;
            total_ms = (entry.debug_sum_total_seconds / sample_count) * 1000.0;
        }
        unsigned long long calls = static_cast<unsigned long long>(entry.debug_sample_count);
        unsigned long long total_frames = static_cast<unsigned long long>(entry.debug_total_frames);
        std::string node_name(entry.name);
        if (node_name.empty()) {
            node_name = "<unnamed>";
        }
        std::printf(
            "%-31.31s | %6.2f | %5u/%-5u | %5u | %5.2f | %11.3f | %12.3f | %3u/%-4u | %5u | %5llu | %6llu | %5u | %5u | %5u\n",
            node_name.c_str(),
            percent,
            entry.ring_size,
            entry.ring_capacity,
            entry.declared_delay_frames,
            static_cast<double>(entry.last_heat),
            proc_ms,
            total_ms,
            entry.debug_min_frames,
            entry.debug_preferred_frames,
            entry.debug_max_frames,
            calls,
            total_frames,
            entry.debug_last_frames,
            entry.debug_last_batches,
            entry.debug_last_channels
        );

        if (entry.tap_count == 0U) {
            continue;
        }

        std::printf("        Tap                      | Ring%%  | Used/Cap | Head/Tail | Readers | Flow(fr/s)\n");
        for (uint32_t tap_idx = 0; tap_idx < entry.tap_count; ++tap_idx) {
            const AmpGraphNodeTapDebugEntry &tap = entry.taps[tap_idx];
            double tap_percent = 0.0;
            if (tap.ring_capacity > 0U) {
                tap_percent = (100.0 * static_cast<double>(tap.ring_size)) / static_cast<double>(tap.ring_capacity);
            }
            std::printf(
                "        %-24.24s | %6.2f | %5u/%-5u | %4u/%-4u | %7u | %9.2f\n",
                tap.name[0] != '\0' ? tap.name : "default",
                tap_percent,
                tap.ring_size,
                tap.ring_capacity,
                tap.head_position,
                tap.tail_position,
                tap.reader_count,
                0.0
            );
        }
    }
}

static void print_usage(const char *exe_name) {
    std::fprintf(
        stderr,
        "Usage: %s [--overlay] [--overlay-refresh=<ms>] [--overlay-final] [--free-clock]\n",
        exe_name ? exe_name : "demo_kpn_native"
    );
}

int main(int argc, char **argv) {
    bool overlay_requested = false;
    uint32_t overlay_refresh_ms = 100U;
    bool free_clock_mode = false;
    bool overlay_final_only = false;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (std::strcmp(arg, "--overlay") == 0) {
            overlay_requested = true;
        } else if (std::strncmp(arg, "--overlay-refresh=", 19) == 0) {
            const char *value = arg + 19;
            char *endptr = nullptr;
            long parsed = std::strtol(value, &endptr, 10);
            if (endptr == value || parsed <= 0 || parsed > 10000) {
                std::fprintf(stderr, "[demo] invalid --overlay-refresh value '%s'\n", value);
                return 1;
            }
            overlay_refresh_ms = static_cast<uint32_t>(parsed);
            overlay_requested = true;
        } else if (std::strcmp(arg, "--overlay-refresh") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[demo] --overlay-refresh expects a value\n");
                return 1;
            }
            const char *value = argv[++i];
            char *endptr = nullptr;
            long parsed = std::strtol(value, &endptr, 10);
            if (endptr == value || parsed <= 0 || parsed > 10000) {
                std::fprintf(stderr, "[demo] invalid --overlay-refresh value '%s'\n", value);
                return 1;
            }
            overlay_refresh_ms = static_cast<uint32_t>(parsed);
            overlay_requested = true;
        } else if (std::strcmp(arg, "--free-clock") == 0) {
            free_clock_mode = true;
            overlay_requested = true;
        } else if (std::strcmp(arg, "--overlay-final") == 0) {
            overlay_final_only = true;
            overlay_requested = false;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[demo] unknown argument '%s'\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (overlay_final_only) {
        overlay_requested = false;
    }
    const uint32_t batches = 1U;
    const double sample_rate = 48000.0;
    const double duration_seconds = 2.0;
    const uint32_t frames = static_cast<uint32_t>(duration_seconds * sample_rate);
    double free_clock_summary_rate = 0.0;
    bool free_clock_summary_ready = false;
    bool final_snapshot_printed = false;

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
    AmpKpnOverlay *overlay = nullptr;

    do {
    const char *const *pitch_inputs = nullptr;
    static const char *const driver_inputs[] = {"pitch"};
    static const char *const osc_inputs[] = {"pitch", "driver"};
    static const char *const fft_inputs[] = {"mix"};
    static const char *const mix_inputs[] = {"osc"};

        static const char pitch_json[] =
            "{\"declared_delay\":0,\"default_slew\":0.0,\"min_freq\":0.0,"
            "\"oversample_ratio\":1,\"supports_v2\":true,"
            "\"fifo_simultaneous_output\":true,\"fifo_release_policy\":\"all\"}";
        static const char driver_json[] =
            "{\"declared_delay\":0,\"mode\":\"piezo\",\"oversample_ratio\":1,\"supports_v2\":true}";
        static const char osc_json[] =
            "{\"accept_reset\":false,\"declared_delay\":0,\"integration_clamp\":1.2,"
            "\"integration_gain\":0.5,\"integration_leak\":0.997,\"mode\":\"op_amp\","\
            "\"oversample_ratio\":1,\"slew_clamp\":1.2,\"slew_rate\":12000.0,\"supports_v2\":true,\"wave\":\"saw\"}";
        static const char fft_json[] =
            "{\"algorithm\":\"radix2\",\"declared_delay\":511,\"enable_remainder\":true,\"oversample_ratio\":1,\"supports_v2\":true,\"window_size\":512,\"hop_size\":256}";
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
                1U,
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
                2U,
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
                "fft",
                "FFTDivisionNode",
                fft_inputs,
                1U,
                fft_json,
                nullptr,
                0U
            ) != 0) {
            std::fprintf(stderr, "[demo] failed to append FFT node\n");
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

        // TODO: Load the compiled plan blob here once the serialized plan is available instead of the nullptr placeholder.
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
        amp_graph_runtime_set_scheduler_mode(runtime, AMP_SCHEDULER_ORDERED);
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

        if (overlay_requested) {
            AmpKpnOverlayConfig config{};
            config.refresh_millis = overlay_refresh_ms;
            config.ansi_only = 1;
            config.clear_on_exit = 0;
            config.enable_free_clock = free_clock_mode ? 1 : 0;
            overlay = amp_kpn_overlay_create(streamer, &config);
            if (overlay != nullptr) {
                if (amp_kpn_overlay_start(overlay) != 0) {
                    amp_kpn_overlay_destroy(overlay);
                    overlay = nullptr;
                }
            }
        }

        AmpGraphStreamerCompletionContract completion_contract{};
        completion_contract.target_produced_frames = frames;
        completion_contract.target_consumed_frames = frames;
        completion_contract.require_ring_drain = 1;
        completion_contract.require_dump_drain = 1;
        completion_contract.idle_timeout_millis = 500U;
        completion_contract.total_timeout_millis = 10000U;

        AmpGraphStreamerCompletionState completion_state{};
        AmpGraphStreamerCompletionVerdict completion_verdict{};

        for (;;) {
            int status = amp_graph_streamer_evaluate_completion(
                streamer,
                &completion_contract,
                &completion_state,
                &completion_verdict
            );
            if (status != 0) {
                std::fprintf(stderr, "[demo] streamer reported error status %d\n", status);
                exit_code = 9;
                break;
            }
            if (completion_verdict.timed_out) {
                std::fprintf(
                    stderr,
                    "[demo] streamer timed out (produced=%llu consumed=%llu ring=%u dump=%u prod_idle=%llu ms cons_idle=%llu ms dump_idle=%llu ms total=%llu ms)\n",
                    static_cast<unsigned long long>(completion_state.produced_frames),
                    static_cast<unsigned long long>(completion_state.consumed_frames),
                    completion_state.ring_size,
                    completion_state.dump_queue_depth,
                    static_cast<unsigned long long>(completion_state.since_producer_progress_millis),
                    static_cast<unsigned long long>(completion_state.since_consumer_progress_millis),
                    static_cast<unsigned long long>(completion_state.since_dump_progress_millis),
                    static_cast<unsigned long long>(completion_state.elapsed_millis)
                );
                exit_code = 9;
                break;
            }
            if (completion_verdict.contract_satisfied) {
                produced_frames = completion_state.produced_frames;
                consumed_frames = completion_state.consumed_frames;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        produced_frames = completion_state.produced_frames;
        consumed_frames = completion_state.consumed_frames;
        if (exit_code != 0) {
            break;
        }

        if (streamer_started) {
            if (overlay) {
                amp_kpn_overlay_destroy(overlay);
                overlay = nullptr;
            }
            amp_graph_streamer_stop(streamer);
            amp_graph_streamer_status(streamer, &produced_frames, &consumed_frames);
            streamer_started = false;
        }

        if (free_clock_mode) {
            double elapsed_s = completion_state.elapsed_millis > 0ULL
                ? static_cast<double>(completion_state.elapsed_millis) / 1000.0
                : 0.0;
            double rate = (elapsed_s > 0.0) ? static_cast<double>(produced_frames) / elapsed_s : 0.0;
            free_clock_summary_rate = rate;
            free_clock_summary_ready = true;
            std::printf(
                "[demo] free-clock throughput: %.2f Hz (produced %llu frames in %llums)\n",
                rate,
                static_cast<unsigned long long>(produced_frames),
                static_cast<unsigned long long>(completion_state.elapsed_millis)
            );
        }

        if (overlay_final_only) {
            double rate = (free_clock_mode && free_clock_summary_ready) ? free_clock_summary_rate : 0.0;
            print_final_overlay_snapshot(runtime, streamer, free_clock_mode, rate);
            final_snapshot_printed = true;
        }

        TapCollection taps{};
        if (!collect_tap_streams(streamer, frames, taps)) {
            std::fprintf(stderr, "[demo] failed to collect tap streams\n");
            exit_code = 10;
            break;
        }
        if (taps.pcm.empty()) {
            std::fprintf(stderr, "[demo] PCM tap stream was empty\n");
            exit_code = 10;
            break;
        }
        if (taps.spectral_power.empty()) {
            std::fprintf(stderr, "[demo] spectral tap stream was empty\n");
            exit_code = 10;
            break;
        }

        fs::path output_dir("output/demo_kpn_native");
        std::error_code dir_error;
        fs::create_directories(output_dir, dir_error);

        const fs::path wav_path = output_dir / "output.wav";
        amp_write_wav16(wav_path, taps.pcm.data(), taps.pcm_channels, frames, sample_rate);
        std::printf(
            "[demo] wrote %s (%u channels, %u frames)\n",
            wav_path.string().c_str(),
            taps.pcm_channels,
            frames
        );

        uint32_t image_width = 0U;
        uint32_t image_height = 0U;
        std::vector<uint8_t> spectrogram = render_spectrogram_image(
            taps.spectral_power,
            taps.spectral_bins,
            taps.spectral_columns,
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
        if (overlay) {
            amp_kpn_overlay_destroy(overlay);
            overlay = nullptr;
        }
        amp_graph_streamer_stop(streamer);
        amp_graph_streamer_status(streamer, &produced_frames, &consumed_frames);
        if (overlay_final_only && !final_snapshot_printed) {
            double rate = (free_clock_mode && free_clock_summary_ready) ? free_clock_summary_rate : 0.0;
            print_final_overlay_snapshot(runtime, streamer, free_clock_mode, rate);
            final_snapshot_printed = true;
        }
    }
    if (overlay) {
        amp_kpn_overlay_destroy(overlay);
        overlay = nullptr;
    }
    if (overlay_final_only && !final_snapshot_printed && streamer != nullptr) {
        double rate = (free_clock_mode && free_clock_summary_ready) ? free_clock_summary_rate : 0.0;
        print_final_overlay_snapshot(runtime, streamer, free_clock_mode, rate);
        final_snapshot_printed = true;
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
