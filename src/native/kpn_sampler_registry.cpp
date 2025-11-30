#include "include/amp_native.h"
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>

struct SamplerEntry {
    std::vector<double> samples;
    uint32_t channels{0};
    size_t read_pos{0};
};

static std::map<std::string, SamplerEntry> g_sampler_registry;
static std::mutex g_sampler_registry_mutex;

extern "C" {

AMP_CAPI int amp_kpn_session_stage_sampler_buffer(
    KpnStreamSession * /*session*/,
    const double *samples,
    size_t frames,
    uint32_t channels,
    const char *node_name
) {
    if (node_name == nullptr || samples == nullptr || frames == 0 || channels == 0) {
        return -1;
    }
    std::string key(node_name);
    size_t total = frames * (size_t)channels;
    try {
        std::lock_guard<std::mutex> lock(g_sampler_registry_mutex);
        SamplerEntry &entry = g_sampler_registry[key];
        entry.samples.assign(samples, samples + total);
        entry.channels = channels;
        entry.read_pos = 0;
    } catch(...) {
        return -2;
    }
    return 0;
}

AMP_CAPI int amp_sampler_unregister(const char *node_name) {
    if (node_name == nullptr) return -1;
    std::lock_guard<std::mutex> lock(g_sampler_registry_mutex);
    auto it = g_sampler_registry.find(node_name);
    if (it == g_sampler_registry.end()) return -2;
    g_sampler_registry.erase(it);
    return 0;
}

AMP_CAPI int amp_sampler_peek(const char *node_name, const double **out_samples, size_t *out_frames, uint32_t *out_channels, size_t *out_read_pos) {
    if (node_name == nullptr || out_samples == nullptr || out_frames == nullptr || out_channels == nullptr || out_read_pos == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_sampler_registry_mutex);
    auto it = g_sampler_registry.find(node_name);
    if (it == g_sampler_registry.end()) {
        return -2;
    }
    const SamplerEntry &entry = it->second;
    if (entry.samples.empty() || entry.channels == 0) {
        return -3;
    }
    *out_samples = entry.samples.data();
    *out_frames = entry.samples.size() / (size_t)entry.channels;
    *out_channels = entry.channels;
    *out_read_pos = entry.read_pos;
    return 0;
}

AMP_CAPI int amp_sampler_advance(const char *node_name, size_t consumed_frames) {
    if (node_name == nullptr) return -1;
    std::lock_guard<std::mutex> lock(g_sampler_registry_mutex);
    auto it = g_sampler_registry.find(node_name);
    if (it == g_sampler_registry.end()) return -2;
    SamplerEntry &entry = it->second;
    size_t total_frames = entry.samples.size() / (size_t)entry.channels;
    if (consumed_frames >= total_frames - entry.read_pos) {
        entry.read_pos = total_frames; // consume to end
    } else {
        entry.read_pos += consumed_frames;
    }
    return 0;
}

} // extern "C"
