#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <memory>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>
#include <deque>
#include <condition_variable>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

extern "C" {
#include "amp_native.h"
}

#if defined(_WIN32) || defined(_WIN64)
#  define AMP_API __declspec(dllexport)
#else
#  define AMP_API
#endif

#if defined(AMP_NATIVE_ENABLE_LOGGING)
extern "C" {
int amp_native_logging_enabled(void);
void amp_log_native_call_external(const char *fn, size_t a, size_t b);
}

static void _log_native_call(const char *fn, size_t a, size_t b) {
    if (!amp_native_logging_enabled()) {
        return;
    }
    if (&amp_log_native_call_external) {
        amp_log_native_call_external(fn, a, b);
    }
}
#  define AMP_LOG_NATIVE_CALL(fn, a, b) _log_native_call((fn), (a), (b))
#else
#  define AMP_LOG_NATIVE_CALL(fn, a, b) ((void)0)
#endif

extern "C" {
extern EdgeRunnerControlHistory *amp_load_control_history(
    const uint8_t *blob,
    size_t blob_len,
    int frames_hint
);
extern void amp_release_control_history(EdgeRunnerControlHistory *history);
extern int amp_run_node(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history
);
extern int amp_run_node_v2(
    const EdgeRunnerNodeDescriptor *descriptor,
    const EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
);
extern void amp_free(double *buffer);
extern void amp_release_state(void *state);
}

using EigenTensor = Eigen::Tensor<double, 3, Eigen::RowMajor>;

struct TensorShape {
    uint32_t batches{1};
    uint32_t channels{1};
    uint32_t frames{1};

    size_t element_count() const {
        return static_cast<size_t>(batches) * static_cast<size_t>(channels) * static_cast<size_t>(frames);
    }
};

struct EigenTensorHolder {
    TensorShape shape;
    EigenTensor values;

    explicit EigenTensorHolder(const TensorShape &shape_)
        : shape(shape_), values(static_cast<long>(std::max<uint32_t>(1, shape_.batches)), static_cast<long>(std::max<uint32_t>(1, shape_.channels)), static_cast<long>(std::max<uint32_t>(1, shape_.frames))) {
        values.setZero();
    }

    double *data() {
        return values.data();
    }

    const double *data() const {
        return values.data();
    }
};

struct ModConnectionInfo {
    std::string source;
    std::string param;
    double scale{1.0};
    int mode{0};
    int channel{-1};
};

struct DefaultParam {
    std::string name;
    TensorShape shape;
    std::vector<double> data;
};

struct ParamBinding {
    TensorShape shape;
    std::vector<double> data;
    bool dirty{true};
    bool use_ring{false};
    uint32_t ring_frames{0};
    uint32_t ring_head{0};
    uint32_t window_frames{0};
    size_t frame_stride{0};
    TensorShape full_shape{};
};

struct RuntimeNode;

struct KahnChannel {
    std::string name;
    std::shared_ptr<EigenTensorHolder> token;

    explicit KahnChannel(std::string name_) : name(std::move(name_)) {}
};

struct RuntimeErrorState {
    int code{0};
    std::string stage;
    std::string node;
    std::string detail;
};

struct RuntimeNode {
    std::string name;
    std::string type_name;
    uint32_t type_id{0};
    std::vector<std::string> audio_inputs;
    std::vector<uint32_t> audio_indices;
    std::vector<ModConnectionInfo> mod_connections;
    std::string params_json;
    std::vector<DefaultParam> defaults;
    std::unordered_map<std::string, ParamBinding> bindings;
    std::vector<TensorShape> buffer_shapes;
    std::shared_ptr<EigenTensorHolder> output;
    uint32_t output_batches{0};
    uint32_t output_channels{0};
    uint32_t output_frames{0};
    std::shared_ptr<EigenTensorHolder> output_ring;
    uint32_t output_ring_capacity{0};
    uint32_t output_ring_head{0};
    uint32_t channel_hint{1};
    void *state{nullptr};
    EdgeRunnerNodeDescriptor descriptor{};
    uint32_t oversample_ratio{1};
    uint32_t declared_delay_frames{0};
    bool supports_v2{true};
    bool has_latest_metrics{false};
    AmpNodeMetrics latest_metrics{};
    double total_heat_accumulated{0.0};
    std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> param_cache;
    std::unordered_map<std::string, size_t> param_cache_index;
    bool param_cache_dirty{true};

    RuntimeNode() = default;

    void finalize_descriptor() {
        descriptor.name = name.c_str();
        descriptor.name_len = name.size();
        descriptor.type_name = type_name.c_str();
        descriptor.type_len = type_name.size();
        descriptor.params_json = params_json.c_str();
        descriptor.params_len = params_json.size();
    }
};

struct DescriptorReader {
    const uint8_t *cursor;
    size_t remaining;

    DescriptorReader(const uint8_t *data, size_t length) : cursor(data), remaining(length) {}

    bool read_u32(uint32_t &out) {
        if (remaining < sizeof(uint32_t)) {
            return false;
        }
        uint32_t value = 0;
        std::memcpy(&value, cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);
        out = value;
        return true;
    }

    bool read_u64(uint64_t &out) {
        if (remaining < sizeof(uint64_t)) {
            return false;
        }
        uint64_t value = 0;
        std::memcpy(&value, cursor, sizeof(uint64_t));
        cursor += sizeof(uint64_t);
        remaining -= sizeof(uint64_t);
        out = value;
        return true;
    }

    bool read_float(float &out) {
        if (remaining < sizeof(float)) {
            return false;
        }
        float value = 0.0f;
        std::memcpy(&value, cursor, sizeof(float));
        cursor += sizeof(float);
        remaining -= sizeof(float);
        out = value;
        return true;
    }

    bool read_string(size_t length, std::string &out) {
        if (remaining < length) {
            return false;
        }
        out.assign(reinterpret_cast<const char *>(cursor), length);
        cursor += length;
        remaining -= length;
        return true;
    }

    bool skip(size_t length) {
        if (remaining < length) {
            return false;
        }
        cursor += length;
        remaining -= length;
        return true;
    }
};

static uint32_t parse_uint_metadata(const std::string &json, const char *key, uint32_t fallback) {
    if (key == nullptr) {
        return fallback;
    }
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }
    if (pos >= json.size()) {
        return fallback;
    }
    if (json[pos] == '-') {
        return fallback;
    }
    uint64_t value = 0;
    bool found_digit = false;
    while (pos < json.size()) {
        unsigned char ch = static_cast<unsigned char>(json[pos]);
        if (!std::isdigit(ch)) {
            break;
        }
        found_digit = true;
        value = value * 10ULL + static_cast<uint64_t>(ch - static_cast<unsigned char>('0'));
        if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return fallback;
        }
        ++pos;
    }
    if (!found_digit) {
        return fallback;
    }
    return static_cast<uint32_t>(value);
}

static bool parse_bool_metadata(const std::string &json, const char *key, bool fallback) {
    if (key == nullptr) {
        return fallback;
    }
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }
    if (pos + 4U <= json.size() && json.compare(pos, 4U, "true") == 0) {
        return true;
    }
    if (pos + 5U <= json.size() && json.compare(pos, 5U, "false") == 0) {
        return false;
    }
    return fallback;
}

struct AmpGraphRuntime {
    std::vector<std::unique_ptr<RuntimeNode>> nodes;
    std::unordered_map<std::string, size_t> node_index;
    std::vector<uint32_t> execution_order;
    std::unordered_map<std::string, std::shared_ptr<KahnChannel>> channels;
    uint32_t sink_index{0};
    uint32_t default_batches{1};
    uint32_t default_frames{0};
    double dsp_sample_rate{0.0};
    RuntimeErrorState last_error;
};

struct StreamDumpChunk {
    std::vector<double> data;
    uint32_t batches{0};
    uint32_t channels{0};
    uint32_t frames{0};
    uint64_t sequence{0};
};

struct AmpGraphStreamer {
    AmpGraphRuntime *runtime{nullptr};
    std::vector<uint8_t> control_blob;
    AmpGraphControlHistory *history{nullptr};
    bool history_owned{false};
    double sample_rate{0.0};
    uint32_t ring_frames{0};
    uint32_t block_frames{0};
    uint32_t batches{0};
    uint32_t channels{0};
    size_t frame_stride{0};
    std::vector<double> ring_buffer;
    std::atomic<uint64_t> write_index{0};
    std::atomic<uint64_t> read_index{0};
    std::atomic<uint64_t> produced_frames{0};
    std::atomic<uint64_t> consumed_frames{0};
    std::deque<StreamDumpChunk> dump_queue;
    std::mutex dump_mutex;
    std::condition_variable dump_cv;
    std::thread worker;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    int frames_hint{0};
    int last_status{0};
    mutable std::mutex status_mutex;
};

static void runtime_clear_error(AmpGraphRuntime *runtime) {
    if (runtime == nullptr) {
        return;
    }
    runtime->last_error.code = 0;
    runtime->last_error.stage.clear();
    runtime->last_error.node.clear();
    runtime->last_error.detail.clear();
}

static void runtime_set_error(
    AmpGraphRuntime *runtime,
    int code,
    const char *stage,
    const RuntimeNode *node,
    std::string detail
) {
    if (runtime == nullptr) {
        return;
    }
    runtime->last_error.code = code;
    runtime->last_error.stage = stage != nullptr ? stage : "";
    runtime->last_error.node = node != nullptr ? node->name : "";
    runtime->last_error.detail = std::move(detail);
}

static TensorShape make_shape(uint32_t batches, uint32_t channels, uint32_t frames) {
    TensorShape shape{};
    shape.batches = std::max<uint32_t>(1U, batches);
    shape.channels = std::max<uint32_t>(1U, channels);
    shape.frames = std::max<uint32_t>(1U, frames);
    return shape;
}

static bool shapes_equal(const TensorShape &lhs, const TensorShape &rhs) {
    return lhs.batches == rhs.batches && lhs.channels == rhs.channels && lhs.frames == rhs.frames;
}

static std::shared_ptr<EigenTensorHolder> make_tensor(const TensorShape &shape) {
    auto tensor = std::make_shared<EigenTensorHolder>(shape);
    return tensor;
}

static bool parse_node_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t length) {
    if (runtime == nullptr || blob == nullptr || length == 0) {
        return false;
    }
    DescriptorReader reader(blob, length);
    uint32_t node_count = 0;
    if (!reader.read_u32(node_count)) {
        return false;
    }
    runtime->nodes.reserve(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        uint32_t type_id = 0;
        uint32_t name_len = 0;
        uint32_t type_len = 0;
        uint32_t audio_count = 0;
        uint32_t mod_count = 0;
        uint32_t param_count = 0;
        uint32_t shape_count = 0;
        uint32_t params_len = 0;
        if (!reader.read_u32(type_id) || !reader.read_u32(name_len) || !reader.read_u32(type_len) ||
            !reader.read_u32(audio_count) || !reader.read_u32(mod_count) || !reader.read_u32(param_count) ||
            !reader.read_u32(shape_count) || !reader.read_u32(params_len)) {
            return false;
        }
        auto node = std::make_unique<RuntimeNode>();
        node->type_id = type_id;
        if (!reader.read_string(name_len, node->name) || !reader.read_string(type_len, node->type_name)) {
            return false;
        }
        for (uint32_t a = 0; a < audio_count; ++a) {
            uint32_t src_len = 0;
            if (!reader.read_u32(src_len)) {
                return false;
            }
            std::string source;
            if (!reader.read_string(src_len, source)) {
                return false;
            }
            node->audio_inputs.push_back(std::move(source));
        }
        for (uint32_t m = 0; m < mod_count; ++m) {
        uint32_t src_len = 0;
        uint32_t param_len = 0;
        uint32_t mode_code = 0;
        float scale = 0.0f;
        uint32_t channel_raw = 0;
        if (!reader.read_u32(src_len) || !reader.read_u32(param_len) || !reader.read_u32(mode_code) ||
            !reader.read_float(scale) || !reader.read_u32(channel_raw)) {
            return false;
        }
        std::string src;
        std::string param;
        if (!reader.read_string(src_len, src) || !reader.read_string(param_len, param)) {
            return false;
        }
        ModConnectionInfo info{};
        info.source = std::move(src);
        info.param = std::move(param);
        info.scale = static_cast<double>(scale);
        info.mode = mode_code == 1U ? 1 : 0;
        info.channel = static_cast<int>(channel_raw);
        node->mod_connections.push_back(std::move(info));
    }
        for (uint32_t p = 0; p < param_count; ++p) {
            uint32_t param_name_len = 0;
            uint32_t batches = 0;
            uint32_t channels = 0;
            uint32_t frames = 0;
            uint64_t blob_len = 0;
            if (!reader.read_u32(param_name_len) || !reader.read_u32(batches) || !reader.read_u32(channels) ||
                !reader.read_u32(frames) || !reader.read_u64(blob_len)) {
                return false;
            }
            std::string param_name;
        if (!reader.read_string(param_name_len, param_name)) {
            return false;
        }
        DefaultParam param{};
        param.name = std::move(param_name);
        param.shape = make_shape(batches, channels, frames);
        size_t value_count = static_cast<size_t>(blob_len) / sizeof(double);
        param.data.resize(value_count);
        if (reader.remaining < blob_len) {
            return false;
        }
        std::memcpy(param.data.data(), reader.cursor, blob_len);
        reader.skip(blob_len);
        node->param_cache_index[param.name] = node->param_cache.size();
        node->param_cache.emplace_back(param.name, std::shared_ptr<EigenTensorHolder>());
        node->defaults.push_back(std::move(param));
        node->param_cache_dirty = true;
        }
        for (uint32_t s = 0; s < shape_count; ++s) {
            uint32_t b = 0, c = 0, f = 0;
            if (!reader.read_u32(b) || !reader.read_u32(c) || !reader.read_u32(f)) {
                return false;
            }
            node->buffer_shapes.push_back(make_shape(b, c, f));
            if (node->channel_hint < c) {
                node->channel_hint = c;
            }
        }
        if (!reader.read_string(params_len, node->params_json)) {
            return false;
        }
        node->finalize_descriptor();
        node->oversample_ratio = std::max<uint32_t>(1U, parse_uint_metadata(node->params_json, "oversample_ratio", 1U));
        node->declared_delay_frames = parse_uint_metadata(node->params_json, "declared_delay", 0U);
        node->supports_v2 = parse_bool_metadata(node->params_json, "supports_v2", node->supports_v2);
        runtime->node_index[node->name] = runtime->nodes.size();
        runtime->nodes.push_back(std::move(node));
    }
    for (auto &entry : runtime->nodes) {
        entry->audio_indices.clear();
        for (const std::string &name : entry->audio_inputs) {
            auto it = runtime->node_index.find(name);
            if (it == runtime->node_index.end()) {
                return false;
            }
            entry->audio_indices.push_back(static_cast<uint32_t>(it->second));
        }
        runtime->channels.emplace(entry->name, std::make_shared<KahnChannel>(entry->name));
    }
    if (!runtime->nodes.empty()) {
        runtime->sink_index = static_cast<uint32_t>(runtime->nodes.size() - 1);
    }
    return true;
}

static bool parse_plan_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t length) {
    if (runtime == nullptr) {
        return false;
    }
    if (blob == nullptr || length == 0) {
        runtime->execution_order.clear();
        for (size_t i = 0; i < runtime->nodes.size(); ++i) {
            runtime->execution_order.push_back(static_cast<uint32_t>(i));
        }
        if (!runtime->execution_order.empty()) {
            runtime->sink_index = runtime->execution_order.back();
        }
        return true;
    }
    if (length < 12U) {
        return false;
    }
    if (std::memcmp(blob, "AMPL", 4) != 0) {
        return false;
    }
    DescriptorReader reader(blob + 4, length - 4);
    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!reader.read_u32(version) || !reader.read_u32(node_count)) {
        return false;
    }
    if (version == 0U || version > 2U) {
        return false;
    }
    if (node_count == 0U) {
        return true;
    }
    runtime->execution_order.assign(node_count, 0U);
    for (uint32_t i = 0; i < node_count; ++i) {
        uint32_t function_id = 0;
        uint32_t name_len = 0;
        uint32_t audio_offset = 0;
        uint32_t audio_span = 0;
        uint32_t param_count = 0;
        uint32_t declared_delay = 0;
        uint32_t oversample_ratio = 1;
        if (!reader.read_u32(function_id) || !reader.read_u32(name_len) || !reader.read_u32(audio_offset) ||
            !reader.read_u32(audio_span) || !reader.read_u32(param_count)) {
            return false;
        }
        if (version >= 2U) {
            if (!reader.read_u32(declared_delay) || !reader.read_u32(oversample_ratio)) {
                return false;
            }
        }
        (void)audio_offset;
        std::string node_name;
        if (!reader.read_string(name_len, node_name)) {
            return false;
        }
        auto it = runtime->node_index.find(node_name);
        if (it == runtime->node_index.end()) {
            return false;
        }
        if (function_id >= runtime->execution_order.size()) {
            return false;
        }
        runtime->execution_order[function_id] = static_cast<uint32_t>(it->second);
        auto &node = runtime->nodes[it->second];
        if (version >= 2U) {
            if (declared_delay > 0U) {
                node->declared_delay_frames = declared_delay;
            }
            if (oversample_ratio > 0U) {
                node->oversample_ratio = oversample_ratio;
            }
        }
        if (audio_span > 0 && node->channel_hint < audio_span) {
            node->channel_hint = audio_span;
        }
        for (uint32_t p = 0; p < param_count; ++p) {
            uint32_t param_len = 0;
            uint32_t cursor = 0;
            uint32_t reserved = 0;
            if (!reader.read_u32(param_len) || !reader.read_u32(cursor) || !reader.read_u32(reserved)) {
                return false;
            }
            std::string param_name;
            if (!reader.read_string(param_len, param_name)) {
                return false;
            }
            (void)param_name;
            (void)cursor;
            (void)reserved;
        }
    }
    if (!runtime->execution_order.empty()) {
        runtime->sink_index = runtime->execution_order.back();
    }
    return true;
}

static std::shared_ptr<EigenTensorHolder> clone_param_tensor(const DefaultParam &param) {
    TensorShape shape = param.shape;
    auto tensor = make_tensor(shape);
    if (!param.data.empty()) {
        size_t copy_size = std::min(param.data.size(), static_cast<size_t>(tensor->values.size()));
        std::memcpy(tensor->data(), param.data.data(), copy_size * sizeof(double));
    }
    tensor->shape = shape;
    return tensor;
}

static std::shared_ptr<EigenTensorHolder> clone_binding_tensor(const ParamBinding &binding) {
    TensorShape shape = binding.shape;
    auto tensor = make_tensor(shape);
    if (!binding.data.empty()) {
        size_t copy_size = std::min(binding.data.size(), static_cast<size_t>(tensor->values.size()));
        std::memcpy(tensor->data(), binding.data.data(), copy_size * sizeof(double));
    }
    tensor->shape = shape;
    return tensor;
}

static std::shared_ptr<EigenTensorHolder> merge_audio_inputs(
    AmpGraphRuntime *runtime,
    RuntimeNode &node,
    std::vector<std::shared_ptr<EigenTensorHolder>> &workspace
) {
    if (runtime == nullptr) {
        return nullptr;
    }
    if (node.audio_indices.empty()) {
        return nullptr;
    }
    if (node.audio_indices.size() == 1U) {
        auto &source = runtime->nodes[node.audio_indices[0]];
        return source->output;
    }
    std::vector<std::shared_ptr<EigenTensorHolder>> sources;
    uint32_t batches = 0;
    uint32_t frames = 0;
    uint32_t total_channels = 0;
    for (uint32_t index : node.audio_indices) {
        if (index >= runtime->nodes.size()) {
            return nullptr;
        }
        auto &source = runtime->nodes[index];
        if (!source->output) {
            return nullptr;
        }
        if (sources.empty()) {
            batches = source->output->shape.batches;
            frames = source->output->shape.frames;
        } else {
            if (source->output->shape.batches != batches || source->output->shape.frames != frames) {
                return nullptr;
            }
        }
        total_channels += source->output->shape.channels;
        sources.push_back(source->output);
    }
    if (sources.empty() || total_channels == 0U) {
        return nullptr;
    }
    TensorShape shape{};
    shape.batches = batches;
    shape.frames = frames;
    shape.channels = total_channels;
    auto merged = make_tensor(shape);
    merged->shape = shape;
    double *dest = merged->data();
    size_t offset = 0;
    for (const auto &src : sources) {
        const double *src_ptr = src->data();
        size_t block = static_cast<size_t>(src->shape.batches) * static_cast<size_t>(src->shape.channels) * static_cast<size_t>(src->shape.frames);
        std::memcpy(dest + offset, src_ptr, block * sizeof(double));
        offset += block;
    }
    workspace.push_back(merged);
    return merged;
}

static void apply_modulation(
    const ModConnectionInfo &info,
    const std::shared_ptr<EigenTensorHolder> &source,
    std::shared_ptr<EigenTensorHolder> &target
) {
    if (!source || !target) {
        return;
    }
    const TensorShape &src_shape = source->shape;
    TensorShape dst_shape = target->shape;
    size_t src_stride = static_cast<size_t>(src_shape.batches) * static_cast<size_t>(src_shape.channels);
    size_t dst_stride = static_cast<size_t>(dst_shape.batches) * static_cast<size_t>(dst_shape.channels);
    if (src_shape.frames != dst_shape.frames || src_stride == 0 || dst_stride == 0) {
        return;
    }
    const double *src_ptr = source->data();
    double *dst_ptr = target->data();
    for (uint32_t frame = 0; frame < dst_shape.frames; ++frame) {
        const double *src_frame = src_ptr + frame * src_stride;
        double *dst_frame = dst_ptr + frame * dst_stride;
        for (uint32_t batch = 0; batch < dst_shape.batches; ++batch) {
            for (uint32_t channel = 0; channel < dst_shape.channels; ++channel) {
                uint32_t src_channel = channel;
                if (info.channel >= 0) {
                    src_channel = static_cast<uint32_t>(info.channel);
                    if (src_channel >= src_shape.channels) {
                        continue;
                    }
                } else if (src_shape.channels <= channel) {
                    src_channel = src_shape.channels - 1U;
                }
                size_t src_index = batch * src_shape.channels + src_channel;
                size_t dst_index = batch * dst_shape.channels + channel;
                double value = src_frame[src_index] * info.scale;
                if (info.mode == 1) {
                    dst_frame[dst_index] *= value;
                } else {
                    dst_frame[dst_index] += value;
                }
            }
        }
    }
}

static const DefaultParam *find_default_param(const RuntimeNode &node, const std::string &name) {
    for (const DefaultParam &param : node.defaults) {
        if (param.name == name) {
            return &param;
        }
    }
    return nullptr;
}

static std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> &build_param_tensors(
    RuntimeNode &node
) {
    for (const auto &binding_entry : node.bindings) {
        if (node.param_cache_index.find(binding_entry.first) == node.param_cache_index.end()) {
            node.param_cache_index[binding_entry.first] = node.param_cache.size();
            node.param_cache.emplace_back(binding_entry.first, std::shared_ptr<EigenTensorHolder>());
            node.param_cache_dirty = true;
        }
    }
    if (!node.param_cache_dirty) {
        return node.param_cache;
    }
    for (auto &entry : node.param_cache) {
        const std::string &name = entry.first;
        std::shared_ptr<EigenTensorHolder> tensor = entry.second;
        auto binding_it = node.bindings.find(name);
        if (binding_it != node.bindings.end()) {
            ParamBinding &binding = binding_it->second;
            if (!tensor || !shapes_equal(tensor->shape, binding.shape)) {
                tensor = make_tensor(binding.shape);
                entry.second = tensor;
            }
            if (binding.use_ring || binding.dirty) {
                tensor->shape = binding.shape;
                double *dst = tensor->data();
                const double *src = binding.data.data();
                size_t frame_stride = binding.frame_stride;
                uint32_t window_frames = binding.window_frames;
                if (!binding.use_ring) {
                    size_t copy_size = std::min(binding.data.size(), static_cast<size_t>(tensor->values.size()));
                    if (copy_size < static_cast<size_t>(tensor->values.size())) {
                        tensor->values.setZero();
                    }
                    if (copy_size > 0U) {
                        std::memcpy(dst, src, copy_size * sizeof(double));
                    }
                } else {
                    tensor->values.setZero();
                    if (frame_stride > 0U && binding.ring_frames > 0U) {
                        for (uint32_t f = 0; f < window_frames; ++f) {
                            uint32_t ring_pos = static_cast<uint32_t>((binding.ring_head + f) % binding.ring_frames);
                            std::memcpy(
                                dst + static_cast<size_t>(f) * frame_stride,
                                src + static_cast<size_t>(ring_pos) * frame_stride,
                                frame_stride * sizeof(double)
                            );
                        }
                    }
                }
            }
            binding.dirty = false;
            continue;
        }
        const DefaultParam *def = find_default_param(node, name);
        if (def == nullptr) {
            if (tensor) {
                tensor->values.setZero();
            }
            continue;
        }
        if (!tensor || !shapes_equal(tensor->shape, def->shape)) {
            tensor = make_tensor(def->shape);
            entry.second = tensor;
        }
        if (!def->data.empty()) {
            size_t copy_size = std::min(def->data.size(), static_cast<size_t>(tensor->values.size()));
            if (copy_size < static_cast<size_t>(tensor->values.size())) {
                tensor->values.setZero();
            }
            std::memcpy(tensor->data(), def->data.data(), copy_size * sizeof(double));
        } else {
            tensor->values.setZero();
        }
        tensor->shape = def->shape;
    }
    node.param_cache_dirty = false;
    return node.param_cache;
}

static std::shared_ptr<EigenTensorHolder> ensure_param_tensor(
    std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> &tensors,
    const std::string &name,
    const TensorShape &fallback_shape
) {
    for (auto &entry : tensors) {
        if (entry.first == name) {
            return entry.second;
        }
    }
    auto tensor = make_tensor(fallback_shape);
    tensor->shape = fallback_shape;
    tensors.emplace_back(name, tensor);
    return tensor;
}

struct HistoryGuard {
    EdgeRunnerControlHistory *ptr;
    bool owned;

    HistoryGuard(EdgeRunnerControlHistory *p, bool take) : ptr(p), owned(take) {}
    ~HistoryGuard() {
        if (owned && ptr != nullptr) {
            amp_release_control_history(ptr);
        }
    }
};

static int execute_runtime_with_history_impl(
    AmpGraphRuntime *runtime,
    EdgeRunnerControlHistory *history,
    bool history_owned,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    if (runtime == nullptr) {
        return -1;
    }
    runtime_clear_error(runtime);
    if (out_buffer == nullptr || out_batches == nullptr || out_channels == nullptr || out_frames == nullptr) {
        runtime_set_error(runtime, -1, "execute_runtime", nullptr, "output buffers must not be null");
        return -1;
    }
    HistoryGuard guard(history, history_owned);
    const EdgeRunnerControlHistory *history_view = history;
    for (auto &entry : runtime->nodes) {
        if (entry->output) {
            entry->output.reset();
        }
        entry->output_batches = 0;
        entry->output_channels = 0;
        entry->output_frames = 0;
    }
    for (auto &channel_entry : runtime->channels) {
        channel_entry.second->token.reset();
    }
    double dsp_rate = runtime->dsp_sample_rate > 0.0 ? runtime->dsp_sample_rate : sample_rate;
    std::vector<std::shared_ptr<EigenTensorHolder>> scratch;
    scratch.reserve(8);
    for (uint32_t order : runtime->execution_order) {
        if (order >= runtime->nodes.size()) {
            std::string detail = std::string("execution order index out of range: ") + std::to_string(order);
            runtime_set_error(runtime, -1, "schedule_bounds", nullptr, std::move(detail));
            return -1;
        }
        RuntimeNode &node = *runtime->nodes[order];
        scratch.clear();
        std::shared_ptr<EigenTensorHolder> audio_tensor = merge_audio_inputs(runtime, node, scratch);
        if (!audio_tensor && !node.audio_indices.empty()) {
            runtime_set_error(runtime, -1, "merge_audio_inputs", &node, "required audio input missing");
            return -1;
        }
        node.has_latest_metrics = false;
        node.total_heat_accumulated = 0.0;
        node.latest_metrics = {};
        uint32_t batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
        uint32_t frames = runtime->default_frames > 0U ? runtime->default_frames : 1U;
        uint32_t input_channels = node.channel_hint > 0U ? node.channel_hint : 1U;
        if (audio_tensor) {
            batches = audio_tensor->shape.batches;
            frames = audio_tensor->shape.frames;
            input_channels = audio_tensor->shape.channels;
        }
        auto &param_tensors = build_param_tensors(node);
        for (const ModConnectionInfo &mod : node.mod_connections) {
            auto channel_it = runtime->channels.find(mod.source);
            if (channel_it == runtime->channels.end()) {
                continue;
            }
            std::shared_ptr<EigenTensorHolder> source_tensor = channel_it->second->token;
            if (!source_tensor) {
                continue;
            }
            TensorShape shape = source_tensor->shape;
            auto target_tensor = ensure_param_tensor(param_tensors, mod.param, shape);
            apply_modulation(mod, source_tensor, target_tensor);
        }
        std::vector<EdgeRunnerParamView> param_views;
        param_views.reserve(param_tensors.size());
        for (auto &entry : param_tensors) {
            EdgeRunnerParamView view{};
            view.name = entry.first.c_str();
            view.batches = entry.second->shape.batches;
            view.channels = entry.second->shape.channels;
            view.frames = entry.second->shape.frames;
            view.data = entry.second->data();
            param_views.push_back(view);
        }
        EdgeRunnerNodeInputs inputs{};
        if (audio_tensor) {
            inputs.audio.has_audio = 1;
            inputs.audio.batches = audio_tensor->shape.batches;
            inputs.audio.channels = audio_tensor->shape.channels;
            inputs.audio.frames = 1;
            inputs.audio.data = audio_tensor->data();
        } else {
            inputs.audio.has_audio = 0;
            inputs.audio.batches = batches;
            inputs.audio.channels = 0;
            inputs.audio.frames = 1;
            inputs.audio.data = nullptr;
        }
        if (!param_views.empty()) {
            inputs.params.count = static_cast<uint32_t>(param_views.size());
            inputs.params.items = param_views.data();
        }
        std::shared_ptr<EigenTensorHolder> node_output;
        double *frame_buffer = nullptr;
        int out_channels = 0;
        void *state = node.state;
        size_t audio_stride = audio_tensor ? static_cast<size_t>(audio_tensor->shape.batches) * static_cast<size_t>(audio_tensor->shape.channels) : 0U;
        std::vector<size_t> param_strides(param_views.size(), 0U);
        for (size_t idx = 0; idx < param_views.size(); ++idx) {
            size_t stride = static_cast<size_t>(param_views[idx].batches) * static_cast<size_t>(param_views[idx].channels);
            param_strides[idx] = stride > 0U ? stride : 1U;
        }
        for (uint32_t frame_index = 0; frame_index < frames; ++frame_index) {
            if (audio_tensor) {
                inputs.audio.data = audio_tensor->data() + frame_index * audio_stride;
            }
            for (size_t view_index = 0; view_index < param_views.size(); ++view_index) {
                if (param_views[view_index].data != nullptr) {
                    param_views[view_index].frames = 1;
                    param_views[view_index].data = param_tensors[view_index].second->data() + frame_index * param_strides[view_index];
                }
            }
            frame_buffer = nullptr;
            out_channels = 0;
            void *state_arg = state;
            bool used_v2 = false;
            if (node.supports_v2) {
                AmpNodeMetrics frame_metrics{};
                int v2_status = amp_run_node_v2(
                    &node.descriptor,
                    &inputs,
                    static_cast<int>(batches),
                    static_cast<int>(input_channels),
                    1,
                    dsp_rate,
                    &frame_buffer,
                    &out_channels,
                    &state_arg,
                    history_view,
                    AMP_EXECUTION_MODE_FORWARD,
                    &frame_metrics
                );
                if (v2_status == AMP_E_UNSUPPORTED) {
                    node.supports_v2 = false;
                    node.has_latest_metrics = false;
                } else if (v2_status != 0) {
                    if (frame_buffer != nullptr) {
                        amp_free(frame_buffer);
                    }
                    std::string detail = std::string("amp_run_node_v2 returned status ") + std::to_string(v2_status)
                        + std::string(" at frame ") + std::to_string(frame_index);
                    runtime_set_error(runtime, v2_status, "run_node_v2", &node, std::move(detail));
                    return -1;
                } else {
                    used_v2 = true;
                    node.has_latest_metrics = true;
                    node.latest_metrics = frame_metrics;
                    node.total_heat_accumulated += static_cast<double>(frame_metrics.accumulated_heat);
                }
            }
            if (!used_v2) {
                int status = amp_run_node(
                    &node.descriptor,
                    &inputs,
                    static_cast<int>(batches),
                    static_cast<int>(input_channels),
                    1,
                    dsp_rate,
                    &frame_buffer,
                    &out_channels,
                    &state_arg,
                    history_view
                );
                if (status != 0 || frame_buffer == nullptr) {
                    if (frame_buffer != nullptr) {
                        amp_free(frame_buffer);
                    }
                    std::string detail;
                    int code = status;
                    if (frame_buffer == nullptr && status == 0) {
                        detail = "amp_run_node returned null buffer";
                        code = -1;
                    } else {
                        detail = std::string("amp_run_node returned status ") + std::to_string(status)
                            + std::string(" at frame ") + std::to_string(frame_index);
                    }
                    runtime_set_error(runtime, code, "run_node", &node, std::move(detail));
                    return -1;
                }
                node.has_latest_metrics = false;
            } else if (frame_buffer == nullptr) {
                runtime_set_error(runtime, -1, "run_node_v2", &node, "amp_run_node_v2 returned null buffer");
                return -1;
            }
            if (!node_output) {
                TensorShape shape{};
                shape.batches = batches;
                shape.channels = static_cast<uint32_t>(out_channels);
                shape.frames = frames;
                node_output = make_tensor(shape);
                node_output->shape = shape;
            }
            if (node.output_ring_capacity == 0U) {
                node.output_ring_capacity = node_output->shape.frames;
            }
            uint32_t history_frames = node.output_ring_capacity > 0U ? node.output_ring_capacity : node_output->shape.frames;
            if (!node.output_ring || !node.output_ring->values.size()
                || node_output->shape.batches != node.output_ring->shape.batches
                || node_output->shape.channels != node.output_ring->shape.channels
                || node.output_ring->shape.frames != history_frames) {
                TensorShape ring_shape{};
                ring_shape.batches = node_output->shape.batches;
                ring_shape.channels = node_output->shape.channels;
                ring_shape.frames = history_frames;
                node.output_ring = make_tensor(ring_shape);
                node.output_ring->shape = ring_shape;
                node.output_ring_head = 0;
            }
            size_t stride = static_cast<size_t>(node_output->shape.batches) * static_cast<size_t>(node_output->shape.channels);
            std::memcpy(
                node_output->data() + frame_index * stride,
                frame_buffer,
                stride * sizeof(double)
            );
            if (node.output_ring) {
                size_t ring_stride = stride;
                uint32_t ring_frames = node.output_ring->shape.frames;
                if (ring_frames > 0U) {
                    uint32_t ring_index = static_cast<uint32_t>((node.output_ring_head + frame_index) % ring_frames);
                    std::memcpy(
                        node.output_ring->data() + static_cast<size_t>(ring_index) * ring_stride,
                        frame_buffer,
                        ring_stride * sizeof(double)
                    );
                }
            }
            amp_free(frame_buffer);
            state = state_arg;
        }
        if (node.output_ring && node.output_ring->shape.frames > 0U) {
            uint32_t ring_frames = node.output_ring->shape.frames;
            node.output_ring_head = static_cast<uint32_t>((node.output_ring_head + frames) % ring_frames);
        }
        for (auto &binding_entry : node.bindings) {
            ParamBinding &binding = binding_entry.second;
            if (binding.use_ring && binding.ring_frames > 0U) {
                binding.ring_head = static_cast<uint32_t>((binding.ring_head + frames) % binding.ring_frames);
                binding.dirty = true;
            }
        }
        if (node.state != state) {
            if (node.state != nullptr) {
                amp_release_state(node.state);
            }
        }
        node.state = state;
        if (!node_output) {
            runtime_set_error(runtime, -1, "node_output", &node, "node produced no output");
            return -1;
        }
        node.output = node_output;
        node.output_batches = node_output->shape.batches;
        node.output_channels = node_output->shape.channels;
        node.output_frames = node_output->shape.frames;
        auto channel_it = runtime->channels.find(node.name);
        if (channel_it != runtime->channels.end()) {
            channel_it->second->token = node_output;
        }
    }
    if (runtime->sink_index >= runtime->nodes.size()) {
        runtime_set_error(runtime, -1, "sink_lookup", nullptr, "sink index out of range");
        return -1;
    }
    RuntimeNode &sink = *runtime->nodes[runtime->sink_index];
    if (!sink.output) {
        runtime_set_error(runtime, -1, "sink_output", &sink, "sink produced no output");
        return -1;
    }
    *out_buffer = sink.output->data();
    *out_batches = sink.output_batches;
    *out_channels = sink.output_channels;
    *out_frames = sink.output_frames;
    return 0;
}

static int execute_runtime_impl(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    EdgeRunnerControlHistory *history = nullptr;
    bool history_owned = false;
    if (control_blob != nullptr && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (history == nullptr) {
            runtime_set_error(runtime, -1, "load_control_history", nullptr, "amp_load_control_history returned null");
            return -1;
        }
        history_owned = true;
    }
    return execute_runtime_with_history_impl(
        runtime,
        history,
        history_owned,
        frames_hint,
        sample_rate,
        out_buffer,
        out_batches,
        out_channels,
        out_frames
    );
}

static int execute_runtime_history_into_impl(
    AmpGraphRuntime *runtime,
    EdgeRunnerControlHistory *history,
    bool history_owned,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    double *native_buffer = nullptr;
    uint32_t batches = 0;
    uint32_t channels = 0;
    uint32_t frames = 0;
    int status = execute_runtime_with_history_impl(
        runtime,
        history,
        history_owned,
        frames_hint,
        sample_rate,
        &native_buffer,
        &batches,
        &channels,
        &frames
    );
    if (status != 0) {
        return status;
    }
    size_t required = static_cast<size_t>(batches) * static_cast<size_t>(channels) * static_cast<size_t>(frames);
    if (out_buffer == nullptr || out_buffer_len < required) {
        runtime_set_error(runtime, -1, "execute_into", nullptr, "output buffer too small");
        return -1;
    }
    std::memcpy(out_buffer, native_buffer, required * sizeof(double));
    *out_batches = batches;
    *out_channels = channels;
    *out_frames = frames;
    return 0;
}

static int execute_runtime_into_impl(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    EdgeRunnerControlHistory *history = nullptr;
    bool history_owned = false;
    if (control_blob != nullptr && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (history == nullptr) {
            runtime_set_error(runtime, -1, "load_control_history", nullptr, "amp_load_control_history returned null");
            return -1;
        }
        history_owned = true;
    }
    return execute_runtime_history_into_impl(
        runtime,
        history,
        history_owned,
        frames_hint,
        sample_rate,
        out_buffer,
        out_buffer_len,
        out_batches,
        out_channels,
        out_frames
    );
}

static size_t streamer_ring_size(const AmpGraphStreamer *streamer) {
    uint64_t w = streamer->write_index.load(std::memory_order_acquire);
    uint64_t r = streamer->read_index.load(std::memory_order_acquire);
    if (w < r) {
        return 0;
    }
    uint64_t diff = w - r;
    if (diff > streamer->ring_frames) {
        diff = streamer->ring_frames;
    }
    return static_cast<size_t>(diff);
}

static size_t streamer_ring_free(const AmpGraphStreamer *streamer) {
    size_t used = streamer_ring_size(streamer);
    if (streamer->ring_frames <= used) {
        return 0;
    }
    return streamer->ring_frames - used;
}

static void streamer_copy_from_ring(
    const AmpGraphStreamer *streamer,
    uint64_t start_index,
    size_t frames,
    double *destination
) {
    if (frames == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) {
        return;
    }
    size_t stride = streamer->frame_stride;
    size_t capacity = streamer->ring_frames;
    size_t remaining = frames;
    size_t dst_offset = 0;
    uint64_t index = start_index;
    while (remaining > 0) {
        size_t ring_pos = static_cast<size_t>(index % capacity);
        size_t contiguous = std::min(remaining, capacity - ring_pos);
        const double *src_ptr = streamer->ring_buffer.data() + static_cast<size_t>(ring_pos) * stride;
        std::memcpy(
            destination + dst_offset * stride,
            src_ptr,
            contiguous * stride * sizeof(double)
        );
        index += contiguous;
        dst_offset += contiguous;
        remaining -= contiguous;
    }
}

static void streamer_copy_to_ring(
    AmpGraphStreamer *streamer,
    uint64_t start_index,
    const double *source,
    size_t frames
) {
    if (frames == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) {
        return;
    }
    size_t stride = streamer->frame_stride;
    size_t capacity = streamer->ring_frames;
    size_t remaining = frames;
    size_t src_offset = 0;
    uint64_t index = start_index;
    while (remaining > 0) {
        size_t ring_pos = static_cast<size_t>(index % capacity);
        size_t contiguous = std::min(remaining, capacity - ring_pos);
        double *dst_ptr = streamer->ring_buffer.data() + static_cast<size_t>(ring_pos) * stride;
        std::memcpy(
            dst_ptr,
            source + src_offset * stride,
            contiguous * stride * sizeof(double)
        );
        index += contiguous;
        src_offset += contiguous;
        remaining -= contiguous;
    }
}

static void streamer_flush_ring_to_dump(AmpGraphStreamer *streamer) {
    size_t current = streamer_ring_size(streamer);
    if (current == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) {
        return;
    }
    StreamDumpChunk chunk;
    chunk.batches = streamer->batches;
    chunk.channels = streamer->channels;
    chunk.frames = static_cast<uint32_t>(current);
    uint64_t start_index = streamer->read_index.load(std::memory_order_acquire);
    chunk.sequence = start_index;
    chunk.data.resize(current * streamer->frame_stride);
    streamer_copy_from_ring(streamer, start_index, current, chunk.data.data());
    {
        std::lock_guard<std::mutex> lock(streamer->dump_mutex);
        streamer->dump_queue.push_back(std::move(chunk));
    }
    streamer->read_index.store(streamer->write_index.load(std::memory_order_acquire), std::memory_order_release);
    streamer->consumed_frames.fetch_add(current, std::memory_order_release);
    streamer->dump_cv.notify_all();
}

static void streamer_enqueue_chunk(
    AmpGraphStreamer *streamer,
    const double *frames,
    size_t frame_count,
    uint32_t batches,
    uint32_t channels,
    uint64_t sequence_start
) {
    if (frame_count == 0 || streamer->frame_stride == 0) {
        return;
    }
    StreamDumpChunk chunk;
    chunk.batches = batches;
    chunk.channels = channels;
    chunk.frames = static_cast<uint32_t>(frame_count);
    chunk.sequence = sequence_start;
    chunk.data.resize(frame_count * streamer->frame_stride);
    std::memcpy(chunk.data.data(), frames, frame_count * streamer->frame_stride * sizeof(double));
    {
        std::lock_guard<std::mutex> lock(streamer->dump_mutex);
        streamer->dump_queue.push_back(std::move(chunk));
    }
    streamer->dump_cv.notify_all();
}

static void streamer_write_frames(
    AmpGraphStreamer *streamer,
    const double *frames,
    size_t frame_count
) {
    if (frame_count == 0) {
        return;
    }
    if (streamer->ring_buffer.empty()) {
        streamer->ring_buffer.resize(static_cast<size_t>(streamer->ring_frames) * streamer->frame_stride);
        std::fill(streamer->ring_buffer.begin(), streamer->ring_buffer.end(), 0.0);
    }
    if (frame_count > streamer->ring_frames) {
        // chunk larger than ring - push directly to dump queue
        streamer_enqueue_chunk(streamer, frames, frame_count, streamer->batches, streamer->channels, streamer->produced_frames.load());
        streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
        return;
    }
    if (streamer_ring_free(streamer) < frame_count) {
        streamer_flush_ring_to_dump(streamer);
    }
    if (streamer_ring_free(streamer) < frame_count) {
        // still not enough due to very small ring - push to dump queue
        streamer_enqueue_chunk(streamer, frames, frame_count, streamer->batches, streamer->channels, streamer->produced_frames.load());
        streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
        return;
    }
    uint64_t start = streamer->write_index.load(std::memory_order_relaxed);
    streamer_copy_to_ring(streamer, start, frames, frame_count);
    streamer->write_index.store(start + frame_count, std::memory_order_release);
    streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
}

static void streamer_worker_main(AmpGraphStreamer *streamer) {
    streamer->running.store(true, std::memory_order_release);
    EdgeRunnerControlHistory *history = streamer->history;
    if (history == nullptr && !streamer->control_blob.empty()) {
        history = amp_load_control_history(
            streamer->control_blob.data(),
            streamer->control_blob.size(),
            streamer->frames_hint > 0 ? streamer->frames_hint : static_cast<int>(streamer->block_frames)
        );
        if (history == nullptr) {
            streamer->last_status = -1;
            streamer->running.store(false, std::memory_order_release);
            return;
        }
        streamer->history = history;
        streamer->history_owned = true;
    }
    double *out_ptr = nullptr;
    uint32_t out_batches = 0;
    uint32_t out_channels = 0;
    uint32_t out_frames = 0;
    while (!streamer->stop_requested.load(std::memory_order_acquire)) {
        out_ptr = nullptr;
        out_batches = out_channels = out_frames = 0;
        int status = execute_runtime_with_history_impl(
            streamer->runtime,
            history,
            false,
            static_cast<int>(streamer->block_frames),
            streamer->sample_rate,
            &out_ptr,
            &out_batches,
            &out_channels,
            &out_frames
        );
        if (status != 0) {
            streamer->last_status = status;
            break;
        }
        if (out_ptr == nullptr || out_frames == 0 || out_channels == 0 || out_batches == 0) {
            continue;
        }
        if (streamer->batches == 0) {
            streamer->batches = out_batches;
            streamer->channels = out_channels;
            streamer->frame_stride = static_cast<size_t>(out_batches) * static_cast<size_t>(out_channels);
        }
        const size_t frame_count = static_cast<size_t>(out_frames);
        std::vector<double> tmp(frame_count * streamer->frame_stride);
        std::memcpy(tmp.data(), out_ptr, tmp.size() * sizeof(double));
        streamer_write_frames(streamer, tmp.data(), frame_count);
    }
    streamer_flush_ring_to_dump(streamer);
    if (streamer->history_owned && history != nullptr) {
        amp_release_control_history(history);
        streamer->history = nullptr;
        streamer->history_owned = false;
    }
    streamer->running.store(false, std::memory_order_release);
    streamer->dump_cv.notify_all();
}

static void clear_runtime(AmpGraphRuntime *runtime) {
    if (runtime == nullptr) {
        return;
    }
    for (auto &node : runtime->nodes) {
        if (node->state != nullptr) {
            amp_release_state(node->state);
            node->state = nullptr;
        }
        node->output_ring.reset();
        node->output_ring_capacity = 0;
        node->output_ring_head = 0;
        node->param_cache.clear();
        node->param_cache_index.clear();
        node->param_cache_dirty = true;
    }
    runtime->nodes.clear();
    runtime->node_index.clear();
    runtime->execution_order.clear();
    runtime->channels.clear();
}

extern "C" {

AMP_API AmpGraphRuntime *amp_graph_runtime_create(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_create", descriptor_len, plan_len);
    auto *runtime = new (std::nothrow) AmpGraphRuntime();
    if (runtime == nullptr) {
        return nullptr;
    }
    runtime->default_batches = 1U;
    runtime->default_frames = 0U;
    runtime->dsp_sample_rate = 0.0;
    if (!parse_node_blob(runtime, descriptor_blob, descriptor_len) || !parse_plan_blob(runtime, plan_blob, plan_len)) {
        clear_runtime(runtime);
        delete runtime;
        return nullptr;
    }
    if (runtime->execution_order.empty()) {
        for (size_t i = 0; i < runtime->nodes.size(); ++i) {
            runtime->execution_order.push_back(static_cast<uint32_t>(i));
        }
    }
    if (!runtime->execution_order.empty()) {
        runtime->sink_index = runtime->execution_order.back();
    }
    return runtime;
}

AMP_API void amp_graph_runtime_destroy(AmpGraphRuntime *runtime) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_destroy", (size_t)(runtime != nullptr), 0);
    if (runtime == nullptr) {
        return;
    }
    clear_runtime(runtime);
    delete runtime;
}

AMP_API int amp_graph_runtime_configure(AmpGraphRuntime *runtime, uint32_t batches, uint32_t frames) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_configure", batches, frames);
    if (runtime == nullptr) {
        return -1;
    }
    runtime->default_batches = batches > 0U ? batches : 1U;
    runtime->default_frames = frames;
    return 0;
}

AMP_API void amp_graph_runtime_set_dsp_sample_rate(AmpGraphRuntime *runtime, double sample_rate) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_set_dsp_sample_rate", static_cast<size_t>(sample_rate), 0);
    if (runtime == nullptr) {
        return;
    }
    runtime->dsp_sample_rate = sample_rate > 0.0 ? sample_rate : 0.0;
}

AMP_API void amp_graph_runtime_clear_params(AmpGraphRuntime *runtime) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_clear_params", (size_t)(runtime != nullptr), 0);
    if (runtime == nullptr) {
        return;
    }
    for (auto &node : runtime->nodes) {
        node->bindings.clear();
        node->param_cache_dirty = true;
    }
}

AMP_API int amp_graph_runtime_set_param(
    AmpGraphRuntime *runtime,
    const char *node_name,
    const char *param_name,
    const double *data,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_set_param", (size_t)batches, (size_t)frames);
    if (runtime == nullptr) {
        return -1;
    }
    runtime_clear_error(runtime);
    if (node_name == nullptr || param_name == nullptr || data == nullptr) {
        runtime_set_error(runtime, -1, "set_param", nullptr, "invalid argument");
        return -1;
    }
    auto it = runtime->node_index.find(node_name);
    if (it == runtime->node_index.end()) {
        runtime_set_error(runtime, -1, "set_param", nullptr, std::string("unknown node: ") + node_name);
        return -1;
    }
    RuntimeNode &node = *runtime->nodes[it->second];
    ParamBinding binding{};
    binding.shape = make_shape(batches, channels, frames);
    TensorShape incoming_shape = binding.shape;
    size_t total = static_cast<size_t>(incoming_shape.batches) * static_cast<size_t>(incoming_shape.channels) * static_cast<size_t>(incoming_shape.frames);
    if (total == 0U) {
        runtime_set_error(runtime, -1, "set_param", &node, "parameter tensor has zero elements");
        return -1;
    }
    const TensorShape *expected_shape = nullptr;
    for (const DefaultParam &param : node.defaults) {
        if (param.name == param_name) {
            expected_shape = &param.shape;
            break;
        }
    }
    if (expected_shape == nullptr) {
        auto existing = node.bindings.find(param_name);
        if (existing != node.bindings.end()) {
            expected_shape = &existing->second.shape;
        }
    }
    uint32_t target_batches = expected_shape != nullptr ? expected_shape->batches : incoming_shape.batches;
    uint32_t target_channels = expected_shape != nullptr ? expected_shape->channels : incoming_shape.channels;
    uint32_t target_frames = incoming_shape.frames;
    if (expected_shape != nullptr && expected_shape->frames > 0U) {
        target_frames = expected_shape->frames;
    } else if (expected_shape == nullptr && runtime->default_frames > 0U) {
        target_frames = runtime->default_frames;
    }
    bool enable_ring = false;
    bool shape_valid = true;
    if (incoming_shape.batches != target_batches || incoming_shape.channels != target_channels) {
        shape_valid = false;
    } else if (incoming_shape.frames != target_frames) {
        if (target_frames > 0U && incoming_shape.frames % target_frames == 0U) {
            enable_ring = true;
        } else {
            shape_valid = false;
        }
    }
    if (!shape_valid) {
        AMP_LOG_NATIVE_CALL(
            "amp_graph_runtime_set_param_shape_mismatch",
            expected_shape != nullptr ? expected_shape->element_count() : 0U,
            incoming_shape.element_count()
        );
#if defined(AMP_NATIVE_ENABLE_LOGGING)
        std::fprintf(
            stderr,
            "amp_graph_runtime_set_param: shape mismatch for %s.%s (expected %u x %u x %u, got %u x %u x %u)\n",
            node_name,
            param_name,
            target_batches,
            target_channels,
            target_frames,
            incoming_shape.batches,
            incoming_shape.channels,
            incoming_shape.frames
        );
#endif
        std::string detail = std::string("shape mismatch for ") + node.name + "." + param_name
            + std::string(" (expected ") + std::to_string(target_batches) + " x "
            + std::to_string(target_channels) + " x " + std::to_string(target_frames)
            + ", got " + std::to_string(incoming_shape.batches) + " x " + std::to_string(incoming_shape.channels)
            + " x " + std::to_string(incoming_shape.frames) + ")";
        runtime_set_error(runtime, -2, "set_param", &node, std::move(detail));
        return -2;
    }
    binding.full_shape = incoming_shape;
    binding.frame_stride = static_cast<size_t>(incoming_shape.batches) * static_cast<size_t>(incoming_shape.channels);
    binding.window_frames = enable_ring ? target_frames : incoming_shape.frames;
    binding.use_ring = enable_ring;
    binding.ring_frames = enable_ring ? incoming_shape.frames : binding.window_frames;
    binding.ring_head = 0;
    binding.shape = make_shape(target_batches, target_channels, binding.window_frames);
    binding.data.resize(total);
    std::memcpy(binding.data.data(), data, total * sizeof(double));
    binding.dirty = true;
    node.output_ring_capacity = std::max(node.output_ring_capacity, binding.ring_frames);
    auto existing_binding = node.bindings.find(param_name);
    if (existing_binding != node.bindings.end()) {
        existing_binding->second = std::move(binding);
    } else {
        node.bindings.emplace(param_name, std::move(binding));
    }
    if (node.param_cache_index.find(param_name) == node.param_cache_index.end()) {
        node.param_cache_index[param_name] = node.param_cache.size();
        node.param_cache.emplace_back(param_name, std::shared_ptr<EigenTensorHolder>());
    }
    node.param_cache_dirty = true;
    return 0;
}

AMP_API int amp_graph_runtime_describe_node(
    AmpGraphRuntime *runtime,
    const char *node_name,
    AmpGraphNodeSummary *summary
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_runtime_describe_node",
        static_cast<size_t>(runtime != nullptr),
        static_cast<size_t>(summary != nullptr)
    );
    if (runtime == nullptr || node_name == nullptr || summary == nullptr) {
        return -1;
    }
    auto it = runtime->node_index.find(std::string(node_name));
    if (it == runtime->node_index.end()) {
        return -1;
    }
    RuntimeNode &node = *runtime->nodes[it->second];
    summary->declared_delay_frames = node.declared_delay_frames;
    summary->oversample_ratio = node.oversample_ratio;
    summary->supports_v2 = node.supports_v2 ? 1 : 0;
    summary->has_metrics = node.has_latest_metrics ? 1 : 0;
    AmpNodeMetrics metrics{};
    if (node.has_latest_metrics) {
        metrics = node.latest_metrics;
    }
    summary->metrics = metrics;
    summary->total_heat_accumulated = node.total_heat_accumulated;
    return 0;
}

AMP_API int amp_graph_runtime_execute(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_execute", (size_t)(control_blob != nullptr ? control_len : 0U), static_cast<size_t>(frames_hint));
    return execute_runtime_impl(runtime, control_blob, control_len, frames_hint, sample_rate, out_buffer, out_batches, out_channels, out_frames);
}

AMP_API int amp_graph_runtime_execute_with_history(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_runtime_execute_with_history",
        static_cast<size_t>(history != nullptr),
        static_cast<size_t>(frames_hint)
    );
    return execute_runtime_with_history_impl(
        runtime,
        history,
        false,
        frames_hint,
        sample_rate,
        out_buffer,
        out_batches,
        out_channels,
        out_frames
    );
}

AMP_API int amp_graph_runtime_execute_into(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_runtime_execute_into",
        (size_t)(control_blob != nullptr ? control_len : 0U),
        static_cast<size_t>(frames_hint)
    );
    return execute_runtime_into_impl(
        runtime,
        control_blob,
        control_len,
        frames_hint,
        sample_rate,
        out_buffer,
        out_buffer_len,
        out_batches,
        out_channels,
        out_frames
    );
}

AMP_API int amp_graph_runtime_execute_history_into(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_runtime_execute_history_into",
        static_cast<size_t>(history != nullptr),
        static_cast<size_t>(frames_hint)
    );
    return execute_runtime_history_into_impl(
        runtime,
        history,
        false,
        frames_hint,
        sample_rate,
        out_buffer,
        out_buffer_len,
        out_batches,
        out_channels,
        out_frames
    );
}

AMP_API int amp_graph_runtime_last_error(
    AmpGraphRuntime *runtime,
    AmpGraphRuntimeErrorInfo *out_error
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_runtime_last_error",
        static_cast<size_t>(runtime != nullptr),
        static_cast<size_t>(out_error != nullptr)
    );
    if (runtime == nullptr || out_error == nullptr) {
        return -1;
    }
    out_error->code = runtime->last_error.code;
    out_error->stage = runtime->last_error.stage.empty() ? nullptr : runtime->last_error.stage.c_str();
    out_error->node = runtime->last_error.node.empty() ? nullptr : runtime->last_error.node.c_str();
    out_error->detail = runtime->last_error.detail.empty() ? nullptr : runtime->last_error.detail.c_str();
    return 0;
}

AMP_API void amp_graph_runtime_buffer_free(double *buffer) {
    (void)buffer;
}

AMP_API AmpGraphControlHistory *amp_graph_history_load(const uint8_t *blob, size_t blob_len, int frames_hint) {
    AMP_LOG_NATIVE_CALL("amp_graph_history_load", blob_len, (size_t)frames_hint);
    return amp_load_control_history(blob, blob_len, frames_hint);
}

AMP_API void amp_graph_history_destroy(AmpGraphControlHistory *history) {
    AMP_LOG_NATIVE_CALL("amp_graph_history_destroy", (size_t)(history != nullptr), 0);
    amp_release_control_history(history);
}

AMP_API AmpGraphStreamer *amp_graph_streamer_create(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    uint32_t ring_frames,
    uint32_t block_frames
) {
    AMP_LOG_NATIVE_CALL(
        "amp_graph_streamer_create",
        static_cast<size_t>(runtime != nullptr),
        static_cast<size_t>(ring_frames)
    );
    if (runtime == nullptr) {
        return nullptr;
    }
    auto *streamer = new (std::nothrow) AmpGraphStreamer();
    if (streamer == nullptr) {
        return nullptr;
    }
    streamer->runtime = runtime;
    if (control_blob != nullptr && control_len > 0U) {
        streamer->control_blob.assign(control_blob, control_blob + control_len);
    }
    streamer->frames_hint = frames_hint;
    streamer->sample_rate = sample_rate;
    streamer->ring_frames = ring_frames > 0U ? ring_frames : block_frames;
    if (streamer->ring_frames == 0U) {
        streamer->ring_frames = 1024U;
    }
    streamer->block_frames = block_frames > 0U ? block_frames : streamer->ring_frames / 2U;
    if (streamer->block_frames == 0U) {
        streamer->block_frames = 256U;
    }
    streamer->write_index.store(0, std::memory_order_relaxed);
    streamer->read_index.store(0, std::memory_order_relaxed);
    streamer->produced_frames.store(0, std::memory_order_relaxed);
    streamer->consumed_frames.store(0, std::memory_order_relaxed);
    streamer->stop_requested.store(false, std::memory_order_relaxed);
    uint32_t batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
    amp_graph_runtime_configure(runtime, batches, streamer->block_frames);
    if (streamer->control_blob.empty()) {
        streamer->history = nullptr;
        streamer->history_owned = false;
    }
    return streamer;
}

AMP_API int amp_graph_streamer_start(AmpGraphStreamer *streamer) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_start", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr) {
        return -1;
    }
    if (streamer->running.load(std::memory_order_acquire)) {
        return 0;
    }
    streamer->stop_requested.store(false, std::memory_order_release);
    streamer->worker = std::thread(streamer_worker_main, streamer);
    return 0;
}

AMP_API void amp_graph_streamer_stop(AmpGraphStreamer *streamer) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_stop", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr) {
        return;
    }
    streamer->stop_requested.store(true, std::memory_order_release);
    if (streamer->worker.joinable()) {
        streamer->worker.join();
    }
}

AMP_API void amp_graph_streamer_destroy(AmpGraphStreamer *streamer) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_destroy", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr) {
        return;
    }
    amp_graph_streamer_stop(streamer);
    if (streamer->history_owned && streamer->history != nullptr) {
        amp_release_control_history(streamer->history);
        streamer->history = nullptr;
        streamer->history_owned = false;
    }
    delete streamer;
}

AMP_API int amp_graph_streamer_available(AmpGraphStreamer *streamer, uint64_t *out_frames) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_available", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr || out_frames == nullptr) {
        return -1;
    }
    *out_frames = streamer_ring_size(streamer);
    return 0;
}

AMP_API int amp_graph_streamer_read(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_read", (size_t)(streamer != nullptr), max_frames);
    if (streamer == nullptr || destination == nullptr || max_frames == 0) {
        return -1;
    }
    size_t available = streamer_ring_size(streamer);
    if (available == 0) {
        if (out_frames) *out_frames = 0;
        if (out_channels) *out_channels = streamer->channels;
        if (out_sequence) *out_sequence = streamer->read_index.load(std::memory_order_acquire);
        return 0;
    }
    size_t to_copy = std::min(max_frames, available);
    uint64_t start_index = streamer->read_index.load(std::memory_order_acquire);
    streamer_copy_from_ring(streamer, start_index, to_copy, destination);
    streamer->read_index.store(start_index + to_copy, std::memory_order_release);
    streamer->consumed_frames.fetch_add(to_copy, std::memory_order_release);
    if (out_frames) {
        *out_frames = static_cast<uint32_t>(to_copy);
    }
    if (out_channels) {
        *out_channels = streamer->channels;
    }
    if (out_sequence) {
        *out_sequence = start_index;
    }
    return 0;
}

AMP_API int amp_graph_streamer_dump_count(AmpGraphStreamer *streamer, uint32_t *out_count) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_dump_count", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr || out_count == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(streamer->dump_mutex);
    *out_count = static_cast<uint32_t>(streamer->dump_queue.size());
    return 0;
}

AMP_API int amp_graph_streamer_pop_dump(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_pop_dump", (size_t)(streamer != nullptr), max_frames);
    if (streamer == nullptr) {
        return -1;
    }
    StreamDumpChunk chunk;
    std::unique_lock<std::mutex> lock(streamer->dump_mutex);
    if (streamer->dump_queue.empty()) {
        if (out_frames) *out_frames = 0;
        if (out_channels) *out_channels = streamer->channels;
        if (out_sequence) *out_sequence = streamer->produced_frames.load(std::memory_order_acquire);
        return 0;
    }
    chunk = std::move(streamer->dump_queue.front());
    streamer->dump_queue.pop_front();
    lock.unlock();
    size_t frames = chunk.frames;
    if (destination == nullptr || max_frames < frames) {
        lock.lock();
        streamer->dump_queue.push_front(std::move(chunk));
        lock.unlock();
        if (out_frames) *out_frames = static_cast<uint32_t>(frames);
        if (out_channels) *out_channels = chunk.channels;
        if (out_sequence) *out_sequence = chunk.sequence;
        return 1;
    }
    std::memcpy(destination, chunk.data.data(), frames * streamer->frame_stride * sizeof(double));
    streamer->consumed_frames.fetch_add(frames, std::memory_order_release);
    if (out_frames) {
        *out_frames = static_cast<uint32_t>(frames);
    }
    if (out_channels) {
        *out_channels = chunk.channels;
    }
    if (out_sequence) {
        *out_sequence = chunk.sequence;
    }
    return 0;
}

AMP_API int amp_graph_streamer_status(
    AmpGraphStreamer *streamer,
    uint64_t *out_produced_frames,
    uint64_t *out_consumed_frames
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_status", (size_t)(streamer != nullptr), 0);
    if (streamer == nullptr) {
        return -1;
    }
    if (out_produced_frames) {
        *out_produced_frames = streamer->produced_frames.load(std::memory_order_acquire);
    }
    if (out_consumed_frames) {
        *out_consumed_frames = streamer->consumed_frames.load(std::memory_order_acquire);
    }
    return streamer->last_status;
}

}  // extern "C"
