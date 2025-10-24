#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
};

struct RuntimeNode;

struct KahnChannel {
    std::string name;
    std::shared_ptr<EigenTensorHolder> token;

    explicit KahnChannel(std::string name_) : name(std::move(name_)) {}
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
    uint32_t channel_hint{1};
    void *state{nullptr};
    EdgeRunnerNodeDescriptor descriptor{};

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

struct AmpGraphRuntime {
    std::vector<std::unique_ptr<RuntimeNode>> nodes;
    std::unordered_map<std::string, size_t> node_index;
    std::vector<uint32_t> execution_order;
    std::unordered_map<std::string, std::shared_ptr<KahnChannel>> channels;
    uint32_t sink_index{0};
    uint32_t default_batches{1};
    uint32_t default_frames{0};
    double dsp_sample_rate{0.0};
};

static TensorShape make_shape(uint32_t batches, uint32_t channels, uint32_t frames) {
    TensorShape shape{};
    shape.batches = std::max<uint32_t>(1U, batches);
    shape.channels = std::max<uint32_t>(1U, channels);
    shape.frames = std::max<uint32_t>(1U, frames);
    return shape;
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
            node->defaults.push_back(std::move(param));
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
        if (!reader.read_u32(function_id) || !reader.read_u32(name_len) || !reader.read_u32(audio_offset) ||
            !reader.read_u32(audio_span) || !reader.read_u32(param_count)) {
            return false;
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

static std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> build_param_tensors(
    RuntimeNode &node
) {
    std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> tensors;
    tensors.reserve(node.defaults.size() + node.bindings.size());
    for (const DefaultParam &param : node.defaults) {
        tensors.emplace_back(param.name, clone_param_tensor(param));
    }
    for (const auto &entry : node.bindings) {
        bool replaced = false;
        for (auto &existing : tensors) {
            if (existing.first == entry.first) {
                existing.second = clone_binding_tensor(entry.second);
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            tensors.emplace_back(entry.first, clone_binding_tensor(entry.second));
        }
    }
    return tensors;
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

static int execute_runtime(
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
    if (runtime == nullptr || out_buffer == nullptr || out_batches == nullptr || out_channels == nullptr || out_frames == nullptr) {
        return -1;
    }
    EdgeRunnerControlHistory *history = nullptr;
    if (control_blob != nullptr && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (history == nullptr) {
            return -1;
        }
    }
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
            if (history != nullptr) {
                amp_release_control_history(history);
            }
            return -1;
        }
        RuntimeNode &node = *runtime->nodes[order];
        scratch.clear();
        std::shared_ptr<EigenTensorHolder> audio_tensor = merge_audio_inputs(runtime, node, scratch);
        if (!audio_tensor && !node.audio_indices.empty()) {
            if (history != nullptr) {
                amp_release_control_history(history);
            }
            return -1;
        }
        uint32_t batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
        uint32_t frames = runtime->default_frames > 0U ? runtime->default_frames : 1U;
        uint32_t input_channels = node.channel_hint > 0U ? node.channel_hint : 1U;
        if (audio_tensor) {
            batches = audio_tensor->shape.batches;
            frames = audio_tensor->shape.frames;
            input_channels = audio_tensor->shape.channels;
        }
        auto param_tensors = build_param_tensors(node);
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
                history
            );
            if (status != 0 || frame_buffer == nullptr) {
                if (frame_buffer != nullptr) {
                    amp_free(frame_buffer);
                }
                if (history != nullptr) {
                    amp_release_control_history(history);
                }
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
            size_t stride = static_cast<size_t>(node_output->shape.batches) * static_cast<size_t>(node_output->shape.channels);
            std::memcpy(
                node_output->data() + frame_index * stride,
                frame_buffer,
                stride * sizeof(double)
            );
            amp_free(frame_buffer);
            state = state_arg;
        }
        if (node.state != state) {
            if (node.state != nullptr) {
                amp_release_state(node.state);
            }
        }
        node.state = state;
        if (!node_output) {
            if (history != nullptr) {
                amp_release_control_history(history);
            }
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
    if (history != nullptr) {
        amp_release_control_history(history);
    }
    if (runtime->sink_index >= runtime->nodes.size()) {
        return -1;
    }
    RuntimeNode &sink = *runtime->nodes[runtime->sink_index];
    if (!sink.output) {
        return -1;
    }
    *out_buffer = sink.output->data();
    *out_batches = sink.output_batches;
    *out_channels = sink.output_channels;
    *out_frames = sink.output_frames;
    return 0;
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
    if (runtime == nullptr || node_name == nullptr || param_name == nullptr || data == nullptr) {
        return -1;
    }
    auto it = runtime->node_index.find(node_name);
    if (it == runtime->node_index.end()) {
        return -1;
    }
    RuntimeNode &node = *runtime->nodes[it->second];
    ParamBinding binding{};
    binding.shape = make_shape(batches, channels, frames);
    size_t total = static_cast<size_t>(binding.shape.batches) * static_cast<size_t>(binding.shape.channels) * static_cast<size_t>(binding.shape.frames);
    if (total == 0U) {
        return -1;
    }
    binding.data.resize(total);
    std::memcpy(binding.data.data(), data, total * sizeof(double));
    node.bindings[param_name] = std::move(binding);
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
    return execute_runtime(runtime, control_blob, control_len, frames_hint, sample_rate, out_buffer, out_batches, out_channels, out_frames);
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

}  // extern "C"
