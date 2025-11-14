#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <memory>
#include <new>
#include <queue>
#include <stdexcept>
#include <string>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <thread>
#include <deque>
#include <condition_variable>
#include <mutex>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

extern "C" {
#include "amp_native.h"
#include "mailbox.h"
#include "amp_debug_alloc.h"
}

#if defined(_WIN32) || defined(_WIN64)
#  define AMP_API __declspec(dllexport)
#else
#  define AMP_API
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

static uint64_t steady_now_millis() {
    using namespace std::chrono;
    return static_cast<uint64_t>(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

static uint64_t elapsed_since(uint64_t now, uint64_t then) {
    if (then == 0 || now <= then) {
        return 0;
    }
    return now - then;
}

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
    double *external_data{nullptr};
    size_t external_count{0};

    explicit EigenTensorHolder(const TensorShape &shape_)
        : shape(shape_),
          values(
              static_cast<long>(std::max<uint32_t>(1U, shape_.batches)),
              static_cast<long>(std::max<uint32_t>(1U, shape_.channels)),
              static_cast<long>(std::max<uint32_t>(1U, shape_.frames))
          ) {
        values.setZero();
        external_count = static_cast<size_t>(values.size());
    }

    double *data() {
        return external_data != nullptr ? external_data : values.data();
    }

    const double *data() const {
        return external_data != nullptr ? external_data : values.data();
    }

    void set_zero() {
        size_t count = shape.element_count();
        if (external_data != nullptr && count > 0U) {
            std::fill(external_data, external_data + count, 0.0);
        } else {
            values.setZero();
        }
    }

    void map_external(double *ptr, size_t count) {
        external_data = ptr;
        external_count = count;
    }

    void clear_external() {
        external_data = nullptr;
        external_count = 0;
    }

    bool has_external() const {
        return external_data != nullptr;
    }

    size_t storage_size() const {
        return shape.element_count();
    }
};

struct SchedulerParams {
    double early_bias{0.5};
    double late_bias{0.5};
    double saturation_bias{1.0};
};

struct VectorizationPolicy {
    uint32_t channel_expand{1};
    uint32_t block_frames{0}; // preferred frames (legacy)
    uint32_t min_block_frames{0};
    uint32_t max_block_frames{0};
    double priority_weight{1.0};
    bool archtypal_mode{false};
};

struct SpectralWorkingSpaceConfig {
    uint32_t duration_frames{0};
    uint32_t frequency_bins{0};
    std::string time_projection_default{"buffered_fill"};
    std::string freq_projection_default{"identity"};
};

struct LaneProjectionConfig {
    std::string input_name;
    std::string time_policy{"buffered_fill"};
    std::string freq_policy{"identity"};
    std::string phase_policy{"preserve"};
};

enum class EdgeRingReleasePolicy : uint8_t {
    AllConsumers = 0,
    PrimaryConsumer = 1
};

struct EdgeRingContract {
    bool simultaneous_availability{true};
    EdgeRingReleasePolicy release_policy{EdgeRingReleasePolicy::AllConsumers};
    uint32_t primary_consumer{std::numeric_limits<uint32_t>::max()};
    double sample_rate_hz{0.0};
    bool sample_rate_free{false};
    double sample_rate_ema_alpha{0.0};
    uint32_t sample_rate_window{0U};
};

struct EdgeRing {
    std::shared_ptr<EigenTensorHolder> storage;
    uint32_t capacity{0};
    uint32_t head{0};
    uint32_t tail{0};
    VectorizationPolicy policy{};
    size_t frame_stride{0};
    std::unordered_map<uint32_t, uint32_t> reader_tails;
    bool constant{false};
    EdgeRingContract contract{};
    uint64_t produced_total{0};
    double nominal_sample_rate{0.0};
    double effective_sample_rate{0.0};
    bool sample_rate_free{false};
    double sample_rate_ema_alpha{0.0};
    uint32_t sample_rate_window{0U};
};

struct EdgeReader {
    std::shared_ptr<EdgeRing> ring;
    uint32_t consumer_index{0};
    std::string tap_name;
    std::string input_name;
    uint32_t producer_node_index{std::numeric_limits<uint32_t>::max()};
    uint32_t producer_output_index{std::numeric_limits<uint32_t>::max()};
    bool hold_if_absent{false};
    bool has_optional_default{false};
    double optional_default_value{0.0};
};

struct TapChannelSemantic {
    std::string role;
    uint32_t components_per_frame{1U};
    bool imag_zero{false};
};

struct OutputTap {
    std::string name;
    std::shared_ptr<EdgeRing> ring;
    EdgeRingContract contract{};
    bool primary{false};
    std::string buffer_class;
    TensorShape declared_shape{};
    bool expose_in_context{false};
    std::vector<TapChannelSemantic> channel_semantics;
};

struct TapOutputBuffer {
    OutputTap *tap{nullptr};
    std::vector<double> scratch;
};

struct ModConnectionInfo {
    std::string source;
    std::string param;
    double scale{1.0};
    int mode{0};     // 0 = add, 1 = mul
    int channel{-1}; // -1 = follow, else fixed
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
    std::shared_ptr<EigenTensorHolder> ring;
    uint32_t ring_frames{0};
    uint32_t block_start{0};
    uint32_t block_frames{0};
    size_t frame_stride{0};

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
    struct DebugFrameCache {
        uint64_t sequence{0};
        uint64_t sample_count{0};
        uint64_t total_frames{0};
        uint64_t total_batches{0};
        uint64_t total_channels{0};
        double sum_processing_seconds{0.0};
        double sum_logging_seconds{0.0};
        double sum_total_seconds{0.0};
        double sum_thread_cpu_seconds{0.0};
    uint64_t metrics_samples{0};
        uint32_t last_frames{0};
        uint32_t last_batches{0};
        uint32_t last_channels{0};
        uint64_t last_timestamp_millis{0};
        AmpNodeMetrics last_metrics{};
    } debug_frame_cache;
    std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> param_cache;
    std::unordered_map<std::string, size_t> param_cache_index;
    bool param_cache_dirty{true};
    VectorizationPolicy vector_policy{};
    SpectralWorkingSpaceConfig spectral_working_space{};
    std::vector<LaneProjectionConfig> lane_projections;
    EdgeRingContract output_contract{};
    std::vector<OutputTap> outputs;
    std::vector<EdgeReader> input_edges;
    bool prefill_only{false};
    uint32_t prefill_frames{0};
    bool constant_node{false};
    std::vector<double> audio_workspace;
    TensorShape audio_workspace_shape{};
    bool prepass_done{false};
    struct InputHoldState {
        std::vector<double> values;
        uint32_t batches{0};
        uint32_t channels{0};
        bool valid{false};
    };
    std::vector<InputHoldState> input_hold_cache;
    bool expose_tap_context{false};

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

static bool metadata_key_exists(const std::string &json, const char *key);
static bool parse_bool_metadata(const std::string &json, const char *key, bool fallback);
static double parse_double_metadata(const std::string &json, const char *key, double fallback);
static std::string sanitize_metadata_key_component(const std::string &value);
static void configure_tap_channel_semantics(const RuntimeNode &node, OutputTap &tap);

static void configure_input_reader_metadata(RuntimeNode &node, EdgeReader &reader, size_t slot) {
    if (slot < node.audio_inputs.size()) {
        reader.input_name = node.audio_inputs[slot];
    } else {
        reader.input_name.clear();
    }
    if (reader.input_name.empty()) {
        reader.hold_if_absent = false;
        reader.has_optional_default = false;
        reader.optional_default_value = 0.0;
        return;
    }
    std::string sanitized = sanitize_metadata_key_component(reader.input_name);
    if (sanitized.empty()) {
        sanitized = "input";
    }
    std::string hold_key = "input_" + sanitized + "_hold_if_absent";
    reader.hold_if_absent = parse_bool_metadata(node.params_json, hold_key.c_str(), false);
    std::string default_key = "input_" + sanitized + "_optional_default";
    if (metadata_key_exists(node.params_json, default_key.c_str())) {
        reader.optional_default_value = parse_double_metadata(node.params_json, default_key.c_str(), 0.0);
        reader.has_optional_default = true;
    } else {
        reader.optional_default_value = 0.0;
        reader.has_optional_default = false;
    }
}

static OutputTap *runtime_primary_output(RuntimeNode &node) {
    for (auto &tap : node.outputs) {
        if (tap.primary) {
            return &tap;
        }
    }
    if (!node.outputs.empty()) {
        return &node.outputs.front();
    }
    return nullptr;
}

static const OutputTap *runtime_primary_output(const RuntimeNode &node) {
    for (const auto &tap : node.outputs) {
        if (tap.primary) {
            return &tap;
        }
    }
    if (!node.outputs.empty()) {
        return &node.outputs.front();
    }
    return nullptr;
}

struct BlockFrameContract {
    uint32_t min_frames{1U};
    uint32_t preferred_frames{1U};
    uint32_t max_frames{1U};
    double priority_weight{1.0};
};

static BlockFrameContract node_block_frame_contract(
    const AmpGraphRuntime *runtime,
    const RuntimeNode &node,
    uint32_t default_frames
);

static void runtime_node_record_debug_frame(RuntimeNode &node, uint32_t batches, uint32_t channels, uint32_t frames) {
    if (frames == 0U) {
        return;
    }

    RuntimeNode::DebugFrameCache &cache = node.debug_frame_cache;
    cache.sequence += 1ULL;
    cache.sample_count += 1ULL;
    cache.total_frames += static_cast<uint64_t>(frames);
    cache.total_batches += static_cast<uint64_t>(std::max<uint32_t>(1U, batches));
    cache.total_channels += static_cast<uint64_t>(std::max<uint32_t>(1U, channels));
    cache.last_frames = frames;
    cache.last_batches = std::max<uint32_t>(1U, batches);
    cache.last_channels = std::max<uint32_t>(1U, channels);
    cache.last_timestamp_millis = steady_now_millis();

}

struct AmpGraphRuntime {
    std::vector<std::unique_ptr<RuntimeNode>> nodes;
    std::unordered_map<std::string, size_t> node_index;
    std::vector<uint32_t> execution_order;
    std::unordered_map<std::string, std::shared_ptr<KahnChannel>> channels;
    std::unordered_map<std::string, std::shared_ptr<EdgeRing>> edge_rings;
    std::vector<std::vector<uint32_t>> dependents;
    std::vector<uint32_t> indegree;
    std::vector<uint32_t> execution_rank;
    SchedulerParams scheduler_params;
    AmpGraphSchedulerMode scheduler_mode{AMP_SCHEDULER_LEARNED};
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
    AmpGraphControlHistory *history{nullptr}; // likely typedef of EdgeRunnerControlHistory
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
    std::atomic<uint64_t> start_time_millis{0};
    std::atomic<uint64_t> last_produced_millis{0};
    std::atomic<uint64_t> last_consumed_millis{0};
    std::atomic<uint64_t> last_dump_millis{0};
    uint64_t expected_output_frames{0};
};

static BlockFrameContract node_block_frame_contract(
    const AmpGraphRuntime *runtime,
    const RuntimeNode &node,
    uint32_t default_frames
) {
    // Derive vector contract bounds using metadata hints and runtime defaults.
    BlockFrameContract contract{};

    double weight = node.vector_policy.priority_weight;
    contract.priority_weight = weight > 0.0 ? weight : 1.0;

    uint32_t hint = default_frames;
    if (hint == 0U && runtime != nullptr) {
        hint = runtime->default_frames;
    }
    if (hint == 0U) {
        hint = node.vector_policy.block_frames;
    }
    if (hint == 0U && node.prefill_frames > 0U) {
        hint = node.prefill_frames;
    }
    if (hint == 0U && node.output_frames > 0U) {
        hint = node.output_frames;
    }
    if (hint == 0U) {
        hint = 1U;
    }

    uint32_t min_frames = node.vector_policy.min_block_frames;
    if (min_frames == 0U) {
        if (node.prefill_only && node.prefill_frames > 0U) {
            min_frames = node.prefill_frames;
        } else {
            min_frames = 1U;
        }
    }
    if (min_frames == 0U) {
        min_frames = 1U;
    }

    uint32_t preferred_frames = node.vector_policy.block_frames;
    if (preferred_frames == 0U) {
        preferred_frames = hint;
    }
    if (preferred_frames == 0U) {
        preferred_frames = min_frames;
    }
    if (node.prefill_frames > 0U) {
        preferred_frames = std::max(preferred_frames, node.prefill_frames);
    }

    uint32_t max_frames = node.vector_policy.max_block_frames;
    if (max_frames == 0U) {
        max_frames = preferred_frames;
        if (hint > 0U) {
            max_frames = std::max(max_frames, hint);
        }
        if (node.prefill_frames > 0U) {
            max_frames = std::max(max_frames, node.prefill_frames);
        }
        if (node.output_ring_capacity > 0U) {
            max_frames = std::max(max_frames, node.output_ring_capacity);
        }
    }

    if (preferred_frames < min_frames) {
        preferred_frames = min_frames;
    }
    if (max_frames < preferred_frames) {
        max_frames = preferred_frames;
    }

    const OutputTap *primary = runtime_primary_output(node);
    if (primary != nullptr && primary->ring != nullptr && runtime != nullptr && runtime->dsp_sample_rate > 0.0) {
        const EdgeRing &ring = *primary->ring;
        double target_rate = ring.effective_sample_rate > 0.0 ? ring.effective_sample_rate : ring.nominal_sample_rate;
        if (target_rate <= 0.0) {
            target_rate = ring.contract.sample_rate_hz;
        }
        if (target_rate > 0.0) {
            double frames_per_update = runtime->dsp_sample_rate / target_rate;
            if (frames_per_update < 1.0) {
                frames_per_update = 1.0;
            }
            uint32_t adjusted = static_cast<uint32_t>(std::max<double>(1.0, std::round(frames_per_update)));
            if (adjusted == 0U) {
                adjusted = 1U;
            }
            if (ring.sample_rate_free || ring.nominal_sample_rate <= 0.0) {
                min_frames = std::max(min_frames, adjusted);
                preferred_frames = std::max(preferred_frames, adjusted);
                max_frames = std::max(max_frames, adjusted);
            } else {
                min_frames = std::max(min_frames, adjusted);
                preferred_frames = std::max(preferred_frames, adjusted);
                max_frames = std::max(max_frames, adjusted);
            }
        }
    }

    contract.min_frames = min_frames;
    contract.preferred_frames = preferred_frames;
    contract.max_frames = max_frames;
    return contract;
}

/*** Forward declarations for functions used before definition ***/
static std::shared_ptr<EdgeRing> ensure_edge_ring(
    AmpGraphRuntime *runtime,
    const std::string &key,
    const VectorizationPolicy &policy
);
static void edge_ring_register_consumer(EdgeRing &ring, uint32_t consumer);
static void runtime_update_scheduler_topology(AmpGraphRuntime *runtime);
static std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> &build_param_tensors(RuntimeNode &node);
static std::shared_ptr<EigenTensorHolder> ensure_param_tensor(
    std::vector<std::pair<std::string, std::shared_ptr<EigenTensorHolder>>> &tensors,
    const std::string &name,
    const TensorShape &fallback_shape
);
static void streamer_write_frames(AmpGraphStreamer *streamer, const double *frames, size_t frame_count);

/*** Descriptor reader ***/
struct DescriptorReader {
    const uint8_t *cursor;
    size_t remaining;

    DescriptorReader(const uint8_t *data, size_t length) : cursor(data), remaining(length) {}

    bool read_u32(uint32_t &out) {
        if (remaining < sizeof(uint32_t)) return false;
        uint32_t value = 0;
        std::memcpy(&value, cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        remaining -= sizeof(uint32_t);
        out = value;
        return true;
    }

    bool read_u64(uint64_t &out) {
        if (remaining < sizeof(uint64_t)) return false;
        uint64_t value = 0;
        std::memcpy(&value, cursor, sizeof(uint64_t));
        cursor += sizeof(uint64_t);
        remaining -= sizeof(uint64_t);
        out = value;
        return true;
    }

    bool read_float(float &out) {
        if (remaining < sizeof(float)) return false;
        float value = 0.0f;
        std::memcpy(&value, cursor, sizeof(float));
        cursor += sizeof(float);
        remaining -= sizeof(float);
        out = value;
        return true;
    }

    bool read_string(size_t length, std::string &out) {
        if (remaining < length) return false;
        out.assign(reinterpret_cast<const char *>(cursor), length);
        cursor += length;
        remaining -= length;
        return true;
    }

    bool skip(size_t length) {
        if (remaining < length) return false;
        cursor += length;
        remaining -= length;
        return true;
    }
};

/*** Small JSON scanners for metadata ***/
static uint32_t parse_uint_metadata(const std::string &json, const char *key, uint32_t fallback) {
    if (key == nullptr) return fallback;
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) ++pos;
    if (pos >= json.size()) return fallback;
    if (json[pos] == '-') return fallback;
    uint64_t value = 0;
    bool found_digit = false;
    while (pos < json.size()) {
        unsigned char ch = static_cast<unsigned char>(json[pos]);
        if (!std::isdigit(ch)) break;
        found_digit = true;
        value = value * 10ULL + static_cast<uint64_t>(ch - static_cast<unsigned char>('0'));
        if (value > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return fallback;
        ++pos;
    }
    if (!found_digit) return fallback;
    return static_cast<uint32_t>(value);
}

static bool parse_bool_metadata(const std::string &json, const char *key, bool fallback) {
    if (key == nullptr) return fallback;
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) ++pos;
    if (pos + 4U <= json.size() && json.compare(pos, 4U, "true") == 0) return true;
    if (pos + 5U <= json.size() && json.compare(pos, 5U, "false") == 0) return false;
    return fallback;
}

static double parse_double_metadata(const std::string &json, const char *key, double fallback) {
    if (key == nullptr) return fallback;
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.push_back('"');
    needle.push_back(':');
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) ++pos;
    if (pos >= json.size()) return fallback;
    const char *start = json.c_str() + pos;
    char *endptr = nullptr;
    double value = std::strtod(start, &endptr);
    if (start == endptr) return fallback;
    return value;
}

static std::string parse_string_metadata(const std::string &json, const char *key, const std::string &fallback = {}) {
    if (key == nullptr) return fallback;
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])) != 0) ++pos;
    if (pos >= json.size() || json[pos] != '"') return fallback;
    ++pos;
    size_t start = pos;
    while (pos < json.size()) {
        unsigned char ch = static_cast<unsigned char>(json[pos]);
        if (ch == '\\') {
            pos += 2U;
            continue;
        }
        if (ch == '"') break;
        ++pos;
    }
    if (pos >= json.size()) return fallback;
    return json.substr(start, pos - start);
}

static bool metadata_key_exists(const std::string &json, const char *key) {
    if (key == nullptr) return false;
    std::string needle;
    needle.reserve(std::strlen(key) + 4U);
    needle.push_back('"');
    needle.append(key);
    needle.append("\":");
    return json.find(needle) != std::string::npos;
}

static std::string sanitize_metadata_key_component(const std::string &value) {
    if (value.empty()) {
        return std::string("input");
    }
    std::string sanitized;
    sanitized.reserve(value.size());
    for (char ch : value) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) != 0) {
            sanitized.push_back(static_cast<char>(std::tolower(uch)));
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized;
}

static std::string trim_metadata_value(const std::string &value) {
    size_t start = 0U;
    size_t end = value.size();
    while (start < end && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1U])) != 0) {
        --end;
    }
    if (start >= end) {
        return std::string();
    }
    return value.substr(start, end - start);
}

static std::vector<std::string> split_metadata_csv(const std::string &csv) {
    std::vector<std::string> tokens;
    if (csv.empty()) {
        return tokens;
    }
    size_t start = 0U;
    while (start < csv.size()) {
        size_t end = csv.find(',', start);
        if (end == std::string::npos) {
            end = csv.size();
        }
        std::string token = trim_metadata_value(csv.substr(start, end - start));
        if (!token.empty()) {
            tokens.push_back(std::move(token));
        }
        if (end == csv.size()) {
            break;
        }
        start = end + 1U;
    }
    return tokens;
}

static std::vector<uint32_t> parse_uint_list_from_csv(const std::string &csv) {
    std::vector<uint32_t> values;
    auto tokens = split_metadata_csv(csv);
    values.reserve(tokens.size());
    for (const std::string &token : tokens) {
        const char *begin = token.c_str();
        char *endptr = nullptr;
        unsigned long parsed = std::strtoul(begin, &endptr, 10);
        if (begin != endptr) {
            if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
                parsed = static_cast<unsigned long>(std::numeric_limits<uint32_t>::max());
            }
            values.push_back(static_cast<uint32_t>(parsed));
        } else {
            values.push_back(0U);
        }
    }
    return values;
}

static std::vector<bool> parse_bool_list_from_csv(const std::string &csv) {
    std::vector<bool> values;
    auto tokens = split_metadata_csv(csv);
    values.reserve(tokens.size());
    for (const std::string &token : tokens) {
        bool flag = false;
        if (!token.empty()) {
            std::string lowered;
            lowered.reserve(token.size());
            for (char ch : token) {
                lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
                flag = true;
            }
        }
        values.push_back(flag);
    }
    return values;
}

static void configure_tap_channel_semantics(const RuntimeNode &node, OutputTap &tap) {
    tap.channel_semantics.clear();
    std::string sanitized = sanitize_metadata_key_component(tap.name);
    if (sanitized.empty()) {
        sanitized = "tap";
    }
    std::string prefix = "tap_" + sanitized + "_";
    std::string roles_key = prefix + "channel_roles";
    std::vector<std::string> roles = split_metadata_csv(
        parse_string_metadata(node.params_json, roles_key.c_str(), "")
    );
    if (roles.empty() && node.type_name == "FFTDivisionNode") {
        roles.emplace_back("pcm");
        roles.emplace_back("spectrum");
    }
    if (roles.empty()) {
        return;
    }
    std::string components_key = prefix + "channel_components";
    std::vector<uint32_t> component_counts = parse_uint_list_from_csv(
        parse_string_metadata(node.params_json, components_key.c_str(), "")
    );
    std::string imag_zero_key = prefix + "channel_imag_zero";
    std::vector<bool> imag_zero_flags = parse_bool_list_from_csv(
        parse_string_metadata(node.params_json, imag_zero_key.c_str(), "")
    );

    for (size_t idx = 0U; idx < roles.size(); ++idx) {
        TapChannelSemantic semantic{};
        semantic.role = roles[idx];
        uint32_t default_components = (semantic.role == "pcm" || semantic.role == "spectrum") ? 2U : 1U;
        if (idx < component_counts.size() && component_counts[idx] > 0U) {
            semantic.components_per_frame = component_counts[idx];
        } else {
            semantic.components_per_frame = default_components;
        }
        if (idx < imag_zero_flags.size()) {
            semantic.imag_zero = imag_zero_flags[idx];
        } else {
            semantic.imag_zero = (semantic.role == "pcm");
        }
        tap.channel_semantics.push_back(std::move(semantic));
    }
}

/*** Error helpers ***/
static void runtime_clear_error(AmpGraphRuntime *runtime) {
    if (!runtime) return;
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
    if (!runtime) return;
    runtime->last_error.code = code;
    runtime->last_error.stage = stage != nullptr ? stage : "";
    runtime->last_error.node = node != nullptr ? node->name : "";
    runtime->last_error.detail = std::move(detail);
}

/*** Shape / math helpers ***/
static TensorShape make_shape(uint32_t batches, uint32_t channels, uint32_t frames) {
    TensorShape shape{};
    shape.batches = std::max<uint32_t>(1U, batches);
    shape.channels = std::max<uint32_t>(1U, channels);
    shape.frames = std::max<uint32_t>(1U, frames);
    return shape;
}

static uint32_t align_frames_up(uint32_t value, uint32_t block) {
    if (block == 0U) return value;
    if (value == 0U) return block;
    uint32_t remainder = value % block;
    if (remainder == 0U) return value;
    uint64_t aligned = static_cast<uint64_t>(value) + static_cast<uint64_t>(block - remainder);
    if (aligned > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
        return std::numeric_limits<uint32_t>::max();
    return static_cast<uint32_t>(aligned);
}

static uint32_t align_frames_down(uint32_t value, uint32_t block) {
    if (block == 0U || value == 0U) return value;
    return (value / block) * block;
}

static bool shapes_equal(const TensorShape &lhs, const TensorShape &rhs) {
    return lhs.batches == rhs.batches && lhs.channels == rhs.channels && lhs.frames == rhs.frames;
}

static std::shared_ptr<EigenTensorHolder> make_tensor(const TensorShape &shape) {
    auto tensor = std::make_shared<EigenTensorHolder>(shape);
    return tensor;
}

/*** Edge ring helpers ***/
static constexpr uint32_t EDGE_RING_HOST_CONSUMER = std::numeric_limits<uint32_t>::max();

static size_t edge_ring_frame_stride(const EdgeRing &ring) {
    if (ring.frame_stride > 0U) return ring.frame_stride;
    if (!ring.storage) return 0U;
    return static_cast<size_t>(ring.storage->shape.batches) * static_cast<size_t>(ring.storage->shape.channels);
}

static uint32_t edge_ring_distance(uint32_t head, uint32_t tail, uint32_t capacity) {
    if (capacity == 0U) return 0U;
    if (head >= tail) return head - tail;
    return capacity - (tail - head);
}

static void edge_ring_recompute_tail(EdgeRing &ring) {
    if (ring.capacity == 0U) {
        ring.tail = 0U;
        return;
    }
    uint32_t new_tail = ring.head % ring.capacity;
    if (ring.contract.release_policy == EdgeRingReleasePolicy::PrimaryConsumer) {
        auto it = ring.reader_tails.find(ring.contract.primary_consumer);
        if (it != ring.reader_tails.end()) {
            ring.tail = it->second % ring.capacity;
            return;
        }
    }
    if (!ring.reader_tails.empty()) {
        for (const auto &entry : ring.reader_tails) {
            uint32_t pos = entry.second % ring.capacity;
            if (pos < new_tail) {
                new_tail = pos;
            }
        }
    }
    ring.tail = new_tail;
}

static void edge_ring_register_consumer(EdgeRing &ring, uint32_t consumer) {
    ring.reader_tails[consumer] = ring.tail;
    if (ring.contract.release_policy == EdgeRingReleasePolicy::PrimaryConsumer &&
        ring.contract.primary_consumer == std::numeric_limits<uint32_t>::max()) {
        ring.contract.primary_consumer = consumer;
    }
}

static void edge_ring_set_consumer_position(EdgeRing &ring, uint32_t consumer, uint32_t position) {
    if (ring.capacity == 0U) {
        ring.reader_tails[consumer] = 0U;
        ring.tail = 0U;
        return;
    }
    ring.reader_tails[consumer] = position % ring.capacity;
    edge_ring_recompute_tail(ring);
}

static uint32_t edge_ring_consumer_tail(const EdgeRing &ring, uint32_t consumer) {
    auto it = ring.reader_tails.find(consumer);
    if (it == ring.reader_tails.end()) {
        return ring.tail;
    }
    return it->second % (ring.capacity == 0U ? 1U : ring.capacity);
}

static uint32_t edge_ring_available_for_consumer(const EdgeRing &ring, uint32_t consumer) {
    if (ring.capacity == 0U) return 0U;
    uint32_t tail = edge_ring_consumer_tail(ring, consumer);
    return edge_ring_distance(ring.head % ring.capacity, tail, ring.capacity);
}

static uint32_t edge_ring_free(const EdgeRing &ring) {
    if (ring.capacity == 0U) return 0U;
    uint32_t used = edge_ring_distance(ring.head % ring.capacity, ring.tail % ring.capacity, ring.capacity);
    if (used >= ring.capacity - 1U) return 0U;
    return (ring.capacity - 1U) - used;
}

static void edge_ring_copy_segment(const EdgeRing &ring, uint32_t start, uint32_t frames, double *buffer, bool to_buffer) {
    if (!ring.storage || ring.capacity == 0U || frames == 0U) return;
    size_t stride = edge_ring_frame_stride(ring);
    if (stride == 0U) return;
    uint32_t begin = start % ring.capacity;
    uint32_t first_span = std::min(frames, ring.capacity - begin);
    double *base = ring.storage->data();
    size_t primary_bytes = static_cast<size_t>(first_span) * stride * sizeof(double);
    if (to_buffer) {
        std::memcpy(buffer, base + static_cast<size_t>(begin) * stride, primary_bytes);
    } else {
        std::memcpy(base + static_cast<size_t>(begin) * stride, buffer, primary_bytes);
    }
    if (frames > first_span) {
        size_t secondary_bytes = static_cast<size_t>(frames - first_span) * stride * sizeof(double);
        if (to_buffer) {
            std::memcpy(buffer + static_cast<size_t>(first_span) * stride, base, secondary_bytes);
        } else {
            std::memcpy(base, buffer + static_cast<size_t>(first_span) * stride, secondary_bytes);
        }
    }
}

static void edge_ring_copy_out(const EdgeRing &ring, uint32_t start, uint32_t frames, double *dst) {
    edge_ring_copy_segment(ring, start, frames, dst, true);
}

static void edge_ring_copy_in(EdgeRing &ring, uint32_t start, const double *src, uint32_t frames) {
    edge_ring_copy_segment(ring, start, frames, const_cast<double *>(src), false);
}

static void edge_ring_write(EdgeRing &ring, const double *src, uint32_t frames) {
    if (frames == 0U || ring.capacity == 0U) return;
    edge_ring_copy_segment(ring, ring.head, frames, const_cast<double *>(src), false);
    ring.head = (ring.head + frames) % ring.capacity;
    ring.produced_total += frames;
}

static void edge_ring_advance_consumer(EdgeRing &ring, uint32_t consumer, uint32_t frames) {
    if (ring.capacity == 0U) return;
    uint32_t tail = edge_ring_consumer_tail(ring, consumer);
    tail = (tail + frames) % ring.capacity;
    edge_ring_set_consumer_position(ring, consumer, tail);
}

/*** Node/ring creation ***/
static std::shared_ptr<EdgeRing> ensure_edge_ring(
    AmpGraphRuntime *runtime,
    const std::string &key,
    const VectorizationPolicy &policy
) {
    if (!runtime) return nullptr;
    auto it = runtime->edge_rings.find(key);
    if (it != runtime->edge_rings.end()) return it->second;
    auto ring = std::make_shared<EdgeRing>();
    ring->policy = policy;
    ring->contract.primary_consumer = EDGE_RING_HOST_CONSUMER;
    ring->nominal_sample_rate = 0.0;
    ring->effective_sample_rate = 0.0;
    ring->sample_rate_free = false;
    ring->sample_rate_ema_alpha = 0.0;
    ring->sample_rate_window = 0U;
    runtime->edge_rings.emplace(key, ring);
    return ring;
}

/*** Topology-dependent init of edge rings ***/
static void runtime_initialize_edge_rings(AmpGraphRuntime *runtime, uint32_t default_frames) {
    if (!runtime) return;
    size_t node_count = runtime->nodes.size();
    uint32_t batches_default = runtime->default_batches > 0U ? runtime->default_batches : 1U;

    for (size_t idx = 0; idx < node_count; ++idx) {
        RuntimeNode &node = *runtime->nodes[idx];
        node.debug_frame_cache = RuntimeNode::DebugFrameCache{};
        if (node.outputs.empty()) {
            OutputTap tap{};
            tap.name = "default";
            tap.contract = node.output_contract;
            tap.primary = true;
            node.outputs.push_back(std::move(tap));
        }

        for (OutputTap &tap : node.outputs) {
            std::string key = node.name;
            if (!tap.name.empty()) {
                key.append("::").append(tap.name);
            }
            if (!tap.ring) {
                tap.ring = ensure_edge_ring(runtime, key, node.vector_policy);
            }
            if (!tap.ring) {
                continue;
            }

            BlockFrameContract contract = node_block_frame_contract(runtime, node, default_frames);
            uint32_t min_frames = std::max<uint32_t>(1U, contract.min_frames);
            uint32_t preferred_frames = std::max<uint32_t>(min_frames, contract.preferred_frames);
            uint32_t max_frames = std::max<uint32_t>(preferred_frames, contract.max_frames);

            uint32_t base_frames = preferred_frames;
            uint32_t capacity = base_frames > 0U ? base_frames : 1U;

            if (!node.prefill_only && base_frames > 0U) {
                uint64_t scaled = static_cast<uint64_t>(base_frames) * 4ULL;
                if (scaled > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                    scaled = std::numeric_limits<uint32_t>::max();
                }
                capacity = std::max<uint32_t>(capacity, static_cast<uint32_t>(scaled));
            }
            capacity = std::max<uint32_t>(capacity, base_frames + 1U);
            capacity = std::max<uint32_t>(capacity, max_frames);
            uint32_t align_unit = std::max<uint32_t>(min_frames, 1U);
            capacity = align_frames_up(capacity, align_unit);

            // Always reserve a parking slot so consumers that need an entire block
            // can observe it without the ring collapsing to empty after wraparound.
            if (capacity < std::numeric_limits<uint32_t>::max()) {
                uint64_t expanded = static_cast<uint64_t>(capacity) + 1ULL;
                if (expanded > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
                    capacity = std::numeric_limits<uint32_t>::max();
                } else {
                    capacity = align_frames_up(static_cast<uint32_t>(expanded), align_unit);
                }
            }
            if (capacity == 0U) capacity = base_frames + 1U;

            uint32_t batches = batches_default;
            if (!node.buffer_shapes.empty() && node.buffer_shapes[0].batches > 0U) {
                batches = node.buffer_shapes[0].batches;
            }
            uint32_t channels = node.channel_hint > 0U ? node.channel_hint : 1U;
            if (!node.buffer_shapes.empty() && node.buffer_shapes[0].channels > 0U) {
                channels = node.buffer_shapes[0].channels;
            }

            TensorShape ring_shape{};
            ring_shape.batches = batches;
            ring_shape.channels = channels;
            ring_shape.frames = capacity;
            if (tap.declared_shape.batches > 0U) {
                ring_shape.batches = tap.declared_shape.batches;
            }
            if (tap.declared_shape.channels > 0U) {
                ring_shape.channels = tap.declared_shape.channels;
            }

            tap.ring->storage = make_tensor(ring_shape);
            tap.ring->storage->shape = ring_shape;
            tap.ring->frame_stride = static_cast<size_t>(ring_shape.batches) * ring_shape.channels;
            tap.ring->capacity = capacity;
            tap.ring->head = 0U;
            tap.ring->tail = 0U;
            tap.ring->reader_tails.clear();
            tap.ring->policy = node.vector_policy;
            tap.ring->constant = node.constant_node;
            tap.ring->contract = tap.contract;
            tap.ring->nominal_sample_rate = tap.contract.sample_rate_hz;
            tap.ring->effective_sample_rate = tap.contract.sample_rate_hz;
            tap.ring->sample_rate_free = tap.contract.sample_rate_free;
            tap.ring->sample_rate_ema_alpha = tap.contract.sample_rate_ema_alpha;
            tap.ring->sample_rate_window = tap.contract.sample_rate_window;
            tap.ring->produced_total = 0U;
            if (tap.ring->contract.primary_consumer == std::numeric_limits<uint32_t>::max()) {
                tap.ring->contract.primary_consumer = EDGE_RING_HOST_CONSUMER;
            }
        }
    }

    for (size_t idx = 0; idx < node_count; ++idx) {
        RuntimeNode &node = *runtime->nodes[idx];
        for (EdgeReader &reader : node.input_edges) {
            if (!reader.ring) continue;
            edge_ring_register_consumer(*reader.ring, reader.consumer_index);
        }
    }
}

/*** Ready check ***/
static bool kpn_node_ready(
    const AmpGraphRuntime *runtime,
    const RuntimeNode &node,
    uint32_t /*node_index*/,
    uint32_t default_frames,
    uint32_t &out_frames
) {
    BlockFrameContract contract = node_block_frame_contract(runtime, node, default_frames);
    uint32_t min_frames = std::max<uint32_t>(1U, contract.min_frames);
    uint32_t preferred_frames = std::max<uint32_t>(min_frames, contract.preferred_frames);
    uint32_t max_frames = std::max<uint32_t>(preferred_frames, contract.max_frames);

    const OutputTap *primary = runtime_primary_output(node);
    if (primary == nullptr || !primary->ring || primary->ring->capacity == 0U) {
        return false;
    }

    uint32_t feasible = max_frames;
    if (feasible == 0U || feasible == std::numeric_limits<uint32_t>::max()) {
        feasible = primary->ring->capacity > 0U ? primary->ring->capacity : preferred_frames;
    }

    for (size_t reader_index = 0; reader_index < node.input_edges.size(); ++reader_index) {
        const EdgeReader &reader = node.input_edges[reader_index];
        uint32_t ready = 0U;
        if (reader.ring) {
            ready = edge_ring_available_for_consumer(*reader.ring, reader.consumer_index);
        }
        bool have_hold_state = reader_index < node.input_hold_cache.size()
            && node.input_hold_cache[reader_index].valid;
        bool can_use_hold = reader.hold_if_absent && (have_hold_state || reader.has_optional_default);
        if (ready < min_frames) {
            if (!can_use_hold) {
                return false;
            }
            ready = min_frames;
        }
        ready = align_frames_down(ready, min_frames);
        if (ready < min_frames) {
            if (!can_use_hold) {
                return false;
            }
            ready = min_frames;
        }
        feasible = std::min(feasible, ready);
    }

    uint32_t free_space = std::numeric_limits<uint32_t>::max();
    bool have_output_ring = false;
    for (const OutputTap &tap : node.outputs) {
        if (!tap.ring) continue;
        have_output_ring = true;
        uint32_t tap_free = edge_ring_free(*tap.ring);
        if (tap_free < min_frames) {
            return false;
        }
        tap_free = align_frames_down(tap_free, min_frames);
        if (tap_free < min_frames) {
            return false;
        }
        free_space = std::min(free_space, tap_free);
    }
    if (!have_output_ring || free_space == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    feasible = std::min(feasible, free_space);

    if (node.input_edges.empty()) {
        uint32_t implicit = align_frames_down(preferred_frames, min_frames);
        if (implicit < min_frames) {
            implicit = min_frames;
        }
        feasible = std::min(feasible, implicit);
    }

    feasible = std::min(feasible, max_frames);
    if (feasible < min_frames) {
        return false;
    }

    uint32_t target = std::min(preferred_frames, feasible);
    target = align_frames_down(target, min_frames);
    if (target < min_frames) {
        target = align_frames_down(feasible, min_frames);
    }
    if (target < min_frames) {
        return false;
    }

    out_frames = target;
    return true;
}

/*** Build param tensors forward decl used above ***/
static const DefaultParam *find_default_param(const RuntimeNode &node, const std::string &name);

/*** Node execution (KPN step) ***/
static int kpn_execute_node_block(
    AmpGraphRuntime *runtime,
    uint32_t node_index,
    uint32_t frames,
    double sample_rate,
    const EdgeRunnerControlHistory *history
) {
    if (!runtime || node_index >= runtime->nodes.size()) return -1;
    if (frames == 0U) return 0;

    RuntimeNode &node = *runtime->nodes[node_index];
    uint32_t default_hint = runtime && runtime->default_frames > 0U
        ? runtime->default_frames
        : frames;
    BlockFrameContract contract = node_block_frame_contract(runtime, node, default_hint);
    OutputTap *primary = runtime_primary_output(node);
    if (primary == nullptr) {
        OutputTap tap{};
        tap.name = "default";
        tap.contract = node.output_contract;
        tap.primary = true;
        node.outputs.push_back(std::move(tap));
        primary = runtime_primary_output(node);
    }
    if (primary == nullptr) {
        return -1;
    }
    if (!primary->ring) {
        std::string key = node.name;
        if (!primary->name.empty()) {
            key.append("::").append(primary->name);
        }
        primary->ring = ensure_edge_ring(runtime, key, node.vector_policy);
    }
    if (!primary->ring) return -1;
    uint32_t ready_frames = 0U;
    if (!kpn_node_ready(runtime, node, node_index, default_hint, ready_frames) || ready_frames < frames) {
        return 1; // not ready
    }

    size_t workspace_batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
    uint32_t total_channels = 0U;

    struct InputCache {
        size_t reader_index{0};
        EdgeReader reader;
        std::vector<double> buffer;
        uint32_t batches{0};
        uint32_t channels{0};
        size_t stride{0};
        bool consumed_from_ring{false};
    };
    std::vector<InputCache> caches;
    caches.reserve(node.input_edges.size());
    if (node.input_hold_cache.size() < node.input_edges.size()) {
        node.input_hold_cache.resize(node.input_edges.size());
    }

    for (size_t reader_index = 0; reader_index < node.input_edges.size(); ++reader_index) {
        const EdgeReader &reader = node.input_edges[reader_index];
        InputCache cache;
        cache.reader_index = reader_index;
        cache.reader = reader;

    RuntimeNode::InputHoldState &hold_state = node.input_hold_cache[reader_index];

        uint32_t ring_batches = 0U;
        uint32_t ring_channels = 0U;
        if (reader.ring && reader.ring->storage) {
            ring_batches = std::max<uint32_t>(1U, reader.ring->storage->shape.batches);
            ring_channels = std::max<uint32_t>(1U, reader.ring->storage->shape.channels);
        }

        uint32_t available = 0U;
        if (reader.ring) {
            available = edge_ring_available_for_consumer(*reader.ring, reader.consumer_index);
        }

        bool have_hold_state = hold_state.valid;
        bool can_use_hold = reader.hold_if_absent && (have_hold_state || reader.has_optional_default);

        bool use_ring_data = reader.ring && reader.ring->storage && available >= frames;
        if (use_ring_data) {
            cache.batches = ring_batches > 0U ? ring_batches : std::max<uint32_t>(1U, hold_state.batches);
            cache.channels = ring_channels > 0U ? ring_channels : std::max<uint32_t>(1U, hold_state.channels);
            cache.stride = cache.batches * cache.channels;
            if (cache.batches == 0U || cache.channels == 0U) {
                cache.batches = 1U;
                cache.channels = 1U;
                cache.stride = 1U;
            }
            if (cache.batches > workspace_batches) workspace_batches = cache.batches;
            cache.buffer.resize(static_cast<size_t>(frames) * cache.stride);
            uint32_t tail = edge_ring_consumer_tail(*reader.ring, reader.consumer_index);
            edge_ring_copy_out(*reader.ring, tail, frames, cache.buffer.data());
            cache.consumed_from_ring = true;

            const double *last_frame = cache.buffer.data() + static_cast<size_t>(frames - 1U) * cache.stride;
            hold_state.values.assign(last_frame, last_frame + cache.stride);
            hold_state.batches = cache.batches;
            hold_state.channels = cache.channels;
            hold_state.valid = true;
        } else {
            cache.batches = ring_batches > 0U ? ring_batches : (hold_state.valid ? hold_state.batches : std::max<uint32_t>(1U, runtime->default_batches));
            if (cache.batches == 0U) {
                cache.batches = 1U;
            }
            cache.channels = ring_channels > 0U ? ring_channels : (hold_state.valid ? hold_state.channels : 1U);
            if (cache.channels == 0U) {
                cache.channels = 1U;
            }
            cache.stride = cache.batches * cache.channels;
            if (cache.batches > workspace_batches) workspace_batches = cache.batches;
            cache.buffer.resize(static_cast<size_t>(frames) * cache.stride);

            std::vector<double> seed;
            if (reader.ring && reader.ring->storage && reader.ring->capacity > 0U && available > 0U) {
                uint32_t effective_batches = ring_batches > 0U ? ring_batches : cache.batches;
                uint32_t effective_channels = ring_channels > 0U ? ring_channels : cache.channels;
                size_t peek_stride = static_cast<size_t>(effective_batches) * static_cast<size_t>(effective_channels);
                if (peek_stride == 0U) {
                    peek_stride = 1U;
                }
                seed.resize(peek_stride);
                uint32_t tail = edge_ring_consumer_tail(*reader.ring, reader.consumer_index);
                uint32_t start = (tail + available - 1U) % reader.ring->capacity;
                edge_ring_copy_out(*reader.ring, start, 1U, seed.data());
                hold_state.values = seed;
                hold_state.batches = effective_batches;
                hold_state.channels = effective_channels;
                hold_state.valid = true;
            }
            if (hold_state.valid && hold_state.values.size() == cache.stride) {
                seed = hold_state.values;
            } else if (hold_state.valid && hold_state.values.size() != cache.stride) {
                seed.resize(cache.stride, hold_state.values.empty() ? 0.0 : hold_state.values.back());
            } else if (reader.has_optional_default) {
                seed.assign(cache.stride, reader.optional_default_value);
            } else {
                seed.assign(cache.stride, 0.0);
            }

            if (!seed.empty()) {
                for (uint32_t f = 0; f < frames; ++f) {
                    double *dst = cache.buffer.data() + static_cast<size_t>(f) * cache.stride;
                    std::memcpy(dst, seed.data(), cache.stride * sizeof(double));
                }
            }
            hold_state.values = seed;
            hold_state.batches = cache.batches;
            hold_state.channels = cache.channels;
            hold_state.valid = !seed.empty() && (reader.has_optional_default || reader.hold_if_absent);
        }

        caches.push_back(std::move(cache));
        total_channels += caches.back().channels;
    }

    size_t workspace_stride = workspace_batches * static_cast<size_t>(std::max<uint32_t>(1U, total_channels));
    if (total_channels > 0U) {
        node.audio_workspace.resize(static_cast<size_t>(frames) * workspace_stride);
        std::fill(node.audio_workspace.begin(), node.audio_workspace.end(), 0.0);
    } else {
        node.audio_workspace.clear();
    }
    node.audio_workspace_shape.batches = static_cast<uint32_t>(workspace_batches);
    node.audio_workspace_shape.channels = total_channels;
    node.audio_workspace_shape.frames = frames;

    size_t channel_offset = 0U;
    for (const InputCache &cache : caches) {
        if (cache.channels == 0U) {
            continue;
        }
        for (uint32_t f = 0; f < frames; ++f) {
            double *dst_frame = node.audio_workspace.data() + static_cast<size_t>(f) * workspace_stride;
            double *dst_block = dst_frame + static_cast<size_t>(channel_offset) * workspace_batches;
            const double *src_block = cache.buffer.data() + static_cast<size_t>(f) * cache.stride;
            if (cache.batches == workspace_batches) {
                std::memcpy(dst_block, src_block, cache.stride * sizeof(double));
            } else if (cache.batches == 1U) {
                for (uint32_t c = 0; c < cache.channels; ++c) {
                    double value = src_block[c];
                    double *dst_channel = dst_block + static_cast<size_t>(c) * workspace_batches;
                    std::fill(dst_channel, dst_channel + workspace_batches, value);
                }
            } else {
                for (uint32_t c = 0; c < cache.channels; ++c) {
                    const double *src_channel = src_block + static_cast<size_t>(c) * cache.batches;
                    double *dst_channel = dst_block + static_cast<size_t>(c) * workspace_batches;
                    size_t copy = std::min<size_t>(workspace_batches, cache.batches);
                    std::memcpy(dst_channel, src_channel, copy * sizeof(double));
                    if (copy < workspace_batches) {
                        std::fill(dst_channel + copy, dst_channel + workspace_batches, src_channel[copy - 1]);
                    }
                }
            }
        }
        channel_offset += cache.channels;
    }

    auto &param_tensors = build_param_tensors(node);
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

    std::vector<TapOutputBuffer> tap_output_buffers;
    std::vector<EdgeRunnerTapBuffer> tap_buffer_views;
    std::vector<EdgeRunnerTapStatus> tap_status_views;
    EdgeRunnerTapContext tap_context{};
    if (node.expose_tap_context && frames > 0U) {
        tap_status_views.reserve(node.outputs.size());
        tap_buffer_views.reserve(node.outputs.size());
        tap_output_buffers.reserve(node.outputs.size());
        for (OutputTap &tap : node.outputs) {
            EdgeRunnerTapStatus status{};
            status.tap_name = tap.name.c_str();
            if (tap.ring) {
                status.subscriber_count = static_cast<uint32_t>(tap.ring->reader_tails.size());
                status.connected = status.subscriber_count > 0 ? 1U : 0U;
                uint32_t primary_consumer = tap.ring->contract.primary_consumer;
                status.primary_consumer_present =
                    (primary_consumer != EDGE_RING_HOST_CONSUMER &&
                     tap.ring->reader_tails.find(primary_consumer) != tap.ring->reader_tails.end())
                        ? 1U
                        : 0U;
            }
            tap_status_views.push_back(status);

            if (tap.primary || !tap.expose_in_context || !tap.ring) {
                continue;
            }
            TapOutputBuffer buffer_entry{};
            buffer_entry.tap = &tap;
            size_t stride = edge_ring_frame_stride(*tap.ring);
            if (stride == 0U) {
                uint32_t batches_hint = tap.ring->storage ? tap.ring->storage->shape.batches : 1U;
                uint32_t channels_hint = tap.ring->storage ? tap.ring->storage->shape.channels : 1U;
                stride = static_cast<size_t>(std::max<uint32_t>(1U, batches_hint) *
                                             std::max<uint32_t>(1U, channels_hint));
            }
            buffer_entry.scratch.resize(static_cast<size_t>(frames) * stride);
            std::fill(buffer_entry.scratch.begin(), buffer_entry.scratch.end(), 0.0);
            tap_output_buffers.push_back(std::move(buffer_entry));

            EdgeRunnerTapBuffer view{};
            view.tap_name = tap.name.c_str();
            view.buffer_class = tap.buffer_class.empty() ? "pcm" : tap.buffer_class.c_str();
            if (tap.ring && tap.ring->storage) {
                view.shape.batches = tap.ring->storage->shape.batches;
                view.shape.channels = tap.ring->storage->shape.channels;
            } else {
                view.shape.batches = 1U;
                view.shape.channels = 1U;
            }
            view.shape.frames = frames;
            view.frame_stride = stride;
            view.data = tap_output_buffers.back().scratch.data();
            tap_buffer_views.push_back(view);
        }
        if (!tap_buffer_views.empty()) {
            tap_context.outputs.items = tap_buffer_views.data();
            tap_context.outputs.count = static_cast<uint32_t>(tap_buffer_views.size());
        }
        if (!tap_status_views.empty()) {
            tap_context.status.items = tap_status_views.data();
            tap_context.status.count = static_cast<uint32_t>(tap_status_views.size());
        }
    }

    EdgeRunnerNodeInputs inputs{};
    if (!node.audio_workspace.empty() && total_channels > 0U) {
        inputs.audio.has_audio = 1;
        inputs.audio.batches = static_cast<uint32_t>(workspace_batches);
        inputs.audio.channels = total_channels;
        inputs.audio.frames = frames;
        inputs.audio.data = node.audio_workspace.data();
    } else {
        inputs.audio.has_audio = 0;
        inputs.audio.batches = static_cast<uint32_t>(workspace_batches);
        inputs.audio.channels = 0U;
        inputs.audio.frames = frames;
        inputs.audio.data = nullptr;
    }
    if (!param_views.empty()) {
        inputs.params.count = static_cast<uint32_t>(param_views.size());
        inputs.params.items = param_views.data();
    }
    if (node.expose_tap_context && (tap_context.outputs.count > 0U || tap_context.status.count > 0U)) {
        inputs.taps = tap_context;
    }

    double *frame_buffer = nullptr;
    int out_channels = 0;
    void *state_arg = node.state;
    bool used_v2 = false;
    int execution_status = 0;

    if (node.supports_v2) {
        AmpNodeMetrics metrics{};
        int v2_status = amp_run_node_v2(
            &node.descriptor,
            &inputs,
            static_cast<int>(workspace_batches),
            static_cast<int>(inputs.audio.channels > 0 ? inputs.audio.channels : node.channel_hint),
            static_cast<int>(frames),
            sample_rate,
            &frame_buffer,
            &out_channels,
            &state_arg,
            history,
            AMP_EXECUTION_MODE_FORWARD,
            &metrics
        );
        if (v2_status == AMP_E_UNSUPPORTED) {
            node.supports_v2 = false;
        } else if (v2_status != 0) {
            execution_status = v2_status;
        } else {
            used_v2 = true;
            node.has_latest_metrics = true;
            node.latest_metrics = metrics;
            node.total_heat_accumulated += static_cast<double>(metrics.accumulated_heat);
        }
    }
    if (!used_v2) {
        int status = amp_run_node(
            &node.descriptor,
            &inputs,
            static_cast<int>(workspace_batches),
            static_cast<int>(inputs.audio.channels > 0 ? inputs.audio.channels : node.channel_hint),
            static_cast<int>(frames),
            sample_rate,
            &frame_buffer,
            &out_channels,
            &state_arg,
            history
        );
        if (status != 0 || frame_buffer == nullptr) {
            if (frame_buffer != nullptr) amp_free(frame_buffer);
            return status != 0 ? status : -1;
        }
        node.has_latest_metrics = false;
    } else if (execution_status != 0 || frame_buffer == nullptr) {
        if (frame_buffer != nullptr) amp_free(frame_buffer);
        return execution_status != 0 ? execution_status : -1;
    }

    auto output_edge = primary->ring;
    if (!output_edge) {
        amp_free(frame_buffer);
        return -1;
    }
    edge_ring_write(*output_edge, frame_buffer, frames);
    uint32_t debug_batches = static_cast<uint32_t>(workspace_batches > 0 ? workspace_batches : 1U);
    uint32_t debug_channels = out_channels > 0
        ? static_cast<uint32_t>(out_channels)
        : std::max<uint32_t>(1U, node.channel_hint);
    runtime_node_record_debug_frame(node, debug_batches, debug_channels, frames);

    TensorShape output_shape{};
    output_shape.batches = static_cast<uint32_t>(workspace_batches);
    output_shape.channels = static_cast<uint32_t>(out_channels > 0 ? out_channels : 1);
    output_shape.frames = frames;
    if (!node.output || !shapes_equal(node.output->shape, output_shape)) {
        node.output = make_tensor(output_shape);
    }
    size_t output_stride = static_cast<size_t>(output_shape.batches) * static_cast<size_t>(output_shape.channels);
    std::memcpy(node.output->data(), frame_buffer, static_cast<size_t>(frames) * output_stride * sizeof(double));
    amp_free(frame_buffer);

    if (!tap_output_buffers.empty()) {
        for (TapOutputBuffer &buffer_entry : tap_output_buffers) {
            if (!buffer_entry.tap || !buffer_entry.tap->ring) continue;
            edge_ring_write(*buffer_entry.tap->ring, buffer_entry.scratch.data(), frames);
        }
    }

    node.output->shape = output_shape;
    node.output_batches = node.output->shape.batches;
    node.output_channels = node.output->shape.channels;
    node.output_frames = node.output->shape.frames;
    node.output_ring = output_edge->storage;
    uint32_t start_index = (output_edge->head + output_edge->capacity - frames) % (output_edge->capacity == 0U ? 1U : output_edge->capacity);
    node.output_ring_head = start_index;
    node.output_ring_capacity = output_edge->capacity;

    if (!node.type_name.empty() && node.type_name == "ResamplerNode" && node.has_latest_metrics && primary != nullptr && primary->ring) {
        double reported_rate = node.latest_metrics.reserved[0];
        if (reported_rate > 0.0) {
            if (primary->ring->sample_rate_free || primary->ring->nominal_sample_rate <= 0.0) {
                primary->ring->effective_sample_rate = reported_rate;
            } else {
                primary->ring->effective_sample_rate = primary->ring->nominal_sample_rate;
            }
        }
    }

    auto channel_it = runtime->channels.find(node.name);
    if (channel_it != runtime->channels.end()) {
        auto &channel = *channel_it->second;
        channel.token = node.output;
        channel.ring = output_edge->storage;
        channel.ring_frames = output_edge->capacity;
        uint32_t channel_block = contract.preferred_frames > 0U
            ? contract.preferred_frames
            : node.output_frames;
        if (channel_block == 0U) {
            channel_block = node.output_frames;
        }
        channel.block_frames = channel_block;
        channel.frame_stride = edge_ring_frame_stride(*output_edge);
        if (channel.ring_frames > 0U && channel.block_frames > 0U) {
            uint32_t span = std::min<uint32_t>(channel.block_frames, channel.ring_frames);
            uint32_t head = output_edge->head % channel.ring_frames;
            uint32_t span_mod = span % channel.ring_frames;
            if (span_mod == 0U) span_mod = channel.ring_frames;
            channel.block_start = (channel.ring_frames + head - span_mod) % channel.ring_frames;
        } else {
            channel.block_start = 0U;
        }
    }

    node.state = state_arg;
    for (const InputCache &cache : caches) {
        if (!cache.consumed_from_ring || !cache.reader.ring) {
            continue;
        }
        edge_ring_advance_consumer(*cache.reader.ring, cache.reader.consumer_index, frames);
    }
    if (output_edge) {
        edge_ring_recompute_tail(*output_edge);
    }
    return 0;
}

/*** Merge audio inputs for non-streaming execute path ***/
static std::shared_ptr<EigenTensorHolder> merge_audio_inputs(
    AmpGraphRuntime *runtime,
    RuntimeNode &node,
    std::vector<std::shared_ptr<EigenTensorHolder>> &workspace
) {
    if (!runtime) return nullptr;
    if (node.audio_indices.empty()) return nullptr;
    if (node.audio_indices.size() == 1U) {
        auto &source = runtime->nodes[node.audio_indices[0]];
        return source->output;
    }
    std::vector<std::shared_ptr<EigenTensorHolder>> sources;
    uint32_t batches = 0;
    uint32_t frames = 0;
    uint32_t total_channels = 0;
    for (uint32_t index : node.audio_indices) {
        if (index >= runtime->nodes.size()) return nullptr;
        auto &source = runtime->nodes[index];
        if (!source->output) return nullptr;
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
    if (sources.empty() || total_channels == 0U) return nullptr;

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

/*** Topology/Schedule ***/
static void runtime_update_scheduler_topology(AmpGraphRuntime *runtime) {
    if (!runtime) return;
    size_t node_count = runtime->nodes.size();
    runtime->dependents.assign(node_count, {});
    runtime->indegree.assign(node_count, 0U);
    runtime->execution_rank.assign(node_count, 0U);

    if (!runtime->execution_order.empty()) {
        for (size_t rank = 0; rank < runtime->execution_order.size(); ++rank) {
            uint32_t node_index = runtime->execution_order[rank];
            if (node_index < runtime->execution_rank.size()) {
                runtime->execution_rank[node_index] = static_cast<uint32_t>(rank);
            }
        }
    }

    for (size_t idx = 0; idx < node_count; ++idx) {
        RuntimeNode &node = *runtime->nodes[idx];
        OutputTap *primary = runtime_primary_output(node);
        if (primary == nullptr) {
            OutputTap tap{};
            tap.name = "default";
            tap.contract = node.output_contract;
            tap.primary = true;
            node.outputs.push_back(std::move(tap));
            primary = runtime_primary_output(node);
        }
        if (primary != nullptr) {
            if (!primary->ring) {
                std::string key = node.name;
                if (!primary->name.empty()) {
                    key.append("::").append(primary->name);
                }
                primary->ring = ensure_edge_ring(runtime, key, node.vector_policy);
            }
            if (primary->ring) {
                primary->ring->reader_tails.erase(static_cast<uint32_t>(idx));
            }
        }
        node.input_edges.clear();
        node.input_hold_cache.clear();
        for (size_t slot = 0; slot < node.audio_indices.size(); ++slot) {
            uint32_t source = node.audio_indices[slot];
            if (source >= node_count) continue;
            runtime->dependents[source].push_back(static_cast<uint32_t>(idx));
            runtime->indegree[idx] += 1U;
            auto &source_node = runtime->nodes[source];
            OutputTap *source_primary = runtime_primary_output(*source_node);
            if (source_primary == nullptr) {
                OutputTap tap{};
                tap.name = "default";
                tap.primary = true;
                tap.contract = source_node->output_contract;
                source_node->outputs.push_back(std::move(tap));
                source_primary = runtime_primary_output(*source_node);
            }
            if (source_primary && !source_primary->ring) {
                std::string key = source_node->name;
                if (!source_primary->name.empty()) {
                    key.append("::").append(source_primary->name);
                }
                    source_primary->ring = ensure_edge_ring(runtime, key, source_node->vector_policy);
            }
            auto edge = source_primary ? source_primary->ring : nullptr;
            EdgeReader reader{};
            reader.ring = edge;
            reader.consumer_index = static_cast<uint32_t>(idx);
            reader.tap_name = source_primary ? source_primary->name : std::string{};
            configure_input_reader_metadata(node, reader, slot);
            reader.producer_node_index = static_cast<uint32_t>(source);
            reader.producer_output_index = 0U;
            if (edge) {
                edge_ring_register_consumer(*edge, reader.consumer_index);
            }
            node.input_edges.push_back(reader);
            node.input_hold_cache.emplace_back();
        }
    }
}

static double compute_scheduler_priority(const AmpGraphRuntime *runtime, uint32_t node_index) {
    if (!runtime || node_index >= runtime->nodes.size()) return 0.0;
    const RuntimeNode &node = *runtime->nodes[node_index];
    const SchedulerParams &params = runtime->scheduler_params;
    BlockFrameContract contract = node_block_frame_contract(
        runtime,
        node,
        runtime->default_frames
    );

    double normalized_order = 0.0;
    if (!runtime->execution_rank.empty() && runtime->execution_order.size() > 1U) {
        double rank = static_cast<double>(runtime->execution_rank[node_index]);
        double denom = static_cast<double>(runtime->execution_order.size() - 1U);
        normalized_order = denom > 0.0 ? rank / denom : 0.0;
    }
    double early_component = (1.0 - normalized_order) * params.early_bias;
    double late_component = normalized_order * params.late_bias;

    double saturation_component = 0.0;
    if (params.saturation_bias != 0.0) {
        double activity = 0.0;
        if (node.has_latest_metrics) {
            activity += node.latest_metrics.processing_time_seconds;
            activity += node.latest_metrics.logging_time_seconds;
            activity += node.latest_metrics.total_time_seconds;
        }
        activity += node.total_heat_accumulated;

        double backlog_penalty = 0.0;
        const OutputTap *primary = runtime_primary_output(node);
        if (primary != nullptr && primary->ring && primary->ring->capacity > 0U) {
            uint32_t capacity = primary->ring->capacity;
            uint32_t head = primary->ring->head % capacity;
            uint32_t tail = primary->ring->tail % capacity;
            uint32_t used = edge_ring_distance(head, tail, capacity);
            if (capacity > 0U) {
                backlog_penalty = static_cast<double>(used) / static_cast<double>(capacity);
            }
        }

        auto channel_it = runtime->channels.find(node.name);
        if (channel_it != runtime->channels.end()) {
            const auto &channel = *channel_it->second;
            if (channel.ring_frames > 0U && channel.block_frames > 0U) {
                double block_fill = static_cast<double>(channel.block_frames) / static_cast<double>(channel.ring_frames);
                activity += block_fill;
            }
        }

        activity -= backlog_penalty;
        saturation_component = activity * params.saturation_bias;
    }

    double vector_component = 0.0;
    if (node.vector_policy.channel_expand > 1U) {
        vector_component += static_cast<double>(node.vector_policy.channel_expand - 1U);
    }
    if (contract.preferred_frames > 0U) {
        vector_component += 1.0 / static_cast<double>(contract.preferred_frames);
    }
    double priority = early_component + late_component + saturation_component + vector_component;
    return priority * contract.priority_weight;
}

static void build_execution_schedule(const AmpGraphRuntime *runtime, std::vector<uint32_t> &out) {
    out.clear();
    if (!runtime) return;
    size_t node_count = runtime->nodes.size();
    if (node_count == 0U) return;

    if (runtime->scheduler_mode == AMP_SCHEDULER_ORDERED || runtime->indegree.size() != node_count) {
        out = runtime->execution_order;
        if (out.empty()) {
            out.reserve(node_count);
            for (size_t i = 0; i < node_count; ++i) out.push_back(static_cast<uint32_t>(i));
        }
        return;
    }

    std::vector<uint32_t> indegree = runtime->indegree;
    std::vector<uint32_t> ready;
    ready.reserve(node_count);
    for (uint32_t idx = 0; idx < indegree.size(); ++idx) {
        if (indegree[idx] == 0U) ready.push_back(idx);
    }

    out.reserve(node_count);
    while (!ready.empty()) {
        size_t best_pos = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < ready.size(); ++i) {
            uint32_t candidate = ready[i];
            double score = compute_scheduler_priority(runtime, candidate);
            if (score > best_score) {
                best_score = score;
                best_pos = i;
            }
        }
        uint32_t node_idx = ready[best_pos];
        ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(best_pos));
        out.push_back(node_idx);
        if (node_idx >= runtime->dependents.size()) continue;
        for (uint32_t dep : runtime->dependents[node_idx]) {
            if (dep >= indegree.size()) continue;
            if (indegree[dep] > 0U) {
                indegree[dep] -= 1U;
                if (indegree[dep] == 0U) ready.push_back(dep);
            }
        }
    }

    if (out.size() != node_count) {
        out = runtime->execution_order;
        if (out.empty()) {
            out.reserve(node_count);
            for (size_t i = 0; i < node_count; ++i) out.push_back(static_cast<uint32_t>(i));
        }
    }
}

/*** Modulation ***/
static void apply_modulation(
    const ModConnectionInfo &info,
    const std::shared_ptr<EigenTensorHolder> &source,
    std::shared_ptr<EigenTensorHolder> &target
) {
    if (!source || !target) return;
    const TensorShape &src_shape = source->shape;
    TensorShape dst_shape = target->shape;
    size_t src_stride = static_cast<size_t>(src_shape.batches) * static_cast<size_t>(src_shape.channels);
    size_t dst_stride = static_cast<size_t>(dst_shape.batches) * static_cast<size_t>(dst_shape.channels);
    if (src_shape.frames != dst_shape.frames || src_stride == 0 || dst_stride == 0) return;

    const double *src_ptr = source->data();
    double *dst_ptr = target->data();
    for (uint32_t frame = 0; frame < dst_shape.frames; ++frame) {
        const double *src_frame = src_ptr + static_cast<size_t>(frame) * src_stride;
        double *dst_frame = dst_ptr + static_cast<size_t>(frame) * dst_stride;
        for (uint32_t batch = 0; batch < dst_shape.batches; ++batch) {
            for (uint32_t channel = 0; channel < dst_shape.channels; ++channel) {
                uint32_t src_channel = channel;
                if (info.channel >= 0) {
                    src_channel = static_cast<uint32_t>(info.channel);
                    if (src_channel >= src_shape.channels) continue;
                } else if (src_shape.channels <= channel) {
                    src_channel = src_shape.channels - 1U;
                }
                size_t src_index = static_cast<size_t>(batch) * src_shape.channels + src_channel;
                size_t dst_index = static_cast<size_t>(batch) * dst_shape.channels + channel;
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

/*** Param helpers ***/
static const DefaultParam *find_default_param(const RuntimeNode &node, const std::string &name) {
    for (const DefaultParam &param : node.defaults) {
        if (param.name == name) return &param;
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
                size_t element_count = binding.shape.element_count();
                if (!binding.use_ring) {
                    if (!binding.data.empty() && element_count > 0U) {
                        tensor->map_external(binding.data.data(), element_count);
                    } else {
                        tensor->clear_external();
                        tensor->set_zero();
                    }
                } else {
                    size_t frame_stride = binding.frame_stride;
                    uint32_t window_frames = binding.window_frames;
                    if (frame_stride > 0U && binding.ring_frames > 0U && window_frames > 0U && element_count > 0U) {
                        uint32_t head = binding.ring_head % binding.ring_frames;
                        if (head + window_frames <= binding.ring_frames) {
                            tensor->map_external(
                                binding.data.data() + static_cast<size_t>(head) * frame_stride,
                                static_cast<size_t>(window_frames) * frame_stride
                            );
                        } else {
                            tensor->clear_external();
                            tensor->set_zero();
                            for (uint32_t f = 0; f < window_frames; ++f) {
                                uint32_t ring_pos = static_cast<uint32_t>((head + f) % binding.ring_frames);
                                std::memcpy(
                                    tensor->data() + static_cast<size_t>(f) * frame_stride,
                                    binding.data.data() + static_cast<size_t>(ring_pos) * frame_stride,
                                    frame_stride * sizeof(double)
                                );
                            }
                        }
                    } else {
                        tensor->clear_external();
                        tensor->set_zero();
                    }
                }
            }
            binding.dirty = false;
            continue;
        }

        const DefaultParam *def = find_default_param(node, name);
        if (def == nullptr) {
            if (tensor) {
                tensor->clear_external();
                tensor->set_zero();
            }
            continue;
        }
        if (!tensor || !shapes_equal(tensor->shape, def->shape)) {
            tensor = make_tensor(def->shape);
            entry.second = tensor;
        }
        tensor->clear_external();
        if (!def->data.empty()) {
            size_t copy_size = std::min(def->data.size(), tensor->storage_size());
            tensor->set_zero();
            std::memcpy(tensor->data(), def->data.data(), copy_size * sizeof(double));
        } else {
            tensor->set_zero();
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
        if (entry.first == name) return entry.second;
    }
    auto tensor = make_tensor(fallback_shape);
    tensor->shape = fallback_shape;
    tensors.emplace_back(name, tensor);
    return tensor;
}

/*** Parse node descriptor blob ***/
static bool parse_node_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t length) {
    if (!runtime || !blob || length == 0) return false;

    DescriptorReader reader(blob, length);
    uint32_t node_count = 0;
    if (!reader.read_u32(node_count)) return false;

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

        // Audio inputs
        for (uint32_t a = 0; a < audio_count; ++a) {
            uint32_t src_len = 0;
            if (!reader.read_u32(src_len)) return false;
            std::string source;
            if (!reader.read_string(src_len, source)) return false;
            node->audio_inputs.push_back(std::move(source));
        }
        node->lane_projections.resize(node->audio_inputs.size());
        for (size_t lane_idx = 0; lane_idx < node->audio_inputs.size(); ++lane_idx) {
            LaneProjectionConfig cfg{};
            cfg.input_name = node->audio_inputs[lane_idx];
            node->lane_projections[lane_idx] = std::move(cfg);
        }

        // Mod connections
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

        // Default params
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
            if (!reader.read_string(param_name_len, param_name)) return false;

            DefaultParam param{};
            param.name = std::move(param_name);
            param.shape = make_shape(batches, channels, frames);

            size_t value_count = static_cast<size_t>(blob_len) / sizeof(double);
            param.data.resize(value_count);

            if (reader.remaining < blob_len) return false;
            std::memcpy(param.data.data(), reader.cursor, blob_len);
            if (!reader.skip(blob_len)) return false;

            node->param_cache_index[param.name] = node->param_cache.size();
            node->param_cache.emplace_back(param.name, std::shared_ptr<EigenTensorHolder>());
            node->defaults.push_back(std::move(param));
            node->param_cache_dirty = true;
        }

        // Buffer shapes (workspace/output hints)
        for (uint32_t s = 0; s < shape_count; ++s) {
            uint32_t b = 0, c = 0, f = 0;
            if (!reader.read_u32(b) || !reader.read_u32(c) || !reader.read_u32(f)) return false;
            node->buffer_shapes.push_back(make_shape(b, c, f));
            if (node->channel_hint < c) node->channel_hint = c;
        }

        // Params JSON
        if (!reader.read_string(params_len, node->params_json)) return false;

        node->finalize_descriptor();
        node->oversample_ratio = std::max<uint32_t>(1U, parse_uint_metadata(node->params_json, "oversample_ratio", 1U));
        node->declared_delay_frames = parse_uint_metadata(node->params_json, "declared_delay", 0U);

        uint32_t vector_expand = parse_uint_metadata(node->params_json, "vector_channel_expand", 1U);
        if (vector_expand == 0U) vector_expand = 1U;
        node->vector_policy.channel_expand = vector_expand;
        node->vector_policy.block_frames = parse_uint_metadata(node->params_json, "vector_block_frames", 0U);
    node->vector_policy.min_block_frames = parse_uint_metadata(node->params_json, "vector_min_block_frames", 0U);
    node->vector_policy.max_block_frames = parse_uint_metadata(node->params_json, "vector_max_block_frames", 0U);
    node->vector_policy.priority_weight = parse_double_metadata(node->params_json, "vector_priority_weight", 1.0);
        node->vector_policy.archtypal_mode = parse_bool_metadata(node->params_json, "vector_archtypal_mode", false);
        node->spectral_working_space.duration_frames = parse_uint_metadata(node->params_json, "working_ft_duration_frames", 0U);
        node->spectral_working_space.frequency_bins = parse_uint_metadata(node->params_json, "working_ft_frequency_bins", 0U);
        std::string working_time_default = parse_string_metadata(
            node->params_json,
            "working_ft_time_projection",
            node->spectral_working_space.time_projection_default
        );
        if (!working_time_default.empty()) {
            node->spectral_working_space.time_projection_default = working_time_default;
        }
        std::string working_freq_default = parse_string_metadata(
            node->params_json,
            "working_ft_frequency_projection",
            node->spectral_working_space.freq_projection_default
        );
        if (!working_freq_default.empty()) {
            node->spectral_working_space.freq_projection_default = working_freq_default;
        }
        node->prefill_frames = parse_uint_metadata(node->params_json, "prefill_frames", 0U);
        node->prefill_only = parse_bool_metadata(node->params_json, "prefill_only", false);
        node->constant_node = parse_bool_metadata(node->params_json, "constant_node", false);
        if (node->constant_node) {
            node->prefill_only = true;
            if (node->prefill_frames == 0U) node->prefill_frames = 1U;
        }
        node->supports_v2 = parse_bool_metadata(node->params_json, "supports_v2", node->supports_v2);
        node->output_contract.simultaneous_availability = parse_bool_metadata(node->params_json, "fifo_simultaneous_output", true);
        std::string release_policy = parse_string_metadata(node->params_json, "fifo_release_policy", "all");
        if (release_policy == "primary" || release_policy == "primary_consumer") {
            node->output_contract.release_policy = EdgeRingReleasePolicy::PrimaryConsumer;
        } else {
            node->output_contract.release_policy = EdgeRingReleasePolicy::AllConsumers;
        }
        node->output_contract.sample_rate_hz = parse_double_metadata(node->params_json, "fifo_sample_rate_hz", 0.0);
        node->output_contract.sample_rate_free = parse_bool_metadata(node->params_json, "fifo_sample_rate_free", false);
        node->output_contract.sample_rate_ema_alpha = parse_double_metadata(node->params_json, "fifo_sample_rate_ema_alpha", 0.0);
        node->output_contract.sample_rate_window = parse_uint_metadata(node->params_json, "fifo_sample_rate_window", 0U);
        uint32_t primary_consumer_index = parse_uint_metadata(
            node->params_json,
            "fifo_primary_consumer",
            std::numeric_limits<uint32_t>::max()
        );
        if (primary_consumer_index != std::numeric_limits<uint32_t>::max()) {
            node->output_contract.primary_consumer = primary_consumer_index;
        }
        if (node->output_contract.primary_consumer == std::numeric_limits<uint32_t>::max()) {
            node->output_contract.primary_consumer = EDGE_RING_HOST_CONSUMER;
        }

        runtime->node_index[node->name] = runtime->nodes.size();
        runtime->nodes.push_back(std::move(node));
    }

    for (auto &entry : runtime->nodes) {
        for (size_t lane_idx = 0; lane_idx < entry->lane_projections.size(); ++lane_idx) {
            LaneProjectionConfig &lane_cfg = entry->lane_projections[lane_idx];
            std::string input_name = lane_cfg.input_name;
            std::string sanitized = sanitize_metadata_key_component(input_name);
            std::string time_key = "projection_" + sanitized + "_time_policy";
            std::string freq_key = "projection_" + sanitized + "_frequency_policy";
            std::string phase_key = "projection_" + sanitized + "_phase_policy";
            lane_cfg.time_policy = parse_string_metadata(
                entry->params_json,
                time_key.c_str(),
                entry->spectral_working_space.time_projection_default
            );
            if (lane_cfg.time_policy.empty()) {
                lane_cfg.time_policy = entry->spectral_working_space.time_projection_default;
            }
            lane_cfg.freq_policy = parse_string_metadata(
                entry->params_json,
                freq_key.c_str(),
                entry->spectral_working_space.freq_projection_default
            );
            if (lane_cfg.freq_policy.empty()) {
                lane_cfg.freq_policy = entry->spectral_working_space.freq_projection_default;
            }
            lane_cfg.phase_policy = parse_string_metadata(
                entry->params_json,
                phase_key.c_str(),
                lane_cfg.phase_policy
            );
            if (lane_cfg.phase_policy.empty()) {
                lane_cfg.phase_policy = "preserve";
            }
        }
    }

    // Resolve audio indices and build channels/rings
    for (auto &entry : runtime->nodes) {
        entry->audio_indices.clear();
        for (const std::string &nm : entry->audio_inputs) {
            auto it = runtime->node_index.find(nm);
            if (it == runtime->node_index.end()) return false;
            entry->audio_indices.push_back(static_cast<uint32_t>(it->second));
        }
        runtime->channels.emplace(entry->name, std::make_shared<KahnChannel>(entry->name));
        if (entry->outputs.empty()) {
            OutputTap tap{};
            tap.name = "default";
            tap.primary = true;
            tap.contract = entry->output_contract;
            tap.buffer_class = "pcm";
            entry->outputs.push_back(std::move(tap));
        } else {
            bool found_primary = false;
            for (auto &tap : entry->outputs) {
                if (tap.buffer_class.empty()) {
                    tap.buffer_class = tap.primary ? "pcm" : "aux";
                }
                if (tap.primary) {
                    found_primary = true;
                    break;
                }
            }
            if (!found_primary) {
                entry->outputs.front().primary = true;
            }
        }
        if (entry->type_name == "FFTDivisionNode") {
            bool enable_spectrum_taps = parse_bool_metadata(entry->params_json, "enable_spectrum_taps", true);
            uint32_t window_size = parse_uint_metadata(entry->params_json, "window_size", 0U);
            if (enable_spectrum_taps && window_size > 0U) {
                uint32_t slot_batches = entry->channel_hint > 0U ? entry->channel_hint : 1U;
                auto tap_exists = [&](const std::string &tap_name) {
                    for (const auto &tap : entry->outputs) {
                        if (tap.name == tap_name) {
                            return true;
                        }
                    }
                    return false;
                };
                if (!tap_exists("spectral_real")) {
                    OutputTap real{};
                    real.name = "spectral_real";
                    real.primary = false;
                    real.contract = entry->output_contract;
                    real.buffer_class = "spectrum";
                    real.declared_shape = make_shape(slot_batches, window_size, 0U);
                    real.expose_in_context = true;
                    entry->outputs.push_back(std::move(real));
                }
                if (!tap_exists("spectral_imag")) {
                    OutputTap imag{};
                    imag.name = "spectral_imag";
                    imag.primary = false;
                    imag.contract = entry->output_contract;
                    imag.buffer_class = "spectrum";
                    imag.declared_shape = make_shape(slot_batches, window_size, 0U);
                    imag.expose_in_context = true;
                    entry->outputs.push_back(std::move(imag));
                }
                entry->expose_tap_context = true;
            }
        }
        for (auto &tap : entry->outputs) {
            configure_tap_channel_semantics(*entry, tap);
        }
    }

    // Create input edges and register consumers
    for (auto &entry : runtime->nodes) {
        entry->input_edges.clear();
        entry->input_hold_cache.clear();
        for (size_t slot = 0; slot < entry->audio_indices.size(); ++slot) {
            uint32_t idx = entry->audio_indices[slot];
            if (idx >= runtime->nodes.size()) return false;
            auto &source = runtime->nodes[idx];
            OutputTap *source_primary = runtime_primary_output(*source);
            if (source_primary == nullptr) {
                OutputTap tap{};
                tap.name = "default";
                tap.primary = true;
                tap.contract = source->output_contract;
                source->outputs.push_back(std::move(tap));
                source_primary = runtime_primary_output(*source);
            }
            if (source_primary && !source_primary->ring) {
                std::string key = source->name;
                if (!source_primary->name.empty()) {
                    key.append("::").append(source_primary->name);
                }
                source_primary->ring = ensure_edge_ring(runtime, key, source->vector_policy);
            }
            auto edge = source_primary ? source_primary->ring : nullptr;
            size_t consumer_index = runtime->node_index[entry->name];
            EdgeReader reader{};
            reader.ring = edge;
            reader.consumer_index = static_cast<uint32_t>(consumer_index);
            reader.tap_name = source_primary ? source_primary->name : std::string{};
            configure_input_reader_metadata(*entry, reader, slot);
            reader.producer_node_index = static_cast<uint32_t>(idx);
            reader.producer_output_index = 0U;
            if (edge) {
                edge_ring_register_consumer(*edge, reader.consumer_index);
            }
            entry->input_edges.push_back(reader);
            entry->input_hold_cache.emplace_back();
        }
    }

    if (!runtime->nodes.empty()) {
        runtime->sink_index = static_cast<uint32_t>(runtime->nodes.size() - 1);
    }
    runtime_update_scheduler_topology(runtime);
    return true;
}

/*** Parse plan blob ***/
static bool parse_plan_blob(AmpGraphRuntime *runtime, const uint8_t *blob, size_t length) {
    if (!runtime) return false;

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

    if (length < 12U) return false;
    if (std::memcmp(blob, "AMPL", 4) != 0) return false;

    DescriptorReader reader(blob + 4, length - 4);
    uint32_t version = 0;
    uint32_t node_count = 0;
    if (!reader.read_u32(version) || !reader.read_u32(node_count)) return false;
    if (version == 0U || version > 2U) return false;
    if (node_count == 0U) return true;

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
            if (!reader.read_u32(declared_delay) || !reader.read_u32(oversample_ratio)) return false;
        }
        (void)audio_offset;

        std::string node_name;
        if (!reader.read_string(name_len, node_name)) return false;

        auto it = runtime->node_index.find(node_name);
        if (it == runtime->node_index.end()) return false;
        if (function_id >= runtime->execution_order.size()) return false;

        runtime->execution_order[function_id] = static_cast<uint32_t>(it->second);
        auto &node = runtime->nodes[it->second];

        if (version >= 2U) {
            if (declared_delay > 0U) node->declared_delay_frames = declared_delay;
            if (oversample_ratio > 0U) node->oversample_ratio = oversample_ratio;
        }
        if (audio_span > 0 && node->channel_hint < audio_span) {
            node->channel_hint = audio_span;
        }

        for (uint32_t p = 0; p < param_count; ++p) {
            uint32_t param_len = 0;
            uint32_t cursor = 0;
            uint32_t reserved = 0;
            if (!reader.read_u32(param_len) || !reader.read_u32(cursor) || !reader.read_u32(reserved)) return false;
            std::string param_name;
            if (!reader.read_string(param_len, param_name)) return false;
            (void)param_name; (void)cursor; (void)reserved;
        }
    }

    if (!runtime->execution_order.empty()) {
        runtime->sink_index = runtime->execution_order.back();
    }
    runtime_update_scheduler_topology(runtime);
    return true;
}

/*** Clone helpers (kept for parity, might be unused in this TU) ***/
static std::shared_ptr<EigenTensorHolder> clone_param_tensor(const DefaultParam &param) {
    TensorShape shape = param.shape;
    auto tensor = make_tensor(shape);
    if (!param.data.empty()) {
        size_t copy_size = std::min(param.data.size(), tensor->storage_size());
        std::memcpy(tensor->data(), param.data.data(), copy_size * sizeof(double));
    }
    tensor->shape = shape;
    return tensor;
}

static std::shared_ptr<EigenTensorHolder> clone_binding_tensor(const ParamBinding &binding) {
    TensorShape shape = binding.shape;
    auto tensor = make_tensor(shape);
    if (!binding.data.empty()) {
        size_t copy_size = std::min(binding.data.size(), tensor->storage_size());
        std::memcpy(tensor->data(), binding.data.data(), copy_size * sizeof(double));
    }
    tensor->shape = shape;
    return tensor;
}

/*** Prepass ***/
static int runtime_execute_prepass(
    AmpGraphRuntime *runtime,
    double sample_rate,
    const EdgeRunnerControlHistory *history,
    uint32_t default_frames
) {
    if (!runtime) return 0;

    for (auto &node_ptr : runtime->nodes) {
        node_ptr->prepass_done = (node_ptr->prefill_frames == 0U);
    }

    bool progress = true;
    while (progress) {
        progress = false;
        for (uint32_t idx : runtime->execution_order) {
            if (idx >= runtime->nodes.size()) continue;
            RuntimeNode &node = *runtime->nodes[idx];
            if (node.prefill_frames == 0U || node.prepass_done) continue;

            uint32_t frames = node.prefill_frames;
            uint32_t ready_frames = 0U;
            if (!kpn_node_ready(runtime, node, idx, frames, ready_frames) || ready_frames < frames) {
                continue;
            }

            int status = kpn_execute_node_block(runtime, idx, frames, sample_rate, history);
            if (status > 0) continue;
            if (status != 0) return status;

            node.prepass_done = true;
            progress = true;
        }
    }

    for (const auto &ptr : runtime->nodes) {
        if (ptr->prefill_frames > 0U && !ptr->prepass_done) {
            return -1;
        }
    }
    return 0;
}

/*** Streamer ring ops ***/
static size_t streamer_ring_size(const AmpGraphStreamer *streamer) {
    uint64_t w = streamer->write_index.load(std::memory_order_acquire);
    uint64_t r = streamer->read_index.load(std::memory_order_acquire);
    if (w < r) return 0;
    uint64_t diff = w - r;
    if (diff > streamer->ring_frames) diff = streamer->ring_frames;
    return static_cast<size_t>(diff);
}

static size_t streamer_ring_free(const AmpGraphStreamer *streamer) {
    size_t used = streamer_ring_size(streamer);
    if (streamer->ring_frames <= used) return 0;
    return streamer->ring_frames - used;
}

static void streamer_copy_from_ring(
    const AmpGraphStreamer *streamer,
    uint64_t start_index,
    size_t frames,
    double *destination
) {
    if (frames == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) return;
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
    if (frames == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) return;
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
    if (current == 0 || streamer->frame_stride == 0 || streamer->ring_buffer.empty()) return;

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
    uint64_t now = steady_now_millis();
    streamer->last_dump_millis.store(now, std::memory_order_release);
    streamer->last_consumed_millis.store(now, std::memory_order_release);
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
    if (frame_count == 0 || streamer->frame_stride == 0) return;
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
    streamer->last_dump_millis.store(steady_now_millis(), std::memory_order_release);
    streamer->dump_cv.notify_all();
}

static void streamer_write_frames(
    AmpGraphStreamer *streamer,
    const double *frames,
    size_t frame_count
) {
    if (frame_count == 0) return;
    if (streamer->ring_buffer.empty()) {
        streamer->ring_buffer.resize(static_cast<size_t>(streamer->ring_frames) * streamer->frame_stride);
        std::fill(streamer->ring_buffer.begin(), streamer->ring_buffer.end(), 0.0);
    }
    if (frame_count > streamer->ring_frames) {
        streamer_enqueue_chunk(streamer, frames, frame_count, streamer->batches, streamer->channels, streamer->produced_frames.load());
        streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
        streamer->last_produced_millis.store(steady_now_millis(), std::memory_order_release);
        return;
    }
    if (streamer_ring_free(streamer) < frame_count) {
        streamer_flush_ring_to_dump(streamer);
    }
    if (streamer_ring_free(streamer) < frame_count) {
        streamer_enqueue_chunk(streamer, frames, frame_count, streamer->batches, streamer->channels, streamer->produced_frames.load());
        streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
        streamer->last_produced_millis.store(steady_now_millis(), std::memory_order_release);
        return;
    }
    uint64_t start = streamer->write_index.load(std::memory_order_relaxed);
    streamer_copy_to_ring(streamer, start, frames, frame_count);
    streamer->write_index.store(start + frame_count, std::memory_order_release);
    streamer->produced_frames.fetch_add(frame_count, std::memory_order_release);
    streamer->last_produced_millis.store(steady_now_millis(), std::memory_order_release);
}

/*** KPN production***
/*** KPN production ***/
static int kpn_streamer_produce(
    AmpGraphStreamer *streamer,
    const EdgeRunnerControlHistory *history,
    uint32_t desired_frames
) {
    if (!streamer || !streamer->runtime) return -1;
    AmpGraphRuntime *runtime = streamer->runtime;

    if (runtime->sink_index >= runtime->nodes.size()) return -1;
    auto &sink_node = *runtime->nodes[runtime->sink_index];

    OutputTap *primary = runtime_primary_output(sink_node);
    if (!primary) return -1;

    if (!primary->ring) {
        std::string key = sink_node.name;
        if (!primary->name.empty()) {
            key.append("::").append(primary->name);
        }
        primary->ring = ensure_edge_ring(runtime, key, sink_node.vector_policy);
    }
    auto sink_edge = primary->ring;
    if (!sink_edge || sink_edge->capacity == 0U) return -1;

    if (streamer->ring_frames == 0U) {
        streamer->ring_frames = std::max<uint32_t>(desired_frames * 16U, sink_edge->capacity);
    }
    if (sink_edge->reader_tails.find(EDGE_RING_HOST_CONSUMER) == sink_edge->reader_tails.end()) {
        edge_ring_register_consumer(*sink_edge, EDGE_RING_HOST_CONSUMER);
    }

    uint32_t host_tail = edge_ring_consumer_tail(*sink_edge, EDGE_RING_HOST_CONSUMER);

    struct ReadyNodeEntry {
        uint32_t node_index{0};
        double priority{0.0};
        bool archtypal{false};
        bool operator<(const ReadyNodeEntry &rhs) const {
            if (priority == rhs.priority) {
                return node_index > rhs.node_index;
            }
            return priority < rhs.priority;
        }
    };

    std::priority_queue<ReadyNodeEntry> ready_queue;
    std::vector<uint32_t> archtypal_ready;

    auto rebuild_ready_queue = [&](void) {
        std::priority_queue<ReadyNodeEntry> empty;
        ready_queue.swap(empty);
        archtypal_ready.clear();
        for (uint32_t idx : runtime->execution_order) {
            if (idx >= runtime->nodes.size()) continue;
            RuntimeNode &candidate = *runtime->nodes[idx];
            if (candidate.prefill_only) continue;
            uint32_t frames = 0U;
            if (!kpn_node_ready(runtime, candidate, idx, streamer->block_frames, frames)) continue;
            double priority = compute_scheduler_priority(runtime, idx);
            ReadyNodeEntry entry{idx, priority, candidate.vector_policy.archtypal_mode};
            ready_queue.push(entry);
            if (entry.archtypal) {
                archtypal_ready.push_back(idx);
            }
        }
    };

    while (edge_ring_available_for_consumer(*sink_edge, EDGE_RING_HOST_CONSUMER) < desired_frames) {
        rebuild_ready_queue();
        bool progress = false;

        if (!archtypal_ready.empty()) {
            for (uint32_t idx : archtypal_ready) {
                if (idx >= runtime->nodes.size()) continue;
                RuntimeNode &node = *runtime->nodes[idx];
                if (node.prefill_only) continue;
                uint32_t frames = 0U;
                if (!kpn_node_ready(runtime, node, idx, streamer->block_frames, frames)) continue;
                int status = kpn_execute_node_block(runtime, idx, frames, streamer->sample_rate, history);
                if (status < 0) return status;
                if (status == 0) {
                    progress = true;
                }
            }
            if (progress) {
                continue;
            }
        }

        while (!ready_queue.empty()) {
            ReadyNodeEntry entry = ready_queue.top();
            ready_queue.pop();
            if (entry.node_index >= runtime->nodes.size()) continue;
            RuntimeNode &node = *runtime->nodes[entry.node_index];
            if (node.prefill_only) continue;
            uint32_t frames = 0U;
            if (!kpn_node_ready(runtime, node, entry.node_index, streamer->block_frames, frames)) continue;
            int status = kpn_execute_node_block(runtime, entry.node_index, frames, streamer->sample_rate, history);
            if (status < 0) return status;
            if (status == 0) {
                progress = true;
                break;
            }
        }

        if (!progress) {
            return -1; // deadlock
        }
    }

    // Drain exactly desired_frames from sink ring for host consumption
    size_t stride = edge_ring_frame_stride(*sink_edge);
    std::vector<double> chunk;
    chunk.resize(static_cast<size_t>(desired_frames) * stride);
    edge_ring_copy_out(*sink_edge, host_tail, desired_frames, chunk.data());
    edge_ring_advance_consumer(*sink_edge, EDGE_RING_HOST_CONSUMER, desired_frames);
    edge_ring_recompute_tail(*sink_edge);

    if (streamer->batches == 0U) {
        streamer->batches = sink_node.output ? sink_node.output->shape.batches
                                             : (runtime->default_batches > 0U ? runtime->default_batches : 1U);
    }
    if (streamer->channels == 0U) {
        streamer->channels = sink_node.output ? sink_node.output->shape.channels : sink_node.channel_hint;
        if (streamer->channels == 0U) streamer->channels = 1U;
    }
    if (streamer->frame_stride == 0U) {
        streamer->frame_stride = static_cast<size_t>(streamer->batches) * static_cast<size_t>(streamer->channels);
    }

    streamer_write_frames(streamer, chunk.data(), desired_frames);
    return 0;
}

/*** Streamer thread ***/
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
            streamer->dump_cv.notify_all();
            return;
        }
        streamer->history = history;
        streamer->history_owned = true;
    }

    if (streamer->block_frames == 0U) {
        streamer->block_frames = (streamer->runtime && streamer->runtime->default_frames > 0U)
            ? streamer->runtime->default_frames : 64U;
    }

    runtime_initialize_edge_rings(streamer->runtime, streamer->block_frames);

    int prepass_status = runtime_execute_prepass(
        streamer->runtime,
        streamer->sample_rate,
        history,
        streamer->block_frames
    );
    if (prepass_status != 0) {
        streamer->last_status = prepass_status;
        streamer->running.store(false, std::memory_order_release);
        streamer->dump_cv.notify_all();
        return;
    }

    uint32_t step_frames = streamer->block_frames > 0U ? streamer->block_frames : 1U;
    while (!streamer->stop_requested.load(std::memory_order_acquire)) {
        int status = kpn_streamer_produce(streamer, history, step_frames);
        if (status != 0) {
            streamer->last_status = status;
            break;
        }
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

/*** Clear runtime ***/
static void clear_runtime(AmpGraphRuntime *runtime) {
    if (!runtime) return;
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
        for (auto &tap : node->outputs) {
            tap.ring.reset();
        }
        node->outputs.clear();
        node->input_edges.clear();
        node->input_hold_cache.clear();
        node->output.reset();
        node->audio_workspace.clear();
    }
    runtime->nodes.clear();
    runtime->node_index.clear();
    runtime->execution_order.clear();
    runtime->channels.clear();
    runtime->edge_rings.clear();
    runtime->dependents.clear();
    runtime->indegree.clear();
    runtime->execution_rank.clear();
    runtime->scheduler_mode = AMP_SCHEDULER_LEARNED;
    runtime->scheduler_params = SchedulerParams{};
}

/*** Non-streaming execute (impl) ***/
static int execute_runtime_with_history_impl(
    AmpGraphRuntime *runtime,
    EdgeRunnerControlHistory *history,
    bool history_owned,
    int /*frames_hint*/,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
) {
    if (!runtime) return -1;
    runtime_clear_error(runtime);

    if (!out_buffer || !out_batches || !out_channels || !out_frames) {
        runtime_set_error(runtime, -1, "execute_runtime", nullptr, "output buffers must not be null");
        return -1;
    }

    struct HistoryGuard {
        EdgeRunnerControlHistory *ptr;
        bool owned;
        HistoryGuard(EdgeRunnerControlHistory *p, bool take) : ptr(p), owned(take) {}
        ~HistoryGuard() { if (owned && ptr) amp_release_control_history(ptr); }
    } guard(history, history_owned);

    const EdgeRunnerControlHistory *history_view = history;

    for (auto &entry : runtime->nodes) {
        entry->output.reset();
        entry->output_batches = 0;
        entry->output_channels = 0;
        entry->output_frames = 0;
        entry->has_latest_metrics = false;
        entry->total_heat_accumulated = 0.0;
        entry->latest_metrics = {};
    }
    for (auto &channel_entry : runtime->channels) {
        auto &channel = *channel_entry.second;
        channel.token.reset();
        channel.ring.reset();
        channel.ring_frames = 0;
        channel.block_start = 0;
        channel.block_frames = 0;
        channel.frame_stride = 0;
    }

    double dsp_rate = runtime->dsp_sample_rate > 0.0 ? runtime->dsp_sample_rate : sample_rate;

    std::vector<std::shared_ptr<EigenTensorHolder>> scratch;
    scratch.reserve(8);

    std::vector<uint32_t> schedule;
    build_execution_schedule(runtime, schedule);
    if (schedule.empty()) {
        schedule = runtime->execution_order;
        if (schedule.empty()) {
            schedule.reserve(runtime->nodes.size());
            for (size_t i = 0; i < runtime->nodes.size(); ++i) {
                schedule.push_back(static_cast<uint32_t>(i));
            }
        }
    }

    for (uint32_t order : schedule) {
        if (order >= runtime->nodes.size()) {
            runtime_set_error(runtime, -1, "schedule_bounds", nullptr,
                              std::string("execution order index out of range: ") + std::to_string(order));
            return -1;
        }
        RuntimeNode &node = *runtime->nodes[order];
        scratch.clear();

        std::shared_ptr<EigenTensorHolder> audio_tensor = merge_audio_inputs(runtime, node, scratch);
        if (!audio_tensor && !node.audio_indices.empty()) {
            runtime_set_error(runtime, -1, "merge_audio_inputs", &node, "required audio input missing");
            return -1;
        }

        uint32_t batches = runtime->default_batches > 0U ? runtime->default_batches : 1U;
        uint32_t frames  = runtime->default_frames > 0U ? runtime->default_frames : 1U;
        uint32_t input_channels = node.channel_hint > 0U ? node.channel_hint : 1U;
        if (audio_tensor) {
            batches = audio_tensor->shape.batches;
            frames  = audio_tensor->shape.frames;
            input_channels = audio_tensor->shape.channels;
        }

        // Params + modulation
        auto &param_tensors = build_param_tensors(node);
        for (const ModConnectionInfo &mod : node.mod_connections) {
            auto it = runtime->channels.find(mod.source);
            if (it == runtime->channels.end()) continue;
            std::shared_ptr<EigenTensorHolder> src = it->second->token;
            if (!src) continue;
            TensorShape shape = src->shape;
            auto dst = ensure_param_tensor(param_tensors, mod.param, shape);
            apply_modulation(mod, src, dst);
        }

        std::vector<EdgeRunnerParamView> param_views;
        param_views.reserve(param_tensors.size());
        for (auto &entry : param_tensors) {
            EdgeRunnerParamView view{};
            view.name     = entry.first.c_str();
            view.batches  = entry.second->shape.batches;
            view.channels = entry.second->shape.channels;
            view.frames   = entry.second->shape.frames;
            view.data     = entry.second->data();
            param_views.push_back(view);
        }

        std::vector<TapOutputBuffer> tap_output_buffers;
        std::vector<EdgeRunnerTapBuffer> tap_buffer_views;
        std::vector<EdgeRunnerTapStatus> tap_status_views;
        EdgeRunnerTapContext tap_context{};
        if (node.expose_tap_context && frames > 0U) {
            tap_output_buffers.reserve(node.outputs.size());
            tap_buffer_views.reserve(node.outputs.size());
            tap_status_views.reserve(node.outputs.size());
            for (OutputTap &tap : node.outputs) {
                EdgeRunnerTapStatus status{};
                status.tap_name = tap.name.c_str();
                status.connected = 1U;
                status.subscriber_count = 1U;
                status.primary_consumer_present = 1U;
                tap_status_views.push_back(status);
                if (tap.primary || !tap.expose_in_context) {
                    continue;
                }
                TapOutputBuffer buffer_entry{};
                buffer_entry.tap = &tap;
                uint32_t tap_batches = tap.declared_shape.batches > 0U
                    ? tap.declared_shape.batches
                    : std::max<uint32_t>(1U, node.channel_hint);
                uint32_t tap_channels = tap.declared_shape.channels > 0U
                    ? tap.declared_shape.channels
                    : (tap.ring && tap.ring->storage
                        ? tap.ring->storage->shape.channels
                        : std::max<uint32_t>(1U, node.channel_hint));
                size_t stride = static_cast<size_t>(tap_batches) * tap_channels;
                buffer_entry.scratch.resize(static_cast<size_t>(frames) * stride);
                std::fill(buffer_entry.scratch.begin(), buffer_entry.scratch.end(), 0.0);
                tap_output_buffers.push_back(std::move(buffer_entry));

                EdgeRunnerTapBuffer view{};
                view.tap_name = tap.name.c_str();
                view.buffer_class = tap.buffer_class.empty() ? "pcm" : tap.buffer_class.c_str();
                view.shape.batches = tap_batches;
                view.shape.channels = tap_channels;
                view.shape.frames = frames;
                view.frame_stride = stride;
                view.data = tap_output_buffers.back().scratch.data();
                tap_buffer_views.push_back(view);
            }
            if (!tap_buffer_views.empty()) {
                tap_context.outputs.items = tap_buffer_views.data();
                tap_context.outputs.count = static_cast<uint32_t>(tap_buffer_views.size());
            }
            if (!tap_status_views.empty()) {
                tap_context.status.items = tap_status_views.data();
                tap_context.status.count = static_cast<uint32_t>(tap_status_views.size());
            }
        }

        EdgeRunnerNodeInputs inputs{};
        if (audio_tensor) {
            inputs.audio.has_audio = 1;
            inputs.audio.batches   = audio_tensor->shape.batches;
            inputs.audio.channels  = audio_tensor->shape.channels;
            inputs.audio.frames    = audio_tensor->shape.frames;
            inputs.audio.data      = audio_tensor->data();
        } else {
            inputs.audio.has_audio = 0;
            inputs.audio.batches   = batches;
            inputs.audio.channels  = 0;
            inputs.audio.frames    = frames;
            inputs.audio.data      = nullptr;
        }
        if (!param_views.empty()) {
            inputs.params.count = static_cast<uint32_t>(param_views.size());
            inputs.params.items = param_views.data();
        }
        if (node.expose_tap_context && (tap_context.outputs.count > 0U || tap_context.status.count > 0U)) {
            inputs.taps = tap_context;
        }

        double *frame_buffer = nullptr;
        int out_channels = 0;
        void *state_arg = node.state;
        bool used_v2 = false;

        bool skip_node_output = false;
        if (node.supports_v2) {
            AmpNodeMetrics frame_metrics{};
            int v2_status = amp_run_node_v2(
                &node.descriptor,
                &inputs,
                static_cast<int>(batches),
                static_cast<int>(input_channels),
                static_cast<int>(frames),
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
            } else {
                used_v2 = true;
                AmpMailboxEntry *mail_entry = (state_arg != NULL)
                    ? amp_node_mailbox_pop(state_arg)
                    : nullptr;
                if (mail_entry != nullptr) {
                    if (frame_buffer != nullptr && frame_buffer != mail_entry->buffer) {
                        amp_free(frame_buffer);
                    }
                    frame_buffer = mail_entry->buffer;
                    out_channels = mail_entry->channels > 0 ? mail_entry->channels : 1;
                    node.latest_metrics = mail_entry->metrics;
                    node.has_latest_metrics = true;
                    node.total_heat_accumulated += static_cast<double>(mail_entry->metrics.accumulated_heat);
                    const int mailbox_status = mail_entry->status;
                    if (mailbox_status == AMP_E_PENDING && frame_buffer == nullptr) {
                        skip_node_output = true;
                    } else if (mailbox_status != 0 && mailbox_status != AMP_E_PENDING) {
                        if (frame_buffer != nullptr) {
                            amp_free(frame_buffer);
                            frame_buffer = nullptr;
                        }
                        amp_mailbox_entry_release(mail_entry);
                        runtime_set_error(
                            runtime,
                            mailbox_status,
                            "run_node_v2",
                            &node,
                            std::string("mailbox entry returned status ") + std::to_string(mailbox_status));
                        return -1;
                    }
                    amp_mailbox_entry_release(mail_entry);
                } else {
                    node.has_latest_metrics = false;
                    node.latest_metrics = frame_metrics;
                    if (v2_status == 0 && frame_buffer != nullptr) {
                        node.has_latest_metrics = true;
                    }
                    node.total_heat_accumulated += static_cast<double>(frame_metrics.accumulated_heat);
                    if (v2_status != 0 && v2_status != AMP_E_PENDING) {
                        std::string detail = std::string("amp_run_node_v2 returned status ") + std::to_string(v2_status);
                        if (frame_buffer != nullptr) {
                            amp_free(frame_buffer);
                            frame_buffer = nullptr;
                        }
                        runtime_set_error(runtime, v2_status, "run_node_v2", &node, std::move(detail));
                        return -1;
                    }
                    if (v2_status == AMP_E_PENDING && frame_buffer == nullptr) {
                        skip_node_output = true;
                    }
                }
            }
        }

        if (!used_v2) {
            int status = amp_run_node(
                &node.descriptor,
                &inputs,
                static_cast<int>(batches),
                static_cast<int>(input_channels),
                static_cast<int>(frames),
                dsp_rate,
                &frame_buffer,
                &out_channels,
                &state_arg,
                history_view
            );
            if (status != 0 || frame_buffer == nullptr) {
                if (frame_buffer) amp_free(frame_buffer);
                std::string detail;
                int code = status;
                if (frame_buffer == nullptr && status == 0) {
                    detail = "amp_run_node returned null buffer";
                    code = -1;
                } else {
                    detail = std::string("amp_run_node returned status ") + std::to_string(status);
                }
                runtime_set_error(runtime, code, "run_node", &node, std::move(detail));
                return -1;
            }
            node.has_latest_metrics = false;
        } else if (!frame_buffer) {
            if (skip_node_output) {
                if (node.state != state_arg) {
                    if (node.state) amp_release_state(node.state);
                }
                node.state = state_arg;
                node.has_latest_metrics = false;
                continue;
            }
            runtime_set_error(runtime, -1, "run_node_v2", &node, "amp_run_node_v2 returned null buffer");
            return -1;
        }

        TensorShape out_shape{};
        out_shape.batches = batches;
        out_shape.channels = static_cast<uint32_t>(out_channels > 0 ? out_channels : 1);
        out_shape.frames = frames;

        auto node_output = make_tensor(out_shape);
        node_output->shape = out_shape;

        size_t frame_stride = static_cast<size_t>(out_shape.batches) * static_cast<size_t>(out_shape.channels);
        std::memcpy(node_output->data(), frame_buffer, static_cast<size_t>(frames) * frame_stride * sizeof(double));
        amp_free(frame_buffer);

        if (node.state != state_arg) {
            if (node.state) amp_release_state(node.state);
        }
        node.state = state_arg;

        node.output = node_output;
        node.output_batches  = node_output->shape.batches;
        node.output_channels = node_output->shape.channels;
        node.output_frames   = node_output->shape.frames;

        runtime_node_record_debug_frame(
            node,
            node_output->shape.batches,
            node_output->shape.channels,
            node_output->shape.frames
        );

        auto channel_it = runtime->channels.find(node.name);
        if (channel_it != runtime->channels.end()) {
            auto &channel = *channel_it->second;
            channel.token = node_output;
            channel.ring.reset();
            channel.ring_frames = 0U;
            BlockFrameContract channel_contract = node_block_frame_contract(
                runtime,
                node,
                runtime ? runtime->default_frames : node_output->shape.frames
            );
            uint32_t channel_block = channel_contract.preferred_frames > 0U
                ? channel_contract.preferred_frames
                : node_output->shape.frames;
            if (channel_block == 0U) {
                channel_block = node_output->shape.frames;
            }
            channel.block_frames = channel_block;
            channel.frame_stride = static_cast<size_t>(node_output->shape.batches) * static_cast<size_t>(node_output->shape.channels);
            channel.block_start = 0U;
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

    *out_buffer  = sink.output->data();
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
    if (control_blob && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (!history) {
            runtime_set_error(runtime, -1, "load_control_history", nullptr, "amp_load_control_history returned null");
            return -1;
        }
        history_owned = true;
    }
    return execute_runtime_with_history_impl(
        runtime, history, history_owned, frames_hint, sample_rate,
        out_buffer, out_batches, out_channels, out_frames
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
    uint32_t batches = 0, channels = 0, frames = 0;
    int status = execute_runtime_with_history_impl(
        runtime, history, history_owned, frames_hint, sample_rate,
        &native_buffer, &batches, &channels, &frames
    );
    if (status != 0) return status;

    size_t required = static_cast<size_t>(batches) * static_cast<size_t>(channels) * static_cast<size_t>(frames);
    if (!out_buffer || out_buffer_len < required) {
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
    if (control_blob && control_len > 0U) {
        history = amp_load_control_history(control_blob, control_len, frames_hint);
        if (!history) {
            runtime_set_error(runtime, -1, "load_control_history", nullptr, "amp_load_control_history returned null");
            return -1;
        }
        history_owned = true;
    }
    return execute_runtime_history_into_impl(
        runtime, history, history_owned, frames_hint, sample_rate,
        out_buffer, out_buffer_len, out_batches, out_channels, out_frames
    );
}

/*** C API ***/
extern "C" {

AMP_API AmpGraphRuntime *amp_graph_runtime_create(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_create", descriptor_len, plan_len);
    auto *runtime = new (std::nothrow) AmpGraphRuntime();
    if (!runtime) return nullptr;

    runtime->default_batches = 1U;
    runtime->default_frames = 0U;
    runtime->dsp_sample_rate = 0.0;
    runtime->scheduler_mode = AMP_SCHEDULER_LEARNED;
    runtime->scheduler_params = SchedulerParams{};

    if (!parse_node_blob(runtime, descriptor_blob, descriptor_len) ||
        !parse_plan_blob(runtime, plan_blob, plan_len)) {
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
    runtime_update_scheduler_topology(runtime);
    return runtime;
}

AMP_API void amp_graph_runtime_destroy(AmpGraphRuntime *runtime) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_destroy", (size_t)(runtime != nullptr), 0);
    if (!runtime) return;
    clear_runtime(runtime);
    delete runtime;
}

AMP_API int amp_graph_runtime_configure(AmpGraphRuntime *runtime, uint32_t batches, uint32_t frames) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_configure", batches, frames);
    if (!runtime) return -1;
    runtime->default_batches = batches > 0U ? batches : 1U;
    runtime->default_frames = frames;
    return 0;
}

AMP_API void amp_graph_runtime_set_dsp_sample_rate(AmpGraphRuntime *runtime, double sample_rate) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_set_dsp_sample_rate", static_cast<size_t>(sample_rate), 0);
    if (!runtime) return;
    runtime->dsp_sample_rate = sample_rate > 0.0 ? sample_rate : 0.0;
}

AMP_API int amp_graph_runtime_set_scheduler_mode(AmpGraphRuntime *runtime, AmpGraphSchedulerMode mode) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_set_scheduler_mode", static_cast<size_t>(mode), 0);
    if (!runtime) return -1;
    switch (mode) {
        case AMP_SCHEDULER_LEARNED: runtime->scheduler_mode = AMP_SCHEDULER_LEARNED; break;
        case AMP_SCHEDULER_ORDERED:
        default: runtime->scheduler_mode = AMP_SCHEDULER_ORDERED; break;
    }
    return 0;
}

AMP_API int amp_graph_runtime_set_scheduler_params(AmpGraphRuntime *runtime, const AmpGraphSchedulerParams *params) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_set_scheduler_params", (size_t)(runtime != nullptr), (size_t)(params != nullptr));
    if (!runtime || !params) return -1;
    runtime->scheduler_params.early_bias = params->early_bias;
    runtime->scheduler_params.late_bias = params->late_bias;
    runtime->scheduler_params.saturation_bias = params->saturation_bias;
    return 0;
}

AMP_API void amp_graph_runtime_clear_params(AmpGraphRuntime *runtime) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_clear_params", (size_t)(runtime != nullptr), 0);
    if (!runtime) return;
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
    if (!runtime) return -1;
    runtime_clear_error(runtime);

    if (!node_name || !param_name || !data) {
        runtime_set_error(runtime, -1, "set_param", nullptr, "invalid argument");
        return -1;
    }

    auto it = runtime->node_index.find(std::string(node_name));
    if (it == runtime->node_index.end()) {
        runtime_set_error(runtime, -1, "set_param", nullptr, std::string("unknown node: ") + node_name);
        return -1;
    }
    RuntimeNode &node = *runtime->nodes[it->second];

    ParamBinding binding{};
    binding.shape = make_shape(batches, channels, frames);
    TensorShape incoming_shape = binding.shape;

    size_t total = static_cast<size_t>(incoming_shape.batches) *
                   static_cast<size_t>(incoming_shape.channels) *
                   static_cast<size_t>(incoming_shape.frames);
    if (total == 0U) {
        runtime_set_error(runtime, -1, "set_param", &node, "parameter tensor has zero elements");
        return -1;
    }

    const TensorShape *expected_shape = nullptr;
    for (const DefaultParam &param : node.defaults) {
        if (param.name == param_name) { expected_shape = &param.shape; break; }
    }
    if (!expected_shape) {
        auto existing = node.bindings.find(param_name);
        if (existing != node.bindings.end()) expected_shape = &existing->second.shape;
    }

    uint32_t target_batches  = expected_shape ? expected_shape->batches  : incoming_shape.batches;
    uint32_t target_channels = expected_shape ? expected_shape->channels : incoming_shape.channels;
    uint32_t target_frames   = incoming_shape.frames;
    if (expected_shape && expected_shape->frames > 0U) {
        target_frames = expected_shape->frames;
    } else if (!expected_shape && runtime->default_frames > 0U) {
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
#if defined(AMP_NATIVE_ENABLE_LOGGING)
        std::fprintf(
            stderr,
            "amp_graph_runtime_set_param: shape mismatch for %s.%s (expected %u x %u x %u, got %u x %u x %u)\n",
            node_name, param_name,
            target_batches, target_channels, target_frames,
            incoming_shape.batches, incoming_shape.channels, incoming_shape.frames
        );
#endif
        std::string detail = std::string("shape mismatch for ") + node.name + "." + param_name
            + " (expected " + std::to_string(target_batches) + " x " + std::to_string(target_channels) + " x " + std::to_string(target_frames)
            + ", got " + std::to_string(incoming_shape.batches) + " x " + std::to_string(incoming_shape.channels) + " x " + std::to_string(incoming_shape.frames) + ")";
        runtime_set_error(runtime, -2, "set_param", &node, std::move(detail));
        return -2;
    }

    binding.full_shape   = incoming_shape;
    binding.frame_stride = static_cast<size_t>(incoming_shape.batches) * static_cast<size_t>(incoming_shape.channels);
    binding.window_frames = enable_ring ? target_frames : incoming_shape.frames;
    BlockFrameContract param_contract = node_block_frame_contract(
        runtime,
        node,
        runtime ? runtime->default_frames : target_frames
    );
    binding.window_frames = std::max(binding.window_frames, param_contract.min_frames);
    binding.window_frames = std::max(binding.window_frames, param_contract.preferred_frames);
    binding.use_ring    = enable_ring;
    binding.ring_frames = enable_ring ? incoming_shape.frames : binding.window_frames;
    binding.ring_head   = 0;
    binding.shape       = make_shape(target_batches, target_channels, binding.window_frames);

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
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_describe_node", (size_t)(runtime != nullptr), (size_t)(summary != nullptr));
    if (!runtime || !node_name || !summary) return -1;

    auto it = runtime->node_index.find(std::string(node_name));
    if (it == runtime->node_index.end()) return -1;

    RuntimeNode &node = *runtime->nodes[it->second];
    summary->declared_delay_frames = node.declared_delay_frames;
    summary->oversample_ratio = node.oversample_ratio;
    summary->supports_v2 = node.supports_v2 ? 1 : 0;
    summary->has_metrics = node.has_latest_metrics ? 1 : 0;
    AmpNodeMetrics metrics{};
    if (node.has_latest_metrics) metrics = node.latest_metrics;
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
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_execute", (size_t)(control_blob ? control_len : 0U), (size_t)frames_hint);
    return execute_runtime_impl(runtime, control_blob, control_len, frames_hint, sample_rate,
                                out_buffer, out_batches, out_channels, out_frames);
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
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_execute_with_history", (size_t)(history != nullptr), (size_t)frames_hint);
    return execute_runtime_with_history_impl(runtime, history, false, frames_hint, sample_rate,
                                             out_buffer, out_batches, out_channels, out_frames);
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
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_execute_into",
                        (size_t)(control_blob ? control_len : 0U), (size_t)frames_hint);
    return execute_runtime_into_impl(runtime, control_blob, control_len, frames_hint, sample_rate,
                                     out_buffer, out_buffer_len, out_batches, out_channels, out_frames);
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
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_execute_history_into", (size_t)(history != nullptr), (size_t)frames_hint);
    return execute_runtime_history_into_impl(runtime, history, false, frames_hint, sample_rate,
                                             out_buffer, out_buffer_len, out_batches, out_channels, out_frames);
}

AMP_API int amp_graph_runtime_last_error(AmpGraphRuntime *runtime, AmpGraphRuntimeErrorInfo *out_error) {
    AMP_LOG_NATIVE_CALL("amp_graph_runtime_last_error", (size_t)(runtime != nullptr), (size_t)(out_error != nullptr));
    if (!runtime || !out_error) return -1;
    out_error->code = runtime->last_error.code;
    out_error->stage = runtime->last_error.stage.empty() ? nullptr : runtime->last_error.stage.c_str();
    out_error->node  = runtime->last_error.node.empty()  ? nullptr : runtime->last_error.node.c_str();
    out_error->detail= runtime->last_error.detail.empty()? nullptr : runtime->last_error.detail.c_str();
    return 0;
}

AMP_API void amp_graph_runtime_buffer_free(double *buffer) {
    (void)buffer; // buffers are owned/freed by amp_free in this design
}

AMP_API AmpGraphControlHistory *amp_graph_history_load(const uint8_t *blob, size_t blob_len, int frames_hint) {
    AMP_LOG_NATIVE_CALL("amp_graph_history_load", blob_len, (size_t)frames_hint);
    return amp_load_control_history(blob, blob_len, frames_hint);
}

AMP_API void amp_graph_history_destroy(AmpGraphControlHistory *history) {
    AMP_LOG_NATIVE_CALL("amp_graph_history_destroy", (size_t)(history != nullptr), 0);
    amp_release_control_history(history);
}

/*** Streamer API ***/
AMP_API AmpGraphStreamer *amp_graph_streamer_create(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    uint32_t ring_frames,
    uint32_t block_frames
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_create", (size_t)(runtime != nullptr), (size_t)ring_frames);
    if (!runtime) return nullptr;
    auto *streamer = new (std::nothrow) AmpGraphStreamer();
    if (!streamer) return nullptr;

    streamer->runtime = runtime;
    if (control_blob && control_len > 0U) {
        streamer->control_blob.assign(control_blob, control_blob + control_len);
    }
    streamer->frames_hint = frames_hint;
    streamer->sample_rate = sample_rate;
    streamer->ring_frames = ring_frames > 0U ? ring_frames : block_frames;
    if (streamer->ring_frames == 0U) streamer->ring_frames = 1024U;
    streamer->block_frames = block_frames > 0U ? block_frames : streamer->ring_frames / 2U;
    if (streamer->block_frames == 0U) streamer->block_frames = 256U;

    streamer->write_index.store(0, std::memory_order_relaxed);
    streamer->read_index.store(0, std::memory_order_relaxed);
    streamer->produced_frames.store(0, std::memory_order_relaxed);
    streamer->consumed_frames.store(0, std::memory_order_relaxed);
    streamer->stop_requested.store(false, std::memory_order_relaxed);
    streamer->start_time_millis.store(0, std::memory_order_relaxed);
    uint64_t now = steady_now_millis();
    streamer->last_produced_millis.store(now, std::memory_order_relaxed);
    streamer->last_consumed_millis.store(now, std::memory_order_relaxed);
    streamer->last_dump_millis.store(now, std::memory_order_relaxed);
    if (frames_hint > 0) {
        streamer->expected_output_frames = static_cast<uint64_t>(frames_hint);
    } else {
        streamer->expected_output_frames = 0;
    }

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
    if (!streamer) return -1;
    if (streamer->running.load(std::memory_order_acquire)) return 0;
    streamer->stop_requested.store(false, std::memory_order_release);
    uint64_t now = steady_now_millis();
    streamer->start_time_millis.store(now, std::memory_order_release);
    streamer->last_produced_millis.store(now, std::memory_order_release);
    streamer->last_consumed_millis.store(now, std::memory_order_release);
    streamer->last_dump_millis.store(now, std::memory_order_release);
    streamer->worker = std::thread(streamer_worker_main, streamer);
    return 0;
}

AMP_API void amp_graph_streamer_stop(AmpGraphStreamer *streamer) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_stop", (size_t)(streamer != nullptr), 0);
    if (!streamer) return;
    streamer->stop_requested.store(true, std::memory_order_release);
    if (streamer->worker.joinable()) streamer->worker.join();
}

AMP_API void amp_graph_streamer_destroy(AmpGraphStreamer *streamer) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_destroy", (size_t)(streamer != nullptr), 0);
    if (!streamer) return;
    amp_graph_streamer_stop(streamer);
    if (streamer->history_owned && streamer->history) {
        amp_release_control_history(streamer->history);
        streamer->history = nullptr;
        streamer->history_owned = false;
    }
    delete streamer;
}

AMP_API int amp_graph_streamer_available(AmpGraphStreamer *streamer, uint64_t *out_frames) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_available", (size_t)(streamer != nullptr), 0);
    if (!streamer || !out_frames) return -1;
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
    if (!streamer || !destination || max_frames == 0) return -1;

    size_t available = streamer_ring_size(streamer);
    if (available == 0) {
        if (out_frames)   *out_frames = 0;
        if (out_channels) *out_channels = streamer->channels;
        if (out_sequence) *out_sequence = streamer->read_index.load(std::memory_order_acquire);
        return 0;
    }

    size_t to_copy = std::min(max_frames, available);
    uint64_t start_index = streamer->read_index.load(std::memory_order_acquire);
    streamer_copy_from_ring(streamer, start_index, to_copy, destination);
    streamer->read_index.store(start_index + to_copy, std::memory_order_release);
    streamer->consumed_frames.fetch_add(to_copy, std::memory_order_release);
    streamer->last_consumed_millis.store(steady_now_millis(), std::memory_order_release);

    if (out_frames)   *out_frames = static_cast<uint32_t>(to_copy);
    if (out_channels) *out_channels = streamer->channels;
    if (out_sequence) *out_sequence = start_index;
    return 0;
}

AMP_API int amp_graph_streamer_dump_count(AmpGraphStreamer *streamer, uint32_t *out_count) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_dump_count", (size_t)(streamer != nullptr), 0);
    if (!streamer || !out_count) return -1;
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
    if (!streamer) return -1;

    StreamDumpChunk chunk;
    std::unique_lock<std::mutex> lock(streamer->dump_mutex);
    if (streamer->dump_queue.empty()) {
        if (out_frames)   *out_frames = 0;
        if (out_channels) *out_channels = streamer->channels;
        if (out_sequence) *out_sequence = streamer->produced_frames.load(std::memory_order_acquire);
        return 0;
    }
    chunk = std::move(streamer->dump_queue.front());
    streamer->dump_queue.pop_front();
    lock.unlock();
    streamer->last_dump_millis.store(steady_now_millis(), std::memory_order_release);

    size_t frames = chunk.frames;
    if (destination == nullptr || max_frames < frames) {
        lock.lock();
        streamer->dump_queue.push_front(std::move(chunk));
        lock.unlock();
        if (out_frames)   *out_frames   = static_cast<uint32_t>(frames);
        if (out_channels) *out_channels = chunk.channels;
        if (out_sequence) *out_sequence = chunk.sequence;
        return 1; // signal retry with bigger buffer
    }

    std::memcpy(destination, chunk.data.data(), frames * streamer->frame_stride * sizeof(double));
    streamer->consumed_frames.fetch_add(frames, std::memory_order_release);
    streamer->last_consumed_millis.store(steady_now_millis(), std::memory_order_release);

    if (out_frames)   *out_frames   = static_cast<uint32_t>(frames);
    if (out_channels) *out_channels = chunk.channels;
    if (out_sequence) *out_sequence = chunk.sequence;
    return 0;
}

AMP_API int amp_graph_streamer_status(
    AmpGraphStreamer *streamer,
    uint64_t *out_produced_frames,
    uint64_t *out_consumed_frames
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_status", (size_t)(streamer != nullptr), 0);
    if (!streamer) return -1;
    if (out_produced_frames) *out_produced_frames = streamer->produced_frames.load(std::memory_order_acquire);
    if (out_consumed_frames) *out_consumed_frames = streamer->consumed_frames.load(std::memory_order_acquire);
    return streamer->last_status;
}

AMP_API int amp_graph_streamer_evaluate_completion(
    AmpGraphStreamer *streamer,
    const AmpGraphStreamerCompletionContract *contract,
    AmpGraphStreamerCompletionState *out_state,
    AmpGraphStreamerCompletionVerdict *out_verdict
) {
    AMP_LOG_NATIVE_CALL("amp_graph_streamer_evaluate_completion", (size_t)(streamer != nullptr), 0);
    if (!streamer) return -1;

    const uint64_t produced = streamer->produced_frames.load(std::memory_order_acquire);
    const uint64_t consumed = streamer->consumed_frames.load(std::memory_order_acquire);
    const uint32_t ring_size = static_cast<uint32_t>(streamer_ring_size(streamer));
    const uint32_t ring_capacity = streamer->ring_frames;
    uint32_t dump_depth = 0U;
    {
        std::lock_guard<std::mutex> lock(streamer->dump_mutex);
        dump_depth = static_cast<uint32_t>(streamer->dump_queue.size());
    }

    const uint64_t now = steady_now_millis();
    const uint64_t start = streamer->start_time_millis.load(std::memory_order_acquire);
    const uint64_t elapsed = start > 0 ? elapsed_since(now, start) : 0;
    const uint64_t since_produced = elapsed_since(now, streamer->last_produced_millis.load(std::memory_order_acquire));
    const uint64_t since_consumed = elapsed_since(now, streamer->last_consumed_millis.load(std::memory_order_acquire));
    const uint64_t since_dump = elapsed_since(now, streamer->last_dump_millis.load(std::memory_order_acquire));

    if (out_state) {
        out_state->produced_frames = produced;
        out_state->consumed_frames = consumed;
        out_state->ring_size = ring_size;
        out_state->ring_capacity = ring_capacity;
        out_state->dump_queue_depth = dump_depth;
        out_state->elapsed_millis = elapsed;
        out_state->since_producer_progress_millis = since_produced;
        out_state->since_consumer_progress_millis = since_consumed;
        out_state->since_dump_progress_millis = since_dump;
        out_state->running = streamer->running.load(std::memory_order_acquire) ? 1 : 0;
    }

    if (out_verdict) {
        out_verdict->contract_satisfied = 0;
        out_verdict->producer_goal_met = 0;
        out_verdict->consumer_goal_met = 0;
        out_verdict->ring_drained = ring_size == 0 ? 1 : 0;
        out_verdict->dump_drained = dump_depth == 0 ? 1 : 0;
        out_verdict->timed_out = 0;
        out_verdict->idle_timeout_triggered = 0;
        out_verdict->total_timeout_triggered = 0;
        out_verdict->inflight_limit_exceeded = 0;
        out_verdict->dump_limit_exceeded = 0;

        const bool has_contract = contract != nullptr;
        const uint64_t target_produced = has_contract && contract->target_produced_frames > 0ULL
            ? contract->target_produced_frames
            : streamer->expected_output_frames;
        const uint64_t target_consumed = has_contract && contract->target_consumed_frames > 0ULL
            ? contract->target_consumed_frames
            : streamer->expected_output_frames;
        bool producer_goal_met = (target_produced == 0ULL) || (produced >= target_produced);
        bool consumer_goal_met = (target_consumed == 0ULL) || (consumed >= target_consumed);
        bool inflight_ok = true;
        bool dump_ok = true;
        bool require_ring_drain = has_contract ? (contract->require_ring_drain != 0)
                                               : (target_produced > 0ULL || target_consumed > 0ULL);
        bool require_dump_drain = has_contract ? (contract->require_dump_drain != 0)
                                               : (target_produced > 0ULL || target_consumed > 0ULL);
        bool idle_timeout_triggered = false;
        bool total_timeout_triggered = false;

        if (has_contract) {
            if (contract->maximum_inflight_frames > 0U && ring_size > contract->maximum_inflight_frames) {
                inflight_ok = false;
                out_verdict->inflight_limit_exceeded = 1;
            }
            if (contract->maximum_dump_depth > 0U && dump_depth > contract->maximum_dump_depth) {
                dump_ok = false;
                out_verdict->dump_limit_exceeded = 1;
            }
            if (contract->idle_timeout_millis > 0U) {
                const uint64_t idle_limit = contract->idle_timeout_millis;
                const bool producer_idle = since_produced >= idle_limit;
                const bool consumer_idle = since_consumed >= idle_limit;
                const bool dump_idle = since_dump >= idle_limit;
                idle_timeout_triggered = producer_idle && consumer_idle && dump_idle;
            }
            if (contract->total_timeout_millis > 0U && elapsed >= contract->total_timeout_millis) {
                total_timeout_triggered = true;
            }
        }

        const bool ring_drained = out_verdict->ring_drained == 1;
        const bool dump_drained = out_verdict->dump_drained == 1;

        bool contract_satisfied = producer_goal_met && consumer_goal_met && inflight_ok && dump_ok;
        if (require_ring_drain) {
            contract_satisfied = contract_satisfied && ring_drained;
        }
        if (require_dump_drain) {
            contract_satisfied = contract_satisfied && dump_drained;
        }

        const bool timed_out = idle_timeout_triggered || total_timeout_triggered;
        out_verdict->producer_goal_met = producer_goal_met ? 1 : 0;
        out_verdict->consumer_goal_met = consumer_goal_met ? 1 : 0;
        out_verdict->contract_satisfied = contract_satisfied ? 1 : 0;
        out_verdict->timed_out = timed_out ? 1 : 0;
        out_verdict->idle_timeout_triggered = idle_timeout_triggered ? 1 : 0;
        out_verdict->total_timeout_triggered = total_timeout_triggered ? 1 : 0;
    }

    return streamer->last_status;
}

AMP_API int amp_graph_runtime_debug_snapshot(
    AmpGraphRuntime *runtime,
    AmpGraphStreamer *streamer,
    AmpGraphNodeDebugEntry *node_entries,
    uint32_t node_capacity,
    AmpGraphDebugSnapshot *snapshot
) {
    if (!runtime || !snapshot) {
        return -1;
    }

    const uint32_t node_count = static_cast<uint32_t>(runtime->nodes.size());
    snapshot->version = 1U;
    snapshot->node_count = node_count;
    snapshot->sink_index = runtime->sink_index;
    snapshot->sample_rate = runtime->dsp_sample_rate;
    snapshot->scheduler_mode = static_cast<uint32_t>(runtime->scheduler_mode);
    snapshot->produced_frames = 0U;
    snapshot->consumed_frames = 0U;
    snapshot->ring_capacity = 0U;
    snapshot->ring_size = 0U;
    snapshot->dump_queue_depth = 0U;

    if (streamer != nullptr) {
        snapshot->produced_frames = streamer->produced_frames.load(std::memory_order_acquire);
        snapshot->consumed_frames = streamer->consumed_frames.load(std::memory_order_acquire);
        snapshot->ring_capacity = streamer->ring_frames;
        snapshot->ring_size = static_cast<uint32_t>(streamer_ring_size(streamer));
        {
            std::lock_guard<std::mutex> lock(streamer->dump_mutex);
            snapshot->dump_queue_depth = static_cast<uint32_t>(streamer->dump_queue.size());
        }
    }

    if (!node_entries || node_capacity < node_count) {
        return static_cast<int>(node_count);
    }

    for (uint32_t i = 0; i < node_count; ++i) {
        RuntimeNode &node = *runtime->nodes[i];
        AmpGraphNodeDebugEntry &entry = node_entries[i];
        std::memset(&entry, 0, sizeof(entry));
        std::snprintf(entry.name, sizeof(entry.name), "%.63s", node.name.c_str());
    entry.declared_delay_frames = node.declared_delay_frames;
    entry.oversample_ratio = node.oversample_ratio;
    entry.supports_v2 = node.supports_v2 ? 1U : 0U;
    entry.prefill_only = node.prefill_only ? 1U : 0U;
    entry.last_heat = node.has_latest_metrics ? node.latest_metrics.accumulated_heat : 0.0f;
    entry.last_processing_time_seconds = node.has_latest_metrics ? node.latest_metrics.processing_time_seconds : 0.0;
    entry.last_total_time_seconds = node.has_latest_metrics ? node.latest_metrics.total_time_seconds : 0.0;
    entry.total_heat_accumulated = node.total_heat_accumulated;
        const RuntimeNode::DebugFrameCache &cache = node.debug_frame_cache;
        uint32_t block_frames_hint = 0U;
        if (streamer && streamer->block_frames > 0U) {
            block_frames_hint = streamer->block_frames;
        } else if (runtime->default_frames > 0U) {
            block_frames_hint = runtime->default_frames;
        }
        BlockFrameContract contract = node_block_frame_contract(runtime, node, block_frames_hint);
        entry.debug_min_frames = contract.min_frames;
        entry.debug_preferred_frames = contract.preferred_frames;
        entry.debug_max_frames = contract.max_frames;
        entry.debug_priority_weight = contract.priority_weight;
        entry.debug_channel_expand = node.vector_policy.channel_expand;
        entry.fifo_simultaneous_availability = node.output_contract.simultaneous_availability ? 1U : 0U;
        entry.fifo_release_policy = static_cast<uint8_t>(node.output_contract.release_policy);
        entry.fifo_primary_consumer = node.output_contract.primary_consumer;
    entry.debug_sequence = cache.sequence;
    entry.debug_sample_count = cache.sample_count;
    entry.debug_total_frames = cache.total_frames;
    entry.debug_total_batches = cache.total_batches;
    entry.debug_total_channels = cache.total_channels;
    entry.debug_metrics_samples = cache.metrics_samples;
    entry.debug_last_timestamp_millis = cache.last_timestamp_millis;
    entry.debug_sum_processing_seconds = cache.sum_processing_seconds;
    entry.debug_sum_logging_seconds = cache.sum_logging_seconds;
    entry.debug_sum_total_seconds = cache.sum_total_seconds;
    entry.debug_sum_thread_cpu_seconds = cache.sum_thread_cpu_seconds;
    entry.debug_last_frames = cache.last_frames;
    entry.debug_last_batches = cache.last_batches;
    entry.debug_last_channels = cache.last_channels;
    entry.tap_count = 0U;

        uint32_t ring_capacity = 0U;
        uint32_t ring_size = 0U;
        uint32_t reader_count = 0U;
        const OutputTap *primary = runtime_primary_output(node);
        if (primary && primary->ring) {
            ring_capacity = primary->ring->capacity;
            reader_count = static_cast<uint32_t>(primary->ring->reader_tails.size());
            if (ring_capacity > 0U) {
                uint32_t head = primary->ring->head % ring_capacity;
                uint32_t tail = primary->ring->tail % ring_capacity;
                ring_size = edge_ring_distance(head, tail, ring_capacity);
            }
        }
        entry.ring_capacity = ring_capacity;
        entry.ring_size = ring_size;
        entry.reader_count = reader_count;

        for (const OutputTap &tap : node.outputs) {
            if (entry.tap_count >= AMP_GRAPH_NODE_MAX_TAPS) {
                break;
            }
            AmpGraphNodeTapDebugEntry &tap_entry = entry.taps[entry.tap_count++];
            std::string tap_name = tap.name.empty() ? std::string("default") : tap.name;
            std::snprintf(tap_entry.name, sizeof(tap_entry.name), "%.31s", tap_name.c_str());
            tap_entry.ring_capacity = 0U;
            tap_entry.ring_size = 0U;
            tap_entry.reader_count = 0U;
            tap_entry.head_position = 0U;
            tap_entry.tail_position = 0U;
            tap_entry.produced_total = 0U;
            if (!tap.ring) {
                continue;
            }
            tap_entry.ring_capacity = tap.ring->capacity;
            tap_entry.reader_count = static_cast<uint32_t>(tap.ring->reader_tails.size());
            if (tap_entry.ring_capacity > 0U) {
                uint32_t head = tap.ring->head % tap_entry.ring_capacity;
                uint32_t tail = tap.ring->tail % tap_entry.ring_capacity;
                tap_entry.head_position = head;
                tap_entry.tail_position = tail;
                tap_entry.ring_size = edge_ring_distance(head, tail, tap_entry.ring_capacity);
            }
            tap_entry.produced_total = tap.ring->produced_total;
        }
    }

    return static_cast<int>(node_count);
}

struct AmpKpnOverlay {
    AmpGraphStreamer *streamer{nullptr};
    AmpKpnOverlayConfig config{100U, 1, 0, 0};
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::thread worker;
    std::vector<AmpGraphNodeDebugEntry> buffer;
    bool cursor_hidden{false};
    double free_clock_hz{0.0};
    bool free_clock_initialized{false};
    uint64_t last_produced_frames{0ULL};
    std::chrono::steady_clock::time_point last_rate_sample{};
};

#if defined(_WIN32)
static void amp_kpn_overlay_enable_vt_mode() {
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (handle == INVALID_HANDLE_VALUE) {
        return;
    }
    DWORD mode = 0;
    if (!GetConsoleMode(handle, &mode)) {
        return;
    }
    if (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) {
        return;
    }
    SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
#else
static void amp_kpn_overlay_enable_vt_mode() {}
#endif

static void amp_kpn_overlay_write(FILE *stream, const char *data, size_t length) {
    if (!stream || !data || length == 0U) {
        return;
    }
    std::fwrite(data, 1U, length, stream);
}

static void amp_kpn_overlay_render(
    AmpKpnOverlay *overlay,
    const AmpGraphDebugSnapshot &snapshot,
    const AmpGraphNodeDebugEntry *entries
) {
    static constexpr double kDefaultEmaAlpha = 0.25; // smoothing factor for EMA; smaller = smoother
    static constexpr double kDefaultWindowSeconds = 1.0; // fallback window for simple average if EMA disabled
    static constexpr bool kUseEma = true;

    if (!overlay) return;
    FILE *out = stdout;
    if (!out) return;

    struct TapHistorySample {
        uint64_t produced_total{0ULL};
        std::chrono::steady_clock::time_point timestamp{};
        double ema_flow{0.0};
    };
    static std::unordered_map<std::string, TapHistorySample> tap_history;

    std::string buffer;
    buffer.reserve(2048U + static_cast<size_t>(snapshot.node_count) * 160U);
    buffer.append("\033[H\033[J");

    char line[256];
    double free_clock = 0.0;
    if (overlay->config.enable_free_clock != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (!overlay->free_clock_initialized) {
            overlay->last_rate_sample = now;
            overlay->last_produced_frames = snapshot.produced_frames;
            overlay->free_clock_hz = 0.0;
            overlay->free_clock_initialized = true;
        } else {
            uint64_t produced_delta = 0ULL;
            if (snapshot.produced_frames >= overlay->last_produced_frames) {
                produced_delta = snapshot.produced_frames - overlay->last_produced_frames;
            }
            double elapsed = std::chrono::duration<double>(now - overlay->last_rate_sample).count();
            if (elapsed > 0.0) {
                double instantaneous = produced_delta / elapsed;
                constexpr double alpha = 0.25;
                if (!std::isfinite(overlay->free_clock_hz) || overlay->free_clock_hz <= 0.0) {
                    overlay->free_clock_hz = instantaneous;
                } else {
                    overlay->free_clock_hz = alpha * instantaneous + (1.0 - alpha) * overlay->free_clock_hz;
                }
                overlay->last_rate_sample = now;
                overlay->last_produced_frames = snapshot.produced_frames;
            }
        }
        free_clock = overlay->free_clock_hz;
    }

    if (overlay->config.enable_free_clock != 0) {
        std::snprintf(
            line,
            sizeof(line),
            "AMP KPN Overlay | nodes:%u sink:%u | mode:%u | sr:%0.1f Hz | free:%0.2f Hz | ring:%u/%u | dumps:%u\n",
            snapshot.node_count,
            snapshot.sink_index,
            snapshot.scheduler_mode,
            snapshot.sample_rate,
            free_clock,
            snapshot.ring_size,
            snapshot.ring_capacity,
            snapshot.dump_queue_depth
        );
    } else {
        std::snprintf(
            line,
            sizeof(line),
            "AMP KPN Overlay | nodes:%u sink:%u | mode:%u | sr:%0.1f Hz | ring:%u/%u | dumps:%u\n",
            snapshot.node_count,
            snapshot.sink_index,
            snapshot.scheduler_mode,
            snapshot.sample_rate,
            snapshot.ring_size,
            snapshot.ring_capacity,
            snapshot.dump_queue_depth
        );
    }
    buffer.append(line);

    std::snprintf(
        line,
        sizeof(line),
        "Produced:%llu Consumed:%llu\n",
        static_cast<unsigned long long>(snapshot.produced_frames),
        static_cast<unsigned long long>(snapshot.consumed_frames)
    );
    buffer.append(line);
    buffer.append("--------------------------------------------------------------------------------\n");
    buffer.append("Node                             | Ring%  | Used/Cap | Delay | Heat  | AvgProc(ms) | AvgTotal(ms) | Min/Pref | Max  | Calls | Frames | LastF | LastB | LastC\n");

    const auto now = std::chrono::steady_clock::now();
    std::unordered_set<std::string> seen_taps;
    seen_taps.reserve(static_cast<size_t>(snapshot.node_count) * 2U);

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
        std::snprintf(
            line,
            sizeof(line),
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
        buffer.append(line);

        if (entry.tap_count == 0U) {
            continue;
        }

        buffer.append("        Tap                      | Ring%  | Used/Cap | Head/Tail | Readers | Flow(fr/s)\n");
        for (uint32_t tap_idx = 0; tap_idx < entry.tap_count; ++tap_idx) {
            const AmpGraphNodeTapDebugEntry &tap = entry.taps[tap_idx];
            double tap_percent = 0.0;
            if (tap.ring_capacity > 0U) {
                tap_percent = (100.0 * static_cast<double>(tap.ring_size)) / static_cast<double>(tap.ring_capacity);
            }

            double flow_rate = 0.0;
            std::string key = node_name;
            key.append("::").append(tap.name);
            auto it = tap_history.find(key);
            if (it != tap_history.end()) {
                double elapsed = std::chrono::duration<double>(now - it->second.timestamp).count();
                if (elapsed > 0.0 && tap.produced_total >= it->second.produced_total) {
                    uint64_t delta = tap.produced_total - it->second.produced_total;
                    double instantaneous = static_cast<double>(delta) / elapsed;
                    if (kUseEma) {
                        double alpha = kDefaultEmaAlpha;
                        double ema = (alpha * instantaneous) + ((1.0 - alpha) * it->second.ema_flow);
                        flow_rate = ema;
                    } else {
                        double window = kDefaultWindowSeconds;
                        if (window <= 0.0) window = elapsed;
                        flow_rate = static_cast<double>(delta) / std::max(window, elapsed);
                    }
                }
            }
            TapHistorySample sample{};
            sample.produced_total = tap.produced_total;
            sample.timestamp = now;
            if (kUseEma) {
                if (it != tap_history.end()) {
                    double ema_prev = it->second.ema_flow;
                    double alpha = kDefaultEmaAlpha;
                    double ema = (flow_rate > 0.0) ? flow_rate : ema_prev;
                    sample.ema_flow = ema;
                } else {
                    sample.ema_flow = flow_rate;
                }
            } else {
                sample.ema_flow = flow_rate;
            }
            tap_history[key] = sample;

            seen_taps.insert(key);

            std::snprintf(
                line,
                sizeof(line),
                "        %-24.24s | %6.2f | %5u/%-5u | %4u/%-4u | %7u | %9.2f\n",
                tap.name[0] != '\0' ? tap.name : "default",
                tap_percent,
                tap.ring_size,
                tap.ring_capacity,
                tap.head_position,
                tap.tail_position,
                tap.reader_count,
                flow_rate
            );
            buffer.append(line);
        }
    }

    if (!tap_history.empty()) {
        for (auto it = tap_history.begin(); it != tap_history.end();) {
            if (seen_taps.find(it->first) == seen_taps.end()) {
                it = tap_history.erase(it);
            } else {
                ++it;
            }
        }
    }

    amp_kpn_overlay_write(out, buffer.data(), buffer.size());
    std::fflush(out);
}

static void amp_kpn_overlay_thread(AmpKpnOverlay *overlay) {
    if (!overlay || !overlay->streamer) {
        return;
    }

    amp_kpn_overlay_enable_vt_mode();

    FILE *out = stdout;
    if (out) {
        std::fputs("\033[?25l", out);
        std::fflush(out);
        overlay->cursor_hidden = true;
    }

    overlay->running.store(true, std::memory_order_release);

    const uint32_t sleep_ms = overlay->config.refresh_millis > 0U ? overlay->config.refresh_millis : 100U;

    while (!overlay->stop_requested.load(std::memory_order_acquire)) {
        AmpGraphDebugSnapshot snapshot{};
        uint32_t capacity = static_cast<uint32_t>(overlay->buffer.size());
        int rc = amp_graph_runtime_debug_snapshot(
            overlay->streamer->runtime,
            overlay->streamer,
            overlay->buffer.data(),
            capacity,
            &snapshot
        );
        if (rc < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            continue;
        }
        if (static_cast<uint32_t>(rc) > capacity) {
            overlay->buffer.resize(static_cast<size_t>(rc));
            continue;
        }

        amp_kpn_overlay_render(overlay, snapshot, overlay->buffer.data());
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }

    if (overlay->cursor_hidden && out) {
        if (overlay->config.clear_on_exit != 0) {
            std::fputs("\033[?25h\033[H\033[J", out);
        } else {
            std::fputs("\033[?25h", out);
        }
        std::fflush(out);
        overlay->cursor_hidden = false;
    }

    overlay->running.store(false, std::memory_order_release);
}

AMP_API AmpKpnOverlay *amp_kpn_overlay_create(
    AmpGraphStreamer *streamer,
    const AmpKpnOverlayConfig *config
) {
    if (!streamer) {
        return nullptr;
    }
    auto *overlay = new (std::nothrow) AmpKpnOverlay();
    if (!overlay) {
        return nullptr;
    }
    overlay->streamer = streamer;
    if (config) {
        overlay->config = *config;
    }
    if (overlay->config.refresh_millis == 0U) {
        overlay->config.refresh_millis = 100U;
    }
    overlay->buffer.resize(32U);
    return overlay;
}

AMP_API int amp_kpn_overlay_start(AmpKpnOverlay *overlay) {
    if (!overlay) {
        return -1;
    }
    if (overlay->running.load(std::memory_order_acquire)) {
        return 0;
    }
    if (overlay->worker.joinable()) {
        overlay->worker.join();
    }
    overlay->stop_requested.store(false, std::memory_order_release);
    overlay->worker = std::thread(amp_kpn_overlay_thread, overlay);
    return 0;
}

AMP_API void amp_kpn_overlay_stop(AmpKpnOverlay *overlay) {
    if (!overlay) {
        return;
    }
    overlay->stop_requested.store(true, std::memory_order_release);
    if (overlay->worker.joinable()) {
        overlay->worker.join();
    }
    overlay->stop_requested.store(false, std::memory_order_release);
}

AMP_API void amp_kpn_overlay_destroy(AmpKpnOverlay *overlay) {
    if (!overlay) {
        return;
    }
    amp_kpn_overlay_stop(overlay);
    delete overlay;
}

} // extern "C"
