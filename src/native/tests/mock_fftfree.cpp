#include "mock_fftfree.hpp"
#include "fftfree/fft_cffi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mock_fftfree {

namespace {

struct FakeContext {
    int n{0};
    int window{0};
    int hop{0};
    bool inverse{false};
    int stft_mode{0};
    bool apply_windows{false};
    int analysis_window{0};
    int synthesis_window{0};
    std::vector<float> pending;
};

std::unordered_map<void *, std::unique_ptr<FakeContext>> g_contexts;
std::vector<BatchCall> g_batch_calls;
std::vector<ComplexCall> g_complex_calls;
std::vector<StreamCall> g_stream_calls;
int g_init_full_calls = 0;
int g_free_calls = 0;

FakeContext *as_context(void *handle) {
    auto it = g_contexts.find(handle);
    return it == g_contexts.end() ? nullptr : it->second.get();
}

int normalised_hop(const FakeContext &ctx) {
    if (ctx.hop > 0) {
        return ctx.hop;
    }
    if (ctx.window > 0) {
        return ctx.window;
    }
    return ctx.n > 0 ? ctx.n : 0;
}

int frame_length(const FakeContext &ctx) {
    if (ctx.n > 0) {
        return ctx.n;
    }
    if (ctx.window > 0) {
        return ctx.window;
    }
    return 0;
}

}  // namespace

void reset_all() {
    g_batch_calls.clear();
    g_complex_calls.clear();
    g_stream_calls.clear();
    g_init_full_calls = 0;
    g_free_calls = 0;
}

void clear_call_history() {
    g_batch_calls.clear();
    g_complex_calls.clear();
    g_stream_calls.clear();
}

const std::vector<BatchCall> &batch_calls() { return g_batch_calls; }
const std::vector<ComplexCall> &complex_calls() { return g_complex_calls; }
const std::vector<StreamCall> &stream_calls() { return g_stream_calls; }

int init_full_call_count() { return g_init_full_calls; }
int free_call_count() { return g_free_calls; }

ContextState describe_handle(void *handle) {
    ContextState state{};
    if (auto *ctx = as_context(handle)) {
        state.n = ctx->n;
        state.window = ctx->window;
        state.hop = ctx->hop;
        state.inverse = ctx->inverse;
        state.stft_mode = ctx->stft_mode;
        state.apply_windows = ctx->apply_windows;
        state.analysis_window = ctx->analysis_window;
        state.synthesis_window = ctx->synthesis_window;
        state.pending_samples = ctx->pending.size();
    }
    return state;
}

}  // namespace mock_fftfree

extern "C" {

void *fft_init_full_v2(
    std::size_t n,
    int threads,
    int lanes,
    int inverse,
    int kernel,
    int radix,
    const void *plan,
    std::size_t plan_bytes,
    int pad_mode,
    int window,
    int hop,
    int stft_mode,
    int transform,
    int phase_mode,
    int reorder,
    int real_policy,
    int freq_policy,
    int time_policy,
    int reserved0,
    int reserved1,
    int silent_crash_reports,
    int apply_windows,
    int apply_ola,
    int analysis_window,
    float analysis_alpha,
    float analysis_beta,
    int synthesis_window,
    float synthesis_alpha,
    float synthesis_beta,
    int window_norm,
    int cola_mode
) {
    (void)threads;
    (void)lanes;
    (void)kernel;
    (void)radix;
    (void)plan;
    (void)plan_bytes;
    (void)pad_mode;
    (void)transform;
    (void)phase_mode;
    (void)reorder;
    (void)real_policy;
    (void)freq_policy;
    (void)time_policy;
    (void)reserved0;
    (void)reserved1;
    (void)silent_crash_reports;
    (void)apply_ola;
    (void)analysis_alpha;
    (void)analysis_beta;
    (void)synthesis_alpha;
    (void)synthesis_beta;
    (void)window_norm;
    (void)cola_mode;

    auto context = std::make_unique<mock_fftfree::FakeContext>();
    context->n = static_cast<int>(n);
    context->window = window;
    context->hop = hop;
    context->inverse = inverse != 0;
    context->stft_mode = stft_mode;
    context->apply_windows = apply_windows != 0;
    context->analysis_window = analysis_window;
    context->synthesis_window = synthesis_window;
    context->pending.clear();
    void *handle = context.get();
    mock_fftfree::g_contexts.emplace(handle, std::move(context));
    ++mock_fftfree::g_init_full_calls;
    return handle;
}

void fft_free(void *handle) {
    if (handle == nullptr) {
        return;
    }
    mock_fftfree::g_contexts.erase(handle);
    ++mock_fftfree::g_free_calls;
}

std::size_t fft_execute_batched(
    void *handle,
    const float *pcm,
    std::size_t samples,
    float *spec_real,
    float *spec_imag,
    float *spec_mag,
    int pad_mode,
    int enable_backup,
    std::size_t frames
) {
    auto call = mock_fftfree::BatchCall{};
    call.handle = handle;
    call.samples = samples;
    call.frames = frames;
    call.pad_mode = pad_mode;
    call.enable_backup = enable_backup;
    if (pcm != nullptr && samples > 0) {
        call.pcm.assign(pcm, pcm + samples);
    }
    if (spec_real != nullptr && pcm != nullptr) {
        std::copy_n(pcm, samples, spec_real);
    }
    if (spec_imag != nullptr && samples > 0) {
        std::fill_n(spec_imag, samples, 0.0f);
    }
    if (spec_mag != nullptr && samples > 0) {
        for (std::size_t i = 0; i < samples; ++i) {
            spec_mag[i] = pcm != nullptr ? std::fabs(pcm[i]) : 0.0f;
        }
    }
    mock_fftfree::g_batch_calls.push_back(std::move(call));
    return frames;
}

std::size_t fft_execute_complex_batched(
    void *handle,
    const float *input_real,
    const float *input_imag,
    std::size_t frames,
    float *pcm_out,
    int pad_mode,
    int enable_backup,
    std::size_t lanes
) {
    auto call = mock_fftfree::ComplexCall{};
    call.handle = handle;
    call.frames = frames;
    call.pad_mode = pad_mode;
    call.enable_backup = enable_backup;
    call.lanes = lanes;
    std::size_t samples = 0;
    if (auto *ctx = mock_fftfree::as_context(handle)) {
        const int frame_len = frame_length(*ctx);
        if (frame_len > 0) {
            samples = static_cast<std::size_t>(frame_len) * frames;
        }
    }
    if (samples == 0 && frames > 0 && lanes > 0) {
        samples = frames * lanes;
    }
    if (input_real != nullptr && samples > 0) {
        call.real.assign(input_real, input_real + samples);
        if (pcm_out != nullptr) {
            std::copy_n(input_real, samples, pcm_out);
        }
    }
    if (input_imag != nullptr && samples > 0) {
        call.imag.assign(input_imag, input_imag + samples);
    }
    mock_fftfree::g_complex_calls.push_back(std::move(call));
    return frames;
}

void *fft_init_ex(
    int n,
    int threads,
    int lanes,
    int inverse,
    int kernel,
    int radix,
    const void *plan,
    std::size_t plan_bytes,
    int pad_mode,
    int window,
    int hop,
    int stft_mode
) {
    return fft_init_full_v2(
        static_cast<std::size_t>(n),
        threads,
        lanes,
        inverse,
        kernel,
        radix,
        plan,
        plan_bytes,
        pad_mode,
        window,
        hop,
        stft_mode,
        FFT_TRANSFORM_C2C,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        FFT_WINDOW_RECT,
        0.0f,
        0.0f,
        FFT_WINDOW_RECT,
        0.0f,
        0.0f,
        FFT_WINDOW_NORM_NONE,
        FFT_COLA_OFF
    );
}

void fft_stream_reset(void *handle) {
    if (auto *ctx = mock_fftfree::as_context(handle)) {
        ctx->pending.clear();
    }
}

std::size_t fft_stream_push_pcm(
    void *handle,
    const float *pcm,
    std::size_t samples,
    float *out_real,
    float *out_imag,
    float *out_mag,
    std::size_t max_frames,
    int flush_mode
) {
    auto call = mock_fftfree::StreamCall{};
    call.handle = handle;
    call.samples = samples;
    call.max_frames = max_frames;
    call.flush_mode = flush_mode;
    if (pcm != nullptr && samples > 0) {
        call.pcm.assign(pcm, pcm + samples);
    }

    auto *ctx = mock_fftfree::as_context(handle);
    if (ctx == nullptr) {
        mock_fftfree::g_stream_calls.push_back(std::move(call));
        return 0;
    }

    if (pcm != nullptr) {
        ctx->pending.insert(ctx->pending.end(), pcm, pcm + samples);
    }

    const int frame_len = frame_length(*ctx);
    const int hop = normalised_hop(*ctx);
    if (frame_len <= 0 || hop <= 0) {
        mock_fftfree::g_stream_calls.push_back(std::move(call));
        return 0;
    }

    std::size_t produced = 0;
    while (produced < max_frames && ctx->pending.size() >= static_cast<std::size_t>(frame_len)) {
        const std::size_t offset = produced * static_cast<std::size_t>(frame_len);
        if (out_real != nullptr) {
            std::copy_n(ctx->pending.begin(), frame_len, out_real + offset);
        }
        if (out_imag != nullptr) {
            for (int i = 0; i < frame_len; ++i) {
                out_imag[offset + static_cast<std::size_t>(i)] = -ctx->pending[static_cast<std::size_t>(i)];
            }
        }
        if (out_mag != nullptr) {
            for (int i = 0; i < frame_len; ++i) {
                out_mag[offset + static_cast<std::size_t>(i)] = std::fabs(ctx->pending[static_cast<std::size_t>(i)]);
            }
        }
        ++produced;
        const std::size_t remove = static_cast<std::size_t>(std::min<int>(hop, frame_len));
        ctx->pending.erase(ctx->pending.begin(), ctx->pending.begin() + static_cast<std::ptrdiff_t>(remove));
    }
    call.produced = produced;
    mock_fftfree::g_stream_calls.push_back(std::move(call));
    return produced;
}

std::size_t fft_stream_pending_frames(void *handle) {
    if (auto *ctx = mock_fftfree::as_context(handle)) {
        const int hop = normalised_hop(*ctx);
        const int frame_len = frame_length(*ctx);
        if (hop <= 0 || frame_len <= 0) {
            return 0;
        }
        if (ctx->pending.size() < static_cast<std::size_t>(frame_len)) {
            return 0;
        }
        const std::size_t available = ctx->pending.size() - static_cast<std::size_t>(frame_len) + static_cast<std::size_t>(hop);
        return available / static_cast<std::size_t>(hop);
    }
    return 0;
}

std::size_t fft_stream_backlog_samples(void *handle) {
    if (auto *ctx = mock_fftfree::as_context(handle)) {
        return ctx->pending.size();
    }
    return 0;
}

}  // extern "C"
