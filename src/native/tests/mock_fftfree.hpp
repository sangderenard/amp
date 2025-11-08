#ifndef AMP_TESTS_MOCK_FFTFREE_HPP
#define AMP_TESTS_MOCK_FFTFREE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mock_fftfree {

struct BatchCall {
    void *handle{nullptr};
    std::vector<float> pcm;
    std::size_t samples{0};
    std::size_t frames{0};
    int pad_mode{0};
    int enable_backup{0};
};

struct ComplexCall {
    void *handle{nullptr};
    std::vector<float> real;
    std::vector<float> imag;
    std::size_t frames{0};
    int pad_mode{0};
    int enable_backup{0};
    std::size_t lanes{0};
};

struct StreamCall {
    void *handle{nullptr};
    std::vector<float> pcm;
    std::size_t samples{0};
    std::size_t max_frames{0};
    int flush_mode{0};
    std::size_t produced{0};
};

struct ContextState {
    int n{0};
    int window{0};
    int hop{0};
    bool inverse{false};
    int stft_mode{0};
    bool apply_windows{false};
    int analysis_window{0};
    int synthesis_window{0};
    std::size_t pending_samples{0};
};

void reset_all();
void clear_call_history();

const std::vector<BatchCall> &batch_calls();
const std::vector<ComplexCall> &complex_calls();
const std::vector<StreamCall> &stream_calls();

int init_full_call_count();
int free_call_count();

ContextState describe_handle(void *handle);

}  // namespace mock_fftfree

#endif  // AMP_TESTS_MOCK_FFTFREE_HPP
