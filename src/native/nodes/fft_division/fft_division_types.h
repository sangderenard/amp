// Forward declaration for pointer use
#if defined(__cplusplus)
struct AmpSpectralMailboxEntry;
struct FftDivTapMailboxCursor {
    const AmpSpectralMailboxEntry *read_cursor = nullptr;
    int last_frame_index = -1;
};
#endif
#pragma once

#if defined(__cplusplus)
#include <cstddef>
#include <cstdint>
#include <vector>

struct FftDivLaneFrameState {
    bool frame_ready{false};
    bool expect_signal{false};
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
};

struct FftDivFilledSlice {
    int tensor_slice{0};
    int scratch_slice{0};
    int view_filled_override{0};
    int wheel_head{0};
    int wheel_tail{0};
    int64_t frame_index{0};
    int64_t pcm_sample_index{0};
    size_t slice_index{0U};
    double timeline_seconds{0.0};
    bool working_tensor_updated{false};
    bool valid{false};
    bool stage4_emitted{false};
    std::vector<FftDivLaneFrameState> lanes;
    std::vector<size_t> lane_frame_offsets;
};

#endif /* defined(__cplusplus) */
