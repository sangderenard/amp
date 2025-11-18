#pragma once

#include <stddef.h>
#include <stdint.h>

static inline int64_t fftdiv_delay_positive_or_default_int64(int64_t value, int64_t fallback) {
    return (value > 0) ? value : fallback;
}

static inline int64_t fftdiv_delay_compute_l_istft(int64_t window_fft, int64_t hop_fft, int64_t hop_working) {
    if (window_fft <= hop_fft) {
        return 0;
    }
    if (hop_fft <= 0) {
        hop_fft = 1;
    }
    if (hop_working <= 0) {
        hop_working = 1;
    }
    const int64_t working_hop_pcm = hop_working * hop_fft;
    if (working_hop_pcm <= 0) {
        return 0;
    }
    const int64_t tail_samples = window_fft - hop_fft;
    return (tail_samples + working_hop_pcm - 1) / working_hop_pcm;
}

static inline int64_t fftdiv_delay_compute_t_max(
    int64_t sample_index,
    int64_t window_fft,
    int64_t hop_fft,
    int64_t hop_working,
    int64_t istft_window,
    int64_t l_istft_override
) {
    if (sample_index < 0) {
        sample_index = 0;
    }
    if (window_fft <= 0) {
        window_fft = 1;
    }
    if (istft_window <= 0) {
        istft_window = window_fft;
    }
    hop_fft = fftdiv_delay_positive_or_default_int64((int)hop_fft, 1);
    hop_working = fftdiv_delay_positive_or_default_int64((int)hop_working, 1);

    int64_t l_istft = l_istft_override;
    if (l_istft < 0) {
        l_istft = fftdiv_delay_compute_l_istft(window_fft, hop_fft, hop_working);
    }

    // k*(n0) = floor(n0 / H_fft)
    const int64_t k_star = sample_index / hop_fft;
    // j*(n0) = floor(k*(n0) / H_work)
    const int64_t j_star = k_star / hop_working;
    // i*(n0) = j*(n0) + L_istft - 1
    const int64_t i_star = j_star + l_istft - 1;
    // t_max(n0) = i*(n0)*H_fft + W_istft - 1
    const int64_t t_max = i_star * hop_fft + istft_window - 1;
    return (t_max > sample_index) ? t_max : sample_index;
}

static inline size_t fftdiv_delay_frames_for_sample(
    size_t sample_index,
    int window_fft,
    int hop_fft,
    int hop_working,
    int istft_window,
    int l_istft_override
) {
    const int64_t t_max = fftdiv_delay_compute_t_max(
        (int64_t)sample_index,
        fftdiv_delay_positive_or_default_int64(window_fft, 1),
        fftdiv_delay_positive_or_default_int64(hop_fft, 1),
        fftdiv_delay_positive_or_default_int64(hop_working, 1),
        fftdiv_delay_positive_or_default_int64(istft_window, 1),
        (int64_t)l_istft_override
    );
    const int64_t delay = t_max - (int64_t)sample_index;
    return (delay > 0) ? (size_t)delay : 0U;
}

static inline size_t fftdiv_delay_frames_for_signal_length(
    size_t total_samples,
    int window_fft,
    int hop_fft,
    int hop_working,
    int istft_window,
    int l_istft_override
) {
    if (total_samples == 0) {
        return 0U;
    }
    const size_t last_sample = total_samples - 1U;
    return fftdiv_delay_frames_for_sample(
        last_sample,
        window_fft,
        hop_fft,
        hop_working,
        istft_window,
        l_istft_override
    );
}
