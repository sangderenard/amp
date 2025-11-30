#pragma once

#include <stddef.h>
#include <stdint.h>

static inline int64_t fftdiv_delay_positive_or_default_int64(int64_t value, int64_t fallback) {
    return (value > 0) ? value : fallback;
}

/*
this inline is only used in one place right? lets get rid of it and just write inside the thing that has all the data what the actual l_istft is, lets just check how many things have been submitted by the sample index, if the pipeline has not filled istft add spectral frames according to the stft / istft overlap/hop and if it is filled, or partially filled, work out how many spectral samples between 1 and the full amount minus 1
*/



static inline int64_t fftdiv_delay_compute_t_max(
    int64_t sample_index,
    int64_t window_fft,
    int64_t hop_fft,
    int64_t hop_working,
    int64_t istft_window,
    int64_t l_istft_override,
    int64_t working_span /* number of working frames the active window spans */
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
    int64_t ola_factor = (istft_window + hop_fft - 1) / hop_fft; // integer ceil    
    if (l_istft < 0) {
        l_istft = (working_span - 1)* hop_fft + ola_factor * hop_fft;
    }
    else {
        l_istft += ola_factor * hop_fft;
    }

    
    const int64_t stft_remainder = sample_index % hop_fft;
    const int64_t working_hop_samples = hop_working * hop_fft;
    int64_t extra_working_hops = 0;
    if (working_span > 1) {
        // integer ceil: (a + b - 1) / b
        extra_working_hops = (working_span - 1 + hop_working - 1) / hop_working;
    }
    else {
        extra_working_hops = 0;
    }
    const int64_t extra_spectral_output = (extra_working_hops > 0) ? extra_working_hops * hop_working * hop_fft - (working_span - 1) : 0;
    l_istft -= extra_spectral_output * hop_fft;
    return extra_working_hops * working_hop_samples + l_istft - stft_remainder;
    
}

static inline size_t fftdiv_delay_frames_for_sample(
    size_t sample_index,
    int window_fft,
    int hop_fft,
    int hop_working,
    int istft_window,
    int l_istft_override,
    int working_span /* working-frame units */
) {
    const int64_t t_max = fftdiv_delay_compute_t_max(
        (int64_t)sample_index,
        fftdiv_delay_positive_or_default_int64(window_fft, 1),
        fftdiv_delay_positive_or_default_int64(hop_fft, 1),
        fftdiv_delay_positive_or_default_int64(hop_working, 1),
        fftdiv_delay_positive_or_default_int64(istft_window, 1),
        (int64_t)l_istft_override,
        (int64_t)working_span
    );
    const int64_t delay = t_max; //- (int64_t)sample_index;
    return (delay > 0) ? (size_t)delay : 0U;
}

static inline size_t fftdiv_delay_frames_for_signal_length(
    size_t total_samples,
    int window_fft,
    int hop_fft,
    int hop_working,
    int istft_window,
    int l_istft_override,
    int working_span /* working-frame units */
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
        l_istft_override,
        working_span
    );
}
