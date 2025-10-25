# NUFFT Mode Specification

This document describes the production requirements for the nonuniform FFT/DFT (NUFFT) mode within the FFT division node. The intent is to extend the existing native C/C++ runtime without introducing any Python fallbacks, keeping the node interchangeable with the present FFT, DFT, and dynamic oscillator algorithms.

## 1. Mode Integration and Compatibility
- **Algorithm selector** – Add `"nufft"` to the node's algorithm options alongside `"fft"`, `"dft"`, and `"dynamic"`. Selection is driven by the JSON descriptor (for example, `"algorithm": "nufft"`) and must reuse the established KPN interface.
- **ABI stability** – The NUFFT branch honours the common node contract: batched `(batch, channel, frame)` I/O, no shape changes, identical ports and parameters, and compliance with `amp_run_node` / `amp_run_node_v2` ABI expectations.
- **Native-only execution** – All NUFFT processing runs inside the native runtime. No Python fallback paths are allowed.

## 2. Windowing and Warm-Up Behaviour
- **Window application** – Respect the configured `window_size` and window type on both the input signal and divisor taps before frequency-domain transforms. Handle invalid sizes as in the FFT mode (clamp to one frame, apply the epsilon stabiliser to divisor magnitudes).
- **Warm-up** – Reuse the safe-division warm-up phase: until a slot accumulates a full window of samples, emit passthrough divided output with an epsilon floor on the divisor magnitude to avoid transients.

## 3. Nonuniform Transform Core (Forward Path)
- **Oversampled grid** – Grid the windowed signal and divisor onto an oversampled frequency lattice using a Kaiser–Bessel convolution kernel. Oversampling factor `σ` (≈1.5–2.0) controls zero padding; kernel half-width `m` (6–8) governs interpolation accuracy.
- **FFT and deapodisation** – Perform a fast FFT on the oversampled grid, then deapodise by dividing by the kernel frequency response to recover a high-resolution spectrum. The procedure degenerates to the standard FFT/DFT path when `σ = 1` and no interpolation kernel is used.
- **Frequency-domain division** – Divide the signal spectrum by the divisor spectrum on a per-bin basis, adding epsilon to the divisor magnitude squared for stability.

## 4. Band Selection and Phase Control
- **Band gating** – Apply the existing `lower_band`, `upper_band`, and `filter_intensity` parameters to gate the NUFFT spectrum. Bins inside the band receive the inside gain; outside bins receive the complementary gain, matching the FFT/DFT logic but benefitting from finer bin spacing.
- **Phase rotation** – After gating and deapodisation, rotate all bins by the per-frame `phase_offset` exactly as in the FFT mode.

## 5. Time-Domain Reconstruction and Continuity
- **Inverse NUFFT** – Reconstruct output frames via the adjoint NUFFT: apply the conjugate Kaiser–Bessel window, run an inverse FFT on the oversampled grid, and degrid/interpolate back to the native sample rate.
- **Continuity** – Maintain frame-to-frame $C^0$ continuity. Manage per-bin phase state (e.g., cumulative `θ[k]`) so slow parameter sweeps remain artefact-free and align with the dynamic oscillator mode's smoothness.

## 6. Backward (Adjoint) Execution Support
- **Adjoint symmetry** – Implement the backward pass as the exact adjoint of the forward NUFFT pipeline. Undo the phase rotation, apply band gating, project gradients onto the oversampled lattice using the same kernel, perform FFT + deapodisation, multiply by the cached divisor spectrum, and finish with an inverse NUFFT to recover input/divisor gradients.
- **State and metrics** – Mirror forward-mode state handling. Report latency and thermal metrics through `AmpNodeMetrics` during backward execution just as in forward mode.

## 7. Parameter Controls and Quality Tuning
- **NUFFT parameters** – Expose tunables such as oversampling ratio `sigma`, kernel half-width `m`, Kaiser–Bessel shape `kb_beta`, and optional grid padding. Allow JSON configuration or provide sensible defaults (e.g., `sigma = 2.0`, `m = 6`). Precompute kernel lookup tables shared by forward and backward paths.
- **FFT compatibility** – Ensure the NUFFT reduces to the standard FFT/DFT behaviour when `sigma = 1.0`, minimal kernel width is selected, and target frequencies align with the uniform grid.

## 8. Performance and Testing Expectations
- **Real-time constraints** – Optimise for real-time execution using vectorised math, preallocated scratch buffers, and accurate latency reporting. Accumulate thermal metrics that reflect NUFFT workloads.
- **Validation** – Provide tests that compare NUFFT results to direct DFT ground truth, verify smooth output for slow sweeps, handle extreme frequencies and gating changes without artefacts, and confirm the adjoint via gradient checks (e.g., vector–Jacobian dot products).

## References
- [FFT Division Node Audit](fft_division_node_audit.md)
- [NUFFT Continuity Audit](nufft_continuity_audit.md)
