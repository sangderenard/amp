# FFTDivisionNode Implementation Audit

## Scope and intent
This note captures the observable behaviour of the native `FFTDivisionNode` as currently
implemented in `src/native/amp_kernels.c`. It focuses on the forward and backward execution
paths, the state that is persisted between firings, and the specific mechanism used to realise
arbitrary bin selection within a finite FFT window. All findings below come from the native C
runtime; no Python fallbacks are provided or allowed under project policy.

## Node surface and state
* **Window and stabiliser configuration** – Each dispatch resolves a JSON-declared
  `window_size`, per-frame `epsilon` stabiliser, and default FFT/window algorithms before
  touching the work buffers. Non-positive window sizes are clamped to one frame, epsilon values
  are floored to `1e-12`, and a non power-of-two size automatically falls back to the slower
  direct DFT path.【F:src/native/amp_kernels.c†L4385-L4574】
* **Algorithm scaffolding** – The selector now recognises `"nufft"`, `"czt"`, and
  `"dynamic"` (dynamic oscillator synthesis) in addition to the existing radix-2 FFT and
  direct DFT options. The new choices currently reuse the DFT kernels but keep their enum value
  alive in state/metrics so the specialised implementations can be dropped in later without
  altering the call surface.【F:src/native/amp_kernels.c†L2759-L2839】【F:src/native/amp_kernels.c†L3049-L3155】【F:src/native/tests/test_fft_division_node.cpp†L312-L360】
* **Dynamic carrier summary** – When the dynamic oscillator algorithm is selected the node scans
  for `carrier_band_{index}` parameter streams, records how many bands were provided, and stores a
  simple aggregate so future oscillator nodes can contribute per-band carriers without changing
  the runtime interface.【F:src/native/amp_kernels.c†L2759-L2839】【F:src/native/amp_kernels.c†L4385-L4574】
* **Parameter inputs** – The node consumes complex divisor taps, optional algorithm/window
  selectors, per-frame stabilisers, phase offsets, and band bounds/intensity controls, all
  retrieved through `EdgeRunnerParamView` wrappers. Slots correspond to `batches × channels` and
  each slot owns its own rolling window in the state struct.【F:src/native/amp_kernels.c†L4174-L4200】【F:src/native/amp_kernels.c†L4317-L4333】
* **State buffers** – `ensure_fft_state_buffers` guarantees contiguous per-slot rings for the
  input signal, complex divisors, phase/band metadata, and scratch FFT workspaces. The node keeps
  the most recent phase/band/filter scalars in `state->u.fftdiv.last_*` so metrics can surface
  what the synthesis path actually used.【F:src/native/amp_kernels.c†L4162-L4296】【F:src/native/amp_kernels.c†L4389-L4409】

## Forward execution behaviour
1. **Warm-up and safe division** – Until the window is filled, each slot outputs the input sample
   divided by a stabilised divisor (real input plus floor on magnitude). This prevents runaway
   gain during startup while waiting for enough history.【F:src/native/amp_kernels.c†L4297-L4308】
2. **Windowed FFT pair** – Once primed, the node applies the selected window to both the signal
   ring buffer and the complex divisor taps, then executes either the radix-2 FFT or the direct
   DFT for both sequences.【F:src/native/amp_kernels.c†L4317-L4338】
3. **Per-bin complex division** – For every frequency bin, the signal FFT is divided by the
   divisor FFT with an epsilon floor on the magnitude squared. The raw quotient is then routed to
   the arbitrary binning stage described below.【F:src/native/amp_kernels.c†L4354-L4367】
4. **Recombination and latency** – The inverse FFT of the gated spectrum is written back to the
   state scratch buffer, and the newest time-domain sample (last index of the IFFT) becomes the
   node’s output for that slot. The node records `position = window_size - 1`, matching the
   declared latency reported through metrics.【F:src/native/amp_kernels.c†L4376-L4387】【F:src/native/tests/test_fft_division_node.cpp†L598-L666】

## Arbitrary binning mechanism
* **Transform core (FFT or DFT)** – Arbitrary bin control always begins from a uniform spectrum.
  Power-of-two windows feed the radix-2 FFT; non power-of-two sizes fall back to the exact direct
  DFT. The NUFFT/CZT/dynamic modes currently route to the DFT scaffolding until their dedicated
  kernels land, so the bin grid still follows the selected window size.【F:src/native/amp_kernels.c†L4334-L4387】
* **Normalised bin coordinate** – After the transform, the code computes `ratio = i /
  (window_size - 1)` for bin index `i`, yielding a fractional position in `[0, 1]` that the UI can
  drive. This coordinate is used only for comparisons; it does not spawn fractional bins on its
  own.【F:src/native/amp_kernels.c†L4367-L4374】
* **Piecewise-constant band gate** – `lower_band` and `upper_band` are clamped, swapped if needed,
  and passed into `compute_band_gain`. Any bin whose `ratio` lands in the interval receives the
  “inside” gain equal to `filter_intensity`; everything else receives the complementary
  “outside” gain. Both gains are floored at `1e-6`, so every discrete FFT bin is either fully
  boosted or fully attenuated by that constant. Because the comparison is binary, fractional
  bounds effectively select the nearest bins whose ratios straddle the requested cut-off—there is
  no interpolation or cross-fade between neighbouring indices.【F:src/native/amp_kernels.c†L2811-L2836】【F:src/native/amp_kernels.c†L4359-L4373】
* **Bin-centric phase steering** – A single `phase_offset` per slot drives a complex rotation after
  the gain. Every bin within the selected range is rotated by the same sine/cosine pair; bins
  outside the band keep that identical rotation even though they receive the opposite gain. The
  rotation therefore shifts phase uniformly across the discrete bins being emitted rather than
  steering an interpolated centre frequency.【F:src/native/amp_kernels.c†L4351-L4373】
* **Divisor FFT capture** – The per-bin divisor spectrum is cached (`div_fft_real/imag`) after the
  transform so the reverse pass can reapply the exact complex ratios that were used when the
  forward path divided the input spectrum.【F:src/native/amp_kernels.c†L4354-L4361】

The outcome is a deterministic, nearest-bin gating scheme: callers may sweep fractional bounds,
but the magnitude response always flips per discrete FFT bin. Arbitrary bin “centres” are
therefore quantised to the underlying transform grid, not resynthesised at new frequencies.

## Backward execution behaviour
The adjoint closely mirrors the forward pass:
1. **Metadata replay** – The recombination buffer stores the emitted time-domain samples, while
   the divisor and metadata rings are rewound through the same memmove pipeline used in the
   forward path. This guarantees alignment between gradients and the history that produced the
   forward sample.【F:src/native/amp_kernels.c†L4531-L4630】
2. **Warm-up product** – During the fill phase, the backward node multiplies the upstream gradient
   by the stabilised divisor so that the derivative of the forward division is respected even
   before the FFT path is live.【F:src/native/amp_kernels.c†L4631-L4649】
3. **FFT/DFT replay** – After warm-up, the divisor history is windowed, transformed, and cached in
   the same spectral buffers, followed by a windowed transform of the recombination buffer (the
   accumulated forward outputs).【F:src/native/amp_kernels.c†L4653-L4705】
4. **Band-aware gradient routing** – The adjoint replays the same ratio comparisons, producing the
   identical step-function gain mask. Gradients inside the band are scaled by the stored
   `filter_intensity`, rotated by the conjugate phase, and divided by that gain floor before being
   multiplied by the cached divisor spectrum. Bins outside the band experience the complementary
   constant gain. No additional interpolation is introduced—the backward flow mirrors the forward
   nearest-bin selection exactly.【F:src/native/amp_kernels.c†L4661-L4740】

## Metrics and observability
* Runtime metrics surface the measured delay, a coarse “heat” estimate derived from algorithmic
  complexity, and the last phase/band/filter scalars for inspection. Total heat is accumulated in
  the node state and grows monotonically across firings.【F:src/native/amp_kernels.c†L4389-L4409】
* The native test suite (`test_fft_division_node.cpp`) verifies that the node reports the declared
  delay, exposes v2 metrics, and retains the most recent metadata in the metrics struct. The same
  harness compares node output to an in-test C++ simulation, covering both FFT and DFT code paths
  plus the new dynamic algorithm stub (with carrier metadata) and the arbitrary bin gating controls.【F:src/native/tests/test_fft_division_node.cpp†L598-L773】

## Audit observations
* The arbitrary bin control is purely algebraic—no lookup tables or Python fallbacks are involved
  in the gating, satisfying the runtime-only requirement.
* Because gain floors at `1e-6`, extremely narrow bands will still leak a minimum amount of energy;
  callers expecting brick-wall rejection must cascade additional filtering.
* The per-slot memmove operations scale with `window_size` for each frame. Large windows and many
  channels will therefore skew towards the direct DFT cost profile and should be profiled if
  deployment scenarios approach that limit.

