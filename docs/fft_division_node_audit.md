# FFTDivisionNode Implementation Audit

## Scope and intent
This audit now references the spectral workstation plan (`docs/spectral_workstation_plan.md`) and
the spectral tensor packing ABI (`docs/spectral_packing_standard.md`). It captures the observable
behaviour of the native `FFTDivisionNode` and documents the ongoing aggregation rewrite that
introduces explicit cache staging. All observations come from the C/C++ runtime—Python fallbacks
remain disallowed by policy.

## Window taxonomy

Three distinct “windowed” constructs coexist inside the node:

1. **Analysis window** – The FFT/DFT/NUFFT transform length (`window_size`). It governs warm-up,
   frequency-bin count, and the stride with which PCM is consumed per slot.
2. **Time-slice ring** – A per-slot circular buffer with length `working_ft_duration_frames`.
   Each entry stores one spectrum produced by the analysis window.
3. **Cache layer** – A higher-level circular buffer with depth `cache_slices` (default `1`). Each
   cache slice holds an entire time-slice ring. Increasing the depth enables smearing/aggregation
   policies while retaining fixed memory bounds.

The internal working tensor now has shape `(lanes, cache_slice, frequency_bin, time_slice)`. When
`cache_slices == 1` the structure collapses back to `(lanes, frequency_bin, time_slice)`, matching
legacy behaviour byte-for-byte.

## Aggregation stages

Two explicit aggregation hooks mediate how data flows through the cache:

* **Lane aggregation (pre-cache)** – Optionally collapses all lanes into lane zero before writing
  into the cache. Disabled by default.
* **Window aggregation (post-cache)** – Reduces along the cache axis when emitting spectra or
  resynthesising PCM. Policies include `latest` (default), `sum`, `mean`, and future bespoke
  reducers. Each lane is reduced independently.

Together these hooks describe when `cache_slices` needs to exceed one: either to smear multiple
time slices within a slot or to hold pre-aggregated results for downstream operators that prefer a
combined view.

## Node surface and state

* **Window and stabiliser configuration** – Each dispatch resolves a JSON-declared
  `window_size`, per-frame `epsilon` stabiliser, and default FFT/window algorithms before
  touching the work buffers. Non-positive window sizes are clamped to one frame, epsilon values
  are floored to `1e-12`, and a non power-of-two size automatically falls back to the slower
  direct DFT path.【F:src/native/amp_kernels.c†L4385-L4574】
* **Algorithm scaffolding** – The selector now recognises `"nufft"`, `"czt"`, and
  `"dynamic"` (dynamic oscillator synthesis) in addition to the existing radix-2 FFT and
  direct DFT options. The dynamic variant bypasses the FFT entirely: each frame builds
  oscillator projections directly in the time domain, advances persistent carrier phases by
  `exp(i·2πf_norm)` for every windowed sample, and emits the newest sample from the carrier
  coefficients while leaving the DFT/FFT branches untouched.【F:src/native/amp_kernels.c†L4520-L4581】【F:src/native/amp_kernels.c†L4690-L4808】
* **Dynamic carrier summary** – When the dynamic oscillator algorithm is selected the node scans
  for `carrier_band_{index}` parameter streams, records how many bands were provided, and stores a
  simple aggregate so future oscillator nodes can contribute per-band carriers without changing
  the runtime interface.【F:src/native/amp_kernels.c†L2759-L2839】【F:src/native/amp_kernels.c†L4385-L4574】
* **Parameter inputs** – The node consumes complex divisor taps, optional algorithm/window
  selectors, per-frame stabilisers, phase offsets, and band bounds/intensity controls, all
  retrieved through `EdgeRunnerParamView` wrappers. Slots correspond to `batches × channels` and
  each slot owns its own rolling window in the state struct.【F:src/native/amp_kernels.c†L4174-L4200】【F:src/native/amp_kernels.c†L4317-L4333】
* **State buffers** – `ensure_fft_state_buffers` guarantees contiguous per-slot rings for the
  input signal, complex divisors, phase/band metadata, scratch FFT workspaces, and a
  `dynamic_phase` array that tracks the accumulated oscillator phase per carrier. The node keeps
  the most recent phase/band/filter scalars in `state->u.fftdiv.last_*` so metrics can surface
  what the synthesis path actually used.【F:src/native/amp_kernels.c†L2147-L2194】【F:src/native/amp_kernels.c†L3444-L3519】

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
4. **Dynamic oscillator resynthesis** – When the dynamic oscillator mode is active the node skips
   the spectral division entirely. Each active carrier is evaluated against the windowed signal by
   accumulating `Σ x_w[n] · e^{-iθ_k[n]}` and (optionally) dividing by the windowed divisor
   projection, then reweighted by the band gate and phase offset before emitting
   `(1/N)·Re{c_k · e^{iθ_k[N-1]}}`. Per-carrier phases are integrated sample-by-sample so the next
  frame resumes from the exact phasor endpoint, and the fallback inverse transform is invoked only
  if no carriers are provided.【F:src/native/amp_kernels.c†L4690-L4808】【F:src/native/tests/test_fft_division_node.cpp†L476-L555】

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
4. **Dynamic adjoint guard** – The backward implementation now refuses to run when the dynamic
  oscillator algorithm is selected, returning `-1` so callers cannot silently reuse the FFT adjoint
  against the non-linear oscillator bank. FFT/DFT paths continue to undo the gain and phase mask
  before multiplying by the cached divisor spectrum.【F:src/native/amp_kernels.c†L4557-L4559】【F:src/native/amp_kernels.c†L5154-L5191】

## Metrics and observability

* Runtime metrics surface the measured delay, a coarse “heat” estimate derived from algorithmic
  complexity, and the last phase/band/filter scalars for inspection. Total heat is accumulated in
  the node state and grows monotonically across firings.【F:src/native/amp_kernels.c†L4389-L4409】
* The native test suite (`test_fft_division_node.cpp`) verifies that the node reports the declared
  delay, exposes v2 metrics, and retains the most recent metadata in the metrics struct. The same
  harness compares node output to an in-test C++ simulation, covering both FFT and DFT code paths,
  the oscillator-bank resynthesis, and the arbitrary bin gating controls.【F:src/native/tests/test_fft_division_node.cpp†L598-L879】

## Diagnostic tooling

The headless gradient harness (`test_fft_noise_gradient`) and its companion Python helper
(`scripts/fft_noise_gradient.py`) now accept an `--algorithm` command line flag, allowing renders to
toggle between the FFT divider implementations (including the dynamic oscillator stub) without
editing JSON descriptors. The flag simply injects the chosen label into the node parameters before
invoking the native renderer.【F:scripts/fft_noise_gradient.py†L63-L86】【F:src/native/tests/test_fft_noise_gradient.cpp†L300-L363】

## Audit observations

* The arbitrary bin control is purely algebraic—no lookup tables or Python fallbacks are involved
  in the gating, satisfying the runtime-only requirement.
* Because gain floors at `1e-6`, extremely narrow bands will still leak a minimum amount of energy;
  callers expecting brick-wall rejection must cascade additional filtering.
* The per-slot memmove operations scale with `window_size` for each frame. Large windows and many
  channels will therefore skew towards the direct DFT cost profile and should be profiled if
  deployment scenarios approach that limit.
* NUFFT and CZT backends must emit spectra that honour the analysis-window bin count before data
  enters the cache; oversampled grids collapse along the time-slice axis so aggregation policies
  remain well defined.



