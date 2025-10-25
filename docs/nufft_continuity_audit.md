# NUFFT Continuity Audit and Reversible Oscillator Action Plan

This document captures the continuity audit script and the staged enablement plan requested for the NUFFT and oscillator family. It is scoped to the native runtime—Python fallbacks remain disallowed per project policy, so every test harness described here must exercise the C runtime entry points (`amp_run_node` / `amp_run_node_v2`).

## Audit Prompt (Deterministic Test Suite Specification)

_All routines operate in double precision. Seed every RNG so reruns reproduce bitwise-identical stimuli._

1. **Uniform-limit sanity (FFT/STFT)**
   - Generate a random signal `x[n]` with power-of-two length `N=2^k`.
   - Apply a Hann window with hop `N/2`, run FFT → IFFT → overlap-add reconstruction.
   - Assert `max|x̂ − x| ≤ 1e−12`.
   - Purpose: pin scaling, phase, and window conventions before auditing nonuniform paths.

2. **Type-2 baseline vs. wandering-bin node (precision race)**
   - Ground truth: direct evaluation `x[n] = Σ_k A_k e^{i2π f_k n / F_s}` using arbitrary `{f_k} ⊂ [0, F_s/2]` for `K ∈ {32, 64}`.
   - Contestants:
     - (a) High-quality NUFFT-2 with `σ ∈ {1.5, 2.0}`, kernel width `m ∈ {6, 8}`, Kaiser–Bessel spreading.
     - (b) The wandering-bin node in both “meandering-bin → delta-lobe” (collapsed kernel) and “tiny-lobe” (3-tap) modes.
   - Metrics (after GCC-PHAT alignment): sample MSE, relative L2, PSNR, STFT spectral MSE (matching window/hop), and aliasing energy above `0.45 F_s`.
   - Passing bar: for four `(N, K)` trials with random `{f_k, A_k}`, the node’s relative L2 ≤ `0.9 ×` the best NUFFT-2 relative L2.

3. **Continuity audit (frequency and output envelopes)**
   - Define time-varying trajectories `f_k[n] = f0 + α n + β sin(2π r n / N)` with small `α`, `β`, and smooth cubic amplitude envelopes `A_k[n]`.
   - Checks:
     - C0 continuity: sample-to-sample jump `|x[n] − x[n−1]|` exhibits no spurious spikes when compared to an analytic oscillator reference.
     - Instantaneous frequency via Hilbert transform matches analytic `f_k[n]` with RMSE ≤ `1e−3 · F_s / N` per frame.
     - Phase unwrapping monotonicity: `unwrap(∠STFT(x))` has no discontinuities beyond window-induced π slips.

4. **Derivative (adjoint) verification**
   - Perform vector–Jacobian checks for perturbations of `A_k[n]`, `f_k[n]`, and `φ0_k` against random `g[n]`, verifying `⟨J·δp, g⟩ ≈ ⟨δp, Jᵗ·g⟩` with tolerance ≤ `1e−10`.
   - Run finite-difference validation on a sparse parameter subset with relative error ≤ `1e−6`.

5. **LCM oversample test (gridding claim)**
   - Choose incommensurate window lengths `L_k` and a base rate; set the node oversample factor to their least common multiple.
   - Demonstrate NUFFT-2 evaluation via oversampled gridding plus deapodization matches the direct oscillator baseline with relative L2 ≤ `1e−9`.

6. **Pathological edge cases**
   - Validate very low (`≈10 Hz`) and very high (`≈0.49 F_s`) frequencies, rapid bin crossings, and abrupt amplitude gates (PolyBLEP if used).
   - Expect no clicks, DC drift, or discontinuities; failures abort the rollout.

Passing items 2–6 allows the team to assert: “continuous, high-quality synthesis from synthetic FFT data, exceeding NUFFT-2 precision.”

## Action Plan: Reversible Algorithm Selection

### A. Common KPN Node Contract
- **Ports:** inputs `{mode, params, optional A_stream, f_stream, t_nodes, f_nodes}`, output `x_stream`.
- **Timing:** emit `L` samples per firing with timestamp `(t0, F_s, L)`.
- **State:** persist `{theta[k], rng_state}`, providing save/restore hooks.
- **Reversibility:** expose `forward()` and exact `adjoint()` (vector–Jacobian products) for every mode.

### B. Modes (mutually exclusive)
Parameter `params.mode ∈ {NUFFT1, NUFFT2, NUFFT3, CZT, DYN_OSC, WANDER_BIN}` selects one algorithm.

1. **NUFFT-2 (uniform time → nonuniform frequency)**
   - Inputs `{A_k (complex), f_k (Hz)}` with optional per-frame trajectories (frequency frozen within frame).
   - Implementation: grid/spread to an oversampled lattice (σ≈1.5–2.0), kernel width `m` (6–8) with Kaiser–Bessel weights, FFT, deapodize.
   - Adjoint: conjugate deapodization, IFFT, and degridding with the same kernel table.
   - Parameters: `{sigma, m, kb_beta, grid_pad}`.

2. **NUFFT-1 (nonuniform time → uniform frequency)**
   - Inputs are time samples at irregular `t_i`; output uniform DFT bins.
   - Implementation: grid samples onto the uniform lattice with the shared kernel, FFT.
   - Adjoint: IFFT followed by degridding back to `{t_i}`.

3. **NUFFT-3 (nonuniform ↔ nonuniform)**
   - Compose NUFFT-1 and NUFFT-2 (or use a dedicated type-3 kernel) with a single FFT in between.
   - Share `σ`, `m`, and kernel tables across both sides.

4. **CZT (Chirp-Z Transform)**
   - Map a frequency arc `[f_start, f_end]` into `M` bins via Bluestein-style convolution.
   - Adjoint mirrors the Bluestein path with conjugated chirps.

5. **Dynamic Oscillator (analytic evaluation)**
   - Directly compute `x[n] = Σ_k A_k[n] cos(θ_k[n])`, advancing phase recurrence and optional 3-tap “tiny lobe” bandwidth shaping.
   - Adjoint reuses the carrier (`i·c`), with prefix sums for `∂L/∂f_k[n]`.
   - Guarantees continuity and exactness; serves as baseline.

6. **Wandering-Bin (enable only after continuity pass)**
   - Extends Dynamic Oscillator with a symmetric mini-lobe tied to a continuously varying bin centre `f_k[n]`.
   - Enable when `f_k[n]` is C¹ per frame and `A_k[n]` is Lipschitz; PolyBLEP gates handle amplitude steps.
   - Adjoint shares the Dynamic Oscillator skeleton plus gradients through lobe weights.

### C. Reversibility and Validation
- Run adjoint tests for every mode per Audit §4.
- Round-trip checks:
  - `NUFFT-1 ∘ (NUFFT-1)ᵗ ≈ I` on random `y`.
  - `NUFFT-2 ∘ (NUFFT-2)ᵗ ≈ I` on random `x`.
  - `NUFFT-3` round-trip likewise.
  - `CZT ∘ CZT⁻¹ ≈ I` (matched chirps).
  - Dynamic Oscillator / Wandering-Bin: numeric VJP vs. finite difference.
- Uniform-limit regression: each mode reduces to the plain FFT path when `σ=1`, `m=0`, and nodes align to the integer grid.

### D. Parameter Schema (`C` struct)
```c
typedef enum { NUFFT1, NUFFT2, NUFFT3, CZT, DYN_OSC, WANDER_BIN } FftMode;

typedef struct {
  FftMode mode;
  int N, K, L;      // frame length, #bins, emit length
  double Fs, sigma; // sample rate, oversample
  int m;            // kernel width
  double kb_beta;   // Kaiser–Bessel beta
  // arrays (structure of arrays layout):
  const double *A;       // [K][L] or [K]
  const double *phi0;    // [K]
  const double *f;       // [K][L] or [K]
  const double *t_nodes; // for NUFFT1/3
  const double *f_nodes; // for NUFFT2/3
  // CZT specific:
  double f_start, f_end;
  int M;
  // state / scratch buffers:
  double *theta;
  double *cr;
  double *si;
} FftNodeParams;
```

### E. Performance Considerations
- Structure-of-arrays layout and block “time stripes” for cache locality.
- Vectorised `sincos` (Sleef/SVML) for phase updates.
- Precompute kernel lookup tables keyed by fractional offset `δ`.
- Expose quality presets (`fast`, `balanced`, `precise`) mapping to `{σ, m}` and tolerance budgets.
- Cache adjoint intermediates (`u = A w`, `i·c`) to avoid redundant computation.

### F. Rollout Sequence
1. Land the Dynamic Oscillator (fast, exact baseline with adjoint).
2. Implement NUFFT-2 (primary deployment target).
3. Add CZT for zoomed-band evaluation.
4. Introduce NUFFT-1, then NUFFT-3 via composition.
5. Execute the full audit suite (Sections 1–6).
6. If continuity holds, expose Wandering-Bin publicly; otherwise keep it behind a development flag until it passes.

Maintain this document in lockstep with the runtime ABI. Any future ABI extensions must be recorded first in `docs/kpn_contract.md` before propagating into the node implementation roadmap.
