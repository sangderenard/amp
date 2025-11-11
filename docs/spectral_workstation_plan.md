# Spectral Workstation Plan (FFTDivisionNode)

This plan documents the ongoing rewrite that turns `FFTDivisionNode` into a
streaming spectral workstation while keeping Python fallbacks off the table.

## Runtime axes

The node tracks three independent axes inside its working tensor:

1. **Analysis window** – Size equals the FFT `window_size`. Each streamed slot
   collects this many PCM samples before a spectrum is emitted.
2. **Time slices** – A per-slot ring buffer that stores successive spectra.
   Retaining multiple slices enables short spectral history without growing the
   cache.
3. **Cache layer** – Optional outer ring that accumulates complete slices. The
   cache depth defaults to one so legacy graphs see no behavioural change.

Together the tensor layout is `(lanes, cache_slice, frequency_bin, time_slice)`.
With a cache depth of one it collapses to the familiar `(lanes, frequency_bin,
 time_slice)` layout used historically.

## Aggregation hooks

Two aggregation stages mediate how cached spectra feed operators:

- **Lane aggregation** (pre-cache) optionally collapses multi-channel spectra
  into a shared view before writing to the cache.
- **Window aggregation** (post-cache) reduces along the cache axis when emitting
  spectra or resynthesising PCM (policies: `latest`, `sum`, `mean`, etc.).

The default configuration (pass-through lanes + `latest`) reproduces the legacy
behaviour byte-for-byte.

## Implementation snapshot

- Forward path uses the streaming FFT backend to produce spectra, mirrors them
  into the working tensor, optionally ingests external spectra, and forwards the
  data to an inverse stream that keeps the PCM tail aligned.
- Operator stack is currently a placeholder; spectral modifications will return
  once the streaming structure settles.
- Backward path remains `AMP_E_UNSUPPORTED` while the adjoint is rebuilt.
- Metrics continue to report window size and algorithm identifiers for tooling.

## Roadmap highlights

1. Reintroduce complex division, band gating, and phase rotation operators on
   top of the streaming scaffolding.
2. Add NUFFT/CZT backends that populate the same tensor layout while respecting
   analysis window counts.
3. Revisit dynamic carrier resynthesis once aggregation policies have test
   coverage.

## Testing guidance

- Exercise only the production runtime paths—no smoke harnesses or Python
  fallbacks.
- Cover the default configuration (`cache_slices = 1`) plus at least one
  aggregated variant in `test_fft_division_node` so ControlDelay invariants stay
  observable.
- Use compact references (Eigen/fftfree, deterministic spectral sums) to verify
  new policies.

## Notes for agents

- Python fallbacks are prohibited; rely solely on the primary native pipeline.
- Keep JSON descriptors explicit: include `io_mode`, cache sizing, aggregation
  keys, and window metadata so downstream components infer intent reliably.
- Preserve the warm-up contract: emit raw PCM until the analysis window fills.
