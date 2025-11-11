# Spectral Tensor ABI for FFTDivisionNode

This note captures the packing convention used when `FFTDivisionNode` exchanges
frequency-domain tensors with the rest of the runtime.

## Layout

- Audio tensors are shaped B×C×F (batches × channels × frames).
- Spectral tensors reuse the same axes. Each spectral frame holds `Nbins`
  complex bins.
- The channel axis stores interleaved complex pairs: `Re, Im, Re, Im, …` for
  each audio channel’s spectrum.

## Packing rules

Let `Nbins` equal the FFT `window_size`. For an audio channel index `a`, bin `k`
(0 ≤ `k` < `Nbins`) and component `comp` (`0 = Re`, `1 = Im`):

```text
channel_index = (a * Nbins + k) * 2 + comp
```

Total spectral channels: `Cin * Nbins * 2`.

## Metadata keys

Descriptors that consume or emit packed spectra should include:

- `"io_mode"`: `"pcm_to_pcm"`, `"pcm_to_spectrum"`, `"spectrum_to_pcm"`,
  `"spectrum_to_spectrum"`, or `"bidi"`.
- `"bins"`: equals `window_size`.
- `"complex_format"`: currently `"split"` (separate real and imaginary lanes).
- `"pack_order"`: currently `"bin_major"`.
- `"window"` / `"hop"`: carry analysis and synthesis window metadata.

## Example

For `window_size = 4`, `Cin = 1`, and per-frame bins:

- Re: `[0.0, 1.0, 0.5, -0.5]`
- Im: `[0.0, -0.5, 0.1, 0.2]`

Packed channel vector: `[0.0, 0.0, 1.0, -0.5, 0.5, 0.1, -0.5, 0.2]`.

## Testing guidance

- Compare packed spectra against a reference FFT (Eigen/fftfree) by reproducing
  the packing scheme above.
- PCM regression tests should still validate warm-up and overlap-add behaviour.

## Notes for agents

- Exercise only the production runtime paths; do not introduce Python fallbacks.
- Keep packing metadata explicit so downstream consumers can infer tensor shapes.
