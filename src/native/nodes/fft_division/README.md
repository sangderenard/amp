# FFT Division Node Package

This directory packages the FFT division node’s assets:

- `contracts/` captures the spectral tap layout together with the passthrough PCM tap required by the demo spectrogram.
- `presets/` stores reusable parameter bundles; `default.json` matches the 512-point analysis run from the KPN demo.
- `src/` is a placeholder for the future native implementation module.

New spectral layouts (different bin counts, alternate PCM taps) belong next to the default files. The build copies the entire folder so runtime tools can discover new presets without manual wiring.
