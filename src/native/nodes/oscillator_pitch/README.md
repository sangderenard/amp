# Oscillator Pitch Node Package

This folder tracks tap contracts and presets for the oscillator pitch helper that slews frequency control signals.

- `contracts/` defines the pitch/root/additive taps the node consumes and the frequency lane it publishes.
- `presets/` supplies defaults referencing the bundled contract for demo graphs.
- `src/` awaits the native source extraction out of `amp_kernels.c`.

Keep the contract aligned with any future slew or routing behaviour so the KPN wheel negotiates correctly.
