# Oscillator Node Package

Assets for the native oscillator live here so the runtime, tooling, and documentation share a single source of truth.

- `contracts/` defines the tap-group contract for the PCM stream emitted by the oscillator.
- `presets/` contains parameter packs (starting with the demo default) that external tools can hot-load.
- `src/` will receive the oscillator implementation once we extract it from `amp_kernels.c`.

Add new tap layouts or presets alongside the existing files; the build copies everything under this directory, so no additional scripting is required when you introduce new oscillator variants.
