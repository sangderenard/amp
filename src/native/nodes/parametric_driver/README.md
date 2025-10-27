# Parametric Driver Node Package

This folder groups everything the runtime needs for the parametric driver node:

- `contracts/` contains tap-group contracts that describe how the node exposes its control bus.
- `presets/` keeps parameter packs that tooling can load without rebuilding the binary.
- `src/` is reserved for the native implementation once `amp_kernels.c` is decomposed. Until then the node still lives in the monolith; this directory intentionally stays empty so the new layout already exists when we extract the code.

The default contract matches the expectations of the native KPN demo: a single control lane that may be multicasted to any downstream modulator. Add new contracts next to `contracts/default.json` and reference them from fresh preset files so build tools can discover the assets automatically.
