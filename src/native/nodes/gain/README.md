# Gain Node Package

Gain node metadata and presets now ship with the tap contract so KPN tooling can negotiate its audio lanes.

- `contracts/` describes the single-producer/single-consumer audio FIFO contract for the gain stage.
- `presets/` captures default gain parameters bound to the bundled contract.
- `src/` will inherit the native implementation once we split it out of `amp_kernels.c`.

Update these files together whenever channel policies or defaults change.
