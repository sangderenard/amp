# Constant Node Package

This package keeps all assets for the native Constant node aligned with the tap-contract driven KPN runtime.

- `contracts/` publishes the default tap-group contract describing the constant control bus.
- `presets/` provides starter parameter packs that point at the bundled contract.
- `src/` will eventually host the extracted native implementation once `amp_kernels.c` is decomposed.

Update the contract or presets alongside behaviour changes so downstream tooling and the wheel stay in sync.
