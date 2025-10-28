# Resampler Node Package

The resampler’s KPN tap contract, presets, and eventual source extraction are managed from this folder.

- `contracts/` specifies the derivative audio tap it outputs and the upstream buffer expectations.
- `presets/` records default rate-tracking parameters tied to the contract.
- `src/` will receive the native implementation when we break up `amp_kernels.c`.

Update the contract and presets alongside runtime changes so wheel negotiation stays faithful.
