# Pitch Quantizer Node Package

Pitch quantizer assets—tap contracts, presets, and future source—reside here for the KPN runtime.

- `contracts/` encodes the control taps consumed by the quantizer and the snapped pitch lane it emits.
- `presets/` carries default scale and slew parameters bound to the contract.
- `src/` is a placeholder until we split the native implementation from `amp_kernels.c`.

Synchronize the contract with behaviour changes so downstream tooling honours the correct tap layout.
