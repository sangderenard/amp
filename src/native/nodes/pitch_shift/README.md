# Pitch Shift Node Package

Tap contracts and presets for the native pitch shifter live in this directory alongside a placeholder for extracted sources.

- `contracts/` outlines the audio FIFO contract and modulation taps the shifter honours.
- `presets/` ships default window/ratio parameters referencing the default contract.
- `src/` will host the native C implementation once it moves out of `amp_kernels.c`.

Revise these assets whenever buffer sizing, delivery policy, or modulation taps change.
