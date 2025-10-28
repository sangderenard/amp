# Subharmonic Node Package

The subharmonic low lifter keeps its tap contract, presets, and future C source here.

- `contracts/` spells out the audio FIFO characteristics for both input and output taps.
- `presets/` stores default mix/drive parameters pointing at the contract.
- `src/` will house the extracted implementation when we split the monolith.

Keep the contract aligned with DSP changes so downstream consumers observe the correct bus semantics.
