# LFO Node Package

The low-frequency oscillator’s tap contract, presets, and future source module live here.

- `contracts/` captures both the optional analysis input tap and the emitted modulation lane.
- `presets/` seeds baseline rate/depth parameters that reference the default contract.
- `src/` provides a landing zone for the extracted implementation.

Revisit the contract whenever the LFO gains new modes or additional taps so downstream KPN runners stay accurate.
