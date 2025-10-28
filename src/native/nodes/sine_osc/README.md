# Sine Oscillator Node Package

This directory co-locates the sine oscillator’s tap contract, presets, and placeholder source staging area.

- `contracts/` explains the modulation taps the oscillator reads and the PCM stream it emits.
- `presets/` declares default frequency/amplitude settings referencing the contract.
- `src/` will inherit the implementation once the monolith is decomposed.

Refresh these assets whenever oscillator parameters or tap layout shifts.
