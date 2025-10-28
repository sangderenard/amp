# Envelope Modulator Node Package

The envelope modulator exposes its tap contract, presets, and eventual source files from this directory.

- `contracts/` defines the control taps the ADSR driver consumes and the dual-lane envelope bus it emits.
- `presets/` provides baseline parameters targeting the default contract.
- `src/` is a staging area for the future extracted native implementation.

Adjust the contract when the envelope gains new taps or timing semantics so the wheel stays authoritative.
