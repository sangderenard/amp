# Native Node Packages

This tree groups node-specific assets so contracts, presets, and source code travel together. Every native node now ships with a
package shaped like:

```text
<node>/
  README.md              # quick reference for maintainers
  contracts/             # tap-group contracts (JSON)
  presets/               # parameter bundles referencing a contract
  src/                   # native implementation files (.c/.cpp/.h)
```

All runtime nodes participating in the modern KPN wheel publish contracts here:

- `constant/`
- `controller/`
- `envelope/`
- `fft_division/`
- `gain/`
- `lfo/`
- `mix/`
- `oscillator/`
- `oscillator_pitch/`
- `parametric_driver/`
- `pitch/`
- `pitch_shift/`
- `resampler/`
- `safety/`
- `sine_osc/`
- `subharmonic/`

The native implementations still live inside `amp_kernels.c`; the `src/.keep` markers reserve space for their eventual extraction.
Build tooling copies the entire `nodes/` tree next to the compiled binaries so presets and tap contracts stay discoverable for
demos, tooling, and hot-loaded graphs.
