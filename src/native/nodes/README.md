# Native Node Packages

This tree groups node-specific assets so contracts, presets, and source code travel together. Every native node should eventually live in a folder with the following shape:

```text
<node>/
  README.md              # quick reference for maintainers
  contracts/             # tap-group contracts (JSON)
  presets/               # parameter bundles referencing a contract
  src/                   # native implementation files (.c/.cpp/.h)
```

The KPN demo currently uses these nodes and each now has a package directory:

- `parametric_driver/`
- `oscillator/`
- `mix/`
- `fft_division/`

At the moment the implementations still sit in `amp_kernels.c`; we keep the `src/.keep` markers so the directories exist ahead of the extraction work. Tooling and the build system copy the entire `nodes/` tree next to the compiled binaries, which makes presets and contracts available for future hot-loading without a rebuild.
