# Mix Node Package

The mixer’s assets are co-located here so contracts, presets, and source code evolve together.

- `contracts/` defines the FIFO contract for the mix bus that ultimately feeds the host.
- `presets/` contains parameter packs (the demo uses `default.json`).
- `src/` is a placeholder for the C implementation once it leaves `amp_kernels.c`.

When you add new tap variants (e.g., surround buses), drop fresh JSON files next to the existing ones. The build picks the entire directory up automatically.
