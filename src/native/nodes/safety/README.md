# Safety Node Package

Safety filtering assets—contracts, presets, and future source files—now live with the node under this directory.

- `contracts/` documents the clipping FIFO policy consumed by downstream host sinks.
- `presets/` provides baseline channel counts and DC filter coefficients referencing the contract.
- `src/` stands by for the implementation once extracted from `amp_kernels.c`.

Keep the contract current so realtime clipping guarantees remain explicit to the KPN wheel.
