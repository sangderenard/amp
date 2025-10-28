# Controller Node Package

Controller node assets now live alongside the tap contract so the KPN wheel can discover its dynamic control bus layout.

- `contracts/` documents the control-lane contract and multicast expectations.
- `presets/` includes starter controller mappings wired to the default contract.
- `src/` holds a placeholder until the native implementation moves out of `amp_kernels.c`.

Keep these resources updated whenever controller outputs or mapping semantics evolve.
