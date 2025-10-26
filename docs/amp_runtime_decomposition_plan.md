# AMP Runtime Decomposition Plan

## Overview
This document records the proposed staged strategy for decomposing the current monolithic `amp_kernels.c` implementation into clearer, modular translation units. The aim is to isolate orthogonal responsibilities, make future maintenance tractable, and establish checkpoints for the upcoming FFT node rewrite.

## 1. Map the Monolith
- Catalogue the principal subsystems that coexist inside `amp_kernels.c`, including allocation and logging wrappers, node dispatch logic (`amp_run_node`), per-node implementations (gain, mix, sine, etc.), FFT division handling, control helpers, and state lifecycle routines.
- Document dependencies between these sections such as shared structures, helper utilities, and macro usage. This dependency map guides safe extraction boundaries and prevents circular include issues during refactoring.

## 2. Define Module Boundaries
- Keep core runtime glue—`amp_run_node`, node kind enumerations, and state lifecycle helpers—in a new `amp_runtime.c` module.
- Move diagnostics and debug allocation wrappers into `amp_debug_alloc.c` with a dedicated header that exposes only the required APIs.
- Consolidate generic utilities (JSON helpers, window functions, clamp helpers) inside `amp_util.c`.
- Extract each substantial node (FFT splitter, envelope, controller, spectral nodes) into dedicated files named `amp_node_<name>.c`, all sharing a common header that defines the node interface.
- Refactor FFT-specific helpers—including state structures and backend hooks—into `amp_node_fft_splitter.c`. Share structure declarations in a new header so both the runtime and backend consumers stay aligned.

## 3. Refactor Iteratively
1. **Extract debug allocation logic**: Introduce the new debug allocation module, adjust includes and macros, and verify the build remains green.
2. **Move utility helpers**: Relocate clamp and JSON utilities, updating references throughout the node files.
3. **Split simple nodes**: Migrate straightforward nodes (gain, mix, sine, safety) to validate the modular pattern and update CMake to compile the new translation units.
4. **Isolate the FFT node**: Create a dedicated file for FFT state management and the forward pass. Co-locate upcoming spectral logic and ensure shared structs reside in a header consumed by the backward stub and runtime.
5. **Iterate with validation**: After every step, execute the existing C++ unit tests (`kpn_unit_test`) and relevant Python suites covering the moved nodes to catch regressions.

## 4. Documentation and Tests
- Update developer documentation (for example, `c-runtime` notes or a new README section) to reflect the new module layout and the updated channel pin map.
- Prepare corresponding test adjustments—especially around `test_fft_spectral_node.py`—once the functional rewrite commences. Add native-focused tests if a C harness is available.

## 5. Tooling and Build Updates
- Modify the build system (e.g., `CMakeLists.txt`) incrementally as files move to keep diffs reviewable.
- Ensure headers maintain clean interfaces without leaking global internals.
- Validate Windows/MSVC and Eigen fallback builds after each decomposition milestone.

## 6. Next Steps
Following this staged approach maintains reviewability, mitigates regression risk, and establishes clear checkpoints before reworking the FFT node to support a single FFT path and the revised channel pin map.
