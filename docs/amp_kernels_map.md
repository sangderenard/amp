# `amp_kernels.c` Subsystem Map

This document inventories the major regions inside `src/native/amp_kernels.c` so we can plan incremental extraction into smaller translation units. Line numbers reference the current file layout.

## Overview

- **File size:** ~3.7k lines of mixed C/C++ (compiled as C++ for Eigen support).
- **Top-level responsibilities:** diagnostics/allocators, binary graph/control loaders, node state lifecycle, per-node DSP implementations, FFT division helpers, runtime dispatch glue.
- **Shared data:** `node_state_t` carries per-node persistent buffers; debug builds rely on `register_alloc`/`unregister_alloc` to correlate memory usage across sections.
- **Runtime contract:** Python fallbacks are not permitted for any of these kernels or utilities; all nodes must execute through the native runtime paths described below.

## Section Inventory

### 1. Diagnostics & Allocation Wrappers (L1-L115)
- **Content:** logging toggles, mutex-protected file handles, debug malloc/memcpy shims, allocation registry, exported logging helpers, and `AMP_LOG_*` macros.【F:src/native/amp_kernels.c†L1-L115】
- **Dependencies:** available everywhere via macro remaps; relies on `AMP_NATIVE_ENABLE_LOGGING` compile flag, OS threading primitives, and `_now_clock_seconds` timing helper (declared later in file).
- **Extraction notes:** can relocate to `amp_debug_alloc.c` by exposing `register_alloc`, `unregister_alloc`, `_log_native_call`, `_gen_wrapper_log`, and the exported `amp_native_logging_*` APIs.

### 2. Binary Plan & History Loaders (L116-L564)
- **Content:** little-endian readers, compiled plan serializer (`amp_load_compiled_plan`/`amp_release_compiled_plan`), control history loader/unloader, and supporting utilities like `read_u32_le`/`read_u64_le`/`read_bytes`.【F:src/native/amp_kernels.c†L116-L564】
- **Dependencies:** uses debug alloc macros (`calloc`, `register_alloc`), the logging helpers for tracing, and descriptor structs from `amp_native.h`.
- **Extraction notes:** largely independent; moving into `amp_runtime_serialization.c` would require sharing the little-endian helpers and allocator wrappers.

### 3. Envelope & Misc DSP Helpers (L565-L1619)
- **Content:** reusable DSP kernels (`subharmonic_process`), ADSR segment helpers (`envelope_start_attack` etc.), scratch buffer globals (`envelope_scratch`), and JSON/string helpers such as `json_copy_string` and `_json_extract_number` used across nodes.【F:src/native/amp_kernels.c†L565-L1619】
- **Dependencies:** relies on logging macros for scratch allocation tracing and on math utilities; envelope helpers share `envelope_scratch_t` defined near the top of the file.
- **Extraction notes:** candidates for an `amp_dsp_util.c` module; ensure scratch state remains singleton or is hoisted into dedicated structs.

### 4. Node State Definitions & Lifecycle (L1620-L2009)
- **Content:** `node_kind_t` enum, `node_state_t` union covering all node types, OSC/driver mode parsers, and `release_node_state`/`fft_state_free_buffers` for lifecycle management.【F:src/native/amp_kernels.c†L1620-L2009】
- **Dependencies:** touches almost every node-specific struct; FFT cleanup depends on helper declared later (`fft_state_free_buffers`).
- **Extraction notes:** keep enum/struct declarations with runtime glue; freeing routines could migrate alongside node implementations once state structs move to headers.

### 5. Controller & Parameter Utilities (L2010-L3163)
- **Content:** parameter lookup (`find_param`), CSV + controller source parsers, generic array helpers (`ensure_param_plane`, `param_total_count`), pitch grid builders, FFT algorithm/window clamps, and dynamic carrier summarizers used by spectral nodes.【F:src/native/amp_kernels.c†L2010-L3163】
- **Dependencies:** `register_alloc` for parse buffers, JSON helpers from Section 3, FFT constants shared with Section 6, and EdgeRunner param structs.
- **Extraction notes:** natural home in `amp_util.c`; ensure `ensure_param_plane` remains accessible to all node files.

### 6. Node Execution Includes (L3164-L3180)
- **Content:** per-node DSP implementations now live under `src/native/nodes/<name>/*.inc` and are included directly into the translation unit. The include block lists Constant, Controller, LFO, Envelope, Pitch, Oscillator Pitch, Subharmonic, Oscillator, Resampler, Parametric Driver, Pitch Shift, Gain, FFT Division (pass-through forward; backward unsupported), Mix, Sine Oscillator, and Safety modules.【F:src/native/amp_kernels.c†L3164-L3180】【F:src/native/nodes/fft_division/fft_division_nodes.inc†L1-L990】
- **Dependencies:** node modules reuse helpers from Sections 3–5 and access `node_state_t` defined earlier.
- **Extraction notes:** nodes are now organised by folder; further refactors can promote these `.inc` files into standalone translation units by updating the build to compile them separately.

### 7. Runtime Glue & API Surface (L3478-L3739)
- **Content:** dump utilities (`maybe_dump_node_output`), node timing instrumentation, state allocation inside `amp_run_node_impl`, node dispatch switch using `determine_node_kind`, export wrappers (`amp_run_node`, `_v2`, `amp_free`, `amp_release_state`).【F:src/native/amp_kernels.c†L3181-L3739】
- **Dependencies:** depends on every preceding section for node execution, uses logging macros, and references `node_state_t`/`node_kind_t` definitions.
- **Extraction notes:** this block becomes the core of `amp_runtime.c`; it will need headers exposing node-specific `run_*` signatures and state structs.

## Dependency Highlights

- Debug logging/alloc registry (Section 1) underpins CSV parsers, controller helpers, and every node that performs heap allocation.【F:src/native/amp_kernels.c†L1-L115】【F:src/native/amp_kernels.c†L2359-L2522】【F:src/native/nodes/controller/controller_node.inc†L1-L396】
- `node_state_t` union bridges runtime glue (Section 7) with node implementations; extracting nodes requires a shared header defining the union or per-node structs.【F:src/native/amp_kernels.c†L1620-L2009】【F:src/native/amp_kernels.c†L3478-L3667】
- Spectral nodes depend on FFT backend hooks and dynamic carrier helpers; backward execution shares buffers and metrics with forward path, so both should migrate together.【F:src/native/amp_kernels.c†L2481-L2912】【F:src/native/nodes/fft_division/fft_division_nodes.inc†L1-L990】
- Controller and Envelope nodes reuse CSV/JSON utilities and envelope scratch space, motivating grouping these helpers when splitting files.【F:src/native/amp_kernels.c†L565-L1619】【F:src/native/nodes/controller/controller_node.inc†L1-L396】【F:src/native/nodes/envelope/envelope_node.inc†L1-L339】

## Extraction Roadmap Notes

- Start by moving diagnostics (Section 1) into a dedicated module; replace macro remaps with exported inline functions/macros to keep call sites stable.
- Relocate parameter/JSON utilities (Sections 3 & 5) next, updating includes for all `run_*` functions that consume them.
- Peel off simpler nodes (Gain, Mix, Sine, Safety) as self-contained translation units once utilities are shared; they mainly depend on Sections 5 and 8.
- Isolate FFT division node by migrating Sections 6/7 related code plus shared state helpers, paving the way for functional rewrites without touching runtime glue.
- Keep runtime dispatch (Section 8) minimal by replacing switch cases with external declarations once node modules provide registration hooks.


Note: See docs/spectral_workstation_plan.md for the current stripped FFT pass-through and the staged spectral-operator roadmap.


Spectral I/O follows the packing standard documented in docs/spectral_packing_standard.md.
