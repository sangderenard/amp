# Native Runtime Diagnostic Reference

This document captures the current structure of AMP's dynamically generated C
runtime and the places where segmentation faults usually appear.  The goal is to
make native crashes reproducible without introducing alternate execution paths
or "smoke" harnesses – the tooling described here exercises the same code that
the interactive UI calls.  All helpers route through
`amp.system.require_native_graph_runtime`, so any missing native components are
treated as hard failures rather than silently falling back to Python.

## Runtime build pipeline

1. **Kernel scaffolding** – `src/native/amp_kernels.c` and
   `src/native/graph_runtime.cpp` contain the DSP kernels and the Kahn process
   network runtime.  The shared library (`libamp_native.*`) is produced by the
   CMake project in `src/native/` and can be built entirely outside Python.
   Configure from the repository root with `cmake -S src/native -B build/native
   -DCMAKE_BUILD_TYPE=Release`, then build the desired targets (for example,
   `cmake --build build/native --target amp_native test_fft_division_node`).
   The project now fetches Eigen and the FFT harness sources automatically via
   `FetchContent`, so the build succeeds without manually initialising
   third-party submodules.  Python only drives the build when no binary is
   available (or when an override requests a rebuild) and never falls back to a
   Python executor.
2. **Graph specialisation** – `AudioGraph.render_block` serialises the Python
   graph into descriptors (`serialize_node_descriptors`) and a compiled plan
   (`serialize_compiled_plan`).  These blobs describe the node ordering,
   parameter shapes, and modulation wiring that the native runtime executes.
3. **Native execution** – During each render the graph exports a
   control-history blob (`ControlDelay.export_control_history_blob`) and hands
   that, together with the compiled plan, to the `NativeGraphExecutor`.  The
   executor loads the standalone CMake-built library via CFFI and refuses to
   proceed if the binary is missing, keeping Python fallbacks strictly
   forbidden.  Failures at this stage usually manifest as heap corruption or
   segmentation faults inside the native code.

## Native logging instrumentation

The instrumentation hooks in `src/native/amp_kernels.c` and
`src/native/graph_runtime.cpp` are compiled out by default so the production build
stays fast.  To opt into native logging:

1. Set the environment variable `AMP_NATIVE_DIAGNOSTICS_BUILD=1` (or "true",
   "yes", "on") before importing the modules so the CMake build enables the
   `AMP_NATIVE_ENABLE_LOGGING` option and includes the logging hooks in the
   resulting binaries.
2. Call `amp_native_logging_set(1)` on the compiled runtime (or
   `amp.diagnostics.enable_py_c_logging(True)` from Python) to activate the log
   sinks.  The runtime ships with logging hard-disabled, so no files are opened
   until an explicit opt-in occurs.

Without the compile-time flag the logging hooks are hard-disabled regardless of
any runtime toggles.

The interplay between these layers means crashes can stem from descriptor drift,
control-history mismatches, or the CFFI module itself; the diagnostics described
below help separate those cases.

## Control history and render flow

`render_audio_block` in `src/amp/runner.py` orchestrates every headless render.
It samples controller history (`AudioGraph.sample_control_tensor`), builds the
parameter block with `app.build_base_params`, then calls
`AudioGraph.render_block`.  That method records the window of control history it
is about to hand to C, writes `logs/last_control_blob.bin` and
`logs/last_control_blob.json`, and finally invokes the native runtime.  Any
mismatch between the control blob and the compiled plan is a likely source of
native crashes; the JSON metadata captures the time window and descriptor sizes
involved so regressions can be reproduced.

## Using the graph diagnostic harness

`scripts/graph_runtime_diagnostic.py` renders the three-oscillator interactive
graph (or any configuration file) with neutral controller curves.  It records
control events through `AudioGraph.record_control_event`, calls the same
`render_audio_block` helper, and writes every generated artifact into a target
folder:

```bash
python scripts/graph_runtime_diagnostic.py --blocks 4 --dump-dir logs/native_debug
```

The script will:

- drive the real control-delay path with zeroed joystick curves so the native
  runtime still sees canonical history data;
- persist each rendered PCM block as `block_XXX.npy` for numerical comparison;
- copy the generated `_amp_ckernels_cffi.c` (when present), write the serialized
  node descriptors and compiled plan, and emit a JSON summary.

Because this harness uses the shared modules (`build_runtime_graph`,
`render_audio_block`, and `NativeGraphExecutor`) it surfaces the same
segmentation faults as the interactive UI while making the compilation artifacts
easy to inspect.

## Triage checklist when C crashes persist

1. Run the diagnostic script to capture descriptors, plan binaries, and PCM
   output for the failing block.
2. Inspect `logs/last_control_blob.json` alongside the plan metadata in the
   dump directory to ensure the C runtime and Python descriptors agree on node
   counts and parameter spans.
3. If the crash occurs before any PCM is written, diff the generated
   `_amp_ckernels_cffi.c` against a known-good build to look for contract drift
   between node implementations and their contracts in `node_contracts.py`.
4. Use the block-level timing metadata to identify whether a specific node is
   repeatedly triggering fallback paths or taking an unusual amount of time; the
   compiled plan JSON lists the node order and parameter slices currently in
   play.

Following this loop keeps the investigation focused on discrepancies between the
Python descriptors, the generated control history, and the native runtime.
