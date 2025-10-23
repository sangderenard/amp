# CFFI Runtime Diagnostic Reference

This document captures the current structure of AMP's dynamically generated C
runtime and the places where segmentation faults usually appear.  The goal is to
make native crashes reproducible without introducing alternate execution paths
or "smoke" harnesses – the tooling described here exercises the same code that
the interactive UI calls.

## Runtime build pipeline

1. **Kernel scaffolding** – `src/amp/c_kernels.py` owns the monolithic
   `C_SRC` string containing all DSP kernels and runtime glue.  The module builds
   an `_amp_ckernels_cffi` extension on import via `ffi.set_source(...)` and
   `ffi.compile(...)`, then copies the generated `.c` file and shared library
   back into the repository so they can be inspected when compilation succeeds.
2. **Graph specialisation** – `AudioGraph.render_block` asks
   `graph_edge_runner.CffiEdgeRunner` to serialise the Python graph into static
   descriptors (`serialize_node_descriptors`) and a compiled plan.  The runner
   is responsible for loading `_amp_ckernels_cffi`, allocating C-ready buffers,
   and marshalling parameters before calling `amp_run_node` for each node.
3. **Native execution** – During each render the graph exports a control-history
   blob (`ControlDelay.export_control_history_blob`) and hands that, together
   with the compiled plan, to the C entry point.  Failures at this stage usually
   manifest as heap corruption or segmentation faults inside the generated C.

The interplay between these layers means crashes can stem from descriptor drift,
control-history mismatches, or the CFFI module itself; the diagnostics described
below help separate those cases.

## Control history and render flow

`render_audio_block` in `src/amp/runner.py` orchestrates every headless render.
It samples controller history (`AudioGraph.sample_control_tensor`), builds the
parameter block with `app.build_base_params`, then calls
`AudioGraph.render_block`.  That method records the window of control history it
is about to hand to C, writes `logs/last_control_blob.bin` and
`logs/last_control_blob.json`, and finally invokes the `CffiEdgeRunner`.  Any
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

- drive the real control-delay path with zeroed joystick curves so the CFFI
  runner still sees canonical history data;
- persist each rendered PCM block as `block_XXX.npy` for numerical comparison;
- copy the generated `_amp_ckernels_cffi.c` (when present), write the serialized
  node descriptors and compiled plan, and emit a JSON summary.

Because this harness uses the shared modules (`build_runtime_graph`,
`render_audio_block`, and `CffiEdgeRunner`) it surfaces the same segmentation
faults as the interactive UI while making the compilation artifacts easy to
inspect.

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
