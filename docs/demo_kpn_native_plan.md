# Demo: native KPN spectrogram (driver → pitch bridge → oscillator → PCM + spectral taps)

This document tracks the implementation plan for a native-only KPN demo that:

- Renders audio using the native KPN runtime and **only** the new native ABI node types (no Python fallbacks; see `AGENTS.md`).
- Integrates the full control pipeline from the legacy pitch demo: parametric driver, pitch-shift module, continuous timebase bridge, and the learned scheduler parameters.
- Publishes FFT analysis data via the new `tap_groups` contracts (`spectral_bins` for magnitudes, `pcm_tap` for passthrough audio) and renders a verification PNG alongside the WAV output.
- Uses a single native runtime instance created from the serialized compiled plan.
- May rebuild the native runtime in this development environment when the DLL/SO is missing.

## High-level architecture

- Build an `AudioGraph` with the following logical pieces:
   - `ParametricDriverNode` → `PitchShiftNode` → optional `ContinuousTimebaseNode` bridge → `OscNode` → `MixNode` sink.
   - `FFTDivisionNode` connected as a tap consumer of the mix output. The PCM stream remains the primary sink while the FFT node exposes `spectral_bins` and `pcm_tap` groups per the new contracts.
   - Scheduler configuration mirrors the legacy demo (learned scheduler with tuned bias params) to maintain timing parity.

- The graph is compiled by the existing Python code (`AudioGraph`) and serialized into two blobs:
  - Node descriptor blob (node descriptors / topology)
  - Compiled plan blob (native job schedule and kernel mapping)

- The demo creates a single native runtime instance via the repository's native ABI (the same path used by `NativeGraphExecutor` / `amp_graph_runtime_create`). That runtime will be reused for the entire render, and the streamer will drain both the PCM ring and the spectral dump queue.

## Key constraints and decisions

- Use only the repo's native kernels. After serializing the compiled plan the script will inspect the plan or node descriptors and abort if any node lacks a native kernel mapping.
- Parameter curves (pitch program, slew limits, pitch-shift ratios, timebase gains) are pre-generated as static B×C×F buffers and delivered through the driver/bridge taps. Export the control-history blob and feed it to the runtime so the scheduler observes the real `ControlDelay` path.
- Rebuilding the native runtime is fine and expected in development. The demo will attempt to load the native library; if not present it will show explicit instructions to run the CMake/MSBuild build steps found in the repo (Windows/CMake guidance). The script itself will not silently skip a build — it will either load the built library or inform the user how to build.

## Implementation steps (script: `scripts/demo_kpn_kpn_native_correct.py`)

1. Parse CLI flags and defaults
   - `--duration` (seconds, default 2.0)
   - `--sr` sample rate (default match repo tests, e.g. 48000)
   - `--out-dir` directory for PNGs (default `output/demo_kpn_spectro`)
   - flags to enable `--play` (attempt playback) and `--display` (pygame spectro)

2. Construct the `AudioGraph` topology
   - Create nodes: `ParametricDriverNode`, `PitchShiftNode`, optional `ContinuousTimebaseNode`, `OscNode`, `MixNode`, `FFTDivisionNode`.
   - Wire pitch modulation (driver → timebase, timebase → oscillator) and audio routing (driver → pitch-shift → timebase → oscillator → mix) to match the legacy Python demo exactly.
   - Connect the FFT node as a tap consumer of the mix output so `spectral_bins` and `pcm_tap` are both active.
   - Configure scheduler mode/params to match the learned scheduler used by the bridge.

3. Serialize
   - Call `audio_graph.serialize_node_descriptors()` and `audio_graph.serialize_compiled_plan()` to produce the blobs to pass to native runtime.

4. Validate compiled plan
   - Inspect node descriptors / plan to ensure each node type maps to a native kernel. If any node lacks native kernel, abort and print the offending node names.

5. Create native runtime
   - Use the repository ABI (the same calls wrapped by `NativeGraphExecutor`) to create a runtime once with the two blobs.

6. Prepare static params and buffers
   - Generate control curves (pitch sequences, driver amplitude/render mix, oscillator slew, pitch-shift ratio, timebase gain) using the legacy helper routines for parity.
   - Create parameter buffers as required (B×C×F layout) with static values for FFT divisors, stabiliser, band limits, etc.
   - Export the control-history blob from the compiled graph and pass it into the streamer so all controller events flow through the authoritative history path.

7. Execute block loop
   - Launch a `NativeGraphStreamer` using the compiled runtime and control history.
   - Drain the PCM ring for audio output and read the dump queue that carries the multicast `spectral_bins` tap.
   - Accumulate spectral magnitude tiles into a 2-D buffer and render a grayscale PNG heatmap. Flatten PCM frames into a WAV file.

8. Optional playback / display
   - If `--play` and the system has a sound device, stream PCM audio blocks to the device.
   - If `--display` and `pygame` is installed, render a scrolling spectro window using the PNG frames or in-memory FFT columns.

9. Cleanup
   - Destroy the native runtime and close any devices.

## Verification and checks

- After serialization, ensure each node has `kernel_impl` set (or the compiled plan indicates native implementation). If not, abort.
- Confirm that PCM output buffers remain valid and are not emptied or replaced by the FFT node. The script will assert that a PCM sink buffer is produced by the runtime.
- Validate that spectral chunks were consumed via `tap_groups.spectral_bins` (dump queue counts, band coverage) before writing the PNG heatmap.
- Write at least one PNG frame to `--out-dir` as proof of correct FFT tap operation.

## Build notes (Windows / CMake)

- If the native library is missing, run the repo's existing build steps (example, run from `build` directory):

```powershell
# from repo root (Windows / PowerShell)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

- After successful build, the Python native loader should be able to find the `amp_native` DLL (or the built artifact). The demo script will attempt to load the native ABI and, if it fails, will print the above instructions.

## AGENTS.md / policy

- This demo intentionally avoids creating any Python-side node fallbacks. All runtime node execution must happen via the native KPN kernels. See `AGENTS.md` for the project's policy on avoiding Python-only harnesses.

## Outputs

- `output/demo_kpn_spectro/*.png` — FFT magnitude heatmap(s) derived from the `spectral_bins` tap group.
- Optional: `output/demo_kpn_spectro/output.wav` if the PCM stream is written to disk.

## Next actions

- Implement the tap-aware streamer collection (PCM ring + spectral dump queue), verify parity with the legacy demo, and run a short validation render that emits both the WAV and PNG assets. Report native build status if the DLL rebuild is required.
