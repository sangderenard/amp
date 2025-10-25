## Demo: native KPN spectrogram (driver → oscillator → PCM stream — FFT is a non-PCM tap and does NOT replace the audio stream)

This document describes the implementation plan for a native-only KPN demo that:

- Renders audio using the native KPN runtime and ONLY the new native ABI node types (no Python node implementations or fallback kernels).
- Produces PNG frames from the FFT splitter node by tapping the PCM audio stream.
- Uses a single native runtime instance created from the serialized compiled plan.
- Is allowed to rebuild the native runtime in this development environment.

### High-level architecture

- Build an `AudioGraph` with the following logical pieces:
  - `Driver` node (control inputs) -> `Oscillator` node -> PCM audio stream sink/mixer.
  - `FFT Splitter` node: this node is constructed so it taps the PCM stream (it should be connected as an additional consumer of the PCM frames), not as a replacement for the PCM output. The PCM audio stream must remain intact and be written to the audio sink or output buffer.

- The graph is compiled by the existing Python code (`AudioGraph`) and serialized into two blobs:
  - Node descriptor blob (node descriptors / topology)
  - Compiled plan blob (native job schedule and kernel mapping)

- The demo creates a single native runtime instance via the repository's native ABI (the same path used by `NativeGraphExecutor` / `amp_graph_runtime_create`). That runtime will be reused for the entire render.

### Key constraints and decisions

- Use only the repo's native kernels. After serializing the compiled plan the script will inspect the plan or node descriptors and abort if any node lacks a native kernel mapping.
- Parameters for the run are static (no time-varying curves). Static parameter buffers will be created and passed to the runtime as B×C×F param views.
- Rebuilding the native runtime is fine and expected in development. The demo will attempt to load the native library; if not present it will show explicit instructions to run the CMake/MSBuild build steps found in the repo (Windows/CMake guidance). The script itself will not silently skip a build — it will either load the built library or inform the user how to build.

### Implementation steps (script: `scripts/demo_kpn_kpn_native_correct.py`)

1. Parse CLI flags and defaults
   - `--duration` (seconds, default 2.0)
   - `--sr` sample rate (default match repo tests, e.g. 48000)
   - `--out-dir` directory for PNGs (default `output/demo_kpn_spectro`)
   - flags to enable `--play` (attempt playback) and `--display` (pygame spectro)

2. Construct the `AudioGraph` topology
   - Create nodes: `ParametricDriverNode` (or repo driver), `OscNode` (or repo oscillator), audio sink/mixer node
   - Add `FFTDivisionNode` (or repo FFT splitter node) and connect it so it receives a tap of the PCM stream (not replacing the PCM sink). That may mean wiring the FFT node's input to the same stream source or connecting its input to the mix output while leaving the mix/sink intact.

3. Serialize
   - Call `audio_graph.serialize_node_descriptors()` and `audio_graph.serialize_compiled_plan()` to produce the blobs to pass to native runtime.

4. Validate compiled plan
   - Inspect node descriptors / plan to ensure each node type maps to a native kernel. If any node lacks native kernel, abort and print the offending node names.

5. Create native runtime
   - Use the repository ABI (the same calls wrapped by `NativeGraphExecutor`) to create a runtime once with the two blobs.

6. Prepare static params and buffers
   - Create parameter buffers as required (B×C×F layout) with static values: oscillator frequency, driver amplitude, FFT settings (window size, hop), sample rate, block size.

7. Execute block loop
   - Loop for the required number of audio blocks (duration / block_size).
   - For each block: call `amp_graph_runtime_execute` (or the wrapper) to fill output PCM buffer and FFT node output buffers.
   - Collect the FFT node's output frames and write PNG(s). Optionally stitch frames into a single spectrogram image at the end.

8. Optional playback / display
   - If `--play` and the system has a sound device, stream PCM audio blocks to the device.
   - If `--display` and `pygame` is installed, render a scrolling spectro window using the PNG frames or in-memory FFT columns.

9. Cleanup
   - Destroy the native runtime and close any devices.

### Verification and checks

- After serialization, ensure each node has `kernel_impl` set (or the compiled plan indicates native implementation). If not, abort.
- Confirm that PCM output buffers remain valid and are not emptied or replaced by the FFT node. The script will assert that a PCM sink buffer is produced by the runtime.
- Write at least one PNG frame to `--out-dir` as proof of correct FFT tap operation.

### Build notes (Windows / CMake)

- If the native library is missing, run the repo's existing build steps (example, run from `build` directory):

```powershell
# from repo root (Windows / PowerShell)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

- After successful build, the Python native loader should be able to find the `amp_native` DLL (or the built artifact). The demo script will attempt to load the native ABI and, if it fails, will print the above instructions.

### AGENTS.md / policy

- This demo intentionally avoids creating any Python-side node fallbacks. All runtime node execution must happen via the native KPN kernels. See `AGENTS.md` for the project's policy on avoiding Python-only harnesses.

### Outputs

- `output/demo_kpn_spectro/*.png` — FFT frame images or a stitched spectrogram.
- Optional: `output/demo_kpn_spectro/output.wav` if the PCM stream is written to disk.

### Next actions

- I will implement the script described above and run a short, non-playing validation run that produces at least one PNG. If the native runtime needs building, I will run the build steps and report results.
