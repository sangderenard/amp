# Native graph runtime policy

Every public entry point that renders an `AudioGraph` must execute entirely inside
AMP's C runtime.  Python implementations remain available purely for developer
instrumentation and are forbidden in production, benchmarking, or diagnostics.

The following rules describe the contract:

1. **Centralised guard** – `amp.system.require_native_graph_runtime` fails fast
   when the C kernels or native runtime cannot be loaded.  Benchmark and
   diagnostic helpers call this before they attempt to build graphs so missing
   toolchains cannot silently fall back to Python code.
2. **Shared render path** – `amp.runner.render_audio_block` is the single entry
   point for headless renders.  It delegates to `AudioGraph.render_block`, which
   locks the C edge runner and raises an error if any node reports Python
   fallbacks.  The same helper powers the interactive UI, the benchmark script,
   and all diagnostic harnesses, guaranteeing identical behaviour.
3. **Native executor** – `AudioGraph.render_block` serialises control history and
   invokes the compiled plan in C, while `NativeGraphExecutor` exposes the same
   interface to external tooling.  Both abstractions reject attempts to run after
   the native module has been torn down or when compiled descriptors drift.

Violations of this policy typically manifest as a `RuntimeError` mentioning
missing C kernels, an unavailable native runtime, or prohibited Python fallbacks.
These errors must be treated as hard failures: update the build environment or
add the missing C implementation rather than suppressing the guard.
