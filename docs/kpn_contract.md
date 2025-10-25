# AMP KPN Node Contract

This contract consolidates the runtime expectations that every AMP Kahn Process Network (KPN) node and oscillator must satisfy. It restates the behaviour described in the native headers and design briefs so contributors can validate implementations without reverse-engineering the C++ sources. All production features, benchmarks, and regression tests **must execute through the native runtime**—Python fallbacks are explicitly disallowed by project policy.

## Invocation Semantics
- **Frame slices:** `amp_run_node` is typically invoked one frame at a time but nodes must gracefully process multi-frame windows when `frames > 1`. Treat the `(batch, channel, frame)` layout as canonical for audio, parameter, and modulation tensors.【F:src/native/include/amp_native.h†L31-L79】
- **State lifecycle:** The runtime may reuse or replace the opaque `state` pointer between calls. Nodes must return any updated state pointer and expect the runtime to release the previous value via `amp_release_state`. Stateful integrators should initialise cleanly when `state == nullptr`.【F:src/native/include/amp_native.h†L60-L69】
- **Allocator contract:** Node outputs are allocated with the runtime allocator and returned via `double **out`. Implementations must set `*out_channels` correctly for each frame slice so the runtime can copy and free the buffer without leaks.【F:src/native/include/amp_native.h†L51-L59】

## Data Ownership and Layout
- **Unified tensors:** Audio inputs, parameter overrides, and modulation sources arrive as contiguous tensors following the `(batch, channel, frame)` ordering. Nodes must never assume channel-major layouts or interleave frames manually.【F:src/native/include/amp_native.h†L31-L79】
- **Control history:** When supplied, the `history` pointer exposes time-indexed automation curves. Treat the history as read-only and sample it deterministically for modulation-dependent behaviour.【F:src/native/include/amp_native.h†L60-L69】
- **FIFO guarantees:** The runtime is migrating toward a contiguous arena-backed FIFO that preserves declared delays and frame ordering. Node implementations must avoid ad-hoc buffering that would hide scheduler timing guarantees documented in the development guidance.【F:docs/kpn_development_guidance.md†L16-L48】

## Modulation and Parameter Binding
- **Parameter surfaces:** Parameter tensors use the same layout as audio inputs. Nodes may receive fewer modulation channels than destination channels and must implement documented add/multiply semantics when applying modulation.【F:src/native/include/amp_native.h†L60-L79】
- **Shape validation:** Overrides supplied through `amp_graph_runtime_set_param` are compared against descriptor defaults. Nodes should emit descriptive errors (and return `-2`) when mismatched shapes are detected to keep diagnostics consistent with the runtime hardening roadmap. The runtime records an `amp_graph_runtime_set_param_shape_mismatch` log entry before returning `-2` so downstream tooling can surface the expected and provided tensor spans.【F:docs/kpn_upgrade_action_plan.md†L18-L33】【F:src/native/graph_runtime.cpp†L906-L955】

## Timing, Delay, and Thermodynamics
- **Declared delays:** Nodes that oversample or perform lookahead processing must declare their nominal delay so downstream scheduling remains aligned. Report delays alongside oversampling ratios inside descriptor metadata and honour the scheduler’s delay-aware dispatch described in the runtime guidance.【F:docs/kpn_development_guidance.md†L5-L48】【F:src/native/include/oscillator_design_notes.h†L96-L126】
- **Thermal side-channel:** Oscillators and other heat-producing nodes expose a reserved `thermo.heat` parameter per frame. The runtime aggregates this signal to enforce thermal budgets; do not gate functional correctness on the presence of the thermal channel, but always publish it when available so diagnostics remain consistent.【F:src/native/include/oscillator_design_notes.h†L121-L140】【F:src/native/tests/test_thermo_param.cpp†L1-L170】
- **No Python fallback:** Thermodynamic reporting, oversampling, and delay declarations must be implemented in the native path. Instrumentation or experiments may prototype elsewhere, but authoritative behaviour always runs inside the C++ runtime.【F:docs/oscillator_design_brief.md†L1-L35】

## Reversible Execution and Instrumentation
- **Forward and backward flows:** Nodes implement a forward renderer (current `amp_run_node`) and provide hooks for reversible/adjoint execution so differentiable workflows can re-use the same contract. Preserve deterministic results whether operating on single frames or batched windows.【F:docs/kpn_development_guidance.md†L20-L33】
- **FFT metadata surfaces:** Frequency-domain processors publish per-frame phase offsets, band bounds, and filter intensities through descriptor parameters. Implementations such as the reversible FFT divider consume these tensors in both forward and backward modes and report the latest values via `AmpNodeMetrics.reserved[0..3]` when running under `amp_run_node_v2`, keeping forward/backward telemetry aligned without Python fallbacks.【F:src/native/amp_kernels.c†L4090-L4269】【F:src/native/tests/test_fft_division_node.cpp†L588-L816】
- **Slew-aware integrators:** Integrations must respect configured slew limits and cooperate with the runtime’s instrumentation layers that monitor traversal latency and thermodynamic output. Avoid per-frame allocations to keep profiling stable.【F:docs/kpn_development_guidance.md†L28-L33】【F:src/native/include/amp_native.h†L70-L79】

## Descriptor Authoring Checklist
When authoring a node descriptor (or serialising it from Python graph assembly), include the following metadata inside `params_json` to keep the runtime contract observable:

```json
{
  "oversample": 4,
  "declared_delay_frames": 128,
  "thermo": {
    "heat_param": "thermo.heat"
  },
  "modulation": {
    "mode": "multiply",
    "channels": 2
  }
}
```

- **Oversample ratio:** Declares the internal oversampling factor so the scheduler can budget compute and align IPLS archetypes.【F:src/native/include/oscillator_design_notes.h†L15-L54】
- **Declared delay:** Specifies the number of frames introduced by internal processing so downstream nodes receive aligned data.【F:src/native/include/oscillator_design_notes.h†L96-L126】
- **Thermo binding:** Names the parameter that surfaces per-frame heat so the runtime can aggregate it across the graph.【F:src/native/include/oscillator_design_notes.h†L121-L140】
- **Modulation schema:** Documents how modulation tensors should be applied, matching the runtime’s add/multiply semantics.【F:src/native/include/amp_native.h†L60-L79】

Keep this checklist in sync with runtime evolutions. When the ABI expands (e.g., `amp_run_node_v2`), update this document first and cross-link any new requirements from the action plan.
