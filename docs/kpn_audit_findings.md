# KPN Runtime & Oscillator Audit Findings

## Scope
This audit covers the native KPN runtime implementation (`src/native/graph_runtime.cpp`), the current native unit test (`src/native/tests/kpn_unit_test.cpp`), and the design collateral introduced in the last merge (`docs/kpn_development_guidance.md`, `docs/oscillator_design_brief.md`, `src/native/include/oscillator_design_notes.h`). The goal is to identify concrete contract gaps, behavioural risks, and immediate remediation opportunities while respecting the project’s no-Python-fallback policy.

## Key Observations

### Runtime invocation model
- `amp_graph_runtime_execute` drives each node in per-frame slices. Inside the node loop, `amp_run_node` is invoked with `frames == 1`, and outputs are copied into tensors while the runtime frees the temporary buffer via `amp_free`.【F:src/native/graph_runtime.cpp†L689-L744】
- Parameter modulations are materialised into temporary tensors that share the `(batch, channel, frame)` convention before each frame slice is dispatched.【F:src/native/graph_runtime.cpp†L666-L708】

### Buffer ownership & state
- Nodes are expected to allocate their own frame buffers; the runtime copies data out and calls `amp_free`, but `amp_graph_runtime_buffer_free` is currently a no-op, so callers must not assume it releases memory. Documentation should clarify this to avoid double-free assumptions.【F:src/native/graph_runtime.cpp†L717-L745】【F:src/native/graph_runtime.cpp†L897-L934】
- Opaque state pointers are preserved across frames, and replacements trigger `amp_release_state` for the old value, matching the contract outlined in `amp_native.h`.【F:src/native/graph_runtime.cpp†L733-L755】

### Parameter overrides
- `amp_graph_runtime_set_param` blindly copies user data after a zero-length guard. It does not validate shape compatibility with defaults, so mismatched overrides could silently alter tensor extents.【F:src/native/graph_runtime.cpp†L905-L932】

### Unit test coverage
- The native test constructs a 4-node graph and validates default/override parameter flows with tight tolerances, exercising `amp_graph_runtime_set_param`, `execute`, and `clear_params` end-to-end. It does not cover modulation channels, state replacement, or declared delays.【F:src/native/tests/kpn_unit_test.cpp†L24-L208】

### Design brief alignment
- The oscillator brief and header emphasise oversampling, thermodynamic reporting, reversible behaviour, and the prohibition of Python fallbacks. These requirements are partially documented in `amp_native.h`, but there is no central contract document for non-C++ readers.【F:docs/oscillator_design_brief.md†L1-L44】【F:src/native/include/oscillator_design_notes.h†L5-L118】【F:src/native/include/amp_native.h†L29-L81】

## Gaps & Risks
1. **Thermodynamic side-channel** – No runtime facility (even via convention) surfaces the required `thermo.heat` reporting. Without a documented reserved parameter, implementations may diverge.
2. **Declared delay metadata** – The runtime lacks a mechanism for nodes to declare processing delay, yet schedulers must honour it per the design brief. Descriptor and plan formats need extension or a documented encoding.
3. **Adjoint/backward path** – The C API only exposes `amp_run_node` forward execution. There is no ABI for backward/adjoint evaluation demanded by the brief.
4. **Shape validation** – Parameter overrides are not validated against defaults, inviting subtle bugs when external tools provide mismatched shapes.
5. **Discoverability** – Design notes live in header comments; there is no standalone markdown that consolidates the contract for contributors who do not read the C++ headers.

## Immediate Remediation Candidates
- Author a `docs/kpn_contract.md` that restates the node contract, including explicit guidance on buffer ownership, modulation semantics, thermodynamic reporting conventions, and the no-Python-fallback stance.
- Add defensive logging/return codes to `amp_graph_runtime_set_param` to flag shape mismatches before data copies.
- Introduce a lightweight native test that exercises a reserved `thermo.heat` parameter to stabilise the side-channel convention without altering the ABI.
- Draft an ABI extension proposal (`docs/abi_extension_proposal.md`) covering declared delays, oversampling metadata, and a backward/adjoint entry point so the team can converge on a design before implementation.

## Follow-Up Questions
- Should declared delay and oversample metadata live in the existing descriptor JSON (`params_json`) or in an extended binary schema shared with the IPLS planner?
- Is the team amenable to introducing `amp_run_node_v2` with an explicit execution-mode enum, or should backward execution reuse the current function signature with flags embedded in `params_json`?
- What logging facilities are acceptable for reporting contract violations in the native runtime (e.g., integration with `amp_native_logging_enabled`)?

## Suggested Next Steps
1. Produce the contract markdown and circulate for review alongside the updated header comments.
2. Implement parameter shape validation and the thermo parameter test to harden existing behaviour.
3. Prepare the ABI extension proposal for team discussion, outlining the migration path that avoids breaking current binaries.
4. Plan CI coverage that builds and runs `kpn_unit_test` (and the new thermo test) on supported platforms without relying on any Python fallback execution paths.

