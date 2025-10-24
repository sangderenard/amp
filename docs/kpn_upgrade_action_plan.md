# KPN + Oscillator Upgrade Plan

This document sequences the work required to align the AMP native KPN runtime, oscillator implementations, and associated tooling with the recently adopted design brief. Each phase contains mid-level deliverables and ready-to-use delegation prompts so agents can execute tasks sequentially without diverging from the no-Python-fallback policy.

## Phase 1 — Complete Investigative Audit (In Progress)
- **Deliverables**: `docs/kpn_audit_findings.md` (this repo), call-outs for runtime gaps, list of recommended quick fixes.
- **Key checks**: runtime invocation pattern, buffer ownership, parameter binding behaviour, current unit test coverage, design-doc compliance.【F:docs/kpn_audit_findings.md†L1-L63】
- **Delegation prompt**:
  ```text
  Read docs/kpn_audit_findings.md and confirm every highlighted gap has a follow-up issue or task in this plan. Update the audit if new discrepancies are uncovered during implementation.
  ```

## Phase 2 — Contract Documentation & Discoverability
- **Goal**: Provide a single authoritative contract reference for contributors who may not inspect C++ headers.
- **Tasks**:
  - Author `docs/kpn_contract.md` summarising runtime/node expectations (frames-first invocation, `(batch, channel, frame)` layout, buffer ownership, state lifecycle, modulation semantics, thermodynamic reporting convention) while reiterating that Python fallbacks are prohibited in production tests and features.【F:src/native/include/amp_native.h†L29-L81】【F:src/native/include/oscillator_design_notes.h†L88-L118】【F:docs/oscillator_design_brief.md†L1-L44】
  - Cross-link the new markdown from existing design docs.
- **Delegation prompt**:
  ```text
  Draft docs/kpn_contract.md using amp_native.h and oscillator_design_notes.h as primary sources. Include an explicit "No Python fallback" notice and an example JSON snippet showing oversample and declared delay metadata inside params_json.
  ```

## Phase 3 — Runtime Hardening (No ABI Change)
- **Goal**: Address immediate runtime risks without altering public signatures.
- **Tasks**:
  - Add shape validation and diagnostic logging to `amp_graph_runtime_set_param` so mismatched overrides return a distinct error code and emit clear log entries.【F:src/native/graph_runtime.cpp†L905-L932】
  - Introduce a lightweight native test (e.g., `test_thermo_param.cpp`) that exercises a reserved `thermo.heat` parameter through `amp_graph_runtime_set_param` and ensures execution remains stable.【F:src/native/tests/kpn_unit_test.cpp†L24-L208】
  - Document the `thermo.heat` convention in the new contract markdown and mention it in `oscillator_design_notes.h` if additional clarity is needed.
- **Delegation prompt**:
  ```text
  Update amp_graph_runtime_set_param to validate incoming tensor shapes against defaults, returning -2 on mismatch with AMP logging enabled. Add src/native/tests/test_thermo_param.cpp that sets a thermo.heat override on a GainNode, runs amp_graph_runtime_execute, and asserts success.
  ```

## Phase 4 — ABI Extension Design (Planning Only)
- **Goal**: Prepare a design proposal for declared delay metadata, oversampling hints, and backward execution.
- **Tasks**:
  - Draft `docs/abi_extension_proposal.md` outlining `amp_run_node_v2`, execution-mode enums, optional side-channel structures, and plan-blob schema updates.【F:docs/kpn_development_guidance.md†L5-L44】
  - Include migration/compatibility strategy and test plan.
- **Delegation prompt**:
  ```text
  Create docs/abi_extension_proposal.md describing amp_run_node_v2 with forward/adjoint modes, plan blob extensions for declared delays and oversample ratios, and a rollout plan that keeps amp_run_node available for legacy nodes.
  ```

## Phase 5 — ABI Extension Implementation (Conditional on Approval)
- **Goal**: Implement the agreed design while maintaining backwards compatibility.
- **Tasks**:
  - Introduce `amp_run_node_v2` in headers and runtime, bridging to legacy nodes until all implementations migrate.
  - Extend descriptor parsing to read delay/oversample metadata and surface it to the scheduler without breaking existing blobs.
  - Update node implementations and tests to use the new entry point where appropriate.
- **Delegation prompt**:
  ```text
  Implement amp_run_node_v2 according to the approved spec, update graph_runtime.cpp to call it when available, and add regression tests covering declared delay propagation and adjoint mode dispatch.
  ```

## Phase 6 — CI & Golden Validation
- **Goal**: Ensure deterministic test execution across platforms through the native runtime.
- **Tasks**:
  - Add GitHub Actions workflows that build the native targets (Windows + Linux), run `kpn_unit_test` plus the new thermo test, and archive logs.
  - Integrate golden oscillator renders that compare native outputs against stored references with documented tolerances.
- **Delegation prompt**:
  ```text
  Add .github/workflows/native-kpn-ci.yml that configures CMake builds on Windows and Linux, runs kpn_unit_test and test_thermo_param, and uploads logs/artifacts for review.
  ```

## Phase 7 — Performance & Monitoring Follow-Up
- **Goal**: Instrument the runtime for oversampling workloads and thermal accounting once functionality stabilises.
- **Tasks**:
  - Develop microbenchmarks for oversampled oscillators to measure throughput and thermal reporting overhead.
  - Add optional runtime logging hooks (guarded by feature flags) for per-node heat accumulation and declared delay validation.
- **Delegation prompt**:
  ```text
  Build microbenchmarks that drive oversampled oscillator graphs through the native runtime, recording per-node execution time and reported thermo.heat. Summarise findings in docs/perf/oversampled_oscillators.md.
  ```

## Checklist Summary
- [x] Audit findings documented (`docs/kpn_audit_findings.md`).
- [ ] Contract markdown published and cross-referenced.
- [ ] Runtime shape validation + thermo test implemented.
- [ ] ABI extension proposal ready for review.
- [ ] ABI extension implemented (post-approval).
- [ ] CI workflows enforcing native test coverage.
- [ ] Performance benchmarks and monitoring notes delivered.

