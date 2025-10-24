# AMP KPN ABI Extension Proposal (v0.1)

This proposal describes an incremental extension to the AMP KPN runtime ABI that preserves the existing `amp_run_node`
contract while introducing explicit delay metadata, oversampling hints, reversible execution support, and thermal
instrumentation. The design builds on the expectations established in the development guidance, audit findings, and node
contract, reaffirming that **all production features and diagnostics must execute through the native runtime with no
Python fallbacks**.【F:docs/kpn_development_guidance.md†L5-L48】【F:docs/kpn_contract.md†L1-L120】【F:docs/kpn_audit_findings.md†L1-L63】

## 1. Goals and Scope
- Maintain backwards compatibility for binaries that only link against `amp_run_node` while enabling richer scheduling
  and differentiation workflows.
- Surface declared delay and oversampling metadata so the scheduler can honour delay-aware readiness guarantees and
  stage-aware archetypes documented in the development guidance.【F:docs/kpn_development_guidance.md†L5-L48】
- Provide a reversible execution entry point with optional metrics reporting so thermodynamic accounting remains
  observable across forward and backward passes.【F:docs/kpn_contract.md†L49-L118】
- Keep migration low-risk by allowing nodes to opt into the new ABI gradually, with explicit capability flags.

## 2. Proposed Entry Point
```c
typedef enum {
    AMP_FORWARD = 0,
    AMP_BACKWARD = 1
} amp_execution_mode_t;

typedef struct {
    uint32_t measured_delay_frames;
    float accumulated_heat;
    float reserved[6];
} amp_node_metrics_t;

int amp_run_node_v2(const amp_node_descriptor_t* desc,
                    amp_state_t** state,
                    const amp_buffer_t* inputs,
                    amp_buffer_t* outputs,
                    amp_execution_mode_t mode,
                    amp_node_metrics_t* metrics);
```

- **Calling convention:** Parameters follow the original ABI layout so existing call sites can upgrade without
  reworking stack discipline. `inputs` and `outputs` remain arrays of `(batch, channel, frame)` tensors allocated by the
  runtime allocator.【F:docs/kpn_contract.md†L1-L118】
- **Mode selection:** `AMP_FORWARD` preserves existing behaviour. `AMP_BACKWARD` instructs the node to consume gradient
  tensors (mirroring the forward outputs order) and emit gradients for its inputs. Nodes that lack an adjoint return
  `AMP_E_UNSUPPORTED`, signalling the runtime to propagate zeros and log the absence for diagnostics.
- **Metrics side-channel:** When `metrics != nullptr`, implementations populate `measured_delay_frames` with the frames
  consumed before outputs become visible and accumulate emitted heat in `accumulated_heat`. The reserved padding keeps
  the structure naturally aligned for future counters (e.g., slew limit hits) without breaking the ABI.

## 3. Descriptor and Plan Extensions
### 3.1 Node Descriptor JSON
Extend each node descriptor with explicit capability and timing metadata:
```json
{
  "name": "GainNode",
  "oversample_ratio": 4,
  "declared_delay": 64,
  "supports_v2": true,
  "params": {
    "gain": 0.5,
    "thermo.heat": 0.0
  }
}
```
- `oversample_ratio` declares a power-of-two multiplier that informs IPLS grouping and allocator reservations.
- `declared_delay` specifies the latency, in frames, that downstream consumers must wait before reading emitted tokens.
- `supports_v2` indicates that `amp_run_node_v2` is available; legacy nodes omit the flag or set it to `false`.
- `thermo.heat` remains the canonical thermal parameter binding so instrumentation can aggregate heat across the graph.

### 3.2 Plan Blob Edge Metadata
Augment edge records with scheduling metadata:
```json
{
  "source": "Oscillator.A",
  "target": "Filter.B",
  "delay_frames": 128,
  "synchrony_gates": [0, 3]
}
```
- `delay_frames` stores FIFO offsets that the runtime applies within the contiguous token arena, ensuring producers run
  early enough to satisfy consumer readiness checks.【F:docs/kpn_development_guidance.md†L5-L48】
- `synchrony_gates` identifies optional gate IDs that must all be satisfied before the consumer executes, enabling
  delay-aware readiness semantics without sacrificing topological ordering guarantees.

## 4. Runtime Behaviour
### 4.1 Scheduling and Token Arena
- The runtime stores declared delays and per-edge offsets alongside channel metadata in the unified token arena, keeping
  FIFO semantics intact while deferring visibility until the declared delay expires.【F:docs/kpn_development_guidance.md†L24-L48】
- Oversampling hints guide stage-aware archetype fusion; nodes sharing a stage and oversample ratio may be fused into
  Eigen kernels while singleton inputs remain valid, matching the IPLS philosophy.

### 4.2 Backward/Adjoint Execution
- During `AMP_BACKWARD`, the runtime presents gradient tensors in the same `(batch, channel, frame)` layout, mirroring
  forward outputs. Nodes write gradients for each input buffer; if an input was unused in forward mode, implementations
  should zero the corresponding gradient tensor and report `AMP_E_SUCCESS` for consistency.
- State reversal follows the existing contract: nodes may mutate `*state` to represent adjoint accumulators. The runtime
  subsequently passes the updated pointer into future invocations, invoking `amp_release_state` on superseded values.

### 4.3 Metrics Propagation
- `amp_node_metrics_t` enables nodes to report per-invocation delay and thermal output through the native path. The
  runtime sums `accumulated_heat` into global counters already used for `thermo.heat` aggregation so thermal budgets
  remain observable without Python instrumentation.【F:docs/kpn_contract.md†L71-L118】
- Consumers that omit the metrics pointer continue operating without allocation changes, preserving legacy behaviour.

## 5. Migration Strategy
1. **Dual entry points:** All nodes continue to export `amp_run_node`. Nodes that implement adjoint logic also export
   `amp_run_node_v2` and set `supports_v2 = true` in their descriptors. Legacy nodes return `AMP_E_UNSUPPORTED` when
   invoked in backward mode.
2. **Runtime negotiation:** When loading a graph, the runtime inspects the descriptor flag. If `supports_v2` is set and
   the function pointer is present, it dispatches to `amp_run_node_v2`; otherwise it falls back to `amp_run_node`.
3. **Descriptor versioning:** Plan blobs include a `"descriptor_version": 2` field when any node uses the extended
   metadata. Older runtimes ignore unknown fields, so existing binaries remain functional.
4. **Testing cadence:** Maintain joint coverage for `amp_run_node` and `amp_run_node_v2` until all production nodes adopt
   the new ABI, after which the legacy entry point may be deprecated through a separate proposal.

## 6. Testing and Validation Plan
- **Unit tests:** Extend the native test suite with cases that verify declared delay propagation, oversample-aware
  scheduling, and backward mode gradient routing. Continue running `test_thermo_param` alongside new fixtures to ensure
  `thermo.heat` remains observable.【F:src/native/tests/test_thermo_param.cpp†L1-L170】
- **Integration tests:** Update `kpn_unit_test` (or successors) to load plan blobs containing oversample and delay fields,
  asserting that readiness gates hold nodes until inputs expire.
- **Performance microbenchmarks:** Measure oversampled oscillator throughput and thermal accumulation under both forward
  and backward modes, recording metrics via `amp_node_metrics_t` to validate instrumentation overhead.
- **CI coverage:** Expand the native CI workflow to build and run the upgraded tests on Linux and Windows, ensuring all
  checks execute through the native runtime with no Python fallbacks.

## 7. Example Descriptor Excerpt
```json
{
  "graph": {
    "descriptor_version": 2,
    "nodes": [
      {
        "name": "OversampledOsc",
        "oversample_ratio": 8,
        "declared_delay": 192,
        "supports_v2": true,
        "params": {
          "frequency_hz": 220.0,
          "thermo.heat": 0.0
        }
      }
    ],
    "edges": [
      {
        "source": "OversampledOsc.out",
        "target": "Downsample.in",
        "delay_frames": 192,
        "synchrony_gates": []
      }
    ]
  }
}
```

## 8. Open Questions
- Define canonical enumerations for synchrony gates (e.g., to distinguish tempo locks vs. modulation gates).
- Confirm whether additional adjoint metadata (such as Jacobian sparsity hints) is required before implementation.
- Validate that the proposed `amp_node_metrics_t` padding accommodates anticipated counters without ABI churn.

## 9. Next Steps
- Circulate this proposal for review with runtime and oscillator maintainers.
- Upon approval, move Phase 5 of the action plan into implementation, ensuring contract documentation and tests stay
  aligned with the no-Python-fallback policy.

