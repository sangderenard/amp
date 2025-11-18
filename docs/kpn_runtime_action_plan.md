# Action Plan Toward DAG + IPLS with Explicit Delays

## Current Baseline Audit

- **Channel semantics:** `KahnChannel` caches only the latest tensor pointer per edge, so there is no FIFO or delay tracking in the native runtime today. 【F:src/native/graph_runtime.cpp†L120-L143】【F:src/native/graph_runtime.cpp†L640-L683】
- **Scheduler behaviour:** `AudioGraph._build_execution_plan` performs a Kahn topological walk over audio and modulation edges without time or gate metadata, producing an order-only plan blob. 【F:src/amp/graph.py†L743-L807】【F:src/amp/graph.py†L1176-L1213】
- **Descriptor payload:** `serialize_node_descriptors` omits delay, synchrony, and thermal fields; it only carries node connectivity, modulation bindings, and parameter buffers. 【F:src/amp/graph.py†L1075-L1160】
- **Execution loop:** `execute_runtime` iterates the compiled order frame-by-frame, immediately running every node whose index appears in the plan and storing its most recent output directly into the channel. No readiness or expiry checks exist. 【F:src/native/graph_runtime.cpp†L640-L809】

## Staged Development Roadmap

1. **Metadata foundation**
   - Extend Python graph descriptors and compiled plan blobs with fields for per-edge delays, optional synchrony gates, node thermal coefficients, and declared process latency.
   - Update the native descriptor parser to ingest and persist the new metadata so later stages can rely on it.
   - Introduce tap contracts (`taps.inputs[]`, `taps.outputs[]`) so nodes declare their input/output FIFO characteristics as independent taps rather than anonymous channels. Documented in `docs/kpn_tap_contract.md`.
2. **Contiguous channel arena**
   - Replace the shared-pointer token map with a contiguous arena allocator that serves FIFO ring buffers per channel and expands geometrically under pressure.
   - Add traversal timestamps (or step counters) to each token and make channel read operations respect delay expiry rules.
   - Convert runtime structures to tap-centric wheels: each tap owns its FIFO contract (capacity, stride, release policy) and is hosted on the threaded KPN wheel.
3. **Scheduler refactor**
   - Introduce a readiness queue that considers both edge delays and synchrony gates before scheduling work.
   - Implement optional IPLS ingestion that maps staged/type-grouped nodes onto Eigen archetypes while falling back to the default KPN order when IPLS data is absent.
   - Ensure the threaded KPN C streamer (`AmpGraphStreamer`) is the only supported execution engine. Python fallbacks or synchronous execute paths are deprecated.
4. **Node contract upgrades**
   - Define the reversible API (`backward()`), slew-aware integrator hooks, delay reporting, and thermal side-channel in the node interface.
   - Audit existing nodes and retrofit them to satisfy the new contract, including vectorised execution paths.
   - Require every node to publish tap groups (`1a`, `1b`, etc.) to describe thematic bundles while keeping tap locks independent.
   - Introduce an explicit completion handshake (`amp_wait_node_completion`) so threaded callers can block on pending frames without spinning; callers must opt into tail draining via `AMP_COMPLETION_DRAIN`, and FFT-oriented nodes must raise completion once their internal wheels drain without relying on implicit flushes.
5. **Instrumentation and validation**
   - Wrap nodes with timing trackers to confirm the reported delays and scheduler honours.
   - Create integration tests that exercise delayed FIFO delivery, synchrony gates, thermal aggregation, and tap negotiation through the threaded KPN streamer.
   - Update all demos (`demo_kpn_*`) and benchmarking harnesses to use the KPN streamer wheels exclusively.

## Milestone Criteria

- IPLS-capable scheduler proven via acceptance tests covering staged execution, delay expiry, and synchrony gates.
- Node implementations demonstrate reversible flows, thermal outputs, and slew-limited integrator compatibility in both vectorised and singleton scenarios.
- Memory diagnostics confirm the contiguous arena maintains FIFO guarantees and doubles capacity under synthetic stress without fragmentation.

Maintain documentation updates alongside each milestone so the guidance in `kpn_development_guidance.md` remains authoritative.
