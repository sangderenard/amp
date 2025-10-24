# KPN Runtime Guidance for Agents and Humans

## Purpose
This document records the shared goals and guardrails for extending the AMP Kahn Process Network (KPN) runtime so that the team can converge on the DAG + IPLS design brief without drifting from project ethos. No Python fallbacks are acceptable—every feature described here must execute through the native runtime pathway. Use this memo as the primary reference when planning or reviewing work around the scheduler, node contracts, and data movement, and consult [`docs/kpn_contract.md`](./kpn_contract.md) for the authoritative node-level contract.

## Scheduling Philosophy
- **Baseline KPN, optional IPLS:** The native scheduler must continue to support the current KPN semantics as the default. When the host supplies an Iterative Partial-Latency Schedule (IPLS), the runtime should honour it. The IPLS description arrives as a data structure that enumerates nodes by execution stage and type cohort; treat that ordering as authoritative.
- **Stage-aware archetypes:** With an IPLS in hand, compile archetypal nodes that bundle Eigen-powered vectorised kernels. These archetypes are selected per stage/type to exploit parallel execution while remaining compatible with single-item processing when necessary.
- **Delay-aware readiness:** Regardless of scheduler mode, no node should start until all mandatory inputs have traversed their declared delays. The scheduler therefore needs time-aware FIFOs and barrier checks before dispatching kernels.

## Node Contract Requirements
Every node implementation—native or wrapped—must satisfy the following:
- Provide both `forward()` (implicit in the current `amp_run_node` contract) and reversible `backward()` behaviours so graph stages can participate in differentiable workflows.
- Operate efficiently on batched, channelised frame blocks but accept singleton cases without special casing.
- Support opt-in execution wrappers that track timing, so instrumentation layers can measure traversal latency.
- Report real, measured processing delay whenever possible; if hardware does not allow measurement, expose a deterministic simulated delay value.
- Publish a thermodynamic side-channel that integrates heat accumulation from the node’s inputs, enabling higher-level thermal simulations.
- Interact with integrators that enforce slew-rate limits up to second-order derivatives and first integrals, including a “no impingement” mode for ideal integration.

## Dataflow and Memory Standards
- **Unified token arena:** All KPN channel caches must live in a single contiguous allocation. Implement a token manager that hands out slices, supports geometric (doubling) growth under pressure, and presents true FIFO ring semantics including reservation for future-delayed tokens.
- **Delayed storage:** Channels must store traversal metadata so tokens remain invisible until their expiry time; delayed entries never surface in consumer reads prematurely.
- **Configurable gather/scatter:** Nodes should be able to select aggregation or parallel fan-in/out behaviour. The runtime must let each node declare how it maps batches, channels, and frames across its inputs and outputs—including bleed and filtering paths required for high-fidelity multiband DSP.
- **Batch-first payloads:** All data delivered to nodes should follow the `(batch, channel, frame)` convention. Nodes map their gather/scatter transformations explicitly to keep multiband routing intelligible.

## Simulation Fidelity Goals
- Preserve accuracy for analog/digital hybrid simulations such as oscillator drivers, BLEP generation, and unlimited FFT-band multi-processing. Scheduling, buffering, and node APIs must not assume band limits or simplified interpolation shortcuts.
- Integrate synchronous gates optionally to stall node execution until every contributor in a gate reports readiness. Expose gates declaratively alongside edge metadata so the scheduler can enforce them without bespoke code paths.

## Collaboration Expectations
- Agents exploring this area are encouraged to inspect existing runtime sources—especially `src/native/graph_runtime.cpp` and `src/amp/graph.py`—before proposing changes. Respect the project’s depth and recorded experiments when suggesting alternatives.
- When tests fail or stubs appear, investigate and resolve them; dormant scaffolding invites bitrot.
- Document any deviations from this guidance immediately, including rationale and plan to realign.
