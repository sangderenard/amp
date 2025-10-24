# High-Fidelity Oscillator Design Brief

This brief codifies the expectations for oscillators implemented within the upgraded AMP KPN/IPLS runtime. Every oscillator node must honor the runtime's C++-first implementation policy—**Python fallbacks are prohibited** in production graphs and in any tooling meant to exercise the oscillator behaviour. For the full runtime and node contract, see [`docs/kpn_contract.md`](./kpn_contract.md).

## 1. Signal Rate and Oversampling Strategy
- Oscillators operate at an intentionally overcranked internal rate chosen for optimal resampling and algorithmic stability.
- The oversampling ratio must be tunable per oscillator family and derivable from IPLS stage metadata so that scheduling can reserve sufficient compute bandwidth.
- Internal kernels are required to maintain numerical headroom for extremely high signal rates without denormal slowdowns or precision loss.

## 2. Periodicity Management and Opportunistic Caching
- Each oscillator must include a periodicity detection/enforcement module that can lock onto steady-state cycles or re-seed unstable waveforms.
- When periodicity is confirmed, nodes should generate opportunistic caches or pre-compiled sample windows so later frames can reuse them with deterministic phase alignment.
- Cache invalidation is driven by control input changes and by heat/slew/timing wrappers so stale waveforms never leak.

## 3. Integration, Simulation, and Driver Library
- Provide a vectorised library of integration and circuit simulation kernels (explicit/implicit integrators, state-variable, wave digital, etc.).
- Offer ideal digital drivers alongside physical simulations (e.g., BLEP/BLAMP, polynomial bandlimited sources) to serve different accuracy/performance trade-offs.
- Kernels must be architected for Eigen-backed batch processing and convertible into archetypal nodes when IPLS metadata is present.

## 4. Multiband Tensor Topology
- Every oscillator processes data using module-level tensors shaped `(batch, channel, bin, window)` to enable intense multiband workflows.
- Gather/scatter policies need to map this tensor layout into the runtime's contiguous FIFO so that bleeding, cross-band filtering, and high-order modulation are configurable per node instance.
- Oscillators must advertise their preferred aggregation strategy to the scheduler for accurate resource planning.

## 5. RCI Simulation Hooks
- Inputs and outputs must optionally route through resistor-capacitor-inductor (RCI) models, supporting per-band impedance differences and transient behaviour.
- The design should allow stacking multiple RCI stages and toggling them independently for inputs, core oscillation loops, and outputs.
- Thermal and energy accounting integrates with the wider node thermodynamic reporting, ensuring RCI branches contribute to the overall heat budget.

## 6. Resampling Library for Frame Finalisation
- Bundle a suite of resamplers (polyphase, sinc, wavelet, spline) capable of converting oversampled frames down to the target KPN network rate without aliasing.
- Resamplers must respect node-level delay declarations so that downsampled frames align with IPLS timing gates.
- Provide hooks to pick resampler quality tiers at runtime, including deterministic low-latency paths for interactive contexts.

## 7. JSON Preset Orchestration
- Oscillators load instruction sets from JSON presets that can encode complex configurations—effectively unifying classic synthesiser topologies inside a single vectorised node.
- Presets should describe waveform families, modulation routings, multiband mappings, RCI stacks, oversampling ratios, and resampler selections.
- The loader validates presets against the node contract, ensures reproducibility (hashable configuration states), and supports hot-reload without breaking FIFO invariants.

## 8. Testing and Verification Expectations
- Unit and integration tests must run through the native runtime path, exercising oversampling, caching, multiband tensors, RCI blocks, and preset loading end-to-end.
- Deterministic reference renders (e.g., golden BLEP/BLEP-integrated sweeps) provide regression coverage for periodicity, resampling, and thermodynamic accounting.
- Performance diagnostics measure throughput under extreme oversampling ratios to guarantee real-time viability before release.

These guidelines are mandatory for any oscillator delivered into the AMP audio graph. Future revisions should capture implementation learnings while keeping the no-Python-fallback principle intact.
