#pragma once

namespace amp::oscillator {

/**
 * @file oscillator_design_notes.h
 * @brief High-fidelity oscillator design expectations for the AMP KPN/IPLS runtime.
 *
 * This header captures the design brief for oscillators that will be implemented
 * inside the upgraded runtime. Every oscillator node must honor the runtime's
 * C++-first implementation policy—Python fallbacks are prohibited in production
 * graphs and in any tooling meant to exercise the oscillator behaviour.
 */
struct OscillatorDesignBrief {
  /**
   * @brief Signal Rate and Oversampling Strategy.
   *
   * - Oscillators operate at an intentionally overcranked internal rate chosen
   *   for optimal resampling and algorithmic stability.
   * - The oversampling ratio must be tunable per oscillator family and
   *   derivable from IPLS stage metadata so that scheduling can reserve
   *   sufficient compute bandwidth.
   * - Internal kernels are required to maintain numerical headroom for
   *   extremely high signal rates without denormal slowdowns or precision loss.
   */
  static void SignalRateAndOversamplingStrategy() {}

  /**
   * @brief Periodicity Management and Opportunistic Caching.
   *
   * - Each oscillator must include a periodicity detection/enforcement module
   *   that can lock onto steady-state cycles or re-seed unstable waveforms.
   * - When periodicity is confirmed, nodes should generate opportunistic caches
   *   or pre-compiled sample windows so later frames can reuse them with
   *   deterministic phase alignment.
   * - Cache invalidation is driven by control input changes and by
   *   heat/slew/timing wrappers so stale waveforms never leak.
   */
  static void PeriodicityManagementAndCaching() {}

  /**
   * @brief Integration, Simulation, and Driver Library.
   *
   * - Provide a vectorised library of integration and circuit simulation
   *   kernels (explicit/implicit integrators, state-variable, wave digital,
   *   etc.).
   * - Offer ideal digital drivers alongside physical simulations (e.g.,
   *   BLEP/BLAMP, polynomial bandlimited sources) to serve different
   *   accuracy/performance trade-offs.
   * - Kernels must be architected for Eigen-backed batch processing and
   *   convertible into archetypal nodes when IPLS metadata is present.
   */
  static void IntegrationSimulationAndDrivers() {}

  /**
   * @brief Multiband Tensor Topology.
   *
   * - Every oscillator processes data using module-level tensors shaped
   *   (batch, channel, bin, window) to enable intense multiband workflows.
   * - Gather/scatter policies need to map this tensor layout into the runtime's
   *   contiguous FIFO so that bleeding, cross-band filtering, and high-order
   *   modulation are configurable per node instance.
   * - Oscillators must advertise their preferred aggregation strategy to the
   *   scheduler for accurate resource planning.
   */
  static void MultibandTensorTopology() {}

  /**
   * @brief RCI Simulation Hooks.
   *
   * - Inputs and outputs must optionally route through resistor-capacitor-
   *   inductor (RCI) models, supporting per-band impedance differences and
   *   transient behaviour.
   * - The design should allow stacking multiple RCI stages and toggling them
   *   independently for inputs, core oscillation loops, and outputs.
   * - Thermal and energy accounting integrates with the wider node
   *   thermodynamic reporting, ensuring RCI branches contribute to the overall
   *   heat budget.
   */
  static void RciSimulationHooks() {}

  /**
   * @brief Resampling Library for Frame Finalisation.
   *
   * - Bundle a suite of resamplers (polyphase, sinc, wavelet, spline) capable
   *   of converting oversampled frames down to the target KPN network rate
   *   without aliasing.
   * - Resamplers must respect node-level delay declarations so that
   *   downsampled frames align with IPLS timing gates.
   * - Provide hooks to pick resampler quality tiers at runtime, including
   *   deterministic low-latency paths for interactive contexts.
   */
  static void ResamplingLibraryForFinalisation() {}

  /**
   * @brief JSON Preset Orchestration.
   *
   * - Oscillators load instruction sets from JSON presets that can encode
   *   complex configurations—effectively unifying classic synthesiser
   *   topologies inside a single vectorised node.
   * - Presets should describe waveform families, modulation routings, multiband
   *   mappings, RCI stacks, oversampling ratios, and resampler selections.
   * - The loader validates presets against the node contract, ensures
   *   reproducibility (hashable configuration states), and supports hot-reload
   *   without breaking FIFO invariants.
   */
  static void JsonPresetOrchestration() {}

  /**
   * @brief Testing and Verification Expectations.
   *
   * - Unit and integration tests must run through the native runtime path,
   *   exercising oversampling, caching, multiband tensors, RCI blocks, and
   *   preset loading end-to-end.
   * - Deterministic reference renders (e.g., golden BLEP/BLEP-integrated sweeps)
   *   provide regression coverage for periodicity, resampling, and
   *   thermodynamic accounting.
   * - Performance diagnostics measure throughput under extreme oversampling
   *   ratios to guarantee real-time viability before release.
   */
  static void TestingAndVerificationExpectations() {}
};

}  // namespace amp::oscillator

// Contract notes: concrete expectations between an "ideal" oscillator node
// implementation and the AMP KPN runtime. These bullets translate design brief
// goals into actionable requirements for node implementers and the runtime:
//
// - Forward/Backward: Each oscillator node must implement a forward-runtime
//   behaviour (produce frames) and provide an optional reversible/adjoint
//   interface for backward/differentiable workflows. The runtime currently
//   exposes an opaque `state` pointer to support persistent integrator state.
//
// - Frame slices: The runtime may call nodes for single-frame slices (frames
//   == 1) or multi-frame windows. Nodes MUST handle both modes. Inputs and
//   outputs are organised (batch, channel, frame). For slice calls the provided
//   `data` pointers will point to the first frame of the slice.
//
// - Oversampling and resampling: Oscillators that run at an internal
//   oversampled rate must present a deterministic resampling/delay contract.
//   When an oscillator down-samples to KPN network rate, it must declare its
//   nominal delay and honour IPLS timing gates so downstream nodes see
//   aligned frames.
//
// - Param modulation: Parameter tensors and modulation sources follow the
//   same tensor conventions as audio (batch, channel, frame). Modulators are
//   applied using add/multiply semantics; nodes should expose trusted hooks to
//   accept pre-applied modulation tensors from the runtime when available.
//
// - Thermodynamic reporting: Oscillators must expose (via params or a
//   documented side-channel) a scalar 'heat' contribution per frame so higher
//   level scheduling may integrate thermal budgets. Prefer binding this to the
//   reserved `thermo.heat` parameter name so the runtime can surface it without
//   additional negotiation. The runtime will consume this data if available—do
//   not rely on its presence for correctness.
//
// - No Python fallback: Per project policy, production oscillator behaviour
//   and tests must run through the native runtime. Tools or experiments may
//   use higher-level languages for prototyping, but not for authoritative
//   regression tests.


