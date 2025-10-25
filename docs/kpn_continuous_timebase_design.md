"""KPN continuous time-base handoff design note.

This document captures the current signal routing in ``scripts/demo_kpn_kpn_native_correct.py``
and evaluates candidate time-scaling algorithms for a continuous kernel phase network (KPN)
handoff between oscillator and driver authorities.
"""

# Overview

The demo graph wires a native-only signal chain with two key paths:

* **Modulation path** – ``OscillatorPitchNode`` (``pitch_programmer``) publishes smoothed pitch
  values that modulate the ``ParametricDriverNode`` frequency input.
* **Audio path** – ``ParametricDriverNode`` (``driver``) renders the transducer waveform that feeds
  the ``OscNode`` (``osc_master``). The oscillator's audio output flows into ``MixNode`` (``mix``),
  which acts as the sink, and ``mix`` also feeds an ``FFTDivisionNode`` tap for offline analysis.

During block rendering the driver receives three envelopes (frequency, amplitude, render_mode)
and the oscillator receives frequency, amplitude, and slew envelopes. The pitch programmer also
tracks a per-block slew limit so that when the driver temporarily holds position its catch-up ramp
remains bounded.

# Pitch authority controls

The demo exposes coarse authority switches that determine which element (driver or oscillator)
actively follows the pitch program:

* ``--pitch-authority`` – values ``oscillator``, ``driver``, ``both``, or ``manual`` determine which
  unit advances the schedule.
* ``--driver-pitch-mode`` / ``--oscillator-pitch-mode`` – ``follow`` or ``hold`` overrides that
  selectively freeze one side when ``--pitch-authority=manual``.
* ``--pitch-direct-depth`` – optional blend that lets the oscillator deviate from the driver
  handoff using the normalised pitch expression; useful when evaluating dual authority modes.
* ``--pitch-slew`` and ``--op-amp-slew`` – independent slew ceilings for the pitch programmer and
  the oscillator's follow loop.

``--pitch-authority=both`` splits the render in two halves: the driver holds while the oscillator
  leads during the first half, then roles swap. ``render_mode`` is biased (≤0.3 vs ≥0.7) so the
  driver/oscillator blend matches the active authority.

# Continuous time-scaling algorithm options

The forthcoming continuous KPN requires a time-base reconciliation layer that can stretch or
compress driver-side PCM while honouring driver vs. oscillator authority. Three families of
algorithms are promising:

## Synchronous Overlap-Add (SOLA)

* **Pros** – Low algorithmic latency (search window ≈ hop size), minimal frequency smearing at
  modest stretch ratios, inexpensive to implement in native code.
* **Cons** – Requires reliable pitch-synchronous cues to avoid phasing; sensitive to rapid authority
  toggles because the overlap alignment is short.
* **Best fit** – Oscillator-led segments where the oscillator dictates the fundamental, allowing the
  SOLA alignment search to lock onto a predictable period.

## Waveform Similarity Overlap-Add (WSOLA)

* **Pros** – Extends SOLA with similarity metrics that survive moderate transients; better handles
  varying stretch ratios and mid-band content, making it robust when the driver momentarily leads.
* **Cons** – Slightly higher computational cost and buffering (needs candidate windows + similarity
  scoring), and requires well-tuned thresholds to avoid artefacts.
* **Best fit** – Driver-led stretches where phase continuity must tolerate the driver's excitations
  (e.g., piezo resonances). Also useful during authority transitions because the similarity window
  can be biased by recent oscillator periods.

## Phase Vocoder (with phase-locking)

* **Pros** – Flexible across wide time-scale factors, preserves long-term phase through accumulated
  bin phases, and integrates naturally with the existing FFT analysis tap.
* **Cons** – Introduces higher latency (analysis & synthesis windows), needs careful magnitude
  preservation to avoid transient dulling, and is costlier in both CPU and memory. Requires
  phase-locking or identity phase coupling to mitigate transient smearing.
* **Best fit** – Hybrid or ``both`` authority regimes where we can amortise the extra latency and
  leverage spectral-domain control to morph between oscillator and driver authority weights.

# Parameter set and latency budget

The continuous time-stretch module should expose:

* ``analysis_window`` / ``synthesis_window`` – size in frames; drives algorithmic latency.
* ``hop_size`` – controls block advance and overlap; needs to stay aligned with the graph block size
  (currently default 512).
* ``similarity_span`` – number of candidate frames for SOLA/WSOLA alignment.
* ``authority_bias`` – scalar [0,1] weighting that shifts similarity matching towards oscillator or
  driver phase cues.
* ``slew_limit`` – ceiling on instantaneous stretch factor change to respect existing slew budgets.
* ``blend_crossfade`` – crossfade duration when switching authority or blend ratios.

Latency must remain below a single render block when oscillator-led so that the oscillator receives
prompt feedback (<512 frames at 48 kHz ≈ 10.7 ms). Driver-led or phase-vocoder operation can accept
up to two blocks (~21 ms) provided ``render_mode`` compensates the added delay on the oscillator
path. The design should keep total buffering symmetrical so ``MixNode`` sees aligned PCM when
authorities hand off.

# Mapping algorithms to authority modes

* **Oscillator-led** – Prefer SOLA with small ``similarity_span`` tuned to the oscillator period and
  aggressive slew limiting. The driver audio should be time-warped to chase the oscillator while the
  oscillator receives minimal delay. ``authority_bias`` ≈ 0.2 keeps phase matching anchored to the
  oscillator history.
* **Driver-led** – Deploy WSOLA with a broader ``similarity_span`` and a higher ``authority_bias``
  (≈0.8) so that driver resonances govern alignment. Allow one extra hop of latency to stabilise the
  driver waveform before the oscillator adjusts its integrator.
* **Split/Both** – When ``--pitch-authority=both``, start in SOLA and pre-fill the phase vocoder state
  so that the mid-render transition can switch to spectral processing without discontinuities.
  ``blend_crossfade`` ensures the render_mode handoff matches the time-stretch mode change.
* **Manual hold/follow** – If either side holds, freeze the time-scale factor (1.0) and bypass
  overlap-add updates to guarantee deterministic re-entry once follow resumes. The module should
  continue emitting buffered audio without invoking Python fallbacks.

This plan aligns the demo's authority toggles with native-friendly time-stretch building blocks
while respecting the project's requirement to avoid Python fallbacks for production signal paths.
