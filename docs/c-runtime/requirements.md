# C runtime requirements for `amp.nodes`

> See also: [Native graph runtime policy](policy.md) for the overarching rule
> that all graph execution flows must remain in C.

This note captures the runtime-facing expectations for every class defined in
`src/amp/nodes.py`. It focuses on the data arity each node handles, any
persistent state carried across processing blocks, the NumPy primitives the
current Python implementation relies on, and how (or whether) those operations
map onto existing CFFI kernels in `src/amp/c_kernels.py`. Any gaps highlight
where a future C implementation would need new primitives.

## Legend
- **Audio/mod inputs** describe the expected batch (`B`), channel (`C`), and
  frame (`F`) layout. "Mod" covers controller inputs supplied through `params`.
- **Persistent state** lists member fields that survive across calls to
  `process` (or equivalent helpers).
- **NumPy operations** enumerates the concrete NumPy calls used today. Helper
  utilities from `amp.utils` or other modules are mentioned when they wrap
  further array work.
- **CFFI mapping** links to kernels that already exist or flags missing
  primitives that a C port would require.

## Class manifest

### `_EnvelopeGroupState`
- **Audio/mod inputs:** None (manages trigger/gate/drone/velocity bundles for
  grouped envelopes).【F:src/amp/nodes.py†L27-L137】
- **Persistent state:** `members`, `_assignments`, `_next_voice`,
  `_block_token`, `_latched_voice` (re-used across blocks).【F:src/amp/nodes.py†L30-L140】
- **NumPy operations:** `np.empty`, `np.zeros`, `np.zeros_like`, `np.full`,
  boolean comparisons, `np.count_nonzero`, `np.arange`, `reshape`, `np.where`,
  `np.copy`, `np.any`.【F:src/amp/nodes.py†L35-L136】
- **CFFI mapping:** No direct C kernels; a C port would need bespoke stateful
  voice-assignment logic.

### `Node`
- **Audio/mod inputs:** Base class—does not process audio directly.【F:src/amp/nodes.py†L173-L201】
- **Persistent state:** `_block_pool`, `_leases` (track pooled buffers across
  blocks).【F:src/amp/nodes.py†L178-L198】
- **NumPy operations:** Uses pooled buffers via `BlockPool`; only calls
  `ndarray.fill` when zeroing outputs.【F:src/amp/nodes.py†L194-L199】
- **CFFI mapping:** No direct C dependency; buffer pooling must be mirrored in
  any C runtime.

### `ConfigNode`
- **Audio/mod inputs:** Delegates to subclasses; stores configuration dict
  `params`.【F:src/amp/nodes.py†L212-L216】
- **Persistent state:** `params` mapping. No NumPy usage on its own.
- **CFFI mapping:** N/A.

### `ControllerNode`
- **Audio/mod inputs:** Consumes only modulation parameters (evaluated
  expressions). Output is `(B, channels, frames)` modulation buffer.【F:src/amp/nodes.py†L218-L438】
- **Persistent state:** Compiled expressions, worker thread state
  (`_task_queue`, `_latest_output`, `_latest_meta`, `_last_error`).【F:src/amp/nodes.py†L231-L320】
- **NumPy operations:** `np.zeros`, `np.array`, `np.zeros_like`, dtype casts,
  `np.require` (via `assert_BCF`), `np.zeros` allocations for outputs, in-place
  assignment, `np.copy`. Uses Python `eval` for expressions.【F:src/amp/nodes.py†L321-L419】
- **CFFI mapping:** No dedicated kernels. A C port would need an expression
  evaluator or precompiled control plan support.

### `SilenceNode`
- **Audio/mod inputs:** Optional audio input for batch size; emits silence with
  configured channel count.【F:src/amp/nodes.py†L440-L447】
- **Persistent state:** None beyond `channels` config.
- **NumPy operations:** Buffer allocation via `allocate_node_buffer` and
  `fill(0.0)`.
- **CFFI mapping:** Would require a trivial zero-fill primitive (could reuse
  generic buffer allocator).

### `ConstantNode`
- **Audio/mod inputs:** Optional audio input for batch size; outputs constant
  value per channel.【F:src/amp/nodes.py†L450-L460】
- **Persistent state:** `value`, `channels`.
- **NumPy operations:** Buffer allocation and `ndarray.fill` with scalar.
- **CFFI mapping:** Needs scalar-fill primitive.

### `SineOscillatorNode`
- **Audio/mod inputs:** No audio passthrough; parameters `frequency` and
  `amplitude` accepted as scalars or `(B,C,F)` arrays.【F:src/amp/nodes.py†L463-L496】
- **Persistent state:** `_phase` array of shape `(B, C)` retained across
  blocks.【F:src/amp/nodes.py†L469-L494】
- **NumPy operations:** `np.full`, `np.cumsum`, modulo arithmetic, `np.sin`,
  `np.multiply` with broadcasting.【F:src/amp/nodes.py†L489-L496】
- **CFFI mapping:** No sine oscillator kernel today. Would need phase advance
  plus sine evaluation in C (akin to existing `phase_advance_*` for `OscNode`).

### `SafetyNode`
- **Audio/mod inputs:** Audio input `(B, ?, F)` normalised to configured
  channel count via `_match_channels`.【F:src/amp/nodes.py†L500-L531】
- **Persistent state:** `_state` per batch/channel DC filter state, reused
  between blocks.【F:src/amp/nodes.py†L503-L520】
- **NumPy operations:** `_match_channels` uses `np.repeat`, `np.concatenate`;
  `np.require` to ensure C layout; `np.clip` post-filter.【F:src/amp/nodes.py†L514-L531】
- **CFFI mapping:** Uses `c_kernels.dc_block_c`; Python fallback explicitly
  forbidden. No missing primitive.

### `DelayNode`
- **Audio/mod inputs:** Single audio input `(B,C,F)`; no separate modulators
  beyond `params` for configuration (unused currently).【F:src/amp/nodes.py†L533-L558】
- **Persistent state:** Circular buffer `self.buf` `(B,C,delay)` and write index
  `self.w` maintained per batch/channel.【F:src/amp/nodes.py†L536-L557】
- **NumPy operations:** `np.zeros`, `np.arange`, modular arithmetic, `ndarray.take`
  with wrap mode, advanced indexing assignments.【F:src/amp/nodes.py†L536-L557】
- **CFFI mapping:** No dedicated delay kernel exists; C runtime would require a
  circular-buffer primitive.

### `LFONode`
- **Audio/mod inputs:** Optional audio input (if `use_input=True`) otherwise
  generates `(B,1,F)` modulation. Parameters include `rate`, `depth`, optional
  slew time.【F:src/amp/nodes.py†L560-L615】
- **Persistent state:** `phase` scalar, `_slew_z0` accumulator for slew filter.【F:src/amp/nodes.py†L565-L609】
- **NumPy operations:** Wave generation via `np.sin`, `np.where`, `np.abs`,
  modulo arithmetic, `np.maximum`, `np.max`, `np.tile`, scalar multiplies.
  Slew path ensures C-contiguous buffer via `np.require`.【F:src/amp/nodes.py†L578-L605】
- **CFFI mapping:** Relies on `c_kernels.lfo_slew_c`. No other wave kernels are
  used; a C rewrite would need waveform generators if Python helpers are not
  available.

### `EnvelopeModulatorNode`
- **Audio/mod inputs:** No audio dependency; expects trigger/gate/drone/velocity
  modulators shaped `(B,1,F)`. Returns `(B,2,F)` (amplitude & reset).【F:src/amp/nodes.py†L617-L795】
- **Persistent state:** Stage machine arrays (`_stage`, `_value`, `_timer`,
  `_velocity`, `_activation_count`, `_release_start`), gate/drone latches,
  optional group assignments, kernel scratch planes.【F:src/amp/nodes.py†L646-L742】
- **NumPy operations:** `np.full`, `np.zeros`, comparisons for boolean masks,
  `np.any`, `np.empty`, `np.copyto`, constructing boolean masks, basic logic.
- **CFFI mapping:** Calls `c_kernels.envelope_process_c` with Python fallback as
  contingency. No extra primitives needed beyond envelope kernel.

### `PitchQuantizerNode`
- **Audio/mod inputs:** No audio path; consumes controller inputs (`input`,
  `root_midi`, optional `span`) to produce frequency modulations `(B,1,F)`.【F:src/amp/nodes.py†L798-L882】
- **Persistent state:** `_last_freq`, `_last_output`, `_last_target` retained per
  batch for slewing.【F:src/amp/nodes.py†L805-L880】
- **NumPy operations:** `np.power`, `np.copy`, `np.linspace`, cubic ramp
  polynomial (`3t^2-2t^3`), broadcasting arithmetic, `np.rint`. Relies on
  `quantizer` helpers for grid warping.【F:src/amp/nodes.py†L824-L880】
- **CFFI mapping:** No existing C kernels. Slew ramp and quantizer lookups would
  need new implementations in C.

### `AmplifierModulatorNode`
- **Audio/mod inputs:** No audio requirement; combines `base`, `control`, and
  optional `mod` arrays (each `(B,1,F)` after validation).【F:src/amp/nodes.py†L885-L919】
- **Persistent state:** None beyond runtime params.
- **NumPy operations:** `np.zeros`, `np.ones`, `np.clip`, elementwise
  multiplication and broadcasting. Uses `assert_BCF` for shape checks.
- **CFFI mapping:** No kernels; would require generic elementwise multiply/clip.

### `OscNode`
- **Audio/mod inputs:** Generates audio `(B,?,F)` from frequency/amplitude
  modulators; optional pan, portamento, slide, chord, subharmonic, arp, reset
  inputs. Accepts optional audio_in only to infer batch size.【F:src/amp/nodes.py†L922-L1210】
- **Persistent state:** Per-node phase (`self.phase`), portamento state,
  dictionaries for additional voices, arpeggiator state, phase buffers.【F:src/amp/nodes.py†L923-L1180】
- **NumPy operations:** `np.asarray`, `np.clip`, `np.where`, `np.zeros`,
  `np.power`, `np.require`, elementwise multiplies, trigonometric `np.sin`,
  `np.cos`; mixing via addition; pan uses trig functions and `np.empty`. Uses
  BLEP helpers from `amp.utils` for waveform shaping.【F:src/amp/nodes.py†L958-L1209】
- **CFFI mapping:** Already calls `phase_advance_c`, `portamento_smooth_c`, and
  `arp_advance_c`. Waveform generation still Python-side; C runtime would need
  BLEP oscillators (e.g., existing `osc_*_blep_c` functions) wired up.

### `SamplerNode`
- **Audio/mod inputs:** Optional audio input (unused) for batch sizing; outputs
  `(B,1,F)` via sampler callback with per-batch params.【F:src/amp/nodes.py†L1212-L1231】
- **Persistent state:** `sampler` reference only.
- **NumPy operations:** `np.empty`; relies on sampler object's `render_into`.
- **CFFI mapping:** No C kernel wrapper—would need sampler integration on the C
  side.

### `MixNode`
- **Audio/mod inputs:** Requires stacked audio input `(B,C,F)`; optionally
  re-channels to `out_channels`. Mod parameters not yet implemented.【F:src/amp/nodes.py†L1233-L1268】
- **Persistent state:** `stats` (`ClipStats`), RMS/peak history arrays stored as
  Python lists of `np.ndarray`.
- **NumPy operations:** `np.sum`, `np.repeat`, `reshape`, `np.zeros` for
  histories (at init). No compression math yet beyond stats update.
- **CFFI mapping:** No kernels; a C mixdown would require sum/repeat primitives
  and optional stats tracking.

### `BiquadNode`
- **Audio/mod inputs:** Expects audio `(B,?,F)`; uses modulation params
  `cutoff`, `Q` (array-like).【F:src/amp/nodes.py†L1270-L1283】
- **Persistent state:** Delegates to `FilterLPBiquad` (not defined here) for
  per-batch state.
- **NumPy operations:** Minimal—relies on filter implementation; uses `np.zeros`
  implicitly through filter setup.
- **CFFI mapping:** No direct kernel; full biquad would need dedicated C code.

### `GainNode`
- **Audio/mod inputs:** Audio `(B,?,F)` scaled by constant or mod param.
- **Persistent state:** `self.gain`.
- **NumPy operations:** Scalar multiplication and optional elementwise multiply
  by provided gain array.【F:src/amp/nodes.py†L1285-L1291】
- **CFFI mapping:** Needs generic multiply primitive.

### `SourceSwitch`
- **Audio/mod inputs:** Delegates to either `OscNode` or `SamplerNode` based on
  shared `state`; audio input only used for batch inference.【F:src/amp/nodes.py†L1293-L1300】
- **Persistent state:** References to child nodes and shared state mapping.
- **NumPy operations:** None locally.
- **CFFI mapping:** Requires host-side orchestration rather than a kernel.

### `ClipStats`
- **Audio/mod inputs:** Utility class—operates on arbitrary arrays passed to
  `update`.【F:src/amp/nodes.py†L1302-L1310】
- **Persistent state:** `last_max`, `last_min`, `last_clipped`.
- **NumPy operations:** `np.max`, `np.min`, `np.any`, `np.abs`.
- **CFFI mapping:** Would need statistics helpers if required in C runtime.

### `SafetyFilterNode`
- **Audio/mod inputs:** Audio `(B,C,F)` filtered with single-pole high-pass.
- **Persistent state:** `prev_in`, `prev_dc`, `_buffer` (resized per batch).
  【F:src/amp/nodes.py†L1312-L1338】
- **NumPy operations:** `np.zeros`, `np.require`, reallocation for buffer.
- **CFFI mapping:** Uses `c_kernels.safety_filter_c`; Python fallback exists.

### `NormalizerCompressorNode`
- **Audio/mod inputs:** Audio `(B,C,F)`; no extra modulators yet.【F:src/amp/nodes.py†L1340-L1378】
- **Persistent state:** RMS/peak history deques (`list` of arrays), `ClipStats`.
- **NumPy operations:** `np.sqrt`, `np.mean`, `np.max`, `np.abs`, `np.stack`,
  `np.concatenate`, `np.full`, `np.sum`, division, `np.tanh` for compression.
- **CFFI mapping:** No kernels. Requires RMS/peak tracking and tanh compressor
  primitives for C port.

### `SubharmonicLowLifterNode`
- **Audio/mod inputs:** Audio `(B,C,F)` only; parameters are scalars sampled per
  block.【F:src/amp/nodes.py†L1380-L1507】
- **Persistent state:** Numerous per-channel arrays (`hp_y`, `lp_y`, `prev`,
  `sign`, `ff2`, optional `ff4`, envelopes, HP output memory) plus init flag.
- **NumPy operations:** `np.zeros`, `np.ones`, `np.asarray`, `np.reshape`,
  scalar math (`math.exp`, etc.), `np.empty_like` for kernel output.
- **CFFI mapping:** Calls `c_kernels.subharmonic_process_c`; Python fallback is
  disallowed.

## Missing primitives summary
- Oscillator family (sine, BLEP variants, delay, gain, mix, quantiser, sampler,
  normaliser/compressor) currently lack C kernels and would require new
  implementations for a full C runtime.
- Existing kernels (`dc_block`, `lfo_slew`, `envelope_process`,
  `phase_advance`, `portamento_smooth`, `arp_advance`, `safety_filter`,
  `subharmonic_process`) already cover their respective nodes.

