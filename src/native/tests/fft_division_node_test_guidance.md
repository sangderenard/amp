# FFT Division Node Test Guidance

This note documents how `test_fft_division_node.cpp` drives the FFT division node and how to extend the harness without polluting the KPN runtime. It is the authoritative reference for test behaviour and guardrails.

## Core principles
- Exercise the same execution path the runtime uses: invoke `amp_run_node_v2`, rely on mailbox/tap delivery, and let ControlDelay history govern readiness. Do **not** introduce alternate pipelines, bypasses, or “smoke” harnesses.
- Keep harness concerns out of the node algorithm: instrumentation, synthetic signals, and tap verification live in the test files and must not leak into the production node implementation.
- Maintain the mailbox contract: populate outputs through persistent mailbox chains, block on `amp_tap_cache_block_until_ready`, then copy via `PopulateLegacy*FromMailbox` helpers. Do not read unpublished internal buffers.
- Python fallbacks are **not** permitted; only the primary native path is acceptable.

## Single-shot path expectations
- Input is preconditioned through an FFT roundtrip to yield a well-behaved waveform of `frames = window_size * 2`.
- Verification reads committed PCM and spectral rows from the staged tap cache after the mailbox chains are ready. Do not reintroduce sleeps; use tap cache blocking helpers instead.
- Metrics are treated as diagnostics only; demand-driven behaviour may alter measured delay.

## Streaming path expectations
- Streaming drives `run_fft_node_streaming` with chunked audio; each chunk must be indexed relative to its own buffer when filling the PCM backlog. Never reuse a global frame cursor to address `audio_base` for chunk data (this previously zeroed later chunks).
- The simulator uses the same chunk cadence as the node (PCM + zero-tail) to produce expected PCM and spectra. Keep the simulator aligned with any runtime changes to tail handling or flush policy.
- Chunk count is `(streaming_frames + streaming_chunk - 1) / streaming_chunk`; the test enforces this and checks state retention and per-call metrics.
- When increasing `kStreamingPasses` or `streaming_chunk`, ensure the tap expectations reflect total frames (no truncation) and validate that mailbox cursor advancement matches committed frame counts.

## Extending the harness
- Add new diagnostics or debug logging only inside the test/harness files; avoid modifying `fft_division_nodes.inc` for test convenience.
- If additional taps are introduced, keep their staging/reading logic in the helpers (`fft_division_mailbox_helpers.h`) and block on tap readiness instead of peeking into node internals.
- Preserve deterministic seeding: synthetic signals should be pure functions of frame index, not random draws.
- Any new test mode must continue to record events through ControlDelay and may not bypass the mailbox history.

## What to avoid
- No ad-hoc “smoke” scripts or alternate sampling paths.
- No direct reads of working tensors, wheels, or ring buffers; only consume through mailboxes/taps.
- No Python fallback or surrogate runtimes; the native path is the sole acceptable path.

Following these rules keeps the harness strictly separated from the production algorithm while still validating the authoritative runtime behaviour.*** End Patch
