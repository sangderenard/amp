# FFT Division Contrast Audit: `work` vs prior `main`

This audit compares the current `work` branch against the last available
`main` snapshot (commit `9670ab6`) with a focus on changes where code was
removed or behaviour materially altered. The goal is to ensure previously
covered responsibilities remain accounted for in the new form of the
implementation.

## Submodule baseline

* `src/native/fftfree` now points to commit `e272444f…` instead of
  `72628dc…`, indicating the FFT backend dependency advanced alongside the
  worker changes. 【107906†L1-L8】

## Behavioural deltas

### Stage halt handling

* **Old (main `9670ab6`)**: `fftdiv_stage_trigger_halt` logged probe samples
  and then called `abort()`, immediately terminating the process.
  ```c
  // main 9670ab6
  fflush(stderr);
  abort();
  ```
* **New (`work`)**: The handler now initiates a graceful shutdown: it marks
  the worker to stop, enables flush-on-stop, shuts down the mailbox, and
  returns without aborting. 【F:src/native/nodes/fft_division/fft_division_nodes.inc†L239-L266】
* **Implication**: Crash-on-zero-signal enforcement is replaced by a
  cooperative drain. Upstream monitoring should rely on log visibility and
  mailbox shutdown rather than process termination.

### Stage 4 tail release semantics

* **Old (main `9670ab6`)**: `fftdiv_window_release_tail` released filled wheel
  slices after Stage 4 emission, updating tail pointers and emitting TRACE
  logs to advance the circular buffer. ```c
  // main 9670ab6
  tail_slice.valid = false;
  tail_slice.stage4_emitted = false;
  *wheel_tail = (*wheel_tail + 1) % wheel_length;
  *wheel_filled -= 1;
  ```
* **New (`work`)**: The function remains for API compatibility but is a
  no-op to avoid interfering with Stage-1 zero-tail semantics, and Stage 4 no
  longer logs tail advancement. 【F:src/native/nodes/fft_division/fft_division_nodes.inc†L336-L352】【F:src/native/nodes/fft_division/fft_division_nodes.inc†L2890-L2914】
* **Implication**: Automatic tail-drain is disabled; any downstream
  expectation of wheel metadata consumption must now rely on Stage-1
  zero-tail management.

### Analytic zero flush

* **Old (main `9670ab6`)**: `fftdiv_flush_with_zeroes` computed an analytic
  tail plan and injected zero PCM frames through the normal input path to
  drain latency, logging detailed parameters in the process. ```c
  // main 9670ab6
  const FftDivTailPlan tail_plan = fftdiv_calculate_tail_plan(...);
  std::vector<double> zeros(...);
  // Analytic zero-injection: push zeros as normal input
  ```
* **New (`work`)**: The function is stubbed out and immediately returns,
  deferring entirely to Stage-1 zero-tail mechanics. 【F:src/native/nodes/fft_division/fft_division_nodes.inc†L3280-L3285】
* **Implication**: Explicit analytic flushing is gone; latency clearance now
  depends solely on the streaming zero-tail path.

### Worker diagnostics

* Additional stderr diagnostics were added around worker start, condition
  waits, and command submission to trace scheduler behaviour without changing
  control flow. 【F:src/native/nodes/fft_division/fft_division_nodes.inc†L3658-L3715】【F:src/native/nodes/fft_division/fft_division_nodes.inc†L3827-L3890】
* **Implication**: Observability improved; production builds should confirm
  whether these prints are acceptable or need gating.

## Coverage check

* Crash enforcement for zero outputs is now handled via graceful shutdown;
  ensure monitoring/CI expectations align with non-aborting behaviour.
* Tail-release and analytic-flush responsibilities have been removed or
  stubbed; confirm that Stage-1 zero-tail logic fully replaces those
  responsibilities for pipeline drainage and latency management.
