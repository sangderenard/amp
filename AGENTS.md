AGENTS GUIDELINES
==================

Policy: Do not create or use ad-hoc "smoke" scripts or tests.

- Rationale: The project has a single authoritative input path for
  controller events: the `AudioGraph` control history (ControlDelay).
  Any test or harness that bypasses or duplicates that history risks
  diverging behaviour between interactive and headless runs.

- Rule: All runtime tests and benchmarking must exercise the real
  code-paths used by the application. Do not add or run scripts named
  or containing the word "smoke" that perform alternate or contrived
  sampling of controls. If a quick manual run is required, run the
  existing benchmark (`amp.system.benchmark_default_graph`) or the
  interactive application in a controlled environment.

- Enforcement: If you spot a file whose name suggests it is a "smoke"
  harness (for example, `_smoke_*.py`), delete it and open a small
  PR describing the reason. This file was removed for that reason.

If you disagree with this policy or need an exception for diagnostics,
open an issue and document the justification and the exact invariant the
diagnostic must preserve (particularly: it must record events into
ControlDelay and not bypass the history).
