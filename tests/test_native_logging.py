from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cffi")


def _pythonpath() -> str:
    return str(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess path quoting differs on Windows")
def test_agent_benchmark_respects_native_logging(tmp_path: Path) -> None:
    script = r"""
import json
import shutil
from pathlib import Path

from amp import diagnostics
from amp.system import benchmark_default_graph


def run(enabled: bool) -> None:
    diagnostics.enable_py_c_logging(enabled)
    benchmark_default_graph(
        frames=32,
        iterations=1,
        sample_rate=44100.0,
        ema_alpha=0.1,
        warmup=0,
        joystick_mode="switch",
        joystick_script_path=None,
    )
    logs = Path("logs")
    entries = []
    if logs.exists():
        entries = sorted(p.name for p in logs.iterdir())
    print(json.dumps({"enabled": enabled, "logs": entries}))


logs_dir = Path("logs")
if logs_dir.exists():
    shutil.rmtree(logs_dir)

run(False)
if logs_dir.exists():
    shutil.rmtree(logs_dir)

run(True)
if logs_dir.exists():
    shutil.rmtree(logs_dir)

run(False)
"""

    repo_root = Path(__file__).resolve().parents[1]
    for pattern in ("_amp_graph_runtime.cpython-*.so", "_amp_ckernels_cffi.cpython-*.so"):
        for path in (repo_root / "src" / "amp").glob(pattern):
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
    for pattern in ("_amp_graph_runtime.cpython-*.so", "_amp_ckernels_cffi.cpython-*.so"):
        for path in repo_root.glob(pattern):
            path.unlink(missing_ok=True)  # type: ignore[arg-type]

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["AMP_NATIVE_DIAGNOSTICS_BUILD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
    )

    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")]
    assert len(lines) == 3, result.stdout

    disabled_first, enabled, disabled_second = lines

    assert disabled_first["enabled"] is False
    assert disabled_first["logs"] == []

    expected_native_logs = {
        "native_alloc_trace.log",
        "native_c_calls.log",
        "native_c_generated.log",
        "native_mem_ops.log",
    }
    assert enabled["enabled"] is True
    assert expected_native_logs.issubset(set(enabled["logs"]))

    assert disabled_second["enabled"] is False
    assert disabled_second["logs"] == []
