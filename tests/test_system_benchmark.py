import copy

import pandas as pd
import numpy as np
import pytest

pytest.importorskip("cffi")

from amp import app as amp_app
from amp import c_kernels, native_runtime
from amp.system import benchmark_default_graph


def _benchmark_kwargs() -> dict[str, object]:
    return {
        "frames": 64,
        "iterations": 1,
        "sample_rate": 44100.0,
        "ema_alpha": 0.1,
        "warmup": 0,
        "joystick_mode": "switch",
        "joystick_script_path": None,
    }


def test_benchmark_refuses_without_c_kernels(monkeypatch):
    monkeypatch.setattr(c_kernels, "AVAILABLE", False, raising=False)
    monkeypatch.setattr(c_kernels, "UNAVAILABLE_REASON", "toolchain missing", raising=False)
    monkeypatch.setattr(native_runtime, "AVAILABLE", True, raising=False)

    def _fail_build(*_args, **_kwargs):
        raise AssertionError("benchmark should abort before building the graph")

    monkeypatch.setattr(amp_app, "build_runtime_graph", _fail_build)

    with pytest.raises(RuntimeError) as excinfo:
        benchmark_default_graph(**_benchmark_kwargs())
    assert "C kernels" in str(excinfo.value)


def test_benchmark_refuses_without_native_runtime(monkeypatch):
    monkeypatch.setattr(c_kernels, "AVAILABLE", True, raising=False)
    monkeypatch.setattr(native_runtime, "AVAILABLE", False, raising=False)
    monkeypatch.setattr(native_runtime, "UNAVAILABLE_REASON", "native runtime disabled", raising=False)

    def _fail_build(*_args, **_kwargs):
        raise AssertionError("benchmark should abort before building the graph")

    monkeypatch.setattr(amp_app, "build_runtime_graph", _fail_build)

    with pytest.raises(RuntimeError) as excinfo:
        benchmark_default_graph(**_benchmark_kwargs())
    assert "Native graph runtime" in str(excinfo.value)


@pytest.mark.skipif(not c_kernels.AVAILABLE, reason="C kernels unavailable")
@pytest.mark.skipif(not native_runtime.AVAILABLE, reason="Native runtime unavailable")
def test_benchmark_uses_interactive_graph_without_python_fallback(monkeypatch):
    captured: dict[str, object] = {}
    original_build = amp_app.build_runtime_graph

    def _capture_graph(sample_rate, state):
        graph, envelope_names, amp_mod_names = original_build(sample_rate, state)
        captured["graph"] = graph
        captured["state"] = state
        captured["envelopes"] = envelope_names
        captured["amp_mods"] = amp_mod_names
        captured["descriptor"] = graph.serialize_node_descriptors()
        reference_graph, _, _ = original_build(sample_rate, copy.deepcopy(state))
        captured["reference_descriptor"] = reference_graph.serialize_node_descriptors()
        return graph, envelope_names, amp_mod_names

    monkeypatch.setattr(amp_app, "build_runtime_graph", _capture_graph)

    df = benchmark_default_graph(
        frames=64,
        iterations=1,
        sample_rate=44100.0,
        ema_alpha=0.1,
        warmup=0,
        joystick_mode="switch",
        joystick_script_path=None,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    graph = captured.get("graph")
    assert graph is not None, "benchmark did not build a runtime graph"
    assert captured["descriptor"] == captured["reference_descriptor"]
    assert hasattr(graph, "_native_executor")


@pytest.mark.skipif(not c_kernels.AVAILABLE, reason="C kernels unavailable")
@pytest.mark.skipif(not native_runtime.AVAILABLE, reason="Native runtime unavailable")
def test_benchmark_invokes_pcm_consumer():
    captured: list[tuple[float, np.ndarray, float]] = []

    def _pcm_consumer(timestamp: float, block: np.ndarray, sample_rate: float) -> None:
        captured.append((timestamp, np.asarray(block).copy(), float(sample_rate)))

    df = benchmark_default_graph(
        frames=32,
        iterations=1,
        sample_rate=44100.0,
        ema_alpha=0.05,
        warmup=0,
        joystick_mode="switch",
        joystick_script_path=None,
        batch_blocks=1,
        pcm_consumer=_pcm_consumer,
    )

    assert isinstance(df, pd.DataFrame)
    assert captured, "PCM consumer did not receive any buffers"
    timestamp, block, sr = captured[0]
    assert isinstance(timestamp, float)
    assert sr == pytest.approx(44100.0)
    assert isinstance(block, np.ndarray)
    assert block.ndim in (2, 3)
    if block.ndim == 3:
        assert block.shape[0] == 1
        block = block[0]
    assert block.shape[0] > 0
    assert block.shape[1] == 32
