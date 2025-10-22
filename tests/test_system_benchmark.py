import copy

import pandas as pd
import pytest

pytest.importorskip("cffi")

from amp import app as amp_app
from amp import c_kernels
from amp.system import benchmark_default_graph


@pytest.mark.skipif(not c_kernels.AVAILABLE, reason="C kernels unavailable")
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
    assert hasattr(graph, "_edge_runner")

    assert captured["descriptor"] == captured["reference_descriptor"]

    runner = graph._ensure_edge_runner()
    assert runner.python_fallback_summary() == {}
