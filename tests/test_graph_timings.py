import numpy as np
import pytest

import amp.graph as graph_module
import amp.graph_edge_runner as graph_edge_runner
from amp import nodes
from amp.graph import AudioGraph


def test_render_block_records_node_timings(monkeypatch):
    times = iter([0.0, 0.001, 0.002, 0.005])

    def fake_perf_counter():
        try:
            return next(times)
        except StopIteration:
            return 0.01

    monkeypatch.setattr(graph_module.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(graph_edge_runner.time, "perf_counter", fake_perf_counter)

    graph = AudioGraph(sample_rate=48_000)
    graph.add_node(nodes.ConstantNode("source", params={"channels": 1, "value": 0.5}))
    graph.add_node(nodes.MixNode("mix", params={"channels": 1}))
    graph.set_sink("mix")
    graph.connect_audio("source", "mix")

    output = graph.render_block(16)
    assert output.shape == (1, 1, 16)

    timings = graph.last_node_timings
    assert timings["source"] == pytest.approx(0.001)
    assert timings["mix"] == pytest.approx(0.003)

    timings["source"] = 0.0
    fresh_timings = graph.last_node_timings
    assert fresh_timings["source"] == pytest.approx(0.001)
