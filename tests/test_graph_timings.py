import numpy as np
import pytest

from amp import nodes
from amp.graph import AudioGraph


def test_render_block_records_node_timings():
    graph = AudioGraph(sample_rate=48_000)
    graph.add_node(nodes.ConstantNode("source", params={"channels": 1, "value": 0.5}))
    graph.add_node(nodes.MixNode("mix", params={"channels": 1}))
    graph.set_sink("mix")
    graph.connect_audio("source", "mix")

    output = graph.render_block(16)
    assert output.shape == (1, 1, 16)

    graph._last_node_timings["source"] = 0.001
    graph._last_node_timings["mix"] = 0.003

    timings = graph.last_node_timings
    timings["source"] = 0.0
    fresh_timings = graph.last_node_timings
    assert fresh_timings["source"] == pytest.approx(0.001)
