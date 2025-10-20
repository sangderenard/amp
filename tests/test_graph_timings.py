import numpy as np
import pytest

import amp.graph as graph_module
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

    class ConstantSource(nodes.Node):
        def __init__(self, name: str, value: float = 0.0) -> None:
            super().__init__(name)
            self.value = float(value)

        def process(self, frames, sample_rate, audio_in, mods, params):
            return np.full((1, 1, frames), self.value, dtype=np.float64)

    class PassthroughSink(nodes.Node):
        def __init__(self, name: str) -> None:
            super().__init__(name)

        def process(self, frames, sample_rate, audio_in, mods, params):
            assert audio_in is not None
            return audio_in

    graph = AudioGraph(sample_rate=48_000)
    graph.add_node(ConstantSource("source", value=0.5))
    graph.add_node(PassthroughSink("sink"))
    graph.connect_audio("source", "sink")
    graph.set_sink("sink")

    output = graph.render_block(16)
    assert output.shape == (1, 1, 16)

    timings = graph.last_node_timings
    assert timings["source"] == pytest.approx(0.001)
    assert timings["sink"] == pytest.approx(0.003)

    timings["source"] = 0.0
    fresh_timings = graph.last_node_timings
    assert fresh_timings["source"] == pytest.approx(0.001)
