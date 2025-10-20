import numpy as np
import pytest

from amp.application import SynthApplication
from amp.config import AppConfig, GraphConfig, NodeConfig, ConnectionConfig, RuntimeConfig
from amp.graph import AudioGraph


def simple_config() -> AppConfig:
    runtime = RuntimeConfig(frames_per_chunk=128, output_channels=2)
    graph = GraphConfig(
        nodes=[
            NodeConfig(name="osc", type="sine_oscillator", params={"frequency": 110.0, "amplitude": 0.1}),
            NodeConfig(name="mix", type="mix", params={"channels": 2}),
            NodeConfig(name="safety", type="safety", params={"channels": 2}),
        ],
        connections=[
            ConnectionConfig(source="osc", target="mix"),
            ConnectionConfig(source="mix", target="safety"),
        ],
        sink="safety",
    )
    return AppConfig(sample_rate=48000, runtime=runtime, graph=graph)


def test_graph_render_shape() -> None:
    config = simple_config()
    graph = AudioGraph.from_config(config.graph, config.sample_rate, config.runtime.output_channels)
    data = graph.render(256)
    assert data.shape == (2, 256)
    assert np.isfinite(data).all()


def test_application_summary_contains_nodes() -> None:
    app = SynthApplication.from_config(simple_config())
    summary = app.summary()
    assert "osc" in summary
    assert "safety" in summary


def test_application_render_defaults_to_runtime_frames() -> None:
    app = SynthApplication.from_config(simple_config())
    data = app.render()
    assert data.shape[1] == app.config.runtime.frames_per_chunk


def test_invalid_node_type_raises() -> None:
    config = simple_config()
    bad_node = NodeConfig(name="mystery", type="unknown", params={})
    config.graph.nodes.append(bad_node)
    with pytest.raises(KeyError):
        AudioGraph.from_config(config.graph, config.sample_rate, config.runtime.output_channels)
