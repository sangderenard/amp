import time

import numpy as np
import pytest

from amp import nodes
from amp.application import SynthApplication
from amp.config import AppConfig, GraphConfig, NodeConfig, ConnectionConfig, RuntimeConfig
from amp.graph import RAW_DTYPE, AudioGraph


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


def test_mod_connection_from_config_uses_named_channel() -> None:
    runtime = RuntimeConfig(frames_per_chunk=64, output_channels=1)
    graph_cfg = GraphConfig(
        nodes=[
            NodeConfig(
                name="ctrl",
                type="controller",
                params={"outputs": {"amp": "signals['amp']"}},
            ),
            NodeConfig(name="osc", type="sine_oscillator", params={"frequency": 220.0, "amplitude": 0.2}),
            NodeConfig(name="mix", type="mix", params={"channels": 1}),
        ],
        connections=[
            ConnectionConfig(
                source="ctrl",
                target="osc",
                kind="mod",
                param="amplitude",
                scale=0.5,
                mode="add",
                channel="amp",
            ),
            ConnectionConfig(source="osc", target="mix"),
        ],
        sink="mix",
    )
    config = AppConfig(sample_rate=48000, runtime=runtime, graph=graph_cfg)

    graph = AudioGraph.from_config(config.graph, config.sample_rate, config.runtime.output_channels)
    mods = graph.mod_connections("osc")
    assert len(mods) == 1
    entry = mods[0]
    assert entry.source == "ctrl"
    assert entry.target == "osc"
    assert entry.param == "amplitude"
    assert entry.mode == "add"
    assert entry.scale == 0.5
    assert entry.channel == 0


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


def test_topo_order_large_graph_is_deterministic_and_fast() -> None:
    graph = AudioGraph(sample_rate=48000)
    node_names = [f"node_{idx}" for idx in range(200)]
    for name in node_names:
        graph.add_node(nodes.ConstantNode(name, params={"channels": 1, "value": 0.0}))

    for idx in range(10, len(node_names)):
        graph.connect_audio(node_names[idx - 10], node_names[idx])

    graph.set_sink(node_names[-1])

    expected_order = node_names
    first_order = list(graph._topo_order())
    second_order = list(graph._topo_order())
    assert first_order == expected_order
    assert second_order == expected_order

    start = time.perf_counter()
    for _ in range(5):
        graph.render_block(32)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"rendering took too long: {elapsed:.3f}s"
