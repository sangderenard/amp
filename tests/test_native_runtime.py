import struct

import numpy as np
import pytest

from amp import nodes
from amp.graph import AudioGraph
from amp.native_runtime import AVAILABLE as NATIVE_AVAILABLE
from amp.native_runtime import NativeGraphExecutor

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native runtime unavailable")


@pytest.fixture
def simple_graph() -> AudioGraph:
    graph = AudioGraph(sample_rate=48_000)
    source = nodes.ConstantNode("source", {"value": 0.25})
    gain = nodes.GainNode("gain")
    graph.add_node(source)
    graph.add_node(gain)
    graph.connect_audio("source", "gain")
    graph.set_sink("gain")
    return graph


def test_graph_render_block_uses_native_runtime(simple_graph: AudioGraph) -> None:
    frames = 16
    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.full((1, 1, frames), 2.0)}}

    output = simple_graph.render_block(frames, sample_rate=48_000, base_params=base_params)

    assert simple_graph._native_executor is not None
    assert output.shape == (1, 1, frames)
    np.testing.assert_allclose(output, 0.5)


def test_serialize_compiled_plan_structure(simple_graph: AudioGraph) -> None:
    plan_blob = simple_graph.serialize_compiled_plan()
    assert plan_blob.startswith(b"AMPL")
    header = struct.unpack_from("<II", plan_blob, 4)
    assert header[0] == 1
    node_count = header[1]
    assert node_count == len(simple_graph.ordered_nodes)


def test_native_executor_runs_directly(simple_graph: AudioGraph) -> None:
    frames = 24
    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.full((1, 1, frames), 1.5)}}
    with NativeGraphExecutor(simple_graph) as executor:
        output = executor.run_block(
            frames,
            sample_rate=48_000.0,
            base_params=base_params,
            control_history_blob=b"",
        )
    assert output.shape == (1, 1, frames)
    np.testing.assert_allclose(output, 0.375)
