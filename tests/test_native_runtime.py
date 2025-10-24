import struct

import numpy as np
import pytest

from amp import nodes
from amp.graph import AudioGraph
from amp.native_runtime import AVAILABLE as NATIVE_AVAILABLE
from amp.native_runtime import NativeGraphExecutor

NATIVE_ONLY = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native runtime unavailable")


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


@NATIVE_ONLY
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
    version, node_count = struct.unpack_from("<II", plan_blob, 4)
    assert version == 1
    assert node_count == len(simple_graph.ordered_nodes)
    offset = 12
    entries = []
    for _ in range(node_count):
        function_id, name_len, audio_offset, audio_span, param_count = struct.unpack_from(
            "<IIIII", plan_blob, offset
        )
        offset += 20
        name = plan_blob[offset : offset + name_len].decode("utf-8")
        offset += name_len
        params = []
        for _ in range(param_count):
            param_len, param_offset, param_span = struct.unpack_from("<III", plan_blob, offset)
            offset += 12
            param_name = plan_blob[offset : offset + param_len].decode("utf-8")
            offset += param_len
            params.append((param_name, param_offset, param_span))
        entries.append((function_id, name, audio_offset, audio_span, params))

    assert [name for _, name, *_ in entries] == ["source", "gain"]
    source_entry = entries[0]
    gain_entry = entries[1]
    assert source_entry[2:] == (0, 0, [])
    assert gain_entry[2] == 0  # audio_offset advances only when edges exist
    assert gain_entry[3] == 1
    assert gain_entry[4] == []


def test_compiled_plan_encodes_mod_spans() -> None:
    graph = AudioGraph(sample_rate=48_000)
    src = nodes.ConstantNode("src", {"value": 0.5})
    mod = nodes.ConstantNode("mod", {"value": 0.25})
    gain = nodes.GainNode("gain")
    graph.add_node(src)
    graph.add_node(mod)
    graph.add_node(gain)
    graph.connect_audio("src", "gain")
    graph.connect_mod("mod", "gain", "gain", scale=0.5, mode="add")
    graph.set_sink("gain")

    plan_blob = graph.serialize_compiled_plan()
    assert plan_blob.startswith(b"AMPL")
    version, node_count = struct.unpack_from("<II", plan_blob, 4)
    assert version == 1
    assert node_count == 3
    offset = 12
    entries = []
    for _ in range(node_count):
        function_id, name_len, audio_offset, audio_span, param_count = struct.unpack_from(
            "<IIIII", plan_blob, offset
        )
        offset += 20
        name = plan_blob[offset : offset + name_len].decode("utf-8")
        offset += name_len
        params = []
        for _ in range(param_count):
            param_len, param_offset, param_span = struct.unpack_from("<III", plan_blob, offset)
            offset += 12
            param_name = plan_blob[offset : offset + param_len].decode("utf-8")
            offset += param_len
            params.append((param_name, param_offset, param_span))
        entries.append((name, audio_offset, audio_span, params))

    names = [entry[0] for entry in entries]
    assert names.count("gain") == 1
    gain_entry = next(entry for entry in entries if entry[0] == "gain")
    assert gain_entry[1] == 0  # no preceding audio edges => offset at origin
    assert gain_entry[2] == 1
    assert gain_entry[3] == [("gain", 0, 1)]


@NATIVE_ONLY
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
