import numpy as np
import pytest

pytest.importorskip("cffi")

from amp.graph import AudioGraph
from amp import nodes
from amp.graph_edge_runner import CffiEdgeRunner


@pytest.fixture
def simple_graph():
    graph = AudioGraph(sample_rate=48000)
    source = nodes.ConstantNode("source", {"value": 0.25})
    gain = nodes.GainNode("gain")
    graph.add_node(source)
    graph.add_node(gain)
    graph.connect_audio("source", "gain")
    graph.set_sink("gain")
    return graph


def test_edge_runner_prepares_audio_and_params(simple_graph):
    frames = 8
    runner = CffiEdgeRunner(simple_graph)
    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.full((1, 1, frames), 2.0)}}
    runner.begin_block(frames, sample_rate=48000, base_params=base_params)

    source_inputs = runner.gather_to("source")
    assert source_inputs.cdata.audio.has_audio == 0
    assert source_inputs.cdata.params.count == 0

    runner.set_node_output("source", np.full((1, 1, frames), 0.25, dtype=np.float64))

    gain_inputs = runner.gather_to("gain")
    assert gain_inputs.cdata.audio.has_audio == 1
    assert gain_inputs.cdata.audio.batches == 1
    assert gain_inputs.cdata.audio.channels == 1
    assert gain_inputs.cdata.audio.frames == frames

    audio_buffer = np.frombuffer(
        runner.ffi.buffer(gain_inputs.cdata.audio.data, frames * 8), dtype=np.float64
    )
    assert np.allclose(audio_buffer.reshape(1, 1, frames), 0.25)

    params = gain_inputs.params
    assert "gain" in params
    assert params["gain"].shape == (1, 1, frames)
    assert np.allclose(params["gain"], 2.0)


def test_edge_runner_applies_modulations():
    frames = 4
    graph = AudioGraph(sample_rate=44100)
    source = nodes.ConstantNode("src", {"value": 0.5})
    mod = nodes.ConstantNode("mod", {"value": 1.0})
    gain = nodes.GainNode("gain")
    graph.add_node(source)
    graph.add_node(mod)
    graph.add_node(gain)
    graph.connect_audio("src", "gain")
    graph.connect_mod("mod", "gain", "gain", scale=0.5, mode="add")
    graph.set_sink("gain")

    runner = CffiEdgeRunner(graph)
    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.ones((1, 1, frames))}}
    runner.begin_block(frames, sample_rate=44100, base_params=base_params)

    runner.set_node_output("src", np.full((1, 1, frames), 0.5, dtype=np.float64))
    runner.set_node_output("mod", np.ones((1, 1, frames), dtype=np.float64))

    gain_inputs = runner.gather_to("gain")
    params = gain_inputs.params
    assert "gain" in params
    expected = np.ones((1, 1, frames), dtype=np.float64) + 0.5 * np.ones((1, 1, frames))
    assert np.allclose(params["gain"], expected)

    # Ensure the CFFI view references the same underlying data
    buffer = np.frombuffer(
        runner.ffi.buffer(
            gain_inputs.cdata.params.items[0].data,
            frames * 8,
        ),
        dtype=np.float64,
    )
    assert np.allclose(buffer.reshape(1, 1, frames), expected)


def test_audio_graph_render_block_uses_edge_runner(simple_graph):
    frames = 16
    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.full((1, 1, frames), 2.0)}}

    output = simple_graph.render_block(frames, sample_rate=48000, base_params=base_params)

    assert simple_graph._edge_runner is not None
    assert output.shape == (1, 1, frames)
    assert np.allclose(output, 0.5)


def test_edge_runner_compiled_plan_metadata(simple_graph):
    runner = CffiEdgeRunner(simple_graph)
    plan = runner.describe_compiled_plan()

    assert plan["version"] == 1
    assert plan["node_count"] == len(simple_graph._nodes)
    ordered = runner.ordered_nodes
    assert tuple(node["name"] for node in plan["nodes"]) == ordered
    assert all("params" in node for node in plan["nodes"])


def test_controller_history_fallback_populates_params():
    frames = 6
    graph = AudioGraph(sample_rate=48000)
    controller = nodes.ControllerNode(
        "controller",
        params={
            "outputs": {
                "trigger": "signals['trigger']",
                "gate": "signals['gate']",
            }
        },
    )
    graph.add_node(controller)

    runner = CffiEdgeRunner(graph)
    runner.begin_block(frames, sample_rate=48000, base_params={"_B": 1, "_C": 1})

    timestamp = 0.0
    trigger_curve = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    gate_curve = np.linspace(0.5, 0.6, frames, dtype=np.float64)
    graph.record_control_event(
        timestamp,
        pitch=np.zeros(1, dtype=np.float64),
        envelope=np.zeros(1, dtype=np.float64),
        extras={"trigger": trigger_curve, "gate": gate_curve},
    )
    blob = graph.control_delay.export_control_history_blob(
        timestamp, timestamp + frames / 48000.0 + 0.001
    )

    lib = runner._ensure_c_kernel()
    handle = runner.gather_to("controller")
    descriptor = runner._descriptor_by_name["controller"]
    desc_struct, keepalive = runner._build_descriptor_struct(descriptor)
    out_ptr = runner.ffi.new("double **")
    out_channels = runner.ffi.new("int *")
    state_ptr = runner.ffi.new("void **", runner.ffi.NULL)
    history_buf = runner.ffi.new("uint8_t[]", blob)
    history_handle = lib.amp_load_control_history(history_buf, len(blob), frames)
    assert history_handle != runner.ffi.NULL
    try:
        status = lib.amp_run_node(
            desc_struct,
            handle.cdata,
            int(handle.batches),
            int(handle.channels),
            int(handle.frames),
            float(runner._sample_rate),
            out_ptr,
            out_channels,
            state_ptr,
            history_handle,
        )
        assert status == 0
        assert int(out_channels[0]) == 2
        total = frames * int(out_channels[0])
        buffer = runner.ffi.buffer(out_ptr[0], total * 8)
        array = np.frombuffer(buffer, dtype=np.float64).reshape(1, int(out_channels[0]), frames)
        np.testing.assert_allclose(array[0, 0], trigger_curve)
        np.testing.assert_allclose(array[0, 1], gate_curve)
    finally:
        lib.amp_release_control_history(history_handle)
        if out_ptr[0] != runner.ffi.NULL:
            lib.amp_free(out_ptr[0])
    _ = keepalive
