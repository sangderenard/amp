import numpy as np
import pytest

from amp import nodes
from amp.graph import AudioGraph
from amp.native_runtime import AVAILABLE, NativeGraphExecutor, get_graph_runtime_impl


@pytest.mark.skipif(not AVAILABLE, reason="native graph runtime unavailable")
def test_native_executor_matches_python_constant_gain():
    frames = 16
    graph = AudioGraph(sample_rate=48000)
    source = nodes.ConstantNode("src", {"value": 0.25})
    gain = nodes.GainNode("gain")
    graph.add_node(source)
    graph.add_node(gain)
    graph.connect_audio("src", "gain")
    graph.set_sink("gain")

    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.full((1, 1, frames), 2.0, dtype=np.float64)}}

    reference = graph.render_block(frames, sample_rate=48000, base_params=base_params)
    blob = graph.control_delay.export_control_history_blob(0.0, frames / 48000.0)

    with NativeGraphExecutor(graph) as executor:
        result = executor.run_block(
            frames,
            48000.0,
            base_params=base_params,
            control_history_blob=blob,
        )

    np.testing.assert_allclose(result, reference)


@pytest.mark.skipif(not AVAILABLE, reason="native graph runtime unavailable")
def test_native_executor_applies_modulations():
    frames = 12
    sample_rate = 44100
    graph = AudioGraph(sample_rate=sample_rate)
    src = nodes.ConstantNode("src", {"value": 0.5})
    mod = nodes.ConstantNode("mod", {"value": 1.0})
    gain = nodes.GainNode("gain")
    graph.add_node(src)
    graph.add_node(mod)
    graph.add_node(gain)
    graph.connect_audio("src", "gain")
    graph.connect_mod("mod", "gain", "gain", scale=0.5, mode="add")
    graph.set_sink("gain")

    base_params = {"_B": 1, "_C": 1, "gain": {"gain": np.ones((1, 1, frames), dtype=np.float64)}}
    reference = graph.render_block(frames, sample_rate=sample_rate, base_params=base_params)
    blob = graph.control_delay.export_control_history_blob(0.0, frames / sample_rate)

    with NativeGraphExecutor(graph) as executor:
        result = executor.run_block(
            frames,
            sample_rate,
            base_params=base_params,
            control_history_blob=blob,
        )

    np.testing.assert_allclose(result, reference)


@pytest.mark.skipif(not AVAILABLE, reason="native graph runtime unavailable")
def test_native_executor_handles_mix_and_safety_pipeline():
    frames = 64
    sample_rate = 44100
    graph = AudioGraph(sample_rate=sample_rate)
    controller = nodes.ControllerNode(
        "ctrl",
        params={"outputs": {"velocity": "signals['velocity']"}},
    )
    carrier = nodes.SineOscillatorNode(
        "carrier",
        params={"frequency": 220.0, "amplitude": 0.15},
    )
    mix = nodes.MixNode("mix", params={"channels": 2})
    safety = nodes.SafetyNode("safety", params={"channels": 2})

    graph.add_node(controller)
    graph.add_node(carrier)
    graph.add_node(mix)
    graph.add_node(safety)

    graph.connect_audio("carrier", "mix")
    graph.connect_audio("mix", "safety")
    graph.connect_mod("ctrl", "carrier", "amplitude", scale=0.5, mode="add")
    graph.set_sink("safety")

    velocity_curve = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    graph.record_control_event(
        0.0,
        pitch=np.zeros(1, dtype=np.float64),
        envelope=np.zeros(1, dtype=np.float64),
        extras={"velocity": velocity_curve},
    )

    reference = graph.render_block(frames, sample_rate=sample_rate)
    blob = graph.control_delay.export_control_history_blob(0.0, frames / sample_rate)

    with NativeGraphExecutor(graph) as executor:
        result = executor.run_block(
            frames,
            sample_rate,
            control_history_blob=blob,
        )

    np.testing.assert_allclose(result, reference, atol=1e-7)


@pytest.mark.skipif(not AVAILABLE, reason="native graph runtime unavailable")
def test_control_history_ingestion_handles_events():
    frames = 8
    sample_rate = 48000
    graph = AudioGraph(sample_rate=sample_rate)
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
        timestamp,
        timestamp + frames / sample_rate + 0.001,
    )
    ffi, lib = get_graph_runtime_impl()
    buf = ffi.new("uint8_t[]", blob)
    history = lib.amp_graph_history_load(buf, len(blob), frames)
    assert history != ffi.NULL
    lib.amp_graph_history_destroy(history)
