import types

import numpy as np

from amp import app
from amp import state as app_state
from amp import utils


class DummyJoy:
    def get_numaxes(self):
        return 6


def _fake_pygame():
    module = types.SimpleNamespace()
    module.K_m = 0
    module.K_k = 1
    module.K_x = 2
    module.K_y = 3
    module.K_b = 4
    module.K_n = 5
    module.K_z = 6
    module.K_PERIOD = 7
    module.K_COMMA = 8
    module.K_SLASH = 9
    return module


def test_default_graph_routes_envelopes_directly_to_oscillators():
    state = app_state.build_default_state(joy=DummyJoy(), pygame=_fake_pygame())
    graph, envelope_names, amp_mod_names = app.build_runtime_graph(48000, state)

    assert len(envelope_names) == 3
    assert amp_mod_names == []
    assert "keyboard_ctrl" in graph._nodes
    assert "joystick_ctrl" in graph._nodes

    for env_name in envelope_names:
        mods = graph.mod_connections(env_name)
        for param in ("velocity", "gate", "drone", "trigger"):
            sources = {conn.source for conn in mods if conn.param == param}
            assert "keyboard_ctrl" in sources, f"{env_name}.{param} missing keyboard controller"
            assert "joystick_ctrl" in sources, f"{env_name}.{param} missing joystick controller"

    if "pitch" in graph._nodes:
        pitch_mods = graph.mod_connections("pitch")
        pitch_inputs = {conn.param: conn.source for conn in pitch_mods}
        assert pitch_inputs.get("input") == "joystick_ctrl"
        assert pitch_inputs.get("span_oct") == "joystick_ctrl"
        assert pitch_inputs.get("root_midi") == "joystick_ctrl"

    for osc_name in ("osc1", "osc2", "osc3"):
        mods = graph.mod_connections(osc_name)
        env_links = [
            entry for entry in mods if entry.param == "amp" and entry.source in envelope_names
        ]
        assert env_links, f"{osc_name} missing envelope amplitude"
        for entry in env_links:
            assert entry.channel == 0
            assert entry.mode == "add"
        reset_links = [
            entry for entry in mods if entry.param == "reset" and entry.source in envelope_names
        ]
        if reset_links:
            for entry in reset_links:
                assert entry.channel == 1
                assert entry.mode == "add"


def test_triggered_envelopes_produce_audible_output():
    state = app_state.build_default_state(joy=DummyJoy(), pygame=_fake_pygame())
    graph, envelope_names, amp_mod_names = app.build_runtime_graph(48000, state)

    frames = 512
    freq = utils.as_BCF(np.full(frames, 220.0), 1, 1, frames, name="freq")
    silence = utils.as_BCF(np.zeros(frames), 1, 1, frames, name="silence")
    velocity = utils.as_BCF(np.full(frames, 0.75), 1, 1, frames, name="velocity")
    trigger = np.zeros(frames, dtype=float)
    trigger[0] = 1.0
    trigger_bcf = utils.as_BCF(trigger, 1, 1, frames, name="trigger")
    ones = utils.as_BCF(np.ones(frames), 1, 1, frames, name="send_reset")
    root = utils.as_BCF(np.full(frames, 60.0), 1, 1, frames, name="root")
    span = utils.as_BCF(np.full(frames, 2.0), 1, 1, frames, name="span")
    cutoff = utils.as_BCF(np.full(frames, 1200.0), 1, 1, frames, name="cutoff")
    q_values = utils.as_BCF(np.full(frames, 0.8), 1, 1, frames, name="q")

    base_params = {"_B": 1, "_C": 1}
    base_params["keyboard_ctrl"] = {
        "trigger": silence,
        "gate": silence,
        "drone": silence,
        "velocity": silence,
    }
    base_params["joystick_ctrl"] = {
        "trigger": trigger_bcf,
        "gate": silence,
        "drone": silence,
        "velocity": velocity,
        "pitch_input": silence,
        "pitch_span": span,
        "pitch_root": root,
        "cutoff": cutoff,
        "q": q_values,
    }
    for osc_name in ("osc1", "osc2", "osc3"):
        if osc_name in graph._nodes:
            base_params.setdefault(osc_name, {})
            base_params[osc_name]["freq"] = freq
            base_params[osc_name]["amp"] = silence
    for env_name in envelope_names:
        base_params[env_name] = {"send_reset": ones}

    output = graph.render(frames, 48000, base_params)
    assert np.max(np.abs(output)) > 1e-3


def test_envelopes_drive_amp_and_reset_control_channels():
    state = app_state.build_default_state(joy=DummyJoy(), pygame=_fake_pygame())
    graph, envelope_names, amp_mod_names = app.build_runtime_graph(48000, state)

    send_resets = state["envelope_params"].get("send_resets", True)

    for osc_name in ("osc1", "osc2", "osc3"):
        if osc_name not in graph._nodes:
            continue
        oscillator = graph._nodes[osc_name]
        amp_conns = [
            conn
            for conn in graph.mod_connections(osc_name)
            if conn.param == "amp" and conn.source in envelope_names
        ]
        assert amp_conns, f"{osc_name} missing envelope amplitude"
        for conn in amp_conns:
            assert conn.channel == 0
            assert conn.mode == "add"

        reset_conns = [
            conn
            for conn in graph.mod_connections(osc_name)
            if conn.source in envelope_names and conn.param == "reset"
        ]
        if getattr(oscillator, "accept_reset", False) and send_resets:
            assert reset_conns, f"{osc_name} should receive reset pulses"
            for conn in reset_conns:
                assert conn.channel == 1
                assert conn.mode == "add"
        else:
            assert not reset_conns

def test_mod_sources_precede_oscillators_in_execution_order():
    state = app_state.build_default_state(joy=DummyJoy(), pygame=_fake_pygame())
    graph, envelope_names, amp_mod_names = app.build_runtime_graph(48000, state)

    order = [node.name for node in graph.ordered_nodes]
    for osc_name in ("osc1", "osc2", "osc3"):
        if osc_name not in order:
            continue
        osc_idx = order.index(osc_name)
        for conn in graph.mod_connections(osc_name):
            source_idx = order.index(conn.source)
            assert source_idx < osc_idx, (
                f"Modulation source {conn.source} should execute before {osc_name}"
            )
