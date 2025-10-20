import types

from amp import app
from amp import state as app_state


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


def test_default_graph_assigns_envelopes_to_all_oscillators():
    state = app_state.build_default_state(joy=DummyJoy(), pygame=_fake_pygame())
    graph, envelope_names = app.build_runtime_graph(48000, state)

    assert len(envelope_names) == 3

    for osc_name in ("osc1", "osc2", "osc3"):
        mods = graph._mod_inputs.get(osc_name, [])
        amp_sources = [entry for entry in mods if entry["target_param"] == "amp"]
        assert any(entry["source"] in envelope_names for entry in amp_sources), (
            f"{osc_name} has no envelope connected"
        )
