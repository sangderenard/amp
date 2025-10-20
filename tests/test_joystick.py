import sys
from types import SimpleNamespace

from amp.application import SynthApplication
from amp.config import AppConfig, GraphConfig, NodeConfig, ConnectionConfig, RuntimeConfig, JoystickConfig
from amp.joystick import JoystickController


def build_minimal_app_config(joystick_enabled: bool) -> AppConfig:
    runtime = RuntimeConfig(
        frames_per_chunk=32,
        output_channels=2,
        joystick=JoystickConfig(
            enabled=joystick_enabled,
            axes={"x": 0},
            buttons={"trigger": 0},
        ),
    )
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
    return AppConfig(sample_rate=44100, runtime=runtime, graph=graph)


def test_application_without_joystick_does_not_create_controller():
    config = build_minimal_app_config(joystick_enabled=False)
    app = SynthApplication.from_config(config)
    assert app.joystick is None
    assert app.joystick_error is None


def test_joystick_unavailable_reports_error(monkeypatch):
    config = build_minimal_app_config(joystick_enabled=True)

    monkeypatch.setitem(sys.modules, "pygame", None)

    app = SynthApplication.from_config(config)
    assert app.joystick is None
    assert app.joystick_error is not None


class _StubJoystick:
    def __init__(self) -> None:
        self._axis = {0: 0.5}
        self._button = {0: 1}

    def init(self) -> None:
        pass

    def get_axis(self, index: int) -> float:
        return self._axis[index]

    def get_button(self, index: int) -> int:
        return self._button[index]

    def quit(self) -> None:
        pass


def test_controller_polls_values(monkeypatch):
    stub = _StubJoystick()

    pygame_stub = SimpleNamespace(
        _init=False,
        joystick=SimpleNamespace(
            _init=False,
            get_init=lambda: False,
            init=lambda: None,
            get_count=lambda: 1,
            Joystick=lambda index: stub,
        ),
        event=SimpleNamespace(pump=lambda: None),
        get_init=lambda: False,
        init=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "pygame", pygame_stub)

    controller = JoystickController.create(JoystickConfig(enabled=True, axes={"x": 0}, buttons={"trigger": 0}))
    state = controller.poll()
    assert state.axes == {"x": 0.5}
    assert state.buttons == {"trigger": True}
    controller.close()


def test_application_with_stubbed_joystick(monkeypatch):
    stub = _StubJoystick()
    pygame_stub = SimpleNamespace(
        _init=False,
        joystick=SimpleNamespace(
            _init=False,
            get_init=lambda: False,
            init=lambda: None,
            get_count=lambda: 1,
            Joystick=lambda index: stub,
        ),
        event=SimpleNamespace(pump=lambda: None),
        get_init=lambda: False,
        init=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "pygame", pygame_stub)
    config = build_minimal_app_config(joystick_enabled=True)

    app = SynthApplication.from_config(config)
    assert app.joystick is not None
    assert app.joystick_error is None
    state = app.poll_joystick()
    assert state is not None
    assert state.axes["x"] == 0.5
    assert state.buttons["trigger"] is True

