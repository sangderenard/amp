import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "render_joystick_audio.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_joystick_audio", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_main_uses_defaults(tmp_path, monkeypatch):
    module = load_module()
    script_path = tmp_path / "song.json"
    script_path.write_text("{}", encoding="utf-8")

    captured_kwargs = {}

    def fake_run_app(**kwargs):
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(module, "run_app", fake_run_app)

    exit_code = module.main([str(script_path)])

    assert exit_code == 0
    assert captured_kwargs["headless"] is True
    assert captured_kwargs["no_audio"] is True
    assert captured_kwargs["headless_joystick_mode"] == "switch"
    assert captured_kwargs["headless_joystick_script"] == str(script_path.resolve())
    expected_output = script_path.with_suffix(".wav").resolve()
    assert captured_kwargs["headless_output"] == str(expected_output)


def test_main_respects_overrides(tmp_path, monkeypatch):
    module = load_module()
    script_path = tmp_path / "song.json"
    script_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "custom_output.raw"

    captured_kwargs = {}

    def fake_run_app(**kwargs):
        captured_kwargs.update(kwargs)
        return 5

    monkeypatch.setattr(module, "run_app", fake_run_app)

    exit_code = module.main(
        [
            str(script_path),
            "--output",
            str(output_path),
            "--frames",
            "512",
            "--iterations",
            "3",
            "--warmup",
            "1",
            "--batch",
            "4",
            "--alpha",
            "0.1",
            "--joystick-mode",
            "axis",
            "--allow-no-joystick",
            "--with-audio",
        ]
    )

    assert exit_code == 5
    assert captured_kwargs["headless_frames"] == 512
    assert captured_kwargs["headless_iterations"] == 3
    assert captured_kwargs["headless_warmup"] == 1
    assert captured_kwargs["headless_batch"] == 4
    assert pytest.approx(captured_kwargs["headless_alpha"], rel=0, abs=1e-9) == 0.1
    assert captured_kwargs["headless_joystick_mode"] == "axis"
    assert captured_kwargs["allow_no_joystick"] is True
    assert captured_kwargs["no_audio"] is False
    coerced_output = Path(captured_kwargs["headless_output"])
    assert coerced_output.suffix == ".wav"
    assert captured_kwargs["headless_joystick_script"] == str(script_path.resolve())
    assert exit_code == 5


def test_main_errors_when_script_missing(monkeypatch):
    module = load_module()

    with pytest.raises(SystemExit):
        module.main(["nonexistent.json"])
