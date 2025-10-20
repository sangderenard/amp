from pathlib import Path

from amp.application import SynthApplication
from amp.cli import main as cli_main
from amp.config import DEFAULT_CONFIG_PATH


def test_cli_headless_renders(tmp_path, capsys):
    exit_code = cli_main(["--headless", "--config", str(DEFAULT_CONFIG_PATH)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Joystick" in captured.out
    assert "Rendered" in captured.out


def test_application_from_file(tmp_path):
    app = SynthApplication.from_file(str(DEFAULT_CONFIG_PATH))
    data = app.render(frames=16)
    assert data.shape == (app.config.runtime.output_channels, 16)
