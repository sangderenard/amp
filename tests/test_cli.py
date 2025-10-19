from pathlib import Path

from amp import SynthApplication
from amp.cli import main as cli_main
from amp.config import DEFAULT_CONFIG_PATH


def test_cli_renders_default(tmp_path, capsys):
    exit_code = cli_main(["--config", str(DEFAULT_CONFIG_PATH), "--frames", "32", "--summary"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Joystick: disabled" in captured.out
    assert "Rendered 32 frames" in captured.out


def test_application_from_file(tmp_path):
    app = SynthApplication.from_file(str(DEFAULT_CONFIG_PATH))
    data = app.render(frames=16)
    assert data.shape == (app.config.runtime.output_channels, 16)
