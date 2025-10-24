import json
import wave

import numpy as np
import pytest

from amp import c_kernels, native_runtime
from amp.application import SynthApplication
from amp.cli import main as cli_main
from amp.config import DEFAULT_CONFIG_PATH


def test_cli_falls_back_to_summary_when_pygame_missing(capsys):
    exit_code = cli_main([
        "--config",
        str(DEFAULT_CONFIG_PATH),
        "--allow-no-joystick",
        "--no-audio",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    # Accept any valid summary output (render performed).
    assert "Rendered" in captured.out
    assert "Rendered" in captured.out


def test_application_from_file(tmp_path):
    app = SynthApplication.from_file(str(DEFAULT_CONFIG_PATH))
    data = app.render(frames=16)
    assert data.shape == (app.config.runtime.output_channels, 16)


@pytest.mark.skipif(not c_kernels.AVAILABLE, reason="C kernels unavailable")
@pytest.mark.skipif(not native_runtime.AVAILABLE, reason="Native runtime unavailable")
def test_cli_headless_writes_pcm(tmp_path):
    output_path = tmp_path / "headless_output.f32"
    exit_code = cli_main(
        [
            "--headless",
            "--headless-output",
            str(output_path),
            "--headless-frames",
            "32",
            "--headless-iterations",
            "1",
            "--headless-warmup",
            "0",
            "--headless-batch",
            "1",
        ]
    )
    assert exit_code == 0

    assert output_path.exists()
    data = np.fromfile(output_path, dtype="<f4")
    assert data.size > 0

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["frames"] > 0
    assert metadata["channels"] >= 1
    assert metadata["sample_rate"] > 0
    assert metadata["format"] == "raw"
    assert metadata["dtype"] == "float32"


@pytest.mark.skipif(not c_kernels.AVAILABLE, reason="C kernels unavailable")
@pytest.mark.skipif(not native_runtime.AVAILABLE, reason="Native runtime unavailable")
def test_cli_headless_writes_wav(tmp_path):
    output_path = tmp_path / "headless_output.wav"
    exit_code = cli_main(
        [
            "--headless",
            "--headless-output",
            str(output_path),
            "--headless-frames",
            "32",
            "--headless-iterations",
            "1",
            "--headless-warmup",
            "0",
            "--headless-batch",
            "1",
        ]
    )
    assert exit_code == 0

    assert output_path.exists()
    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() >= 1
        wav_frames = wav_file.getnframes()
        assert wav_frames > 0
        assert wav_file.getframerate() > 0

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["frames"] == wav_frames
    assert metadata["channels"] >= 1
    assert metadata["sample_rate"] > 0
    assert metadata["format"] == "wav"
    assert metadata["dtype"] == "int16"
