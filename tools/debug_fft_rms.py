from pathlib import Path
import numpy as np
import traceback

import importlib.util
from pathlib import Path as _Path

# Load the test module by file path (tests/ isn't a package here)
spec = importlib.util.spec_from_file_location(
    "test_fft_spectral_node",
    _Path(__file__).resolve().parents[1] / "tests" / "test_fft_spectral_node.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    try:
        ffi, lib = mod._load_fft_interface()

        frames = 512
        window_size = 256
        oversample_ratio = 4
        batches = 32
        channels = 384
        slot_count = batches * channels
        sample_rate = 48_000.0

        (
            time_axis,
            release_schedule,
            gate,
            audio_slots,
            curves,
        ) = mod._generate_spectral_instruction_set(frames, batches, channels, window_size)

        descriptor = ffi.new("EdgeRunnerNodeDescriptor *")
        name_buf = ffi.new("char[]", b"fft_spectral")
        type_buf = ffi.new("char[]", b"FFTDivisionNode")
        params_json = (
            "{" "\"window_size\":"
            + str(window_size)
            + ",\"stabilizer\":1e-9,\"epsilon\":1e-12,\"declared_delay\":"
            + str(window_size - 1)
            + ",\"oversample_ratio\":"
            + str(oversample_ratio)
            + ",\"supports_v2\":true}"
        ).encode("utf-8")
        params_buf = ffi.new("char[]", params_json)
        descriptor.name = name_buf
        descriptor.name_len = len(b"fft_spectral")
        descriptor.type_name = type_buf
        descriptor.type_len = len(b"FFTDivisionNode")
        descriptor.params_json = params_buf
        descriptor.params_len = len(params_json)

        audio_flat = mod._flatten_frameslots(audio_slots)
        audio_ptr = ffi.from_buffer("double[]", audio_flat)
        audio_view = ffi.new("EdgeRunnerAudioView *")
        audio_view.has_audio = 1
        audio_view.batches = batches
        audio_view.channels = channels
        audio_view.frames = frames
        audio_view.data = audio_ptr

        param_names = (
            "divisor",
            "divisor_imag",
            "phase_offset",
            "lower_band",
            "upper_band",
            "filter_intensity",
            "stabilizer",
        )

        param_views = ffi.new("EdgeRunnerParamView[]", len(param_names))
        keepalive = [name_buf, type_buf, params_buf, audio_ptr]
        for idx, name in enumerate(param_names):
            array = mod._flatten_frameslots(curves[name])
            param_name_buf = ffi.new("char[]", name.encode("utf-8"))
            buf_ptr = ffi.from_buffer("double[]", array)
            view = param_views[idx]
            view.name = param_name_buf
            view.batches = batches
            view.channels = channels
            view.frames = frames
            view.data = buf_ptr
            keepalive.extend([param_name_buf, array, buf_ptr])

        param_set = ffi.new("EdgeRunnerParamSet *")
        param_set.count = len(param_names)
        param_set.items = param_views

        inputs = ffi.new("EdgeRunnerNodeInputs *")
        inputs.audio = audio_view[0]
        inputs.params = param_set[0]

        out_buffer = ffi.new("double **")
        out_channels = ffi.new("int *")
        state_ptr = ffi.new("void **")
        metrics = ffi.new("AmpNodeMetrics *")

        rc = lib.amp_run_node_v2(
            descriptor,
            inputs,
            batches,
            channels,
            frames,
            sample_rate,
            out_buffer,
            out_channels,
            state_ptr,
            ffi.NULL,
            ffi.cast("AmpExecutionMode", 1),
            metrics,
        )

        print("amp_run_node_v2 rc:", rc)
        if out_buffer[0] != ffi.NULL:
            total = slot_count * frames
            pcm = np.frombuffer(
                ffi.buffer(out_buffer[0], total * np.dtype(np.float64).itemsize),
                dtype=np.float64,
            ).copy()
        else:
            print("No output buffer returned")
            return

        pcm_matrix = pcm.reshape(frames, slot_count)
        rms = np.sqrt(np.mean(pcm_matrix**2, axis=0))
        print("RMS min:", float(rms.min()))
        print("RMS max:", float(rms.max()))
        print("RMS max/min:", float(rms.max() / rms.min()))
        print("RMS sample (first 20):", rms[:20].tolist())
        print("metrics.measured_delay_frames:", int(metrics.measured_delay_frames))
        print("metrics.reserved[:5]:", [float(metrics.reserved[i]) for i in range(5)])
        # Print the last values from the parameter curves for comparison
        phase_flat = curves["phase_offset"].reshape(frames, -1)
        lower_flat = curves["lower_band"].reshape(frames, -1)
        upper_flat = curves["upper_band"].reshape(frames, -1)
        intensity_flat = curves["filter_intensity"].reshape(frames, -1)
        print("phase_flat[-1,-1]:", float(phase_flat[-1, -1]))
        print("lower_flat[-1,-1]:", float(lower_flat[-1, -1]))
        print("upper_flat[-1,-1]:", float(upper_flat[-1, -1]))
        print("intensity_flat[-1,-1]:", float(intensity_flat[-1, -1]))
        print("window_size:", window_size)

    except Exception:
        traceback.print_exc()
    finally:
        try:
            if out_buffer is not None and out_buffer[0] != ffi.NULL:
                lib.amp_free(out_buffer[0])
        except Exception:
            pass
        try:
            if state_ptr is not None and state_ptr[0] != ffi.NULL:
                lib.amp_release_state(state_ptr[0])
        except Exception:
            pass


if __name__ == "__main__":
    main()
