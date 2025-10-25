import traceback

from amp.graph import AudioGraph
from amp.nodes import FFTDivisionNode, MixNode, OscNode, ParametricDriverNode
from amp.native_runtime import NativeGraphExecutor


def build_graph(sample_rate: int) -> AudioGraph:
    graph = AudioGraph(sample_rate=sample_rate, output_channels=1)
    driver = ParametricDriverNode("driver", mode="piezo")
    osc = OscNode("osc", wave="saw", mode="integrator", accept_reset=False, integration_leak=0.997, integration_gain=0.5)
    mix = MixNode("mix", params={"channels": 1})
    fft = FFTDivisionNode("fft", params={"window_size": 512, "oversample_ratio": 1, "declared_delay": 511, "supports_v2": True})
    graph.add_node(driver)
    graph.add_node(osc)
    graph.add_node(mix)
    graph.add_node(fft)
    graph.connect_mod("driver", "osc", "freq", scale=40.0, mode="add")
    graph.connect_audio("osc", "mix")
    graph.connect_audio("mix", "fft")
    graph.set_sink("mix")
    return graph


def main() -> None:
    sr = 48000
    graph = build_graph(sr)
    with NativeGraphExecutor(graph) as executor:
        frames = 256
        import numpy as np

        driver_freq = np.full(frames, 2.0)
        driver_amp = np.ones(frames)
        osc_freq = np.full(frames, 330.0)
        osc_amp = np.full(frames, 0.5)
        fft_defaults = {
            "divisor": np.ones(frames),
            "divisor_imag": np.zeros(frames),
            "phase_offset": np.zeros(frames),
            "lower_band": np.zeros(frames),
            "upper_band": np.ones(frames),
            "filter_intensity": np.ones(frames),
            "stabilizer": np.full(frames, 1.0e-9),
        }

        def pack(name, values):
            arr = np.asarray(values, dtype=np.float64)
            return np.require(arr[np.newaxis, np.newaxis, :], requirements=("C",))

        params = {
            "driver": {"frequency": pack("f", driver_freq), "amplitude": pack("a", driver_amp)},
            "osc": {"freq": pack("o", osc_freq), "amp": pack("oa", osc_amp)},
            "fft": {key: pack(key, value) for key, value in fft_defaults.items()},
        }

        print("running block...")
        pcm = executor.run_block(frames, sr, base_params=params)
        print("block done", pcm.shape)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
