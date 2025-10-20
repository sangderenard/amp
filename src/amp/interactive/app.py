"""Joystick-controlled interactive synthesiser."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from ..graph import AudioGraph
from . import config, menu, nodes, persistence, quantizer, utils


class _NullJoystick:
    """Fallback joystick that returns neutral values."""

    def __init__(self, axes: int = 6, buttons: int = 12) -> None:
        self._axes = max(axes, 0)
        self._buttons = max(buttons, 0)

    def init(self) -> None:  # pragma: no cover - trivial
        pass

    def get_numaxes(self) -> int:
        return self._axes

    def get_numbuttons(self) -> int:
        return self._buttons

    def get_axis(self, index: int) -> float:
        return 0.0

    def get_button(self, index: int) -> int:
        return 0

    def quit(self) -> None:  # pragma: no cover - trivial
        pass


def _load_sampler(state: dict) -> Optional[nodes.Sampler]:
    path = state.get("sample_file")
    if not path or not os.path.isfile(path):
        print("[Sampler] sample.wav not found (sampler available after you add a file).")
        return None
    try:
        sampler = nodes.Sampler(path, loop=True)
        print(f"[Sampler] Loaded '{os.path.basename(path)}' at {sampler.file_sr} Hz")
        return sampler
    except Exception as exc:  # pragma: no cover - depends on local files
        print(f"[Sampler] Disabled: {exc}")
        return None


def _pick_output_device_and_rate(sd):
    try:
        dev = sd.query_devices(None, "output")
        sr = int(dev.get("default_samplerate", 48000))
        return dev["index"], sr
    except Exception:
        for device in sd.query_devices():
            if device.get("max_output_channels", 0) > 0:
                return device["index"], int(device.get("default_samplerate", 48000))
        raise RuntimeError("No audio output device found.")


def run(
    *,
    allow_no_joystick: bool = False,
    no_audio: bool = False,
    headless: bool = False,
    config_path: str | None = None,
) -> int:
    """Launch the original joystick-driven player.

    Parameters
    ----------
    allow_no_joystick:
        When true the app will operate with a neutral virtual joystick so it can
        start for inspection or testing without hardware attached.
    no_audio:
        Skip initialising the sounddevice output stream.  Useful for CI where no
        PortAudio backend is available.
    headless:
        Run the graph without creating a pygame window.  This uses the
        configuration system to render a single buffer for verification.
    config_path:
        Optional configuration override when running in headless mode.
    """

    if headless:
        from ..application import SynthApplication
        from ..config import DEFAULT_CONFIG_PATH, load_configuration

        cfg = load_configuration(config_path or DEFAULT_CONFIG_PATH)
        app = SynthApplication.from_config(cfg)
        print(app.summary())
        buffer = app.render()
        peak = float(buffer.max())
        trough = float(buffer.min())
        print(
            f"Rendered {buffer.shape[1]} frames @ {cfg.sample_rate} Hz "
            f"(peak {peak:.3f}, trough {trough:.3f})"
        )
        if app.joystick_error and not app.joystick:
            print(f"Warning: {app.joystick_error}")
        return 0

    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised only when pygame missing
        print(f"pygame is required for the interactive application: {exc}")
        return 1

    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - exercised only when sounddevice missing
        print(f"sounddevice is required for the interactive application: {exc}")
        return 1

    pygame.init()
    pygame.joystick.init()

    pygame.display.set_caption("Controller Synth (Graph)")
    pygame.display.set_mode((600, 180))
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 14)

    if pygame.joystick.get_count() == 0:
        if not allow_no_joystick:
            print("No joystick. Connect controller and restart.")
            return 1
        joy = _NullJoystick()
        print("[Joystick] Running with virtual controller (all controls neutral).")
    else:
        joy = pygame.joystick.Joystick(0)
        joy.init()

    sample_rate = 44100
    freq_target = 220.0
    freq_current = 220.0
    amp_target = 0.0
    amp_current = 0.0
    cutoff_target = 1000.0
    cutoff_current = 1000.0
    q_target = 0.8
    q_current = 0.8
    gate_momentary = False
    drone_on = False

    waves = ["sine", "square", "saw", "triangle"]

    state = {
        "base_token": "12tet/full",
        "root_midi": 60,
        "free_variant": "continuous",
        "waves": waves,
        "wave_idx": 0,
        "filter_types": ["lowpass", "highpass", "bandpass", "notch", "peaking"],
        "filter_type": "lowpass",
        "filter_axis_cutoff": min(4, joy.get_numaxes() - 1) if joy.get_numaxes() > 4 else 3,
        "filter_axis_q": min(5, joy.get_numaxes() - 1) if joy.get_numaxes() > 5 else 4,
        "peaking_gain_db": 6.0,
        "source_type": "osc",
        "sample_file": os.path.join(os.path.dirname(__file__), "sample.wav"),
        "mod_wave_types": ["sine", "square", "saw", "triangle"],
        "mod_wave_idx": 0,
        "mod_rate_hz": 4.0,
        "mod_depth": 0.5,
        "mod_route": "both",
        "mod_use_input": False,
        "mod_slew_ms": 5.0,
        "keymap": {
            "toggle_menu": pygame.K_m,
            "open_keymap": pygame.K_k,
            "wave_next": pygame.K_x,
            "mode_next": pygame.K_y,
            "drone_toggle": pygame.K_b,
            "toggle_source": pygame.K_n,
            "free_variant_next": pygame.K_z,
            "root_up": pygame.K_PERIOD,
            "root_down": pygame.K_COMMA,
            "root_reset": pygame.K_SLASH,
        },
        "buttonmap": {
            4: {"token": "12tet/full"},
            5: {"token": "FREE"},
        },
        "bumper_priority": [4, 5],
        "double_tap_window": 0.33,
        "free_variant_button": 6,
    }

    persistence.load_mappings(state)

    sampler = _load_sampler(state)

    def build_graph(fs: int, runtime_state: dict) -> AudioGraph:
        use_subharm = runtime_state.get("use_subharm", True)
        use_normalizer = runtime_state.get("use_normalizer", True)
        use_hardclip = runtime_state.get("use_hardclip", False)

        osc_waves = ["sine", "square", "saw"]
        osc_nodes = []
        pan_lfos = []
        am_lfos = []
        for i, wave in enumerate(osc_waves):
            osc = nodes.OscNode(f"osc{i+1}", wave=wave)
            pan_lfo = nodes.LFONode(f"pan_lfo{i+1}", wave="sine", rate_hz=0.2 + 0.1 * i, depth=1.0)
            am_lfo = nodes.LFONode(f"am_lfo{i+1}", wave="sine", rate_hz=4.0 + i, depth=0.5)
            osc_nodes.append(osc)
            pan_lfos.append(pan_lfo)
            am_lfos.append(am_lfo)

        graph_obj = AudioGraph(fs, output_channels=2)
        for osc in osc_nodes:
            graph_obj.add_node(osc)
        for pan in pan_lfos:
            graph_obj.add_node(pan)
        for am in am_lfos:
            graph_obj.add_node(am)

        for i, osc in enumerate(osc_nodes):
            graph_obj.connect_mod(am_lfos[i].name, osc.name, "amp", scale=1.0, mode="add")

        if use_subharm:
            subharm = nodes.SubharmonicGeneratorNode("subharm", n_ch=1, mix=0.4, divisions=(2, 3))
            graph_obj.add_node(subharm)
            for osc in osc_nodes:
                graph_obj.connect_audio(osc.name, subharm.name)
            mixer_in = subharm.name
        else:
            mixer_in = osc_nodes[0].name

        mixer = nodes.MixNode(
            "mixer",
            out_channels=2,
            alc=runtime_state.get("use_normalizer", True),
            compression="tanh" if not runtime_state.get("use_hardclip", False) else "clip",
        )
        graph_obj.add_node(mixer)
        graph_obj.connect_audio(mixer_in, mixer.name)

        safety = nodes.SafetyFilterNode("safety", fs, n_ch=2)
        graph_obj.add_node(safety)
        graph_obj.connect_audio(mixer.name, safety.name)
        graph_obj.set_sink(safety.name)

        return graph_obj

    graph = build_graph(sample_rate, state)
    menu_instance = menu.Menu(state)
    menu_instance.toggle()
    menu_instance.draw()

    prev_buttons = [0] * joy.get_numbuttons()
    button_last_press: dict[int, float] = {}
    button_latch: dict[int, bool] = {}

    utils._scratch.ensure(config.MAX_FRAMES)

    def audio_callback(outdata, frames, time_info, status):
        nonlocal sample_rate, freq_current, amp_current, cutoff_current, q_current, graph

        sr = sample_rate
        utils._scratch.ensure(frames)

        f = utils.cubic_ramp(freq_current, freq_target, frames, utils._scratch.f[:frames])
        a = utils.cubic_ramp(amp_current, amp_target, frames, utils._scratch.a[:frames])
        c = utils.cubic_ramp(cutoff_current, cutoff_target, frames, utils._scratch.c[:frames])
        q = utils.cubic_ramp(q_current, q_target, frames, utils._scratch.q[:frames])

        B, C = 1, 1
        base_params = {
            "_B": B,
            "_C": C,
            "source": {
                "freq": utils.as_BCF(f, B, C, frames, name="freq"),
                "amp": utils.as_BCF(a, B, C, frames, name="amp"),
            },
            "filter": {
                "cutoff": utils.as_BCF(c, B, C, frames, name="cutoff"),
                "Q": utils.as_BCF(q, B, C, frames, name="Q"),
            },
        }

        y = graph.render(frames, sr, base_params)
        y = utils.assert_BCF(y, name="sink")
        if y.shape[0] != 1 or y.shape[1] not in (1, outdata.shape[1]):
            raise RuntimeError(f"Device expects (1,{outdata.shape[1]},F), got {y.shape}")

        chans = min(outdata.shape[1], y.shape[1])
        for ch in range(chans):
            outdata[:, ch] = y[0, ch].astype(np.float32, copy=False)

        freq_current = float(f[-1])
        amp_current = float(a[-1])
        cutoff_current = float(c[-1])
        q_current = float(q[-1])

    running = True

    def audio_thread():
        nonlocal sample_rate, graph, running
        try:
            dev_index, dev_sr = _pick_output_device_and_rate(sd)
            sample_rate = dev_sr
            sd.default.device = (None, dev_index)
            print(f"\n[Audio] Device #{dev_index} @ {sample_rate} Hz")
            graph = build_graph(sample_rate, state)
            with sd.OutputStream(
                device=dev_index,
                channels=2,
                dtype="float32",
                samplerate=sample_rate,
                blocksize=256,
                latency="low",
                callback=audio_callback,
            ):
                while running:
                    time.sleep(0.002)
        except sd.PortAudioError as exc:
            print("[Audio] Start failed:", exc)
            raise

    if not no_audio:
        threading.Thread(target=audio_thread, daemon=True).start()
    else:
        print("[Audio] Skipping output initialisation (no-audio mode).")

    clock = pygame.time.Clock()

    def draw_status(screen, lines):
        screen.fill((0, 0, 0))
        y = 10
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += 16
        pygame.display.flip()

    print("Menu:'m'  LS=pitch/vel  RS=filter.  A=gate  B=drone  X=wave  Y=cycle  N=source (osc/sampler).")
    print("Bumpers: hold=momentary, double-tap=latch. RB default FREE; LB default 12tet/full.")
    print("Z cycles FREE variant. </> root down/up, / resets root.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return 0
                if event.type == pygame.KEYDOWN:
                    if event.key == state["keymap"].get("toggle_menu", pygame.K_m):
                        menu_instance.toggle()
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("toggle_source", pygame.K_n):
                        state["source_type"] = "sampler" if state["source_type"] == "osc" else "osc"
                        graph = build_graph(sample_rate, state)
                        print(f"Source → {state['source_type'].upper()}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("wave_next", pygame.K_x):
                        state["wave_idx"] = (state["wave_idx"] + 1) % len(state["waves"])
                        graph = build_graph(sample_rate, state)
                        print(f"Waveform → {state['waves'][state['wave_idx']]}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("free_variant_next", pygame.K_z):
                        i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                        state["free_variant"] = quantizer.FREE_VARIANTS[(i + 1) % len(quantizer.FREE_VARIANTS)]
                        print(f"FREE variant → {state['free_variant']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_up", pygame.K_PERIOD):
                        state["root_midi"] = min(127, state["root_midi"] + 1)
                        print(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_down", pygame.K_COMMA):
                        state["root_midi"] = max(0, state["root_midi"] - 1)
                        print(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_reset", pygame.K_SLASH):
                        state["root_midi"] = 60
                        print("Root MIDI → 60 (C4)")
                        menu_instance.draw()

            pygame.event.pump()

            fvb = state.get("free_variant_button", 6)
            if fvb < len(prev_buttons):
                pressed = bool(joy.get_button(fvb))
                if pressed and not prev_buttons[fvb]:
                    i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                    state["free_variant"] = quantizer.FREE_VARIANTS[(i + 1) % len(quantizer.FREE_VARIANTS)]
                    print(f"FREE variant → {state['free_variant']}")
                    menu_instance.draw()

            nowt = time.time()
            for btn_idx, cfg in state["buttonmap"].items():
                if btn_idx >= len(prev_buttons):
                    continue
                pressed_now = bool(joy.get_button(btn_idx))
                pressed_prev = prev_buttons[btn_idx]
                if pressed_now and not pressed_prev:
                    last_t = button_last_press.get(btn_idx, 0.0)
                    if nowt - last_t <= state["double_tap_window"]:
                        button_latch[btn_idx] = not button_latch.get(btn_idx, False)
                        print(f"Latch[{btn_idx}] → {'ON' if button_latch[btn_idx] else 'OFF'}  ({cfg['token']})")
                        menu_instance.draw()
                    button_last_press[btn_idx] = nowt

            effective_token = state["base_token"]
            for b in state.get("bumper_priority", [4, 5]):
                cfg = state["buttonmap"].get(b)
                if not cfg:
                    continue
                if b < len(prev_buttons) and joy.get_button(b):
                    effective_token = cfg["token"]
                    break
            else:
                for b in state.get("bumper_priority", [4, 5]):
                    if state["buttonmap"].get(b) and button_latch.get(b, False):
                        effective_token = state["buttonmap"][b]["token"]
                        break

            lx, ly = joy.get_axis(0), joy.get_axis(1)
            ax_cut = state.get("filter_axis_cutoff", 3)
            ax_q = state.get("filter_axis_q", 4)
            rx_val = joy.get_axis(ax_cut) if ax_cut < joy.get_numaxes() else 0.0
            ry_val = joy.get_axis(ax_q) if ax_q < joy.get_numaxes() else 0.0
            rx01 = (rx_val + 1.0) * 0.5
            ry01 = (1.0 - ry_val) * 0.5
            cutoff_target = utils.expo_map(rx01, 80.0, 8000.0)
            q_target = 0.5 + ry01 * (12.0 - 0.5)

            root_midi = state.get("root_midi", 60)
            root_f = quantizer.Quantizer.midi_to_freq(root_midi)
            span_oct = float(state.get("free_span_oct", 2.0))
            grid = quantizer.get_reference_grid_cents(state, effective_token)
            if quantizer.is_free_mode_token(effective_token):
                N = max(1, len(utils._grid_sorted(grid)[0]))
                u = lx * span_oct * N
                fv = state.get("free_variant", "continuous")
                if fv == "continuous":
                    cents = lx * span_oct * 1200.0
                elif fv == "weighted":
                    cents = quantizer.grid_warp_inverse(u, grid)
                else:
                    cents = quantizer.grid_warp_inverse(round(u), grid)
            else:
                cents_unq = lx * span_oct * 1200.0
                u = quantizer.grid_warp_forward(cents_unq, grid)
                cents = quantizer.grid_warp_inverse(round(u), grid)
            freq_target = root_f * (2.0 ** (cents / 1200.0))

            amp_target = max(0.0, 1.0 - (ly + 1.0) / 2.0)

            gate_momentary = bool(joy.get_button(0))
            bB = joy.get_button(1)
            if bB and not prev_buttons[1]:
                drone_on = not drone_on
                print(f"Drone → {'ON' if drone_on else 'OFF'}")
            bX = joy.get_button(2)
            if bX and not prev_buttons[2]:
                state["wave_idx"] = (state["wave_idx"] + 1) % len(state["waves"])
                graph = build_graph(sample_rate, state)
                print(f"Waveform → {state['waves'][state['wave_idx']]}")
                menu_instance.draw()

            bY = joy.get_button(3)
            if bY and not prev_buttons[3]:
                t, m = quantizer.token_to_tuning_mode(state["base_token"])
                if t == "12tet":
                    names = list(quantizer.Quantizer.DIATONIC_MODES.keys())
                    idx = names.index(m) if (m in names) else -1
                    m2 = names[(idx + 1) % len(names)] if idx >= 0 else names[0]
                    state["base_token"] = f"12tet/{m2}"
                    print(f"Base token → {state['base_token']}")
                    menu_instance.draw()
                else:
                    et = ["12tet/full", "19tet/full", "31tet/full", "53tet/full"]
                    try:
                        idx = et.index(state["base_token"])
                        state["base_token"] = et[(idx + 1) % len(et)]
                    except ValueError:
                        state["base_token"] = et[0]
                    print(f"Base token  {state['base_token']}")
                    menu_instance.draw()

            for i in range(len(prev_buttons)):
                prev_buttons[i] = joy.get_button(i)

            lines = [
                f"Eff:{effective_token:<16} Base:{state['base_token']:<16} Free:{state['free_variant']:<10} Src:{state['source_type']}",
                f"Root:{state['root_midi']:3d}  Freq:{freq_target:7.2f}Hz  Amp:{amp_target:.2f}",
                f"Cut:{cutoff_target:7.1f}Hz  Q:{q_target:4.2f}  Filter:{state['filter_type']}",
                f"Wave:{state['waves'][state['wave_idx']]}  Drone:{'ON' if drone_on else 'OFF'}  LFO:{state['mod_wave_types'][state['mod_wave_idx']]}/{state['mod_rate_hz']:.2f}Hz d={state['mod_depth']:.2f} src={'input' if state['mod_use_input'] else 'free'}",
            ]
            screen = pygame.display.get_surface()
            if screen:
                draw_status(screen, lines)
            print("\r" + " | ".join(lines), end="")

            clock.tick(120)
    finally:
        running = False
        if hasattr(joy, "quit"):
            joy.quit()

    return 0


__all__ = ["run"]
