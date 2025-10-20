"""Joystick-controlled synthesiser application."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional, cast

import numpy as np

from .graph import AudioGraph
from . import menu, nodes, persistence, quantizer, state as app_state, utils


def build_runtime_graph(
    fs: int, runtime_state: dict
) -> tuple[AudioGraph, list[str], list[str]]:
    """Construct the default audio graph for the given runtime state.

    This helper returns the populated :class:`AudioGraph` instance together with
    the list of envelope modulators and any downstream amplifier modulators (the
    default graph currently returns none).  Keeping this as a module-level
    function allows tests to inspect the generated graph without needing to spin
    up the interactive application loop.
    """

    use_subharm = runtime_state.get("use_subharm", True)
    use_normalizer = runtime_state.get("use_normalizer", True)
    use_hardclip = runtime_state.get("use_hardclip", False)

    osc_waves = ["sine", "square", "saw"]
    osc_nodes = []
    pan_lfos = []
    am_lfos = []
    env_nodes = []
    for i, wave in enumerate(osc_waves):
        osc = nodes.OscNode(f"osc{i+1}", wave=wave)
        pan_lfo = nodes.LFONode(f"pan_lfo{i+1}", wave="sine", rate_hz=0.2 + 0.1 * i, depth=1.0)
        am_lfo = nodes.LFONode(f"am_lfo{i+1}", wave="sine", rate_hz=4.0 + i, depth=0.5)
        osc_nodes.append(osc)
        pan_lfos.append(pan_lfo)
        am_lfos.append(am_lfo)

    graph_obj = AudioGraph(fs, output_channels=2)

    keyboard_ctrl = nodes.ControllerNode(
        "keyboard_ctrl",
        params={
            "outputs": {
                "trigger": "signals['trigger']",
                "gate": "signals['gate']",
                "drone": "signals['drone']",
                "velocity": "signals['velocity']",
            }
        },
    )
    joystick_ctrl = nodes.ControllerNode(
        "joystick_ctrl",
        params={
            "outputs": {
                "trigger": "signals['trigger']",
                "gate": "signals['gate']",
                "drone": "signals['drone']",
                "velocity": "signals['velocity']",
                "cutoff": "signals['cutoff']",
                "q": "signals['q']",
                "pitch_input": "signals['pitch_input']",
                "pitch_span": "signals['pitch_span']",
                "pitch_root": "signals['pitch_root']",
            }
        },
    )

    graph_obj.add_node(keyboard_ctrl)
    graph_obj.add_node(joystick_ctrl)

    for osc in osc_nodes:
        graph_obj.add_node(osc)
    for pan in pan_lfos:
        graph_obj.add_node(pan)
    for am in am_lfos:
        graph_obj.add_node(am)

    nodes.EnvelopeModulatorNode.reset_groups()

    pitch_node = nodes.PitchQuantizerNode("pitch", runtime_state)
    graph_obj.add_node(pitch_node)

    env_cfg = runtime_state.get("envelope_params", {})
    poly_mode = runtime_state.get("polyphony_mode", "strings")
    default_voice_count = len(osc_nodes) if poly_mode == "piano" else 1
    voice_count = int(runtime_state.get("polyphony_voices", default_voice_count))
    if voice_count < 1:
        voice_count = 1
    voice_count = min(len(osc_nodes), voice_count)
    group_name = "voices" if voice_count > 1 else None
    for i in range(voice_count):
        env = nodes.EnvelopeModulatorNode(
            f"env{i+1}",
            attack_ms=env_cfg.get("attack_ms", 12.0),
            hold_ms=env_cfg.get("hold_ms", 8.0),
            decay_ms=env_cfg.get("decay_ms", 90.0),
            sustain_level=env_cfg.get("sustain_level", 0.65),
            sustain_ms=env_cfg.get("sustain_ms", 0.0),
            release_ms=env_cfg.get("release_ms", 220.0),
            send_resets=env_cfg.get("send_resets", True),
            group=group_name,
        )
        env_nodes.append(env)
        graph_obj.add_node(env)

    controllers = (keyboard_ctrl, joystick_ctrl)

    def _connect_controller(
        controller: nodes.ControllerNode,
        target: str,
        param: str,
        channel_name: str,
        *,
        scale: float = 1.0,
        mode: str = "add",
    ) -> None:
        try:
            channel = controller.output_index(channel_name)
        except KeyError:
            return
        graph_obj.connect_mod(
            controller.name,
            target,
            param,
            scale=scale,
            mode=mode,
            channel=channel,
        )

    for env in env_nodes:
        for controller in controllers:
            _connect_controller(controller, env.name, "velocity", "velocity")
            _connect_controller(controller, env.name, "gate", "gate")
            _connect_controller(controller, env.name, "drone", "drone")
            _connect_controller(controller, env.name, "trigger", "trigger")

    if pitch_node is not None:
        for controller in controllers:
            _connect_controller(controller, pitch_node.name, "input", "pitch_input")
            _connect_controller(controller, pitch_node.name, "span_oct", "pitch_span")
            _connect_controller(controller, pitch_node.name, "root_midi", "pitch_root")

    for i, osc in enumerate(osc_nodes):
        env = env_nodes[i % len(env_nodes)] if env_nodes else None
        if env is not None:
            graph_obj.connect_mod(env.name, osc.name, "amp", scale=1.0, mode="add", channel=0)
            if getattr(osc, "accept_reset", True) and env_cfg.get("send_resets", True):
                graph_obj.connect_mod(env.name, osc.name, "reset", scale=1.0, mode="add", channel=1)
        graph_obj.connect_mod(pitch_node.name, osc.name, "freq", scale=1.0, mode="add", channel=0)
        graph_obj.connect_mod(am_lfos[i].name, osc.name, "amp", scale=0.5, mode="mul")
        graph_obj.connect_mod(pan_lfos[i].name, osc.name, "pan", scale=1.0, mode="add")

    if use_subharm:
        subharm = nodes.SubharmonicLowLifterNode(
            "subharm",
            fs,
            band_lo=70.0,
            band_hi=160.0,
            mix=0.6,
            drive=1.2,
            out_hp=25.0,
            use_div4=True,
        )
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

    graph_obj.set_sink(mixer.name)

    return graph_obj, [env.name for env in env_nodes], []
from .config import DEFAULT_CONFIG_PATH, load_configuration


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
    """Launch the synthesiser.

    Parameters
    ----------
    allow_no_joystick:
        When true the app will operate with a neutral virtual joystick so it can
        start for inspection or testing without hardware attached.
    no_audio:
        Skip initialising the sounddevice output stream.  Useful for CI where no
        PortAudio backend is available.
    headless:
        Render the configured graph without initialising pygame.  This is useful
        for automated verification of graph behaviour.
    config_path:
        Optional configuration override used when the application needs to
        fall back to a summary render (for example when pygame is unavailable).
    """

    cfg_path = config_path or str(DEFAULT_CONFIG_PATH)

    def render_summary(reason: str, *, cleanup: bool = False) -> int:
        from .application import SynthApplication

        if cleanup:
            try:  # pragma: no cover - depends on pygame availability
                import pygame as _pygame

                _pygame.quit()
            except Exception:
                pass

        cfg = load_configuration(cfg_path)
        app = SynthApplication.from_config(cfg)
        print(reason)
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

    if headless:
        return render_summary("Headless run requested.")

    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - exercised only when pygame missing
        return render_summary(f"pygame unavailable, running summary instead: {exc}")

    if no_audio:
        sd = None
    else:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - exercised only when sounddevice missing
            return render_summary(f"sounddevice unavailable, running summary instead: {exc}")

    pygame.init()
    pygame.joystick.init()

    pygame.display.set_caption("Controller Synth (Graph)")
    pygame.display.set_mode((1280, 800))
    pygame.font.init()
    font = pygame.font.SysFont("monospace", 16)
    font_small = pygame.font.SysFont("monospace", 12)

    if pygame.joystick.get_count() == 0:
        if not allow_no_joystick:
            print("No joystick. Connect controller and restart.")
            return 1
        joy = _NullJoystick()
        print("[Joystick] Running with virtual controller (all controls neutral).")
    else:
        joy = pygame.joystick.Joystick(0)
        joy.init()

    state = app_state.build_default_state(joy=joy, pygame=pygame)
    persistence.load_mappings(state)
    sampler = _load_sampler(state)

    sample_rate = 44100
    freq_target = 220.0
    freq_current = 220.0
    velocity_target = 0.0
    velocity_current = 0.0
    cutoff_target = 1000.0
    cutoff_current = 1000.0
    q_target = 0.8
    q_current = 0.8
    gate_momentary = False
    root_midi_value = state.get("root_midi", 60)
    pitch_input_value = 0.0
    pitch_span_value = float(state.get("free_span_oct", 2.0))
    pitch_effective_token = state.get("base_token", "12tet/full")
    pitch_free_variant = state.get("free_variant", "continuous")

    envelope_modes = ("envelope", "drone", "hold")
    envelope_mode_idx = 0
    envelope_mode = envelope_modes[envelope_mode_idx]
    pending_trigger = False

    graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
    pitch_node = graph._nodes.get("pitch")
    menu_instance = menu.Menu(state)
    menu_instance.toggle()
    menu_instance.draw()

    def cycle_envelope_mode() -> None:
        nonlocal envelope_mode_idx, envelope_mode, pending_trigger

        envelope_mode_idx = (envelope_mode_idx + 1) % len(envelope_modes)
        envelope_mode = envelope_modes[envelope_mode_idx]
        if envelope_mode == "drone":
            pending_trigger = True
        elif envelope_mode == "hold" and gate_momentary:
            pending_trigger = True
        print(f"Envelope mode → {envelope_mode.title()}")
        menu_instance.draw()

    prev_buttons = [0] * joy.get_numbuttons()
    button_last_press: dict[int, float] = {}
    button_latch: dict[int, bool] = {}

    utils._scratch.ensure(app_state.MAX_FRAMES)

    momentary_prev = False
    drone_prev = False

    def audio_callback(outdata, frames, time_info, status):
        nonlocal sample_rate, freq_current, freq_target, velocity_current, cutoff_current, q_current, graph
        nonlocal momentary_prev, drone_prev, envelope_mode, pending_trigger, amp_mod_names
        nonlocal pitch_node, pitch_input_value, pitch_span_value, pitch_effective_token
        nonlocal root_midi_value, pitch_free_variant

        sr = sample_rate
        utils._scratch.ensure(frames)

        v = utils.cubic_ramp(velocity_current, velocity_target, frames, utils._scratch.a[:frames])
        c = utils.cubic_ramp(cutoff_current, cutoff_target, frames, utils._scratch.c[:frames])
        q = utils.cubic_ramp(q_current, q_target, frames, utils._scratch.q[:frames])

        mode = envelope_mode
        momentary_now = gate_momentary
        drone_now = mode == "drone"

        trigger_now = False
        if pending_trigger:
            trigger_now = True
            pending_trigger = False

        if mode in ("envelope", "hold") and momentary_now and not momentary_prev:
            trigger_now = True
        elif mode == "drone" and drone_now and not drone_prev:
            trigger_now = True

        gate_now = mode == "hold" and momentary_now
        trigger_signal = np.zeros(frames, dtype=utils.RAW_DTYPE)
        if trigger_now:
            trigger_signal[0] = 1.0
        gate_signal = np.full(frames, 1.0 if gate_now else 0.0, dtype=utils.RAW_DTYPE)
        drone_signal = np.full(frames, 1.0 if drone_now else 0.0, dtype=utils.RAW_DTYPE)

        B, C = 1, 1
        base_params: dict[str, dict[str, np.ndarray]] = {"_B": B, "_C": C}

        zero_ctrl = utils.as_BCF(0.0, B, 1, frames, name="ctrl.zero")
        base_params["keyboard_ctrl"] = {
            "trigger": zero_ctrl,
            "gate": zero_ctrl,
            "drone": zero_ctrl,
            "velocity": zero_ctrl,
        }

        trigger_bcf = utils.as_BCF(trigger_signal, B, 1, frames, name="ctrl.trigger")
        gate_bcf = utils.as_BCF(gate_signal, B, 1, frames, name="ctrl.gate")
        drone_bcf = utils.as_BCF(drone_signal, B, 1, frames, name="ctrl.drone")
        velocity_bcf = utils.as_BCF(v, B, 1, frames, name="ctrl.velocity")
        cutoff_bcf = utils.as_BCF(c, B, 1, frames, name="ctrl.cutoff")
        q_bcf = utils.as_BCF(q, B, 1, frames, name="ctrl.q")
        pitch_input_bcf = utils.as_BCF(pitch_input_value, B, 1, frames, name="ctrl.pitch_input")
        pitch_span_bcf = utils.as_BCF(pitch_span_value, B, 1, frames, name="ctrl.pitch_span")
        pitch_root_bcf = utils.as_BCF(root_midi_value, B, 1, frames, name="ctrl.pitch_root")

        base_params["joystick_ctrl"] = {
            "trigger": trigger_bcf,
            "gate": gate_bcf,
            "drone": drone_bcf,
            "velocity": velocity_bcf,
            "cutoff": cutoff_bcf,
            "q": q_bcf,
            "pitch_input": pitch_input_bcf,
            "pitch_span": pitch_span_bcf,
            "pitch_root": pitch_root_bcf,
        }

        pitch_ref = pitch_node
        if pitch_ref is None or pitch_ref.name not in graph._nodes:
            pitch_ref = graph._nodes.get("pitch")
            pitch_node = pitch_ref
        if pitch_ref is not None:
            pitch_ref.update_mode(
                effective_token=pitch_effective_token,
                free_variant=pitch_free_variant,
                span_oct=pitch_span_value,
            )

        osc_names = [name for name in ("osc1", "osc2", "osc3") if name in graph._nodes]
        for name in osc_names:
            base_params[name] = {
                "freq": utils.as_BCF(0.0, B, 1, frames, name=f"{name}.freq"),
                "amp": utils.as_BCF(0.0, B, 1, frames, name=f"{name}.amp"),
            }

        if envelope_names:
            send_reset_flag = state.get("envelope_params", {}).get("send_resets", True)
            send_reset_bcf = utils.as_BCF(
                1.0 if send_reset_flag else 0.0,
                B,
                1,
                frames,
                name="env.send_reset",
            )
            for env_name in envelope_names:
                base_params[env_name] = {"send_reset": send_reset_bcf}

        if amp_mod_names:
            amp_base = utils.as_BCF(v, B, 1, frames, name="amp_mod.base")
            for name in amp_mod_names:
                base_params[name] = {"base": amp_base}

        y = graph.render(frames, sr, base_params)
        y = utils.assert_BCF(y, name="sink")
        if y.shape[0] != 1 or y.shape[1] not in (1, outdata.shape[1]):
            raise RuntimeError(f"Device expects (1,{outdata.shape[1]},F), got {y.shape}")

        chans = min(outdata.shape[1], y.shape[1])
        for ch in range(chans):
            outdata[:, ch] = y[0, ch].astype(np.float32, copy=False)

        if pitch_ref is not None:
            last = pitch_ref.last_output
            targets = pitch_ref.last_target
            if last is not None and last.size > 0:
                freq_current = float(np.asarray(last)[0, -1]) if last.ndim == 2 else float(last[-1])
            if targets is not None and targets.size > 0:
                freq_target = float(np.asarray(targets)[-1])
        velocity_current = float(v[-1])
        cutoff_current = float(c[-1])
        q_current = float(q[-1])
        momentary_prev = momentary_now
        drone_prev = drone_now

    running = True
    audio_failures: list[Exception] = []

    if sd is None:
        print("[Audio] Skipping output initialisation (no-audio mode).")
    else:
        try:
            dev_index, dev_sr = _pick_output_device_and_rate(sd)
        except Exception as exc:
            return render_summary(
                f"Audio initialisation failed, running summary instead: {exc}",
                cleanup=True,
            )

        sample_rate = dev_sr
        sd.default.device = (None, dev_index)
        print(f"\n[Audio] Device #{dev_index} @ {sample_rate} Hz")
        graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
        pitch_node = graph._nodes.get("pitch")

        def audio_thread() -> None:
            nonlocal running
            try:
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
            except Exception as exc:  # pragma: no cover - depends on audio backend
                audio_failures.append(exc)
                running = False

        threading.Thread(target=audio_thread, daemon=True).start()

    clock = pygame.time.Clock()

    def _extract_node_stats(node: nodes.Node) -> list[str]:
        stats: list[str] = []
        for key, value in vars(node).items():
            if key.startswith("_"):
                continue
            if isinstance(value, (bool, int)):
                stats.append(f"{key}={value}")
            elif isinstance(value, float):
                stats.append(f"{key}={value:.2f}")
            elif isinstance(value, str):
                stats.append(f"{key}={value}")
        return stats[:4]

    def _node_colour(node: nodes.Node) -> tuple[int, int, int]:
        if isinstance(node, nodes.ControllerNode):
            return (48, 140, 196)
        if isinstance(node, nodes.OscNode):
            return (48, 92, 180)
        if isinstance(node, nodes.LFONode):
            return (34, 135, 92)
        if isinstance(node, nodes.EnvelopeModulatorNode):
            return (128, 84, 168)
        if isinstance(node, nodes.MixNode):
            return (150, 94, 40)
        if isinstance(node, nodes.SubharmonicLowLifterNode):
            return (92, 110, 160)
        return (70, 70, 90)

    def draw_visualisation(
        screen,
        lines: list[str],
        freq_target: float,
        velocity_target: float,
    ) -> None:
        if screen is None:
            return

        screen.fill((10, 10, 16))

        nodes_in_graph = list(getattr(graph, "_nodes", {}).keys())
        if not nodes_in_graph:
            pygame.display.flip()
            return

        node_levels = getattr(graph, "last_node_levels", {})

        width, height = screen.get_size()
        margin = 32
        tile_w = 280
        tile_h = 210

        grouped: dict[str, list[str]] = {}
        for name in nodes_in_graph:
            node = graph._nodes.get(name)
            if isinstance(node, nodes.EnvelopeModulatorNode) and getattr(node, "group", None):
                grouped.setdefault(node.group, []).append(name)

        entries: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        for name in nodes_in_graph:
            node = graph._nodes.get(name)
            if isinstance(node, nodes.EnvelopeModulatorNode) and getattr(node, "group", None):
                group_name = node.group
                if group_name in seen_groups:
                    continue
                members = [member for member in grouped.get(group_name, []) if member in graph._nodes]
                if members:
                    entries.append({"kind": "group", "group": group_name, "names": members})
                seen_groups.add(group_name)
            else:
                entries.append({"kind": "single", "name": name})

        display_count = len(entries)
        cols = max(1, min(display_count, max(1, (width - margin) // (tile_w + margin))))
        rows = (display_count + cols - 1) // cols
        total_height = rows * (tile_h + margin) + margin
        if total_height > height - 160:
            available = max(height - 200, tile_h + margin)
            tile_h = max(120, available // max(rows, 1) - margin)

        layout: dict[str, pygame.Rect] = {}
        entry_rects: list[tuple[dict[str, Any], pygame.Rect]] = []
        for idx, entry in enumerate(entries):
            row = idx // cols
            col = idx % cols
            x = margin + col * (tile_w + margin)
            y = margin + row * (tile_h + margin)
            rect = pygame.Rect(x, y, tile_w, tile_h)
            entry_rects.append((entry, rect))
            if entry["kind"] == "single":
                node_name = cast(str, entry["name"])
                layout[node_name] = rect
            else:
                names = cast(list[str], entry["names"])
                if not names:
                    continue
                header_h = font.get_height() + 12
                inner_top = rect.y + header_h
                inner_height = rect.height - header_h - 10
                inner_available = max(inner_height, 0)
                spacing = 6 if len(names) > 1 else 0
                min_slot = 48
                min_total = len(names) * min_slot + spacing * (len(names) - 1)
                effective_height = max(inner_available, min_total)
                slot_h = (effective_height - spacing * (len(names) - 1)) // len(names)
                slot_h = max(min_slot, slot_h)
                total_stack = len(names) * slot_h + spacing * (len(names) - 1)
                start_y = inner_top + max(0, (inner_available - total_stack) // 2)
                max_bottom = rect.bottom - 6
                if start_y + total_stack > max_bottom:
                    start_y = max(inner_top, max_bottom - total_stack)
                for member in names:
                    layout[member] = pygame.Rect(rect.x + 8, start_y, rect.width - 16, slot_h)
                    start_y += slot_h + spacing

        centres = {name: rect.center for name, rect in layout.items()}

        audio_colour = (90, 160, 240)
        mod_colour = (200, 140, 255)

        group_visuals: list[tuple[str, pygame.Rect, int]] = []
        for entry, rect in entry_rects:
            if entry["kind"] == "group":
                names = cast(list[str], entry["names"])
                group_name = cast(str, entry["group"])
                member_count = len(names)
                pygame.draw.rect(screen, (36, 28, 64), rect, border_radius=14)
                pygame.draw.rect(screen, (210, 210, 235), rect, width=2, border_radius=14)
                group_visuals.append((group_name, rect, member_count))

        for source, targets in getattr(graph, "_audio_successors", {}).items():
            if source not in centres:
                continue
            start = centres[source]
            for target in targets:
                if target not in centres:
                    continue
                pygame.draw.line(screen, audio_colour, start, centres[target], 3)

        for target, conns in getattr(graph, "_mod_inputs", {}).items():
            if target not in centres:
                continue
            end = centres[target]
            for conn in conns:
                start = centres.get(conn.source)
                if start is None:
                    continue
                pygame.draw.line(screen, mod_colour, start, end, 2)

        def _brighten(colour: tuple[int, int, int], amount: int = 18) -> tuple[int, int, int]:
            return tuple(min(255, max(0, c + amount)) for c in colour)

        def render_node_tile(name: str, rect: pygame.Rect, *, header: str | None = None, tint: bool = False) -> None:
            node = graph._nodes.get(name)
            if node is None:
                return
            colour = _node_colour(node)
            if tint:
                colour = _brighten(colour)
            border_width = 1 if rect.height < 120 else 2
            pygame.draw.rect(screen, colour, rect, border_radius=10)
            pygame.draw.rect(screen, (220, 220, 220), rect, width=border_width, border_radius=10)

            header_font = font if rect.height >= 140 else font_small
            header_text = header or name
            title = header_font.render(header_text, True, (250, 250, 250))
            screen.blit(title, (rect.x + 10, rect.y + 6))

            lines_to_render: list[str] = []
            if isinstance(node, nodes.OscNode):
                lines_to_render.extend(
                    [
                        f"wave={node.wave}",
                        f"freq≈{freq_target:.1f}Hz",
                        f"vel≈{velocity_target:.2f}",
                    ]
                )
            else:
                lines_to_render.extend(_extract_node_stats(node))

            info_start_y = rect.y + header_font.get_height() + 8
            max_lines = 5 if rect.height >= 160 else 3 if rect.height >= 120 else 2
            rendered_lines = lines_to_render[:max_lines]
            for idx_line, text_line in enumerate(rendered_lines):
                text_surface = font_small.render(text_line, True, (240, 240, 240))
                screen.blit(
                    text_surface,
                    (rect.x + 12, info_start_y + idx_line * (font_small.get_height() + 2)),
                )

            levels = node_levels.get(name)
            if isinstance(levels, np.ndarray) and levels.size:
                vu_rows: list[tuple[str, float]] = []
                for batch_idx in range(levels.shape[0]):
                    for channel_idx in range(levels.shape[1]):
                        label = f"B{batch_idx}C{channel_idx}"
                        vu_rows.append((label, float(levels[batch_idx, channel_idx])))
                if vu_rows:
                    vu_bar_h = 12 if rect.height >= 140 else 10
                    row_spacing = vu_bar_h + 4
                    base_y = rect.bottom - (len(vu_rows) * row_spacing) - 8
                    min_y = info_start_y + len(rendered_lines) * (font_small.get_height() + 2) + 4
                    base_y = max(base_y, min_y)
                    bar_x = rect.x + 12
                    bar_w = rect.width - 24
                    for idx_row, (label, value) in enumerate(vu_rows):
                        row_y = base_y + idx_row * row_spacing
                        label_surface = font_small.render(label, True, (210, 210, 210))
                        screen.blit(label_surface, (bar_x, row_y))
                        label_w = label_surface.get_width() + 6
                        meter_w = max(24, bar_w - label_w)
                        meter_rect = pygame.Rect(bar_x + label_w, row_y, meter_w, vu_bar_h)
                        pygame.draw.rect(screen, (30, 30, 46), meter_rect, border_radius=4)
                        level = max(0.0, value)
                        colour_scale = min(1.0, level)
                        if level > 1.0:
                            vu_colour = (230, 60, 60)
                        elif colour_scale > 0.85:
                            vu_colour = (235, 180, 60)
                        else:
                            vu_colour = (90, 200, 120)
                        fill_w = int(meter_w * min(1.0, colour_scale))
                        if fill_w > 0:
                            fill_rect = pygame.Rect(bar_x + label_w, row_y, fill_w, vu_bar_h)
                            pygame.draw.rect(screen, vu_colour, fill_rect, border_radius=4)
                        peak_text = font_small.render(f"{level:0.2f}", True, (190, 190, 190))
                        peak_x = bar_x + label_w + meter_w + 4
                        peak_x = min(peak_x, rect.right - peak_text.get_width() - 6)
                        screen.blit(peak_text, (peak_x, row_y))

        for entry, rect in entry_rects:
            if entry["kind"] == "group":
                names = cast(list[str], entry["names"])
                for member in names:
                    render_node_tile(member, layout[member], tint=True)
            else:
                node_name = cast(str, entry["name"])
                render_node_tile(node_name, layout[node_name])

        for group_name, rect, member_count in group_visuals:
            header = font.render(f"{group_name} [{member_count}]", True, (235, 235, 245))
            screen.blit(header, (rect.x + 12, rect.y + 8))

        legend_x = margin
        legend_y = height - (len(lines) + 3) * (font.get_height() + 4)
        if legend_y < rows * (tile_h + margin) + margin:
            legend_y = rows * (tile_h + margin) + margin

        pygame.draw.line(screen, audio_colour, (legend_x, legend_y), (legend_x + 24, legend_y), 3)
        label_audio = font_small.render("audio", True, (200, 200, 200))
        screen.blit(label_audio, (legend_x + 32, legend_y - font_small.get_height() // 2))

        legend_y += font_small.get_height() + 8
        pygame.draw.line(screen, mod_colour, (legend_x, legend_y), (legend_x + 24, legend_y), 2)
        label_mod = font_small.render("mod", True, (200, 200, 200))
        screen.blit(label_mod, (legend_x + 32, legend_y - font_small.get_height() // 2))

        info_y = legend_y + font_small.get_height() + 12
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (margin, info_y))
            info_y += font.get_height() + 4

        pygame.display.flip()

    print(
        "Menu:'m'  LS=pitch/vel  RS=filter.  A=trigger  B=mode  X=wave  Y=cycle  N=source (osc/sampler)."
    )
    print("Bumpers: hold=momentary, double-tap=latch. RB default FREE; LB default 12tet/full.")
    print("Z cycles FREE variant. </> root down/up, / resets root.")

    try:
        while True:
            if audio_failures:
                pygame.quit()
                return render_summary(
                    f"Audio initialisation failed, running summary instead: {audio_failures[0]}",
                    cleanup=False,
                )
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
                        graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                        pitch_node = graph._nodes.get("pitch")
                        print(f"Source → {state['source_type'].upper()}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("wave_next", pygame.K_x):
                        state["wave_idx"] = (state["wave_idx"] + 1) % len(state["waves"])
                        graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                        pitch_node = graph._nodes.get("pitch")
                        print(f"Waveform → {state['waves'][state['wave_idx']]}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("free_variant_next", pygame.K_z):
                        i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                        state["free_variant"] = quantizer.FREE_VARIANTS[(i + 1) % len(quantizer.FREE_VARIANTS)]
                        pitch_free_variant = state["free_variant"]
                        print(f"FREE variant → {state['free_variant']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("drone_toggle", pygame.K_b):
                        cycle_envelope_mode()
                    elif event.key == state["keymap"].get("root_up", pygame.K_PERIOD):
                        state["root_midi"] = min(127, state["root_midi"] + 1)
                        root_midi_value = state["root_midi"]
                        print(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_down", pygame.K_COMMA):
                        state["root_midi"] = max(0, state["root_midi"] - 1)
                        root_midi_value = state["root_midi"]
                        print(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_reset", pygame.K_SLASH):
                        state["root_midi"] = 60
                        root_midi_value = state["root_midi"]
                        print("Root MIDI → 60 (C4)")
                        menu_instance.draw()

            pygame.event.pump()

            fvb = state.get("free_variant_button", 6)
            if fvb < len(prev_buttons):
                pressed = bool(joy.get_button(fvb))
                if pressed and not prev_buttons[fvb]:
                    i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                    state["free_variant"] = quantizer.FREE_VARIANTS[(i + 1) % len(quantizer.FREE_VARIANTS)]
                    pitch_free_variant = state["free_variant"]
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
            span_oct = float(state.get("free_span_oct", 2.0))
            pitch_input_value = lx
            pitch_span_value = span_oct
            pitch_effective_token = effective_token
            pitch_free_variant = state.get("free_variant", "continuous")
            root_midi_value = root_midi

            velocity_target = max(0.0, 1.0 - (ly + 1.0) / 2.0)

            gate_momentary = bool(joy.get_button(0))
            bB = joy.get_button(1)
            if bB and not prev_buttons[1]:
                cycle_envelope_mode()
            bX = joy.get_button(2)
            if bX and not prev_buttons[2]:
                state["wave_idx"] = (state["wave_idx"] + 1) % len(state["waves"])
                graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                pitch_node = graph._nodes.get("pitch")
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
                    graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                    pitch_node = graph._nodes.get("pitch")
                    print(f"Base token → {state['base_token']}")
                    menu_instance.draw()
                else:
                    et = ["12tet/full", "19tet/full", "31tet/full", "53tet/full"]
                    try:
                        idx = et.index(state["base_token"])
                        state["base_token"] = et[(idx + 1) % len(et)]
                    except ValueError:
                        state["base_token"] = et[0]
                    graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                    pitch_node = graph._nodes.get("pitch")
                    print(f"Base token  {state['base_token']}")
                    menu_instance.draw()

            for i in range(len(prev_buttons)):
                prev_buttons[i] = joy.get_button(i)

            lines = [
                f"Eff:{effective_token:<16} Base:{state['base_token']:<16} Free:{state['free_variant']:<10} Src:{state['source_type']}",
                f"Root:{state['root_midi']:3d}  Freq:{freq_target:7.2f}Hz  Vel:{velocity_target:.2f}",
                f"Cut:{cutoff_target:7.1f}Hz  Q:{q_target:4.2f}  Filter:{state['filter_type']}",
                f"Wave:{state['waves'][state['wave_idx']]}  Mode:{envelope_mode.title():<9}  LFO:{state['mod_wave_types'][state['mod_wave_idx']]}/{state['mod_rate_hz']:.2f}Hz d={state['mod_depth']:.2f} src={'input' if state['mod_use_input'] else 'free'}",
            ]
            screen = pygame.display.get_surface()
            if screen:
                draw_visualisation(screen, lines, freq_target, velocity_target)
            print("\r" + " | ".join(lines), end="")

            clock.tick(120)
    finally:
        running = False
        if hasattr(joy, "quit"):
            joy.quit()

    return 0


__all__ = ["run", "build_runtime_graph"]
