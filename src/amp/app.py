"""Joystick-controlled synthesiser application."""

from __future__ import annotations


import math
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Optional, cast
import traceback

# Import the modular controller monitor
from .controller_monitor import ControllerMonitor

import numpy as np

from .graph import AudioGraph
from . import menu, nodes, persistence, quantizer, state as app_state, utils
from .controls import _assign_control


CacheKey = tuple[str, str, tuple[int, int, int], str]


class AsyncThrottledPrinter:
    """Background printer that rate limits console output."""

    def __init__(
        self,
        *,
        window_seconds: float = 0.75,
        max_messages: int = 8,
    ) -> None:
        self._queue: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        self._history: deque[float] = deque()
        self._history_window = window_seconds
        self._max_messages = max_messages
        self._history_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            text, end = item
            try:
                sys.stdout.write(text)
                sys.stdout.write(end)
                sys.stdout.flush()
            finally:
                self._queue.task_done()

    def _prune_history(self, now: float) -> None:
        while self._history and now - self._history[0] > self._history_window:
            self._history.popleft()

    def emit(self, message: str, *, end: str = "\n", force: bool = False) -> bool:
        """Queue *message* for printing when under the rate limit.

        Returns ``True`` when the message is enqueued for output."""

        now = time.monotonic()
        with self._history_lock:
            self._prune_history(now)
            if not force and len(self._history) >= self._max_messages:
                return False
            self._history.append(now)
        self._queue.put((message, end))
        return True

    def flush(self) -> None:
        """Block until queued messages have been printed."""

        self._queue.join()

    def close(self) -> None:
        """Stop the worker thread after flushing pending messages."""

        self.flush()
        self._queue.put(None)
        self._queue.join()


STATUS_PRINTER = AsyncThrottledPrinter()


class TextSurfaceCache:
    """Cache rendered text surfaces keyed by node and content."""

    def __init__(self) -> None:
        self._cache: dict[CacheKey, Any] = {}
        self._node_keys: dict[str, set[CacheKey]] = {}
        self._frame_usage: dict[str, set[CacheKey]] = {}

    def start_frame(self) -> None:
        """Initialise tracking for a new frame."""

        self._frame_usage = {}

    def fetch(
        self,
        node_key: str,
        text: str,
        colour: tuple[int, int, int],
        font_key: str,
        renderer: Callable[[], Any],
    ) -> Any:
        """Return a cached surface, rendering when no cache entry exists."""

        key = (node_key, text, colour, font_key)
        surface = self._cache.get(key)
        if surface is None:
            surface = renderer()
            self._cache[key] = surface
        self._frame_usage.setdefault(node_key, set()).add(key)
        return surface

    def finish_frame(self) -> None:
        """Drop cache entries no longer referenced this frame."""

        for node_key, used in self._frame_usage.items():
            previous = self._node_keys.get(node_key)
            if previous:
                for stale in previous - used:
                    self._cache.pop(stale, None)
            self._node_keys[node_key] = set(used)

        stale_nodes = [node for node in self._node_keys if node not in self._frame_usage]
        for node in stale_nodes:
            for key in self._node_keys[node]:
                self._cache.pop(key, None)
            del self._node_keys[node]

        self._frame_usage = {}

    def invalidate(self, node_key: str) -> None:
        """Remove cached entries for the given node."""

        cached = self._node_keys.pop(node_key, None)
        if not cached:
            return
        for key in cached:
            self._cache.pop(key, None)


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


def build_base_params(
    graph: AudioGraph,
    state: dict,
    frames: int,
    cache: dict,
    envelope_names: list[str],
    amp_mod_names: list[str],
    joystick_curves: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Construct the base parameter dict used for rendering blocks.

    This is shared with the benchmarking helper so the interactive app and
    headless benchmark use identical parameter shaping and defaults.
    """


    # All controller data must be sourced from the control history (via AudioGraph/ControlDelay)
    # Sample the control history for the current block
    sampled = graph.sample_control_tensor(getattr(graph, 'latest_time', 0.0), frames)
    sampled_extras = sampled.get("extras", {})

    # Keyboard controller (if present in history/extras)
    base_params: dict[str, dict[str, np.ndarray]] = {"_B": 1, "_C": 1}
    base_params["keyboard_ctrl"] = {
        "trigger": _assign_control(cache, "keyboard.trigger", frames, sampled_extras.get("keyboard_trigger", 0.0)),
        "gate": _assign_control(cache, "keyboard.gate", frames, sampled_extras.get("keyboard_gate", 0.0)),
        "drone": _assign_control(cache, "keyboard.drone", frames, sampled_extras.get("keyboard_drone", 0.0)),
        "velocity": _assign_control(cache, "keyboard.velocity", frames, sampled_extras.get("keyboard_velocity", 0.0)),
    }

    # Joystick controller: always use history-backed data
    joystick_params: dict[str, np.ndarray] = {}
    for key in (
        "trigger",
        "gate",
        "drone",
        "velocity",
        "cutoff",
        "q",
        "pitch_input",
        "pitch_span",
        "pitch_root",
    ):
        # Use value from history/extras, or fallback to state for static params
        if key in ("pitch_span", "pitch_root"):
            value = sampled_extras.get(key, float(state.get("free_span_oct", 2.0)) if key == "pitch_span" else float(state.get("root_midi", 60)))
        else:
            value = sampled_extras.get(key, 0.0)
        joystick_params[key] = _assign_control(cache, f"joystick.{key}", frames, value)
    base_params["joystick_ctrl"] = joystick_params

    # Remove any direct update of pitch_node from here; all pitch info must flow through history

    osc_names = [name for name in ("osc1", "osc2", "osc3") if name in graph._nodes]
    for idx, name in enumerate(osc_names):
        # Oscillator params should also be history-backed if modulated; otherwise, use static defaults
        freq = sampled_extras.get(f"{name}_freq", 110.0 * (idx + 2))
        amp = sampled_extras.get(f"{name}_amp", 0.3 if idx == 0 else 0.25)
        base_params[name] = {
            "freq": _assign_control(cache, f"{name}.freq", frames, freq),
            "amp": _assign_control(cache, f"{name}.amp", frames, amp),
        }

    if envelope_names:
        send_reset = _assign_control(cache, "envelope.send_reset", frames, 1.0)
        for env_name in envelope_names:
            base_params[env_name] = {"send_reset": send_reset}

    if amp_mod_names:
        amp_base = joystick_params["velocity"]
        for mod_name in amp_mod_names:
            base_params[mod_name] = {"base": amp_base}

    return base_params
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
        STATUS_PRINTER.emit(
            "[Sampler] sample.wav not found (sampler available after you add a file).",
            force=True,
        )
        return None
    try:
        sampler = nodes.Sampler(path, loop=True)
        STATUS_PRINTER.emit(
            f"[Sampler] Loaded '{os.path.basename(path)}' at {sampler.file_sr} Hz",
            force=True,
        )
        return sampler
    except Exception as exc:  # pragma: no cover - depends on local files
        STATUS_PRINTER.emit(f"[Sampler] Disabled: {exc}", force=True)
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
        STATUS_PRINTER.emit(reason, force=True)
        STATUS_PRINTER.emit(app.summary(), force=True)
        buffer = app.render()
        peak = float(buffer.max())
        trough = float(buffer.min())
        STATUS_PRINTER.emit(
            f"Rendered {buffer.shape[1]} frames @ {cfg.sample_rate} Hz "
            f"(peak {peak:.3f}, trough {trough:.3f})",
            force=True,
        )
        node_timings = getattr(app.graph, "last_node_timings", None)
        if node_timings:
            STATUS_PRINTER.emit("Node timings (ms):", force=True)
            for name, duration in sorted(node_timings.items(), key=lambda item: item[1], reverse=True):
                STATUS_PRINTER.emit(f"  {name:<24} {duration * 1000.0:7.3f}", force=True)
        if app.joystick_error and not app.joystick:
            STATUS_PRINTER.emit(f"Warning: {app.joystick_error}", force=True)
        STATUS_PRINTER.flush()
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

    text_cache = TextSurfaceCache()
    last_freq_target_display: float | None = None
    last_velocity_target_display: float | None = None
    last_node_levels_snapshot: dict[str, np.ndarray] = {}
    last_node_timings_snapshot: dict[str, float] = {}
    status_signature_pending: tuple[Any, ...] | None = None
    last_status_signature: tuple[Any, ...] | None = None
    last_console_signature: tuple[Any, ...] | None = None
    last_timing_alert: bool | None = None


    # Controller polling rate (Hz) - must be integer divisor of audio rate (e.g., 200Hz for 44.1kHz/48kHz)
    controller_poll_rate = 200
    controller_poll_interval = 1.0 / controller_poll_rate

    if pygame.joystick.get_count() == 0:
        if not allow_no_joystick:
            STATUS_PRINTER.emit("No joystick. Connect controller and restart.", force=True)
            return 1
        joy = _NullJoystick()
        STATUS_PRINTER.emit(
            "[Joystick] Running with virtual controller (all controls neutral).",
            force=True,
        )
    else:
        joy = pygame.joystick.Joystick(0)
        joy.init()

    state = app_state.build_default_state(joy=joy, pygame=pygame)
    persistence.load_mappings(state)
    sampler = _load_sampler(state)

    # --- ControllerMonitor setup ---
    # All controller input is now handled by ControllerMonitor, which writes to control history.
    # No direct polling or caching of controller state outside the history-backed thread.
    controller_monitor = ControllerMonitor(
        poll_fn=None,  # Actual polling is handled inside ControllerMonitor
        control_history=graph,
        poll_interval=controller_poll_interval,
        audio_frame_rate=sample_rate,
    )
    controller_monitor.start()

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
        STATUS_PRINTER.emit(f"Envelope mode → {envelope_mode.title()}")
        menu_instance.draw()


    # Remove direct polling and history writing from main loop; all input is now handled by ControllerMonitor

    callback_timing_samples: deque[dict[str, Any]] = deque(maxlen=256)
    callback_timing_lock = threading.Lock()
    last_callback_started = 0.0
    ema_alpha = 0.05
    render_ema: float | None = None
    produced_ema: float | None = None
    period_ema: float | None = None
    # HUD-level slow EMAs (milliseconds) to ensure display changes slowly.
    # Use a time-constant based EMA so smoothing is stable regardless of frame rate.
    # HUD smoothing time constants (set long so EMAs update very slowly)
    hud_time_constant_node = 30.0
    hud_time_constant_global = 30.0
    hud_last_update = time.perf_counter()
    hud_render_ms: float | None = None
    hud_produced_ms: float | None = None
    hud_period_ms: float | None = None
    # Per-node HUD EMAs (milliseconds)
    hud_node_timings: dict[str, float] = {}

    pcm_queue: queue.Queue[tuple[np.ndarray, dict[str, Any]]] | None = None
    # Holds chunks that couldn't be enqueued immediately so we never
    # silently drop producer data. A background requeue daemon will
    # attempt to push these back into `pcm_queue` and emit warnings.
    stashed_chunks: deque[tuple[np.ndarray, dict[str, Any]]] = deque()
    queue_capacity = 0
    queue_depth_display = 0
    audio_blocksize = 256
    producer_batch_blocks = 8
    producer_max_batch_blocks = 64
    # Aim to fill the queue fully when possible (aggressive pre-rendering).
    producer_fill_target = 1.0
    producer_stop_event = threading.Event()
    producer_thread_obj: threading.Thread | None = None

    queue_stats_lock = threading.Lock()
    queue_stats = {
        "min_depth": float("inf"),
        "sum_depth": 0.0,
        "samples": 0,
        "underflows": 0,
        "last_report": time.monotonic(),
        "interval": 5.0,
    }

    # Accumulate repeated underrun messages so the UI/logs are not flooded.
    audio_underrun_accum: int = 0
    audio_underrun_last_emit: float = 0.0
    # How often (seconds) to flush accumulated underrun messages
    audio_underrun_flush_interval: float = 2.0

    control_tensors: dict[str, np.ndarray] = {}

    # Efficiency exploration data (producer records these)
    efficiency_lock = threading.Lock()
    # deque of tuples: (batch_blocks, measured_efficiency)
    efficiency_points: deque[tuple[int, float]] = deque(maxlen=4096)
    # Snapshot of the current preferred batch size for UI
    preferred_batch_snapshot: int | None = None
    # Debug throttling for STATUS_PRINTER emits
    efficiency_debug_last_emit: float = 0.0
    efficiency_debug_emit_interval: float = 5.0

    # Use module-level control helpers to centralise cache behaviour and
    # ensure both interactive and headless runners create identical buffers.
    from .controls import _assign_control as _assign_control_cache

    def _render_audio_frames(frames: int) -> tuple[np.ndarray, dict[str, Any]]:
        # All controller state is now handled by ControllerMonitor and control history.

        sr = sample_rate
        start_time = time.perf_counter()
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
        # Delegate construction of base_params to the shared function so the
        # interactive path uses identical shaping to headless runs. Pass the
        # local control_tensors dict as the cache argument expected by the
        # cache-based helper.
        # Sample joystick and pitch/envelope history from the graph so the
        # renderer consumes inputs derived from the ControlDelay. This keeps
        # the interactive and headless paths identical with respect to
        # interpolation and read-ahead semantics.
        start_time = time.perf_counter()
        sampled = graph.sample_control_tensor(start_time, frames)
        sampled_extras = sampled.get("extras", {})

        joystick_curves = {
            "trigger": sampled_extras.get("trigger", np.full(frames, 0.0, dtype=utils.RAW_DTYPE)),
            "gate": sampled_extras.get("gate", np.full(frames, 1.0 if gate_now else 0.0, dtype=utils.RAW_DTYPE)),
            "drone": sampled_extras.get("drone", np.full(frames, 1.0 if drone_now else 0.0, dtype=utils.RAW_DTYPE)),
            "velocity": sampled_extras.get("velocity", v),
            "cutoff": sampled_extras.get("cutoff", c),
            "q": sampled_extras.get("q", q),
            "pitch_input": sampled_extras.get("pitch_input", np.full(frames, pitch_input_value, dtype=utils.RAW_DTYPE)),
            "pitch_span": sampled_extras.get("pitch_span", float(pitch_span_value)),
            "pitch_root": sampled_extras.get("pitch_root", float(root_midi_value)),
        }

        # Use shared runner for rendering
        from .runner import render_audio_block
        audio_block, meta = render_audio_block(
            graph,
            start_time,
            frames,
            sr,
            joystick_curves,
            state,
            envelope_names,
            amp_mod_names,
            control_tensors,
        )
        y = utils.assert_BCF(audio_block, name="sink")
        if y.shape[0] != 1:
            raise RuntimeError(f"Device expects single batch output, got {y.shape}")
        buffer = np.swapaxes(y[0], 0, 1).astype(np.float64, copy=False)
        # ...existing code for updating freq_current, velocity_current, etc.
        render_duration = time.perf_counter() - start_time
        meta["render_duration"] = render_duration
        meta["allotted_time"] = (frames / sr) if sr else 0.0
        meta["produced_time"] = meta["allotted_time"]
        return buffer, meta

    def audio_callback(outdata, frames, time_info, status):
        nonlocal sample_rate, graph, last_callback_started, pcm_queue, queue_depth_display, audio_blocksize, queue_capacity
        nonlocal render_ema, produced_ema, period_ema
        nonlocal audio_underrun_accum, audio_underrun_last_emit, audio_underrun_flush_interval

        sr = sample_rate
        start_time = time.perf_counter()
        audio_blocksize = frames

        chunk: np.ndarray | None = None
        meta: dict[str, Any] | None = None
        queue_underflow = False

        if pcm_queue is not None:
            try:
                chunk, meta = pcm_queue.get_nowait()
            except queue.Empty:
                queue_underflow = True
                chunk = None
                meta = None
            else:
                queue_depth_display = pcm_queue.qsize()

        if chunk is None:
            # Graceful underrun handling: do not crash the app. Instead,
            # synthesize silence for this callback and accumulate a collapsed
            # log message to avoid flooding the logs/UI. Emit the accumulated
            # message at most once per `audio_underrun_flush_interval` seconds.
            now = time.monotonic()
            audio_underrun_accum += 1
            if now - audio_underrun_last_emit >= audio_underrun_flush_interval:
                if audio_underrun_accum > 1:
                    STATUS_PRINTER.emit(
                        f"[AudioCallback] Audio callback: PCM queue empty when data expected x{audio_underrun_accum}",
                        force=True,
                    )
                else:
                    STATUS_PRINTER.emit(
                        "[AudioCallback] Audio callback: PCM queue empty when data expected",
                        force=True,
                    )
                audio_underrun_accum = 0
                audio_underrun_last_emit = now

            # Produce silent chunk matching the expected frames and channel count
            try:
                chans = outdata.shape[1]
            except Exception:
                chans = 2
            chunk = np.zeros((frames, chans), dtype=np.float32)
            # Provide minimal metadata so downstream logic can still operate
            meta = {
                "render_duration": 0.0,
                "allotted_time": (frames / sample_rate) if sample_rate else 0.0,
                "produced_frames": frames,
                "produced_time": (frames / sample_rate) if sample_rate else 0.0,
                "batch_blocks": 1,
                "batch_index": 0,
                "queue_underflow": True,
            }

        # If chunk length mismatches the callback frames, do NOT silently
        # pad/truncate. This indicates a producer/renderer bug which must be
        # diagnosed at source. Surface as an error.
        if chunk.shape[0] != frames:
            tb = (
                f"Audio callback: chunk length {chunk.shape[0]} does not match "
                f"callback frames {frames}"
            )
            STATUS_PRINTER.emit(f"[AudioCallback] {tb}", force=True)
            exc = RuntimeError(tb)
            audio_failures.append(exc)
            running = False
            return

        queue_underflow = queue_underflow or bool(meta.get("queue_underflow"))

        outdata.fill(0.0)
        chans = min(outdata.shape[1], chunk.shape[1])
        # Convert to float32 only at the audio output boundary to preserve
        # higher internal precision throughout the DSP pipeline.
        if chunk.dtype != np.float32:
            chunk_to_write = chunk.astype(np.float32, copy=False)
        else:
            chunk_to_write = chunk
        outdata[:, :chans] = chunk_to_write[:, :chans]

        if pcm_queue is not None:
            queue_depth_display = pcm_queue.qsize()
        else:
            queue_depth_display = 0

        now_monotonic = time.monotonic()
        with queue_stats_lock:
            queue_stats["min_depth"] = min(queue_stats["min_depth"], queue_depth_display)
            queue_stats["sum_depth"] += float(queue_depth_display)
            queue_stats["samples"] += 1
            if queue_underflow:
                queue_stats["underflows"] += 1
            interval = queue_stats.get("interval", 5.0)
            should_report = (
                queue_capacity
                and queue_stats["samples"]
                and now_monotonic - queue_stats["last_report"] >= interval
            )
            if should_report:
                min_depth = queue_stats["min_depth"]
                if math.isinf(min_depth):
                    min_depth = queue_depth_display
                avg_depth = queue_stats["sum_depth"] / max(1, queue_stats["samples"])
                STATUS_PRINTER.emit(
                    (
                        f"[Audio] Queue stats: min={int(min_depth)}/{queue_capacity} "
                        f"avg={avg_depth:.1f} underflows={queue_stats['underflows']}"
                    ),
                    force=True,
                )
                queue_stats["min_depth"] = float("inf")
                queue_stats["sum_depth"] = 0.0
                queue_stats["samples"] = 0
                queue_stats["underflows"] = 0
                queue_stats["last_report"] = now_monotonic

        end_time = time.perf_counter()
        period = 0.0
        if last_callback_started:
            period = start_time - last_callback_started
        last_callback_started = start_time

        status_obj = status or 0
        underrun = bool(
            getattr(status_obj, "output_underflow", False)
            or getattr(status_obj, "input_underflow", False)
        )
        if queue_underflow:
            underrun = True

        # Do NOT accept missing metadata silently. Require the producer to
        # supply authoritative timing information for each chunk.
        if meta is None:
            tb = "Audio callback: missing metadata for dequeued chunk"
            STATUS_PRINTER.emit(f"[AudioCallback] {tb}", force=True)
            exc = RuntimeError(tb)
            audio_failures.append(exc)
            running = False
            return
        if "render_duration" not in meta or "produced_frames" not in meta or "produced_time" not in meta or "allotted_time" not in meta:
            tb = f"Audio callback: incomplete metadata keys: {list(meta.keys())}"
            STATUS_PRINTER.emit(f"[AudioCallback] {tb}", force=True)
            exc = RuntimeError(tb)
            audio_failures.append(exc)
            running = False
            return
        render_duration = float(meta["render_duration"])
        produced_frames = int(meta["produced_frames"])
        produced_time = float(meta["produced_time"])
        allotted_time = float(meta["allotted_time"])

        def _ema(previous: float | None, value: float) -> float:
            if previous is None:
                return value
            return previous + ema_alpha * (value - previous)

        render_ema = _ema(render_ema, render_duration)
        produced_ema = _ema(produced_ema, produced_time)
        period_ema = _ema(period_ema, period)

        # Require explicit batch metadata rather than falling back to 1.
        if "batch_blocks" not in meta or "batch_index" not in meta:
            tb = f"Audio callback: missing batch metadata: {list(meta.keys())}"
            STATUS_PRINTER.emit(f"[AudioCallback] {tb}", force=True)
            exc = RuntimeError(tb)
            audio_failures.append(exc)
            running = False
            return
        batch_blocks = int(meta["batch_blocks"])
        batch_index = int(meta["batch_index"])

        sample = {
            "render_duration": render_duration,
            "callback_period": period,
            "allotted_time": allotted_time,
            "produced_frames": produced_frames,
            "produced_time": produced_time,
            "render_ema": render_ema,
            "produced_ema": produced_ema,
            "period_ema": period_ema,
            "batch_blocks": batch_blocks,
            "batch_index": batch_index,
            "underrun": underrun,
            "queue_depth": queue_depth_display,
            "queue_capacity": queue_capacity,
            "queue_underflow": queue_underflow,
        }
        node_timings_meta = meta.get("node_timings")
        if node_timings_meta:
            sample["node_timings"] = dict(node_timings_meta)

        with callback_timing_lock:
            callback_timing_samples.append(sample)

    running = True
    audio_failures: list[Exception] = []

    if sd is None:
        STATUS_PRINTER.emit("[Audio] Skipping output initialisation (no-audio mode).", force=True)
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
        STATUS_PRINTER.emit(f"\n[Audio] Device #{dev_index} @ {sample_rate} Hz", force=True)
        graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
        pitch_node = graph._nodes.get("pitch")

        # Larger queue to allow more aggressive pre-rendering.
        pcm_queue = queue.Queue[tuple[np.ndarray, dict[str, Any]]](maxsize=128)
        queue_capacity = pcm_queue.maxsize
        producer_stop_event.clear()

        def producer_thread() -> None:
            nonlocal running, audio_blocksize
            try:
                # Adaptive backoff used when the queue is full. This starts
                # very small (low-latency retry) and exponentially backs off
                # to avoid busy spinning if the consumer lags.
                put_backoff = 0.0001
                put_backoff_max = 0.05
                # Preferred batch size driven by real renders (never probe).
                preferred_batch_blocks = producer_batch_blocks
                # EMA for measured efficiency = produced_time / render_duration
                efficiency_ema: float | None = None
                efficiency_alpha = 0.2
                # Maintain an EMA of the estimated gradient (d efficiency / d batch)
                grad_ema: float | None = None
                grad_alpha = 0.4
                # Keep a tiny recent map of measured efficiencies by batch
                # size so we can compute a centered finite-difference using
                # the +1 / -1 neighbourhood (hyperlocal landscape). This
                # materialises local landscape without doing dedicated
                # probe renders — we alternate small perturbations so each
                # render is still a real render that will be consumed.
                last_eff_map: dict[int, float] = {}
                # Probe cycle: 0 (preferred), +1, -1, repeat. This ensures
                # we sample both neighbours frequently to form a centre
                # difference estimate.
                probe_sequence = (0, 1, -1)
                probe_idx = 0
                # Greedy ascent: render at the current preferred size and
                # rely on the finite-difference gradient EMA to step the
                # preferred size. No explicit trials or perturbations are
                # performed — we do not render solely to probe.
                # Minimum significant gradient threshold to avoid thrash.
                grad_threshold = 1e-6

                while running and not producer_stop_event.is_set():
                    block_frames = max(1, audio_blocksize)
                    # Batch selection MUST be independent of cache occupancy.
                    # Always render according to the learned
                    # `preferred_batch_blocks` with a small alternating
                    # perturbation to estimate the gradient. Do NOT change
                    # the candidate based on queue backlog; the enqueue
                    # loop will briefly back off if the queue is full.
                    # Conservative candidate selection:
                    # - If in cooldown, keep the preferred size.
                    # - Otherwise, occasionally perform a one-block *upwards*
                    #   trial to measure whether larger batches improve
                    #   efficiency. Trials happen infrequently (trial_interval)
                    #   and are not alternated every render, avoiding the
                    #   previously-observed every-other-single behaviour.
                    # Greedy candidate: always render at the integer
                    # preferred batch size. No perturbation or trial is
                    # performed; preferred_batch_blocks is adjusted only by
                    # the gradient-EMA logic below when evidence supports
                    # an increase or decrease.
                    # Choose a candidate batch that is a small hyperlocal
                    # perturbation around the preferred size. We cycle
                    # through preferred, +1, -1 so we get neighbour
                    # measurements for a centered gradient estimate.
                    base_pref = int(preferred_batch_blocks)
                    probe = probe_sequence[probe_idx]
                    probe_idx = (probe_idx + 1) % len(probe_sequence)
                    candidate = base_pref + probe
                    # Clamp to allowed range.
                    candidate = min(producer_max_batch_blocks, max(producer_batch_blocks, candidate))
                    batch_blocks = candidate
                    total_frames = block_frames * max(1, batch_blocks)
                    # Render exactly once for the chosen batch size. Do NOT
                    # perform additional probe renders — every render must be
                    # a real render that will be consumed. Choose the largest
                    # batch required to fill the queue target (as calculated
                    # above) and render it in one go.
                    buffer, meta = _render_audio_frames(total_frames)

                    # Update batching policy from the actual measured efficiency
                    # Require producer metadata to be present and trust the
                    # renderer to return the requested number of frames.
                    # Warn (do not crash) if the renderer did not provide
                    # expected metadata or honoured the requested frames.
                    if meta is None:
                        STATUS_PRINTER.emit(
                            "[AudioProducer] warning: renderer returned no metadata",
                            force=True,
                        )
                        meta = {}
                    if "produced_time" not in meta or "render_duration" not in meta:
                        STATUS_PRINTER.emit(
                            f"[AudioProducer] warning: incomplete renderer metadata: {list(meta.keys())}",
                            force=True,
                        )
                    produced_time = float(meta.get("produced_time", (buffer.shape[0] / sample_rate) if sample_rate else 0.0))
                    render_duration = float(meta.get("render_duration", 0.0))
                    if buffer.shape[0] != total_frames:
                        STATUS_PRINTER.emit(
                            f"[AudioProducer] warning: renderer returned {buffer.shape[0]} frames but {total_frames} were requested",
                            force=True,
                        )
                    efficiency = float("inf") if render_duration <= 0.0 else (produced_time / render_duration)

                    # Do not record per-render here; record per-chunk when chunks
                    # are actually enqueued below so the x value matches the
                    # chunk metadata (chunk_meta['batch_blocks']).

                    # Smooth the observed efficiency (useful for display and
                    # to reduce noise before computing finite differences).
                    if efficiency_ema is None:
                        efficiency_ema = efficiency
                    else:
                        efficiency_ema = efficiency_ema + efficiency_alpha * (efficiency - efficiency_ema)

                    # Record this measurement in the small recent map so we
                    # can compute centred finite differences over the ±1
                    # neighbourhood. Keep the map tiny to avoid memory use.
                    try:
                        last_eff_map[int(batch_blocks)] = float(efficiency)
                        # Keep only the most recent few sizes (e.g. 9)
                        if len(last_eff_map) > 9:
                            # drop the oldest entry (arbitrary eviction)
                            oldest = sorted(last_eff_map.keys())[0]
                            del last_eff_map[oldest]
                    except Exception:
                        pass

                    # Compute a hyperlocal gradient estimate around the
                    # current integer preferred size using a centred
                    # difference when both neighbours are present.
                    grad: float | None = None
                    p = int(preferred_batch_blocks)
                    eff_p = last_eff_map.get(p)
                    eff_p_plus = last_eff_map.get(p + 1)
                    eff_p_minus = last_eff_map.get(p - 1)
                    if eff_p_plus is not None and eff_p_minus is not None:
                        # centred difference (per-block)
                        grad = (eff_p_plus - eff_p_minus) / 2.0
                    elif eff_p_plus is not None and eff_p is not None:
                        grad = (eff_p_plus - eff_p) / 1.0
                    elif eff_p_minus is not None and eff_p is not None:
                        grad = (eff_p - eff_p_minus) / 1.0

                    if grad is not None:
                        if grad_ema is None:
                            grad_ema = grad
                        else:
                            grad_ema = grad_ema + grad_alpha * (grad - grad_ema)

                    # Move preferred size by one block in the direction
                    # of increasing efficiency if the EMA'd gradient is
                    # significantly non-zero. This yields gradual,
                    # stable steps instead of jumping to bounds.
                    if grad_ema is not None:
                        if grad_ema > grad_threshold:
                            preferred_batch_blocks = min(producer_max_batch_blocks, preferred_batch_blocks + 1)
                        elif grad_ema < -grad_threshold:
                            preferred_batch_blocks = max(producer_batch_blocks, preferred_batch_blocks - 1)

                    # Split the returned buffer into full blocks and a final
                    # partial remainder (if any). Do NOT drop or mutate
                    # frames; emit warnings for mismatches.
                    total_samples = buffer.shape[0]
                    full_blocks = total_samples // block_frames
                    remainder = total_samples % block_frames
                    slices = full_blocks + (1 if remainder else 0)
                    if slices == 0:
                        # Nothing produced; warn and skip.
                        STATUS_PRINTER.emit(
                            "[AudioProducer] warning: renderer produced zero frames",
                            force=True,
                        )
                        continue
                    per_chunk_duration = float(render_duration) / float(slices) if slices else 0.0
                    allotted_per_chunk = float(meta.get("allotted_time", (total_samples / sample_rate) if sample_rate else 0.0)) / float(slices)
                    # Emit full-block slices
                    for idx in range(full_blocks):
                        start = idx * block_frames
                        end = start + block_frames
                        chunk = buffer[start:end].copy()
                        chunk_meta = {
                            "render_duration": per_chunk_duration,
                            "allotted_time": allotted_per_chunk,
                            "produced_frames": block_frames,
                            "produced_time": block_frames / sample_rate if sample_rate else 0.0,
                            "batch_blocks": full_blocks + (1 if remainder else 0),
                            "batch_index": idx,
                            "queue_underflow": False,
                        }
                        node_timings_meta = meta.get("node_timings") if idx == 0 else None
                        if node_timings_meta:
                            chunk_meta["node_timings"] = dict(node_timings_meta)
                        # Aggressive non-blocking put with adaptive sleep/backoff
                        # if the queue is full. Use put_nowait to avoid the
                        # 50ms timeout and allow fast retries while backing off
                        # if the queue remains saturated.
                        try:
                            pcm_queue.put_nowait((chunk, chunk_meta))
                            # Record per-chunk efficiency point for UI
                            try:
                                with efficiency_lock:
                                    # Record the producer's chosen batch size (candidate)
                                    efficiency_points.append((int(batch_blocks), float(efficiency)))
                                    preferred_batch_snapshot = int(preferred_batch_blocks)
                            except Exception:
                                pass
                        except queue.Full:
                            STATUS_PRINTER.emit(
                                "[AudioProducer] PCM queue full: stashing chunk for requeue",
                                force=True,
                            )
                            stashed_chunks.append((chunk, chunk_meta))
                            break
                    # Handle the remainder partial chunk, if any
                    if remainder:
                        start = full_blocks * block_frames
                        end = start + remainder
                        chunk = buffer[start:end].copy()
                        chunk_meta = {
                            "render_duration": per_chunk_duration,
                            "allotted_time": allotted_per_chunk,
                            "produced_frames": remainder,
                            "produced_time": remainder / sample_rate if sample_rate else 0.0,
                            "batch_blocks": 1,
                            "batch_index": full_blocks,
                            "queue_underflow": False,
                        }
                        node_timings_meta = meta.get("node_timings") if full_blocks == 0 else None
                        if node_timings_meta:
                            chunk_meta["node_timings"] = dict(node_timings_meta)
                        try:
                            pcm_queue.put_nowait((chunk, chunk_meta))
                            try:
                                with efficiency_lock:
                                    # Record the producer's chosen batch size (candidate)
                                    efficiency_points.append((int(batch_blocks), float(efficiency)))
                                    preferred_batch_snapshot = int(preferred_batch_blocks)
                            except Exception:
                                pass
                        except queue.Full:
                            STATUS_PRINTER.emit(
                                "[AudioProducer] PCM queue full: stashing remainder chunk for requeue",
                                force=True,
                            )
                            stashed_chunks.append((chunk, chunk_meta))
                        if not running or producer_stop_event.is_set():
                            break
            except Exception as exc:  # pragma: no cover - depends on audio backend
                # Capture full traceback for diagnostics
                tb = traceback.format_exc()
                STATUS_PRINTER.emit(f"[AudioProducer] Exception:\n{tb}", force=True)
                audio_failures.append(exc)
                running = False

        producer_thread_obj = threading.Thread(target=producer_thread, name="AudioProducer", daemon=True)
        producer_thread_obj.start()

        def _requeue_daemon() -> None:
            """Background helper that attempts to move stashed chunks
            back into `pcm_queue`. Emits warnings when queue remains
            saturated. Runs as a daemon so it does not block shutdown.
            """
            try:
                while running:
                    try:
                        item = None
                        # Pop leftmost stashed chunk if available
                        if stashed_chunks:
                            item = stashed_chunks.popleft()
                        if item is None:
                            time.sleep(0.01)
                            continue
                        chunk, chunk_meta = item
                        # Block briefly while attempting to put so we do not
                        # spin or lose data. If the queue is still full after
                        # a short timeout, emit a warning and retry.
                        try:
                            # Put the full (chunk, meta) item back into the queue.
                            # Use a blocking put with a short timeout so the daemon
                            # does not spin if the consumer is briefly saturated.
                            pcm_queue.put((chunk, chunk_meta), block=True, timeout=0.1)
                        except Exception:
                            # Could not place it; re-stash and back off.
                            STATUS_PRINTER.emit(
                                "[RequeueDaemon] warning: pcm_queue still full, will retry",
                                force=True,
                            )
                            stashed_chunks.appendleft(item)
                            time.sleep(0.05)
                    except Exception:
                        # Ensure the daemon doesn't die silently.
                        STATUS_PRINTER.emit(
                            f"[RequeueDaemon] exception: {traceback.format_exc()}",
                            force=True,
                        )
                        time.sleep(0.1)
            except Exception:
                STATUS_PRINTER.emit(f"[RequeueDaemon] fatal: {traceback.format_exc()}", force=True)

        threading.Thread(target=_requeue_daemon, name="RequeueDaemon", daemon=True).start()

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
                tb = traceback.format_exc()
                STATUS_PRINTER.emit(f"[AudioThread] Exception:\n{tb}", force=True)
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
        lines: list[Any],
        freq_target: float,
        velocity_target: float,
    ) -> None:
        nonlocal last_freq_target_display, last_velocity_target_display
        nonlocal last_node_levels_snapshot, last_node_timings_snapshot
        nonlocal status_signature_pending, last_status_signature

        text_cache.start_frame()
        try:
            if screen is None:
                return

            screen.fill((10, 10, 16))

            nodes_in_graph = list(getattr(graph, "_nodes", {}).keys())
            if not nodes_in_graph:
                pygame.display.flip()
                return

            pending_signature = status_signature_pending
            if pending_signature is not None and pending_signature != last_status_signature:
                text_cache.invalidate("__status__")
                last_status_signature = pending_signature

            node_levels = getattr(graph, "last_node_levels", {})
            node_timings = getattr(graph, "last_node_timings", {})

            freq_changed = (
                last_freq_target_display is None
                or not np.isclose(freq_target, last_freq_target_display, atol=1e-4)
            )
            vel_changed = (
                last_velocity_target_display is None
                or not np.isclose(velocity_target, last_velocity_target_display, atol=1e-4)
            )
            if freq_changed:
                last_freq_target_display = freq_target
            if vel_changed:
                last_velocity_target_display = velocity_target
            if freq_changed or vel_changed:
                for name in nodes_in_graph:
                    node = graph._nodes.get(name)
                    if isinstance(node, nodes.OscNode):
                        text_cache.invalidate(name)

            updated_levels: dict[str, np.ndarray] = {}
            updated_timings: dict[str, float] = {}
            changed_nodes: set[str] = set()
            for name in nodes_in_graph:
                levels = node_levels.get(name)
                if isinstance(levels, np.ndarray):
                    updated_levels[name] = np.array(levels, copy=True)
                    prev = last_node_levels_snapshot.get(name)
                    if prev is None or prev.shape != levels.shape or not np.allclose(prev, levels, atol=1e-4):
                        changed_nodes.add(name)
                timing_value = node_timings.get(name)
                if timing_value is not None:
                    timing_float = float(timing_value)
                    updated_timings[name] = timing_float
                    prev_timing = last_node_timings_snapshot.get(name)
                    if prev_timing is None or abs(prev_timing - timing_float) > 1e-6:
                        changed_nodes.add(name)
                elif name in last_node_timings_snapshot:
                    changed_nodes.add(name)
            for removed in set(last_node_levels_snapshot) - set(updated_levels):
                changed_nodes.add(removed)
            for removed in set(last_node_timings_snapshot) - set(updated_timings):
                changed_nodes.add(removed)
            if changed_nodes:
                for node_name in changed_nodes:
                    text_cache.invalidate(node_name)
            last_node_levels_snapshot = updated_levels
            last_node_timings_snapshot = updated_timings

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

            def _render_text(
                node_key: str,
                text: str,
                colour: tuple[int, int, int],
                font_obj,
            ):
                font_key = "large" if font_obj is font else "small"
                return text_cache.fetch(
                    node_key,
                    text,
                    colour,
                    font_key,
                    lambda: font_obj.render(text, True, colour),
                )

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
                title = _render_text(name, header_text, (250, 250, 250), header_font)
                screen.blit(title, (rect.x + 10, rect.y + 6))

                lines_to_render: list[str] = []
                node_time = node_timings.get(name)
                # Prefer the HUD per-node EMA (ms) for display when available so
                # per-node timing boxes don't jump every frame.
                hud_node_time = hud_node_timings.get(name)
                # Do NOT fall back to instantaneous values — display only the
                # HUD-smoothed per-node EMA when available. This prevents noisy
                # instant timings from showing up in the GUI.
                if hud_node_time is not None:
                    lines_to_render.append(f"time={hud_node_time:0.3f}ms")
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
                    text_surface = _render_text(name, text_line, (240, 240, 240), font_small)
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
                            label_surface = _render_text(name, label, (210, 210, 210), font_small)
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
                            peak_text = _render_text(name, f"{level:0.2f}", (190, 190, 190), font_small)
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
                header = _render_text(
                    f"group:{group_name}",
                    f"{group_name} [{member_count}]",
                    (235, 235, 245),
                    font,
                )
                screen.blit(header, (rect.x + 12, rect.y + 8))

            legend_x = margin
            legend_y = height - (len(lines) + 3) * (font.get_height() + 4)
            if legend_y < rows * (tile_h + margin) + margin:
                legend_y = rows * (tile_h + margin) + margin

            pygame.draw.line(screen, audio_colour, (legend_x, legend_y), (legend_x + 24, legend_y), 3)
            label_audio = _render_text("__legend__", "audio", (200, 200, 200), font_small)
            screen.blit(label_audio, (legend_x + 32, legend_y - font_small.get_height() // 2))

            legend_y += font_small.get_height() + 8
            pygame.draw.line(screen, mod_colour, (legend_x, legend_y), (legend_x + 24, legend_y), 2)
            label_mod = _render_text("__legend__", "mod", (200, 200, 200), font_small)
            screen.blit(label_mod, (legend_x + 32, legend_y - font_small.get_height() // 2))

            info_y = legend_y + font_small.get_height() + 12
            for line in lines:
                if isinstance(line, tuple):
                    text_value, colour = line
                else:
                    text_value = cast(str, line)
                    colour = (255, 255, 255)
                text = _render_text("__status__", text_value, colour, font)
                screen.blit(text, (margin, info_y))
                info_y += font.get_height() + 4

            # Efficiency exploration scatter inset (UI telemetry)
            try:
                with efficiency_lock:
                    points = list(efficiency_points)
                    pref = preferred_batch_snapshot
            except Exception:
                points = []
                pref = None

            if points:
                inset_w = 220
                inset_h = 120
                inset_x = width - inset_w - margin
                inset_y = margin
                pygame.draw.rect(screen, (18, 18, 24), (inset_x, inset_y, inset_w, inset_h), border_radius=6)
                pygame.draw.rect(screen, (120, 120, 140), (inset_x, inset_y, inset_w, inset_h), width=1, border_radius=6)

                # Use exact numeric batch sizes on the x axis (no jitter).
                batches = [float(p[0]) for p in points]
                effs = [p[1] for p in points]

                orig_min_b = float(min(batches))
                orig_max_b = float(max(batches))
                span = orig_max_b - orig_min_b
                # Use the exact data range for scaling. If there's no span,
                # expand a bit so the axis is not degenerate.
                if span <= 0.0:
                    min_b = orig_min_b - 1.0
                    max_b = orig_max_b + 1.0
                else:
                    min_b = orig_min_b
                    max_b = orig_max_b

                min_e, max_e = min(effs), max(effs)
                if abs(max_e - min_e) < 1e-6:
                    min_e = 0.0
                    max_e = max_e + 1.0

                def x_of(b: float) -> int:
                    return int(inset_x + 8 + (inset_w - 16) * ((b - min_b) / (max_b - min_b)))

                def y_of(e: float) -> int:
                    return int(inset_y + inset_h - 8 - (inset_h - 16) * ((e - min_e) / (max_e - min_e)))

                label = _render_text("__eff_title__", f"efficiency explorer", (200, 200, 200), font_small)
                screen.blit(label, (inset_x + 8, inset_y + 4))

                # Debug overlay: show computed x-range and a few batch values
                try:
                    dbg_lines = []
                    dbg_lines.append(f"count={len(points)}")
                    dbg_lines.append(f"min={min_b:.3f}")
                    dbg_lines.append(f"max={max_b:.3f}")
                    sample_batches = ",".join(str(int(x)) for x in batches[:6])
                    if len(batches) > 6:
                        sample_batches += ",.."
                    dbg_lines.append(f"b:{sample_batches}")
                    dbg_text = " | ".join(dbg_lines)
                    dbg_surf = _render_text("__eff_dbg__", dbg_text, (200, 200, 160), font_small)
                    screen.blit(dbg_surf, (inset_x + 8, inset_y + inset_h - dbg_surf.get_height() - 6))
                except Exception:
                    pass

                # Occasional log of sampled batches for debugging
                try:
                    nowt = time.monotonic()
                    if nowt - efficiency_debug_last_emit >= efficiency_debug_emit_interval:
                        sample_batches = ",".join(str(int(x)) for x in batches[:8])
                        if len(batches) > 8:
                            sample_batches += ",.."
                        STATUS_PRINTER.emit(f"[EffDebug] count={len(points)} batches={sample_batches}")
                        efficiency_debug_last_emit = nowt
                except Exception:
                    pass

                # plot last N points to avoid excessive draw cost
                subset = list(zip(batches, effs))[-512:]
                for b, e in subset:
                    try:
                        px = x_of(b)
                        py = y_of(e)
                        pygame.draw.circle(screen, (120, 220, 120), (px, py), 2)
                    except Exception:
                        continue

                # autoscaled scatter only; no extra ticks or labels

            pygame.display.flip()
        finally:
            text_cache.finish_frame()

    STATUS_PRINTER.emit(
        "Menu:'m'  LS=pitch/vel  RS=filter.  A=trigger  B=mode  X=wave  Y=cycle  N=source (osc/sampler).",
        force=True,
    )
    STATUS_PRINTER.emit(
        "Bumpers: hold=momentary, double-tap=latch. RB default FREE; LB default 12tet/full.",
        force=True,
    )
    STATUS_PRINTER.emit("Z cycles FREE variant. </> root down/up, / resets root.", force=True)

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
                        STATUS_PRINTER.emit(f"Source → {state['source_type'].upper()}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("wave_next", pygame.K_x):
                        state["wave_idx"] = (state["wave_idx"] + 1) % len(state["waves"])
                        graph, envelope_names, amp_mod_names = build_runtime_graph(sample_rate, state)
                        pitch_node = graph._nodes.get("pitch")
                        STATUS_PRINTER.emit(f"Waveform → {state['waves'][state['wave_idx']]}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("free_variant_next", pygame.K_z):
                        i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                        state["free_variant"] = quantizer.FREE_VARIANTS[(i + 1) % len(quantizer.FREE_VARIANTS)]
                        pitch_free_variant = state["free_variant"]
                        STATUS_PRINTER.emit(f"FREE variant → {state['free_variant']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("drone_toggle", pygame.K_b):
                        cycle_envelope_mode()
                    elif event.key == state["keymap"].get("root_up", pygame.K_PERIOD):
                        state["root_midi"] = min(127, state["root_midi"] + 1)
                        root_midi_value = state["root_midi"]
                        STATUS_PRINTER.emit(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_down", pygame.K_COMMA):
                        state["root_midi"] = max(0, state["root_midi"] - 1)
                        root_midi_value = state["root_midi"]
                        STATUS_PRINTER.emit(f"Root MIDI → {state['root_midi']}")
                        menu_instance.draw()
                    elif event.key == state["keymap"].get("root_reset", pygame.K_SLASH):
                        state["root_midi"] = 60
                        root_midi_value = state["root_midi"]
                        STATUS_PRINTER.emit("Root MIDI → 60 (C4)")
                        menu_instance.draw()

            pygame.event.pump()


            # All controller input is now handled by ControllerMonitor and written to control history.
            # If UI needs to know the latest state, it can query controller_monitor.get_latest_curves(),
            # but all graph and node access must use the history.

            # All controller state for graph/nodes is now sourced from control history.
            # If UI needs to display the latest controller state, use controller_monitor.get_latest_curves().
            # Remove all direct access to axes_current, buttons_current, prev_buttons, etc.

            with callback_timing_lock:
                timing_snapshot = list(callback_timing_samples)

            render_mean_ms = render_peak_ms = allotted_ms = 0.0
            produced_mean_ms = produced_peak_ms = produced_latest_ms = 0.0
            period_mean_ms = period_peak_ms = 0.0
            # Smooth (slow-changing) EMAs are provided by the audio callback samples.
            render_ema_ms = produced_ema_ms = period_ema_ms = 0.0
            underrun_recent = False
            if timing_snapshot:
                render_values = [entry["render_duration"] for entry in timing_snapshot]
                render_mean_ms = sum(render_values) / len(render_values) * 1000.0
                render_peak_ms = max(render_values) * 1000.0
                allotted_ms = timing_snapshot[-1]["allotted_time"] * 1000.0

                # Exclude samples that were silence produced due to underrun
                produced_values_all = [
                    entry["produced_time"]
                    for entry in timing_snapshot
                    if entry.get("produced_time") is not None
                ]
                produced_values = [
                    entry["produced_time"]
                    for entry in timing_snapshot
                    if entry.get("produced_time") is not None and not entry.get("queue_underflow", False)
                ]
                if produced_values:
                    produced_mean_ms = sum(produced_values) / len(produced_values) * 1000.0
                    produced_peak_ms = max(produced_values) * 1000.0
                    # Use the most recent non-underrun sample if available
                    produced_latest_ms = next((e.get("produced_time") for e in reversed(timing_snapshot) if e.get("produced_time") is not None and not e.get("queue_underflow", False)), 0.0) * 1000.0
                else:
                    # No valid non-underrun produced samples; keep produced_* at 0
                    produced_mean_ms = 0.0
                    produced_peak_ms = 0.0
                    produced_latest_ms = 0.0

                period_values = [entry["callback_period"] for entry in timing_snapshot if entry["callback_period"] > 0.0]
                if period_values:
                    period_mean_ms = sum(period_values) / len(period_values) * 1000.0
                    period_peak_ms = max(period_values) * 1000.0
                # For display purposes prefer the most-recent non-underrun
                # timing sample's batch size. This prevents a silent
                # underrun-produced chunk (which reports batch_blocks=1)
                # from causing the UI to show a single produced batch when
                # the producer had been rendering larger batches.
                last_non_underrun = next((e for e in reversed(timing_snapshot) if not e.get("queue_underflow", False)), None)
                if last_non_underrun is not None:
                    batch_blocks = max(1, int(last_non_underrun.get("batch_blocks", 1)))
                else:
                    batch_blocks = max(1, int(timing_snapshot[-1].get("batch_blocks", 1)))
                if timing_snapshot[-1].get("render_ema") is not None:
                    render_ema_ms = timing_snapshot[-1]["render_ema"] * 1000.0
                if timing_snapshot[-1].get("produced_ema") is not None:
                    produced_ema_ms = timing_snapshot[-1]["produced_ema"] * 1000.0
                if timing_snapshot[-1].get("period_ema"):
                    period_ema_ms = timing_snapshot[-1]["period_ema"] * 1000.0
                queue_depth_recent = timing_snapshot[-1].get("queue_depth", 0)
                queue_capacity_recent = timing_snapshot[-1].get("queue_capacity", 0)
                underrun_recent = any(entry.get("underrun") for entry in timing_snapshot)

            current_status_signature = (
                int(state["root_midi"]),
                round(float(freq_target), 3),
                round(float(velocity_target), 3),
                round(float(cutoff_target), 2),
                round(float(q_target), 3),
                state["filter_type"],
                state["source_type"],
                state["waves"][state["wave_idx"]],
                envelope_mode,
                state["mod_wave_types"][state["mod_wave_idx"]],
                round(float(state["mod_rate_hz"]), 3),
                round(float(state["mod_depth"]), 3),
                bool(state["mod_use_input"]),
                pitch_effective_token,
                state["base_token"],
                state["free_variant"],
            )
            status_signature_pending = current_status_signature

            base_line_colour = (255, 255, 255)
            lines_with_colour: list[tuple[str, tuple[int, int, int]]] = [
                (
                    f"Eff:{effective_token:<16} Base:{state['base_token']:<16} Free:{state['free_variant']:<10} Src:{state['source_type']}",
                    base_line_colour,
                ),
                (
                    f"Root:{state['root_midi']:3d}  Freq:{freq_target:7.2f}Hz  Vel:{velocity_target:.2f}",
                    base_line_colour,
                ),
                (
                    f"Cut:{cutoff_target:7.1f}Hz  Q:{q_target:4.2f}  Filter:{state['filter_type']}",
                    base_line_colour,
                ),
                (
                    f"Wave:{state['waves'][state['wave_idx']]}  Mode:{envelope_mode.title():<9}  LFO:{state['mod_wave_types'][state['mod_wave_idx']]}/{state['mod_rate_hz']:.2f}Hz d={state['mod_depth']:.2f} src={'input' if state['mod_use_input'] else 'free'}",
                    base_line_colour,
                ),
            ]

            console_lines = [entry[0] for entry in lines_with_colour]

            if timing_snapshot:
                # Use instantaneous aggregated measures (ms) to update HUD EMAs
                inst_render_ms = render_mean_ms
                # Prefer the latest non-underrun produced time, fall back to mean
                inst_produced_ms = produced_latest_ms if produced_latest_ms else produced_mean_ms
                inst_period_ms = period_mean_ms

                # Compute time-based alphas for HUD EMAs from loop dt -> alpha = 1 - exp(-dt / tau)
                now_t = time.perf_counter()
                dt = now_t - hud_last_update if now_t > hud_last_update else 0.0
                # clamp dt to a very small maximum to avoid large alpha spikes when UI stalls
                dt = max(1e-6, min(0.02, dt))
                hud_last_update = now_t

                hud_alpha_global = 1.0 - math.exp(-dt / hud_time_constant_global)
                hud_alpha_node = 1.0 - math.exp(-dt / hud_time_constant_node)

                def _hud_update(prev: float | None, value: float, alpha: float) -> float:
                    if prev is None:
                        # seed new EMAs with the first observed value
                        return value
                    return prev + alpha * (value - prev)

                # Update global EMAs with the global alpha
                hud_render_ms = _hud_update(hud_render_ms, float(inst_render_ms or 0.0), hud_alpha_global)
                hud_produced_ms = _hud_update(hud_produced_ms, float(inst_produced_ms or 0.0), hud_alpha_global)
                hud_period_ms = _hud_update(hud_period_ms, float(inst_period_ms or 0.0), hud_alpha_global)

                render_display = hud_render_ms or 0.0
                produced_display = hud_produced_ms or 0.0
                period_display = hud_period_ms or 0.0

                render_text = f"render {render_display:5.2f}ms pk {render_peak_ms:5.2f}ms"
                # Only show produced audio stats when we have at least one
                # non-underrun produced sample. Silence produced due to
                # underrun should not be displayed as completed audio.
                audio_section = ""
                if produced_mean_ms or produced_peak_ms or produced_latest_ms:
                    audio_section = f"audio {produced_display:5.2f}ms"
                    if batch_blocks > 1:
                        audio_section += f" ({batch_blocks}×)"

                # Build a stable `audio_text` variable. When there are no
                # non-underrun produced samples, avoid attaching numeric
                # produced-time stats but still display a minimal placeholder
                # (e.g. "audio") so the UI shows an audio section during an
                # underrun period.
                if audio_section:
                    audio_text = audio_section
                else:
                    audio_text = "audio" if underrun_recent else ""

                # Append aggregate numbers only when we actually have
                # non-underrun produced measurements.
                if produced_mean_ms or produced_peak_ms:
                    audio_text += f" avg {produced_mean_ms:5.2f}ms pk {produced_peak_ms:5.2f}ms"
                if produced_ema_ms:
                    audio_text += f" ema {produced_ema_ms:5.2f}ms"

                if audio_text:
                    budget_text = f"Audio: {render_text} | {audio_text} | budget {allotted_ms:5.2f}ms"
                else:
                    budget_text = f"Audio: {render_text} | budget {allotted_ms:5.2f}ms"
                period_text = ""
                if period_mean_ms or period_peak_ms or period_ema_ms:
                    period_text = (
                        f" period {period_mean_ms:5.2f}/{period_peak_ms:5.2f}ms"
                    )
                    if period_ema_ms:
                        period_text += f" ema {period_ema_ms:5.2f}ms"
                queue_text = ""
                if queue_capacity_recent:
                    queue_text = f" queue {queue_depth_recent}/{queue_capacity_recent}"
                underrun_text = " UNDERRUN" if underrun_recent else ""
                timing_text = budget_text + period_text + queue_text + underrun_text
                timing_colour = (240, 120, 120) if underrun_recent else (180, 220, 255)
                # Strict rule: if the most-recent dequeued chunk was an
                # underrun / silence produced by the callback (queue_underflow
                # or underrun), do NOT print the bottom timing line. This
                # ensures we don't report produced silence. Leave other UI
                # elements unchanged.
                last_sample = timing_snapshot[-1] if timing_snapshot else {}
                last_was_underrun = bool(last_sample.get("queue_underflow") or last_sample.get("underrun"))
                # If the most-recent dequeued chunk was silence produced by
                # the callback (queue_underflow/underrun), do not print the
                # bottom timing line at all. This strictly avoids reporting
                # produced silence. All other UI output remains unchanged.
                if not last_was_underrun:
                    lines_with_colour.append((timing_text, timing_colour))
                    console_lines.append(timing_text)
                latest_node_timings = timing_snapshot[-1].get("node_timings") or {}
                # Update per-node HUD EMAs from the latest node timings (convert to ms)
                if latest_node_timings:
                    for name, duration in latest_node_timings.items():
                        value_ms = float(duration) * 1000.0
                        prev = hud_node_timings.get(name)
                        hud_node_timings[name] = _hud_update(prev, value_ms, hud_alpha_node)

                    sorted_nodes = sorted(
                        latest_node_timings.items(), key=lambda item: item[1], reverse=True
                    )
                    top_entries = ", ".join(
                        f"{name}:{hud_node_timings.get(name, duration * 1000.0):0.2f}ms" for name, duration in sorted_nodes[:3]
                    )
                    nodes_text = f"Nodes: {top_entries}"
                    lines_with_colour.append((nodes_text, (200, 210, 255)))
                    console_lines.append(nodes_text)
            screen = pygame.display.get_surface()
            if screen:
                draw_visualisation(screen, lines_with_colour, freq_target, velocity_target)
            should_emit = False
            if (
                last_console_signature is None
                or current_status_signature != last_console_signature
            ):
                should_emit = True
            elif last_timing_alert is None or underrun_recent != last_timing_alert:
                should_emit = True

            if should_emit:
                STATUS_PRINTER.emit("\r" + " | ".join(console_lines), end="")

            last_console_signature = current_status_signature
            last_timing_alert = underrun_recent

            clock.tick(120)
    finally:
        running = False
        producer_stop_event.set()
        if producer_thread_obj is not None:
            producer_thread_obj.join(timeout=1.0)
        if pcm_queue is not None:
            try:
                while True:
                    pcm_queue.get_nowait()
            except queue.Empty:
                pass
        if hasattr(joy, "quit"):
            try:
                joy.quit()
            except Exception:
                # Defensive: some pygame setups may raise when joystick subsystem
                # has already been shut down; ignore errors during cleanup.
                pass
        STATUS_PRINTER.flush()

    return 0


__all__ = ["run", "build_runtime_graph"]
