# amp.py
import os, json, time, threading
import pygame, sounddevice as sd
from amp import config, utils, quantizer, persistence, menu, nodes, audio_graph

# =========================
# Runtime setup
# =========================
pygame.init(); pygame.joystick.init()
pygame.display.set_caption("Controller Synth (Graph)")
pygame.display.set_mode((600, 180))
pygame.font.init(); font=pygame.font.SysFont("monospace",14)

if pygame.joystick.get_count()==0:
    print("No joystick. Connect controller and restart."); raise SystemExit
joy=pygame.joystick.Joystick(0); joy.init()

SAMPLE_RATE = 44100
freq_target=220.0; freq_current=220.0
amp_target=0.0; amp_current=0.0
cutoff_target=1000.0; cutoff_current=1000.0
q_target=0.8; q_current=0.8
gate_momentary=False; drone_on=False

# Default state
waves=["sine","square","saw","triangle"]

state={
    # Base selection (tuning/mode token) + root
    "base_token":"12tet/full",       # equal temperament, full set
    "root_midi": 60,                 # C4
    # FREE mode variant (affects pitch mapping only when effective_token == "FREE")
    "free_variant":"continuous",     # "continuous" | "weighted" | "stepped"
    # Waves, filter, source
    "waves":waves, "wave_idx":0,
    "filter_types":["lowpass","highpass","bandpass","notch","peaking"],
    "filter_type":"lowpass",
    "filter_axis_cutoff": min(4, joy.get_numaxes()-1) if joy.get_numaxes()>4 else 3,
    "filter_axis_q":      min(5, joy.get_numaxes()-1) if joy.get_numaxes()>5 else 4,
    "peaking_gain_db":6.0,
    "source_type":"osc",
    "sample_file": os.path.join(os.path.dirname(__file__), "sample.wav"),
    # LFO / modulator
    "mod_wave_types":["sine","square","saw","triangle"],
    "mod_wave_idx":0,
    "mod_rate_hz":4.0,
    "mod_depth":0.5,
    "mod_route":"both",   # "freq","amp","both"
    "mod_use_input":False,
    "mod_slew_ms":5.0,
    # Controls
    "keymap":{
        "toggle_menu": pygame.K_m, "open_keymap": pygame.K_k,
        "wave_next": pygame.K_x, "mode_next": pygame.K_y, "drone_toggle": pygame.K_b,
        "toggle_source": pygame.K_n,
        "free_variant_next": pygame.K_z,
        "root_up": pygame.K_PERIOD, "root_down": pygame.K_COMMA, "root_reset": pygame.K_SLASH,
    },
    # Bumpers as mode switches (no behavior flags; hold=momentary, double-tap=latch)
    # token is either "FREE" or "<tuning>/<mode>"
    "buttonmap": {
        4: {"token": "12tet/full"},  # LB: equal temperament full-grid
        5: {"token": "FREE"},        # RB: FREE (uses current base grid for weighted/stepped)
    },
    "bumper_priority": [4,5],
    "double_tap_window": 0.33,
    "free_variant_button": 6,   # (optional) controller button to cycle FREE variant
}
persistence.load_mappings(state)

# optional sampler
sampler=None
if os.path.isfile(state["sample_file"]):
    try:
        sampler=nodes.Sampler(state["sample_file"],loop=True)
        print(f"[Sampler] Loaded '{os.path.basename(state['sample_file'])}' at {sampler.file_sr} Hz")
    except Exception as e:
        print(f"[Sampler] Disabled: {e}")
else:
    print("[Sampler] sample.wav not found (sampler available after you add a file).")

# graph build
def build_graph(fs, state):
    # Config toggles (can be set in state)
    use_subharm = state.get("use_subharm", True)
    use_normalizer = state.get("use_normalizer", True)
    use_hardclip = state.get("use_hardclip", False)

    # Oscillator setup
    osc_waves = ["sine", "square", "saw"]
    osc_nodes = []
    pan_lfos = []
    am_lfos = []
    for i, wave in enumerate(osc_waves):
        osc = nodes.OscNode(f"osc{i+1}", wave=wave)
        pan_lfo = nodes.LFONode(f"pan_lfo{i+1}", wave="sine", rate_hz=0.2 + 0.1*i, depth=1.0)
        am_lfo = nodes.LFONode(f"am_lfo{i+1}", wave="sine", rate_hz=4.0 + i, depth=0.5)
        osc_nodes.append(osc)
        pan_lfos.append(pan_lfo)
        am_lfos.append(am_lfo)

    # Add nodes to graph
    G = audio_graph.AudioGraph(fs)
    for osc in osc_nodes: G.add_node(osc)
    for pan in pan_lfos: G.add_node(pan)
    for am in am_lfos: G.add_node(am)

    # AM modulation (vibrato intensity)
    for i, osc in enumerate(osc_nodes):
        G.connect_mod(am_lfos[i].name, osc.name, "amp", scale=1.0, mode="add")

    # Pan modulation (stereo placement)
    # Each oscillator will have a pan value modulated by its pan LFO
    # The mixer will use these pan values for stereo placement
    # We'll pass pan as a param to the mixer node
    # For now, collect pan signals for later use

    # Subharmonic generator
    if use_subharm:
        subharm = nodes.SubharmonicGeneratorNode("subharm", n_ch=1, mix=0.4, divisions=(2,3))
        G.add_node(subharm)
        for osc in osc_nodes:
            G.connect_audio(osc.name, subharm.name)
        mixer_in = subharm.name
    else:
        mixer_in = osc_nodes[0].name  # If only one osc, connect directly

    # Mixer node (reduces to stereo)
    mixer = nodes.MixNode("mixer", out_channels=2, alc=state.get("use_normalizer", True), compression="tanh" if not state.get("use_hardclip", False) else "clip")
    G.add_node(mixer)
    G.connect_audio(mixer_in, mixer.name)

    # Safety filter
    safety = nodes.SafetyFilterNode("safety", fs, n_ch=2)
    G.add_node(safety)
    G.connect_audio(mixer.name, safety.name)
    G.set_sink(safety.name)

    return G

graph = build_graph(SAMPLE_RATE, state)
menu_instance = menu.Menu(state)
menu_instance.toggle(); menu_instance.draw()

# LB/RB
prev_buttons=[0]*joy.get_numbuttons()
button_last_press = {}  # {button_index: last_press_time}
button_latch      = {}  # {button_index: bool}

# device select
def _pick_output_device_and_rate():
    try:
        dev=sd.query_devices(None,'output')
        sr=int(dev.get('default_samplerate',48000))
        return dev['index'], sr
    except Exception:
        for i,d in enumerate(sd.query_devices()):
            if d.get('max_output_channels',0)>0:
                return i, int(d.get('default_samplerate',48000))
        raise RuntimeError("No audio output device found.")

# audio callback
def audio_callback(outdata, frames, time_info, status):
    global SAMPLE_RATE, freq_current, amp_current, cutoff_current, q_current
    sr = SAMPLE_RATE
    utils._scratch.ensure(frames)

    # ramps
    f = utils.cubic_ramp(freq_current, freq_target, frames, utils._scratch.f[:frames])
    a = utils.cubic_ramp(amp_current, amp_target, frames, utils._scratch.a[:frames])
    c = utils.cubic_ramp(cutoff_current, cutoff_target, frames, utils._scratch.c[:frames])
    q = utils.cubic_ramp(q_current, q_target, frames, utils._scratch.q[:frames])

    # Build base_params for graph
    B, C = 1, 1
    base_params = {
        "_B": B, "_C": C,
        "source": {"freq": utils.as_BCF(f, B, C, frames, name="freq"),
                   "amp":  utils.as_BCF(a, B, C, frames, name="amp")},
        "filter": {"cutoff": utils.as_BCF(c, B, C, frames, name="cutoff"),
                   "Q":      utils.as_BCF(q, B, C, frames, name="Q")},
    }

    # --- Graph rendering ---
    y = graph.render(frames, sr, base_params)  # (B,C,F)
    y = utils.assert_BCF(y, name="sink")
    if y.shape[0] != 1 or y.shape[1] not in (1, outdata.shape[1]):
        raise RuntimeError(f"Device expects (1,{outdata.shape[1]},F), got {y.shape}")

    # Output assignment (stereo)
    chans = min(outdata.shape[1], y.shape[1])
    for ch in range(chans):
        outdata[:, ch] = y[0, ch].astype(np.float32, copy=False)

    # update tails
    freq_current = float(f[-1])
    amp_current = float(a[-1])
    cutoff_current = float(c[-1])
    q_current = float(q[-1])

def audio_thread():
    global SAMPLE_RATE, graph
    try:
        dev_index, dev_sr = _pick_output_device_and_rate()
        SAMPLE_RATE = dev_sr
        sd.default.device=(None, dev_index)
        print(f"\n[Audio] Device #{dev_index} @ {SAMPLE_RATE} Hz")
        graph = build_graph(SAMPLE_RATE, state)
        with sd.OutputStream(
            device=dev_index, channels=2, dtype='float32',
            samplerate=SAMPLE_RATE, blocksize=256, latency='low',
            callback=audio_callback
        ):
            while running: time.sleep(0.002)
    except sd.PortAudioError as e:
        print("[Audio] Start failed:", e); raise

running=True
threading.Thread(target=audio_thread,daemon=True).start()

# =========================
# Main loop / UI
# =========================
clock=pygame.time.Clock()
def draw_status(screen, lines):
    screen.fill((0,0,0)); y=10
    for line in lines:
        text=font.render(line,True,(255,255,255)); screen.blit(text,(10,y)); y+=16
    pygame.display.flip()

print("Menu:'m'  LS=pitch/vel  RS=filter.  A=gate  B=drone  X=wave  Y=cycle  N=source (osc/sampler).")
print("Bumpers: hold=momentary, double-tap=latch. RB default FREE; LB default 12tet/full.")
print("Z cycles FREE variant. </> root down/up, / resets root.")

while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False; pygame.quit(); raise SystemExit
        if event.type==pygame.KEYDOWN:
            if event.key==state["keymap"].get("toggle_menu",pygame.K_m):
                menu_instance.toggle(); menu_instance.draw()
            elif event.key==state["keymap"].get("toggle_source",pygame.K_n):
                state["source_type"]="sampler" if state["source_type"]=="osc" else "osc"
                graph = build_graph(SAMPLE_RATE, state); print(f"Source → {state['source_type'].upper()}"); menu_instance.draw()
            elif event.key==state["keymap"].get("wave_next",pygame.K_x):
                state["wave_idx"]=(state["wave_idx"]+1)%len(state["waves"])
                graph = build_graph(SAMPLE_RATE, state); print(f"Waveform → {state['waves'][state['wave_idx']]}"); menu_instance.draw()
            elif event.key==state["keymap"].get("free_variant_next",pygame.K_z):
                i = quantizer.FREE_VARIANTS.index(state["free_variant"])
                state["free_variant"] = quantizer.FREE_VARIANTS[(i+1)%len(quantizer.FREE_VARIANTS)]
                print(f"FREE variant → {state['free_variant']}"); menu_instance.draw()
            elif event.key==state["keymap"].get("root_up",pygame.K_PERIOD):
                state["root_midi"] = min(127, state["root_midi"]+1)
                print(f"Root MIDI → {state['root_midi']}"); menu_instance.draw()
            elif event.key==state["keymap"].get("root_down",pygame.K_COMMA):
                state["root_midi"] = max(0, state["root_midi"]-1)
                print(f"Root MIDI → {state['root_midi']}"); menu_instance.draw()
            elif event.key==state["keymap"].get("root_reset",pygame.K_SLASH):
                state["root_midi"] = 60; print("Root MIDI → 60 (C4)"); menu_instance.draw()

    pygame.event.pump()

    # Optional controller button to cycle FREE variant
    fvb = state.get("free_variant_button", 6)
    if fvb < len(prev_buttons):
        pressed = bool(joy.get_button(fvb))
        if pressed and not prev_buttons[fvb]:
            i = quantizer.FREE_VARIANTS.index(state["free_variant"])
            state["free_variant"] = quantizer.FREE_VARIANTS[(i+1)%len(quantizer.FREE_VARIANTS)]
            print(f"FREE variant → {state['free_variant']}"); menu_instance.draw()

    # --- Bumpers: hold=momentary, double-tap=latch ---
    nowt = time.time()
    for btn_idx, cfg in state["buttonmap"].items():
        if btn_idx >= len(prev_buttons): continue
        pressed_now = bool(joy.get_button(btn_idx))
        pressed_prev = prev_buttons[btn_idx]
        if pressed_now and not pressed_prev:
            last_t = button_last_press.get(btn_idx, 0.0)
            if nowt - last_t <= state["double_tap_window"]:
                button_latch[btn_idx] = not button_latch.get(btn_idx, False)
                print(f"Latch[{btn_idx}] → {'ON' if button_latch[btn_idx] else 'OFF'}  ({cfg['token']})")
                menu_instance.draw()
            button_last_press[btn_idx] = nowt

    # Resolve effective token: holds > latches > base_token
    effective_token = state["base_token"]
    # momentary holds
    for b in state.get("bumper_priority", [4,5]):
        cfg = state["buttonmap"].get(b)
        if not cfg: continue
        if b < len(prev_buttons) and joy.get_button(b):
            effective_token = cfg["token"]
            break
    else:
        # no hold; look for latches by priority
        for b in state.get("bumper_priority", [4,5]):
            if state["buttonmap"].get(b) and button_latch.get(b, False):
                effective_token = state["buttonmap"][b]["token"]
                break

    # Axes
    lx,ly = joy.get_axis(0), joy.get_axis(1)
    ax_cut=state.get("filter_axis_cutoff",3)
    ax_q  =state.get("filter_axis_q",4)
    rx_val = joy.get_axis(ax_cut) if ax_cut<joy.get_numaxes() else 0.0
    ry_val = joy.get_axis(ax_q)   if ax_q  <joy.get_numaxes() else 0.0
    rx01=(rx_val+1.0)*0.5; ry01=(1.0-ry_val)*0.5
    cutoff_target = utils.expo_map(rx01,80.0,8000.0)
    q_target      = 0.5 + ry01*(12.0-0.5)

    # --- Pitch mapping (LS -> freq) honoring FREE/WEIGHTED/STEPPED ---
    root_midi = state.get("root_midi", 60)
    root_f = quantizer.Quantizer.midi_to_freq(root_midi)
    span_oct = float(state.get("free_span_oct", 2.0))
    # Use the resolved effective_token from above
    grid = quantizer.get_reference_grid_cents(state, effective_token)
    if quantizer.is_free_mode_token(effective_token):
        N = max(1, len(utils._grid_sorted(grid)[0]))
        u = lx * span_oct * N
        fv = state.get("free_variant", "continuous")
        if fv == "continuous":
            cents = lx * span_oct * 1200.0
        elif fv == "weighted":

            cents = quantizer.grid_warp_inverse(u, grid)
        else:  # "stepped"
            cents = quantizer.grid_warp_inverse(round(u), grid)
    else:
        cents_unq = lx * span_oct * 1200.0
        u = quantizer.grid_warp_forward(cents_unq, grid)
        cents = quantizer.grid_warp_inverse(round(u), grid)
    freq_target = root_f * (2.0 ** (cents / 1200.0))

    # Amp from LY (up loud)
    amp_target=max(0.0, 1.0 - (ly + 1.0)/2.0)

    # Buttons A/B/X
    gate_momentary = bool(joy.get_button(0)) # A gate
    bB=joy.get_button(1)                     # B drone
    if bB and not prev_buttons[1]:
        drone_on=not drone_on; print(f"Drone → {'ON' if drone_on else 'OFF'}")
    bX=joy.get_button(2)
    if bX and not prev_buttons[2]:
        state["wave_idx"]=(state["wave_idx"]+1)%len(state["waves"])
        graph = build_graph(SAMPLE_RATE, state); print(f"Waveform → {state['waves'][state['wave_idx']]}"); menu_instance.draw()

    # Y can cycle a common diatonic mode set (affects only if your base_token uses 12tet/that mode)
    bY=joy.get_button(3)
    if bY and not prev_buttons[3]:
        # If base_token is 12tet/<mode>, cycle through diatonic modes
        t, m = quantizer.token_to_tuning_mode(state["base_token"])
        if t == "12tet":
            names = list(quantizer.Quantizer.DIATONIC_MODES.keys())
            idx = names.index(m) if (m in names) else -1
            m2 = names[(idx+1)%len(names)] if idx>=0 else names[0]
            state["base_token"] = f"12tet/{m2}"
            print(f"Base token → {state['base_token']}"); menu_instance.draw()
        else:
            # Cycle ET full variants quickly as a convenience
            et = ["12tet/full","19tet/full","31tet/full","53tet/full"]
            try:
                idx = et.index(state["base_token"])
                state["base_token"] = et[(idx+1)%len(et)]
            except ValueError:
                state["base_token"] = et[0]
            print(f"Base token → {state['base_token']}"); menu_instance.draw()

    for i in range(len(prev_buttons)): prev_buttons[i]=joy.get_button(i)

    # Status
    lines=[
        f"Eff:{effective_token:<16} Base:{state['base_token']:<16} Free:{state['free_variant']:<10} Src:{state['source_type']}",
        f"Root:{state['root_midi']:3d}  Freq:{freq_target:7.2f}Hz  Amp:{amp_target:.2f}",
        f"Cut:{cutoff_target:7.1f}Hz  Q:{q_target:4.2f}  Filter:{state['filter_type']}",
        f"Wave:{state['waves'][state['wave_idx']]}  Drone:{'ON' if drone_on else 'OFF'}  LFO:{state['mod_wave_types'][state['mod_wave_idx']]}/{state['mod_rate_hz']:.2f}Hz d={state['mod_depth']:.2f} src={'input' if state['mod_use_input'] else 'free'}"
    ]
    screen=pygame.display.get_surface()
    if screen: draw_status(screen,lines)
    print("\r"+" | ".join(lines), end="")

    clock.tick(120)
