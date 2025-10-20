import os

# =========================
# Settings / fidelity
# =========================
RAW_DTYPE = 'float64'
MAX_FRAMES = 4096

# =========================
# Persistence
# =========================
MAPPINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mappings.json")
MAX_UNDO = 10

# =========================
# Tuning/Mode token helpers
# =========================
FREE_VARIANTS = ("continuous", "weighted", "stepped")

# Default state
waves = ["sine", "square", "saw", "triangle"]

state = {
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