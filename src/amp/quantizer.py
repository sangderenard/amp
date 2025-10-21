# quantizer.py
import numpy as np
import math

from .utils import RAW_DTYPE, _grid_sorted

# =========================
# Quantizer dictionaries
# =========================
class Quantizer:
    # Diatonic degrees in 12-TET (values are semitone counts within octave)
    DIATONIC_MODES = {
        "ionian":[0,2,4,5,7,9,11],
        "dorian":[0,2,3,5,7,9,10],
        "phrygian":[0,1,3,5,7,8,10],
        "lydian":[0,2,4,6,7,9,11],
        "mixolydian":[0,2,4,5,7,9,10],
        "aeolian":[0,2,3,5,7,8,10],
        "locrian":[0,1,3,5,6,8,10],
    }
    # Cents-based scale presets (per-octave degrees in cents)
    TUNING_MODES = {
        # “Equal temperaments”
        "12tet_full": [i*100.0 for i in range(12)],
        "19tet_full": [i*(1200.0/19.0) for i in range(19)],
        "31tet_full": [i*(1200.0/31.0) for i in range(31)],
        "53tet_full": [i*(1200.0/53.0) for i in range(53)],
        # Microtonal / modal approximations (examples)
        "raga_bhairav_approx": [0, 90, 400, 500, 700, 810, 1100],
        "raga_todi_approx":    [0, 90, 290, 600, 700, 790, 1090],
        "raga_yaman_approx":   [0, 200, 390, 600, 700, 900, 1080],
        "just_major": [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27],
        "just_minor": [0, 203.91, 315.64, 498.04, 701.96, 792.69, 1017.60],
        "pythagorean": [0, 203.91, 407.82, 498.04, 701.96, 905.87, 1109.78],
    }

    @staticmethod
    def midi_to_freq(m): return 440.0 * (2.0**((m-69.0)/12.0))

# =========================
# Tuning/Mode token helpers
# =========================
FREE_VARIANTS = ("continuous","weighted","stepped")

def is_free_mode_token(tok: str) -> bool:
    return tok == "FREE"

# ----- Equal-distance grid warping (per-degree spacing uniform) -----
def grid_warp_forward(cents, grid_cents):
    """Map cents -> fractional degree units."""

    g, g_ext = _grid_sorted(grid_cents)
    N = int(g.size)
    cents_arr = np.asarray(cents, dtype=RAW_DTYPE)
    scalar = np.isscalar(cents) or cents_arr.ndim == 0

    c_mod = np.mod(cents_arr, 1200.0)
    octs = np.floor(cents_arr / 1200.0)

    idx = np.searchsorted(g_ext, c_mod, side="right") - 1
    idx = np.clip(idx, 0, N - 1)
    lower = g_ext[idx]
    upper = g_ext[idx + 1]
    denom = np.maximum(upper - lower, 1e-9)
    t = (c_mod - lower) / denom
    u_mod = idx + t
    out = octs * N + u_mod

    if scalar:
        return float(np.asarray(out, dtype=RAW_DTYPE).reshape(-1)[0])
    return out.astype(RAW_DTYPE, copy=False)

def grid_warp_inverse(u, grid_cents):
    """Map degree units -> cents."""

    g, g_ext = _grid_sorted(grid_cents)
    N = int(g.size)
    u_arr = np.asarray(u, dtype=RAW_DTYPE)
    scalar = np.isscalar(u) or u_arr.ndim == 0

    octs = np.floor(u_arr / N)
    u_mod = u_arr - octs * N
    idx = np.floor(u_mod).astype(np.intp)
    idx = np.clip(idx, 0, N - 1)
    frac = u_mod - idx
    lower = g_ext[idx]
    upper = g_ext[idx + 1]
    c_mod = lower + frac * (upper - lower)
    out = octs * 1200.0 + c_mod

    if scalar:
        return float(np.asarray(out, dtype=RAW_DTYPE).reshape(-1)[0])
    return out.astype(RAW_DTYPE, copy=False)

def token_to_tuning_mode(token: str):
    """Return (tuning, mode) or ('FREE', None).
       Examples:
         '12tet/full' -> ('12tet','full')
         '12tet/ionian' -> ('12tet','ionian')
         'cents/raga_bhairav_approx' -> ('cents','raga_bhairav_approx')
         'FREE' -> ('FREE', None)
    """
    if token == "FREE":
        return "FREE", None
    if "/" not in token:
        return token, "full"
    t, m = token.split("/", 1)
    return t, m

def grid_for_token(token: str):
    """Return per-octave degrees in cents for the given token."""
    t, m = token_to_tuning_mode(token)
    if t == "FREE":
        return []  # not used directly
    # Equal temperaments
    if t.endswith("tet"):
        try:
            N = int(t[:-3])
            step = 1200.0 / float(N)
        except Exception:
            N = 12; step = 100.0
        if m == "full" or m is None:
            return [i*step for i in range(N)]
        # 12-TET named diatonic modes
        if t == "12tet" and m in Quantizer.DIATONIC_MODES:
            return [d*100.0 for d in Quantizer.DIATONIC_MODES[m]]
        # fallback to full set for other ETs
        return [i*step for i in range(N)]
    # Cents-based banks
    if t == "cents":
        if m in Quantizer.TUNING_MODES:
            return Quantizer.TUNING_MODES[m][:]
        # fallback: 12tet full
        return [i*100.0 for i in range(12)]
    # Unknown -> default 12tet full
    return [i*100.0 for i in range(12)]

def get_reference_grid_cents(state, effective_token: str):
    """
    When you're *in* FREE (any variant), we still need a reference grid to define steps
    for WEIGHTED/STEPPED. Use the user's current non-free selection as the reference.
    """
    if is_free_mode_token(effective_token):
        base_tok = state.get("base_token", "12tet/full")
        if is_free_mode_token(base_tok):
            return [i*100.0 for i in range(12)]
        return grid_for_token(base_tok)
    else:
        return grid_for_token(effective_token)

def quantize_to_grid_cents(cents_target, grid_cents):
    best = None; best_err = 1e12
    # Search few octaves around target
    for k in range(-4,5):
        base = 1200.0*k
        for d in grid_cents:
            c = base + d
            err = abs(c - cents_target)
            if err < best_err:
                best, best_err = c, err
    return best if best is not None else cents_target

def weighted_blend_to_grid(cents_unq, grid_cents, sigma=45.0):
    c_near = quantize_to_grid_cents(cents_unq, grid_cents)
    d = abs(cents_unq - c_near)
    w = math.exp(- (d / max(sigma, 1e-9))**2)
    return (1.0 - w)*cents_unq + w*c_near