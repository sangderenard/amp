# utils.py
import math
import os

import numpy as np

# =========================
# Settings / fidelity
# =========================
RAW_DTYPE = np.float64
MAX_FRAMES = 4096

# =========================
# Persistence
# =========================
MAPPINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mappings.json")
MAX_UNDO = 10

def as_BF(x, B, F, dtype=RAW_DTYPE):
    """
    Coerce input `x` into shape (B, F):
    - Scalar -> (B, F) filled with the scalar value.
    - (F,) -> (B, F) broadcasted along the batch axis.
    - (B, F) -> Copy of the input.
    """
    if np.isscalar(x):
        return np.full((B, F), float(x), dtype)
    x = np.asarray(x)
    if x.ndim == 1 and x.shape[0] == F:
        return np.broadcast_to(x, (B, F)).astype(dtype, copy=True)
    if x.shape == (B, F):
        return x.astype(dtype, copy=True)
    raise ValueError(f"Cannot broadcast to (B, F): {x.shape}")


# =========================
# Add strict shape helpers (top of file)
# =========================
def as_BCF(x, B, C, F, *, name="tensor"):
    """Coerce x to (B,C,F). Accepts scalar, (F,), (B,F), (C,F), (B,1,F), (1,C,F), (B,C,F)."""
    if np.isscalar(x):
        return np.full((B, C, F), float(x), RAW_DTYPE)

    a = np.asarray(x, dtype=RAW_DTYPE)

    # 1D (F,) -> (1,1,F) -> (B,C,F)
    if a.ndim == 1 and a.shape[0] == F:
        a = a[None, None, :]

    # 2D -> resolve ambiguity:
    elif a.ndim == 2:
        if a.shape[1] == F:
            # Could be (B,F) or (C,F); prefer (B,F) if it matches, else (C,F)
            if a.shape[0] == B:
                a = a[:, None, :]
            else:
                a = a[None, :, :]
        else:
            raise ValueError(f"{name}: 2D must be (*,F); got {a.shape}")

    # 3D ok
    elif a.ndim == 3:
        pass
    else:
        raise ValueError(f"{name}: rank {a.ndim} not in {1,2,3}; shape={a.shape}")

    # Broadcast to (B,C,F)
    if a.shape[0] == 1 and B > 1: a = np.broadcast_to(a, (B, a.shape[1], F)).copy()
    if a.shape[1] == 1 and C > 1: a = np.broadcast_to(a, (a.shape[0], C, F)).copy()

    if a.shape != (B, C, F):
        raise ValueError(f"{name}: got {a.shape}, expected {(B, C, F)}")
    return a

def assert_BCF(x, *, name="tensor"):
    a = np.asarray(x)
    if a.ndim != 3:
        raise ValueError(f"{name}: expected (B,C,F), got rank {a.ndim}, shape={a.shape}")
    return a

def cubic_ramp(y0,y1,n,out=None):
    if out is None: out=np.empty(n,RAW_DTYPE)
    if n<=0: return out
    t=np.linspace(0.0,1.0,n,endpoint=False,dtype=RAW_DTYPE)
    out[:] = y0 + (y1-y0)*(3*t*t - 2*t*t*t)
    return out

_dc_prev=0.0; _prev_in=0.0
def dc_block(sig,a=0.995):
    global _dc_prev,_prev_in
    y=_scratch.tmp[:len(sig)]
    x_prev=_prev_in; y_prev=_dc_prev
    for i,x in enumerate(sig):
        y_prev=a*y_prev + x - x_prev
        x_prev=x; y[i]=y_prev
    _dc_prev,_prev_in=y_prev,x_prev
    return y

def soft_clip(sig,drive=1.2):
    return np.tanh(sig*drive)*(1.0/drive)

def expo_map(norm,lo,hi):
    base=hi/lo
    return lo*(base**norm)

# =========================
# Oscillator (polyBLEP)
# =========================
def _polyblep_arr(t,dt):
    out=np.zeros_like(t)
    m=t<dt
    if np.any(m):
        x=t[m]/np.maximum(dt[m],1e-20)
        out[m]=x+x-x*x-1.0
    m=t>(1.0-dt)
    if np.any(m):
        x=(t[m]-1.0)/np.maximum(dt[m],1e-20)
        out[m]=x*x+x+x+1.0
    return out

def osc_sine(ph): return np.sin(2*np.pi*ph,dtype=RAW_DTYPE)
def osc_saw_blep(ph,dphi):
    t=ph; y=2.0*t-1.0; return y-_polyblep_arr(t,dphi)
def osc_square_blep(ph,dphi,pw=0.5):
    t=ph; y=np.where(t<pw,1.0,-1.0)
    y-=_polyblep_arr(t,dphi)
    t2=(t+(1.0-pw))%1.0
    y+=_polyblep_arr(t2,dphi)
    return y
_tri_state=0.0
def osc_triangle_blep(ph,dphi):
    global _tri_state
    sq=osc_square_blep(ph,dphi)
    leak=0.9995; y=np.empty_like(sq); s=_tri_state
    for i,v in enumerate(sq):
        s=leak*s + (1.0-leak)*v
        y[i]=s
    _tri_state=s
    return y

def make_wave_hq(name,phase,dphi):
    if name=="sine": return osc_sine(phase)
    if name=="saw": return osc_saw_blep(phase,dphi)
    if name=="square": return osc_square_blep(phase,dphi)
    if name=="triangle": return osc_triangle_blep(phase,dphi)
    return np.zeros_like(phase)