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
# Scratch buffers
# =========================


class _ScratchBuffers:
    """Reusable scratch space for intermediate render buffers.

    The interactive application allocates several temporary buffers every time it
    schedules a ramp.  Reusing numpy arrays avoids a flood of short-lived
    allocations during realtime rendering.
    """

    __slots__ = ("capacity", "tmp", "f", "a", "c", "q")

    def __init__(self) -> None:
        self.capacity = 0
        self.tmp = np.empty(0, RAW_DTYPE)
        self.f = np.empty(0, RAW_DTYPE)
        self.a = np.empty(0, RAW_DTYPE)
        self.c = np.empty(0, RAW_DTYPE)
        self.q = np.empty(0, RAW_DTYPE)

    def ensure(self, frames: int) -> None:
        """Guarantee buffers large enough for ``frames`` samples."""

        if frames <= 0:
            return

        if frames <= self.capacity:
            return

        # Round up to the next power of two so reallocation costs are amortised.
        new_capacity = 1 << max(0, frames - 1).bit_length()
        self.tmp = np.empty(new_capacity, RAW_DTYPE)
        self.f = np.empty(new_capacity, RAW_DTYPE)
        self.a = np.empty(new_capacity, RAW_DTYPE)
        self.c = np.empty(new_capacity, RAW_DTYPE)
        self.q = np.empty(new_capacity, RAW_DTYPE)
        self.capacity = new_capacity


_scratch = _ScratchBuffers()

# =========================
# Resampling
# =========================


def lanczos_resample(buffer, src_rate, dst_rate, dst_frames, *, a=8):
    if src_rate <= 0.0 or dst_rate <= 0.0:
        raise ValueError("sample rates must be positive")
    if dst_frames <= 0:
        raise ValueError("dst_frames must be positive")
    array = np.asarray(buffer, dtype=RAW_DTYPE)
    if array.ndim == 0:
        return np.full(dst_frames, float(array), dtype=RAW_DTYPE)
    src_frames = int(array.shape[-1])
    if src_frames == 0:
        shape = array.shape[:-1] + (dst_frames,)
        return np.zeros(shape, dtype=RAW_DTYPE)
    if np.isclose(src_rate, dst_rate) and src_frames == dst_frames:
        return np.array(array, copy=True)

    ratio = float(src_rate) / float(dst_rate)
    positions = np.arange(dst_frames, dtype=RAW_DTYPE) * ratio
    base = np.floor(positions).astype(np.int64)
    frac = positions - base

    taps = int(max(1, a))
    offsets = np.arange(-taps + 1, taps + 1, dtype=np.int64)
    window = offsets.shape[0]

    indices = base[:, None] + offsets[None, :]
    phase = frac[:, None] - offsets[None, :]
    kernel = np.sinc(phase) * np.sinc(phase / taps)
    kernel[np.abs(phase) >= taps] = 0.0

    valid = (indices >= 0) & (indices < src_frames)
    kernel *= valid.astype(RAW_DTYPE)
    sums = kernel.sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    kernel /= sums

    clipped = np.clip(indices, 0, src_frames - 1)
    samples = np.take(array, clipped, axis=-1)
    leading = samples.shape[:-2]
    samples = samples.reshape((-1, dst_frames, window))
    weights = kernel[np.newaxis, :, :]
    resampled = np.sum(samples * weights, axis=2)
    resampled = resampled.reshape(leading + (dst_frames,))
    return np.require(resampled, dtype=RAW_DTYPE, requirements=("C",))

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


def _grid_sorted(grid_cents):
    """Return a sorted tuning grid and a one-octave extension.

    Parameters
    ----------
    grid_cents:
        Iterable of per-octave degree positions expressed in cents.

    Returns
    -------
    (g, g_ext):
        ``g`` is the sorted array of degree positions.  ``g_ext`` appends a
        final element one octave above the first degree so callers can safely
        index ``i+1`` when iterating over segments.
    """

    g = np.asarray(sorted(grid_cents), dtype=RAW_DTYPE)
    if g.size < 2:
        # Safe fallback: chromatic 12TET one-octave grid.
        g = np.arange(12, dtype=RAW_DTYPE) * 100.0
    g_ext = np.concatenate([g, [g[0] + 1200.0]])
    return g, g_ext


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
    sig=np.asarray(sig,dtype=RAW_DTYPE)
    n=sig.shape[0]
    _scratch.ensure(n)
    y=_scratch.tmp[:n]
    if n==0:
        return y

    if a==0.0:
        diffs=_scratch.a[:n]
        diffs[0]=sig[0]-_prev_in
        if n>1:
            np.subtract(sig[1:],sig[:-1],out=diffs[1:])
        y[:]=diffs
        _dc_prev=float(y[-1])
        _prev_in=float(sig[-1])
        return y

    diffs=_scratch.a[:n]
    diffs[0]=sig[0]-_prev_in
    if n>1:
        np.subtract(sig[1:],sig[:-1],out=diffs[1:])

    powers=_scratch.f[:n]
    powers[0]=1.0
    if n>1:
        powers[1:]=a
        np.multiply.accumulate(powers,out=powers)

    accum=_scratch.c[:n]
    np.divide(diffs,powers,out=accum)
    np.add.accumulate(accum,out=accum)
    accum+=a*_dc_prev
    np.multiply(accum,powers,out=y)

    _dc_prev=float(y[-1])
    _prev_in=float(sig[-1])
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
    try:
        from . import c_kernels
        if getattr(c_kernels, "AVAILABLE", False):
            # c_kernels expects contiguous arrays and flattens as needed
            return c_kernels._polyblep_arr_c(t, dt)
    except Exception:
        pass
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
    try:
        from . import c_kernels
        if getattr(c_kernels, "AVAILABLE", False):
            return c_kernels.osc_saw_blep_c(ph, dphi)
    except Exception:
        pass
    t=ph; y=2.0*t-1.0; return y-_polyblep_arr(t,dphi)
def osc_square_blep(ph,dphi,pw=0.5):
    try:
        from . import c_kernels
        if getattr(c_kernels, "AVAILABLE", False):
            return c_kernels.osc_square_blep_c(ph, dphi, float(pw))
    except Exception:
        pass
    t=ph; y=np.where(t<pw,1.0,-1.0)
    y-=_polyblep_arr(t,dphi)
    t2=(t+(1.0-pw))%1.0
    y+=_polyblep_arr(t2,dphi)
    return y
_tri_state = None

def osc_triangle_blep(ph, dphi):
    global _tri_state
    try:
        from . import c_kernels
        if getattr(c_kernels, "AVAILABLE", False):
            if _tri_state is None:
                _tri_state = np.zeros(ph.shape[0], dtype=RAW_DTYPE) if ph.ndim == 2 else np.zeros(1, dtype=RAW_DTYPE)
            out = c_kernels.osc_triangle_blep_c(ph, dphi, _tri_state)
            return out
    except Exception:
        pass

    if _tri_state is None or (isinstance(_tri_state, np.ndarray) and _tri_state.shape[0] != ph.shape[0]):
        # initialize per-batch triangle state
        _tri_state = np.zeros(ph.shape[0], dtype=RAW_DTYPE)
    sq = osc_square_blep(ph, dphi)
    leak = 0.9995
    y = np.empty_like(sq)
    # per-batch state vector
    s = _tri_state.copy() if isinstance(_tri_state, np.ndarray) else np.full(ph.shape[0], float(_tri_state), dtype=RAW_DTYPE)
    # iterate frames and update per-batch state
    B = ph.shape[0]
    F = ph.shape[1]
    for i in range(F):
        v = sq[:, i]
        s = leak * s + (1.0 - leak) * v
        y[:, i] = s
    _tri_state = s
    return y

def make_wave_hq(name,phase,dphi):
    if name=="sine": return osc_sine(phase)
    if name=="saw": return osc_saw_blep(phase,dphi)
    if name=="square": return osc_square_blep(phase,dphi)
    if name=="triangle": return osc_triangle_blep(phase,dphi)
    return np.zeros_like(phase)