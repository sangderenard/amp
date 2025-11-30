"""Small utilities to map FFT magnitudes to display-friendly formats.

This re-implements the magnitude->dB and magnitude->u8 mapping used by
the fftfree `fft_to_png.py` helper, but with a compact, testable API.
"""
from typing import Literal, Optional
import warnings
import numpy as np


def mag_to_db(mag_mat: np.ndarray, db_ref: Literal["global", "frame"] = "global", db_floor: float = -80.0, eps: float = 1e-12) -> np.ndarray:
    """Convert linear magnitude matrix to dB values.

    - mag_mat: 2D array shaped (bins, frames) containing non-negative magnitudes.
    - db_ref: 'global' to normalize by global max, 'frame' to normalize per-frame.
    - db_floor: floor in dB (negative number) to clamp lower values.
    - eps: small value to avoid log(0).

    Returns array of same shape (bins, frames) in dB clipped to [db_floor, 0.0].
    """
    m = np.array(mag_mat, dtype=np.float64, copy=True)
    if m.size == 0:
        return m
    # Choose reference
    if db_ref == "frame":
        # per-frame max
        ref = np.max(m, axis=0, keepdims=True) + eps
    else:
        ref = float(np.max(m)) + eps
    # Normalize and clamp
    if db_ref == "frame":
        m_norm = m / ref
    else:
        m_norm = m / ref
    m_norm = np.clip(m_norm, eps, 1.0)
    db = 20.0 * np.log10(m_norm)
    db = np.clip(db, db_floor, 0.0)
    return db


def mag_to_u8(mag_mat: np.ndarray, scale: Literal["db", "log1p", "linear"] = "log1p", db_ref: Literal["global", "frame"] = "global", db_floor: float = -80.0, eps: float = 1e-12) -> np.ndarray:
    """Convert magnitude matrix to uint8 image data (0..255).

    - mag_mat: 2D array shaped (bins, frames).
    - scale: 'db'|'log1p'|'linear' matching fftfree CLI options.
    - db_ref/db_floor/eps behave as in `mag_to_db`.

    Returns a 2D uint8 array shaped (bins, frames).
    """
    m = np.array(mag_mat, dtype=np.float64, copy=True)
    if m.size == 0:
        warnings.warn("fft_viz.mag_to_u8: received empty magnitude array", UserWarning)
        return m.astype(np.uint8)

    if scale == "linear":
        a = m.copy()
        min_val = float(np.min(a))
        a -= min_val
        max_val = float(np.max(a))
        if max_val == 0.0:
            warnings.warn("fft_viz.mag_to_u8: linear scale input has zero dynamic range", UserWarning)
            return np.full_like(a, 128, dtype=np.uint8)
        a /= (max_val + 1e-8)
        a = np.clip(a * 255.0, 0.0, 255.0)
        return a.astype(np.uint8)

    if scale == "db":
        # Compute normalized dB per-frame or global
        db = mag_to_db(m, db_ref=db_ref, db_floor=db_floor, eps=eps)
        # Detect constant-db (no dynamic range)
        if db.size == 0:
            warnings.warn("fft_viz.mag_to_u8: db mapping produced empty array", UserWarning)
            return db.astype(np.uint8)
        if np.nanmax(db) - np.nanmin(db) < 1e-12:
            warnings.warn("fft_viz.mag_to_u8: db mapping has no dynamic range (constant), returning mid-gray image", UserWarning)
            return np.full_like(db, 128, dtype=np.uint8)
        # Map db in [db_floor, 0] -> [0,255]
        norm = (db - db_floor) / (0.0 - db_floor + 1e-12)
        u8 = np.uint8(np.clip(norm * 255.0, 0.0, 255.0))
        return u8

    # log1p mapping heuristic similar to fftfree: choose k from median/ref
    ref = float(np.max(m)) if np.max(m) > 0 else 1.0
    med = float(np.median(m)) if m.size else 0.0
    if med > 0:
        k = 1.0 / med
    else:
        k = 1.0 / ref
    num = np.log1p(k * m)
    den = np.log1p(k * ref) + 1e-12
    norm = np.clip(num / den, 0.0, 1.0)
    return np.uint8(np.clip(norm * 255.0, 0.0, 255.0))


def autoscale_image(arr: np.ndarray, method: Literal['percentile', 'hist_eq', 'none']='percentile', low: float=1.0, high: float=99.0, gamma: float=1.0) -> np.ndarray:
    """Autoscale a 2D image-like array to uint8 using a simple high-quality method.

    - arr: 2D array, can be float or uint8. Interpreted as intensity; higher is brighter.
    - method: 'percentile' (default) does percentile contrast stretch; 'hist_eq' does histogram equalization; 'none' returns casted array.
    - low/high: percentiles for clipping when method='percentile'. Values in [0,100].
    - gamma: apply gamma correction after scaling (gamma=1.0 = no change).

    Returns uint8 array with same shape.
    """
    if arr is None:
        return np.zeros((0,0), dtype=np.uint8)
    a = np.asarray(arr)
    if a.size == 0:
        warnings.warn("fft_viz.autoscale_image: received empty array", UserWarning)
        return a.astype(np.uint8)

    # Convert to float in [0,1]
    if a.dtype == np.uint8:
        f = a.astype(np.float64) / 255.0
    else:
        # scale by max abs if floats; if all <=1 assume already in [0,1]
        if np.nanmax(a) > 1.0 or np.nanmin(a) < 0.0:
            # bring to positive range
            amin = float(np.nanmin(a))
            a_shift = a - amin
            amax = float(np.nanmax(a_shift))
            if amax <= 0:
                f = np.clip(a_shift, 0.0, 1.0)
            else:
                f = a_shift / amax
        else:
            f = np.clip(a.astype(np.float64), 0.0, 1.0)

    if method == 'none':
        out = np.clip(np.power(f, 1.0/gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
        return out

    if method == 'percentile':
        # compute percentiles robustly ignoring NaN
        lo = np.nanpercentile(f, low)
        hi = np.nanpercentile(f, high)
        if hi <= lo or hi - lo < 1e-12:
            # fallback to simple scale by max
            lo = 0.0
            hi = np.nanmax(f)
            if hi <= 0:
                warnings.warn("fft_viz.autoscale_image: data has no dynamic range; returning mid-gray image", UserWarning)
                return np.full_like(f, 128, dtype=np.uint8)
        # stretch
        s = (f - lo) / (hi - lo)
        s = np.clip(s, 0.0, 1.0)
        if gamma != 1.0 and gamma > 0:
            s = np.power(s, 1.0/gamma)
        return (np.clip(s * 255.0, 0.0, 255.0)).astype(np.uint8)

    if method == 'hist_eq':
        # histogram equalization on flattened data
        flat = (f.flatten())
        # mask NaNs
        mask = ~np.isnan(flat)
        vals = flat[mask]
        if vals.size == 0:
            return np.zeros_like(f, dtype=np.uint8)
        # compute histogram
        hist, bins = np.histogram(vals, bins=256, range=(0.0, 1.0), density=False)
        cdf = np.cumsum(hist).astype(np.float64)
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
        # map vals via cdf
        # digitize assigns bins; use interpolation
        inds = np.searchsorted(bins[:-1], vals, side='right') - 1
        inds = np.clip(inds, 0, 255)
        mapped = cdf[inds]
        out = np.zeros_like(flat, dtype=np.uint8)
        out[mask] = np.clip(mapped * 255.0, 0.0, 255.0).astype(np.uint8)
        return out.reshape(f.shape)

    # default: return clipped cast
    return np.clip(f * 255.0, 0.0, 255.0).astype(np.uint8)
