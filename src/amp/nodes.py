# nodes.py
import math
import numpy as np

from .utils import (
    as_BCF,
    assert_BCF,
    dc_block,
    make_wave_hq,
    osc_saw_blep,
    osc_square_blep,
    osc_triangle_blep,
    soft_clip,
)
from .state import RAW_DTYPE, MAX_FRAMES


def _ensure_bcf(audio_in, frames: int, *, name: str):
    if audio_in is None:
        return None, 1
    array = assert_BCF(audio_in, name=name)
    if array.shape[2] != frames:
        raise ValueError(f"{name}: expected {frames} frames, got {array.shape[2]}")
    return array, array.shape[0]


def _match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if data.shape[1] > channels:
        return data[:, :channels, :]
    pad = np.zeros((data.shape[0], channels - data.shape[1], data.shape[2]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=1)



# =========================
# Graph nodes
# =========================
class Node:
    def __init__(self,name): self.name=name
    def process(self,frames,sr,audio_in,mods,params): raise NotImplementedError


class ConfigNode(Node):
    def __init__(self, name, params=None):
        super().__init__(name)
        self.params = dict(params or {})


class SilenceNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))

    def process(self, frames, sr, audio_in, mods, params):
        _, batches = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        return np.zeros((batches, self.channels, frames), dtype=RAW_DTYPE)


class ConstantNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))
        self.value = float(self.params.get("value", 0.0))

    def process(self, frames, sr, audio_in, mods, params):
        _, batches = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        buffer = np.full((batches, self.channels, frames), self.value, dtype=RAW_DTYPE)
        return buffer


class SineOscillatorNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 1))
        self.frequency = float(self.params.get("frequency", 440.0))
        self.amplitude = float(self.params.get("amplitude", 0.5))
        phase = float(self.params.get("phase", 0.0)) % 1.0
        self._phase = np.array([[phase]], dtype=RAW_DTYPE)

    def process(self, frames, sr, audio_in, mods, params):
        _, batches = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        channels = self.channels
        freq = as_BCF(
            params.get("frequency", self.frequency),
            batches,
            channels,
            frames,
            name=f"{self.name}.frequency",
        )
        amp = as_BCF(
            params.get("amplitude", self.amplitude),
            batches,
            channels,
            frames,
            name=f"{self.name}.amplitude",
        )
        if self._phase.shape != (batches, channels):
            self._phase = np.full((batches, channels), self._phase[0, 0], dtype=RAW_DTYPE)
        dphi = freq / float(sr)
        phase = (self._phase[..., None] + np.cumsum(dphi, axis=2)) % 1.0
        self._phase = phase[..., -1]
        wave = np.sin(2.0 * np.pi * phase, dtype=RAW_DTYPE)
        return wave * amp


class SafetyNode(ConfigNode):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        self.channels = int(self.params.get("channels", 2))
        self.dc_alpha = float(self.params.get("dc_alpha", 0.995))
        self._state: dict[int, np.ndarray] = {}

    def process(self, frames, sr, audio_in, mods, params):
        audio, batches = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        if audio is None:
            data = np.zeros((batches, self.channels, frames), dtype=RAW_DTYPE)
        else:
            data = _match_channels(audio, self.channels)
        out = np.empty_like(data)
        for batch in range(batches):
            state = self._state.get(batch)
            if state is None or state.shape[0] != self.channels:
                state = np.zeros(self.channels, dtype=RAW_DTYPE)
            for ch in range(self.channels):
                dc = state[ch]
                for i in range(frames):
                    dc = self.dc_alpha * dc + (1.0 - self.dc_alpha) * data[batch, ch, i]
                    out[batch, ch, i] = data[batch, ch, i] - dc
                state[ch] = dc
            self._state[batch] = state
        np.clip(out, -1.0, 1.0, out=out)
        return out

class DelayNode(Node):
    def __init__(self,name,delay_samples=64):
        super().__init__(name)
        self.delay = delay_samples
        self.buf = np.zeros((1, 1, delay_samples), RAW_DTYPE)  # (B,C,D)
        self.w   = np.zeros((1, 1), dtype=int)                 # (B,C)

    def _ensure(self, B, C):
        if self.buf.shape[:2] != (B, C):
            self.buf = np.zeros((B, C, self.delay), RAW_DTYPE)
            self.w   = np.zeros((B, C), dtype=int)

    def process(self, frames, sr, audio_in, mods, params):
        x = np.zeros((1,1,frames), RAW_DTYPE) if audio_in is None else assert_BCF(audio_in, name="delay.in")
        B, C, F = x.shape
        self._ensure(B, C)

        out = np.empty_like(x)
        idxs = (self.w[..., None] + np.arange(F)[None, None, :]) % self.delay  # (B,C,F)
        out[:] = self.buf.take(idxs, axis=2, mode='wrap')
        self.buf[np.arange(B)[:,None,None], np.arange(C)[None,:,None], idxs] = x
        self.w = (self.w + F) % self.delay
        return out

class LFONode(Node):
    def __init__(self,name,wave="sine",rate_hz=4.0,depth=0.5,use_input=False,slew_ms=0.0):
        super().__init__(name)
        self.wave=wave; self.rate=rate_hz; self.depth=depth
        self.use_input=use_input; self.slew_ms=slew_ms
        self.phase=0.0
    def _make(self,ph):
        if self.wave=="sine": return np.sin(2*np.pi*ph,dtype=RAW_DTYPE)
        if self.wave=="square": return np.where((ph%1.0)<0.5,1.0,-1.0).astype(RAW_DTYPE)
        if self.wave=="saw": return (2.0*((ph%1.0)) - 1.0).astype(RAW_DTYPE)
        if self.wave=="triangle": return (2.0*np.abs(2.0*(ph%1.0)-1.0)-1.0).astype(RAW_DTYPE)
        return np.zeros_like(ph)
    def process(self,frames,sr,audio_in,mods,params):
        B = audio_in.shape[0] if self.use_input and audio_in is not None else 1
        C = 1; F = frames
        if self.use_input and audio_in is not None:
            x = assert_BCF(audio_in, name="lfo.in")
            m = np.maximum(1e-12, np.max(np.abs(x), axis=(1,2)))  # (B,)
            out = (x[:, :1, :] / m[:, None, None]) * float(self.depth)
        else:
            t = (self.phase + np.arange(F)*(self.rate/sr)) % 1.0
            self.phase = float(t[-1])
            wave = self._make(t)  # (F,)
            out = np.tile(wave, (B,1,1))  # (B,1,F)
            out *= float(self.depth)
        if self.slew_ms > 0:
            alpha = 1.0 - math.exp(-1.0/(sr*(self.slew_ms/1000.0)))
            z = np.zeros((B,1), RAW_DTYPE)
            for i in range(F):
                z = z + alpha * (out[:,:,i:i+1] - z)
                out[:,:,i] = z[:,0]
        return out  # (B,1,F)

class OscNode(Node):
    def __init__(self, name, wave="sine"):
        super().__init__(name); self.wave = wave; self.phase = None

    def process(self, frames, sr, audio_in, mods, params):
        B = audio_in.shape[0] if audio_in is not None else 1
        C = 1
        F = frames

        f = as_BCF(params.get("freq", 0.0), B, C, F, name="osc.freq")[:,0,:]  # (B,F)
        a = as_BCF(params.get("amp",  1.0), B, C, F, name="osc.amp") [:,0,:]  # (B,F)

        dphi = f / float(sr)
        if self.phase is None or self.phase.shape[0] != B:
            self.phase = np.zeros(B, RAW_DTYPE)
        ph = (self.phase[:, None] + np.cumsum(dphi, axis=1)) % 1.0
        self.phase = ph[:, -1]

        if self.wave == "sine":
            w = np.sin(2*np.pi*ph, dtype=RAW_DTYPE)
        elif self.wave == "saw":
            w = osc_saw_blep(ph, dphi)
        elif self.wave == "square":
            w = osc_square_blep(ph, dphi)
        elif self.wave == "triangle":
            w = osc_triangle_blep(ph, dphi)
        else:
            w = np.zeros_like(ph)

        return (w * a)[:, None, :]  # (B,1,F)

class SamplerNode(Node):
    def __init__(self, name, sampler):
        super().__init__(name)
        self.sampler = sampler
    def process(self,frames,sr,audio_in,mods,params):
        _, B = _ensure_bcf(audio_in, frames, name=f"{self.name}.in")
        out = np.empty((B, 1, frames), RAW_DTYPE)
        rate = params.get("rate", 1.0)
        tr = params.get("transpose", 0.0)
        gain = params.get("gain", 1.0)
        # Vectorized: if params are arrays, use them per channel, else broadcast
        for b in range(B):
            self.sampler.render_into(
                out[b, 0],
                sr,
                rate[b] if isinstance(rate, np.ndarray) and rate.shape[0] == B else rate,
                tr[b] if isinstance(tr, np.ndarray) and tr.shape[0] == B else tr,
                gain[b] if isinstance(gain, np.ndarray) and gain.shape[0] == B else gain
            )
        return out

class MixNode(Node):
    """
    Mixes N input bundles (num_inputs, channels, frames) down to (channels, frames).
    Applies per-channel gain, expands mono to stereo if needed, and supports ALC and compression.
    """

    def __init__(self, name, params=None, *, out_channels=2, alc=True, compression="tanh"):
        super().__init__(name)
        cfg = dict(params or {}) if params is not None else {}
        if params is not None and not isinstance(params, dict):
            raise TypeError("MixNode params must be a mapping")
        self.out_channels = int(cfg.get("channels", out_channels))
        self.alc = bool(cfg.get("alc", alc))
        self.compression = cfg.get("compression", compression)
        self.stats = ClipStats()
        # For ALC
        self.rms_hist = [np.zeros(self.out_channels, dtype=RAW_DTYPE) for _ in range(256)]
        self.peak_hist = [np.zeros(self.out_channels, dtype=RAW_DTYPE) for _ in range(256)]
        self.alpha = 0.5
        self.attack = 16
        self.sustain = 128
        self.decay = 256

    def process(self, frames, sr, audio_in, mods, params):
        if audio_in is None:
            return np.zeros((1, self.out_channels, frames), dtype=RAW_DTYPE)
        x = assert_BCF(audio_in, name="mix.in")  # (B,C,F)
        B, C, F = x.shape
        y = np.sum(x, axis=1, keepdims=True)     # (B,1,F)
        if self.out_channels > 1:
            y = np.repeat(y, self.out_channels, axis=1)  # (B,outC,F)

        # (optional) ALC/compression can operate per-channel over y[:,c,:] here

        self.stats.update(y.reshape(-1, F))
        return y

class BiquadNode(Node):
    def __init__(self,name,fs,ftype="lowpass"):
        super().__init__(name); self.ftype=ftype; self.filt=FilterLPBiquad(fs)
        self.peaking_db=6.0
    def process(self,frames,sr,audio_in,mods,params):
        if audio_in is None: return np.zeros((1,frames),RAW_DTYPE)
        cutoff=params.get("cutoff", 1000.0)  # Default cutoff to 1000.0 Hz if not provided
        Q=params.get("Q", 0.707)  # Default Q to 0.707 if not provided
        # cutoff, Q: (B,F)
        B = audio_in.shape[0]
        self.filt.ensure_B(B)
        # Use last frame for filter update (per batch)
        self.filt.update(float(cutoff[-1,-1]), float(Q[-1,-1]), self.ftype, self.peaking_db)
        return self.filt.process(audio_in)

class GainNode(Node):
    def __init__(self,name,gain=1.0): super().__init__(name); self.gain=gain
    def process(self,frames,sr,audio_in,mods,params):
        if audio_in is None: return np.zeros((1,frames),RAW_DTYPE)
        g=params.get("gain",None)
        if g is None: return audio_in*self.gain
        return audio_in*g

class SourceSwitch(Node):
    def __init__(self,name,osc_node:OscNode,samp_node:SamplerNode|None, state):
        super().__init__(name); self.osc=osc_node; self.samp=samp_node; self.state=state
    def process(self,frames,sr,audio_in,mods,params):
        B = audio_in.shape[0] if audio_in is not None else 1
        if self.state["source_type"]=="sampler" and self.samp is not None:
            return self.samp.process(frames,sr,audio_in,mods,{"rate":1.0,"transpose":0.0,"gain":1.0})
        return self.osc.process(frames,sr,audio_in,mods,params)

class ClipStats:
    def __init__(self):
        self.last_max = 0.0
        self.last_min = 0.0
        self.last_clipped = False
    def update(self, arr):
        self.last_max = float(np.max(arr))
        self.last_min = float(np.min(arr))
        self.last_clipped = np.any(np.abs(arr) > 1.0)

class SafetyFilterNode(Node):
    def __init__(self, name, sr, n_ch=2):
        super().__init__(name)
        self.a = 0.995
        self.prev_in = np.zeros((1, n_ch), RAW_DTYPE)  # (B,C)
        self.prev_dc = np.zeros((1, n_ch), RAW_DTYPE)

    def _ensure(self, B, C):
        if self.prev_in.shape != (B, C):
            self.prev_in = np.zeros((B, C), RAW_DTYPE)
            self.prev_dc = np.zeros((B, C), RAW_DTYPE)

    def process(self, frames, sr, audio_in, mods, params):
        x = assert_BCF(audio_in, name="safety.in")  # (B,C,F)
        B, C, F = x.shape
        self._ensure(B,C)
        y = np.empty_like(x)
        pi, pd = self.prev_in, self.prev_dc
        for i in range(F):
            pd = self.a * pd + x[:,:,i] - pi
            pi = x[:,:,i]
            y[:,:,i] = pd
        self.prev_in, self.prev_dc = pi, pd
        return y

class NormalizerCompressorNode(Node):
    def __init__(self, name, n_ch=2, alpha=0.5, attack=16, sustain=128, decay=256):
        super().__init__(name)
        self.n_ch = n_ch
        self.alpha = alpha
        self.attack = attack
        self.sustain = sustain
        self.decay = decay
        self.rms_hist = [np.zeros(n_ch, dtype=RAW_DTYPE) for _ in range(decay)]
        self.peak_hist = [np.zeros(n_ch, dtype=RAW_DTYPE) for _ in range(decay)]
        self.stats = ClipStats()
    def process(self, frames, sr, audio_in, mods, params):
        # Compute RMS and peak over moving window (attack/sustain/decay)
        rms = np.sqrt(np.mean(audio_in**2, axis=1))
        peak = np.max(np.abs(audio_in), axis=1)
        self.rms_hist.pop(0)
        self.rms_hist.append(rms)
        self.peak_hist.pop(0)
        self.peak_hist.append(peak)
        # Weighted window: attack, sustain, decay
        def weighted_avg(hist, attack, sustain, decay):
            hist_arr = np.stack(hist)
            weights = np.concatenate([
                np.full(attack, 1.0),
                np.full(sustain, 0.5),
                np.full(decay, 0.25)
            ])
            weights = weights[:hist_arr.shape[0]]
            weights = weights / np.sum(weights)
            return np.sum(hist_arr * weights[:, None], axis=0)
        rms_val = weighted_avg(self.rms_hist, self.attack, self.sustain, self.decay)
        peak_val = weighted_avg(self.peak_hist, self.attack, self.sustain, self.decay)
        norm_val = self.alpha * rms_val + (1.0 - self.alpha) * peak_val + 1e-8
        # Normalize
        normed = audio_in / norm_val[:, None]
        # Tanh soft-knee compression
        compressed = np.tanh(normed)
        self.stats.update(compressed)
        return compressed

class SubharmonicLowLifterNode(Node):
    """Low-lifter style subharmonic enhancer operating purely on audio input."""

    def __init__(
        self,
        name,
        sample_rate,
        *,
        band_lo=70.0,
        band_hi=160.0,
        mix=0.5,
        drive=1.0,
        out_hp=25.0,
        use_div4=False,
    ):
        super().__init__(name)
        self.sr = float(sample_rate)
        self.band_lo = float(band_lo)
        self.band_hi = float(band_hi)
        self.mix = float(mix)
        self.drive = float(drive)
        self.out_hp = float(out_hp)
        self.use_div4 = bool(use_div4)
        self._init = False

    # --- coefficient helpers -------------------------------------------------
    def _alpha_lp(self, fc, sr):
        return 1.0 - math.exp(-2.0 * math.pi * max(fc, 1.0) / sr)

    def _alpha_hp(self, fc, sr):
        fc = max(fc, 1.0)
        rc = 1.0 / (2.0 * math.pi * fc)
        return rc / (rc + 1.0 / sr)

    def _ensure_state(self, B, C):
        if self._init and self.prev.shape == (B, C):
            return
        self.hp_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.lp_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.prev = np.zeros((B, C), dtype=RAW_DTYPE)
        self.sign = np.zeros((B, C), dtype=np.int8)
        self.ff2 = np.ones((B, C), dtype=np.int8)
        self.sub2_lp = np.zeros((B, C), dtype=RAW_DTYPE)
        self.env = np.zeros((B, C), dtype=RAW_DTYPE)
        self.hp_out_y = np.zeros((B, C), dtype=RAW_DTYPE)
        self.hp_out_x = np.zeros((B, C), dtype=RAW_DTYPE)
        if self.use_div4:
            self.ff4 = np.ones((B, C), dtype=np.int8)
            self.ff4_count = np.zeros((B, C), dtype=np.int32)
            self.sub4_lp = np.zeros((B, C), dtype=RAW_DTYPE)
        else:
            self.ff4 = None
            self.ff4_count = None
            self.sub4_lp = None
        self._init = True

    @staticmethod
    def _scalar_param(params, key, default):
        value = params.get(key)
        if value is None:
            return float(default)
        arr = np.asarray(value, dtype=RAW_DTYPE)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.reshape(-1)[-1])

    def process(self, frames, sr, audio_in, mods, params):
        if audio_in is None:
            return np.zeros((1, 1, frames), dtype=RAW_DTYPE)

        x = assert_BCF(audio_in, name=f"{self.name}.in")
        B, C, F = x.shape

        self._ensure_state(B, C)

        # Allow slow parameter modulation by sampling the last provided value.
        sr = float(sr or self.sr)
        band_lo = self._scalar_param(params, "band_lo", self.band_lo)
        band_hi = self._scalar_param(params, "band_hi", self.band_hi)
        mix = self._scalar_param(params, "mix", self.mix)
        drive = self._scalar_param(params, "drive", self.drive)
        out_hp = self._scalar_param(params, "out_hp", self.out_hp)

        a_hp_in = self._alpha_hp(band_lo, sr)
        a_lp_in = self._alpha_lp(band_hi, sr)
        a_sub2 = self._alpha_lp(max(band_hi / 3.0, 30.0), sr)
        a_sub4 = self._alpha_lp(max(band_hi / 5.0, 20.0), sr) if self.use_div4 else None
        a_env_attack = self._alpha_lp(100.0, sr)
        a_env_release = self._alpha_lp(5.0, sr)
        a_hp_out = self._alpha_hp(out_hp, sr)

        y = np.empty_like(x)

        for t in range(F):
            xt = x[:, :, t]

            # Bandpass driver: simple HP then LP
            self.hp_y = a_hp_in * (self.hp_y + xt - self.prev)
            self.prev = xt
            bp = self.lp_y + a_lp_in * (self.hp_y - self.lp_y)
            self.lp_y = bp

            abs_bp = np.abs(bp)
            self.env = np.where(
                abs_bp > self.env,
                self.env + a_env_attack * (abs_bp - self.env),
                self.env + a_env_release * (abs_bp - self.env),
            )

            prev_sign = self.sign
            sign_now = (bp > 0.0).astype(np.int8) * 2 - 1
            pos_zc = (prev_sign < 0) & (sign_now > 0)
            self.sign = sign_now

            self.ff2 = np.where(pos_zc, -self.ff2, self.ff2)

            if self.use_div4:
                self.ff4_count = np.where(pos_zc, self.ff4_count + 1, self.ff4_count)
                toggle4 = pos_zc & (self.ff4_count >= 2)
                self.ff4 = np.where(toggle4, -self.ff4, self.ff4)
                self.ff4_count = np.where(toggle4, 0, self.ff4_count)

            sq2 = self.ff2.astype(RAW_DTYPE)
            self.sub2_lp = self.sub2_lp + a_sub2 * (sq2 - self.sub2_lp)
            sub = self.sub2_lp

            if self.use_div4 and self.sub4_lp is not None and self.ff4 is not None:
                sq4 = self.ff4.astype(RAW_DTYPE)
                self.sub4_lp = self.sub4_lp + a_sub4 * (sq4 - self.sub4_lp)
                sub = sub + 0.6 * self.sub4_lp

            sub = np.tanh(drive * sub) * (self.env + 1e-6)

            dry = xt
            wet = sub
            out_t = (1.0 - mix) * dry + mix * wet

            y_prev = self.hp_out_y
            x_prev = self.hp_out_x
            hp = a_hp_out * (y_prev + out_t - x_prev)
            self.hp_out_y = hp
            self.hp_out_x = out_t
            y[:, :, t] = hp

        return y


NODE_TYPES = {
    "silence": SilenceNode,
    "constant": ConstantNode,
    "sine": SineOscillatorNode,
    "sine_oscillator": SineOscillatorNode,
    "mix": MixNode,
    "safety": SafetyNode,
    "subharmonic_low_lifter": SubharmonicLowLifterNode,
}


__all__ = [
    "NODE_TYPES",
    "Node",
    "SilenceNode",
    "ConstantNode",
    "SineOscillatorNode",
    "MixNode",
    "SafetyNode",
    "SubharmonicLowLifterNode",
]
