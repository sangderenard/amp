#!/usr/bin/env python3
"""
Simple GUI to run/visualize FFT division node (simulate or run external binary).

Usage:
    - Requires: Python 3.8+, numpy, Pillow (PIL)
  - Optional: a compiled `test_fft_division_node` executable; you can point the GUI to it.

This GUI provides parameter fields (window size, hop, frames), a Run button
and two image panes: waveform and spectral magnitude (horizontal orientation). Uses Pillow for PNG previews.
"""
import os
import sys
import threading
import subprocess
import shlex
import time
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    print("Tkinter is required but not available.")
    raise

import numpy as np
# When this script is executed from a different CWD (for example from build/Debug),
# Python's import machinery may not find the `tools` package. Ensure the project
# root (parent of this `tools` package) is on `sys.path` so `import tools` works.
from pathlib import Path as _Path
_proj_root = str(_Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from tools.fft_viz import mag_to_db, mag_to_u8, autoscale_image
# PIL for fast PNG preview (required for PNG-only mode)
try:
    from PIL import Image, ImageTk
    _has_pil = True
except Exception:
    Image = None
    ImageTk = None
    _has_pil = False

# Optional cffi integration for in-process native call
try:
    from cffi import FFI
    _has_cffi = True
except Exception:
    FFI = None
    _has_cffi = False

# Cached native lib (cffi) handle
_amp_native_lib = None
_amp_native_ffi = None

def get_amp_native_lib():
    """Try to load amp_native.dll via cffi. Returns cffi lib or raises.
    Search order: environment var AMP_NATIVE_DLL_PATH, build/Debug, build/Release, PATH."""
    global _amp_native_lib
    if _amp_native_lib is not None:
        return _amp_native_lib
    if not _has_cffi:
        raise RuntimeError("cffi not available")
    ffi = FFI()
    # export the rich session API plus the legacy helper
    ffi.cdef("""
    int kpn_run_fft_division_from_buffer(const double *samples, unsigned long long frames, const char *params_json, const char *dump_prefix, int chunk_frames);

    typedef struct KpnStreamSession KpnStreamSession;
    KpnStreamSession *amp_kpn_session_create_from_blobs(const uint8_t *descriptor_blob, size_t descriptor_len, const uint8_t *plan_blob, size_t plan_len, int frames_hint, double sample_rate, uint32_t ring_frames, uint32_t block_frames);
    int amp_kpn_session_start(KpnStreamSession *session);
    void amp_kpn_session_stop(KpnStreamSession *session);
    void amp_kpn_session_destroy(KpnStreamSession *session);
    int amp_kpn_session_available(KpnStreamSession *session, unsigned long long *out_frames);
    int amp_kpn_session_read(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
    int amp_kpn_session_dump_count(KpnStreamSession *session, uint32_t *out_count);
    int amp_kpn_session_pop_dump(KpnStreamSession *session, double *destination, size_t max_frames, uint32_t *out_frames, uint32_t *out_channels, unsigned long long *out_sequence);
    int amp_kpn_session_status(KpnStreamSession *session, unsigned long long *out_produced_frames, unsigned long long *out_consumed_frames);
    int amp_kpn_session_stage_sampler_buffer(KpnStreamSession *session, const double *samples, size_t frames, uint32_t channels, const char *node_name);
    int amp_sampler_unregister(const char *node_name);
    """)

    # Try env var
    env_path = os.environ.get("AMP_NATIVE_DLL_PATH")
    candidates = []
    if env_path:
        candidates.append(env_path)
    # typical cmake multi-config output
    cwd = Path(__file__).resolve().parents[1]
    build = cwd / "build"
    candidates.extend([str(build / "Debug" / "amp_native.dll"), str(build / "Release" / "amp_native.dll")])
    # final fallback - let cffi search PATH
    last_exc = None
    for p in candidates:
        if not p:
            continue
        try:
            _amp_native_lib = ffi.dlopen(str(p))
            # store ffi so callers can allocate buffers with the same FFI instance
            globals()['_amp_native_ffi'] = ffi
            return _amp_native_lib
        except Exception as e:
            last_exc = e
    try:
        _amp_native_lib = ffi.dlopen(None)  # try process / PATH
        globals()['_amp_native_ffi'] = ffi
        return _amp_native_lib
    except Exception as e:
        last_exc = e
    raise RuntimeError("Failed to load amp_native via cffi") from last_exc

def call_kpn_run_fft_division(samples: np.ndarray, params_json: str = None, dump_prefix: str = None, chunk_frames: int = 0):
    """Call native kpn_run_fft_division_from_buffer via cffi.
    Returns integer result (0 == success)."""
    if not _has_cffi:
        raise RuntimeError("cffi not installed")
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a numpy array")
    arr = np.ascontiguousarray(samples, dtype=np.float64)
    ffi = FFI()
    lib = get_amp_native_lib()
    # prepare pointers
    ptr = ffi.cast("const double *", arr.ctypes.data)
    frames = int(arr.size)
    params_p = ffi.NULL
    dump_p = ffi.NULL
    if params_json is not None:
        params_bytes = params_json.encode("utf-8")
        params_p = ffi.new("char[]", params_bytes)
    if dump_prefix is not None:
        dump_bytes = dump_prefix.encode("utf-8")
        dump_p = ffi.new("char[]", dump_bytes)
    rc = lib.kpn_run_fft_division_from_buffer(ptr, frames, params_p, dump_p, int(chunk_frames))
    return rc


def load_wav_mono(path: str):
    """Load a WAV file (PCM) using the stdlib `wave` module.
    Returns (signal: np.ndarray(float64), sample_rate:int).
    Supports 8-bit unsigned, 16-bit signed, 32-bit signed PCM. Multi-channel is mixed to mono by averaging."""
    import wave
    with wave.open(path, 'rb') as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    if sampwidth == 1:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        # 8-bit WAV is unsigned [0,255]
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        arr = arr / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(data, dtype=np.int32).astype(np.float64)
        arr = arr / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)

    return arr.astype(np.float64), fr


def resample_signal(sig: np.ndarray, orig_sr: int, target_sr: int):
    """Simple linear-resample of 1D numpy array to target sample rate."""
    if orig_sr == target_sr:
        return sig
    import math
    orig_len = sig.shape[0]
    new_len = int(math.ceil(orig_len * float(target_sr) / float(orig_sr)))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float64)
    old_idx = np.linspace(0, orig_len - 1, num=orig_len)
    new_idx = np.linspace(0, orig_len - 1, num=new_len)
    new_sig = np.interp(new_idx, old_idx, sig).astype(np.float64)
    return new_sig


# Use fftfree-inspired mapping in `tools.fft_viz` (mag_to_db, mag_to_u8)


def load_audio_file(path: str):
    """Generic audio loader. Tries multiple backends in order:
    1. soundfile (pysoundfile)
    2. pydub (ffmpeg)
    3. stdlib wave (PCM WAV only)

    Returns (mono_float64_signal, sample_rate)
    """
    # 1) try soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True, dtype='float64')
        # data shape: (frames, channels)
        if data.ndim == 1 or data.shape[1] == 1:
            mono = data.reshape(-1)
        else:
            mono = data.mean(axis=1)
        return mono.astype(np.float64), int(sr)
    except Exception:
        pass

    # 2) try pydub (requires ffmpeg available)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples())
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        # convert integer samples to float in [-1,1]
        sw = seg.sample_width
        if sw == 1:
            samples = (samples.astype(np.float64) - 128.0) / 128.0
        elif sw == 2:
            samples = samples.astype(np.int16).astype(np.float64) / 32768.0
        elif sw == 4:
            samples = samples.astype(np.int32).astype(np.float64) / 2147483648.0
        else:
            samples = samples.astype(np.float64)
        return samples.astype(np.float64), int(sr)
    except Exception:
        pass

    # 3) fallback to simple WAV loader
    return load_wav_mono(path)





class FFTDivisionGUI:
    def __init__(self, master):
        self.master = master
        master.title("FFT Division Node Harness GUI")

        # Params frame
        params = ttk.LabelFrame(master, text="Parameters")
        params.grid(row=0, column=0, sticky="nw", padx=6, pady=6)

        ttk.Label(params, text="Window size:").grid(row=0, column=0, sticky="w")
        self.window_var = tk.IntVar(value=4)
        ttk.Entry(params, textvariable=self.window_var, width=8).grid(row=0, column=1)

        ttk.Label(params, text="Hop size:").grid(row=1, column=0, sticky="w")
        self.hop_var = tk.IntVar(value=1)
        ttk.Entry(params, textvariable=self.hop_var, width=8).grid(row=1, column=1)

        ttk.Label(params, text="Frames:").grid(row=2, column=0, sticky="w")
        self.frames_var = tk.IntVar(value=8)
        ttk.Entry(params, textvariable=self.frames_var, width=8).grid(row=2, column=1)

        ttk.Label(params, text="Window kind:").grid(row=3, column=0, sticky="w")
        self.window_kind = tk.StringVar(value="hann")
        ttk.Combobox(params, textvariable=self.window_kind, values=["hann", "rect"], width=6).grid(row=3, column=1)

        # Working convolution controls (wwin, whop)
        ttk.Label(params, text="WWIN (working window):").grid(row=4, column=0, sticky="w")
        self.wwin_var = tk.IntVar(value=4)
        ttk.Entry(params, textvariable=self.wwin_var, width=8).grid(row=4, column=1)

        ttk.Label(params, text="WHOP (working hop):").grid(row=5, column=0, sticky="w")
        self.whop_var = tk.IntVar(value=1)
        ttk.Entry(params, textvariable=self.whop_var, width=8).grid(row=5, column=1)
        
        # Streaming controls
        self.streaming_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params, text="Streaming mode", variable=self.streaming_var).grid(row=9, column=0, columnspan=2, sticky="w")
        ttk.Label(params, text="Chunk frames:").grid(row=10, column=0, sticky="w")
        self.chunk_frames_var = tk.IntVar(value=1024)
        ttk.Entry(params, textvariable=self.chunk_frames_var, width=8).grid(row=10, column=1)
        self.live_update_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Live update", variable=self.live_update_var).grid(row=11, column=0, columnspan=2, sticky="w")
        # dB floor slider for live spectrogram scaling (value is positive, e.g. 80 -> -80 dB)
        ttk.Label(params, text="dB floor:").grid(row=6, column=0, sticky="w")
        self.db_floor_var = tk.IntVar(value=80)
        db_scale = tk.Scale(params, from_=20, to=160, orient=tk.HORIZONTAL, variable=self.db_floor_var, command=self.on_db_floor_change)
        db_scale.grid(row=6, column=1, columnspan=1, sticky="we")

        # Display mode controls: allow selecting mapping mode and dB reference
        ttk.Label(params, text="Display mode:").grid(row=7, column=0, sticky="w")
        self.display_scale = tk.StringVar(value="db")
        ttk.Combobox(params, textvariable=self.display_scale, values=["db", "log1p", "linear"], width=8).grid(row=7, column=1)
        ttk.Label(params, text="dB ref:").grid(row=8, column=0, sticky="w")
        self.db_ref_var = tk.StringVar(value="global")
        ttk.Combobox(params, textvariable=self.db_ref_var, values=["global", "frame"], width=8).grid(row=8, column=1)

        # External binary
        binframe = ttk.LabelFrame(master, text="External Run (optional)")
        binframe.grid(row=1, column=0, sticky="nw", padx=6, pady=6)
        self.exec_path_var = tk.StringVar(value="")
        ttk.Entry(binframe, textvariable=self.exec_path_var, width=40).grid(row=0, column=0, columnspan=2)
        ttk.Button(binframe, text="Browse...", command=self.browse_exec).grid(row=0, column=2)
        ttk.Label(binframe, text="Timeout (s):").grid(row=1, column=0, sticky="w")
        self.timeout_var = tk.IntVar(value=10)
        ttk.Entry(binframe, textvariable=self.timeout_var, width=6).grid(row=1, column=1, sticky="w")

        # Audio file loader
        audioframe = ttk.LabelFrame(master, text="Audio Input")
        audioframe.grid(row=3, column=0, sticky="nw", padx=6, pady=6)
        self.audio_path_var = tk.StringVar(value="")
        ttk.Entry(audioframe, textvariable=self.audio_path_var, width=40).grid(row=0, column=0, columnspan=2)
        ttk.Button(audioframe, text="Load WAV...", command=self.browse_wav).grid(row=0, column=2)
        self.audio_info_var = tk.StringVar(value="No audio loaded")
        ttk.Label(audioframe, textvariable=self.audio_info_var).grid(row=1, column=0, columnspan=3, sticky="w")
        self.loaded_signal = None
        self.loaded_rate = None

        # Run button and status
        control = ttk.Frame(master)
        control.grid(row=2, column=0, sticky="nw", padx=6, pady=6)
        self.run_button = ttk.Button(control, text="Run (external)", command=self.on_run_external)
        self.run_button.grid(row=0, column=0, padx=(0, 6))
        self.run_inproc_button = ttk.Button(control, text="Run (in-process)", command=self.on_run_inprocess)
        self.run_inproc_button.grid(row=0, column=1, padx=(0, 6))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control, textvariable=self.status_var).grid(row=1, column=0, columnspan=3, sticky="w")

        # Plot frames
        plotframe = ttk.Frame(master)
        plotframe.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=6, pady=6)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # Waveform and spectrogram will be rendered as PIL images into
        # Tkinter Labels. This GUI is PNG-only (no matplotlib).
        # Fixed display sizes (spectrogram and waveform) to mimic matplotlib
        self._spec_w = 800
        self._spec_h = 320
        self._wave_w = self._spec_w
        self._wave_h = 80
        # Make plotframe a fixed-size container so the window doesn't jump
        plotframe.configure(width=self._spec_w, height=self._spec_h + self._wave_h)
        try:
            plotframe.grid_propagate(False)
        except Exception:
            pass
        self.wave_label = ttk.Label(plotframe)
        self.wave_label.pack(fill=tk.X, expand=False)

        # Fast PNG preview label (spectrogram)
        self.spec_preview_label = ttk.Label(plotframe)
        self.spec_preview_label.pack(fill=tk.BOTH, expand=True)

        self._current_thread = None

        # storage for last spectrogram magnitudes so slider can update display
        self._last_mag_mat = None  # shape: (bins, frames)
        self._floor_after_id = None
        self._spec_photo = None
        self._use_png_preview = _has_pil
        # preview generation counter to drop stale results
        self._preview_gen = 0

    def browse_exec(self):
        p = filedialog.askopenfilename(title="Select binary (test_fft_division_node)")
        if p:
            self.exec_path_var.set(p)

    def browse_wav(self):
        p = filedialog.askopenfilename(title="Select audio file", filetypes=[("Audio files","*.wav;*.flac;*.mp3;*.ogg;*.m4a;*.aiff"), ("All files","*")])
        if not p:
            return
        self.audio_path_var.set(p)
        try:
            sig, sr = load_audio_file(p)
            self.loaded_signal = sig
            self.loaded_rate = sr
            self.audio_info_var.set(f"Loaded: {Path(p).name} — {sig.size} frames @ {sr} Hz")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load WAV: {e}")
            self.loaded_signal = None
            self.loaded_rate = None
            self.audio_info_var.set("No audio loaded")


    def set_status(self, text: str):
        self.status_var.set(text)
        self.master.update_idletasks()

    def on_db_floor_change(self, val):
        """Called by the dB floor slider; debounce and update display from last mag."""
        try:
            if self._floor_after_id:
                self.master.after_cancel(self._floor_after_id)
        except Exception:
            pass
        # schedule update after 100ms
        self._floor_after_id = self.master.after(100, self._render_display_from_last_mag)

    def _render_display_from_last_mag(self):
        """Recompute display from stored magnitude matrix (no native re-run)."""
        self._floor_after_id = None
        if self._last_mag_mat is None:
            return
        try:
            db_floor = -float(self.db_floor_var.get())
        except Exception:
            db_floor = -80.0
        try:
            db = mag_to_db(self._last_mag_mat, db_ref="global", db_floor=db_floor, eps=1e-12)
            # db has shape (bins, frames); transpose for imshow (frames horizontal)
            img = db.T
            # Update fast PNG preview (compute off-main-thread and update label)
            if not self._use_png_preview:
                # PIL is required for PNG-only mode
                self.set_status("Pillow is not installed — spectrogram preview unavailable")
                return
            gen = self._preview_gen + 1
            self._preview_gen = gen

            def worker(local_gen, mag_mat, db_floor_local):
                try:
                    # Use user-selected display mapping and reference
                    scale = self.display_scale.get() if hasattr(self, 'display_scale') else "db"
                    db_ref = self.db_ref_var.get() if hasattr(self, 'db_ref_var') else "global"
                    u8 = mag_to_u8(mag_mat, scale=scale, db_ref=db_ref, db_floor=db_floor_local, eps=1e-12)
                except Exception:
                    # schedule fallback on main thread
                    try:
                        self.master.after(0, lambda: self._fallback_to_preview(db_floor_local, local_gen))
                    except Exception:
                        pass
                    return

                def apply_preview():
                    if local_gen != self._preview_gen:
                        return
                    try:
                        # autoscale spectrogram for better contrast
                        adj = autoscale_image(u8.T, method='percentile', low=1.0, high=99.0, gamma=1.0)
                        pil = Image.fromarray(adj, mode='L')
                        # Resize to fixed spectrogram box (do not preserve aspect ratio)
                        pil = pil.resize((self._spec_w, self._spec_h), resample=Image.BILINEAR)
                        self._spec_photo = ImageTk.PhotoImage(pil)
                        self.spec_preview_label.configure(image=self._spec_photo)
                        self.master.update_idletasks()
                    except Exception:
                        self._fallback_to_preview(db_floor_local, local_gen)

                try:
                    self.master.after(0, apply_preview)
                except Exception:
                    pass

            try:
                t = threading.Thread(target=worker, args=(gen, self._last_mag_mat.copy(), db_floor), daemon=True)
                t.start()
            except Exception:
                # synchronous fallback
                try:
                    scale = self.display_scale.get() if hasattr(self, 'display_scale') else "db"
                    db_ref = self.db_ref_var.get() if hasattr(self, 'db_ref_var') else "global"
                    u8 = mag_to_u8(self._last_mag_mat, scale=scale, db_ref=db_ref, db_floor=db_floor, eps=1e-12)
                    adj = autoscale_image(u8.T, method='percentile', low=1.0, high=99.0, gamma=1.0)
                    pil = Image.fromarray(adj, mode='L')
                    pil = pil.resize((self._spec_w, self._spec_h), resample=Image.BILINEAR)
                    self._spec_photo = ImageTk.PhotoImage(pil)
                    self.spec_preview_label.configure(image=self._spec_photo)
                    self.master.update_idletasks()
                except Exception:
                    self._fallback_to_preview(db_floor, gen)
        except Exception as e:
            # don't crash UI on render errors
            self.set_status(f"Render update failed: {e}")

    def _fallback_to_preview(self, db_floor_local, gen):
        """Fallback when preview generation fails: clear preview and set status."""
        try:
            if gen != self._preview_gen:
                return
            # clear preview image
            try:
                self.spec_preview_label.configure(image="")
                self._spec_photo = None
            except Exception:
                pass
            self.set_status("Spectrogram preview unavailable (render failed)")
        except Exception as e:
            self.set_status(f"Preview fallback failed: {e}")

    def on_run(self):
        # Deprecated: single-run simulation removed. Use external run.
        self.on_run_external()

    def on_run_external(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress")
            return
        path = self.exec_path_var.get().strip()
        if not path:
            messagebox.showwarning("No executable", "Please select the external executable path first.")
            return
        t = threading.Thread(target=self._run_external, args=(path,), daemon=True)
        self._current_thread = t
        t.start()

    def _run_external(self, path: str):
        self.set_status(f"Running external: {Path(path).name}")
        timeout = max(1, int(self.timeout_var.get()))

        # Prepare a temporary prefix for the harness to dump arrays
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="fftgui_")
        prefix = os.path.join(tmpdir, "dump")

        args = [path, "--window", str(self.window_var.get()), "--frames", str(self.frames_var.get()), "--hop", str(self.hop_var.get())]
        # pass working window/hop to external binary
        try:
            args.extend(["--wwin", str(int(self.wwin_var.get())), "--whop", str(int(self.whop_var.get()))])
        except Exception:
            pass

        # If a file was loaded in the GUI, write a truncated temporary WAV
        # and expose its path via env var `FFT_GUI_TRUNCATED_INPUT` so external
        # harnesses can pick it up if they support that convention.
        env = os.environ.copy()
        env["FFT_GUI_DUMP_PREFIX"] = prefix
        if self.loaded_signal is not None and self.loaded_signal.size > 0:
            try:
                import soundfile as sf
                use_sf = True
            except Exception:
                use_sf = False
            try:
                requested_N = int(self.frames_var.get())
            except Exception:
                requested_N = None
            sig_to_write = self.loaded_signal.astype(np.float64)
            if requested_N is not None and requested_N >= 0 and sig_to_write.size > requested_N:
                sig_to_write = sig_to_write[:requested_N]
            # write truncated WAV to temp file
            import tempfile
            try:
                tmpwav = tempfile.NamedTemporaryFile(prefix="fftgui_input_", suffix=".wav", delete=False)
                tmpwav_path = tmpwav.name
                tmpwav.close()
                sr = int(self.loaded_rate) if self.loaded_rate else 48000
                if use_sf:
                    sf.write(tmpwav_path, sig_to_write, sr, format='WAV', subtype='PCM_16')
                else:
                    # fallback using wave + int16 conversion
                    import wave
                    import struct
                    with wave.open(tmpwav_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        # clip and convert
                        clipped = np.clip(sig_to_write, -1.0, 1.0)
                        ints = (clipped * 32767.0).astype(np.int16)
                        wf.writeframes(ints.tobytes())
                env["FFT_GUI_TRUNCATED_INPUT"] = tmpwav_path
                # Inform user via status
                self.set_status(f"External run: prepared truncated input {os.path.basename(tmpwav_path)}")
            except Exception as e:
                # Non-fatal: continue without truncated input
                self.set_status(f"External run: failed to write truncated input: {e}")

        try:
            proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            status_msg = f"Exit {proc.returncode}. stdout {len(out)} bytes, stderr {len(err)} bytes"
            self.set_status(status_msg)

            # Try to read dumped files
            def read_vector(path):
                if not os.path.exists(path):
                    return None
                with open(path, "r") as f:
                    first = f.readline().strip()
                    try:
                        n = int(first)
                    except Exception:
                        return None
                    vals = []
                    for _ in range(n):
                        line = f.readline()
                        if not line:
                            break
                        vals.append(float(line.strip()))
                    return np.array(vals, dtype=float)

            def read_spectral(path):
                if not os.path.exists(path):
                    return None
                with open(path, "r") as f:
                    # Robust reader: header normally is "frames bins" but some
                    # dump writers may emit "bins frames" or mismatch. Read all
                    # numeric values and attempt to infer shape if needed.
                    header_line = f.readline().strip()
                    header = header_line.split()
                    vals = []
                    # read remaining numeric lines
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            vals.append(float(line))
                        except Exception:
                            # ignore non-numeric lines
                            continue

                    if len(header) >= 2:
                        try:
                            frames = int(header[0]); bins = int(header[1])
                        except Exception:
                            # try swapped
                            try:
                                bins = int(header[0]); frames = int(header[1])
                            except Exception:
                                frames = None; bins = None
                    else:
                        frames = None; bins = None

                    nvals = len(vals)
                    if frames is not None and bins is not None and frames * bins == nvals:
                        arr = np.array(vals, dtype=float).reshape((frames, bins))
                        return arr

                    # Try swapped header if provided
                    if frames is not None and bins is not None and frames * bins != nvals:
                        if bins * frames == nvals:
                            arr = np.array(vals, dtype=float).reshape((frames, bins))
                            return arr
                        # try swap
                        if bins * frames == nvals:
                            arr = np.array(vals, dtype=float).reshape((bins, frames)).T
                            return arr

                    # If header missing or doesn't match, try to infer dims.
                    if nvals == 0:
                        return None
                    # Heuristic: try to use known window size as bins
                    try:
                        guessed_bins = int(self.window_var.get())
                        if guessed_bins > 0 and nvals % guessed_bins == 0:
                            guessed_frames = nvals // guessed_bins
                            arr = np.array(vals, dtype=float).reshape((guessed_frames, guessed_bins))
                            # inform user via status
                            self.set_status(f"Parsed spectral dump with inferred shape ({guessed_frames}x{guessed_bins})")
                            return arr
                    except Exception:
                        pass

                    # Last resort: assume 2D with 1 frame
                    arr = np.array(vals, dtype=float).reshape((1, nvals))
                    self.set_status(f"Parsed spectral dump with fallback shape (1x{nvals})")
                    return arr

            first_pcm = read_vector(prefix + "_first_pcm.txt")
            first_real = read_spectral(prefix + "_first_spec_real.txt")
            first_imag = read_spectral(prefix + "_first_spec_imag.txt")

            expected_pcm = read_vector(prefix + "_expected_pcm.txt")
            expected_real = read_spectral(prefix + "_expected_spec_real.txt")
            expected_imag = read_spectral(prefix + "_expected_spec_imag.txt")

            if first_pcm is None and expected_pcm is None:
                # open log window so user can inspect stdout/stderr
                logwin = tk.Toplevel(self.master)
                logwin.title("External run output")
                txt = tk.Text(logwin, wrap="none", width=120, height=40)
                txt.pack(fill=tk.BOTH, expand=True)
                txt.insert("1.0", "=== STDOUT ===\n")
                txt.insert("end", out + "\n")
                txt.insert("end", "=== STDERR ===\n")
                txt.insert("end", err + "\n")
                self.set_status("Completed: no dump files found; see logs")
                return

            # Prefer dumped 'first' run if available, otherwise expected
            pcm_to_plot = first_pcm if first_pcm is not None else expected_pcm
            # If both real+imag spectral dumps are available, compute magnitude.
            spec_to_plot = None
            if first_real is not None and first_imag is not None:
                try:
                    spec_to_plot = np.sqrt(np.square(first_real) + np.square(first_imag))
                except Exception:
                    spec_to_plot = np.abs(first_real)
            elif first_real is not None:
                # fallback: use absolute of real part
                spec_to_plot = np.abs(first_real)
            elif expected_real is not None and expected_imag is not None:
                try:
                    spec_to_plot = np.sqrt(np.square(expected_real) + np.square(expected_imag))
                except Exception:
                    spec_to_plot = np.abs(expected_real)
            elif expected_real is not None:
                spec_to_plot = np.abs(expected_real)
            if spec_to_plot is None:
                # nothing to plot spectrally
                spec_to_plot = np.zeros((1, max(1, int(self.window_var.get()))))

            # Update plots on main thread
            self.master.after(0, lambda: self._update_plots(pcm_to_plot, spec_to_plot))

        except subprocess.TimeoutExpired:
            self.set_status("External run timed out")
        except Exception as e:
            self.set_status(f"External run error: {e}")

    def on_run_inprocess(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Busy", "A run is already in progress")
            return
        if self.streaming_var.get():
            t = threading.Thread(target=self._run_inprocess_streaming, daemon=True, args=())
        else:
            t = threading.Thread(target=self._run_inprocess, daemon=True, args=())
        self._current_thread = t
        t.start()

    def _run_inprocess_streaming(self):
        """Run the native processor in streaming mode by delivering chunks
        of the signal to the in-process native function and collecting
        per-chunk dumps as they are produced, appending to live buffers
        and updating the display.
        """
        self.set_status("Streaming in-process run (session drain)...")
        try:
            if not _has_cffi:
                self.set_status("cffi not installed; cannot run in-process streaming")
                return

            # Prepare signal as in _run_inprocess
            if self.loaded_signal is not None and self.loaded_signal.size > 0:
                sig = self.loaded_signal.astype(np.float64)
                sr = int(self.loaded_rate) if self.loaded_rate else 48000
                if sr != 48000:
                    sig = resample_signal(sig, sr, 48000)
                try:
                    requested_N = int(self.frames_var.get())
                except Exception:
                    requested_N = sig.size
                if requested_N >= 0 and sig.size > requested_N:
                    sig = sig[:requested_N]
            else:
                try:
                    N = int(self.frames_var.get())
                except Exception:
                    N = -1
                if N < 0:
                    N = 48000
                t = np.arange(N, dtype=float) / max(1.0, float(max(1, N)))
                sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
                sr = 48000

            try:
                chunk = int(self.chunk_frames_var.get())
                if chunk <= 0:
                    chunk = 1024
            except Exception:
                chunk = 1024

            # Acquire native lib and FFI
            lib = get_amp_native_lib()
            ffi = globals().get('_amp_native_ffi')
            if ffi is None:
                # defensive: create a new FFI (will likely not match the lib)
                ffi = FFI()

            # Create a generic session. Passing NULL blobs will let native
            # code fall back to the demo runtime (mirrors demo_kpn_native).
            session = lib.amp_kpn_session_create_from_blobs(ffi.NULL, 0, ffi.NULL, 0, int(sig.size), float(sr), max(65536, chunk * 8), int(chunk))
            if session == ffi.NULL or session is None:
                self.set_status("Failed to create native session")
                return

            rc = lib.amp_kpn_session_start(session)
            if rc != 0:
                self.set_status(f"Failed to start session (rc={rc})")
                try:
                    lib.amp_kpn_session_destroy(session)
                except Exception:
                    pass
                return

            accum_pcm = []
            accum_spec = None
            guessed_bins = max(1, int(self.window_var.get()))

            # Run session loop inside try/finally so we always perform
            # cleanup (unregister sampler, stop & destroy session).
            try:
                # Stage the entire input signal into the sampler registry under
                # a conventional node name 'sampler'. The runtime's graph must
                # contain a node instance with that name (or the user can modify
                # graphs to match). The registry copies the data so the Python
                # array may be short-lived.
                try:
                    arr_c = np.ascontiguousarray(sig, dtype=np.float64)
                    ptr = ffi.cast("const double *", arr_c.ctypes.data)
                    lib.amp_kpn_session_stage_sampler_buffer(session, ptr, arr_c.size, 1, ffi.new("char[]", b"sampler"))
                except Exception:
                    pass

                idle_cycles = 0
                max_idle = 200  # ~10s at 0.05s poll
                poll_interval = 0.05

                while True:
                    # Check available PCM frames
                    avail_ptr = ffi.new("unsigned long long *")
                    lib.amp_kpn_session_available(session, avail_ptr)
                    avail = int(avail_ptr[0])

                    if avail > 0:
                        # read in chunks up to `chunk`
                        to_read = min(avail, chunk)
                        out_frames_p = ffi.new("uint32_t *")
                        out_ch_p = ffi.new("uint32_t *")
                        seq_p = ffi.new("unsigned long long *")
                        # guess channels: use guessed_bins (spectral) or 1 (pcm)
                        max_channels_guess = max(1, guessed_bins)
                        buf = ffi.new("double[]", to_read * max_channels_guess)
                        r = lib.amp_kpn_session_read(session, buf, to_read, out_frames_p, out_ch_p, seq_p)
                        if r == 0:
                            nframes = int(out_frames_p[0])
                            nch = int(out_ch_p[0])
                            if nframes > 0 and nch > 0:
                                raw = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                                arr = raw.reshape((nframes, nch))
                                if nch == 1:
                                    accum_pcm.append(arr[:, 0])
                                elif nch == guessed_bins:
                                    # assume this is spectral payload: frames x bins
                                    spec_chunk = np.abs(arr)
                                    if accum_spec is None:
                                        accum_spec = spec_chunk
                                    else:
                                        try:
                                            accum_spec = np.concatenate([accum_spec, spec_chunk], axis=0)
                                        except Exception:
                                            # ignore concatenation errors
                                            pass
                                else:
                                    # Unknown channel count: try to treat as PCM if single-channel per-frame
                                    if nch == 1:
                                        accum_pcm.append(arr[:, 0])
                                    else:
                                        # try treat as spectral with nch as bins
                                        spec_chunk = np.abs(arr)
                                        if accum_spec is None:
                                            accum_spec = spec_chunk
                                        else:
                                            try:
                                                accum_spec = np.concatenate([accum_spec, spec_chunk], axis=0)
                                            except Exception:
                                                pass
                        idle_cycles = 0

                    # Drain any dump queue entries (spectral taps may appear here)
                    dump_count_p = ffi.new("uint32_t *")
                    lib.amp_kpn_session_dump_count(session, dump_count_p)
                    dump_count = int(dump_count_p[0])
                    while dump_count > 0:
                        out_frames_p = ffi.new("uint32_t *")
                        out_ch_p = ffi.new("uint32_t *")
                        seq_p = ffi.new("unsigned long long *")
                        # allocate a buffer large enough for chunk x guessed_bins
                        buf = ffi.new("double[]", max(1, chunk * guessed_bins))
                        r = lib.amp_kpn_session_pop_dump(session, buf, chunk, out_frames_p, out_ch_p, seq_p)
                        if r == 0:
                            nframes = int(out_frames_p[0]); nch = int(out_ch_p[0])
                            if nframes > 0 and nch > 0:
                                raw = np.frombuffer(ffi.buffer(buf, nframes * nch * 8), dtype=np.float64).copy()
                                arr = raw.reshape((nframes, nch))
                                spec_chunk = np.abs(arr)
                                if accum_spec is None:
                                    accum_spec = spec_chunk
                                else:
                                    try:
                                        accum_spec = np.concatenate([accum_spec, spec_chunk], axis=0)
                                    except Exception:
                                        pass
                        dump_count_p = ffi.new("uint32_t *")
                        lib.amp_kpn_session_dump_count(session, dump_count_p)
                        dump_count = int(dump_count_p[0])
                        idle_cycles = 0

                    # update GUI
                    if self.live_update_var.get():
                        to_plot_pcm = None
                        if len(accum_pcm) > 0:
                            try:
                                to_plot_pcm = np.concatenate(accum_pcm, axis=0)
                            except Exception:
                                to_plot_pcm = np.hstack(accum_pcm)
                        to_plot_spec = accum_spec if accum_spec is not None else np.zeros((1, max(1, guessed_bins)))
                        try:
                            self.master.after(0, lambda p=to_plot_pcm, s=to_plot_spec: self._update_plots(p, s))
                        except Exception:
                            pass

                    # termination heuristic: when no data for a number of idle cycles
                    if avail == 0 and dump_count == 0:
                        idle_cycles += 1
                    else:
                        idle_cycles = 0

                    if idle_cycles > max_idle:
                        break

                    time.sleep(poll_interval)
            finally:
                # Best-effort cleanup: unregister sampler entry and stop/destroy session
                try:
                    lib.amp_sampler_unregister(ffi.new("char[]", b"sampler"))
                except Exception:
                    pass
                try:
                    lib.amp_kpn_session_stop(session)
                except Exception:
                    pass
                try:
                    lib.amp_kpn_session_destroy(session)
                except Exception:
                    pass

            self.set_status("In-process streaming session finished")

        except Exception as e:
            self.set_status(f"In-process streaming (session) error: {e}")

    def _run_inprocess(self):
        self.set_status("Running in-process via amp_native.dll")
        try:
            if not _has_cffi:
                self.set_status("cffi not installed; cannot run in-process")
                return
            # Use loaded audio if present, otherwise build a simple mono test signal
            if self.loaded_signal is not None and self.loaded_signal.size > 0:
                sig = self.loaded_signal.astype(np.float64)
                sr = int(self.loaded_rate) if self.loaded_rate else 48000
                # resample if necessary to 48k
                if sr != 48000:
                    sig = resample_signal(sig, sr, 48000)
                # Respect GUI 'Frames' parameter by truncating the signal
                try:
                    requested_N = int(self.frames_var.get())
                except Exception:
                    requested_N = sig.size
                if requested_N >= 0 and sig.size > requested_N:
                    sig = sig[:requested_N]
            else:
                # If frames == -1, treat as "full signal" — but when no
                # loaded signal exists we'll use a reasonable default length
                # for generated test signal (1s @ 48k). This avoids crashes
                # from np.arange(-1).
                try:
                    N = int(self.frames_var.get())
                except Exception:
                    N = -1
                if N < 0:
                    N = 48000
                t = np.arange(N, dtype=float) / max(1.0, float(max(1, N)))
                sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

            # Prepare params JSON (window_size only)
            # include working window and hop so native in-process run can use them
            try:
                wwin = int(self.wwin_var.get())
            except Exception:
                wwin = int(self.window_var.get())
            try:
                whop = int(self.whop_var.get())
            except Exception:
                whop = int(self.hop_var.get())
            params = '{{"window_size": %d, "wwin": %d, "whop": %d}}' % (int(self.window_var.get()), wwin, whop)

            # Prepare temp dump prefix so native writes dumps we can reuse
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix="fftgui_inproc_")
            prefix = os.path.join(tmpdir, "dump")

            try:
                # Call into native with truncated or generated signal
                rc = call_kpn_run_fft_division(sig, params_json=params, dump_prefix=prefix, chunk_frames=0)
            except Exception as e:
                self.set_status(f"In-process call failed: {e}")
                return

            # Reuse same read helpers as external run
            def read_vector(path):
                if not os.path.exists(path):
                    return None
                with open(path, "r") as f:
                    first = f.readline().strip()
                    try:
                        n = int(first)
                    except Exception:
                        return None
                    vals = []
                    for _ in range(n):
                        line = f.readline()
                        if not line:
                            break
                        vals.append(float(line.strip()))
                    return np.array(vals, dtype=float)

            def read_spectral(path):
                if not os.path.exists(path):
                    return None
                with open(path, "r") as f:
                    header = f.readline().strip().split()
                    if len(header) < 2:
                        return None
                    frames = int(header[0]); bins = int(header[1])
                    vals = []
                    for _ in range(frames * bins):
                        line = f.readline()
                        if not line:
                            break
                        vals.append(float(line.strip()))
                    if len(vals) != frames * bins:
                        return None
                    arr = np.array(vals, dtype=float).reshape((frames, bins))
                    return arr

            first_pcm = read_vector(prefix + "_first_pcm.txt")
            first_real = read_spectral(prefix + "_first_spec_real.txt")
            first_imag = read_spectral(prefix + "_first_spec_imag.txt")

            if first_pcm is None:
                self.set_status("In-process run completed but no dump files found")
                return

            # If both real/imag present, compute magnitude; else take abs(real)
            if first_real is not None and first_imag is not None:
                try:
                    spec_to_plot = np.sqrt(np.square(first_real) + np.square(first_imag))
                except Exception:
                    spec_to_plot = np.abs(first_real)
            elif first_real is not None:
                spec_to_plot = np.abs(first_real)
            else:
                spec_to_plot = np.zeros((1, max(1, int(self.window_var.get()))))
            self.master.after(0, lambda: self._update_plots(first_pcm, spec_to_plot))
            self.set_status(f"In-process run completed (rc={rc})")
        except Exception as e:
            self.set_status(f"In-process error: {e}")

    def _run_simulate(self):
        try:
            self.set_status("Simulating...")
            W = int(self.window_var.get())
            H = int(self.hop_var.get())
            try:
                N = int(self.frames_var.get())
            except Exception:
                N = 1024
            if N < 0:
                # no loaded signal in simulate path; use sensible default
                N = 1024
            kind = self.window_kind.get()

            sig = generate_test_signal(N)
            spec = stft(sig, W, H, kind)
            mag = np.abs(spec)

            # Update plots on the main thread via after
            self.master.after(0, lambda: self._update_plots(sig, mag))
            self.set_status("Simulate: done")
        except Exception as e:
            self.set_status(f"Simulation error: {e}")

    def _update_plots(self, sig: np.ndarray, mag: np.ndarray):
        

        # spectral magnitude
        # mag is (frames, bins) — fft_viz expects (bins, frames), so transpose.
        mag_mat = mag.T
        # Use fftfree defaults: dB mapping by default with db_floor -80 dB
        db_floor = -80.0
        try:
            _ = mag_to_db(mag_mat, db_ref="global", db_floor=db_floor, eps=1e-12)
        except Exception:
            # ensure conversion doesn't raise
            pass
        # store last magnitude matrix for live updates (bins, frames)
        self._last_mag_mat = mag_mat
        # Render waveform as a small grayscale image
        try:
            h = min(self._wave_h, 256)
            img = np.zeros((h, sig.shape[0]), dtype=np.uint8)
            center = h // 2
            if sig.max() - sig.min() > 1e-12:
                s = (sig - sig.min()) / (sig.max() - sig.min()) * 2.0 - 1.0
            else:
                s = sig
            for i, val in enumerate(s):
                y = int(center + val * (center - 2))
                y = max(0, min(h - 1, y))
                img[y, i] = 255
            adj_wave = autoscale_image(img.astype('uint8'), method='percentile', low=1.0, high=99.0, gamma=1.0)
            pil_wave = Image.fromarray(adj_wave, mode='L')
            # Always scale waveform into fixed box (no aspect preservation)
            pil_wave = pil_wave.resize((self._wave_w, self._wave_h), resample=Image.BILINEAR)
            self._wave_photo = ImageTk.PhotoImage(pil_wave)
            self.wave_label.configure(image=self._wave_photo)
        except Exception:
            # ignore waveform render errors
            pass

        # Schedule spectrogram preview render using existing display logic
        try:
            self.master.after(0, self._render_display_from_last_mag)
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = FFTDivisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
