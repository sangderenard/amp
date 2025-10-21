"""
Modular controller input monitor for both interactive and benchmark modes.
Handles polling (real or synthetic), writes to control history, and prepares/resamples curves.
"""
import threading
import time
import numpy as np

class ControllerMonitor:
    def __init__(self, poll_fn, control_history, poll_interval=0.005, audio_frame_rate=44100):
        """
        poll_fn: function returning (axes, buttons) at each poll
        control_history: object with .record_control_event(timestamp, axes, buttons, extras)
        poll_interval: seconds between polls (controller polling rate)
        audio_frame_rate: target frame rate for resampling curves
        """
        self.poll_fn = poll_fn
        self.control_history = control_history
        self.poll_interval = poll_interval
        self.audio_frame_rate = audio_frame_rate
        self.running = False
        self.thread = None
        self._lock = threading.Lock()
        self._latest_curves = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.running:
            timestamp = time.perf_counter()
            axes, buttons = self.poll_fn()
            # Prepare extras for control history (can be extended)
            extras = {
                "axes": np.asarray(axes, dtype=np.float32),
                "buttons": np.asarray(buttons, dtype=np.float32),
            }
            self.control_history.record_control_event(timestamp, axes, buttons, extras)
            # Prepare curves resampled to audio frame rate (example: just store latest)
            with self._lock:
                self._latest_curves = self._prepare_curves(axes, buttons)
            time.sleep(self.poll_interval)

    def _prepare_curves(self, axes, buttons):
        # Example: resample axes/buttons to audio frame rate (can be extended)
        # For now, just return as-is
        return {
            "axes": np.asarray(axes, dtype=np.float32),
            "buttons": np.asarray(buttons, dtype=np.float32),
        }

    def get_latest_curves(self):
        with self._lock:
            return self._latest_curves

# Usage:
# - For interactive: pass a poll_fn that reads from hardware
# - For benchmark: pass a poll_fn that generates synthetic input
# - Both use the same ControllerMonitor instance
