from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from enum import IntEnum
from typing import Callable, Mapping, List, Optional, Tuple

import numpy as np

from .utils import lanczos_resample
from . import native_build

AVAILABLE = False
_IMPL = None
UNAVAILABLE_REASON: str | None = None

native_build.ensure_toolchain_env()
_BUILD_CONFIG = native_build.get_build_config()
_LOGGING_ENABLED = _BUILD_CONFIG.logging_enabled

_LIBRARY_OVERRIDE_ENV = "AMP_NATIVE_RUNTIME_PATH"
_FORCE_REBUILD_ENV = "AMP_NATIVE_FORCE_REBUILD"


def _library_name() -> str:
    if sys.platform == "win32":
        return "amp_native.dll"
    if sys.platform == "darwin":
        return "libamp_native.dylib"
    return "libamp_native.so"


def _native_root() -> Path:
    return Path(__file__).resolve().parents[1] / "native"


def _build_dir() -> Path:
    return _native_root() / "build"


def _parse_force_flag() -> bool:
    value = os.environ.get(_FORCE_REBUILD_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _ensure_library_path(force: bool = False) -> Path:
    override = os.environ.get(_LIBRARY_OVERRIDE_ENV, "")
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Native runtime override path not found: {path}")
        return path

    build_dir = _build_dir()
    release_candidate = build_dir / "Release" / _library_name()
    candidates = [
        build_dir / _library_name(),
        release_candidate,
        _native_root() / _library_name(),
    ]
    for candidate in candidates:
        if candidate.exists() and not force:
            return candidate

    return _build_native_library(force=force)


def _build_native_library(force: bool = False) -> Path:
    native_dir = _native_root()
    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = ["cmake", "-S", str(native_dir), "-B", str(build_dir)]
    if _LOGGING_ENABLED:
        cmake_args.append("-DAMP_NATIVE_ENABLE_LOGGING=ON")

    extra_config = os.environ.get("AMP_NATIVE_CMAKE_ARGS", "")
    if extra_config:
        cmake_args.extend(shlex.split(extra_config))

    env = native_build.command_environment()

    subprocess.run(cmake_args, check=True, cwd=native_dir, env=env)

    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
    if force and sys.platform != "win32":
        build_cmd.append("--clean-first")
    subprocess.run(build_cmd, check=True, cwd=native_dir, env=env)

    library_name = _library_name()
    candidates = [build_dir / library_name]
    if sys.platform == "win32":
        candidates.extend((build_dir / "Release").glob("*.dll"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to locate built native library '{library_name}' in {build_dir}")

_CDEF = """
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef struct AmpGraphRuntime AmpGraphRuntime;
typedef struct AmpGraphControlHistory AmpGraphControlHistory;
typedef struct EdgeRunnerControlHistory EdgeRunnerControlHistory;
typedef struct {
    uint32_t measured_delay_frames;
    float accumulated_heat;
    double processing_time_seconds;
    double logging_time_seconds;
    double total_time_seconds;
    double thread_cpu_time_seconds;
    double reserved[6];
} AmpNodeMetrics;
typedef enum {
    AMP_EXECUTION_MODE_FORWARD = 0,
    AMP_EXECUTION_MODE_BACKWARD = 1
} AmpExecutionMode;
typedef struct {
    uint32_t declared_delay_frames;
    uint32_t oversample_ratio;
    int supports_v2;
    int has_metrics;
    AmpNodeMetrics metrics;
    double total_heat_accumulated;
} AmpGraphNodeSummary;
typedef enum {
    AMP_SCHEDULER_ORDERED = 0,
    AMP_SCHEDULER_LEARNED = 1
} AmpGraphSchedulerMode;
typedef struct {
    double early_bias;
    double late_bias;
    double saturation_bias;
} AmpGraphSchedulerParams;
typedef struct {
    int code;
    const char *stage;
    const char *node;
    const char *detail;
} AmpGraphRuntimeErrorInfo;
AmpGraphRuntime *amp_graph_runtime_create(
    const uint8_t *descriptor_blob,
    size_t descriptor_len,
    const uint8_t *plan_blob,
    size_t plan_len
);
void amp_graph_runtime_destroy(AmpGraphRuntime *runtime);
int amp_graph_runtime_configure(AmpGraphRuntime *runtime, uint32_t batches, uint32_t frames);
void amp_graph_runtime_set_dsp_sample_rate(AmpGraphRuntime *runtime, double sample_rate);
int amp_graph_runtime_set_scheduler_mode(AmpGraphRuntime *runtime, AmpGraphSchedulerMode mode);
int amp_graph_runtime_set_scheduler_params(AmpGraphRuntime *runtime, const AmpGraphSchedulerParams *params);
void amp_graph_runtime_clear_params(AmpGraphRuntime *runtime);
int amp_graph_runtime_set_param(
    AmpGraphRuntime *runtime,
    const char *node_name,
    const char *param_name,
    const double *data,
    uint32_t batches,
    uint32_t channels,
    uint32_t frames
);
int amp_graph_runtime_describe_node(
    AmpGraphRuntime *runtime,
    const char *node_name,
    AmpGraphNodeSummary *summary
);
int amp_graph_runtime_execute(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
int amp_graph_runtime_execute_with_history(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double **out_buffer,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
int amp_graph_runtime_execute_into(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
int amp_graph_runtime_execute_history_into(
    AmpGraphRuntime *runtime,
    AmpGraphControlHistory *history,
    int frames_hint,
    double sample_rate,
    double *out_buffer,
    size_t out_buffer_len,
    uint32_t *out_batches,
    uint32_t *out_channels,
    uint32_t *out_frames
);
int amp_graph_runtime_last_error(
    AmpGraphRuntime *runtime,
    AmpGraphRuntimeErrorInfo *out_error
);
void amp_graph_runtime_buffer_free(double *buffer);
AmpGraphControlHistory *amp_graph_history_load(const uint8_t *blob, size_t blob_len, int frames_hint);
void amp_graph_history_destroy(AmpGraphControlHistory *history);
int amp_run_node_v2(
    const struct EdgeRunnerNodeDescriptor *descriptor,
    const struct EdgeRunnerNodeInputs *inputs,
    int batches,
    int channels,
    int frames,
    double sample_rate,
    double **out_buffer,
    int *out_channels,
    void **state,
    const EdgeRunnerControlHistory *history,
    AmpExecutionMode mode,
    AmpNodeMetrics *metrics
);
int amp_native_logging_enabled(void);
void amp_native_logging_set(int enabled);
typedef struct AmpGraphStreamer AmpGraphStreamer;
AmpGraphStreamer *amp_graph_streamer_create(
    AmpGraphRuntime *runtime,
    const uint8_t *control_blob,
    size_t control_len,
    int frames_hint,
    double sample_rate,
    uint32_t ring_frames,
    uint32_t block_frames
);
int amp_graph_streamer_start(AmpGraphStreamer *streamer);
void amp_graph_streamer_stop(AmpGraphStreamer *streamer);
void amp_graph_streamer_destroy(AmpGraphStreamer *streamer);
int amp_graph_streamer_available(AmpGraphStreamer *streamer, uint64_t *out_frames);
int amp_graph_streamer_read(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
);
int amp_graph_streamer_dump_count(AmpGraphStreamer *streamer, uint32_t *out_count);
int amp_graph_streamer_pop_dump(
    AmpGraphStreamer *streamer,
    double *destination,
    size_t max_frames,
    uint32_t *out_frames,
    uint32_t *out_channels,
    uint64_t *out_sequence
);
int amp_graph_streamer_status(
    AmpGraphStreamer *streamer,
    uint64_t *out_produced_frames,
    uint64_t *out_consumed_frames
);
"""


class SchedulerMode(IntEnum):
    ORDERED = 0
    LEARNED = 1


def _load_impl() -> tuple["cffi.FFI", object]:
    global AVAILABLE, _IMPL, UNAVAILABLE_REASON
    if _IMPL is not None:
        AVAILABLE = True
        UNAVAILABLE_REASON = None
        return _IMPL
    try:
        import cffi

        ffi = cffi.FFI()
        ffi.cdef(_CDEF)
        force = _parse_force_flag()
        library_path = _ensure_library_path(force=force)
        lib = ffi.dlopen(str(library_path))
        _IMPL = (ffi, lib)
        AVAILABLE = True
        UNAVAILABLE_REASON = None
    except Exception as exc:  # pragma: no cover - depends on build tooling
        AVAILABLE = False
        UNAVAILABLE_REASON = f"Failed to load native graph runtime: {exc}"
        raise
    return _IMPL


def get_graph_runtime_impl() -> tuple["cffi.FFI", object]:
    """Return the (ffi, lib) pair for the native graph runtime."""

    return _load_impl()


def set_native_logging_enabled(enabled: bool) -> None:
    """Propagate the bridge logging preference to the native runtime."""

    try:
        _, lib = _load_impl()
    except Exception:
        return
    setter = getattr(lib, "amp_native_logging_set", None)
    if setter is None:
        return
    try:
        setter(1 if enabled else 0)
    except Exception:
        return


class NativeGraphExecutor:
    """Small shim that evaluates an AudioGraph via the native runtime."""

    def __init__(self, graph) -> None:
        if graph is None:
            raise ValueError("graph must be provided")
        self._graph = graph
        self.ffi, self.lib = get_graph_runtime_impl()
        descriptor_blob = graph.serialize_node_descriptors()
        plan_blob = graph.serialize_compiled_plan()
        desc_buf = self.ffi.new("uint8_t[]", descriptor_blob)
        if plan_blob:
            plan_buf = self.ffi.new("uint8_t[]", plan_blob)
            plan_ptr = plan_buf
            plan_len = len(plan_blob)
        else:
            plan_buf = None
            plan_ptr = self.ffi.NULL
            plan_len = 0
        runtime = self.lib.amp_graph_runtime_create(
            desc_buf,
            len(descriptor_blob),
            plan_ptr,
            plan_len,
        )
        if runtime == self.ffi.NULL:
            self._runtime = self.ffi.NULL
            raise RuntimeError("failed to create native graph runtime instance")
        self._runtime = runtime
        self._lock = threading.Lock()
        self._set_dsp_sample_rate: Callable[[object, float], object] | None
        try:
            self._set_dsp_sample_rate = self.lib.amp_graph_runtime_set_dsp_sample_rate
        except AttributeError:
            self._set_dsp_sample_rate = None
        self._set_scheduler_mode: Callable[[object, int], int] | None
        try:
            self._set_scheduler_mode = self.lib.amp_graph_runtime_set_scheduler_mode
        except AttributeError:
            self._set_scheduler_mode = None
        self._set_scheduler_params: Callable[[object, object], int] | None
        try:
            self._set_scheduler_params = self.lib.amp_graph_runtime_set_scheduler_params
        except AttributeError:
            self._set_scheduler_params = None
        if self._set_scheduler_mode is not None:
            try:
                self._set_scheduler_mode(self._runtime, int(SchedulerMode.LEARNED))
            except Exception:
                pass
        if self._set_scheduler_params is not None:
            try:
                params = self.ffi.new("AmpGraphSchedulerParams *")
                params.early_bias = 0.5
                params.late_bias = 0.5
                params.saturation_bias = 1.0
                self._set_scheduler_params(self._runtime, params)
            except Exception:
                pass
        self._param_cache: dict[str, dict[str, np.ndarray]] = {}
        self._history_handle = self.ffi.NULL
        self._history_blob: bytes | None = None
        self._history_frames_hint: int | None = None
        self._history_buffer = None

    def __enter__(self) -> "NativeGraphExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_runtime", self.ffi.NULL) not in (None, self.ffi.NULL):
            self.lib.amp_graph_runtime_destroy(self._runtime)
            self._runtime = self.ffi.NULL
        if getattr(self, "_history_handle", self.ffi.NULL) not in (None, self.ffi.NULL):
            try:
                self.lib.amp_graph_history_destroy(self._history_handle)
            except Exception:
                pass
        self._history_handle = self.ffi.NULL
        self._history_blob = None
        self._history_frames_hint = None
        self._history_buffer = None
        self._param_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _ensure_history(self, blob: bytes | memoryview | bytearray, frames_hint: int):
        if isinstance(blob, memoryview):
            blob_bytes = blob.tobytes()
        else:
            blob_bytes = bytes(blob)
        if (
            getattr(self, "_history_handle", self.ffi.NULL) not in (None, self.ffi.NULL)
            and self._history_blob == blob_bytes
            and self._history_frames_hint == frames_hint
        ):
            return self._history_handle
        if getattr(self, "_history_handle", self.ffi.NULL) not in (None, self.ffi.NULL):
            try:
                self.lib.amp_graph_history_destroy(self._history_handle)
            except Exception:
                pass
        buffer_obj = self.ffi.new("uint8_t[]", blob_bytes)
        handle = self.lib.amp_graph_history_load(buffer_obj, len(blob_bytes), frames_hint)
        if handle == self.ffi.NULL:
            raise RuntimeError("failed to load control history")
        self._history_handle = handle
        self._history_blob = blob_bytes
        self._history_frames_hint = frames_hint
        self._history_buffer = buffer_obj
        return handle

    def set_scheduler_mode(self, mode: SchedulerMode | int) -> None:
        if self._set_scheduler_mode is None:
            raise RuntimeError("native runtime does not expose scheduler mode controls")
        if isinstance(mode, SchedulerMode):
            value = int(mode)
        else:
            value = int(SchedulerMode(mode))
        self._set_scheduler_mode(self._runtime, value)

    def set_scheduler_params(self, early_bias: float, late_bias: float, saturation_bias: float) -> None:
        if self._set_scheduler_params is None:
            raise RuntimeError("native runtime does not expose scheduler parameter controls")
        params = self.ffi.new("AmpGraphSchedulerParams *")
        params.early_bias = float(early_bias)
        params.late_bias = float(late_bias)
        params.saturation_bias = float(saturation_bias)
        self._set_scheduler_params(self._runtime, params)

    def _bind_base_params(
        self,
        batches: int,
        dsp_frames: int,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None,
    ) -> list[np.ndarray]:
        keepalive: list[np.ndarray] = []
        existing_keys = {
            (node_name, param_name)
            for node_name, params in self._param_cache.items()
            for param_name in params.keys()
        }
        existing_cache = self._param_cache
        new_cache: dict[str, dict[str, np.ndarray]] = {}
        if base_params:
            desired_keys: set[tuple[str, str]] = set()
            for node_name, params in base_params.items():
                if node_name.startswith("_"):
                    continue
                for param_name in params.keys():
                    desired_keys.add((node_name, param_name))
            need_reset = desired_keys != existing_keys
            if not need_reset:
                for node_name, params in base_params.items():
                    if node_name.startswith("_"):
                        continue
                    node_cache = existing_cache.get(node_name, {})
                    for param_name, array in params.items():
                        prev = node_cache.get(param_name)
                        if prev is None:
                            continue
                        if prev.shape != np.shape(array):
                            need_reset = True
                            break
                    if need_reset:
                        break
            if need_reset:
                self.lib.amp_graph_runtime_clear_params(self._runtime)
                existing_cache = {}
            for node_name, params in base_params.items():
                if node_name.startswith("_"):
                    continue
                node_cache = existing_cache.get(node_name, {})
                new_node_cache: dict[str, np.ndarray] = {}
                node_bytes = node_name.encode("utf-8")
                for param_name, array in params.items():
                    arr = np.asarray(array, dtype=np.float64)
                    if arr.ndim != 3:
                        raise ValueError(
                            f"param '{param_name}' for node '{node_name}' must be BxCxF"
                        )
                    arr_c = np.require(arr, requirements=("C",))
                    prev = node_cache.get(param_name)
                    if (
                        prev is not None
                        and prev.shape == arr_c.shape
                        and np.array_equal(prev, arr_c)
                    ):
                        new_node_cache[param_name] = prev
                        continue
                    keepalive.append(arr_c)
                    ptr = self.ffi.from_buffer("double[]", arr_c)
                    status = self.lib.amp_graph_runtime_set_param(
                        self._runtime,
                        node_bytes,
                        param_name.encode("utf-8"),
                        ptr,
                        arr_c.shape[0],
                        arr_c.shape[1],
                        arr_c.shape[2],
                    )
                    if int(status) != 0:
                        message = self._format_error(
                            f"failed to bind param '{param_name}' for node '{node_name}'",
                            self.last_error(),
                        )
                        raise RuntimeError(message)
                    new_node_cache[param_name] = np.array(arr_c, copy=True)
                if new_node_cache:
                    new_cache[node_name] = new_node_cache
            self._param_cache = new_cache
        else:
            if existing_keys:
                self.lib.amp_graph_runtime_clear_params(self._runtime)
            self._param_cache = {}
        return keepalive

    def _decode_optional_string(self, pointer) -> str | None:
        if pointer in (None, self.ffi.NULL):
            return None
        return self.ffi.string(pointer).decode("utf-8", "replace")

    def last_error(self) -> dict[str, object] | None:
        """Return the last error reported by the native runtime, if any."""

        if getattr(self, "_runtime", self.ffi.NULL) in (None, self.ffi.NULL):
            return None
        info = self.ffi.new("AmpGraphRuntimeErrorInfo *")
        rc = self.lib.amp_graph_runtime_last_error(self._runtime, info)
        if int(rc) != 0:
            return None
        error = {
            "code": int(info.code),
            "stage": self._decode_optional_string(info.stage),
            "node": self._decode_optional_string(info.node),
            "detail": self._decode_optional_string(info.detail),
        }
        if error["code"] == 0 and not error["stage"] and not error["node"] and not error["detail"]:
            return None
        return error

    def _format_error(self, prefix: str, error: dict[str, object] | None) -> str:
        if not error:
            return prefix
        parts: list[str] = []
        code = error.get("code")
        if code is not None:
            parts.append(f"code={code}")
        stage = error.get("stage")
        if stage:
            parts.append(f"stage={stage}")
        node = error.get("node")
        if node:
            parts.append(f"node={node}")
        detail = error.get("detail")
        if detail:
            parts.append(str(detail))
        detail_str = ", ".join(parts)
        return f"{prefix} ({detail_str})" if detail_str else prefix

    def run_block(
        self,
        frames: int,
        sample_rate: float,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        control_history_blob: bytes | None = None,
        timeout: float | None = None,
        out_buffer: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._runtime == self.ffi.NULL:
            raise RuntimeError("native runtime has been closed")
        if frames <= 0:
            raise ValueError("frames must be positive")
        output_frames = int(frames)
        audio_rate = float(sample_rate)
        if audio_rate <= 0.0:
            raise ValueError("sample_rate must be positive")
        dsp_rate = float(getattr(self._graph, "dsp_sample_rate", audio_rate) or audio_rate)
        ratio = dsp_rate / audio_rate
        dsp_frames = int(math.ceil(output_frames * ratio)) if ratio > 0 else output_frames
        with self._lock:
            batches = 1
            if base_params and "_B" in base_params:
                batches = int(base_params["_B"])
            self.lib.amp_graph_runtime_configure(self._runtime, batches, dsp_frames)
            if self._set_dsp_sample_rate is not None:
                self._set_dsp_sample_rate(self._runtime, float(dsp_rate))
            keepalive = self._bind_base_params(batches, dsp_frames, base_params)
            if out_buffer is not None:
                if out_buffer.dtype != np.float64:
                    raise TypeError("out_buffer must use float64 samples")
                if out_buffer.ndim != 3:
                    raise ValueError("out_buffer must have shape (batches, channels, frames)")
                if not out_buffer.flags.c_contiguous:
                    raise ValueError("out_buffer must be C-contiguous")
                out_target = out_buffer
                target_ptr = self.ffi.cast("double *", out_target.ctypes.data)
            else:
                out_target = None
                target_ptr = None
            history_handle = self.ffi.NULL
            ctrl_blob_bytes = b""
            if control_history_blob:
                history_handle = self._ensure_history(control_history_blob, dsp_frames)
            if history_handle == self.ffi.NULL:
                ctrl_blob_bytes = control_history_blob or b""
                if ctrl_blob_bytes:
                    ctrl_buf = self.ffi.new("uint8_t[]", ctrl_blob_bytes)
                    keepalive.append(ctrl_buf)
            else:
                ctrl_buf = self.ffi.NULL
            out_ptr = self.ffi.new("double **")
            out_batches = self.ffi.new("uint32_t *")
            out_channels = self.ffi.new("uint32_t *")
            out_frames = self.ffi.new("uint32_t *")
            if history_handle != self.ffi.NULL:
                if target_ptr is not None:
                    status = self.lib.amp_graph_runtime_execute_history_into(
                        self._runtime,
                        history_handle,
                        dsp_frames,
                        float(dsp_rate),
                        target_ptr,
                        out_target.size,
                        out_batches,
                        out_channels,
                        out_frames,
                    )
                else:
                    status = self.lib.amp_graph_runtime_execute_with_history(
                        self._runtime,
                        history_handle,
                        dsp_frames,
                        float(dsp_rate),
                        out_ptr,
                        out_batches,
                        out_channels,
                        out_frames,
                    )
            else:
                if target_ptr is not None:
                    status = self.lib.amp_graph_runtime_execute_into(
                        self._runtime,
                        ctrl_buf if ctrl_blob_bytes else self.ffi.NULL,
                        len(ctrl_blob_bytes),
                        dsp_frames,
                        float(dsp_rate),
                        target_ptr,
                        out_target.size,
                        out_batches,
                        out_channels,
                        out_frames,
                    )
                else:
                    status = self.lib.amp_graph_runtime_execute(
                        self._runtime,
                        ctrl_buf if ctrl_blob_bytes else self.ffi.NULL,
                        len(ctrl_blob_bytes),
                        dsp_frames,
                        float(dsp_rate),
                        out_ptr,
                        out_batches,
                        out_channels,
                        out_frames,
                    )
            if int(status) != 0:
                message = self._format_error(
                    f"native runtime execution failed (status {int(status)})",
                    self.last_error(),
                )
                raise RuntimeError(message)
            if target_ptr is not None:
                if (
                    int(out_batches[0]) != out_target.shape[0]
                    or int(out_channels[0]) != out_target.shape[1]
                    or int(out_frames[0]) != out_target.shape[2]
                ):
                    raise RuntimeError("output shape mismatch for provided buffer")
                array = out_target
            else:
                total = int(out_batches[0]) * int(out_channels[0]) * int(out_frames[0])
                buffer = self.ffi.buffer(out_ptr[0], total * np.dtype(np.float64).itemsize)
                array = np.frombuffer(buffer, dtype=np.float64).copy().reshape(
                    int(out_batches[0]), int(out_channels[0]), int(out_frames[0])
                )
        if int(out_frames[0]) != output_frames or not np.isclose(dsp_rate, audio_rate):
            array = lanczos_resample(array, dsp_rate, audio_rate, output_frames)
        return np.nan_to_num(array, copy=False)
    def create_streamer(
        self,
        *,
        total_frames: int,
        sample_rate: float,
        base_params: Mapping[str, Mapping[str, np.ndarray]] | None = None,
        control_history_blob: bytes | None = None,
        ring_frames: int | None = None,
        block_frames: int | None = None,
    ) -> "NativeGraphStreamer":
        if self._runtime == self.ffi.NULL:
            raise RuntimeError("native runtime has been closed")
        if total_frames <= 0:
            raise ValueError("total_frames must be positive")
        audio_rate = float(sample_rate)
        if audio_rate <= 0.0:
            raise ValueError("sample_rate must be positive")
        dsp_rate = float(getattr(self._graph, "dsp_sample_rate", audio_rate) or audio_rate)
        ratio = dsp_rate / audio_rate
        dsp_frames = int(math.ceil(total_frames * ratio)) if ratio > 0 else total_frames
        ring_frames = int(ring_frames or total_frames)
        if ring_frames <= 0:
            raise ValueError("ring_frames must be positive")
        default_block = min(ring_frames, max(1, getattr(self._graph, "block_size", ring_frames)))
        block_frames = int(block_frames or default_block)
        if block_frames <= 0:
            block_frames = 1
        with self._lock:
            batches = 1
            if base_params and "_B" in base_params:
                batches = int(base_params["_B"])
            self.lib.amp_graph_runtime_configure(self._runtime, batches, dsp_frames)
            if self._set_dsp_sample_rate is not None:
                self._set_dsp_sample_rate(self._runtime, float(dsp_rate))
            keepalive = self._bind_base_params(batches, dsp_frames, base_params)
            ctrl_blob_bytes = control_history_blob or b""
            ctrl_buf = self.ffi.new("uint8_t[]", ctrl_blob_bytes) if ctrl_blob_bytes else self.ffi.NULL
            streamer_ptr = self.lib.amp_graph_streamer_create(
                self._runtime,
                ctrl_buf if ctrl_blob_bytes else self.ffi.NULL,
                len(ctrl_blob_bytes),
                dsp_frames,
                float(dsp_rate),
                ring_frames,
                block_frames,
            )
            if streamer_ptr == self.ffi.NULL:
                raise RuntimeError("failed to create graph streamer")
            if ctrl_blob_bytes:
                keepalive.append(ctrl_buf)
        streamer = NativeGraphStreamer(
            executor=self,
            ptr=streamer_ptr,
            keepalive=keepalive,
            batches=batches,
            ring_frames=ring_frames,
            total_frames=total_frames,
            block_frames=block_frames,
            dsp_frames=dsp_frames,
            sample_rate=float(dsp_rate),
        )
        return streamer


class NativeGraphStreamer:
    """Wraps the native streaming runtime and exposes low-latency ring access.

    This intentionally pairs Python orchestration with the C hot path; the native
    streamer continues to run independently, and Python is limited to configuration
    and data collection. This co-development is an accepted exception to the
    "no Python fallback" guidance because the DSP work remains entirely in C.
    """

    def __init__(
        self,
        *,
        executor: NativeGraphExecutor,
        ptr,
        keepalive: list[np.ndarray],
        batches: int,
        ring_frames: int,
        total_frames: int,
        block_frames: int,
        dsp_frames: int,
        sample_rate: float,
    ) -> None:
        self._executor = executor
        self.ffi = executor.ffi
        self.lib = executor.lib
        self._streamer = ptr
        self._keepalive: List[object] = list(keepalive)
        self._batches = batches
        self._channels: Optional[int] = None
        self._ring_frames = ring_frames
        self._total_frames = total_frames
        self._block_frames = block_frames
        self._dsp_frames = dsp_frames
        self._sample_rate = sample_rate
        self._running = False

    def _check_open(self) -> None:
        if self._streamer in (None, self.ffi.NULL):
            raise RuntimeError("streamer has been closed")

    def start(self) -> None:
        self._check_open()
        rc = self.lib.amp_graph_streamer_start(self._streamer)
        if int(rc) != 0:
            raise RuntimeError("failed to start streamer")
        self._running = True

    def stop(self) -> None:
        if self._streamer in (None, self.ffi.NULL):
            return
        self.lib.amp_graph_streamer_stop(self._streamer)
        self._running = False

    def close(self) -> None:
        if self._streamer in (None, self.ffi.NULL):
            return
        try:
            self.stop()
        finally:
            self.lib.amp_graph_streamer_destroy(self._streamer)
            self._streamer = self.ffi.NULL

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "NativeGraphStreamer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def status(self) -> Tuple[int, int, int]:
        self._check_open()
        produced = self.ffi.new("uint64_t *")
        consumed = self.ffi.new("uint64_t *")
        rc = self.lib.amp_graph_streamer_status(self._streamer, produced, consumed)
        return int(produced[0]), int(consumed[0]), int(rc)

    def dump_count(self) -> int:
        self._check_open()
        count = self.ffi.new("uint32_t *")
        rc = self.lib.amp_graph_streamer_dump_count(self._streamer, count)
        if int(rc) != 0:
            raise RuntimeError("failed to query dump queue")
        return int(count[0])

    def _pop_dump(self, remaining_frames: int) -> Optional[Tuple[int, np.ndarray]]:
        self._check_open()
        frames_ptr = self.ffi.new("uint32_t *")
        channels_ptr = self.ffi.new("uint32_t *")
        sequence_ptr = self.ffi.new("uint64_t *")
        max_frames = max(remaining_frames, 1)
        channels_guess = self._channels or 1
        buffer = np.empty((self._batches, channels_guess, max_frames), dtype=np.float64)
        rc = self.lib.amp_graph_streamer_pop_dump(
            self._streamer,
            self.ffi.from_buffer("double[]", buffer),
            max_frames,
            frames_ptr,
            channels_ptr,
            sequence_ptr,
        )
        if int(rc) == 1:
            needed = int(frames_ptr[0])
            if needed <= 0:
                return None
            channels_guess = max(int(channels_ptr[0]) or channels_guess, 1)
            buffer = np.empty((self._batches, channels_guess, needed), dtype=np.float64)
            rc = self.lib.amp_graph_streamer_pop_dump(
                self._streamer,
                self.ffi.from_buffer("double[]", buffer),
                needed,
                frames_ptr,
                channels_ptr,
                sequence_ptr,
            )
        if int(rc) != 0:
            raise RuntimeError("failed to pop dump chunk")
        frames = int(frames_ptr[0])
        if frames == 0:
            return None
        channels = max(int(channels_ptr[0]), 1)
        self._channels = channels
        chunk = buffer[:, :channels, :frames].copy()
        return int(sequence_ptr[0]), chunk

    def _read_ring(self, remaining_frames: int) -> Optional[Tuple[int, np.ndarray]]:
        self._check_open()
        available_ptr = self.ffi.new("uint64_t *")
        rc = self.lib.amp_graph_streamer_available(self._streamer, available_ptr)
        if int(rc) != 0:
            raise RuntimeError("failed to query ring buffer")
        available = int(available_ptr[0])
        if available == 0:
            return None
        to_read = min(remaining_frames, available)
        frames_ptr = self.ffi.new("uint32_t *")
        channels_ptr = self.ffi.new("uint32_t *")
        sequence_ptr = self.ffi.new("uint64_t *")
        channels_guess = self._channels or 1
        buffer = np.empty((self._batches, channels_guess, to_read), dtype=np.float64)
        rc = self.lib.amp_graph_streamer_read(
            self._streamer,
            self.ffi.from_buffer("double[]", buffer),
            to_read,
            frames_ptr,
            channels_ptr,
            sequence_ptr,
        )
        if int(rc) != 0:
            raise RuntimeError("failed to read from ring buffer")
        frames = int(frames_ptr[0])
        if frames == 0:
            return None
        channels = max(int(channels_ptr[0]), 1)
        self._channels = channels
        chunk = buffer[:, :channels, :frames].copy()
        return int(sequence_ptr[0]), chunk

    def collect(self, expected_frames: Optional[int] = None) -> np.ndarray:
        self._check_open()
        target = expected_frames if expected_frames is not None else self._total_frames
        pieces: List[Tuple[int, np.ndarray]] = []
        gathered = 0
        while True:
            chunk = self._pop_dump(target - gathered)
            if chunk is None:
                break
            pieces.append(chunk)
            gathered += chunk[1].shape[2]
        ring_chunk = self._read_ring(target - gathered)
        if ring_chunk is not None:
            pieces.append(ring_chunk)
            gathered += ring_chunk[1].shape[2]
        if not pieces:
            return np.zeros((self._batches, self._channels or 0, 0), dtype=np.float64)
        pieces.sort(key=lambda item: item[0])
        data = np.concatenate([arr for _, arr in pieces], axis=2)
        return data


__all__ = [
    "AVAILABLE",
    "UNAVAILABLE_REASON",
    "get_graph_runtime_impl",
    "NativeGraphExecutor",
    "NativeGraphStreamer",
    "SchedulerMode",
]

try:  # Attempt eager load so AVAILABLE reflects the environment
    _load_impl()
except Exception:
    # Leave AVAILABLE/UNAVAILABLE_REASON as set by _load_impl
    pass
