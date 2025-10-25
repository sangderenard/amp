from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .utils import lanczos_resample

AVAILABLE = False
_IMPL = None
UNAVAILABLE_REASON: str | None = None

_DIAGNOSTIC_BUILD = os.environ.get("AMP_NATIVE_DIAGNOSTICS_BUILD", "")
_LOGGING_ENABLED = _DIAGNOSTIC_BUILD.lower() in ("1", "true", "yes", "on")

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
    candidates = [build_dir / _library_name(), _native_root() / _library_name()]
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

    subprocess.run(cmake_args, check=True, cwd=native_dir)

    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
    if force and sys.platform != "win32":
        build_cmd.append("--clean-first")
    subprocess.run(build_cmd, check=True, cwd=native_dir)

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
    float reserved[6];
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
"""
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

    def __enter__(self) -> "NativeGraphExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_runtime", self.ffi.NULL) not in (None, self.ffi.NULL):
            self.lib.amp_graph_runtime_destroy(self._runtime)
            self._runtime = self.ffi.NULL

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

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
            self.lib.amp_graph_runtime_clear_params(self._runtime)
            self.lib.amp_graph_runtime_configure(self._runtime, batches, dsp_frames)
            if self._set_dsp_sample_rate is not None:
                self._set_dsp_sample_rate(self._runtime, float(dsp_rate))
            keepalive: list[np.ndarray] = []
            if base_params:
                for node_name, params in base_params.items():
                    if node_name.startswith("_"):
                        continue
                    for param_name, array in params.items():
                        arr = np.asarray(array, dtype=np.float64)
                        if arr.ndim != 3:
                            raise ValueError(
                                f"param '{param_name}' for node '{node_name}' must be BxCxF"
                            )
                        arr_c = np.require(arr, requirements=("C",))
                        keepalive.append(arr_c)
                        ptr = self.ffi.from_buffer("double[]", arr_c)
                        status = self.lib.amp_graph_runtime_set_param(
                            self._runtime,
                            node_name.encode("utf-8"),
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
            ctrl_blob = control_history_blob or b""
            ctrl_buf = self.ffi.new("uint8_t[]", ctrl_blob) if ctrl_blob else self.ffi.NULL
            out_ptr = self.ffi.new("double **")
            out_batches = self.ffi.new("uint32_t *")
            out_channels = self.ffi.new("uint32_t *")
            out_frames = self.ffi.new("uint32_t *")
            status = self.lib.amp_graph_runtime_execute(
                self._runtime,
                ctrl_buf if ctrl_blob else self.ffi.NULL,
                len(ctrl_blob),
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
            total = int(out_batches[0]) * int(out_channels[0]) * int(out_frames[0])
            buffer = self.ffi.buffer(out_ptr[0], total * np.dtype(np.float64).itemsize)
            array = np.frombuffer(buffer, dtype=np.float64).copy().reshape(
                int(out_batches[0]), int(out_channels[0]), int(out_frames[0])
            )
        if int(out_frames[0]) != output_frames or not np.isclose(dsp_rate, audio_rate):
            array = lanczos_resample(array, dsp_rate, audio_rate, output_frames)
        return np.nan_to_num(array, copy=False)


__all__ = [
    "AVAILABLE",
    "UNAVAILABLE_REASON",
    "get_graph_runtime_impl",
    "NativeGraphExecutor",
]

try:  # Attempt eager load so AVAILABLE reflects the environment
    _load_impl()
except Exception:
    # Leave AVAILABLE/UNAVAILABLE_REASON as set by _load_impl
    pass
