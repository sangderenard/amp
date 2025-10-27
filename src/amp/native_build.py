from __future__ import annotations

"""Helpers for configuring the native AMP toolchain.

This module centralises detection of compiler binaries, logging flags, and
shared build arguments so both the CMake and cffi entry points make consistent
choices.  It prefers explicit environment configuration but falls back to
probing the local PATH for suitable compilers when nothing is configured.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
import shlex
import shutil
import sys
from typing import Iterable, Mapping

_LOGGING_ENV = "AMP_NATIVE_DIAGNOSTICS_BUILD"
_EXTRA_COMPILE_ENV = "AMP_NATIVE_EXTRA_COMPILE_ARGS"
_EXTRA_LINK_ENV = "AMP_NATIVE_EXTRA_LINK_ARGS"
_CC_OVERRIDE_ENV = "AMP_NATIVE_CC"
_CXX_OVERRIDE_ENV = "AMP_NATIVE_CXX"


@dataclass(frozen=True)
class NativeBuildConfig:
    """Canonical toolchain configuration shared across build paths."""

    compile_args: tuple[str, ...]
    link_args: tuple[str, ...]
    logging_enabled: bool
    c_compiler: str | None
    cxx_compiler: str | None


def _parse_extra_args(env_var: str) -> tuple[str, ...]:
    value = os.environ.get(env_var, "").strip()
    if not value:
        return ()
    return tuple(shlex.split(value))


def _candidate_compilers() -> tuple[tuple[str, ...], tuple[str, ...]]:
    if sys.platform == "win32":
        # Windows primarily relies on MSVC. If users need clang-cl or mingw,
        # they can provide explicit overrides via AMP_NATIVE_CC/CXX.
        return (("cl",), ("cl",))
    # POSIX – try canonical C first, then specific vendors.
    return (("cc", "clang", "gcc"), ("c++", "clang++", "g++"))


def _select_compiler(
    env_name: str,
    override_env: str,
    candidates: Iterable[str],
) -> str | None:
    override = os.environ.get(override_env, "").strip()
    if override:
        return override
    existing = os.environ.get(env_name, "").strip()
    if existing:
        return existing
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return None


@lru_cache(maxsize=1)
def get_build_config() -> NativeBuildConfig:
    logging_enabled = os.environ.get(_LOGGING_ENV, "").lower() in {"1", "true", "yes", "on"}

    compile_args: list[str] = []
    if logging_enabled:
        compile_args.append("-DAMP_NATIVE_ENABLE_LOGGING")
    if sys.platform == "win32":
        compile_args.extend(["/std:c++17", "/TP"])
    else:
        compile_args.extend(["-std=c++17", "-pthread"])

    extra_compile = list(_parse_extra_args(_EXTRA_COMPILE_ENV))
    compile_args.extend(extra_compile)

    link_args: list[str] = []
    if sys.platform != "win32":
        link_args.append("-pthread")
    link_args.extend(_parse_extra_args(_EXTRA_LINK_ENV))

    c_candidates, cxx_candidates = _candidate_compilers()
    c_compiler = _select_compiler("CC", _CC_OVERRIDE_ENV, c_candidates)
    cxx_compiler = _select_compiler("CXX", _CXX_OVERRIDE_ENV, cxx_candidates)

    return NativeBuildConfig(
        compile_args=tuple(compile_args),
        link_args=tuple(link_args),
        logging_enabled=logging_enabled,
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
    )


def ensure_toolchain_env() -> None:
    """Guarantee that CC/CXX are exported for subprocess consumers."""

    config = get_build_config()
    if config.c_compiler and not os.environ.get("CC"):
        os.environ["CC"] = config.c_compiler
    if config.cxx_compiler and not os.environ.get("CXX"):
        os.environ["CXX"] = config.cxx_compiler


def command_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment mapping containing detected compiler overrides."""

    config = get_build_config()
    env = dict(os.environ if base is None else base)
    if config.c_compiler:
        env.setdefault("CC", config.c_compiler)
    if config.cxx_compiler:
        env.setdefault("CXX", config.cxx_compiler)
    return env
