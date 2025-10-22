"""Module entry point."""

from __future__ import annotations

from .cli import main

# Enable faulthandler early so fatal native crashes (SIGSEGV/etc) produce a
# traceback written to disk. This helps diagnose crashes that bypass Python
# exception machinery.
try:
    import faulthandler, os
    os.makedirs("logs", exist_ok=True)
    faulthandler_log = open(os.path.join("logs", "faulthandler.log"), "a")
    faulthandler.enable(faulthandler_log)
except Exception:
    try:
        # Best-effort to enable default faulthandler output
        import faulthandler as _fh
        _fh.enable()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        rv = main()
        try:
            # Best-effort debug output to show the exit code when the
            # application returns; helps diagnose silent exits.
            print(f"[__main__] main() returned: {rv!r}")
        except Exception:
            pass
        raise SystemExit(rv)
    except SystemExit as se:
        # Write a synchronous diagnostic for SystemExit so we capture
        # exits that are raised before the normal return path.
        try:
            import sys as _sys, os as _os, traceback as _tb
            text = f"[__main__] SystemExit raised with code={se.code!r}\n"
            try:
                _sys.stderr.write(text)
                _sys.stderr.flush()
            except Exception:
                pass
            try:
                _os.makedirs("logs", exist_ok=True)
                with open(_os.path.join("logs", "exit_diagnostics.log"), "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        except Exception:
            pass
        raise
    except Exception:
        try:
            import sys as _sys, os as _os, traceback as _tb
            text = "[__main__] uncaught exception:\n" + _tb.format_exc()
            try:
                _sys.stderr.write(text)
                _sys.stderr.flush()
            except Exception:
                pass
            try:
                _os.makedirs("logs", exist_ok=True)
                with open(_os.path.join("logs", "exit_diagnostics.log"), "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        except Exception:
            pass
        raise
