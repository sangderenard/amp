"""Build c_kernels cffi extension into a persistent binary.

Usage:
    python scripts/build_c_kernels.py

This will create a compiled extension `_amp_ckernels_cffi.*` in the project root.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path('src').resolve()))
from amp import c_kernels

print('Attempting to compile C kernels (if not already compiled)')
# The module compiled lazily on import; importing should have attempted compilation
print('C_AVAILABLE =', getattr(c_kernels, 'AVAILABLE', False))

if not getattr(c_kernels, 'AVAILABLE', False):
    print('Attempting to re-import to trigger build...')
    import importlib
    try:
        importlib.reload(c_kernels)
        print('Re-import done; AVAILABLE =', getattr(c_kernels, 'AVAILABLE', False))
    except Exception as e:
        print('Re-import failed:', e)

print('Done')
