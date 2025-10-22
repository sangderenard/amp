from cffi import FFI
import os

def build_test_module():
    ffi = FFI()
    ffi.cdef("""
    void test_run_graph(const char *control_blob, unsigned long control_size, const char *node_blob, unsigned long node_size, double *out_buf, unsigned int batches, unsigned int channels, unsigned int frames);
    """)
    this_dir = os.path.dirname(__file__)
    c_file = os.path.join(this_dir, 'c_test.c')
    with open(c_file, 'r', encoding='utf-8') as f:
        source = f.read()
    # Build a module named _amp_c_test in the package directory
    module_name = "_amp_c_test"
    ffi.set_source(module_name, source)
    ffi.compile(verbose=False)
    # import the compiled module
    import importlib
    mod = importlib.import_module(module_name)
    return ffi, mod.lib

def get_test_lib():
    # Try importing a prebuilt module using importlib to avoid static-import warnings
    try:
        import importlib

        _mod = importlib.import_module("_amp_c_test")
        from cffi import FFI as _FFI

        return (_FFI(), _mod.lib)
    except Exception:
        # If not present, attempt to build the test module in-place. This will
        # require a working C toolchain and Python headers on the host system.
        try:
            return build_test_module()
        except Exception:
            return None
