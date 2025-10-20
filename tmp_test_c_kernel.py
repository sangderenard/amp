import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from amp import c_kernels
rng = np.random.RandomState(123)
x = (rng.rand(2,128).astype('float64')*2-1)
r = 0.95
alpha = 0.05
z0 = np.zeros(2,dtype='float64')
try:
    out_c = c_kernels.lfo_slew_c(x, r, alpha, z0.copy())
    print('C called ok, out_c.shape=', out_c.shape)
except Exception as e:
    print('C call failed:', e)
    out_c = None
out_py = c_kernels.lfo_slew_py(x, r, alpha, z0.copy())
print('py out shape=', out_py.shape)
if out_c is not None:
    import numpy as _np
    print('max abs diff:', float(_np.max(_np.abs(out_c - out_py))))
    print('allclose (1e-12):', bool(_np.allclose(out_c, out_py, rtol=1e-12, atol=1e-12)))
