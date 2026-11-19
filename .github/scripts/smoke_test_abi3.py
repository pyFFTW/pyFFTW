"""Basic checks of an installed stable ABI (abi3) pyfftw wheel.

Meant to be run with a Python version different from the one used to build the
wheel (e.g. 3.13, 3.14 or 3.15 for a cp312-abi3 wheel), from a directory where
the local ``pyfftw`` source package can not shadow the installed one.
"""

import os
import sys
import sysconfig
from pathlib import Path

import numpy as np

import pyfftw
import pyfftw.interfaces.numpy_fft as fft_interface

print(f"Python {sys.version}")
print(f"pyfftw {pyfftw.__version__} from {pyfftw.__file__}")

# 1. The installed extension has to be the stable ABI one, not a
#    version-specific build that would have been compiled from source.
#    abi3 extensions are named pyfftw.abi3.so on Linux and macOS, but
#    simply pyfftw.pyd on Windows.
ext_files = sorted(
    p.name
    for p in Path(pyfftw.__file__).parent.glob("pyfftw.*")
    if p.suffix in (".so", ".pyd")
)
print("extension modules:", ext_files)
assert len(ext_files) == 1, "expected exactly one pyfftw extension module"
if sys.platform == "win32":
    assert ext_files[0] == "pyfftw.pyd", ext_files
else:
    assert ext_files[0].endswith(".abi3.so"), ext_files
assert "cpython-" not in ext_files[0], "a version-specific extension module was found"

# 2. We really test another Python version than the one used for the build.
build_version = os.environ.get("PYFFTW_ABI3_BUILD_PYTHON", "3.12")
running = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"wheel built with Python {build_version}, running Python {running}")
assert running != build_version, "this test should not run with the build Python"
assert not sysconfig.get_config_var("Py_GIL_DISABLED"), (
    "abi3 wheels are not compatible with free-threaded builds"
)

rng = np.random.default_rng(0)

# 3. Low level builders API, complex and real transforms, several precisions.
for dtype in ("complex64", "complex128", "clongdouble"):
    a = pyfftw.empty_aligned((8, 16), dtype=dtype)
    a[:] = rng.random(a.shape) + 1j * rng.random(a.shape)
    fft = pyfftw.builders.fft2(a)
    res = fft()
    tol = 1e-4 if dtype == "complex64" else 1e-10
    np.testing.assert_allclose(res, np.fft.fft2(a), rtol=tol, atol=tol)
    ifft = pyfftw.builders.ifft2(res)
    np.testing.assert_allclose(ifft(), a, rtol=tol, atol=tol)

x = pyfftw.empty_aligned(1024, dtype="float64")
x[:] = rng.random(x.shape)
rfft = pyfftw.builders.rfft(x)
np.testing.assert_allclose(rfft(), np.fft.rfft(x), rtol=1e-10, atol=1e-10)

# 4. Class API: FFTW object, update_arrays and normalization.
in_arr = pyfftw.empty_aligned(64, dtype="complex128")
out_arr = pyfftw.empty_aligned(64, dtype="complex128")
plan = pyfftw.FFTW(in_arr, out_arr)
in_arr[:] = rng.random(64)
plan()
np.testing.assert_allclose(out_arr, np.fft.fft(in_arr), rtol=1e-10, atol=1e-10)
new_in = pyfftw.empty_aligned(64, dtype="complex128")
new_in[:] = rng.random(64)
plan.update_arrays(new_in, out_arr)
plan()
np.testing.assert_allclose(out_arr, np.fft.fft(new_in), rtol=1e-10, atol=1e-10)

# 5. Multithreading (uses the libfftw3_threads symbols of the bundled lib).
y = pyfftw.empty_aligned((64, 64), dtype="complex128")
y[:] = rng.random(y.shape)
fft_mt = pyfftw.builders.fftn(y, threads=2)
np.testing.assert_allclose(fft_mt(), np.fft.fftn(y), rtol=1e-10, atol=1e-10)

# 6. Drop-in numpy interface (with its cache).
z = rng.random((32, 32))
np.testing.assert_allclose(
    fft_interface.fft2(z), np.fft.fft2(z), rtol=1e-10, atol=1e-10
)
np.testing.assert_allclose(
    fft_interface.rfftn(z), np.fft.rfftn(z), rtol=1e-10, atol=1e-10
)

# 7. Utilities implemented in the extension module.
a = pyfftw.simd_alignment
assert a >= 1 and (a & (a - 1)) == 0, a  # positive power of two
assert pyfftw.is_n_byte_aligned(pyfftw.empty_aligned(10, dtype="float64", n=a), a)
assert pyfftw.next_fast_len(1031) >= 1031
wisdom = pyfftw.export_wisdom()
assert len(wisdom) == 3
pyfftw.forget_wisdom()

# 8. Errors raised from the extension are properly propagated.
try:
    pyfftw.FFTW(in_arr, pyfftw.empty_aligned(32, dtype="complex128"))
except ValueError:
    pass
else:
    raise AssertionError("FFTW should reject arrays with incompatible shapes")

print("abi3 smoke test OK")
