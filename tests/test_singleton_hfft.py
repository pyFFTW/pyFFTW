import numpy as np
import pytest

from pyfftw.interfaces import numpy_fft, scipy_fft


@pytest.mark.parametrize("interface", [numpy_fft, scipy_fft])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.complex128])
@pytest.mark.parametrize("n", [1, 2, 5, 16])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_singleton_hfft_with_explicit_length(interface, dtype, n, norm):
    data = np.array([2], dtype=dtype)
    original = data.copy()

    actual = interface.hfft(data, n=n, norm=norm)

    np.testing.assert_allclose(actual, np.fft.hfft(data, n=n, norm=norm), rtol=1e-6)
    np.testing.assert_array_equal(data, original)


def test_singleton_hfft_without_length_is_invalid():
    with pytest.raises(ValueError):
        numpy_fft.hfft(np.array([2.0]))
