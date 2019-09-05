# Copyright 2019, The pyFFTW developers
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import pyfftw
import numpy

try:
    import scipy
    scipy_version = scipy.__version__
except ImportError:
    scipy_version = '0.0.0'

from distutils.version import LooseVersion
has_scipy_fft = LooseVersion(scipy_version) >= LooseVersion('1.4.0')

if has_scipy_fft:
    import scipy.fft
    import scipy.signal
    from pyfftw.interfaces import scipy_fft

import unittest
from .test_pyfftw_base import run_test_suites, miss
from . import test_pyfftw_numpy_interface

'''pyfftw.interfaces.scipy_fft just wraps pyfftw.interfaces.numpy_fft.

All the tests here just check that the call is made correctly.
'''

funcs = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
         'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
         'hfft', 'ihfft')

acquired_names = ('dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
                  'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
                  'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq')

def make_complex_data(shape, dtype):
    ar, ai = dtype(numpy.random.randn(2, *shape))
    return ar + 1j*ai

def make_r2c_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

def make_c2r_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

# reuse from numpy tests
make_complex_data = test_pyfftw_numpy_interface.make_complex_data
complex_dtypes = test_pyfftw_numpy_interface.complex_dtypes
real_dtypes = test_pyfftw_numpy_interface.real_dtypes

io_dtypes = {
        'complex': (complex_dtypes, make_complex_data),
        'r2c': (real_dtypes, make_r2c_real_data),
        'c2r': (real_dtypes, make_c2r_real_data)}

@unittest.skipIf(not has_scipy_fft, 'scipy.fft is unavailable')
class InterfacesScipyFFTTestSimple(unittest.TestCase):
    ''' A simple test suite for a simple implementation.
    '''

    @unittest.skipIf(*miss('64'))
    def test_scipy_backend(self):
        a = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)
        b = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)

        a[:] = (numpy.random.randn(*a.shape) +
                1j*numpy.random.randn(*a.shape))
        b[:] = (numpy.random.randn(*b.shape) +
                1j*numpy.random.randn(*b.shape))

        scipy_c = scipy.signal.fftconvolve(a, b)

        with scipy.fft.set_backend(scipy_fft, only=True):
            scipy_replaced_c = scipy.signal.fftconvolve(a, b)

        self.assertTrue(numpy.allclose(scipy_c, scipy_replaced_c))

    def test_acquired_names(self):
        for each_name in acquired_names:

            fft_attr = getattr(scipy.fft, each_name)
            acquired_attr = getattr(scipy_fft, each_name)

            self.assertIs(fft_attr, acquired_attr)


# Construct all the test classes automatically.
test_cases = []
for each_func in funcs:
    class_name = 'InterfacesScipyFFTTest' + each_func.upper()

    parent_class_name = 'InterfacesNumpyFFTTest' + each_func.upper()
    parent_class = getattr(test_pyfftw_numpy_interface, parent_class_name)

    class_dict = {'validator_module': scipy.fft if has_scipy_fft else None,
                  'test_interface': scipy_fft if has_scipy_fft else None,
                  'io_dtypes': io_dtypes,
                  'overwrite_input_flag': 'overwrite_x',
                  'default_s_from_shape_slicer': slice(None),
                  'threads_arg_name': 'workers'}

    cls = type(class_name, (parent_class,), class_dict)
    cls = unittest.skipIf(not has_scipy_fft, "scipy.fft is not available")(cls)

    globals()[class_name] = cls
    test_cases.append(cls)

test_cases.append(InterfacesScipyFFTTestSimple)
test_set = None


if __name__ == '__main__':
    run_test_suites(test_cases, test_set)
