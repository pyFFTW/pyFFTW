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

from packaging.version import Version
has_scipy_fft = Version(scipy_version) >= Version('1.4.0')

if has_scipy_fft:
    import scipy.fft
    import scipy.signal
    from pyfftw.interfaces import scipy_fft

import unittest
from pyfftw import _supported_types
from .test_pyfftw_base import run_test_suites, miss
from . import test_pyfftw_numpy_interface

'''pyfftw.interfaces.scipy_fft just wraps pyfftw.interfaces.numpy_fft.

All the tests here just check that the call is made correctly.
'''

funcs = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
         'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
         'hfft', 'ihfft')

acquired_names = ('hfft2', 'ihfft2', 'hfftn', 'ihfftn',
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


# InterfacesScipyR2RFFTTest is mostly the same as the ones defined in
# test_pyfftw_scipy_interface.py, but call the functions from scipy.fft instead
# of scipy.fftpack and test additional normalization modes.

if '64' in _supported_types:
    default_floating_type = numpy.float64
elif '32' in _supported_types:
    default_floating_type = numpy.float32
elif 'ld' in _supported_types:
    default_floating_type = numpy.longdouble
atol_dict = dict(f=1e-5, d=1e-7, g=1e-7)
rtol_dict = dict(f=1e-4, d=1e-5, g=1e-5)
transform_types = [1, 2, 3, 4]

if Version(scipy_version) >= Version('1.6.0'):
    # all norm options aside from None
    scipy_norms = [None, 'ortho', 'forward', 'backward']
else:
    scipy_norms = [None, 'ortho']


@unittest.skipIf(not has_scipy_fft, 'scipy.fft is unavailable')
class InterfacesScipyR2RFFTTest(unittest.TestCase):
    ''' Class template for building the scipy real to real tests.
    '''

    # unittest is not very smart and will always turn this class into a test,
    # even though it is not on the list. Hence mark test-dependent values as
    # constants (so this particular test ends up being run twice).
    func_name = 'dct'
    float_type = default_floating_type
    atol = atol_dict['f']
    rtol = rtol_dict['f']

    def setUp(self):
        self.scipy_func = getattr(scipy.fft, self.func_name)
        self.pyfftw_func = getattr(scipy_fft, self.func_name)
        self.ndims = numpy.random.randint(1, high=3)
        self.axis = numpy.random.randint(0, high=self.ndims)
        self.shape = numpy.random.randint(2, high=10, size=self.ndims)
        self.data = numpy.random.rand(*self.shape).astype(self.float_type)
        self.data_copy = self.data.copy()

        if self.func_name in ['dctn', 'idctn', 'dstn', 'idstn']:
            self.kwargs = dict(axes=(self.axis, ))
        else:
            self.kwargs = dict(axis=self.axis)

    def test_unnormalized(self):
        '''Test unnormalized pyfftw transformations against their scipy
        equivalents.
        '''
        for transform_type in transform_types:
            data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                          overwrite_x=False, **self.kwargs)
            self.assertEqual(numpy.linalg.norm(self.data - self.data_copy), 0.0)
            data_hat_s = self.scipy_func(self.data, type=transform_type,
                                         overwrite_x=False, **self.kwargs)
            self.assertTrue(numpy.allclose(data_hat_p, data_hat_s,
                                           atol=self.atol, rtol=self.rtol))

    def test_normalized(self):
        '''Test normalized against scipy results. Note that scipy does
        not support normalization for all transformations.
        '''
        for norm in scipy_norms:
            for transform_type in transform_types:
                data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                              norm=norm,
                                              overwrite_x=False, **self.kwargs)
                if norm == 'ortho':
                    self.assertEqual(
                        numpy.linalg.norm(self.data - self.data_copy), 0.0
                    )
                data_hat_s = self.scipy_func(self.data, type=transform_type,
                                             norm=norm,
                                             overwrite_x=False, **self.kwargs)

                if not numpy.allclose(data_hat_p, data_hat_s, atol=self.atol, rtol=self.rtol):
                    info = f"{self.func_name=}, {norm=}, {transform_type=}"
                    print(info)
                    print("scipy: ", data_hat_s)
                    print("ratio:  ", data_hat_p / data_hat_s)
                    raise ValueError(info)

    def test_normalization_inverses(self):
        '''Test normalization in all of the pyfftw scipy wrappers.
        '''
        for transform_type in transform_types:
            inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[transform_type]
            forward = self.pyfftw_func(self.data, type=transform_type,
                                       norm='ortho',
                                       overwrite_x=False, **self.kwargs)
            result = self.pyfftw_func(forward, type=inverse_type,
                                      norm='ortho',
                                      overwrite_x=False, **self.kwargs)
            if not numpy.allclose(self.data, result, atol=self.atol, rtol=self.rtol):
                info = f"{self.func_name=}, norm='ortho', {transform_type=}"
                print(info)
                print("ratio: ", result / self.data)
                raise ValueError(info)


@unittest.skipIf(not has_scipy_fft, 'scipy.fft is unavailable')
class InterfacesScipyR2RFFTNTest(InterfacesScipyR2RFFTTest):
    ''' Class template for building the scipy real to real tests.
    '''

    # unittest is not very smart and will always turn this class into a test,
    # even though it is not on the list. Hence mark test-dependent values as
    # constants (so this particular test ends up being run twice).
    func_name = 'dctn'
    float_type = default_floating_type
    atol = atol_dict['f']
    rtol = rtol_dict['f']

    def setUp(self):
        self.scipy_func = getattr(scipy.fft, self.func_name)
        self.pyfftw_func = getattr(scipy_fft, self.func_name)
        self.ndims = numpy.random.randint(1, high=3)
        self.shape = numpy.random.randint(2, high=10, size=self.ndims)
        self.data = numpy.random.rand(*self.shape).astype(self.float_type)
        self.data_copy = self.data.copy()
        # random subset of axes
        self.axes = tuple(range(0, numpy.random.randint(0, high=self.ndims)))
        self.kwargs = dict(axes=self.axes)

    def test_axes_none(self):
        '''Test transformation over all axes.
        '''
        for transform_type in transform_types:
            data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                          overwrite_x=False, axes=None)
            self.assertEqual(numpy.linalg.norm(self.data - self.data_copy), 0.0)
            data_hat_s = self.scipy_func(self.data, type=transform_type,
                                         overwrite_x=False, axes=None)
            self.assertTrue(numpy.allclose(data_hat_p, data_hat_s,
                                           atol=self.atol, rtol=self.rtol))

    def test_axes_scalar(self):
        '''Test transformation over a single, scalar axis.
        '''
        for transform_type in transform_types:
            data_hat_p = self.pyfftw_func(self.data, type=transform_type,
                                          overwrite_x=False, axes=-1)
            self.assertEqual(numpy.linalg.norm(self.data - self.data_copy), 0.0)
            data_hat_s = self.scipy_func(self.data, type=transform_type,
                                         overwrite_x=False, axes=-1)
            self.assertTrue(numpy.allclose(data_hat_p, data_hat_s,
                                           atol=self.atol, rtol=self.rtol))


# Construct all the test classes automatically.
test_cases = []
# Construct the r2r test classes.
for floating_type, floating_name in [[numpy.float32, 'Float32'],
                                     [numpy.float64, 'Float64']]:
    if floating_type == numpy.float32 and '32' not in _supported_types:
        # skip single precision tests if library is unavailable
        continue
    elif floating_type == numpy.float64 and '64' not in _supported_types:
        # skip double precision tests if library is unavailable
        continue

    real_transforms = ('dct', 'idct', 'dst', 'idst')
    real_transforms_nd = ('dctn', 'idctn', 'dstn', 'idstn')
    real_transforms += real_transforms_nd

    dt_char = numpy.dtype(floating_type).char
    atol = atol_dict[dt_char]
    rtol = rtol_dict[dt_char]

    # test-cases where only one axis is transformed
    for transform_name in real_transforms:
        class_name = ('InterfacesScipyR2RFFTTest' + transform_name.upper() +
                      floating_name)

        globals()[class_name] = type(
            class_name,
            (InterfacesScipyR2RFFTTest,),
            {'func_name': transform_name,
             'float_type': floating_type,
             'atol': atol,
             'rtol': rtol})

        test_cases.append(globals()[class_name])

    # n-dimensional test-cases
    for transform_name in real_transforms_nd:
        class_name = ('InterfacesScipyR2RFFTNTest' + transform_name.upper() +
                      floating_name)

        globals()[class_name] = type(
            class_name,
            (InterfacesScipyR2RFFTNTest,),
            {'func_name': transform_name,
             'float_type': floating_type,
             'atol': atol,
             'rtol': rtol})

        test_cases.append(globals()[class_name])

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
