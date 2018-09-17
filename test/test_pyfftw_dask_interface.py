# Copyright 2015 Knowledge Economy Developments Ltd
#
# Henry Gomersall
# heng@kedevelopments.co.uk
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

from pyfftw import interfaces

try:
    interfaces.dask_fft
except AttributeError:
    interfaces.dask_fft = None

try:
    # Kludge to skip dask tests when mkl_fft is present
    import mkl_fft
    interfaces.dask_fft = None
except ImportError:
    pass

from .test_pyfftw_base import run_test_suites
from .test_pyfftw_numpy_interface import complex_dtypes, real_dtypes
from ._get_default_args import get_default_args

from distutils.version import LooseVersion
import unittest
import numpy

if interfaces.dask_fft:
    import dask.array as da
    from dask.array import fft as da_fft
    from dask.array.fft import fft_wrap
else:
    da = None
    da_fft = None
    fft_wrap = None

import warnings
import copy
warnings.filterwarnings('always')

def make_complex_data(shape, dtype):
    ar, ai = dtype(numpy.random.randn(2, *shape))
    ac = ar + 1j*ai

    return da.from_array(ac, chunks=shape)

def make_real_data(shape, dtype):
    ar = dtype(numpy.random.randn(*shape))

    return da.from_array(ar, chunks=shape)

def _dask_array_fft_has_norm_kwarg():
    """returns True if dask.array's fft supports the norm keyword argument
    """

    return False

functions = {
        'fft': 'complex',
        'fft2': 'complex',
        'fftn': 'complex',
        'ifft': 'complex',
        'ifft2': 'complex',
        'ifftn': 'complex',
        'rfft': 'r2c',
        'rfft2': 'r2c',
        'rfftn': 'r2c',
        'irfft': 'c2r',
        'irfft2': 'c2r',
        'irfftn': 'c2r',
        'hfft': 'c2r',
        'ihfft': 'r2c'}

acquired_names = ('fft_wrap', 'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift')

@unittest.skipIf(
    not interfaces.dask_fft,
    "dask interface is not available, so skipping tests."
)
class InterfacesDaskFFTTestModule(unittest.TestCase):
    ''' A really simple test suite to check the module works as expected.
    '''

    def test_acquired_names(self):
        for each_name in acquired_names:

            da_fft_attr = getattr(da_fft, each_name)
            acquired_attr = getattr(interfaces.dask_fft, each_name)

            self.assertIs(da_fft_attr, acquired_attr)


@unittest.skipIf(
    not interfaces.dask_fft,
    "dask interface is not available, so skipping tests."
)
class InterfacesDaskFFTTestFFT(unittest.TestCase):

    io_dtypes = {
            'complex': (complex_dtypes, make_complex_data),
            'r2c': (real_dtypes, make_real_data),
            'c2r': (complex_dtypes, make_complex_data)}

    validator_module = da_fft
    test_wrapped_interface = interfaces.numpy_fft
    test_interface = interfaces.dask_fft
    func = 'fft'
    axes_kw = 'axis'
    default_s_from_shape_slicer = slice(-1, None)

    test_shapes = (
            ((100,), {}),
            ((128, 64), {'axis': 0}),
            ((128, 32), {'axis': -1}),
            ((59, 100), {}),
            ((59, 99), {'axis': -1}),
            ((59, 99), {'axis': 0}),
            ((32, 32, 4), {'axis': 1}),
            ((32, 32, 2), {'axis': 1, 'norm': 'ortho'}),
            ((64, 128, 16), {}),
            )

    realinv = False
    has_norm_kwarg = _dask_array_fft_has_norm_kwarg()

    @property
    def test_data(self):
        for test_shape, kwargs in self.test_shapes:
            axes = self.axes_from_kwargs(kwargs)
            s = self.s_from_kwargs(test_shape, kwargs)

            if not self.has_norm_kwarg and 'norm' in kwargs:
                kwargs.pop('norm')

            if self.realinv:
                test_shape = list(test_shape)
                test_shape[axes[-1]] = test_shape[axes[-1]]//2 + 1
                test_shape = tuple(test_shape)

            yield test_shape, s, kwargs

    def __init__(self, *args, **kwargs):

        super(InterfacesDaskFFTTestFFT, self).__init__(*args, **kwargs)

        # Assume python 3, but keep backwards compatibility
        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def validate(self, array_type, test_shape, dtype,
                 s, kwargs):

        # Do it without the cache

        # without:
        interfaces.cache.disable()
        self._validate(array_type, test_shape, dtype, s, kwargs)

    def munge_input_array(self, array, kwargs):
        return array

    def _validate(self, array_type, test_shape, dtype,
                  s, kwargs):

        input_array = self.munge_input_array(
                array_type(test_shape, dtype), kwargs)

        orig_input_array = input_array

        np_input_array = numpy.asarray(input_array)

        if np_input_array.dtype == 'clongdouble':
            np_input_array = numpy.complex128(input_array)

        elif np_input_array.dtype == 'longdouble':
            np_input_array = numpy.float64(input_array)

        da_input_array = da.from_array(np_input_array,
                                       chunks=np_input_array.shape)


        with warnings.catch_warnings(record=True) as w:
            # We catch the warnings so as to pick up on when
            # a complex array is turned into a real array

            if 'axes' in kwargs:
                validator_kwargs = {'axes': kwargs['axes']}
            elif 'axis' in kwargs:
                validator_kwargs = {'axis': kwargs['axis']}
            else:
                validator_kwargs = {}

            if self.has_norm_kwarg and 'norm' in kwargs:
                validator_kwargs['norm'] = kwargs['norm']

            try:
                test_out_array = getattr(self.validator_module, self.func)(
                        da_input_array, s, **validator_kwargs)

            except Exception as e:
                interface_exception = None
                try:
                    getattr(self.test_interface, self.func)(
                            input_array, s, **kwargs)
                except Exception as _interface_exception:
                    # It's necessary to assign the exception to the
                    # already defined variable in Python 3.
                    # See http://www.python.org/dev/peps/pep-3110/#semantic-changes
                    interface_exception = _interface_exception

                # If the test interface raised, so must this.
                self.assertEqual(type(interface_exception), type(e),
                        msg='Interface exception raised. ' +
                        'Testing for: ' + repr(e))
                return

            output_array = getattr(self.test_interface, self.func)(
                    input_array, s, **kwargs)

            if (functions[self.func] == 'r2c'):
                if numpy.iscomplexobj(input_array):
                    if len(w) > 0:
                        # Make sure a warning is raised
                        self.assertIs(
                                w[-1].category, numpy.ComplexWarning)

        self.assertTrue(
                numpy.allclose(output_array, test_out_array,
                    rtol=1e-2, atol=1e-4))

        if numpy.asanyarray(input_array).real.dtype == numpy.float16:
            # FFTW output will never be single precision for half precision
            # inputs as there is no half-precision FFTW routine
            input_precision_dtype = numpy.float32
        else:
            input_precision_dtype = numpy.asanyarray(input_array).real.dtype

        self.assertEqual(input_precision_dtype,
                output_array.real.dtype)

        self.assertTrue(numpy.allclose(input_array,
                orig_input_array))

        return output_array

    def axes_from_kwargs(self, kwargs):
        default_args = get_default_args(
            getattr(self.test_wrapped_interface, self.func))

        if 'axis' in kwargs:
            axes = (kwargs['axis'],)

        elif 'axes' in kwargs:
            axes = kwargs['axes']
            if axes is None:
                axes = default_args['axes']

        else:
            if 'axis' in default_args:
                # default 1D
                axes = (default_args['axis'],)
            else:
                # default nD
                axes = default_args['axes']

        if axes is None:
            axes = (-1,)

        return axes

    def s_from_kwargs(self, test_shape, kwargs):
        ''' Return either a scalar s or a tuple depending on
        whether axis or axes is specified
        '''
        default_args = get_default_args(
            getattr(self.test_wrapped_interface, self.func))

        if 'axis' in kwargs:
            s = test_shape[kwargs['axis']]

        elif 'axes' in kwargs:
            axes = kwargs['axes']
            if axes is not None:
                s = []
                for each_axis in axes:
                    s.append(test_shape[each_axis])
            else:
                # default nD
                s = []
                try:
                    for each_axis in default_args['axes']:
                        s.append(test_shape[each_axis])
                except TypeError:
                    try:
                        s = list(test_shape[
                            self.default_s_from_shape_slicer])
                    except TypeError:
                        # We had an integer as the default, so force
                        # it to be a list
                        s = [test_shape[self.default_s_from_shape_slicer]]

        else:
            if 'axis' in default_args:
                # default 1D
                s = test_shape[default_args['axis']]
            else:
                # default nD
                s = []
                try:
                    for each_axis in default_args['axes']:
                        s.append(test_shape[each_axis])
                except TypeError:
                    s = None

        return s

    def test_valid(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]

        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                s = None

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)


    def test_same_sized_s(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)

    def test_bigger_s(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                try:
                    for each_axis, length in enumerate(s):
                        s[each_axis] += 2
                except TypeError:
                    s += 2

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)


    def test_smaller_s(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                try:
                    for each_axis, length in enumerate(s):
                        s[each_axis] -= 2
                except TypeError:
                    s -= 2

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)

    def check_arg(self, arg, arg_test_values, array_type, test_shape,
            dtype, s, kwargs):
        '''Check that the correct arg is passed to the builder'''
        # We trust the builders to work as expected when passed
        # the correct arg (the builders have their own unittests).

        return_values = []
        input_array = array_type(test_shape, dtype)

        def fake_fft(*args, **kwargs):
            return_values.append((args, kwargs))
            return (args, kwargs)

        try:

            # Replace the function that is to be used
            real_fft = getattr(self.test_interface, self.func)
            setattr(self.test_interface, self.func, fake_fft)

            _kwargs = kwargs.copy()

            for each_value in arg_test_values:
                _kwargs[arg] = each_value
                builder_args = getattr(self.test_interface, self.func)(
                input_array.copy(), s, **_kwargs)

                self.assertTrue(builder_args[1][arg] == each_value)

            # make sure it was called
            self.assertTrue(len(return_values) > 0)
        except:
            raise

        finally:
            # Make sure we set it back
            setattr(self.test_interface, self.func, real_fft)

        # Validate it aswell
        for each_value in arg_test_values:
            _kwargs[arg] = each_value
            builder_args = getattr(self.test_interface, self.func)(
            input_array.copy(), s, **_kwargs)

            self.validate(array_type, test_shape, dtype, s, _kwargs)


    def test_bigger_and_smaller_s(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            i = -1
            for test_shape, s, kwargs in self.test_data:

                try:
                    for each_axis, length in enumerate(s):
                        s[each_axis] += i * 2
                        i *= i
                except TypeError:
                    s += i * 2
                    i *= i

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)


    def test_dtype_coercian(self):
        # Make sure we input a dtype that needs to be coerced
        if functions[self.func] == 'r2c':
            dtype_tuple = self.io_dtypes['complex']
        else:
            dtype_tuple = self.io_dtypes['r2c']

        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                s = None

                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)


    def test_input_maintained(self):
        '''Test to make sure the input is maintained by default.
        '''
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                input_array = dtype_tuple[1](test_shape, dtype)

                orig_input_array = input_array.copy()

                getattr(self.test_interface, self.func)(
                        input_array, s, **kwargs)

                self.assertTrue(
                        numpy.alltrue(input_array == orig_input_array))


class InterfacesDaskFFTTestFFT2(InterfacesDaskFFTTestFFT):
    axes_kw = 'axes'
    func = 'ifft2'
    has_norm_kwarg = False
    test_shapes = (
            ((128, 64), {'axes': None}),
            ((128, 32), {'axes': None}),
            ((128, 32, 4), {'axes': (0, 2)}),
            ((59, 100), {'axes': (-2, -1)}),
            ((32, 32), {'axes': (-2, -1), 'norm': 'ortho'}),
            ((64, 128, 16), {'axes': (0, 2)}),
            ((4, 6, 8, 4), {'axes': (0, 3)}),
            )

    invalid_args = ()

    def test_shape_and_s_different_lengths(self):
        dtype_tuple = self.io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, _kwargs in self.test_data:
                kwargs = copy.copy(_kwargs)
                try:
                    s = s[1:]
                except TypeError:
                    self.skipTest('Not meaningful test on 1d arrays.')

                # Convert empty tuples to None
                s = s if s else None

                del kwargs['axes']
                self.validate(dtype_tuple[1],
                        test_shape, dtype, s, kwargs)


class InterfacesDaskFFTTestFFTN(InterfacesDaskFFTTestFFT2):
    func = 'ifftn'
    has_norm_kwarg = False
    test_shapes = (
            ((128, 32, 4), {'axes': None}),
            ((64, 128, 16), {'axes': (0, 1, 2)}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1)}),
            ((4, 6, 4, 4), {'axes': (0, 3, 1), 'norm': 'ortho'}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1, 2)}),
            )


class InterfacesDaskFFTTestIFFT(InterfacesDaskFFTTestFFT):
    func = 'ifft'
    has_norm_kwarg = False

class InterfacesDaskFFTTestIFFT2(InterfacesDaskFFTTestFFT2):
    func = 'ifft2'
    has_norm_kwarg = False

class InterfacesDaskFFTTestIFFTN(InterfacesDaskFFTTestFFTN):
    func = 'ifftn'
    has_norm_kwarg = False

class InterfacesDaskFFTTestRFFT(InterfacesDaskFFTTestFFT):
    func = 'rfft'
    has_norm_kwarg = False

class InterfacesDaskFFTTestRFFT2(InterfacesDaskFFTTestFFT2):
    func = 'rfft2'
    has_norm_kwarg = False

class InterfacesDaskFFTTestRFFTN(InterfacesDaskFFTTestFFTN):
    func = 'rfftn'
    has_norm_kwarg = False

class InterfacesDaskFFTTestIRFFT(InterfacesDaskFFTTestFFT):
    func = 'irfft'
    realinv = True
    has_norm_kwarg = False

class InterfacesDaskFFTTestIRFFT2(InterfacesDaskFFTTestFFT2):
    func = 'irfft2'
    has_norm_kwarg = False
    realinv = True

class InterfacesDaskFFTTestIRFFTN(InterfacesDaskFFTTestFFTN):
    func = 'irfftn'
    has_norm_kwarg = False
    realinv = True

class InterfacesDaskFFTTestHFFT(InterfacesDaskFFTTestFFT):
    func = 'hfft'
    realinv = True
    has_norm_kwarg = False

class InterfacesDaskFFTTestIHFFT(InterfacesDaskFFTTestFFT):
    func = 'ihfft'
    has_norm_kwarg = False

test_cases = (
        InterfacesDaskFFTTestModule,
        InterfacesDaskFFTTestFFT,
        InterfacesDaskFFTTestFFT2,
        InterfacesDaskFFTTestFFTN,
        InterfacesDaskFFTTestIFFT,
        InterfacesDaskFFTTestIFFT2,
        InterfacesDaskFFTTestIFFTN,
        InterfacesDaskFFTTestRFFT,
        InterfacesDaskFFTTestRFFT2,
        InterfacesDaskFFTTestRFFTN,
        InterfacesDaskFFTTestIRFFT,
        InterfacesDaskFFTTestIRFFT2,
        InterfacesDaskFFTTestIRFFTN,
        InterfacesDaskFFTTestHFFT,
        InterfacesDaskFFTTestIHFFT)

test_set = None

if __name__ == '__main__':
    run_test_suites(test_cases, test_set)
