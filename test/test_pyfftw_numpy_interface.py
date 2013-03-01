# Copyright 2012 Knowledge Economy Developments Ltd
# 
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pyfftw import n_byte_align_empty, n_byte_align, interfaces

from .test_pyfftw_base import run_test_suites

import unittest
import numpy
from numpy import fft as np_fft
import inspect
import warnings
warnings.filterwarnings('always')

complex_dtypes = (numpy.complex64, numpy.complex128, numpy.clongdouble)
real_dtypes = (numpy.float32, numpy.float64, numpy.longdouble)

def make_complex_data(shape, dtype):
    ar, ai = dtype(numpy.random.randn(2, *shape))
    return ar + 1j*ai

def make_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))


io_dtypes = {
    'complex': (complex_dtypes, make_complex_data),
    'r2c': (real_dtypes, make_real_data),
    'c2r': (complex_dtypes, make_complex_data)}

functions = {
        'fft': 'complex',
        'ifft': 'complex', 
        'rfft': 'r2c',
        'irfft': 'c2r',
        'rfftn': 'r2c',
        'irfftn': 'c2r',
        'rfft2': 'r2c',
        'irfft2': 'c2r',
        'fft2': 'complex',
        'ifft2': 'complex',
        'fftn': 'complex',
        'ifftn': 'complex'}

acquired_names = ('hfft', 'ihfft', 'fftfreq', 'fftshift', 'ifftshift')

class InterfacesNumpyFFTTestModule(unittest.TestCase):
    ''' A really simple test suite to check the module works as expected.
    '''

    def test_acquired_names(self):
        for each_name in acquired_names:

            numpy_fft_attr = getattr(numpy.fft, each_name)
            acquired_attr = getattr(interfaces.numpy_fft, each_name)

            self.assertIs(numpy_fft_attr, acquired_attr)


class InterfacesNumpyFFTTestFFT(unittest.TestCase):

    func = 'fft'
    axes_kw = 'axis'
    test_shapes = (
            ((100,), {}),
            ((128, 64), {'axis': 0}),
            ((128, 32), {'axis': -1}),
            ((59, 100), {}),
            ((32, 32, 4), {'axis': 1}),
            ((64, 128, 16), {}),
            )

    # invalid_s_shapes is:
    # (size, invalid_args, error_type, error_string)
    invalid_args = (
            ((100,), ((100, 200),), TypeError, ''),
            ((100, 200), ((100, 200),), TypeError, ''),
            ((100,), (100, (-2, -1)), TypeError, ''),
            ((100,), (100, -20), IndexError, ''))

    realinv = False

    @property
    def test_data(self):
        for test_shape, kwargs in self.test_shapes:
            axes = self.axes_from_kwargs(kwargs)
            s = self.s_from_kwargs(test_shape, kwargs)

            if self.realinv:
                test_shape = list(test_shape)
                test_shape[axes[-1]] = test_shape[axes[-1]]//2 + 1
                test_shape = tuple(test_shape)

            yield test_shape, s, kwargs

    def __init__(self, *args, **kwargs):

        super(InterfacesNumpyFFTTestFFT, self).__init__(*args, **kwargs)

        # Assume python 3, but keep backwards compatibility
        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def validate(self, array_type, test_shape, dtype, 
            s, kwargs):

        # Do it without the cache

        # without:
        interfaces.cache.disable()
        self._validate(array_type, test_shape, dtype, s, kwargs)

    def _validate(self, array_type, test_shape, dtype, 
            s, kwargs):

        input_array = array_type(test_shape, dtype)
        orig_input_array = input_array.copy()

        if input_array.dtype == 'clongdouble':
            np_input_array = numpy.complex128(input_array)

        elif input_array.dtype == 'longdouble':
            np_input_array = numpy.float64(input_array)

        else:
            np_input_array = input_array

        with warnings.catch_warnings(record=True) as w:
            # We catch the warnings so as to pick up on when
            # a complex array is turned into a real array

            output_array = getattr(interfaces.numpy_fft, self.func)(
                    input_array.copy(), s, **kwargs)

            if 'axes' in kwargs:
                axes = {'axes': kwargs['axes']}
            elif 'axis' in kwargs:
                axes = {'axis': kwargs['axis']}
            else:
                axes = {}

            test_out_array = getattr(np_fft, self.func)(
                    np_input_array.copy(), s, **axes)

            if (functions[self.func] == 'r2c'):
                if numpy.iscomplexobj(input_array):
                    if len(w) > 0:
                        # Make sure a warning is raised
                        self.assertIs(
                                w[-1].category, numpy.ComplexWarning)
        
        self.assertTrue(
                numpy.allclose(output_array, test_out_array, 
                    rtol=1e-2, atol=1e-4))

        if (not 'overwrite_input' in kwargs or 
                not kwargs['overwrite_input']):
            self.assertTrue(numpy.allclose(input_array,
                orig_input_array))

        return output_array

    def axes_from_kwargs(self, kwargs):
        
        argspec = inspect.getargspec(getattr(interfaces.numpy_fft, self.func))
        default_args = dict(list(zip(
            argspec.args[-len(argspec.defaults):], argspec.defaults)))

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
        argspec = inspect.getargspec(getattr(interfaces.numpy_fft, self.func))
        default_args = dict(list(zip(
            argspec.args[-len(argspec.defaults):], argspec.defaults)))

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
                    s = [test_shape[-1]]

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
        dtype_tuple = io_dtypes[functions[self.func]]
        
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                s = None

                self.validate(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)


    def test_fail_on_invalid_s_or_axes(self):
        dtype_tuple = io_dtypes[functions[self.func]]
        
        for dtype in dtype_tuple[0]:

            for test_shape, args, exception, e_str in self.invalid_args:
                input_array = dtype_tuple[1](test_shape, dtype)
                
                self.assertRaisesRegex(exception, e_str,
                        getattr(interfaces.numpy_fft, self.func), 
                        *((input_array,) + args))


    def test_same_sized_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                self.validate(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)

    def test_bigger_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
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
        dtype_tuple = io_dtypes[functions[self.func]]
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
            real_fft = getattr(interfaces.numpy_fft, self.func)
            setattr(interfaces.numpy_fft, self.func, fake_fft)

            _kwargs = kwargs.copy()

            for each_value in arg_test_values:
                _kwargs[arg] = each_value
                builder_args = getattr(interfaces.numpy_fft, self.func)(
                input_array.copy(), s, **_kwargs)
                
                self.assertTrue(builder_args[1][arg] == each_value)

            # make sure it was called
            self.assertTrue(len(return_values) > 0)
        except:
            raise
        
        finally:
            # Make sure we set it back
            setattr(interfaces.numpy_fft, self.func, real_fft)

        # Validate it aswell        
        for each_value in arg_test_values:
            _kwargs[arg] = each_value
            builder_args = getattr(interfaces.numpy_fft, self.func)(
            input_array.copy(), s, **_kwargs)

            self.validate(array_type, test_shape, dtype, s, _kwargs)

    def test_auto_align_input(self):
        dtype_tuple = io_dtypes[functions[self.func]]

        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                self.check_arg('auto_align_input', (True, False),
                        dtype_tuple[1], test_shape, dtype, s, kwargs)

    def test_auto_contiguous_input(self):
        dtype_tuple = io_dtypes[functions[self.func]]

        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                self.check_arg('auto_contiguous', (True, False), 
                        dtype_tuple[1], test_shape, dtype, s, kwargs)

    def test_bigger_and_smaller_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
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
            dtype_tuple = io_dtypes['complex']
        else:
            dtype_tuple = io_dtypes['r2c']

        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:
                s = None

                self.validate(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)


    def test_planner_effort(self):
        '''Test the planner effort arg
        '''
        dtype_tuple = io_dtypes[functions[self.func]]
        test_shape = (16,)
        
        for dtype in dtype_tuple[0]:
            s = None
            if self.axes_kw == 'axis':
                kwargs = {'axis': -1}
            else:
                kwargs = {'axes': (-1,)}

            for each_effort in ('FFTW_ESTIMATE', 'FFTW_MEASURE', 
                    'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'):
                
                kwargs['planner_effort'] = each_effort
                
                self.validate(
                        dtype_tuple[1], test_shape, dtype, s, kwargs)

            kwargs['planner_effort'] = 'garbage'

            self.assertRaisesRegex(ValueError, 'Invalid planner effort',
                    self.validate, 
                    *(dtype_tuple[1], test_shape, dtype, s, kwargs))

    def test_threads_arg(self):
        '''Test the threads argument
        '''
        dtype_tuple = io_dtypes[functions[self.func]]
        test_shape = (16,)
        
        for dtype in dtype_tuple[0]:
            s = None
            if self.axes_kw == 'axis':
                kwargs = {'axis': -1}
            else:
                kwargs = {'axes': (-1,)}

            self.check_arg('threads', (1, 2, 5, 10), 
                        dtype_tuple[1], test_shape, dtype, s, kwargs)

            kwargs['threads'] = 'bleh'
            
            # Should not work
            self.assertRaises(TypeError,
                    self.validate, 
                    *(dtype_tuple[1], test_shape, dtype, s, kwargs))


    def test_overwrite_input(self):
        '''Test the overwrite_input flag
        '''
        dtype_tuple = io_dtypes[functions[self.func]]
        
        for dtype in dtype_tuple[0]:
            for test_shape, s, _kwargs in self.test_data:
                s = None

                kwargs = _kwargs.copy()
                self.validate(dtype_tuple[1], test_shape, dtype, s, kwargs)
                
                self.check_arg('overwrite_input', (True, False),
                        dtype_tuple[1], test_shape, dtype, s, kwargs)

    def test_input_maintained(self):
        '''Test to make sure the input is maintained by default.
        '''
        dtype_tuple = io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                input_array = dtype_tuple[1](test_shape, dtype)

                orig_input_array = input_array.copy()
                
                getattr(interfaces.numpy_fft, self.func)(
                        input_array, s, **kwargs)

                self.assertTrue(
                        numpy.alltrue(input_array == orig_input_array))


class InterfacesNumpyFFTTestIFFT(InterfacesNumpyFFTTestFFT):
    func = 'ifft'

class InterfacesNumpyFFTTestRFFT(InterfacesNumpyFFTTestFFT):
    func = 'rfft'

class InterfacesNumpyFFTTestIRFFT(InterfacesNumpyFFTTestFFT):
    func = 'irfft'
    realinv = True    

class InterfacesNumpyFFTTestFFT2(InterfacesNumpyFFTTestFFT):
    axes_kw = 'axes'    
    func = 'ifft2'
    test_shapes = (
            ((128, 64), {'axes': None}),
            ((128, 32), {'axes': None}),
            ((128, 32, 4), {'axes': (0, 2)}),
            ((59, 100), {'axes': (-2, -1)}),
            ((64, 128, 16), {'axes': (0, 2)}),
            ((4, 6, 8, 4), {'axes': (0, 3)}),
            )
    
    invalid_args = (
            ((100,), ((100, 200),), ValueError, 'Shape error'),
            ((100, 200), ((100, 200, 100),), ValueError, 'Shape error'),
            ((100,), ((100, 200), (-3, -2, -1)), ValueError, 'Shape error'),
            ((100, 200), (100, -1), TypeError, ''),
            ((100, 200), ((100, 200), (-3, -2)), IndexError, 'Invalid axes'),
            ((100, 200), ((100,), (-3,)), IndexError, 'Invalid axes'))


class InterfacesNumpyFFTTestIFFT2(InterfacesNumpyFFTTestFFT2):
    func = 'ifft2'

class InterfacesNumpyFFTTestRFFT2(InterfacesNumpyFFTTestFFT2):
    func = 'rfft2'

class InterfacesNumpyFFTTestIRFFT2(InterfacesNumpyFFTTestFFT2):
    func = 'irfft2'
    realinv = True    

class InterfacesNumpyFFTTestFFTN(InterfacesNumpyFFTTestFFT2):
    func = 'ifftn'
    test_shapes = (
            ((128, 32, 4), {'axes': None}),
            ((64, 128, 16), {'axes': (0, 1, 2)}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1)}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1, 2)}),
            )

class InterfacesNumpyFFTTestIFFTN(InterfacesNumpyFFTTestFFTN):
    func = 'ifftn'

class InterfacesNumpyFFTTestRFFTN(InterfacesNumpyFFTTestFFTN):
    func = 'rfftn'

class InterfacesNumpyFFTTestIRFFTN(InterfacesNumpyFFTTestFFTN):
    func = 'irfftn'
    realinv = True    


test_cases = (
        InterfacesNumpyFFTTestModule,
        InterfacesNumpyFFTTestFFT,
        InterfacesNumpyFFTTestIFFT,
        InterfacesNumpyFFTTestRFFT,
        InterfacesNumpyFFTTestIRFFT,
        InterfacesNumpyFFTTestFFT2,
        InterfacesNumpyFFTTestIFFT2,
        InterfacesNumpyFFTTestRFFT2,
        InterfacesNumpyFFTTestIRFFT2,
        InterfacesNumpyFFTTestFFTN,
        InterfacesNumpyFFTTestIFFTN,
        InterfacesNumpyFFTTestRFFTN,
        InterfacesNumpyFFTTestIRFFTN)

#test_set = {'InterfacesNumpyFFTTestIRFFT2': ('test_bigger_s',)}
test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
