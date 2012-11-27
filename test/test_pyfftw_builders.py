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

from pyfftw import builders, n_byte_align_empty, n_byte_align, FFTW
import pyfftw.builders.builders as _builders

import unittest
import numpy
from numpy import fft as np_fft
import inspect

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

one_d_shapes = (
        ((100,), {}),
        ((128, 64), {'axis': 0}),
        ((128, 32), {'axis': -1}),
        ((59, 100), {}),
        ((32, 32, 4), {'axis': 1}),
        ((64, 128, 16), {}),
        )

two_d_shapes = (
        ((128, 64), {'axes': None}),
        ((128, 32), {'axes': None}),
        ((128, 32, 4), {'axes': (0, 2)}),
        ((59, 100), {'axes': (-2, -1)}),
        ((64, 128, 16), {'axes': (0, 2)}),
        ((4, 6, 8, 4), {'axes': (0, 3)}),
        )

n_d_shapes = (
        ((128, 32, 4), {'axes': None}),
        ((64, 128, 16), {'axes': (0, 1, 2)}),
        ((4, 6, 8, 4), {'axes': (0, 3, 1)}),
        ((4, 6, 8, 4), {'axes': (0, 3, 1, 2)}),
        )

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


class BuildersTestFFT(unittest.TestCase):

    func = 'fft'
    test_shapes = one_d_shapes
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

    def validate_pyfftw_object(self, array_type, test_shape, dtype, 
            s, kwargs):

        input_array = array_type(test_shape, dtype)

        FFTW_object = getattr(builders, self.func)(
                input_array.copy(), s, **kwargs)

        output_array = FFTW_object(input_array.copy())

        if input_array.dtype == 'clongdouble':
            input_array = numpy.complex128(input_array)

        elif input_array.dtype == 'longdouble':
            input_array = numpy.float64(input_array)

        test_out_array = getattr(np_fft, self.func)(
                input_array.copy(), s, **kwargs)
        
        self.assertTrue(
                numpy.allclose(output_array, test_out_array, 
                    rtol=1e-2, atol=1e-4))

        return FFTW_object

    def axes_from_kwargs(self, kwargs):
        
        argspec = inspect.getargspec(getattr(builders, self.func))
        default_args = dict(zip(
            argspec.args[-len(argspec.defaults):], argspec.defaults))

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
        argspec = inspect.getargspec(getattr(builders, self.func))
        default_args = dict(zip(
            argspec.args[-len(argspec.defaults):], argspec.defaults))

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

                FFTW_object = self.validate_pyfftw_object(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)

                self.assertTrue(type(FFTW_object) == FFTW)

    def test_fail_on_invalid_s(self):
        pass

    def test_fail_on_invalid_axes(self):
        pass

    def test_same_sized_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                FFTW_object = self.validate_pyfftw_object(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)

                self.assertTrue(type(FFTW_object) == FFTW)
    
    def test_bigger_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                try:
                    for each_axis, length in enumerate(s):
                        s[each_axis] += 2
                except TypeError:
                    s += 2

                FFTW_object = self.validate_pyfftw_object(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)

                self.assertTrue(
                        type(FFTW_object) == _builders._FFTWWrapper)

    def test_smaller_s(self):
        dtype_tuple = io_dtypes[functions[self.func]]
        for dtype in dtype_tuple[0]:
            for test_shape, s, kwargs in self.test_data:

                try:
                    for each_axis, length in enumerate(s):
                        s[each_axis] -= 2
                except TypeError:
                    s -= 2

                FFTW_object = self.validate_pyfftw_object(dtype_tuple[1], 
                        test_shape, dtype, s, kwargs)

                self.assertTrue(
                        type(FFTW_object) == _builders._FFTWWrapper)                

    def test_s_subset(self):
        pass

    def test_flags(self):
        pass

    def test_auto_align_input(self):
        pass

    def test_n_threads(self):
        pass

class BuildersTestIFFT(BuildersTestFFT):
    func = 'ifft'

class BuildersTestRFFT(BuildersTestFFT):
    func = 'rfft'

class BuildersTestIRFFT(BuildersTestFFT):
    func = 'irfft'
    realinv = True    

class BuildersTestFFT2(BuildersTestFFT):
    func = 'ifft2'
    test_shapes = two_d_shapes

class BuildersTestIFFT2(BuildersTestFFT2):
    func = 'ifft2'

class BuildersTestRFFT2(BuildersTestFFT2):
    func = 'rfft2'

class BuildersTestIRFFT2(BuildersTestFFT2):
    func = 'irfft2'
    realinv = True    

class BuildersTestFFTN(BuildersTestFFT2):
    func = 'ifftn'
    test_shapes = n_d_shapes

class BuildersTestIFFTN(BuildersTestFFTN):
    func = 'ifftn'

class BuildersTestRFFTN(BuildersTestFFTN):
    func = 'rfftn'

class BuildersTestIRFFTN(BuildersTestFFTN):
    func = 'irfftn'
    realinv = True    


class BuildersTestFFTWWrapper(unittest.TestCase):
    '''This basically reimplements the FFTW.__call__ tests, with
    a few tweaks.
    '''

    def setUp(self):

        self.input_array_slicer = [slice(None), slice(256)]
        self.FFTW_array_slicer = [slice(128), slice(None)]
        
        self.input_array = n_byte_align_empty((128, 512), 16, 
                dtype='complex128')
        self.output_array = n_byte_align_empty((256, 256), 16,
                dtype='complex128')

        self.internal_array = n_byte_align_empty((256, 256), 16, 
                dtype='complex128')

        self.fft = _builders._FFTWWrapper(self.internal_array, 
                self.output_array,
                input_array_slicer=self.input_array_slicer,
                FFTW_array_slicer=self.FFTW_array_slicer)

        self.input_array[:] = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        self.internal_array[:] = 0
        self.internal_array[self.FFTW_array_slicer] = (
                self.input_array[self.input_array_slicer])

    def update_arrays(self, input_array, output_array):
        '''Does what the internal update arrays does for an FFTW
        object but with a reslicing.
        '''
        internal_input_array = self.fft.get_input_array()
        internal_output_array = self.fft.get_output_array()

        internal_input_array[self.FFTW_array_slicer] = (
                input_array[self.input_array_slicer])

        self.fft(output_array=output_array)

    def test_call(self):
        '''Test a call to an instance of the class.
        '''

        self.input_array[:] = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

    
    def test_call_with_positional_input_update(self):
        '''Test the class call with a positional input update.
        '''

        input_array = n_byte_align(
                (numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = self.fft(n_byte_align(input_array.copy(), 16)).copy()

        self.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

        
    def test_call_with_keyword_input_update(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = n_byte_align(
                numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape), 16)

        output_array = self.fft(
            input_array=n_byte_align(input_array.copy(), 16)).copy()

        self.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
    
        
    def test_call_with_keyword_output_update(self):
        '''Test the class call with a keyword output update.
        '''
        output_array = n_byte_align(
            (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
                output_array=n_byte_align(output_array.copy(), 16)).copy()

        
        self.update_arrays(self.input_array, output_array)
        self.fft.execute()

        self.assertTrue(
                numpy.alltrue(returned_output_array == output_array))

    def test_call_with_positional_updates(self):
        '''Test the class call with a positional array updates.
        '''
        
        input_array = n_byte_align((numpy.random.randn(*self.input_array.shape) 
            + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = n_byte_align((numpy.random.randn(*self.output_array.shape) 
            + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
            n_byte_align(input_array.copy(), 16),
            n_byte_align(output_array.copy(), 16)).copy()

        self.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))

    def test_call_with_keyword_updates(self):
        '''Test the class call with a positional output update.
        '''
        
        input_array = n_byte_align(
                (numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = n_byte_align(
                (numpy.random.randn(*self.output_array.shape)
                    + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
                output_array=n_byte_align(output_array.copy(), 16),
                input_array=n_byte_align(input_array.copy(), 16)).copy()

        self.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))
    
    def test_call_with_different_input_dtype(self):
        '''Test the class call with an array with a different input dtype
        '''
        input_array = n_byte_align(numpy.complex64(
                numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = self.fft(n_byte_align(input_array.copy(), 16)).copy()

        _input_array = numpy.asarray(input_array,
                dtype=self.input_array.dtype)

        self.update_arrays(_input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
    
    def test_call_with_list_input(self):
        '''Test the class call with a list rather than an array
        '''

        output_array = self.fft().copy()

        test_output_array = self.fft(self.input_array.tolist()).copy()

        self.assertTrue(numpy.alltrue(output_array == test_output_array))


    def test_call_with_invalid_update(self):
        '''Test the class call with an invalid update.
        '''

        new_shape = self.input_array.shape + (2, )
        invalid_array = (numpy.random.randn(*new_shape) 
                + 1j*numpy.random.randn(*new_shape))
        
        self.assertRaises(ValueError, self.fft, 
                *(),
                **{'output_array':invalid_array})

        self.assertRaises(ValueError, self.fft, 
                *(),
                **{'input_array':invalid_array})

    
    def test_call_with_invalid_output_striding(self):
        '''Test the class call with an invalid strided output update.
        '''
        # Add an extra dimension to bugger up the striding
        new_shape = self.output_array.shape + (2,)
        output_array = n_byte_align(numpy.random.randn(*new_shape) 
                + 1j*numpy.random.randn(*new_shape), 16)

        self.assertRaisesRegexp(ValueError, 'Invalid output striding',
                self.fft, **{'output_array': output_array[:,:,1]})

    def test_call_with_different_striding(self):
        '''Test the input update with different strides to internal array.
        '''
        input_array_shape = self.input_array.shape + (2,)
        internal_array_shape = self.internal_array.shape

        internal_array = n_byte_align(
                numpy.random.randn(*internal_array_shape) 
                + 1j*numpy.random.randn(*internal_array_shape), 16)

        fft =  _builders._FFTWWrapper(internal_array, self.output_array,
                input_array_slicer=self.input_array_slicer,
                FFTW_array_slicer=self.FFTW_array_slicer)
        
        test_output_array = fft().copy()

        new_input_array = n_byte_align_empty(input_array_shape, 16,
                dtype=internal_array.dtype)
        new_input_array[:] = 0

        new_input_array[:,:,0][self.input_array_slicer] = (
                internal_array[self.FFTW_array_slicer])

        new_output = fft(new_input_array[:,:,0]).copy()

        # Test the test!
        self.assertTrue(
                new_input_array[:,:,0].strides != internal_array.strides)

        self.assertTrue(numpy.alltrue(test_output_array == new_output))

    def test_call_with_copy_with_missized_array_error(self):
        '''Force an input copy with a missized array.
        '''
        shape = list(self.input_array.shape + (2,))
        shape[0] += 1

        input_array = n_byte_align(numpy.random.randn(*shape) 
                + 1j*numpy.random.randn(*shape), 16)

        self.assertRaisesRegexp(ValueError, 'Invalid input shape',
                self.fft, **{'input_array': input_array[:,:,0]})

    def test_call_with_normalisation_on(self):
        _input_array = n_byte_align_empty(self.internal_array.shape, 16,
                dtype='complex128')
        
        ifft = _builders._FFTWWrapper(self.output_array, _input_array, 
                direction='FFTW_BACKWARD',
                input_array_slicer=slice(None),
                FFTW_array_slicer=slice(None))

        self.fft(normalise_idft=True) # Shouldn't make any difference
        ifft(normalise_idft=True)

        self.assertTrue(numpy.allclose(
            self.input_array[self.input_array_slicer], 
            _input_array[self.FFTW_array_slicer]))

    def test_call_with_normalisation_off(self):
        
        _input_array = n_byte_align_empty(self.internal_array.shape, 16,
                dtype='complex128')

        ifft = _builders._FFTWWrapper(self.output_array, _input_array, 
                direction='FFTW_BACKWARD',
                input_array_slicer=slice(None),
                FFTW_array_slicer=slice(None))

        self.fft(normalise_idft=True) # Shouldn't make any difference
        ifft(normalise_idft=False)

        _input_array /= ifft.N

        self.assertTrue(numpy.allclose(
            self.input_array[self.input_array_slicer], 
            _input_array[self.FFTW_array_slicer]))

    def test_call_with_normalisation_default(self):
        _input_array = n_byte_align_empty(self.internal_array.shape, 16,
                dtype='complex128')

        ifft = _builders._FFTWWrapper(self.output_array, _input_array, 
                direction='FFTW_BACKWARD',
                input_array_slicer=slice(None),
                FFTW_array_slicer=slice(None))

        self.fft()
        ifft()

        # Scaling is performed by default
        self.assertTrue(numpy.allclose(
            self.input_array[self.input_array_slicer], 
            _input_array[self.FFTW_array_slicer]))


class BuildersTestUtilities(unittest.TestCase):

    def test_setup_input_slicers(self):
        inputs = (
                ((4, 5), (4, 5)),
                ((4, 4), (3, 5)),
                ((4, 5), (3, 5)),
                )

        outputs = (
                ([slice(None), slice(None)], [slice(None), slice(None)]),
                ([slice(0, 3), slice(None)], [slice(None), slice(0, 4)]),
                ([slice(0, 3), slice(None)], [slice(None), slice(None)]),
                )

        for _input, _output in zip(inputs, outputs):
            self.assertEqual(
                    _builders._setup_input_slicers(*_input),
                    _output)



    def test_compute_array_shapes(self):
        # inputs are:
        # (a.shape, s, axes, inverse, real)
        inputs = (
                ((4, 5), (4, 5), (-2, -1), False, False),
                ((4, 5), (4, 5), (-1, -2), False, False),
                ((4, 5), (4, 5), (-1, -2), True, False),
                ((4, 5), (4, 5), (-1, -2), True, True),
                ((4, 5), (4, 5), (-2, -1), True, True),
                ((4, 5), (4, 5), (-2, -1), False, True),
                ((4, 5), (4, 5), (-1, -2), False, True),
                ((4, 5, 6), (4, 5), (-2, -1), False, False),
                ((4, 5, 6), (5, 6), (-2, -1), False, False),
                ((4, 5, 6), (3, 5), (-3, -1), False, False),
                ((4, 5, 6), (4, 5), (-2, -1), True, False),
                ((4, 5, 6), (3, 5), (-3, -1), True, False),
                ((4, 5, 6), (4, 5), (-2, -1), True, True),
                ((4, 5, 6), (3, 5), (-3, -1), True, True),
                ((4, 5, 6), (4, 5), (-2, -1), False, True),
                ((4, 5, 6), (3, 5), (-3, -1), False, True),
                )

        outputs = (
                ((4, 5), (4, 5)),
                ((5, 4), (5, 4)),
                ((5, 4), (5, 4)),
                ((3, 4), (5, 4)),
                ((4, 3), (4, 5)),
                ((4, 5), (4, 3)),
                ((5, 4), (3, 4)),
                ((4, 4, 5), (4, 4, 5)),
                ((4, 5, 6), (4, 5, 6)),
                ((3, 5, 5), (3, 5, 5)),
                ((4, 4, 5), (4, 4, 5)),
                ((3, 5, 5), (3, 5, 5)),
                ((4, 4, 3), (4, 4, 5)),
                ((3, 5, 3), (3, 5, 5)),
                ((4, 4, 5), (4, 4, 3)),
                ((3, 5, 5), (3, 5, 3)),
                )

        for _input, output in zip(inputs, outputs):
            shape, s, axes, inverse, real = _input
            a = numpy.empty(shape)

            self.assertEqual(
                    _builders._compute_array_shapes(a, s, axes, inverse, real),
                    output)

    def test_compute_array_shapes_invalid_axes(self):

        a = numpy.zeros((3, 4))
        s = (3, 4)
        test_axes = ((1, 2, 3),)

        for each_axes in test_axes:

            args = (a, s, each_axes, False, False)
            self.assertRaisesRegexp(IndexError, 'Invalid axes', 
                    _builders._compute_array_shapes, *args)

    def _call_cook_nd_args(self, arg_tuple):
        a = numpy.zeros(arg_tuple[0])
        args = ('s', 'axes', 'invreal')
        arg_dict = {'a': a}
        for arg_name, arg in zip(args, arg_tuple[1:]):
            if arg is not None:
                arg_dict[arg_name] = arg

        return _builders._cook_nd_args(**arg_dict)

    def test_cook_nd_args_normal(self):
        # inputs are (a.shape, s, axes, invreal)
        # None corresponds to no argument
        inputs = (
                ((2, 3), None, (-1,), False),
                ((2, 3), (5, 6), (-2, -1), False),
                ((2, 3), (5, 6), (-1, -2), False),
                ((2, 3), None, (-1, -2), False),
                ((2, 3, 5), (5, 6), (-1, -2), False),
                ((2, 3, 5), (5, 6), None, False),
                ((2, 3, 5), None, (-1, -2), False),
                ((2, 3, 5), None, (-1, -3), False))

        outputs = (
                ((3,), (-1,)),
                ((5, 6), (-2, -1)),
                ((5, 6), (-1, -2)),
                ((3, 2), (-1, -2)),
                ((5, 6), (-1, -2)),
                ((5, 6), (-2, -1)),
                ((5, 3), (-1, -2)),
                ((5, 2), (-1, -3))
                )

        for each_input, each_output in zip(inputs, outputs):
            self.assertEqual(self._call_cook_nd_args(each_input),
                    each_output)
    
    def test_cook_nd_args_invreal(self):

        # inputs are (a.shape, s, axes, invreal)
        # None corresponds to no argument
        inputs = (
                ((2, 3), None, (-1,), True),
                ((2, 3), (5, 6), (-2, -1), True),
                ((2, 3), (5, 6), (-1, -2), True),
                ((2, 3), None, (-1, -2), True),
                ((2, 3, 5), (5, 6), (-1, -2), True),
                ((2, 3, 5), (5, 6), None, True),
                ((2, 3, 5), None, (-1, -2), True),
                ((2, 3, 5), None, (-1, -3), True))

        outputs = (
                ((4,), (-1,)),
                ((5, 6), (-2, -1)),
                ((5, 6), (-1, -2)),
                ((3, 2), (-1, -2)),
                ((5, 6), (-1, -2)),
                ((5, 6), (-2, -1)),
                ((5, 4), (-1, -2)),
                ((5, 2), (-1, -3))
                )

        for each_input, each_output in zip(inputs, outputs):
            self.assertEqual(self._call_cook_nd_args(each_input),
                    each_output)


    def test_cook_nd_args_invalid_inputs(self):
        # inputs are (a.shape, s, axes, invreal)
        # None corresponds to no argument
        inputs = (
                ((2, 3), (1,), (-1, -2), None),
                ((2, 3), (2, 3, 4), (-3, -2, -1), None),
                )

        # all the inputs should yield an error
        for each_input in inputs:
            self.assertRaisesRegexp(ValueError, 'Shape error', 
                    self._call_cook_nd_args, *(each_input,))

test_cases = (
        BuildersTestFFTWWrapper,
        BuildersTestUtilities,
        BuildersTestFFT,
        BuildersTestIFFT,
        BuildersTestIRFFT,
        BuildersTestIRFFT,
        BuildersTestFFT2,
        BuildersTestIFFT2,
        BuildersTestIRFFT2,
        BuildersTestIRFFT2,
        BuildersTestFFTN,
        BuildersTestIFFTN,
        BuildersTestIRFFTN,
        BuildersTestIRFFTN)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
