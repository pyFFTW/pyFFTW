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

from pyfftw import FFTW, n_byte_align, n_byte_align_empty
import numpy
from timeit import Timer

import unittest

def timer_test_setup(fft_length = 2048, vectors = 256):
    a = numpy.complex64(numpy.random.randn(vectors,fft_length)
            +1j*numpy.random.randn(vectors,fft_length))
    b = numpy.complex64(numpy.random.randn(vectors,fft_length)
            +1j*numpy.random.randn(vectors,fft_length))
    
    fft_class = FFTW(a,b, flags=['FFTW_MEASURE'])

    # We need to refill a with data as it gets clobbered by
    # initialisation.
    a[:] = numpy.complex64(numpy.random.randn(vectors,fft_length)
            +1j*numpy.random.randn(vectors,fft_length))


    return fft_class, a, b

timer_setup = '''
import numpy
try:
    from __main__ import timer_test_setup
except:
    from test_pyfftw import timer_test_setup

fft_class,a,b= timer_test_setup()
'''
def timer():
    from timeit import Timer
    N = 100
    t = Timer(stmt="fft_class.execute()", setup=timer_setup)
    t_numpy_fft = Timer(stmt="numpy.fft.fft(a)", setup=timer_setup)
    
    t_numpy_fft = Timer(stmt="numpy.fft.fft(a)", setup=timer_setup)
    
    t_str = ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'
    t_numpy_str = ("%.2f" % (1000.0/N*t_numpy_fft.timeit(N)))+' ms'

    print ('One run: '+ t_str + ' (versus ' + t_numpy_str + ' for numpy.fft)')

def timer_with_array_update():
    from timeit import Timer
    N = 100
    # We can update the arrays with the original arrays. We're
    # really just trying to test the code path, which should be
    # fixed regardless of the array size.
    t = Timer(
            stmt="fft_class.update_arrays(a,b); fft_class.execute()", 
            setup=timer_setup)
        
    print ('One run: '+ ("%.2f" % (1000.0/N*t.timeit(N)))+' ms')

class Complex64FFTWTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(Complex64FFTWTest, self).__init__(*args, **kwargs)
        self.make_shapes()

    def setUp(self):
        
        self.input_dtype = numpy.complex64
        self.output_dtype = numpy.complex64
        self.np_fft_comparison = numpy.fft.fft
        return

    def create_test_arrays(self, input_shape, output_shape):
        a = self.input_dtype(numpy.random.randn(*input_shape)
                +1j*numpy.random.randn(*input_shape))

        b = self.output_dtype(numpy.random.randn(*output_shape)
                +1j*numpy.random.randn(*output_shape))

        return a, b

    def tearDown(self):
        
        return

    def make_shapes(self):
        self.input_shapes = {
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (15, 256, 2048)}

        self.output_shapes = {
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (15, 256, 2048)}

    def reference_fftn(self, a, axes):
        return numpy.fft.fftn(a, axes=axes)

    def timer_routine(self, pyfftw_callable, numpy_fft_callable):

        N = 100

        t = Timer(stmt=pyfftw_callable)
        t_numpy_fft = Timer(stmt=numpy_fft_callable)
    
        t_str = ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'
        t_numpy_str = ("%.2f" % (1000.0/N*t_numpy_fft.timeit(N)))+' ms'

        print ('One run: '+ t_str + \
                ' (versus ' + t_numpy_str + ' for numpy.fft)')
    
    def test_time(self):
        
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes)

        self.timer_routine(fft.execute, 
                lambda: self.np_fft_comparison(a))
        self.assertTrue(True)

    def test_time_with_array_update(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes)
        
        def fftw_callable():
            fft.update_arrays(a,b)
            fft.execute()

        self.timer_routine(fftw_callable, 
                lambda: self.np_fft_comparison(a))

        self.assertTrue(True)

    def run_validate_fft(self, a, b, axes, fft=None, ifft=None, 
            force_unaligned_data=False, create_array_copies=True):
        ''' Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.

        If force_unaligned_data is True, the flag FFTW_UNALIGNED
        will be passed to the fftw routines.
        '''

        if create_array_copies:
            # Don't corrupt the original mutable arrays
            a = a.copy()
            b = b.copy()

        a_orig = a.copy()

        flags = ['FFTW_ESTIMATE']

        if force_unaligned_data:
            flags.append('FFTW_UNALIGNED')
        
        if fft == None:
            fft = FFTW(a,b,axes=axes,
                    direction='FFTW_FORWARD',flags=flags)
        else:
            fft.update_arrays(a,b)

        if ifft == None:
            ifft = FFTW(b,a,axes=axes,
                    direction='FFTW_BACKWARD',flags=flags)
        else:
            ifft.update_arrays(b,a)


        a[:] = a_orig

        # Test the forward FFT by comparing it to the result from numpy.fft
        fft.execute()
        ref_b = self.reference_fftn(a, axes=axes)

        # This is actually quite a poor relative error, but it still
        # sometimes fails. I assume that numpy.fft has different internals
        # to fftw.
        self.assertTrue(numpy.allclose(b, ref_b, rtol=1e-2, atol=1e-3))
        
        # Test the inverse FFT by comparing the result to the starting
        # value (which is scaled as per FFTW being unnormalised).
        ifft.execute()
        # The scaling is the product of the lengths of the fft along
        # the axes along which the fft is taken.
        scaling = numpy.prod(numpy.array(a.shape)[list(axes)])

        self.assertTrue(numpy.allclose(a/scaling, a_orig, rtol=1e-2, atol=1e-3))
        return fft, ifft

    def test_1d(self):
        in_shape = self.input_shapes['1d']
        out_shape = self.output_shapes['1d']

        axes=(0,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.run_validate_fft(a, b, axes)
       
    def test_multiple_1d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.run_validate_fft(a, b, axes)

    def test_default_args(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        a, b = self.create_test_arrays(in_shape, out_shape)
        
        fft = FFTW(a,b)
        fft.execute()
        ref_b = self.reference_fftn(a, axes=(-1,))
        self.assertTrue(numpy.allclose(b, ref_b, rtol=1e-2, atol=1e-3))

    def test_2d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.run_validate_fft(a, b, axes, create_array_copies=False)
    
    def test_multiple_2d(self):
        in_shape = self.input_shapes['3d']
        out_shape = self.output_shapes['3d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.run_validate_fft(a, b, axes, create_array_copies=False)

    def test_3d(self):
        in_shape = self.input_shapes['3d']
        out_shape = self.output_shapes['3d']
        
        axes=(0, 1, 2)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.run_validate_fft(a, b, axes, create_array_copies=False)

    def test_missized_fail(self):
        in_shape = self.input_shapes['2d']
        _out_shape = self.output_shapes['2d']

        out_shape = (_out_shape[0]+1, _out_shape[1])
        
        axes=(0,1)
        a, b = self.create_test_arrays(in_shape, out_shape)
    
        self.assertRaises(ValueError, FFTW, *(a,b, axes))
    
    def test_missized_nonfft_axes_fail(self):
        in_shape = self.input_shapes['3d']
        _out_shape = self.output_shapes['3d']
        out_shape = (_out_shape[0], _out_shape[1]+1, _out_shape[2])
        
        axes=(0, 2)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.assertRaises(ValueError, FFTW, *(a,b, axes))

    def test_extra_dimension_fail(self):
        in_shape = self.input_shapes['2d']
        _out_shape = self.output_shapes['2d']        
        out_shape = (2, _out_shape[0], _out_shape[1])
        
        axes=(1, 2)
        a, b = self.create_test_arrays(in_shape, out_shape)
    
        self.assertRaises(ValueError, FFTW, *(a,b))


    def test_f_contiguous_1d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(0,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Taking the transpose just makes the array F contiguous
        a = a.transpose()
        b = b.transpose()

        self.run_validate_fft(a, b, axes, create_array_copies=False)

    def test_non_contiguous_2d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, 300:2041:9]
        # b needs to be the same size
        b_sliced = b[20:146:2, 100:1458:7]

        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

    def test_non_contiguous_2d_in_3d(self):
        in_shape = (256, 4, 2048)
        out_shape = in_shape
        axes=(0,2)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, :, 300:2041:9]
        # b needs to be the same size
        b_sliced = b[20:146:2, :, 100:1458:7]

        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

    def test_different_dtypes_fail(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        a_ = numpy.complex64(a)
        b_ = numpy.complex128(b)
        self.assertRaises(ValueError, FFTW, *(a_,b_))

        a_ = numpy.complex128(a)
        b_ = numpy.complex64(b)
        self.assertRaises(ValueError, FFTW, *(a_,b_))

    def test_update_data(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes)

        a, b = self.create_test_arrays(in_shape, out_shape)
        
        self.run_validate_fft(a, b, axes, fft=fft, ifft=ifft)

    def test_update_data_with_stride_error(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes, 
                create_array_copies=False)

        in_shape = (in_shape[0]+2, in_shape[1]+2)
        out_shape = (out_shape[0]+2, out_shape[1]+2)

        a_, b_ = self.create_test_arrays(in_shape, out_shape)

        a_ = a_[2:,2:]
        b_ = b_[2:,2:]

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes),
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a,b_,axes),
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})

    def test_update_data_with_shape_error(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes)

        in_shape = (in_shape[0]-10, in_shape[1])
        out_shape = (out_shape[0], out_shape[1]+5)

        a_, b_ = self.create_test_arrays(in_shape, out_shape)

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b_,axes),
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), 
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})
    
    def test_update_unaligned_data_with_FFTW_UNALIGNED(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        a = n_byte_align(a, 16)
        b = n_byte_align(b, 16)

        fft, ifft = self.run_validate_fft(a, b, axes, 
                force_unaligned_data=True)

        a, b = self.create_test_arrays(in_shape, out_shape)

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(in_shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.input_dtype).reshape(*in_shape)
        a_[:] = a 
        
        b__ = n_byte_align_empty(
                numpy.prod(out_shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.output_dtype).reshape(*out_shape)
        b_[:] = b

        self.run_validate_fft(a, b_, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b_, axes, fft=fft, ifft=ifft)

    def test_update_data_with_unaligned_original(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(in_shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.input_dtype).reshape(*in_shape)
        a_[:] = a
        
        b__ = n_byte_align_empty(
                numpy.prod(out_shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.output_dtype).reshape(*out_shape)
        b_[:] = b
        
        fft, ifft = self.run_validate_fft(a_, b_, axes, 
                force_unaligned_data=True)
        
        self.run_validate_fft(a, b_, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b_, axes, fft=fft, ifft=ifft)


    def test_update_data_with_alignment_error(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        a = n_byte_align(a, 16)
        b = n_byte_align(b, 16)

        fft, ifft = self.run_validate_fft(a, b, axes)
        
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(in_shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.input_dtype).reshape(*in_shape)
        a_[:] = a 
        
        b__ = n_byte_align_empty(
                numpy.prod(out_shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.output_dtype).reshape(*out_shape)
        b_[:] = b
     
        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a,b_,axes), 
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), 
                **{'fft':fft, 'ifft':ifft, 'create_array_copies':False})

    def test_invalid_axes(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-3,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.assertRaises(ValueError, FFTW, *(a,b,axes))

        axes=(10,)
        self.assertRaises(ValueError, FFTW, *(a,b,axes))

class Complex128FFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.complex128
        self.output_dtype = numpy.complex128
        self.np_fft_comparison = numpy.fft.fft        
        return

class ComplexLongDoubleFFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.clongdouble
        self.output_dtype = numpy.clongdouble
        self.np_fft_comparison = self.reference_fftn       
        return

    def reference_fftn(self, a, axes):

        # numpy.fft.fftn doesn't support complex256 type,
        # so we need to compare to a lower precision type.
        a = numpy.complex128(a)
        return numpy.fft.fftn(a, axes=axes)

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time(self):
        pass

    @unittest.skip('numpy.fft has issues with this dtype.')    
    def test_time_with_array_update(self):
        pass

class RealForwardDoubleFFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.float64
        self.output_dtype = numpy.complex128 
        self.np_fft_comparison = numpy.fft.rfft
        return  
    
    def make_shapes(self):
        self.input_shapes = {
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (15, 256, 2048)}

        self.output_shapes = {
                '1d': (1025,),
                '2d': (256, 1025),
                '3d': (15, 256, 1025)}

    def create_test_arrays(self, input_shape, output_shape):
        a = self.input_dtype(numpy.random.randn(*input_shape))

        b = self.output_dtype(numpy.random.randn(*output_shape)
                +1j*numpy.random.randn(*output_shape))

        return a, b
    
    def reference_fftn(self, a, axes):

        return numpy.fft.rfftn(a, axes=axes)
    
    def test_wrong_direction_fail(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.assertRaises(ValueError, FFTW, *(a,b),
                **{'direction':'FFTW_BACKWARD'})

    def test_non_contiguous_2d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, 300:2041:9]
        # b needs to be compatible
        b_sliced = b[20:146:2, 100:786:7]

        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

    def test_non_contiguous_2d_in_3d(self):
        in_shape = (256, 4, 2048)
        out_shape = in_shape
        axes=(0,2)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, :, 300:2041:9]
        # b needs to be compatible
        b_sliced = b[20:146:2, :, 100:786:7]

        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

class RealForwardSingleFFTWTest(RealForwardDoubleFFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.float32
        self.output_dtype = numpy.complex64
        self.np_fft_comparison = numpy.fft.rfft        
        return 

class RealForwardLongDoubleFFTWTest(RealForwardDoubleFFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.longdouble
        self.output_dtype = numpy.clongdouble 
        self.np_fft_comparison = numpy.fft.rfft
        return

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time(self):
        pass

    @unittest.skip('numpy.fft has issues with this dtype.')    
    def test_time_with_array_update(self):
        pass

    def reference_fftn(self, a, axes):

        a = numpy.float64(a)
        return numpy.fft.rfftn(a, axes=axes)

class RealBackwardDoubleFFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.complex128
        self.output_dtype = numpy.float64
        self.np_fft_comparison = numpy.fft.irfft       
        return  
    
    def make_shapes(self):

        self.input_shapes = {
                '1d': (1025,),
                '2d': (256, 1025),
                '3d': (15, 256, 1025)}

        self.output_shapes = {
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (15, 256, 2048)}

    def create_test_arrays(self, input_shape, output_shape, axes=None):

        a = self.input_dtype(numpy.random.randn(*input_shape)
                +1j*numpy.random.randn(*input_shape))
        
        b = self.output_dtype(numpy.random.randn(*output_shape))

        # We fill a by doing the forward FFT from b.
        # This means that the relevant bits that should be purely
        # real will be (for example the zero freq component). 
        # This is easier than writing a complicate system to work it out.
        try:
            if axes == None:
                fft = FFTW(b,a,direction='FFTW_FORWARD')
            else:
                fft = FFTW(b,a,direction='FFTW_FORWARD', axes=axes)

            b[:] = self.output_dtype(numpy.random.randn(*output_shape))
            
            fft.execute()
            
            scaling = numpy.prod(numpy.array(a.shape))
            a = self.input_dtype(a/scaling)

        except ValueError:
            # In this case, we assume that it was meant to error,
            # so we can return what we want.
            pass

        b = self.output_dtype(numpy.random.randn(*output_shape))
        
        return a, b

    def run_validate_fft(self, a, b, axes, fft=None, ifft=None, 
            force_unaligned_data=False, create_array_copies=True):
        ''' *** EVERYTHING IS FLIPPED AROUND BECAUSE WE ARE
        VALIDATING AN INVERSE FFT ***
        
        Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.

        If force_unaligned_data is True, the flag FFTW_UNALIGNED
        will be passed to the fftw routines.
        '''
        if create_array_copies:
            # Don't corrupt the original mutable arrays
            a = a.copy()
            b = b.copy()

        a_orig = a.copy()

        flags = ['FFTW_ESTIMATE']

        if force_unaligned_data:
            flags.append('FFTW_UNALIGNED')

        if ifft == None:
            ifft = FFTW(a,b,axes=axes,
                    direction='FFTW_BACKWARD',flags=flags)
        else:
            ifft.update_arrays(a,b)

        if fft == None:
            fft = FFTW(b,a,axes=axes,
                    direction='FFTW_FORWARD',flags=flags)
        else:
            fft.update_arrays(b,a)


        a[:] = a_orig
        # Test the inverse FFT by comparing it to the result from numpy.fft
        ifft.execute()

        a[:] = a_orig
        ref_b = self.reference_fftn(a, axes=axes)

        # The scaling is the product of the lengths of the fft along
        # the axes along which the fft is taken.
        scaling = numpy.prod(numpy.array(b.shape)[list(axes)])

        # This is actually quite a poor relative error, but it still
        # sometimes fails. I assume that numpy.fft has different internals
        # to fftw.
        self.assertTrue(numpy.allclose(b/scaling, ref_b, rtol=1e-2, atol=1e-3))
        
        # Test the FFT by comparing the result to the starting
        # value (which is scaled as per FFTW being unnormalised).
        fft.execute()

        self.assertTrue(numpy.allclose(a/scaling, a_orig, rtol=1e-2, atol=1e-3))
        return fft, ifft

    def test_time_with_array_update(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes)

        def fftw_callable():
            fft.update_arrays(b,a)
            fft.execute()

        self.timer_routine(fftw_callable, 
                lambda: self.np_fft_comparison(a))

        self.assertTrue(True)
    
    def reference_fftn(self, a, axes):
        # This needs to be an inverse
        return numpy.fft.irfftn(a, axes=axes)
    
    def test_wrong_direction_fail(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.assertRaises(ValueError, FFTW, *(a,b),
                **{'direction':'FFTW_FORWARD'})

    def test_default_args(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        a, b = self.create_test_arrays(in_shape, out_shape)
        
        # default args should fail for backwards transforms
        # (as the default is FFTW_FORWARD)
        self.assertRaises(ValueError, FFTW, *(a,b))

    def test_non_contiguous_2d(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']
        
        axes=(-2,-1)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[20:146:2, 100:786:7]
        # b needs to be compatible
        b_sliced = b[12:200:3, 300:2041:9]

        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

    @unittest.skipIf(numpy.version.version <= '1.6.1',
            'numpy.fft <= 1.6.1 has a bug that causes this test to fail.')
    def test_non_contiguous_2d_in_3d(self):
        in_shape = (256, 4, 1025)
        out_shape = (256, 4, 2048)
        axes=(0,2)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Some arbitrary and crazy slicing
        a_sliced = a[20:146:2, :, 100:786:7]
        # b needs to be compatible
        b_sliced = b[12:200:3, :, 300:2041:9]
        
        # The data doesn't work, so we need to generate it for the 
        # correct size
        a_, b_ = self.create_test_arrays(a_sliced.shape, b_sliced.shape, axes=axes)

        # And then copy it into the non contiguous array
        a_sliced[:] = a_
        b_sliced[:] = b_
        
        self.run_validate_fft(a_sliced, b_sliced, axes, create_array_copies=False)

class RealBackwardSingleFFTWTest(RealBackwardDoubleFFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.complex64
        self.output_dtype = numpy.float32 
        self.np_fft_comparison = numpy.fft.irfft

        return 

class RealBackwardLongDoubleFFTWTest(RealBackwardDoubleFFTWTest):
    
    def setUp(self):

        self.input_dtype = numpy.clongdouble
        self.output_dtype = numpy.longdouble 
        self.np_fft_comparison = numpy.fft.irfft        
        return

    def reference_fftn(self, a, axes):

        a = numpy.complex128(a)
        return numpy.fft.irfftn(a, axes=axes)

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time(self):
        pass

    @unittest.skip('numpy.fft has issues with this dtype.')    
    def test_time_with_array_update(self):
        pass


class NByteAlignTest(unittest.TestCase):

    def setUp(self):

        return

    def tearDown(self):
        
        return

    def test_n_byte_align_empty(self):
        shape = (10,10)
        # Test a few alignments and dtypes
        for each in [(3, 'float64'),
                (7, 'float64'),
                (9, 'float32'),
                (16, 'int64'),
                (24, 'bool'),
                (23, 'complex64'),
                (63, 'complex128'),
                (64, 'int8')]:

            n = each[0]
            b = n_byte_align_empty(shape, n, dtype=each[1])
            self.assertTrue(b.ctypes.data%n == 0)
            self.assertTrue(b.dtype == each[1])            

    def test_n_byte_align(self):
        shape = (10,10)
        a = numpy.random.randn(*shape)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            b = n_byte_align(a, n)
            self.assertTrue(b.ctypes.data%n == 0)


test_cases = (
        Complex64FFTWTest,
        Complex128FFTWTest,
        ComplexLongDoubleFFTWTest,
        RealForwardDoubleFFTWTest,
        RealForwardSingleFFTWTest,
        RealForwardLongDoubleFFTWTest,
        RealBackwardDoubleFFTWTest,
        RealBackwardSingleFFTWTest,
        RealBackwardLongDoubleFFTWTest,
        NByteAlignTest,
        )

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
