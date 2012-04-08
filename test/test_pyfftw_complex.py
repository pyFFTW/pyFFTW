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

from test_pyfftw_base import FFTWBaseTest

class Complex64FFTWTest(FFTWBaseTest):

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

test_cases = (
        Complex64FFTWTest,
        Complex128FFTWTest,
        ComplexLongDoubleFFTWTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)

