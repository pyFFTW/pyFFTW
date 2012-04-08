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

from test_pyfftw_complex import Complex64FFTWTest

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

test_cases = (
        RealForwardDoubleFFTWTest,
        RealForwardSingleFFTWTest,
        RealForwardLongDoubleFFTWTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)

del Complex64FFTWTest
