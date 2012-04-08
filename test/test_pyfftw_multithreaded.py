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

class Complex64MultiThreadedTest(FFTWBaseTest):
    
    def run_multithreaded_test(self, threads):
        in_shape = self.input_shapes['2d'];
        out_shape = self.output_shapes['2d']
        
        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        fft, ifft = self.run_validate_fft(a, b, axes, threads=threads)

        fft_, ifft_ = self.run_validate_fft(a, b, axes, threads=1)

        self.timer_routine(fft.execute, fft_.execute, 
                comparison_string='singled threaded')
        self.assertTrue(True)


    def test_2_threads(self):
        self.run_multithreaded_test(2)

    def test_4_threads(self):
        self.run_multithreaded_test(4)

    def test_7_threads(self):
        self.run_multithreaded_test(7)        

    def test_25_threads(self):
        self.run_multithreaded_test(25)        

class Complex128MultiThreadedTest(Complex64MultiThreadedTest):
    
    def setUp(self):

        self.input_dtype = numpy.complex128
        self.output_dtype = numpy.complex128
        self.np_fft_comparison = numpy.fft.fft        
        return

class ComplexLongDoubleMultiThreadedTest(Complex64MultiThreadedTest):
    
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

test_cases = (
        Complex64MultiThreadedTest,
        Complex128MultiThreadedTest,
        ComplexLongDoubleMultiThreadedTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
