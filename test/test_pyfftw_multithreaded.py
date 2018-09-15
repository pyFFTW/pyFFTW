# Copyright 2014 Knowledge Economy Developments Ltd
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

from pyfftw import FFTW
import numpy
from timeit import Timer

from .test_pyfftw_base import run_test_suites, miss, np_fft

import unittest

from .test_pyfftw_base import FFTWBaseTest

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

@unittest.skipIf(*miss('64'))
class Complex128MultiThreadedTest(Complex64MultiThreadedTest):

    def setUp(self):

        self.input_dtype = numpy.complex128
        self.output_dtype = numpy.complex128
        self.np_fft_comparison = np_fft.fft
        return

@unittest.skipIf(*miss('ld'))
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
        return np_fft.fftn(a, axes=axes)

test_cases = (
        Complex64MultiThreadedTest,
        Complex128MultiThreadedTest,
        ComplexLongDoubleMultiThreadedTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
