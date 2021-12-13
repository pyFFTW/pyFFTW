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

from .test_pyfftw_base import run_test_suites, miss, require, np_fft

import unittest

from .test_pyfftw_complex import Complex64FFTWTest

class RealForwardDoubleFFTWTest(Complex64FFTWTest):

    def setUp(self):
        require(self, '64')

        self.input_dtype = numpy.float64
        self.output_dtype = numpy.complex128
        self.np_fft_comparison = np_fft.rfft

        self.direction = 'FFTW_FORWARD'

    def make_shapes(self):
        self.input_shapes = {
                'small_1d': (16,),
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (5, 256, 2048)}

        self.output_shapes = {
                'small_1d': (9,),
                '1d': (1025,),
                '2d': (256, 1025),
                '3d': (5, 256, 1025)}

    def create_test_arrays(self, input_shape, output_shape, axes=None):
        a = self.input_dtype(numpy.random.randn(*input_shape))

        b = self.output_dtype(numpy.random.randn(*output_shape)
                +1j*numpy.random.randn(*output_shape))

        return a, b

    def reference_fftn(self, a, axes):

        return np_fft.rfftn(a, axes=axes)

    def test_wrong_direction_fail(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']

        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        with self.assertRaisesRegex(ValueError, 'Invalid direction'):
            FFTW(a, b, direction='FFTW_BACKWARD')

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

@unittest.skipIf(*miss('32'))
class RealForwardSingleFFTWTest(RealForwardDoubleFFTWTest):

    def setUp(self):

        self.input_dtype = numpy.float32
        self.output_dtype = numpy.complex64
        self.np_fft_comparison = np_fft.rfft

        self.direction = 'FFTW_FORWARD'

@unittest.skipIf(*miss('ld'))
class RealForwardLongDoubleFFTWTest(RealForwardDoubleFFTWTest):

    def setUp(self):

        self.input_dtype = numpy.longdouble
        self.output_dtype = numpy.clongdouble
        self.np_fft_comparison = np_fft.rfft

        self.direction = 'FFTW_FORWARD'

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time(self):
        pass

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time_with_array_update(self):
        pass

    def reference_fftn(self, a, axes):

        a = numpy.float64(a)
        return np_fft.rfftn(a, axes=axes)

test_cases = (
        RealForwardDoubleFFTWTest,
        RealForwardSingleFFTWTest,
        RealForwardLongDoubleFFTWTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)

del Complex64FFTWTest
