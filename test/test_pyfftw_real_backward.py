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


from pyfftw import FFTW, forget_wisdom
import numpy
from timeit import Timer
import time

from .test_pyfftw_base import run_test_suites, miss, require, np_fft

import unittest

from .test_pyfftw_complex import Complex64FFTWTest

class RealBackwardDoubleFFTWTest(Complex64FFTWTest):

    def setUp(self):
        require(self, '64')

        self.input_dtype = numpy.complex128
        self.output_dtype = numpy.float64
        self.np_fft_comparison = np_fft.irfft

        self.direction = 'FFTW_BACKWARD'

    def make_shapes(self):

        self.input_shapes = {
                'small_1d': (9,),
                '1d': (1025,),
                '2d': (256, 1025),
                '3d': (5, 256, 1025)}

        self.output_shapes = {
                'small_1d': (16,),
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (5, 256, 2048)}

    def test_invalid_args_raise(self):
        in_shape = self.input_shapes['1d']
        out_shape = self.output_shapes['1d']

        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # Note "thread" is incorrect, it should be "threads"
        self.assertRaises(TypeError, FFTW, a, b, axes,
                          direction='FFTW_BACKWARD', thread=4)

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
            force_unaligned_data=False, create_array_copies=True,
            threads=1, flags=('FFTW_ESTIMATE',)):
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

        flags = list(flags)

        if force_unaligned_data:
            flags.append('FFTW_UNALIGNED')

        if ifft == None:
            ifft = FFTW(a, b, axes=axes, direction='FFTW_BACKWARD',
                    flags=flags, threads=threads)
        else:
            ifft.update_arrays(a,b)

        if fft == None:
            fft = FFTW(b, a, axes=axes, direction='FFTW_FORWARD',
                    flags=flags, threads=threads)
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
        self.assertEqual(ifft.N, scaling)
        self.assertEqual(fft.N, scaling)

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
        return np_fft.irfftn(a, axes=axes)

    def test_wrong_direction_fail(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']

        axes=(-1,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        with self.assertRaisesRegex(ValueError, 'Invalid direction'):
            FFTW(a, b, direction='FFTW_FORWARD')

    def test_planning_time_limit(self):
        in_shape = self.input_shapes['1d']
        out_shape = self.output_shapes['1d']

        axes=(0,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        # run this a few times
        runs = 10
        t1 = time.time()
        for n in range(runs):
            forget_wisdom()
            fft = FFTW(b, a, axes=axes)

        unlimited_time = (time.time() - t1)/runs

        time_limit = (unlimited_time)/8

        # Now do it again but with an upper limit on the time
        t1 = time.time()
        for n in range(runs):
            forget_wisdom()
            fft = FFTW(b, a, axes=axes, planning_timelimit=time_limit)

        limited_time = (time.time() - t1)/runs

        import sys
        if sys.platform == 'win32':
            # Give a 4x margin on windows. The timers are low
            # precision and FFTW seems to take longer anyway
            self.assertTrue(limited_time < time_limit*4)
        else:
            # Otherwise have a 2x margin
            self.assertTrue(limited_time < time_limit*2)

    def test_invalid_planning_time_limit(self):
        in_shape = self.input_shapes['1d']
        out_shape = self.output_shapes['1d']

        axes=(0,)
        a, b = self.create_test_arrays(in_shape, out_shape)

        self.assertRaisesRegex(TypeError, 'Invalid planning timelimit',
                FFTW, *(b, a, axes), **{'planning_timelimit': 'foo'})

    def test_default_args(self):
        in_shape = self.input_shapes['2d']
        out_shape = self.output_shapes['2d']

        a, b = self.create_test_arrays(in_shape, out_shape)

        # default args should fail for backwards transforms
        # (as the default is FFTW_FORWARD)
        with self.assertRaisesRegex(ValueError, 'Invalid direction'):
            FFTW(a, b)

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

    def test_non_monotonic_increasing_axes(self):
        super(RealBackwardDoubleFFTWTest,
                self).test_non_monotonic_increasing_axes()

@unittest.skipIf(*miss('32'))
class RealBackwardSingleFFTWTest(RealBackwardDoubleFFTWTest):

    def setUp(self):

        self.input_dtype = numpy.complex64
        self.output_dtype = numpy.float32
        self.np_fft_comparison = np_fft.irfft

        self.direction = 'FFTW_BACKWARD'

@unittest.skipIf(*miss('ld'))
class RealBackwardLongDoubleFFTWTest(RealBackwardDoubleFFTWTest):

    def setUp(self):

        self.input_dtype = numpy.clongdouble
        self.output_dtype = numpy.longdouble
        self.np_fft_comparison = np_fft.irfft

        self.direction = 'FFTW_BACKWARD'

    def reference_fftn(self, a, axes):

        a = numpy.complex128(a)
        return np_fft.irfftn(a, axes=axes)

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time(self):
        pass

    @unittest.skip('numpy.fft has issues with this dtype.')
    def test_time_with_array_update(self):
        pass

test_cases = (
        RealBackwardDoubleFFTWTest,
        RealBackwardSingleFFTWTest,
        RealBackwardLongDoubleFFTWTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)

del Complex64FFTWTest
