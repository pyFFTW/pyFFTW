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
            force_unaligned_data=False, create_array_copies=True,
            threads=1):
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

test_cases = (
        RealBackwardDoubleFFTWTest,
        RealBackwardSingleFFTWTest,
        RealBackwardLongDoubleFFTWTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)

del Complex64FFTWTest

