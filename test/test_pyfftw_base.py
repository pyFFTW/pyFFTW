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

class FFTWBaseTest(unittest.TestCase):
    
    def reference_fftn(self, a, axes):
        return numpy.fft.fftn(a, axes=axes)

    def __init__(self, *args, **kwargs):

        super(FFTWBaseTest, self).__init__(*args, **kwargs)
        self.make_shapes()

    def setUp(self):
        
        self.input_dtype = numpy.complex64
        self.output_dtype = numpy.complex64
        self.np_fft_comparison = numpy.fft.fft
        return

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

    def create_test_arrays(self, input_shape, output_shape):
        a = self.input_dtype(numpy.random.randn(*input_shape)
                +1j*numpy.random.randn(*input_shape))

        b = self.output_dtype(numpy.random.randn(*output_shape)
                +1j*numpy.random.randn(*output_shape))

        return a, b

    def timer_routine(self, pyfftw_callable, numpy_fft_callable,
            comparison_string='numpy.fft'):

        N = 100

        t = Timer(stmt=pyfftw_callable)
        t_numpy_fft = Timer(stmt=numpy_fft_callable)
    
        t_str = ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'
        t_numpy_str = ("%.2f" % (1000.0/N*t_numpy_fft.timeit(N)))+' ms'

        print ('One run: '+ t_str + \
                ' (versus ' + t_numpy_str + ' for ' + comparison_string + \
                ')')


    def run_validate_fft(self, a, b, axes, fft=None, ifft=None, 
            force_unaligned_data=False, create_array_copies=True, 
            threads=1):
        ''' Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.

        If force_unaligned_data is True, the flag FFTW_UNALIGNED
        will be passed to the fftw routines.

        The threads argument runs the validation with multiple threads.
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
            fft = FFTW(a,b,axes=axes, direction='FFTW_FORWARD',
                    flags=flags, threads=threads)
        else:
            fft.update_arrays(a,b)

        if ifft == None:
            ifft = FFTW(b, a, axes=axes, direction='FFTW_BACKWARD',
                    flags=flags, threads=threads)
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


