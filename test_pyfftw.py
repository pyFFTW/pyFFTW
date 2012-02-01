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

from pyfftw import ComplexFFTW, n_byte_align, n_byte_align_empty
import numpy

import unittest

def timer_test_setup(fft_length = 2048, vectors = 256):
    a = numpy.complex64(numpy.random.rand(vectors,fft_length)
            +1j*numpy.random.rand(vectors,fft_length))
    b = numpy.complex64(numpy.random.rand(vectors,fft_length)
            +1j*numpy.random.rand(vectors,fft_length))
    
    fft_class = ComplexFFTW(a,b, flags=['FFTW_MEASURE'])

    # We need to refill a with data as it gets clobbered by
    # initialisation.
    a[:] = numpy.complex64(numpy.random.rand(vectors,fft_length)
            +1j*numpy.random.rand(vectors,fft_length))


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

    def setUp(self):

        self.dtype = numpy.complex64
        return

    def tearDown(self):
        
        return

    def reference_fftn(self, a, axes):
        return numpy.fft.fftn(a, axes=axes)

    def test_time(self):
        timer()
        self.assertTrue(True)

    def test_time_with_array_update(self):
        timer_with_array_update()
        self.assertTrue(True)

    def run_validate_fft(self, a, b, axes, fft=None, ifft=None, 
            force_unaligned_data=False):
        ''' Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.

        If force_unaligned_data is True, the flag FFTW_UNALIGNED
        will be passed to the fftw routines.
        '''
        a_orig = a.copy()
        foo = False

        flags = ['FFTW_ESTIMATE']

        if force_unaligned_data:
            flags.append('FFTW_UNALIGNED')

        if fft == None:
            fft = ComplexFFTW(a,b,axes=axes,
                    direction='FFTW_FORWARD',flags=flags)
        else:
            fft.update_arrays(a,b)
            foo = True

        if ifft == None:
            ifft = ComplexFFTW(b,a,axes=axes,
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
        self.assertTrue(numpy.allclose(b, ref_b, rtol=1e-2))

        # Test the inverse FFT by comparing the result to the starting
        # value (which is scaled as per FFTW being unnormalised).
        ifft.execute()
        # The scaling is the product of the lengths of the fft along
        # the axes along which the fft is taken.
        scaling = numpy.prod(numpy.array(a.shape)[axes])

        self.assertTrue(numpy.allclose(a_orig*scaling, a, rtol=5e-3))

        return fft, ifft

    def test_1d(self):
        shape = (2048,)
        axes=[0]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)
       
    def test_multiple_1d(self):
        shape = (256, 2048)
        axes=[-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_2d(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)
    
    def test_multiple_2d(self):
        shape = (15, 256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_3d(self):
        shape = (15, 256, 2048)
        axes=[0, 1, 2]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_missized_fail(self):
        shape = (256, 2048)
        axes=[1, 2]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        shape = (257, 2048)
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
    
        self.assertRaises(ValueError, ComplexFFTW, *(a,b))

    def test_f_contiguous_1d(self):
        shape = (256, 2048)
        axes=[-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape)).transpose()
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape)).transpose()

        self.run_validate_fft(a, b, axes)

    def test_non_contiguous_2d(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, 300:2041:9]
        # b needs to be the same size
        b_sliced = b[20:146:2, 100:1458:7]

        self.run_validate_fft(a_sliced, b_sliced, axes)

    def test_non_contiguous_2d_in_3d(self):
        shape = (256, 4, 2048)
        axes=[0,2]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, :, 300:2041:9]
        # b needs to be the same size
        b_sliced = b[20:146:2, :, 100:1458:7]

        self.run_validate_fft(a_sliced, b_sliced, axes)

    def test_different_dtypes_fail(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex128(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, ComplexFFTW, *(a,b))

        a = numpy.complex64(a)
        b = numpy.complex128(b)
        self.assertRaises(ValueError, ComplexFFTW, *(a,b))

    def test_update_data(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes, fft=fft, ifft=ifft)

    def test_update_data_with_stride_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        shape = (258, 2050)
        a_ = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))[2:,2:]
        b_ = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))[2:,2:]

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a,b_,axes), **{'fft':fft, 'ifft':ifft})

    def test_update_data_with_shape_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        shape = (250, 2048)
        a_ = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b_ = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b_,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})
    
    def test_update_unaligned_data_with_FFTW_UNALIGNED(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        a = n_byte_align(a, 16)
        b = n_byte_align(b, 16)

        fft, ifft = self.run_validate_fft(a, b, axes, 
                force_unaligned_data=True)

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.dtype).reshape(*shape)
        a_[:] = a 
        
        b__ = n_byte_align_empty(
                numpy.prod(shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.dtype).reshape(*shape)
        b_[:] = b

        self.run_validate_fft(a, b_, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b_, axes, fft=fft, ifft=ifft)

    def test_update_data_with_unaligned_original(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.dtype).reshape(*shape)
        a_[:] = a
        
        b__ = n_byte_align_empty(
                numpy.prod(shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.dtype).reshape(*shape)
        b_[:] = b
        
        fft, ifft = self.run_validate_fft(a_, b_, axes, 
                force_unaligned_data=True)
        
        self.run_validate_fft(a, b_, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b, axes, fft=fft, ifft=ifft)
        self.run_validate_fft(a_, b_, axes, fft=fft, ifft=ifft)


    def test_update_data_with_alignment_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        a = n_byte_align(a, 16)
        b = n_byte_align(b, 16)

        fft, ifft = self.run_validate_fft(a, b, axes)
        
        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a__ = n_byte_align_empty(
                numpy.prod(shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=self.dtype).reshape(*shape)
        a_[:] = a 
        
        b__ = n_byte_align_empty(
                numpy.prod(shape)*b.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=self.dtype).reshape(*shape)
        b_[:] = b
     
        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a,b_,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})

    def test_invalid_axes(self):
        shape = (256, 2048)
        axes=[-3]
        a = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = self.dtype(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, ComplexFFTW, *(a,b,axes))

        axes=[10]
        self.assertRaises(ValueError, ComplexFFTW, *(a,b,axes))

class Complex128FFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.dtype = numpy.complex128
        return

class ComplexLongDoubleFFTWTest(Complex64FFTWTest):
    
    def setUp(self):

        self.dtype = numpy.clongdouble
        return

    def reference_fftn(self, a, axes):

        # numpy.fft.fftn doesn't support complex256 type,
        # so we need to compare to a lower precision type.
        a = numpy.complex128(a)
        return numpy.fft.fftn(a, axes=axes)

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
        a = numpy.random.rand(*shape)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            b = n_byte_align(a, n)
            self.assertTrue(b.ctypes.data%n == 0)


test_cases = (
        Complex64FFTWTest,
        Complex128FFTWTest,
        ComplexLongDoubleFFTWTest,
        NByteAlignTest)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)

