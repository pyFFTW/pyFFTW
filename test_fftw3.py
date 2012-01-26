from fftw3 import ComplexFFTW
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
try:
    from __main__ import timer_test_setup
except:
    from test_fftw3 import timer_test_setup

fft_class,a,b= timer_test_setup()
'''
def timer():
    from timeit import Timer
    N = 100
    t = Timer(stmt="fft_class.execute()", setup=timer_setup)
        
    print 'One run: '+ ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'

def timer_with_array_update():
    from timeit import Timer
    N = 100
    # We can update the arrays with the original arrays. We're
    # really just trying to test the code path, which should be
    # fixed regardless of the array size.
    t = Timer(
            stmt="fft_class.update_arrays(a,b); fft_class.execute()", 
            setup=timer_setup)
        
    print 'One run: '+ ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'

class ComplexFFTWTest(unittest.TestCase):

    def setUp(self):

        return

    def tearDown(self):
        
        return

    def test_time(self):
        timer()
        self.assertTrue(True)

    def test_time_with_array_update(self):
        timer_with_array_update()
        self.assertTrue(True)

    def run_validate_fft(self, a, b, axes, fft=None, ifft=None):
        ''' Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.
        '''
        a_orig = a.copy()
        foo = False

        if fft == None:
            fft = ComplexFFTW(a,b,axes=axes,
                    direction='FFTW_FORWARD',flags=['FFTW_ESTIMATE'])
        else:
            fft.update_arrays(a,b)
            foo = True

        if ifft == None:
            ifft = ComplexFFTW(b,a,axes=axes,
                    direction='FFTW_BACKWARD',flags=['FFTW_ESTIMATE'])
        else:
            ifft.update_arrays(b,a)


        a[:] = a_orig

        # Test the forward FFT by comparing it to the result from numpy.fft
        fft.execute()
        ref_b = numpy.fft.fftn(a, axes=axes)

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
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)
       
    def test_multiple_1d(self):
        shape = (256, 2048)
        axes=[-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_2d(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)
    
    def test_multiple_2d(self):
        shape = (15, 256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_3d(self):
        shape = (15, 256, 2048)
        axes=[0, 1, 2]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes)

    def test_missized_fail(self):
        shape = (256, 2048)
        axes=[1, 2]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        shape = (257, 2048)
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
    
        self.assertRaises(ValueError, ComplexFFTW, *(a,b))

    def test_f_contiguous_1d(self):
        shape = (256, 2048)
        axes=[-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape)).transpose()
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape)).transpose()

        self.run_validate_fft(a, b, axes)

    def test_non_contiguous_2d(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        # Some arbitrary and crazy slicing
        a_sliced = a[12:200:3, 300:2041:9]
        # b needs to be the same size
        b_sliced = b[20:146:2, 100:1458:7]

        self.run_validate_fft(a_sliced, b_sliced, axes)

    def test_non_complex64_fail(self):
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
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.run_validate_fft(a, b, axes, fft=fft, ifft=ifft)

    def test_update_data_with_stride_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        shape = (258, 2050)
        a_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))[2:,2:]
        b_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))[2:,2:]

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a,b_,axes), **{'fft':fft, 'ifft':ifft})

    def test_update_data_with_shape_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        shape = (250, 2048)
        a_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b_,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})

    def test_update_data_with_alignment_error(self):
        shape = (256, 2048)
        axes=[-2,-1]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        fft, ifft = self.run_validate_fft(a, b, axes)

        shape = (250, 2048)
        a_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b_ = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b_,axes), **{'fft':fft, 'ifft':ifft})

        self.assertRaises(ValueError, self.run_validate_fft, 
                *(a_,b,axes), **{'fft':fft, 'ifft':ifft})

    def test_invalid_axes(self):
        shape = (256, 2048)
        axes=[-3]
        a = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))
        b = numpy.complex64(numpy.random.rand(*shape)
                +1j*numpy.random.rand(*shape))

        self.assertRaises(ValueError, ComplexFFTW, *(a,b,axes))

        axes=[10]
        self.assertRaises(ValueError, ComplexFFTW, *(a,b,axes))

if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(
            ComplexFFTWTest)
    unittest.TextTestRunner(verbosity=2).run(test_suite)

