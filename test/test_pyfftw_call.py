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


from pyfftw import (
        FFTW, empty_aligned, byte_align)

from .test_pyfftw_base import run_test_suites, miss, require
import numpy
import unittest

@unittest.skipIf(*miss('64'))
class FFTWCallTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(FFTWCallTest, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def setUp(self):

        self.input_array = empty_aligned((256, 512), dtype='complex128', n=16)
        self.output_array = empty_aligned((256, 512), dtype='complex128', n=16)

        self.fft = FFTW(self.input_array, self.output_array)

        self.input_array[:] = (numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape))


    def test_call(self):
        '''Test a call to an instance of the class.
        '''

        self.input_array[:] = (numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))


    def test_call_with_positional_input_update(self):
        '''Test the class call with a positional input update.
        '''

        input_array = byte_align(
                (numpy.random.randn(*self.input_array.shape)
                    + 1j*numpy.random.randn(*self.input_array.shape)), n=16)

        output_array = self.fft(byte_align(input_array.copy(), n=16)).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

    def test_call_with_keyword_input_update(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = byte_align(
                numpy.random.randn(*self.input_array.shape)
                    + 1j*numpy.random.randn(*self.input_array.shape), n=16)

        output_array = self.fft(
            input_array=byte_align(input_array.copy(), n=16)).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))


    def test_call_with_keyword_output_update(self):
        '''Test the class call with a keyword output update.
        '''
        output_array = byte_align(
            (numpy.random.randn(*self.output_array.shape)
                + 1j*numpy.random.randn(*self.output_array.shape)), n=16)

        returned_output_array = self.fft(
                output_array=byte_align(output_array.copy(), n=16)).copy()

        self.fft.update_arrays(self.input_array, output_array)
        self.fft.execute()

        self.assertTrue(
                numpy.alltrue(returned_output_array == output_array))

    def test_call_with_positional_updates(self):
        '''Test the class call with a positional array updates.
        '''

        input_array = byte_align((numpy.random.randn(*self.input_array.shape)
            + 1j*numpy.random.randn(*self.input_array.shape)), n=16)

        output_array = byte_align((numpy.random.randn(*self.output_array.shape)
            + 1j*numpy.random.randn(*self.output_array.shape)), n=16)

        returned_output_array = self.fft(
            byte_align(input_array.copy(), n=16),
            byte_align(output_array.copy(), n=16)).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))

    def test_call_with_keyword_updates(self):
        '''Test the class call with a positional output update.
        '''

        input_array = byte_align(
                (numpy.random.randn(*self.input_array.shape)
                    + 1j*numpy.random.randn(*self.input_array.shape)), n=16)

        output_array = byte_align(
                (numpy.random.randn(*self.output_array.shape)
                    + 1j*numpy.random.randn(*self.output_array.shape)), n=16)

        returned_output_array = self.fft(
                output_array=byte_align(output_array.copy(), n=16),
                input_array=byte_align(input_array.copy(), n=16)).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))

    def test_call_with_different_input_dtype(self):
        '''Test the class call with an array with a different input dtype
        '''
        input_array = byte_align(numpy.complex64(
                numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape)), n=16)

        output_array = self.fft(byte_align(input_array.copy(), n=16)).copy()

        _input_array = byte_align(numpy.asarray(input_array,
                dtype=self.input_array.dtype), n=16)

        self.assertTrue(_input_array.dtype != input_array.dtype)

        self.fft.update_arrays(_input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

    def test_call_with_list_input(self):
        '''Test the class call with a list rather than an array
        '''

        output_array = self.fft().copy()

        test_output_array = self.fft(self.input_array.tolist()).copy()

        self.assertTrue(numpy.alltrue(output_array == test_output_array))


    def test_call_with_invalid_update(self):
        '''Test the class call with an invalid update.
        '''

        new_shape = self.input_array.shape + (2, )
        invalid_array = (numpy.random.randn(*new_shape)
                + 1j*numpy.random.randn(*new_shape))

        self.assertRaises(ValueError, self.fft,
                *(),
                **{'output_array':invalid_array})

        self.assertRaises(ValueError, self.fft,
                *(),
                **{'input_array':invalid_array})


    @unittest.skipIf(*miss('32'))
    def test_call_with_auto_input_alignment(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = (numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(
                input_array=byte_align(input_array.copy(), n=16)).copy()

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a = input_array
        a__ = empty_aligned(numpy.prod(a.shape)*a.itemsize+1, dtype='int8',
                            n=16)

        a_ = a__[1:].view(dtype=a.dtype).reshape(*a.shape)
        a_[:] = a

        # Just confirm that a usual update will fail
        self.assertRaisesRegex(ValueError, 'Invalid input alignment',
                self.fft.update_arrays, *(a_, self.output_array))

        self.fft(a_, self.output_array)

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

        # now try with a single byte offset and SIMD off
        ar, ai = numpy.float32(numpy.random.randn(2, 257))
        a = ar[1:] + 1j*ai[1:]

        b = a.copy()

        a_size = len(a.ravel())*a.itemsize

        update_array = numpy.frombuffer(
                numpy.zeros(a_size + 1, dtype='int8')[1:].data,
                dtype=a.dtype).reshape(a.shape)

        fft = FFTW(a, b, flags=('FFTW_UNALIGNED',))
        # Confirm that a usual update will fail (it's not on the
        # byte boundary)
        self.assertRaisesRegex(ValueError, 'Invalid input alignment',
                fft.update_arrays, *(update_array, b))

        fft(update_array, b)

    def test_call_with_invalid_output_striding(self):
        '''Test the class call with an invalid strided output update.
        '''
        # Add an extra dimension to bugger up the striding
        new_shape = self.output_array.shape + (2,)
        output_array = byte_align(numpy.random.randn(*new_shape)
                + 1j*numpy.random.randn(*new_shape), n=16)

        self.assertRaisesRegex(ValueError, 'Invalid output striding',
                self.fft, **{'output_array': output_array[:,:,1]})

    def test_call_with_different_striding(self):
        '''Test the input update with different strides to internal array.
        '''
        shape = self.input_array.shape + (2,)

        input_array = byte_align(numpy.random.randn(*shape)
                + 1j*numpy.random.randn(*shape), n=16)

        fft = FFTW(input_array[:,:,0], self.output_array)

        test_output_array = fft().copy()

        new_input_array = byte_align(
                input_array[:, :, 0].copy(), n=16)

        new_output = fft(new_input_array).copy()

        # Test the test!
        self.assertTrue(new_input_array.strides != input_array[:,:,0].strides)

        self.assertTrue(numpy.alltrue(test_output_array == new_output))

    def test_call_with_copy_with_missized_array_error(self):
        '''Force an input copy with a missized array.
        '''
        shape = list(self.input_array.shape + (2,))
        shape[0] += 1

        input_array = byte_align(numpy.random.randn(*shape)
                + 1j*numpy.random.randn(*shape), n=16)

        fft = FFTW(self.input_array, self.output_array)

        self.assertRaisesRegex(ValueError, 'Invalid input shape',
                self.fft, **{'input_array': input_array[:,:,0]})

    def test_call_with_unaligned(self):
        '''Make sure the right thing happens with unaligned data.
        '''
        input_array = (numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(
                input_array=byte_align(input_array.copy(), n=16)).copy()

        input_array = byte_align(input_array, n=16)
        output_array = byte_align(output_array, n=16)

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a = byte_align(input_array.copy(), n=16)
        a__ = empty_aligned(numpy.prod(a.shape)*a.itemsize+1, dtype='int8',
                            n=16)

        a_ = a__[1:].view(dtype=a.dtype).reshape(*a.shape)
        a_[:] = a

        # Create a different second array the same way
        b = byte_align(output_array.copy(), n=16)
        b__ = empty_aligned(numpy.prod(b.shape)*a.itemsize+1, dtype='int8',
                            n=16)

        b_ = b__[1:].view(dtype=b.dtype).reshape(*b.shape)
        b_[:] = a

        # Set up for the first array
        fft = FFTW(input_array, output_array)
        a_[:] = a
        output_array = fft().copy()

        # Check a_ is not aligned...
        self.assertRaisesRegex(ValueError, 'Invalid input alignment',
                self.fft.update_arrays, *(a_, output_array))

        # and b_ too
        self.assertRaisesRegex(ValueError, 'Invalid output alignment',
                self.fft.update_arrays, *(input_array, b_))

        # But it should still work with the a_
        fft(a_)

        # However, trying to update the output will raise an error
        self.assertRaisesRegex(ValueError, 'Invalid output alignment',
                self.fft.update_arrays, *(input_array, b_))

        # Same with SIMD off
        fft = FFTW(input_array, output_array, flags=('FFTW_UNALIGNED',))
        fft(a_)
        self.assertRaisesRegex(ValueError, 'Invalid output alignment',
                self.fft.update_arrays, *(input_array, b_))

    def test_call_with_normalisation_on(self):
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        ifft = FFTW(self.output_array, _input_array,
                direction='FFTW_BACKWARD')

        self.fft(normalise_idft=True) # Shouldn't make any difference
        ifft(normalise_idft=True)

        self.assertTrue(numpy.allclose(self.input_array, _input_array))

    def test_call_with_normalisation_off(self):
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        ifft = FFTW(self.output_array, _input_array,
                direction='FFTW_BACKWARD')

        self.fft(normalise_idft=True) # Shouldn't make any difference
        ifft(normalise_idft=False)

        _input_array /= ifft.N

        self.assertTrue(numpy.allclose(self.input_array, _input_array))

    def test_call_with_normalisation_default(self):
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        ifft = FFTW(self.output_array, _input_array,
                direction='FFTW_BACKWARD')

        self.fft()
        ifft()

        # Scaling is performed by default
        self.assertTrue(numpy.allclose(self.input_array, _input_array))

    @unittest.skipIf(*miss('32', '64'))
    def test_call_with_normalisation_precision(self):
        '''The normalisation should use a double precision scaling.
        '''
        # Should be the case for double inputs...
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        self.fft()
        ifft = FFTW(self.output_array, _input_array,
                direction='FFTW_BACKWARD')

        ref_output = ifft(normalise_idft=False).copy()/numpy.float64(ifft.N)
        test_output = ifft(normalise_idft=True).copy()

        self.assertTrue(numpy.alltrue(ref_output == test_output))

        # ... and single inputs.
        _input_array = empty_aligned((256, 512), dtype='complex64', n=16)

        ifft = FFTW(numpy.array(self.output_array, _input_array.dtype),
                    _input_array,
                    direction='FFTW_BACKWARD')

        ref_output = ifft(normalise_idft=False).copy()/numpy.float64(ifft.N)
        test_output = ifft(normalise_idft=True).copy()

        self.assertTrue(numpy.alltrue(ref_output == test_output))

    def test_call_with_ortho_on(self):
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        ifft = FFTW(self.output_array, _input_array,
                    direction='FFTW_BACKWARD')

        self.fft(ortho=True, normalise_idft=False)

        # ortho case preserves the norm in forward direction
        self.assertTrue(
            numpy.allclose(numpy.linalg.norm(self.input_array),
                           numpy.linalg.norm(self.output_array)))

        ifft(ortho=True, normalise_idft=False)

        # ortho case preserves the norm in backward direction
        self.assertTrue(
            numpy.allclose(numpy.linalg.norm(_input_array),
                           numpy.linalg.norm(self.output_array)))

        self.assertTrue(numpy.allclose(self.input_array, _input_array))

        # cant select both ortho and normalise_idft
        self.assertRaisesRegex(ValueError, 'Invalid options',
                               self.fft, normalise_idft=True, ortho=True)
        # cant specify orth=True with default normalise_idft=True
        self.assertRaisesRegex(ValueError, 'Invalid options',
                               self.fft, ortho=True)

    def test_call_with_ortho_off(self):
        _input_array = empty_aligned((256, 512), dtype='complex128', n=16)

        ifft = FFTW(self.output_array, _input_array,
                    direction='FFTW_BACKWARD')

        self.fft(ortho=False)
        ifft(ortho=False)

        # Scaling by normalise_idft is performed by default
        self.assertTrue(numpy.allclose(self.input_array, _input_array))

test_cases = (
        FFTWCallTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
