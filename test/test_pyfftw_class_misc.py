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
        FFTW, empty_aligned, is_byte_aligned, simd_alignment)
import pyfftw

from .test_pyfftw_base import run_test_suites, miss, require

import unittest
import numpy
import warnings

# FFTW tests that don't seem to fit anywhere else

class FFTWMiscTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(FFTWMiscTest, self).__init__(*args, **kwargs)

        # Assume python 3, but keep backwards compatibility
        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp


    def setUp(self):
        require(self, '64')

        self.input_array = empty_aligned((256, 512), dtype='complex128', n=16)
        self.output_array = empty_aligned((256, 512), dtype='complex128', n=16)

        self.fft = FFTW(self.input_array, self.output_array)

        self.output_array[:] = (numpy.random.randn(*self.output_array.shape)
                + 1j*numpy.random.randn(*self.output_array.shape))

    def test_aligned_flag(self):
        '''Test to see if the aligned flag is correct
        '''
        fft = FFTW(self.input_array, self.output_array)
        self.assertTrue(fft.simd_aligned)

        fft = FFTW(self.input_array, self.output_array,
                flags=('FFTW_UNALIGNED',))

        self.assertFalse(fft.simd_aligned)

    @unittest.skipIf(*miss('32'))
    def test_flags(self):
        '''Test to see if the flags are correct
        '''
        fft = FFTW(self.input_array, self.output_array)
        self.assertEqual(fft.flags, ('FFTW_MEASURE',))

        fft = FFTW(self.input_array, self.output_array,
                flags=('FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED'))
        self.assertEqual(fft.flags, ('FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED'))

        # Test an implicit flag
        _input_array = empty_aligned(256, dtype='complex64', n=16)
        _output_array = empty_aligned(256, dtype='complex64', n=16)

        # These are guaranteed to be misaligned (due to dtype size == 8)
        input_array = _input_array[:-1]
        output_array = _output_array[:-1]
        u_input_array = _input_array[1:]
        u_output_array = _output_array[1:]

        fft = FFTW(input_array, u_output_array)
        self.assertEqual(fft.flags, ('FFTW_MEASURE', 'FFTW_UNALIGNED'))

        fft = FFTW(u_input_array, output_array)
        self.assertEqual(fft.flags, ('FFTW_MEASURE', 'FFTW_UNALIGNED'))

        fft = FFTW(u_input_array, u_output_array)
        self.assertEqual(fft.flags, ('FFTW_MEASURE', 'FFTW_UNALIGNED'))

    @unittest.skipIf(*miss('32'))
    def test_differing_aligned_arrays_update(self):
        '''Test to see if the alignment code is working as expected
        '''

        # Start by creating arrays that are only on various byte
        # alignments (4, 16 and 32)
        _input_array = empty_aligned(len(self.input_array.ravel())*2+5,
                                     dtype='float32', n=32)
        _output_array = empty_aligned(len(self.output_array.ravel())*2+5,
                                      dtype='float32', n=32)

        _input_array[:] = 0
        _output_array[:] = 0

        input_array_4 = (
                numpy.frombuffer(_input_array[1:-4].data, dtype='complex64')
                .reshape(self.input_array.shape))
        output_array_4 = (
                numpy.frombuffer(_output_array[1:-4].data, dtype='complex64')
                .reshape(self.output_array.shape))

        input_array_16 = (
                numpy.frombuffer(_input_array[4:-1].data, dtype='complex64')
                .reshape(self.input_array.shape))
        output_array_16 = (
                numpy.frombuffer(_output_array[4:-1].data, dtype='complex64')
                .reshape(self.output_array.shape))

        input_array_32 = (
                numpy.frombuffer(_input_array[:-5].data, dtype='complex64')
                .reshape(self.input_array.shape))
        output_array_32 = (
                numpy.frombuffer(_output_array[:-5].data, dtype='complex64')
                .reshape(self.output_array.shape))

        input_arrays = {4: input_array_4,
                16: input_array_16,
                32: input_array_32}

        output_arrays = {4: output_array_4,
                16: output_array_16,
                32: output_array_32}

        alignments = (4, 16, 32)

        # Test the arrays are aligned on 4 bytes...
        self.assertTrue(is_byte_aligned(input_arrays[4], n=4))
        self.assertTrue(is_byte_aligned(output_arrays[4], n=4))

        # ...and on 16...
        self.assertFalse(is_byte_aligned(input_arrays[4], n=16))
        self.assertFalse(is_byte_aligned(output_arrays[4], n=16))
        self.assertTrue(is_byte_aligned(input_arrays[16], n=16))
        self.assertTrue(is_byte_aligned(output_arrays[16], n=16))

        # ...and on 32...
        self.assertFalse(is_byte_aligned(input_arrays[16], n=32))
        self.assertFalse(is_byte_aligned(output_arrays[16], n=32))
        self.assertTrue(is_byte_aligned(input_arrays[32], n=32))
        self.assertTrue(is_byte_aligned(output_arrays[32], n=32))

        if len(pyfftw.pyfftw._valid_simd_alignments) > 0:
            max_align = pyfftw.pyfftw._valid_simd_alignments[0]
        else:
            max_align = simd_alignment

        for in_align in alignments:
            for out_align in alignments:
                expected_align = min(in_align, out_align, max_align)
                fft = FFTW(input_arrays[in_align], output_arrays[out_align])

                self.assertTrue(fft.input_alignment == expected_align)
                self.assertTrue(fft.output_alignment == expected_align)

                for update_align in alignments:

                    if update_align < expected_align:
                        # This should fail (not aligned properly)
                        self.assertRaisesRegex(ValueError,
                                'Invalid input alignment',
                                fft.update_arrays,
                                input_arrays[update_align],
                                output_arrays[out_align])

                        self.assertRaisesRegex(ValueError,
                                'Invalid output alignment',
                                fft.update_arrays,
                                input_arrays[in_align],
                                output_arrays[update_align])

                    else:
                        # This should work (and not segfault!)
                        fft.update_arrays(input_arrays[update_align],
                                output_arrays[out_align])
                        fft.update_arrays(input_arrays[in_align],
                                output_arrays[update_align])
                        fft.execute()


    def test_get_input_array(self):
        '''Test to see the get_input_array method returns the correct thing
        '''
        with warnings.catch_warnings(record=True) as w:
            # This method is deprecated, so check the deprecation warning
            # is raised.
            warnings.simplefilter("always")
            input_array = self.fft.get_input_array()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        self.assertIs(self.input_array, input_array)

    def test_get_output_array(self):
        '''Test to see the get_output_array method returns the correct thing
        '''
        with warnings.catch_warnings(record=True) as w:
            # This method is deprecated, so check the deprecation warning
            # is raised.
            warnings.simplefilter("always")
            output_array = self.fft.get_output_array()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

        self.assertIs(self.output_array, output_array)

    def test_input_array(self):
        '''Test to see the input_array property returns the correct thing
        '''
        self.assertIs(self.input_array, self.fft.input_array)

    def test_output_array(self):
        '''Test to see the output_array property returns the correct thing
        '''
        self.assertIs(self.output_array, self.fft.output_array)

    def test_input_strides(self):
        '''Test to see if the input_strides property returns the correct thing
        '''
        self.assertEqual(self.fft.input_strides, self.input_array.strides)

        new_input_array = self.input_array[::2, ::4]
        new_output_array = self.output_array[::2, ::4]

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.input_strides, new_input_array.strides)

    def test_output_strides(self):
        '''Test to see if the output_strides property returns the correct thing
        '''
        self.assertEqual(self.fft.output_strides, self.output_array.strides)

        new_input_array = self.output_array[::2, ::4]
        new_output_array = self.output_array[::2, ::4]

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.output_strides, new_output_array.strides)

    def test_input_shape(self):
        '''Test to see if the input_shape property returns the correct thing
        '''
        self.assertEqual(self.fft.input_shape, self.input_array.shape)

        new_input_array = self.input_array[::2, ::4]
        new_output_array = self.output_array[::2, ::4]

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.input_shape, new_input_array.shape)

    def test_output_strides(self):
        '''Test to see if the output_shape property returns the correct thing
        '''
        self.assertEqual(self.fft.output_shape, self.output_array.shape)

        new_input_array = self.output_array[::2, ::4]
        new_output_array = self.output_array[::2, ::4]

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.output_shape, new_output_array.shape)

    @unittest.skipIf(*miss('32'))
    def test_input_dtype(self):
        '''Test to see if the input_dtype property returns the correct thing
        '''
        self.assertEqual(self.fft.input_dtype, self.input_array.dtype)

        new_input_array = numpy.complex64(self.input_array)
        new_output_array = numpy.complex64(self.output_array)

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.input_dtype, new_input_array.dtype)

    @unittest.skipIf(*miss('32'))
    def test_output_dtype(self):
        '''Test to see if the output_dtype property returns the correct thing
        '''
        self.assertEqual(self.fft.output_dtype, self.output_array.dtype)

        new_input_array = numpy.complex64(self.input_array)
        new_output_array = numpy.complex64(self.output_array)

        new_fft = FFTW(new_input_array, new_output_array)

        self.assertEqual(new_fft.output_dtype, new_output_array.dtype)

    def test_direction_property(self):
        '''Test to see if the direction property returns the correct thing
        '''
        self.assertEqual(self.fft.direction, 'FFTW_FORWARD')

        new_fft = FFTW(self.input_array, self.output_array,
                direction='FFTW_BACKWARD')

        self.assertEqual(new_fft.direction, 'FFTW_BACKWARD')

    def test_axes_property(self):
        '''Test to see if the axes property returns the correct thing
        '''
        self.assertEqual(self.fft.axes, (1,))

        new_fft = FFTW(self.input_array, self.output_array, axes=(-1, -2))
        self.assertEqual(new_fft.axes, (1, 0))

        new_fft = FFTW(self.input_array, self.output_array, axes=(-2, -1))
        self.assertEqual(new_fft.axes, (0, 1))

        new_fft = FFTW(self.input_array, self.output_array, axes=(1, 0))
        self.assertEqual(new_fft.axes, (1, 0))

        new_fft = FFTW(self.input_array, self.output_array, axes=(1,))
        self.assertEqual(new_fft.axes, (1,))

        new_fft = FFTW(self.input_array, self.output_array, axes=(0,))
        self.assertEqual(new_fft.axes, (0,))

    def test_ortho_property(self):
        '''ortho property defaults to False
        '''
        self.assertEqual(self.fft.ortho, False)

        newfft = FFTW(self.input_array, self.output_array, ortho=True,
                      normalise_idft=False)
        self.assertEqual(newfft.ortho, True)

    def test_normalise_idft_property(self):
        '''normalise_idft property defaults to True
        '''
        self.assertEqual(self.fft.normalise_idft, True)

        newfft = FFTW(self.input_array, self.output_array,
                      normalise_idft=False)
        self.assertEqual(newfft.normalise_idft, False)

    def test_invalid_normalisation(self):
        # both ortho and normalise_idft cannot be True
        self.assertRaisesRegex(
            ValueError, 'Invalid options: ortho',
            FFTW, self.input_array, self.output_array,
            direction='FFTW_BACKWARD', ortho=True, normalise_idft=True)

test_cases = (
        FFTWMiscTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
