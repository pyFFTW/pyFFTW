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

from pyfftw import (
        FFTW, n_byte_align_empty, is_n_byte_aligned, simd_alignment)
import pyfftw

from .test_pyfftw_base import run_test_suites

import unittest
import numpy

# FFTW tests that don't seem to fit anywhere else

class FFTWMiscTest(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):

        super(FFTWMiscTest, self).__init__(*args, **kwargs)

        # Assume python 3, but keep backwards compatibility
        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp


    def setUp(self):

        self.input_array = n_byte_align_empty((256, 512), 16, 
                dtype='complex128')
        self.output_array = n_byte_align_empty((256, 512), 16,
                dtype='complex128')

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

    def test_flags(self):
        '''Test to see if the flags are correct
        '''
        fft = FFTW(self.input_array, self.output_array)
        self.assertEqual(fft.flags, ('FFTW_MEASURE',))

        fft = FFTW(self.input_array, self.output_array, 
                flags=('FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED'))
        self.assertEqual(fft.flags, ('FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED'))

        # Test an implicit flag
        _input_array = n_byte_align_empty(256, 16, dtype='complex64')
        _output_array = n_byte_align_empty(256, 16, dtype='complex64')

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

    def test_differing_aligned_arrays_update(self):
        '''Test to see if the alignment code is working as expected
        '''

        # Start by creating arrays that are only on various byte 
        # alignments (4, 16 and 32)
        _input_array = n_byte_align_empty(
                len(self.input_array.ravel())*2+5,
                32, dtype='float32')
        _output_array = n_byte_align_empty(
                len(self.output_array.ravel())*2+5,
                32, dtype='float32')

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
        self.assertTrue(is_n_byte_aligned(input_arrays[4], 4))
        self.assertTrue(is_n_byte_aligned(output_arrays[4], 4))

        # ...and on 16...
        self.assertFalse(is_n_byte_aligned(input_arrays[4], 16))
        self.assertFalse(is_n_byte_aligned(output_arrays[4], 16))
        self.assertTrue(is_n_byte_aligned(input_arrays[16], 16))
        self.assertTrue(is_n_byte_aligned(output_arrays[16], 16))

        # ...and on 32...
        self.assertFalse(is_n_byte_aligned(input_arrays[16], 32))
        self.assertFalse(is_n_byte_aligned(output_arrays[16], 32))
        self.assertTrue(is_n_byte_aligned(input_arrays[32], 32))
        self.assertTrue(is_n_byte_aligned(output_arrays[32], 32))

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

        self.assertIs(self.input_array, self.fft.get_input_array())

    def test_get_output_array(self):
        '''Test to see the get_output_array method returns the correct thing
        '''

        self.assertIs(self.output_array, self.fft.get_output_array())

test_cases = (
        FFTWMiscTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
