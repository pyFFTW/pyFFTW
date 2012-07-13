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

from pyfftw import (
        FFTW, n_byte_align_empty, n_byte_align)

import numpy
import unittest

class FFTWCallTest(unittest.TestCase):
   
    def setUp(self):

        self.input_array = n_byte_align_empty((256, 512), 16, 
                dtype='complex128')
        self.output_array = n_byte_align_empty((256, 512), 16,
                dtype='complex128')

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

        input_array = n_byte_align(
                (numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = self.fft(n_byte_align(input_array.copy(), 16)).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
        
    def test_call_with_keyword_input_update(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = n_byte_align(
                numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape), 16)

        output_array = self.fft(
            input_array=n_byte_align(input_array.copy(), 16)).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
    
        
    def test_call_with_keyword_output_update(self):
        '''Test the class call with a keyword output update.
        '''
        output_array = n_byte_align(
            (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
                output_array=n_byte_align(output_array.copy(), 16)).copy()

        self.fft.update_arrays(self.input_array, output_array)
        self.fft.execute()

        self.assertTrue(
                numpy.alltrue(returned_output_array == output_array))

    def test_call_with_positional_updates(self):
        '''Test the class call with a positional array updates.
        '''
        
        input_array = n_byte_align((numpy.random.randn(*self.input_array.shape) 
            + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = n_byte_align((numpy.random.randn(*self.output_array.shape) 
            + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
            n_byte_align(input_array.copy(), 16),
            n_byte_align(output_array.copy(), 16)).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))

    def test_call_with_keyword_updates(self):
        '''Test the class call with a positional output update.
        '''
        
        input_array = n_byte_align(
                (numpy.random.randn(*self.input_array.shape) 
                    + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = n_byte_align(
                (numpy.random.randn(*self.output_array.shape)
                    + 1j*numpy.random.randn(*self.output_array.shape)), 16)

        returned_output_array = self.fft(
                output_array=n_byte_align(output_array.copy(), 16),
                input_array=n_byte_align(input_array.copy(), 16)).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))
    
    def test_call_with_different_input_dtype(self):
        '''Test the class call with an array with a different input dtype
        '''
        input_array = n_byte_align(numpy.complex64(
                numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape)), 16)

        output_array = self.fft(n_byte_align(input_array.copy(), 16)).copy()

        _input_array = numpy.asarray(input_array,
                dtype=self.input_array.dtype)

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

    def test_call_with_auto_input_alignment(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(
                input_array=n_byte_align(input_array.copy(), 16)).copy()

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a = input_array
        a__ = n_byte_align_empty(
                numpy.prod(a.shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=a.dtype).reshape(*a.shape)
        a_[:] = a

        # Just confirm that a usual update will fail
        self.assertRaisesRegexp(ValueError, 'Invalid input alignment', 
                self.fft.update_arrays, *(a_, self.output_array))
        
        self.fft(a_, self.output_array)

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
    
    
    def test_call_with_invalid_input_striding(self):
        '''Test the class call with an invalid strided input update.
        '''
        # Add an extra dimension to bugger up the striding
        new_shape = self.input_array.shape + (2,)
        input_array = n_byte_align(numpy.random.randn(*new_shape) 
                + 1j*numpy.random.randn(*new_shape), 16)

        self.assertRaisesRegexp(ValueError, 'Invalid input striding',
                self.fft, *(input_array[:,:,1],))

    def test_call_with_invalid_internal_striding(self):
        '''Test the input update with non C contiguous internal input array.
        '''
        shape = self.input_array.shape + (2,)

        input_array = n_byte_align(numpy.random.randn(*shape) 
                + 1j*numpy.random.randn(*shape), 16)

        fft = FFTW(input_array[:, :, 1], self.output_array)

        # set up a new array with an incorrect dtype to try to force
        # a copy.
        new_input_array = n_byte_align(numpy.complex64(
            numpy.random.randn(*shape) 
            + 1j*numpy.random.randn(*shape)), 16)

        self.assertRaisesRegexp(ValueError, 'Invalid internal striding',
                fft, *(new_input_array[:,:,1],))

    def test_call_with_unaligned(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(
                input_array=n_byte_align(input_array.copy(), 16)).copy()

        # Offset by one from 16 byte aligned to guarantee it's not
        # 16 byte aligned
        a = input_array.copy()
        a__ = n_byte_align_empty(
                numpy.prod(a.shape)*a.itemsize+1, 16, dtype='int8')
        
        a_ = a__[1:].view(dtype=a.dtype).reshape(*a.shape)
        a_[:] = a

        # Create a different second array the same way
        b = input_array.copy()
        b__ = n_byte_align_empty(
                numpy.prod(b.shape)*a.itemsize+1, 16, dtype='int8')
        
        b_ = b__[1:].view(dtype=b.dtype).reshape(*b.shape)
        b_[:] = a

        # Set up for the first array
        fft = FFTW(a_, self.output_array)
        a_[:] = a
        output_array = fft().copy()

        # Check b is not aligned...
        self.assertRaisesRegexp(ValueError, 'Invalid input alignment', 
                self.fft.update_arrays, *(b_, self.output_array))
        
        # But it should still work with the new array
        fft(b_)

        self.assertTrue(numpy.alltrue(output_array == self.output_array))

        
test_cases = (
        FFTWCallTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
