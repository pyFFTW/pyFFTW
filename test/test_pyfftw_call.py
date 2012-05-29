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
        FFTW, n_byte_align_empty)

import numpy
import unittest

class FFTWCallTest(unittest.TestCase):
   
    def setUp(self):

        self.input_array = n_byte_align_empty((256, 512), 16, 
                dtype='complex128')
        self.output_array = n_byte_align_empty((256, 512), 16,
                dtype='complex128')

        self.fft = FFTW(self.input_array, self.output_array)

        self.output_array[:] = (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape))

    
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

        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(input_array.copy()).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
        
    def test_call_with_keyword_input_update(self):
        '''Test the class call with a keyword input update.
        '''
        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = self.fft(input_array=input_array.copy()).copy()

        self.fft.update_arrays(input_array, self.output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(output_array == self.output_array))
    
        
    def test_call_with_keyword_output_update(self):
        '''Test the class call with a keyword output update.
        '''
        output_array = (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape))

        returned_output_array = self.fft(
                output_array=output_array.copy()).copy()

        self.fft.update_arrays(self.input_array, output_array)
        self.fft.execute()

        self.assertTrue(
                numpy.alltrue(returned_output_array == output_array))

    def test_call_with_positional_updates(self):
        '''Test the class call with a positional array updates.
        '''
        
        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape))

        returned_output_array = self.fft(input_array.copy(), 
                output_array.copy()).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))

    def test_call_with_keyword_updates(self):
        '''Test the class call with a positional output update.
        '''
        
        input_array = (numpy.random.randn(*self.input_array.shape) 
                + 1j*numpy.random.randn(*self.input_array.shape))

        output_array = (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape))

        returned_output_array = self.fft(
                output_array=output_array.copy(),
                input_array=input_array.copy()).copy()

        self.fft.update_arrays(input_array, output_array)
        self.fft.execute()

        self.assertTrue(numpy.alltrue(returned_output_array == output_array))
    
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


test_cases = (
        FFTWCallTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
