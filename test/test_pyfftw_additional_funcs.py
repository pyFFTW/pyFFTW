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

import unittest
import numpy

class FFTWAdditionalFuncsTest(unittest.TestCase):
   
    def setUp(self):

        self.input_array = n_byte_align_empty((256, 512), 16, 
                dtype='complex128')
        self.output_array = n_byte_align_empty((256, 512), 16,
                dtype='complex128')

        self.fft = FFTW(self.input_array, self.output_array)

        self.output_array[:] = (numpy.random.randn(*self.output_array.shape) 
                + 1j*numpy.random.randn(*self.output_array.shape))

    def test_get_input_array(self):
        '''Test to see the get_input_array method returns the correct thing
        '''

        self.assertIs(self.input_array, self.fft.get_input_array())

    def test_get_output_array(self):
        '''Test to see the get_output_array method returns the correct thing
        '''

        self.assertIs(self.output_array, self.fft.get_output_array())

test_cases = (
        FFTWAdditionalFuncs,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
