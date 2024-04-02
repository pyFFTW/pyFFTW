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


from .test_pyfftw_base import run_test_suites

import numpy
import unittest
import pyfftw

class NoGILTest(unittest.TestCase):


    def __init__(self, *args, **kwargs):

        super(NoGILTest, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def setUp(self):

        self.input_array = pyfftw.empty_aligned((256, 512), dtype='complex128', n=16)
        self.output_array = pyfftw.empty_aligned((256, 512), dtype='complex128', n=16)

        self.fft = pyfftw.FFTW(self.input_array, self.output_array)

        self.input_array[:] = (numpy.random.randn(*self.input_array.shape)
                + 1j*numpy.random.randn(*self.input_array.shape))

    def test_call_with_keyword_output_update(self):
        '''Test a call to an instance of the class.
        '''

        output_array = pyfftw.byte_align(
            (numpy.random.randn(*self.output_array.shape)
                + 1j*numpy.random.randn(*self.output_array.shape)), n=16)

        returned_output_array = self.fft(
                output_array=pyfftw.byte_align(output_array.copy(), n=16)).copy()

        self.fft.update_arrays(self.input_array, output_array)
        self.fft.execute()
        
        self.assertTrue(
                numpy.all(returned_output_array == output_array))

    def test_call_with_keyword_output_update_nogil(self):
        '''Test a call to an instance of the class.
        '''

        output_array = pyfftw.byte_align(
            (numpy.random.randn(*self.output_array.shape)
                + 1j*numpy.random.randn(*self.output_array.shape)), n=16)

        returned_output_array = self.fft(
                output_array=pyfftw.byte_align(output_array.copy(), n=16)).copy()

        self.fft.update_arrays(self.input_array, output_array)
        
        pyfftw.execute_in_nogil(self.fft)

        self.assertTrue(
                numpy.all(returned_output_array == output_array))


test_cases = (
        NoGILTest)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
