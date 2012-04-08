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

from pyfftw import n_byte_align, n_byte_align_empty
import numpy
from timeit import Timer

import unittest

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
        a = numpy.random.randn(*shape)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            b = n_byte_align(a, n)
            self.assertTrue(b.ctypes.data%n == 0)

test_cases = (
        NByteAlignTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
