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
        FFTW, n_byte_align_empty, 
        export_wisdom, import_wisdom, forget_wisdom)

import numpy
import cPickle

import unittest

class FFTWWisdomTest(unittest.TestCase):
    
    def generate_wisdom(self):
        for each_dtype in (numpy.complex128, numpy.complex64, 
                numpy.clongdouble):

            a = n_byte_align_empty((1,1024), 16, each_dtype)
            b = n_byte_align_empty(a.shape, 16, dtype=a.dtype)
            fft = FFTW(a,b)


    def test_export(self):

        forget_wisdom()

        before_wisdom = export_wisdom()

        self.generate_wisdom()

        after_wisdom = export_wisdom()

        for n in range(0,2):
            self.assertNotEqual(before_wisdom[n], after_wisdom[n])
    
    def test_import(self):

        forget_wisdom()

        self.generate_wisdom()

        after_wisdom = export_wisdom()

        forget_wisdom()
        before_wisdom = export_wisdom()

        success = import_wisdom(after_wisdom)

        for n in range(0,2):
            self.assertNotEqual(before_wisdom[n], after_wisdom[n])

        self.assertEqual(success, (True, True, True))


test_cases = (
        FFTWWisdomTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
