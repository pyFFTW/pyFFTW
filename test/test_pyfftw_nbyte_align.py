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

from pyfftw import n_byte_align, n_byte_align_empty, is_n_byte_aligned
import numpy
from timeit import Timer

from .test_pyfftw_base import run_test_suites

import unittest

class NByteAlignTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(NByteAlignTest, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

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

    def test_integer_shape(self):
        shape = 100
        a = numpy.random.randn(shape)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            b = n_byte_align(a, n)
            self.assertTrue(b.ctypes.data%n == 0)

    def test_is_n_byte_aligned(self):
        a = n_byte_align_empty(100, 16)
        self.assertTrue(is_n_byte_aligned(a, 16))

        a = n_byte_align_empty(100, 5)
        self.assertTrue(is_n_byte_aligned(a, 5))

        a = n_byte_align_empty(100, 16, dtype='float32')[1:]
        self.assertFalse(is_n_byte_aligned(a, 16))
        self.assertTrue(is_n_byte_aligned(a, 4))

    def test_is_n_byte_aligned_fail_with_non_array(self):

        a = [1, 2, 3, 4]
        self.assertRaisesRegex(TypeError, 'Invalid array',
                is_n_byte_aligned, a, 16)

    def test_n_byte_align_fail_with_non_array(self):

        a = [1, 2, 3, 4]
        self.assertRaisesRegex(TypeError, 'Invalid array',
                n_byte_align, a, 16)

    def test_n_byte_align_consistent_data(self):
        shape = (10,10)
        a = numpy.int16(numpy.random.randn(*shape)*16000)
        b = numpy.float64(numpy.random.randn(*shape))
        c = numpy.int8(numpy.random.randn(*shape)*255)

        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            d = n_byte_align(a, n)
            self.assertTrue(numpy.array_equal(a, d))

            d = n_byte_align(b, n)
            self.assertTrue(numpy.array_equal(b, d))

            d = n_byte_align(c, n)
            self.assertTrue(numpy.array_equal(c, d))

    def test_n_byte_align_different_dtypes(self):
        shape = (10,10)
        a = numpy.int16(numpy.random.randn(*shape)*16000)
        b = numpy.float64(numpy.random.randn(*shape))
        c = numpy.int8(numpy.random.randn(*shape)*255)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            d = n_byte_align(a, n)
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.__class__ == a.__class__)

            d = n_byte_align(b, n)
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.__class__ == b.__class__)

            d = n_byte_align(c, n)
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.__class__ == c.__class__)

    def test_n_byte_align_set_dtype(self):
        shape = (10,10)
        a = numpy.int16(numpy.random.randn(*shape)*16000)
        b = numpy.float64(numpy.random.randn(*shape))
        c = numpy.int8(numpy.random.randn(*shape)*255)
        # Test a few alignments
        for n in [3, 7, 9, 16, 24, 23, 63, 64]:
            d = n_byte_align(a, n, dtype='float32')
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.dtype == 'float32')

            d = n_byte_align(b, n, dtype='float32')
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.dtype == 'float32')

            d = n_byte_align(c, n, dtype='float64')
            self.assertTrue(d.ctypes.data%n == 0)
            self.assertTrue(d.dtype == 'float64')


test_cases = (
        NByteAlignTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
