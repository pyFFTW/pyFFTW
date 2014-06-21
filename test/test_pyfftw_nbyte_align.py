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
