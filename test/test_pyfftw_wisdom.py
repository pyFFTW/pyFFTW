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
        FFTW, empty_aligned,
        export_wisdom, import_wisdom, forget_wisdom)

from .test_pyfftw_base import run_test_suites

import numpy
import pickle

import unittest

class FFTWWisdomTest(unittest.TestCase):

    def generate_wisdom(self):
        for each_dtype in (numpy.complex128, numpy.complex64,
                numpy.clongdouble):

            a = empty_aligned((1,1024), each_dtype, n=16)
            b = empty_aligned(a.shape, dtype=a.dtype, n=16)
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

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
