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
        export_wisdom, import_wisdom, forget_wisdom,
        _supported_types, _supported_nptypes_complex)

from .test_pyfftw_base import run_test_suites

import numpy
import pickle
import sys

import unittest

class FFTWWisdomTest(unittest.TestCase):

    def generate_wisdom(self):
        for each_dtype in _supported_nptypes_complex:
            n = 16
            a = empty_aligned((1,1024), each_dtype, n=n)
            b = empty_aligned(a.shape, dtype=a.dtype, n=n)
            fft = FFTW(a,b)


    def compare_single(self, prec, before, after):
        # skip over unsupported data types where wisdom is the empty string
        if  prec in _supported_types:
            # wisdom not updated for ld, at least on appveyor; e.g.
            # https://ci.appveyor.com/project/hgomersall/pyfftw/build/job/vweyed25jx8oxxcb
            if prec == 'ld' and sys.platform.startswith("win"):
                pass
            else:
                self.assertNotEqual(before, after)
        else:
            self.assertEqual(before, b'')
            self.assertEqual(before, after)


    def compare(self, before, after):
        for prec, ind in zip(['64', '32', 'ld'], [0,1,2]):
            self.compare_single(prec, before[ind], after[ind])


    def test_export(self):

        forget_wisdom()

        before_wisdom = export_wisdom()

        self.generate_wisdom()

        after_wisdom = export_wisdom()

        self.compare(before_wisdom, after_wisdom)

    def test_import(self):

        forget_wisdom()

        self.generate_wisdom()

        after_wisdom = export_wisdom()

        forget_wisdom()
        before_wisdom = export_wisdom()

        success = import_wisdom(after_wisdom)

        self.compare(before_wisdom, after_wisdom)

        self.assertEqual(success, tuple([x in _supported_types for x in ['64', '32', 'ld']]))


test_cases = (
        FFTWWisdomTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
