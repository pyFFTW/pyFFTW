# Copyright 2017 Knowledge Economy Developments Ltd
#
# Henry Gomersall
# heng@kedevelopments.co.uk
# Frederik Beaujean
# Frederik.Beaujean@lmu.de
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
    interfaces,
    _all_types_np, _all_types_human_readable, _supported_types
    )
from pyfftw.builders._utils import _rc_dtype_pairs
import numpy as np
import unittest
import warnings

@unittest.skipIf(len(_all_types_human_readable) == len(_supported_types), "All data types available")
class FFTWPartialTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(FFTWPartialTest, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def test_failure(self):
        for dtype, npdtype in zip(['32', '64', 'ld'], [np.complex64, np.complex128, np.clongdouble]):
            if dtype == 'ld' and np.dtype(np.clongdouble) == np.dtype(np.complex128):
                # skip this test on systems where clongdouble is complex128
                continue
            if dtype not in _supported_types:
                a = empty_aligned((1,1024), npdtype, n=16)
                b = empty_aligned(a.shape, dtype=a.dtype, n=16)
                msg = "Rebuild pyFFTW with support for %s precision!" % _all_types_human_readable[dtype]
                with self.assertRaisesRegex(NotImplementedError, msg):
                    FFTW(a,b)


    def conversion(self, missing, alt1, alt2):
        '''If the ``missing`` precision is not available, the builder should convert to
           ``alt1`` precision. If that isn't available either, it should fall back to
           ``alt2``. If input precision is lost, a warning should be emitted.

        '''

        missing, alt1, alt2 = [np.dtype(x) for x in (missing, alt1, alt2)]
        if _all_types_np[missing]  in _supported_types:
            return

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            itemsize = alt1.itemsize
            a = empty_aligned((1, 512), dtype=missing)
            b = interfaces.numpy_fft.fft(a)
            res = _rc_dtype_pairs.get(alt1.char, None)
            if res is not None:
                self.assertEqual(b.dtype, res)
            else:
                itemsize = alt2.itemsize
                self.assertEqual(b.dtype, _rc_dtype_pairs[alt2.char])

            if itemsize < missing.itemsize:
                print(itemsize, missing.itemsize)
                assert len(w) == 1
                assert "Narrowing conversion" in str(w[-1].message)
                print("Found narrowing conversion from %d to %d bytes" % (missing.itemsize, itemsize))
            else:
                assert len(w) == 0


    def test_conversion(self):
        self.conversion('float32', 'float64', 'longdouble')
        self.conversion('float64', 'longdouble', 'single')
        self.conversion('longdouble', 'float64', 'float32')

test_cases = (
        FFTWPartialTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
