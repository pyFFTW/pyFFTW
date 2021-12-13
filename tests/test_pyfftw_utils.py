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


from .test_pyfftw_base import run_test_suites

import unittest
import pyfftw
import platform
import os
from numpy.testing import assert_, assert_equal

def get_cpus_info():

    if 'Linux' in platform.system():
        # A simple /proc/cpuinfo parser
        with open(os.path.join('/', 'proc','cpuinfo'), 'r') as f:

            cpus_info = []
            idx = 0
            for line in f.readlines():
                if line.find(':') < 0:
                    idx += 1
                    continue

                key, values = [each.strip() for each in line.split(':')]

                try:
                    cpus_info[idx][key] = values
                except IndexError:
                    cpus_info.append({key: values})

    else:
        cpus_info = None

    return cpus_info

class UtilsTest(unittest.TestCase):

    def setUp(self):

        return

    def tearDown(self):

        return

    @unittest.skipIf('Linux' not in platform.system(),
            'Skipping as we only have it set up for Linux at present.')
    def test_get_alignment(self):
        cpus_info = get_cpus_info()

        for each_cpu in cpus_info:
            if 'avx' in each_cpu['flags']:
                self.assertTrue(pyfftw.simd_alignment == 32)
            elif 'sse' in each_cpu['flags']:
                self.assertTrue(pyfftw.simd_alignment == 16)
            else:
                self.assertTrue(pyfftw.simd_alignment == 1)


class NextFastLenTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super(NextFastLenTest, self).__init__(*args, **kwargs)

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def test_next_fast_len(self):

        def nums():
            for j in range(1, 1000):
                yield j
            yield 2**5 * 3**5 * 4**5 + 1

        for n in nums():
            m = pyfftw.next_fast_len(n)
            msg = "n=%d, m=%d" % (n, m)

            assert_(m >= n, msg)

            # check regularity
            k = m
            num11 = num13 = 0
            # These factors come from the description in the FFTW3 docs:
            #     http://fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs
            for d in [2, 3, 5, 7, 11, 13]:
                while True:
                    a, b = divmod(k, d)
                    if b == 0:
                        k = a
                        if d in [11, 13]:
                            # only allowed to match 11 or 13 once
                            if num11 > 0 or num13 > 0:
                                break
                            if d == 11:
                                num11 += 1
                            else:
                                num13 += 1
                    else:
                        break
            assert_equal(k, 1, err_msg=msg)

    def test_next_fast_len_strict(self):
        strict_test_cases = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 11: 11, 13: 13,
            14: 14, 15: 15, 16: 16, 17: 18, 1021: 1024,

            # 2 * 3 * 5 * 7 * 11
            2310: 2310,
            2310 - 1: 2310,
            # 2 * 3 * 5 * 7 * 13
            2730: 2730,
            2730 - 1: 2730,
            # 2**2 * 3**2 * 5**2 * 7**2 * 11
            485100: 485100,
            485100-1: 485100,
            # 2**2 * 3**2 * 5**2 * 7**2 * 13
            573300: 573300,
            573300-1: 573300,

            # more than one multiple of 11 or 13 is not accepted
            # 2 * 3 * 5 * 7 * 11**2
            25410: 25872,
            # 2 * 3 * 5 * 7 * 13**2
            35490: 35672,
            # 2 * 3 * 5 * 7 * 11 * 13
            30030: 30576,

        }
        for x, y in strict_test_cases.items():
            assert_equal(pyfftw.next_fast_len(x), y)

test_cases = (
        UtilsTest,
        NextFastLenTest)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
