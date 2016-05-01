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

test_cases = (
        UtilsTest,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
