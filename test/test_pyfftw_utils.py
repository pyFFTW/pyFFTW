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

