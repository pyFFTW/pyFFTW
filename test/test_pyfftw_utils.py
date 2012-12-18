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


import unittest
from pyfftw import pyfftw
import platform
import os

def get_cpus_info():
    
    if 'Linux' in platform.system():
        # A simple /proc/cpuinfo parser
        f = open(os.path.join('/', 'proc','cpuinfo'), 'r')

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
                self.assertTrue(pyfftw.get_alignment() == 32)
            elif 'sse' in each_cpu['flags']:
                self.assertTrue(pyfftw.get_alignment() == 16)
            else:
                self.assertTrue(pyfftw.get_alignment() == 1)

    @unittest.skipIf('Linux' not in platform.system(),
            'Skipping as we only have it set up for Linux at present.')
    def test_correct_cpuinfo_lookup(self):
        utils_cpuinfo = pyfftw.get_cpuinfo()
        cpus_info = get_cpus_info()
        
        for each_cpu in cpus_info:
            for each_flag in utils_cpuinfo:
                if utils_cpuinfo[each_flag]:
                    self.assertTrue(each_flag in each_cpu['flags'])
                else:
                    self.assertTrue(each_flag not in each_cpu['flags'])

    def test_cpuinfo_lookup(self):
        cpuinfo = pyfftw.get_cpuinfo()

        for each_flag in cpuinfo:
            self.assertTrue(each_flag in ('avx', 'sse', 'sse2'))

test_cases = (
        UtilsTest,)

if __name__ == '__main__':

    suite = unittest.TestSuite()

    for test_class in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    unittest.TextTestRunner(verbosity=2).run(suite)
