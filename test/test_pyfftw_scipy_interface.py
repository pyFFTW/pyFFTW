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

from pyfftw.interfaces import scipy_fftpack
import pyfftw
import numpy
import scipy
import scipy.fftpack
import scipy.signal

import unittest
from .test_pyfftw_base import run_test_suites


'''pyfftw.interfaces.scipy_fftpack just wraps pyfftw.interfaces.numpy_fft.

All the tests here just check that the call is made correctly.
'''

funcs = ('fft','ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 
           'rfft', 'irfft')

acquired_names = ('dct', 'idct', 'diff', 'tilbert', 'itilbert', 'hilbert', 
        'ihilbert', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff', 'shift', 
        'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'convolve', 
        '_fftpack')

def numpy_fft_replacement(a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous):

    return (a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous)

class InterfacesScipyFFTPackTestFFT(unittest.TestCase):
    ''' A really simple test suite to check simple implementation.
    '''

    def test_scipy_overwrite(self):
        scipy_fftn = scipy.signal.signaltools.fftn
        scipy_ifftn = scipy.signal.signaltools.ifftn

        a = pyfftw.n_byte_align_empty((128, 64), 16, dtype='complex128')
        b = pyfftw.n_byte_align_empty((128, 64), 16, dtype='complex128')

        a[:] = (numpy.random.randn(*a.shape) + 
                1j*numpy.random.randn(*a.shape))
        b[:] = (numpy.random.randn(*b.shape) + 
                1j*numpy.random.randn(*b.shape))


        scipy_c = scipy.signal.fftconvolve(a, b)

        scipy.signal.signaltools.fftn = scipy_fftpack.fftn
        scipy.signal.signaltools.ifftn = scipy_fftpack.ifftn

        scipy_replaced_c = scipy.signal.fftconvolve(a, b)

        self.assertTrue(numpy.allclose(scipy_c, scipy_replaced_c))

        scipy.signal.signaltools.fftn = scipy_fftn
        scipy.signal.signaltools.ifftn = scipy_ifftn

    def test_funcs(self):

        for each_func in funcs:
            func_being_replaced = getattr(scipy_fftpack, each_func)

            #create args (8 of them)
            args = []
            for n in range(8):
                args.append(object())

            args = tuple(args)

            try:
                setattr(scipy_fftpack, each_func, 
                        numpy_fft_replacement)

                return_args = getattr(scipy_fftpack, each_func)(*args)
                for n, each_arg in enumerate(args):
                    # Check that what comes back is what is sent
                    # (which it should be)
                    self.assertIs(each_arg, return_args[n])
            except:
                raise

            finally:
                setattr(scipy_fftpack, each_func, 
                        func_being_replaced)

    def test_acquired_names(self):
        for each_name in acquired_names: 

            fftpack_attr = getattr(scipy.fftpack, each_name)
            acquired_attr = getattr(scipy_fftpack, each_name)

            self.assertIs(fftpack_attr, acquired_attr)


test_cases = (
        InterfacesScipyFFTPackTestFFT,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
