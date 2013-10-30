# Copyright 2013 Knowledge Economy Developments Ltd
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
import test_pyfftw_numpy_interface

'''pyfftw.interfaces.scipy_fftpack just wraps pyfftw.interfaces.numpy_fft.

All the tests here just check that the call is made correctly.
'''

funcs = ('fft','ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 
           'rfft', 'irfft')

acquired_names = ('dct', 'idct', 'diff', 'tilbert', 'itilbert', 'hilbert', 
        'ihilbert', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff', 'shift', 
        'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'convolve', 
        '_fftpack')

def make_complex_data(shape, dtype):
    ar, ai = dtype(numpy.random.randn(2, *shape))
    return ar + 1j*ai

def make_r2c_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

def make_c2r_real_data(shape, dtype):
    return dtype(numpy.random.randn(*shape))

make_complex_data = test_pyfftw_numpy_interface.make_complex_data

complex_dtypes = test_pyfftw_numpy_interface.complex_dtypes
real_dtypes = test_pyfftw_numpy_interface.real_dtypes

def numpy_fft_replacement(a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous):

    return (a, s, axes, overwrite_input, planner_effort, 
        threads, auto_align_input, auto_contiguous)

io_dtypes = {
        'complex': (complex_dtypes, make_complex_data),
        'r2c': (real_dtypes, make_r2c_real_data),
        'c2r': (real_dtypes, make_c2r_real_data)}

class InterfacesScipyFFTPackTestSimple(unittest.TestCase):
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


# Construct all the test classes automatically.
built_classes = []
for each_func in funcs:

    class_name = 'InterfacesScipyFFTPackTest' + each_func.upper()

    parent_class_name = 'InterfacesNumpyFFTTest' + each_func.upper()
    parent_class = getattr(test_pyfftw_numpy_interface, parent_class_name)

    class_dict = {'validator_module': scipy.fftpack, 
                'test_interface': scipy_fftpack,
                'io_dtypes': io_dtypes,
                'overwrite_input_flag': 'overwrite_x',
                'default_s_from_shape_slicer': slice(None)}

    globals()[class_name] = type(class_name,
            (parent_class,), class_dict)

    built_classes.append(globals()[class_name])

built_classes = tuple(built_classes)

test_cases = (
        InterfacesScipyFFTPackTestSimple,) + built_classes

test_set = None
#test_set = {'InterfacesScipyFFTPackTestIFFTN': ['test_auto_align_input']}


if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
