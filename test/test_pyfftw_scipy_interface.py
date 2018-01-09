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

from pyfftw.interfaces import scipy_fftpack
import pyfftw
import numpy

try:
    import scipy
    import scipy.fftpack
    import scipy.signal

except ImportError:
    scipy_missing = True

else:
    scipy_missing = False

import unittest
from .test_pyfftw_base import run_test_suites, miss
from . import test_pyfftw_numpy_interface

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

if scipy.__version__ < '0.19':
    # Older scipy will raise an error for inputs of type float16, so we
    # cannot validate transforms with float16 input vs. scipy.fftpack
    complex_dtypes = pyfftw._supported_nptypes_complex
    real_dtypes = pyfftw._supported_nptypes_real
else:
    # reuse all dtypes from numpy tests (including float16)
    complex_dtypes = test_pyfftw_numpy_interface.complex_dtypes
    real_dtypes = test_pyfftw_numpy_interface.real_dtypes

# Remove long double because scipy explicitly doesn't support it
complex_dtypes = [x for x in complex_dtypes if x != numpy.clongdouble]
real_dtypes = [x for x in real_dtypes if x != numpy.longdouble]

def numpy_fft_replacement(a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous):

    return (a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous)

io_dtypes = {
        'complex': (complex_dtypes, make_complex_data),
        'r2c': (real_dtypes, make_r2c_real_data),
        'c2r': (real_dtypes, make_c2r_real_data)}

@unittest.skipIf(scipy_missing, 'scipy is not installed, so this feature is'
                 'unavailable')
class InterfacesScipyFFTPackTestSimple(unittest.TestCase):
    ''' A really simple test suite to check simple implementation.
    '''

    @unittest.skipIf(*miss('64'))
    def test_scipy_overwrite(self):

        new_style_scipy_fftn = False
        try:
            scipy_fftn = scipy.signal.signaltools.fftn
            scipy_ifftn = scipy.signal.signaltools.ifftn
        except AttributeError:
            scipy_fftn = scipy.fftpack.fftn
            scipy_ifftn = scipy.fftpack.ifftn
            new_style_scipy_fftn = True

        a = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)
        b = pyfftw.empty_aligned((128, 64), dtype='complex128', n=16)

        a[:] = (numpy.random.randn(*a.shape) +
                1j*numpy.random.randn(*a.shape))
        b[:] = (numpy.random.randn(*b.shape) +
                1j*numpy.random.randn(*b.shape))


        scipy_c = scipy.signal.fftconvolve(a, b)

        if new_style_scipy_fftn:
            scipy.fftpack.fftn = scipy_fftpack.fftn
            scipy.fftpack.ifftn = scipy_fftpack.ifftn

        else:
            scipy.signal.signaltools.fftn = scipy_fftpack.fftn
            scipy.signal.signaltools.ifftn = scipy_fftpack.ifftn

        scipy_replaced_c = scipy.signal.fftconvolve(a, b)

        self.assertTrue(numpy.allclose(scipy_c, scipy_replaced_c))

        if new_style_scipy_fftn:
            scipy.fftpack.fftn = scipy_fftn
            scipy.fftpack.ifftn = scipy_ifftn

        else:
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

    # unlike numpy, none of the scipy functions support the norm kwarg
    globals()[class_name].has_norm_kwarg = False

    built_classes.append(globals()[class_name])

built_classes = tuple(built_classes)

test_cases = (
        InterfacesScipyFFTPackTestSimple,) + built_classes

test_set = None
#test_set = {'InterfacesScipyFFTPackTestIFFTN': ['test_auto_align_input']}


if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
