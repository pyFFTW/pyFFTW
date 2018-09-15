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

from pyfftw import FFTW, _supported_types, _all_types_human_readable
import numpy
import struct
from timeit import Timer

import unittest

try:
    import mkl_fft
    # mkl_fft monkeypatches numpy.fft so explicitly import from fftpack instead
    from numpy.fft import fftpack as np_fft
except ImportError:
    from numpy import fft as np_fft

def miss(*xs):
    '''Skip test if the precisions in the iterable `xs` are not available.'''
    msg = 'Requires %s' % _all_types_human_readable[xs[0]]
    for x in xs[1:]:
        msg += ' and %s' % _all_types_human_readable[x]
    msg += ' precision.'
    skip = not all(x in _supported_types for x in xs)
    return (skip, msg)

def require(self, *xs):
    skip, msg = miss(*xs)
    if skip:
        self.skipTest(msg)

class FFTWBaseTest(unittest.TestCase):

    def reference_fftn(self, a, axes):
        return np_fft.fftn(a, axes=axes)

    def __init__(self, *args, **kwargs):

        super(FFTWBaseTest, self).__init__(*args, **kwargs)
        self.make_shapes()

        if not hasattr(self, 'assertRaisesRegex'):
            self.assertRaisesRegex = self.assertRaisesRegexp

    def setUp(self):

        require(self, '32')

        self.input_dtype = numpy.complex64
        self.output_dtype = numpy.complex64
        self.np_fft_comparison = np_fft.fft

        self.direction = 'FFTW_FORWARD'
        return

    def tearDown(self):

        return

    def get_input_dtype_alignment(self):
        return self.input_dtype([]).real.dtype.alignment

    def get_output_dtype_alignment(self):
        return self.input_dtype([]).real.dtype.alignment

    def make_shapes(self):
        self.input_shapes = {
                'small_1d': (16,),
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (5, 256, 2048)}

        self.output_shapes = {
                'small_1d': (16,),
                '1d': (2048,),
                '2d': (256, 2048),
                '3d': (5, 256, 2048)}

    def create_test_arrays(self, input_shape, output_shape, axes=None):
        a = self.input_dtype(numpy.random.randn(*input_shape)
                +1j*numpy.random.randn(*input_shape))

        b = self.output_dtype(numpy.random.randn(*output_shape)
                +1j*numpy.random.randn(*output_shape))

        return a, b

    def timer_routine(self, pyfftw_callable, numpy_fft_callable,
            comparison_string='numpy.fft'):

        N = 100

        t = Timer(stmt=pyfftw_callable)
        t_numpy_fft = Timer(stmt=numpy_fft_callable)

        t_str = ("%.2f" % (1000.0/N*t.timeit(N)))+' ms'
        t_numpy_str = ("%.2f" % (1000.0/N*t_numpy_fft.timeit(N)))+' ms'

        print('One run: '+ t_str + \
                ' (versus ' + t_numpy_str + ' for ' + comparison_string + \
                ')')


    def run_validate_fft(self, a, b, axes, fft=None, ifft=None,
            force_unaligned_data=False, create_array_copies=True,
            threads=1, flags=('FFTW_ESTIMATE',)):
        ''' Run a validation of the FFTW routines for the passed pair
        of arrays, a and b, and the axes argument.

        a and b are assumed to be the same shape (but not necessarily
        the same layout in memory).

        fft and ifft, if passed, should be instantiated FFTW objects.

        If force_unaligned_data is True, the flag FFTW_UNALIGNED
        will be passed to the fftw routines.

        The threads argument runs the validation with multiple threads.

        flags is passed to the creation of the FFTW object.
        '''

        if create_array_copies:
            # Don't corrupt the original mutable arrays
            a = a.copy()
            b = b.copy()

        a_orig = a.copy()

        flags = list(flags)

        if force_unaligned_data:
            flags.append('FFTW_UNALIGNED')

        if fft == None:
            fft = FFTW(a,b,axes=axes, direction='FFTW_FORWARD',
                    flags=flags, threads=threads)
        else:
            fft.update_arrays(a,b)

        if ifft == None:
            ifft = FFTW(b, a, axes=axes, direction='FFTW_BACKWARD',
                    flags=flags, threads=threads)
        else:
            ifft.update_arrays(b,a)


        a[:] = a_orig

        # Test the forward FFT by comparing it to the result from numpy.fft
        fft.execute()
        ref_b = self.reference_fftn(a, axes=axes)

        # This is actually quite a poor relative error, but it still
        # sometimes fails. I assume that numpy.fft has different internals
        # to fftw.
        self.assertTrue(numpy.allclose(b, ref_b, rtol=1e-2, atol=1e-3))

        # Test the inverse FFT by comparing the result to the starting
        # value (which is scaled as per FFTW being unnormalised).
        ifft.execute()
        # The scaling is the product of the lengths of the fft along
        # the axes along which the fft is taken.
        scaling = numpy.prod(numpy.array(a.shape)[list(axes)])

        self.assertEqual(ifft.N, scaling)
        self.assertEqual(fft.N, scaling)

        self.assertTrue(numpy.allclose(a/scaling, a_orig, rtol=1e-2, atol=1e-3))
        return fft, ifft


def run_test_suites(test_suites, run_tests=None):
    '''From each test case (derived from TestCase) in test_suites,
    load and run all the test cases within.

    If run_tests is not None, then it should be a dictionary with
    keys being the test suite class name, and the values being
    a list of test methods to run. Alternatively, the key can
    be 'all' in which case all the test suites will be run with
    the provided list of test suites.
    '''
    suite = unittest.TestSuite()

    for test_class in test_suites:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)

        if run_tests is not None:
            if test_class.__name__ in run_tests:
                this_suite_run = set(run_tests[test_class.__name__])
            else:
                this_suite_run = set()

            if 'all' in run_tests:
                this_suite_run = this_suite_run.union(run_tests['all'])

            _tests = []
            for each_test in tests:
                if (each_test.id().split('.')[-1] in this_suite_run):
                    _tests.append(each_test)

            tests = _tests

        suite.addTests(tests)


    unittest.TextTestRunner(verbosity=2).run(suite)
