#
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

import copy

from pyfftw import interfaces, builders
import numpy

import unittest
from .test_pyfftw_base import run_test_suites
from .test_pyfftw_numpy_interface import InterfacesNumpyFFTTestFFT

import threading
import time

import os

'''Test the caching functionality of the interfaces package.
'''

class InterfacesNumpyFFTCacheTestFFT(InterfacesNumpyFFTTestFFT):
    test_shapes = (
            ((100,), {}),
            ((128, 64), {'axis': 0}),
            ((128, 32), {'axis': -1}),
            ((32, 64), {}),
            )

    def validate(self, array_type, test_shape, dtype,
                 s, kwargs, copy_func=copy.copy):

        # Do it with the cache
        interfaces.cache.enable()
        output = self._validate(array_type, test_shape, dtype, s, kwargs,
                                copy_func=copy_func)
        output2 = self._validate(array_type, test_shape, dtype, s, kwargs,
                                 copy_func=copy_func)

        self.assertIsNot(output, output2)

        # Turn it off to finish
        interfaces.cache.disable()

class CacheSpecificInterfacesUtils(unittest.TestCase):

    def test_slow_lookup_no_race_condition(self):
        '''Checks that lookups in _utils longer than the keepalive time are ok.
        '''
        # Any old size, it doesn't matter
        data_shape = (128,)

        # Monkey patch the module with a custom _Cache object
        _Cache_class = interfaces.cache._Cache
        class _SlowLookupCache(_Cache_class):

            def _lookup(self, key):
                return _Cache_class.lookup(self, key)

            def lookup(self, key):
                time.sleep(0.1)
                return self._lookup(key)

        try:
            interfaces.cache._Cache = _SlowLookupCache

            interfaces.cache.enable()

            # something shortish
            interfaces.cache.set_keepalive_time(0.001)

            ar, ai = numpy.random.randn(*(2,) + data_shape)
            a = ar + 1j*ai

            # Both the following should work without exception
            # (even if it fails to get from the cache)
            interfaces.numpy_fft.fft(a)
            interfaces.numpy_fft.fft(a)

            interfaces.cache.disable()

        finally:
            # Revert the monkey patching
            interfaces.cache._Cache = _Cache_class


class InterfacesCacheTest(unittest.TestCase):

    def test_missing_threading(self):
        self.assertIs(interfaces.cache._fftw_cache, None)

        mod_threading = interfaces.cache._threading
        interfaces.cache._threading = None

        with self.assertRaises(ImportError):
            interfaces.cache.enable()

        interfaces.cache._threading = mod_threading

    def test_is_enabled(self):
        self.assertIs(interfaces.cache._fftw_cache, None)

        interfaces.cache.enable()
        self.assertTrue(interfaces.cache.is_enabled())

        interfaces.cache.disable()
        self.assertFalse(interfaces.cache.is_enabled())

    def test_cache_enable_disable(self):

        self.assertIs(interfaces.cache._fftw_cache, None)

        interfaces.cache.enable()
        self.assertIsInstance(
                interfaces.cache._fftw_cache, interfaces.cache._Cache)

        interfaces.cache.disable()
        self.assertIs(interfaces.cache._fftw_cache, None)

    def test_set_keepalive_time(self):
        with self.assertRaises(interfaces.cache.CacheError):
            interfaces.cache.set_keepalive_time(10)

        interfaces.cache.enable()
        interfaces.cache.set_keepalive_time(10)

        self.assertTrue(
                interfaces.cache._fftw_cache.keepalive_time == 10.0)

        interfaces.cache.disable()


class CacheTest(unittest.TestCase):

    def test_cache_parent_thread_ended(self):
        '''Test ending cache parent thread ends cache thread.
        '''
        # Firstly make sure we've exited any lingering threads from other
        # tests.
        time.sleep(0.1)

        self.assertTrue(threading.active_count() == 1)

        def cache_parent_thread():
            cache = interfaces.cache._Cache()
            time.sleep(0.5)

        parent_t = threading.Thread(target=cache_parent_thread)
        parent_t.start()

        time.sleep(0.1)
        # Check it's running
        self.assertTrue(threading.active_count() == 3)

        parent_t.join()
        time.sleep(0.1)
        # Check both threads have exited properly
        self.assertTrue(threading.active_count() == 1)

    def test_delete_cache_object(self):
        '''Test deleting a cache object ends cache thread.
        '''
        # Firstly make sure we've exited any lingering threads from other
        # tests.
        time.sleep(0.2)
        self.assertTrue(threading.active_count() == 1)

        _cache = interfaces.cache._Cache()
        time.sleep(0.2)
        self.assertTrue(threading.active_count() == 2)

        del _cache
        time.sleep(0.2)
        self.assertTrue(threading.active_count() == 1)

    def test_insert_and_lookup_item(self):
        _cache = interfaces.cache._Cache()

        key = 'the key'

        test_array = numpy.random.randn(16)
        obj = builders.fft(test_array)
        _cache.insert(obj, key)

        self.assertIs(_cache.lookup(key), obj)

    def test_invalid_lookup(self):
        _cache = interfaces.cache._Cache()

        key = 'the key'

        test_array = numpy.random.randn(16)
        obj = builders.fft(test_array)
        _cache.insert(obj, key)

        self.assertRaises(KeyError, _cache.lookup, 'wrong_key')

    def test_keepalive_time_update(self):
        _cache = interfaces.cache._Cache()

        # The default
        self.assertEqual(_cache.keepalive_time, 0.1)

        _cache.set_keepalive_time(0.3)
        self.assertEqual(_cache.keepalive_time, 0.3)

        _cache.set_keepalive_time(10.0)
        self.assertEqual(_cache.keepalive_time, 10.0)

        _cache.set_keepalive_time('0.2')
        self.assertEqual(_cache.keepalive_time, 0.2)

        with self.assertRaises(ValueError):
            _cache.set_keepalive_time('foo')

        with self.assertRaises(TypeError):
            _cache.set_keepalive_time([])

    def test_contains(self):
        _cache = interfaces.cache._Cache()

        key = 'the key'

        test_array = numpy.random.randn(16)
        obj = builders.fft(test_array)
        _cache.insert(obj, key)

        self.assertTrue(key in _cache)
        self.assertFalse('Not a key' in _cache)

    def test_objects_removed_after_keepalive(self):
        _cache = interfaces.cache._Cache()

        key = 'the key'

        test_array = numpy.random.randn(16)
        obj = builders.fft(test_array)
        _cache.insert(obj, key)

        self.assertIs(_cache.lookup(key), obj)

        keepalive_time = _cache.keepalive_time

        if os.name == 'nt':
            # A hack to keep appveyor from falling over here. I suspect the
            # contention is too much to work properly. Either way, let's
            # assume it's a windows problem for now...
            time.sleep(keepalive_time * 8)
        else:
            # Relax a bit more otherwise
            time.sleep(keepalive_time * 4)

        self.assertRaises(KeyError, _cache.lookup, key)

        _cache.insert(obj, key)
        old_keepalive_time = _cache.keepalive_time
        _cache.set_keepalive_time(old_keepalive_time * 4)

        self.assertIs(_cache.lookup(key), obj)

        time.sleep(old_keepalive_time * 3.5)
        # still should be there
        self.assertIs(_cache.lookup(key), obj)

        if os.name == 'nt':
            # As above, but with a bit longer
            time.sleep(old_keepalive_time * 16)
        else:
            time.sleep(old_keepalive_time * 8)

        self.assertRaises(KeyError, _cache.lookup, key)

class InterfacesNumpyFFTCacheTestIFFT(InterfacesNumpyFFTCacheTestFFT):
    func = 'ifft'

class InterfacesNumpyFFTCacheTestRFFT(InterfacesNumpyFFTCacheTestFFT):
    func = 'rfft'

class InterfacesNumpyFFTCacheTestIRFFT(InterfacesNumpyFFTCacheTestFFT):
    func = 'irfft'
    realinv = True

class InterfacesNumpyFFTCacheTestFFT2(InterfacesNumpyFFTCacheTestFFT):
    axes_kw = 'axes'
    func = 'ifft2'
    test_shapes = (
            ((128, 64), {'axes': None}),
            ((128, 32), {'axes': None}),
            ((32, 64), {'axes': (-2, -1)}),
            ((4, 6, 8, 4), {'axes': (0, 3)}),
            )

    invalid_args = (
            ((100,), ((100, 200),), ValueError, 'Shape error'),
            ((100, 200), ((100, 200, 100),), ValueError, 'Shape error'),
            ((100,), ((100, 200), (-3, -2, -1)), ValueError, 'Shape error'),
            ((100, 200), (100, -1), TypeError, ''),
            ((100, 200), ((100, 200), (-3, -2)), IndexError, 'Invalid axes'),
            ((100, 200), ((100,), (-3,)), IndexError, 'Invalid axes'))


class InterfacesNumpyFFTCacheTestIFFT2(InterfacesNumpyFFTCacheTestFFT2):
    func = 'ifft2'

class InterfacesNumpyFFTCacheTestRFFT2(InterfacesNumpyFFTCacheTestFFT2):
    func = 'rfft2'

class InterfacesNumpyFFTCacheTestIRFFT2(InterfacesNumpyFFTCacheTestFFT2):
    func = 'irfft2'
    realinv = True

class InterfacesNumpyFFTCacheTestFFTN(InterfacesNumpyFFTCacheTestFFT2):
    func = 'ifftn'
    test_shapes = (
            ((128, 32, 4), {'axes': None}),
            ((64, 128, 16), {'axes': (0, 1, 2)}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1)}),
            ((4, 6, 8, 4), {'axes': (0, 3, 1, 2)}),
            )

class InterfacesNumpyFFTCacheTestIFFTN(InterfacesNumpyFFTCacheTestFFTN):
    func = 'ifftn'

class InterfacesNumpyFFTCacheTestRFFTN(InterfacesNumpyFFTCacheTestFFTN):
    func = 'rfftn'

class InterfacesNumpyFFTCacheTestIRFFTN(InterfacesNumpyFFTCacheTestFFTN):
    func = 'irfftn'
    realinv = True

test_cases = (
        CacheTest,
        InterfacesCacheTest,
        CacheSpecificInterfacesUtils,
        InterfacesNumpyFFTCacheTestFFT,
        InterfacesNumpyFFTCacheTestIFFT,
        InterfacesNumpyFFTCacheTestRFFT,
        InterfacesNumpyFFTCacheTestIRFFT,
        InterfacesNumpyFFTCacheTestFFT2,
        InterfacesNumpyFFTCacheTestIFFT2,
        InterfacesNumpyFFTCacheTestRFFT2,
        InterfacesNumpyFFTCacheTestIRFFT2,
        InterfacesNumpyFFTCacheTestFFTN,
        InterfacesNumpyFFTCacheTestIFFTN,
        InterfacesNumpyFFTCacheTestRFFTN,
        InterfacesNumpyFFTCacheTestIRFFTN,)

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
