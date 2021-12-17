#!/usr/bin/env python
#
# Copyright 2015 Knowledge Economy Developments Ltd
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

'''
During calls to functions implemented in :mod:`pyfftw.interfaces`, a
:class:`pyfftw.FFTW` object is necessarily created. Although the time to
create a new :class:`pyfftw.FFTW` is short (assuming that the planner
possesses the necessary wisdom to create the plan immediately), it may
still take longer than a short transform.

This module implements a method by which objects that are created through
:mod:`pyfftw.interfaces` are temporarily cached. If an equivalent
transform is then performed within a short period, the object is acquired
from the cache rather than a new one created. The equivalency is quite
conservative and in practice means that if any of the arguments change, or
if the properties of the array (shape, strides, dtype) change in any way, then
the cache lookup will fail.

The cache temporarily stores a copy of any interim :class:`pyfftw.FFTW`
objects that are created. If they are not used for some period of time,
which can be set with :func:`pyfftw.interfaces.cache.set_keepalive_time`,
then they are removed from the cache (liberating any associated memory).
The default keepalive time is 0.1 seconds.

Enable the cache by calling :func:`pyfftw.interfaces.cache.enable`.
Disable it by calling :func:`pyfftw.interfaces.cache.disable`. By default,
the cache is disabled.

Note that even with the cache enabled, there is a fixed overhead associated
with lookups. This means that for small transforms, the overhead may exceed
the transform. At this point, it's worth looking at using :class:`pyfftw.FFTW`
directly.

When the cache is enabled, the module spawns a new thread to keep track
of the objects. If :mod:`threading` is not available, then the cache
is not available and trying to use it will raise an ImportError exception.

The actual implementation of the cache is liable to change, but the
documented API is stable.
'''

try:
    import threading as _threading
    _threading_import_error = None
except ImportError as e:
    _threading_import_error = e
    _threading = None

import time
import weakref

__all__ = ['enable', 'disable', 'set_keepalive_time']

_fftw_cache = None

class CacheError(Exception):
    pass

def enable():
    '''Enable the cache.
    '''
    global _fftw_cache

    if _threading is not None:
        if _fftw_cache is None:
            _fftw_cache = _Cache()
    else:
        raise ImportError(_threading_import_error)

def disable():
    '''Disable the cache.
    '''
    global _fftw_cache
    _fftw_cache = None

def is_enabled():
    '''Return whether the cache is currently enabled.
    '''
    if _fftw_cache is None:
        return False
    else:
        return True

def set_keepalive_time(keepalive_time):
    '''Set the minimum time in seconds for which any :mod:`pyfftw.FFTW` object
    in the cache is kept alive.

    When the cache is enabled, the interim objects that are used through
    a :mod:`pyfftw.interfaces` function are cached for the time set through
    this function. If the object is not used for the that time, it is
    removed from the cache. Using the object zeros the timer.

    The time is not precise, and sets a minimum time to be alive. In
    practice, it may be quite a bit longer before the object is
    deleted from the cache (due to implementational details - e.g. contention
    from other threads).
    '''
    global _fftw_cache

    if _fftw_cache is None:
        raise CacheError('Cache is not currently enabled')
    else:
        _fftw_cache.set_keepalive_time(keepalive_time)

class _Cache(object):

    @property
    def keepalive_time(self):
        return self._keepalive_time

    def __init__(self, keepalive_time=0.1):

        self._cache_dict = {}
        self.set_keepalive_time(keepalive_time)

        # A set of objects to be kept alive during the next cull
        self._keepalive_set = set()

        self._cull_lock = _threading.Lock()
        self._keepalive_set_lock = _threading.Lock()

        self.initialised = _threading.Event()

        self._parent_thread = _threading.current_thread()
        self._close_thread_now = _threading.Event()

        self._initialised = _threading.Event()
        self._initialised.clear() # Explicitly clear it for clarity

        self._thread_object = _threading.Thread(target=_Cache._run,
                args=(weakref.proxy(self), ), name='PyFFTWCacheThread')

        self._thread_object.daemon = True
        self._thread_object.start()

        while not self._initialised.is_set():
            # This loop is necessary to stop the main thread doing
            # anything until the exception handler in _run can deal with
            # the object being deleted.
            pass

    def __del__(self):
        # Wait until the thread object has quit before
        # exiting (which it will because a reference error will
        # be raised).
        try:
            self._close_thread_now.set()

        except TypeError:
            # Not sure what's going on here, but IPython baulks on exit
            pass

    def __contains__(self, key):
        return key in self._cache_dict

    def _run(self):

        last_cull_time = time.time()

        try:
            self._initialised.set()

            while True:
                if (not self._parent_thread.is_alive() or
                    self._close_thread_now.is_set()):
                    break

                if time.time() - last_cull_time > self._keepalive_time:
                    # Perform a cull
                    last_cull_time = time.time()

                    with self._cull_lock:
                        # Operate on a copy of the cache dict
                        # so lookups continue.
                        new_cache_dict = self._cache_dict.copy()

                        with self._keepalive_set_lock:
                            # Work out which should be culled
                            cull_set = set(new_cache_dict).difference(
                                    self._keepalive_set)

                            self._keepalive_set = set()

                        for each_key in cull_set:
                            del new_cache_dict[each_key]

                        # Necessarily atomic, so no problem with
                        # the lookups continuing
                        self._cache_dict = new_cache_dict

                time.sleep(self._wakeup_time)

        except ReferenceError:
            pass

    def set_keepalive_time(self, keepalive_time=0.1):
        '''Set the minimum time in seconds for which any object in the cache
        is kept alive.

        The time is not precise, and sets a minimum time to be alive. In
        practice, it may be up to twice as long before the object is
        deleted from the cache (due to implementational details).
        '''
        self._keepalive_time = float(keepalive_time)

        if self._keepalive_time/2 > 0.1:
            self._wakeup_time = 0.1
        else:
            self._wakeup_time = self._keepalive_time/2

    def _refresh(self, key):
        '''Refresh the object referenced by key to stop it being culled
        on the next round.
        '''
        with self._keepalive_set_lock:
            self._keepalive_set.add(key)

    def insert(self, obj, key):
        '''Insert the passed object into the cache, referenced by key,
        a hashable.
        '''
        with self._cull_lock:
            self._cache_dict[key] = obj
            self._refresh(key)

    def lookup(self, key):
        '''Lookup the object referenced by key and return it, refreshing
        the cache at the same time.
        '''
        self._refresh(key)
        return self._cache_dict[key]
