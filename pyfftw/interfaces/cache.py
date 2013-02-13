#!/usr/bin/env python
#
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

'''
During calls to functions implemented in :mod:`pyfftw.interfaces`, a
:class:`pyfftw.FFTW` object is necessarily created. Although the time to
create a new :class:`pyfftw.FFTW` is short (assuming that the planner
possesses the necessary wisdom to create the plan immediately), it may
still take longer than a short transform.

This module implements a method by which objects that are created through
:mod:`pyfftw.interfaces` are temporarily cached. If an equivalent
transform is then performed within a short period, the object is acquired
from the cache rather than a new one created.

Enable the cache by calling :func:`pyfftw.interfaces.cache.enable`. 
Disable it by calling :func:`pyfftw.interfaces.cache.disable`. By default,
the cache is disabled.

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
    _threading == None

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
    practice, it may be up to twice as long before the object is
    deleted from the cache (due to implmentational details).
    '''
    global _fftw_cache
    
    if _fftw_cache is None:
        raise CacheError('Cache is not currently enabled')
    else:
        _fftw_cache.set_keepalive_time(keepalive_time)

class _Cache(object):

    @property
    def keepalive_time(self):
        return self.__keepalive_time

    def __init__(self, keepalive_time=0.1):

        self.__cache_dict = {}
        self.set_keepalive_time(keepalive_time)

        # A set of objects to be kept alive during the next cull
        self.__keepalive_set = set()

        self.__cull_lock = _threading.Lock()
        self.__keepalive_set_lock = _threading.Lock()

        self.initialised = _threading.Event()

        self.__parent_thread = _threading.current_thread()

        self.__initialised = _threading.Event()
        self.__initialised.clear() # Explicitly clear it for clarity
        
        self.__thread_object = _threading.Thread(target=_Cache._run,
                args=(weakref.proxy(self), ))

        self.__thread_object.daemon = True
        self.__thread_object.start()

        while not self.__initialised.is_set():
            # This loop is necessary to stop the main thread doing 
            # anything until the exception handler in _run can deal with 
            # the object being deleted.
            pass

    def __del__(self):
        # Wait until the thread object has quit before
        # exiting (which it will because a reference error will
        # be raised).
        self.__thread_object.join()

    def __contains__(self, key):
        return key in self.__cache_dict

    def _run(self):

        self.__initialised.set()

        last_cull_time = time.time()

        try:
            while True:
                if not self.__parent_thread.is_alive():
                    break

                if time.time() - last_cull_time > self.__keepalive_time:
                    # Perform a cull
                    last_cull_time = time.time()

                    with self.__cull_lock:
                        # Operate on a copy of the cache dict
                        # so lookups continue.
                        new_cache_dict = self.__cache_dict.copy()

                        with self.__keepalive_set_lock:
                            # Work out which should be culled
                            cull_set = set(new_cache_dict).difference(
                                    self.__keepalive_set)

                            self.__keepalive_set = set()

                        for each_key in cull_set:
                            del new_cache_dict[each_key]

                        # Necessarily atomic, so no problem with
                        # the lookups continuing
                        self.__cache_dict = new_cache_dict

                time.sleep(self.__wakeup_time)

        except ReferenceError:
            pass

    def set_keepalive_time(self, keepalive_time=0.1):
        '''Set the minimum time in seconds for which any object in the cache
        is kept alive.

        The time is not precise, and sets a minimum time to be alive. In 
        practice, it may be up to twice as long before the object is
        deleted from the cache (due to implementational details).
        '''
        self.__keepalive_time = float(keepalive_time)

        if self.__keepalive_time/2 > 0.1:
            self.__wakeup_time = 0.1
        else:
            self.__wakeup_time = self.__keepalive_time/2

    def __refresh(self, key):
        '''Refresh the object referenced by key to stop it being culled
        on the next round.
        '''
        with self.__keepalive_set_lock:
            self.__keepalive_set.add(key)

    def insert(self, obj, key):
        '''Insert the passed object into the cache, referenced by key, 
        a hashable.
        '''
        with self.__cull_lock:
            self.__cache_dict[key] = obj
            self.__refresh(key)

    def lookup(self, key):
        '''Lookup the object referenced by key and return it, refreshing
        the cache at the same time.
        '''
        self.__refresh(key)        
        return self.__cache_dict[key]

