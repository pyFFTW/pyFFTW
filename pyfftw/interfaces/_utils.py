#!/usr/bin/env python
#
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

'''
Utility functions for the interfaces routines
'''

import pyfftw.builders as builders
import pyfftw
from . import cache

def _Xfftn(a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous, 
        calling_func):

    reload_after_transform = False

    try:
        s = tuple(s)
    except TypeError:
        pass

    try:
        axes = tuple(axes)
    except TypeError:
        pass

    if calling_func in ('irfft2', 'irfftn'):
        # overwrite_input is not an argument to irfft2 or irfftn
        args = (a, s, axes, planner_effort, threads, 
                auto_align_input, auto_contiguous)

        if not overwrite_input:
            # Only irfft2 and irfftn have overwriting the input
            # as the default (and so require the input array to 
            # be reloaded).
            reload_after_transform = True
    else:
        args = (a, s, axes, overwrite_input, planner_effort, threads, 
                auto_align_input, auto_contiguous)
    
    if cache.is_enabled():
        key = (calling_func, a.shape, a.strides, a.dtype, s.__hash__(), 
                axes.__hash__(), args[3:])

        try:
            if key in cache._fftw_cache:
                FFTW_object = cache._fftw_cache.lookup(key)
            else:
                FFTW_object = None

        except KeyError:
            # This occurs if the object has fallen out of the cache between
            # the check and the lookup
            FFTW_object = None

    if not cache.is_enabled() or FFTW_object is None:

        # If we're going to create a new FFTW object, we need to copy
        # the input array to preserve it, otherwise we can't actually
        # take the transform of the input array! (in general, we have
        # to assume that the input array will be destroyed during 
        # planning).
        a_copy = a.copy()

        FFTW_object = getattr(builders, calling_func)(*args)
    
        # Only copy if the input array is what was actually used
        # (otherwise it shouldn't be overwritten)
        if FFTW_object.get_input_array() is a:
            a[:] = a_copy

        if cache.is_enabled():
            cache._fftw_cache.insert(FFTW_object, key)
        
        output_array = FFTW_object()

    else:
        if reload_after_transform:
            a_copy = a.copy()

        orig_output_array = FFTW_object.get_output_array()
        output_shape = orig_output_array.shape
        output_dtype = orig_output_array.dtype
        output_alignment = FFTW_object.output_alignment

        output_array = pyfftw.n_byte_align_empty(output_shape, 
                output_alignment, output_dtype)

        FFTW_object(input_array=a, output_array=output_array)
    
    if reload_after_transform:
        a[:] = a_copy

    return output_array
