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

_default_args = {}

def _Xfftn(a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous, 
        calling_func):

    reload_after_transform = False

    if calling_func in ('irfft2', 'irfftn'):
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

    output_array = FFTW_object()

    if reload_after_transform:
        a[:] = a_copy

    return output_array
