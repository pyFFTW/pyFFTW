#!/usr/bin/env python
#
# Copyright 2014 Knowledge Economy Developments Ltd
#
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# Michael McNeil Forbes
# michael.forbes+pyfftw@gmail.com
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
Utility functions for the interfaces routines
'''

import pyfftw.builders as builders
import pyfftw
import numpy
from . import cache


def _Xfftn(a, s, axes, overwrite_input, planner_effort,
        threads, auto_align_input, auto_contiguous,
        calling_func, normalise_idft=True, ortho=False):

    work_with_copy = False

    a = numpy.asanyarray(a)

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
        args = (planner_effort, threads, auto_align_input, auto_contiguous)

        if not overwrite_input:
            # Only irfft2 and irfftn have overwriting the input
            # as the default (and so require the input array to
            # be reloaded).
            work_with_copy = True
    else:
        args = (overwrite_input, planner_effort, threads,
                auto_align_input, auto_contiguous)

        if not a.flags.writeable:
            # Special case of a locked array - always work with a
            # copy.  See issue #92.
            work_with_copy = True

            if overwrite_input:
                raise ValueError('overwrite_input cannot be True when the ' +
                                 'input array flags.writeable is False')

    if work_with_copy:
        # We make the copy before registering the key so that the
        # copy's stride information will be cached since this will be
        # used for planning.  Make sure the copy is byte aligned to
        # prevent further copying
        a_original = a
        a = pyfftw.empty_aligned(shape=a.shape, dtype=a.dtype)
        a[...] = a_original

    if cache.is_enabled():
        alignment = a.ctypes.data % pyfftw.simd_alignment

        key = (calling_func, a.shape, a.strides, a.dtype, s.__hash__(),
               axes.__hash__(), alignment, args)

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

        # If we're going to create a new FFTW object and are not
        # working with a copy, then we need to copy the input array to
        # preserve it, otherwise we can't actually  take the transform
        # of the input array! (in general, we have to assume that the
        # input array will be destroyed during planning).
        if not work_with_copy:
            a_copy = a.copy()

        planner_args = (a, s, axes) + args

        FFTW_object = getattr(builders, calling_func)(*planner_args)

        # Only copy if the input array is what was actually used
        # (otherwise it shouldn't be overwritten)
        if not work_with_copy and FFTW_object.input_array is a:
            a[:] = a_copy

        if cache.is_enabled():
            cache._fftw_cache.insert(FFTW_object, key)

        output_array = FFTW_object(normalise_idft=normalise_idft, ortho=ortho)

    else:
        orig_output_array = FFTW_object.output_array
        output_shape = orig_output_array.shape
        output_dtype = orig_output_array.dtype
        output_alignment = FFTW_object.output_alignment

        output_array = pyfftw.empty_aligned(
            output_shape, output_dtype, n=output_alignment)

        FFTW_object(input_array=a, output_array=output_array,
                normalise_idft=normalise_idft, ortho=ortho)

    return output_array
