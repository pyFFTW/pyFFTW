#!/usr/bin/env python
#
# Copyright 2014 Knowledge Economy Developments Ltd
# Copyright 2014 David Wells
#
# Henry Gomersall
# heng@kedevelopments.co.uk
# David Wells
# drwells <at> vt.edu
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
A set of utility functions for use with the builders. Users should
not need to use the functions directly, but they are included here for
completeness and to aid with understanding of what is happening behind
the scenes.

Certainly, users may encounter instances of
:class:`~pyfftw.builders._utils._FFTWWrapper`.

Everything documented in this module is *not* part of the public API
and may change in future versions.
'''

import multiprocessing
import pyfftw
import numpy
import warnings
from .. import _threading_type
from .. import config

__all__ = ['_FFTWWrapper', '_rc_dtype_pairs', '_default_dtype', '_Xfftn',
        '_setup_input_slicers', '_compute_array_shapes', '_precook_1d_args',
        '_cook_nd_args']

_valid_efforts = ('FFTW_ESTIMATE', 'FFTW_MEASURE',
        'FFTW_PATIENT', 'FFTW_EXHAUSTIVE')

# Looking up a real dtype in here returns the complex complement of the same
# precision, and vice versa.
# It is necessary to use .char as the keys due to MSVC mapping long
# double to double and the way that numpy handles this.
_rc_dtype_pairs = {}
_default_dtype = None

# Double precision is the default default precision. Prefer casting to higher precision if
# possible. If missing, long double is mapped to double and so we lose less
# precision than by converting to single.
if '64' in pyfftw._supported_types:
    _default_dtype = numpy.dtype('float64')
    _rc_dtype_pairs.update({
        numpy.dtype('float64').char: numpy.dtype('complex128'),
        numpy.dtype('complex128').char: numpy.dtype('float64')})
if 'ld' in pyfftw._supported_types:
    if _default_dtype is None:
        _default_dtype = numpy.dtype('longdouble')
    _rc_dtype_pairs.update({
         numpy.dtype('longdouble').char: numpy.dtype('clongdouble'),
         numpy.dtype('clongdouble').char: numpy.dtype('longdouble')})
if '32' in pyfftw._supported_types:
    if _default_dtype is None:
        _default_dtype = numpy.dtype('float32')
    _rc_dtype_pairs.update({
        numpy.dtype('float32').char: numpy.dtype('complex64'),
        numpy.dtype('complex64').char: numpy.dtype('float32')})
if _default_dtype is None:
    raise NotImplementedError("No default precision available")


def _default_effort(effort):
    if effort is None:
        return config.PLANNER_EFFORT
    else:
        return effort


def _default_threads(threads):
    if threads is None:
        if config.NUM_THREADS <= 0:
            return multiprocessing.cpu_count()
        return config.NUM_THREADS
    else:
        if threads > 1 and _threading_type is None:
            raise ValueError("threads > 1 requested, but pyFFTW was not built "
                             "with multithreaded FFTW.")
        elif threads <= 0:
            return multiprocessing.cpu_count()
        return threads


def _unitary(norm):
    """_unitary() utility copied from numpy"""
    if norm not in (None, "ortho"):
        raise ValueError("Invalid norm value %s, should be None or \"ortho\"."
                         % norm)
    return norm is not None


def _norm_args(norm):
    """ pass the proper normalization-related keyword arguments. """
    if _unitary(norm):
        ortho = True
        normalise_idft = False
    else:
        ortho = False
        normalise_idft = True
    return dict(normalise_idft=normalise_idft, ortho=ortho)


def _Xfftn(a, s, axes, overwrite_input,
        planner_effort, threads, auto_align_input, auto_contiguous,
        avoid_copy, inverse, real, normalise_idft=True, ortho=False):
    '''Generic transform interface for all the transforms. No
    defaults exist. The transform must be specified exactly.
    '''
    a_orig = a
    invreal = inverse and real

    if inverse:
        direction = 'FFTW_BACKWARD'
    else:
        direction = 'FFTW_FORWARD'

    if planner_effort not in _valid_efforts:
        raise ValueError('Invalid planner effort: ', planner_effort)

    s, axes = _cook_nd_args(a, s, axes, invreal)

    input_shape, output_shape = _compute_array_shapes(
            a, s, axes, inverse, real)

    a_is_complex = numpy.iscomplexobj(a)

    # Make the input dtype correct by transforming to an available type
    if a.dtype.char not in _rc_dtype_pairs:
        dtype = _default_dtype
        if a.dtype == numpy.dtype('float16') and '32' in pyfftw._supported_types:
            # convert half-precision to single precision, if available
            dtype = numpy.dtype('float32')

        # warn when losing precision but not when using a higher precision
        if dtype.itemsize < a.dtype.itemsize:
            warnings.warn("Narrowing conversion from %s to %s precision" % (a.dtype, dtype))

        if not real or inverse:
            # It's going to be complex
            dtype = numpy.dtype(_rc_dtype_pairs[dtype.char])

        # finally convert the input array
        a = numpy.asarray(a, dtype=dtype)
    elif not (real and not inverse) and not a_is_complex:
        # We need to make it a complex dtype
        a = numpy.asarray(a, dtype=_rc_dtype_pairs[a.dtype.char])

    elif (real and not inverse) and a_is_complex:
        # It should be real
        a = numpy.asarray(a, dtype=_rc_dtype_pairs[a.dtype.char])

    # Make the output dtype correct
    if not real:
        output_dtype = a.dtype

    else:
        output_dtype = _rc_dtype_pairs[a.dtype.char]

    if not avoid_copy:
        a_copy = a.copy()

    output_array = pyfftw.empty_aligned(output_shape, output_dtype)

    flags = [planner_effort]

    if not auto_align_input:
        flags.append('FFTW_UNALIGNED')

    if overwrite_input:
        flags.append('FFTW_DESTROY_INPUT')

    if not a.shape == input_shape:

        if avoid_copy:
            raise ValueError('Cannot avoid copy: '
                    'The transform shape is not the same as the array size. '
                    '(from avoid_copy flag)')

        # This means we need to use an _FFTWWrapper object
        # and so need to create slicers.
        update_input_array_slicer, FFTW_array_slicer = (
                _setup_input_slicers(a.shape, input_shape))

        # Also, the input array will be a different shape to the shape of
        # `a`, so we need to create a new array.
        input_array = pyfftw.empty_aligned(input_shape, a.dtype)

        FFTW_object = _FFTWWrapper(input_array, output_array, axes, direction,
                flags, threads, input_array_slicer=update_input_array_slicer,
                FFTW_array_slicer=FFTW_array_slicer,
                normalise_idft=normalise_idft, ortho=ortho)

        # We copy the data back into the internal FFTW object array
        internal_array = FFTW_object.input_array
        internal_array[:] = 0
        internal_array[FFTW_array_slicer] = (
                a_copy[update_input_array_slicer])

    else:
        # Otherwise we can use `a` as-is

        input_array = a

        if auto_contiguous:
            # We only need to create a new array if it's not already
            # contiguous
            if not (a.flags['C_CONTIGUOUS'] or a.flags['F_CONTIGUOUS']):
                if avoid_copy:
                    raise ValueError('Cannot avoid copy: '
                            'The input array is not contiguous and '
                            'auto_contiguous is set. (from avoid_copy flag)')

                input_array = pyfftw.empty_aligned(a.shape, a.dtype)

        if (auto_align_input and not pyfftw.is_byte_aligned(input_array)):

            if avoid_copy:
                raise ValueError('Cannot avoid copy: '
                        'The input array is not aligned and '
                        'auto_align is set. (from avoid_copy flag)')

            input_array = pyfftw.byte_align(input_array)


        FFTW_object = pyfftw.FFTW(input_array, output_array, axes, direction,
                flags, threads, normalise_idft=normalise_idft, ortho=ortho)

        if not avoid_copy:
            # Copy the data back into the (likely) destroyed array
            FFTW_object.input_array[:] = a_copy

    return FFTW_object


class _FFTWWrapper(pyfftw.FFTW):
    ''' A class that wraps :class:`pyfftw.FFTW`, providing a slicer on the input
    stage during calls to :meth:`~pyfftw.builders._utils._FFTWWrapper.__call__`.
    '''

    def __init__(self, input_array, output_array, axes=[-1],
            direction='FFTW_FORWARD', flags=['FFTW_MEASURE'],
            threads=1, input_array_slicer=None, FFTW_array_slicer=None,
            normalise_idft=True, ortho=False):
        '''The arguments are as per :class:`pyfftw.FFTW`, but with the addition
        of 2 keyword arguments: ``input_array_slicer`` and
        ``FFTW_array_slicer``.

        These arguments represent 2 slicers: ``input_array_slicer`` slices
        the input array that is passed in during a call to instances of this
        class, and ``FFTW_array_slicer`` slices the internal array.

        The arrays that are returned from both of these slicing operations
        should be the same size. The data is then copied from the sliced
        input array into the sliced internal array.
        '''

        self._input_array_slicer = input_array_slicer
        self._FFTW_array_slicer = FFTW_array_slicer
        self._normalise_idft = normalise_idft
        self._ortho = ortho

        if 'FFTW_DESTROY_INPUT' in flags:
            self._input_destroyed = True
        else:
            self._input_destroyed = False

        pyfftw.FFTW.__init__(self, input_array, output_array,
                             axes, direction, flags, threads)

    def __call__(self, input_array=None, output_array=None,
            normalise_idft=None, ortho=None):
        '''Wrap :meth:`pyfftw.FFTW.__call__` by firstly slicing the
        passed-in input array and then copying it into a sliced version
        of the internal array. These slicers are set at instantiation.

        When input array is not ``None``, this method always results in
        a copy. Consequently, the alignment and dtype are maintained in
        the internal array.

        ``output_array`` and ``normalise_idft`` are passed through to
        :meth:`pyfftw.FFTW.__call__` untouched.
        '''

        if input_array is not None:
            # Do the update here (which is a copy, so it's alignment
            # safe etc).

            internal_input_array = self.input_array
            input_array = numpy.asanyarray(input_array)

            if self._input_destroyed:
                internal_input_array[:] = 0

            sliced_internal = internal_input_array[self._FFTW_array_slicer]
            sliced_input = input_array[self._input_array_slicer]

            if sliced_internal.shape != sliced_input.shape:
                raise ValueError('Invalid input shape: '
                        'The new input array should be the same shape '
                        'as the input array used to instantiate the '
                        'object.')

            sliced_internal[:] = sliced_input

        if normalise_idft is None:
            normalise_idft = self._normalise_idft

        if ortho is None:
            ortho = self._ortho

        output = super(_FFTWWrapper, self).__call__(input_array=None,
                output_array=output_array, normalise_idft=normalise_idft,
                ortho=ortho)

        return output


def _setup_input_slicers(a_shape, input_shape):
    ''' This function returns two slicers that are to be used to
    copy the data from the input array to the FFTW object internal
    array, which can then be passed to _FFTWWrapper:

    ``(update_input_array_slicer, FFTW_array_slicer)``

    On calls to :class:`~pyfftw.builders._utils._FFTWWrapper` objects,
    the input array is copied in as:

    ``FFTW_array[FFTW_array_slicer] = input_array[update_input_array_slicer]``
    '''

    # default the slicers to include everything
    update_input_array_slicer = (
            [slice(None)]*len(a_shape))
    FFTW_array_slicer = [slice(None)]*len(a_shape)

    # iterate over each dimension and modify the slicer and FFTW dimension
    for axis in range(len(a_shape)):

        if a_shape[axis] > input_shape[axis]:
            update_input_array_slicer[axis] = (
                    slice(0, input_shape[axis]))

        elif a_shape[axis] < input_shape[axis]:
            FFTW_array_slicer[axis] = (
                    slice(0, a_shape[axis]))
            update_input_array_slicer[axis] = (
                    slice(0, a_shape[axis]))

        else:
            # If neither of these, we use the whole dimension.
            update_input_array_slicer[axis] = (
                    slice(0, a_shape[axis]))

    return tuple(update_input_array_slicer), tuple(FFTW_array_slicer)

def _compute_array_shapes(a, s, axes, inverse, real):
    '''Given a passed in array ``a``, and the rest of the arguments
    (that have been fleshed out with
    :func:`~pyfftw.builders._utils._cook_nd_args`), compute
    the shape the input and output arrays need to be in order
    to satisfy all the requirements for the transform. The input
    shape *may* be different to the shape of a.

    returns:
    ``(input_shape, output_shape)``
    '''
    # Start with the shape of a
    orig_domain_shape = list(a.shape)
    fft_domain_shape = list(a.shape)

    try:
        for n, axis in enumerate(axes):
            orig_domain_shape[axis] = s[n]
            fft_domain_shape[axis] = s[n]

        if real:
            fft_domain_shape[axes[-1]] = s[-1]//2 + 1

    except IndexError:
        raise IndexError('Invalid axes: '
                'At least one of the passed axes is invalid.')

    if inverse:
        input_shape = fft_domain_shape
        output_shape = orig_domain_shape
    else:
        input_shape = orig_domain_shape
        output_shape = fft_domain_shape

    return tuple(input_shape), tuple(output_shape)

def _precook_1d_args(a, n, axis):
    '''Turn ``*(n, axis)`` into ``(s, axes)``
    '''
    if n is not None:
        s = [int(n)]
    else:
        s = None

    # Force an error with an invalid axis
    a.shape[axis]

    return s, (axis,)

def _cook_nd_args(a, s=None, axes=None, invreal=False):
    '''Similar to :func:`numpy.fft.fftpack._cook_nd_args`.
    '''

    if axes is None:
        if s is None:
            len_s = len(a.shape)
        else:
            len_s = len(s)

        axes = list(range(-len_s, 0))

    if s is None:
        s = list(numpy.take(a.shape, axes))

        if invreal:
            s[-1] = (a.shape[axes[-1]] - 1) * 2


    if len(s) != len(axes):
        raise ValueError('Shape error: '
                'Shape and axes have different lengths.')

    if len(s) > len(a.shape):
        raise ValueError('Shape error: '
                'The length of s or axes cannot exceed the dimensionality '
                'of the input array, a.')

    return tuple(s), tuple(axes)
