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

'''A set of builder functions that return FFTW objects. The interface
to create these objects is mostly the same as `numpy.fft
<http://docs.scipy.org/doc/numpy/reference/routines.fft.html>`_, only
instead of the call returning the result of the FFT, an FFTW object is
returned that performs that FFT operation when it is called.

In the case where ``s`` dictates that the passed-in input array be
copied into a different processing array, the returned FFTW object 
is a child class of ``pyfftw.FFTW``, _FFTWWrapper, which wraps the 
call method in order to correctly perform that copying. That is, 
subsequent calls to the object (i.e. through ``__call__``) should
occur with an input array that can be sliced to the same size as
the expected internal array. Note that a side effect of this is
that subsequent calls to the object can be made with an array that
is *bigger* than the original (but not smaller).

Only the call method is wrapped: ``update_arrays`` still expects an
array with the correct size, alignment, dtype etc for the underlying
FFTW object.

When the internal input array is bigger along any axis than the input
array that is passed in (due to ``s`` dictating a larger size), then the
extra entries are padded with zeros. This is a one time action. If the
internal input array is then extracted using
:ref:`FFTW.get_input_array()<FFTW_get_input_array>`, it is possible to
persistently fill the padding space with whatever the user desires, so
subsequent calls with a new input only overwrite the values that aren't
padding (even if the array that is used for the call is bigger than the
original - see the point above about bigger arrays being sliced to
fit).

The precision of the FFT operation is acquired from the input array.
If an array is passed in that is not of float type, or is of an
unknown float type, an attempt is made to convert the array to a
double precision array. This results in a copy being made.

If an array of the incorrect complexity is passed in (e.g. a complex
array is passed to a real transform routine, or vice-versa), then an
attempt is made to convert the array to an array of the correct
complexity. This results in a copy being made.

The following functions are supported. They can be used with the
same calling signature as their respective functions in ``numpy.fft``.

'fft','ifft', 'rfft', 'irfft', 'rfftn',
'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn'

The first caveat is that the dtype of the input array must match the
transform. For example, for ``fft`` and ``ifft``, the dtype must
be complex, for ``rfft`` it must be real, and so on. The other point to
note from this is that the precision of the transform matches the 
precision of the input array. So, if a single precision input array
is passed in, then a single precision transform will be used.

The second caveat is that repeated axes are handled differently: with 
the returned FFTW object, axes that are repeated in the axes argument
are considered only once, as compared to ``numpy.fft`` in which
repeated axes results in the DFT being taken along that axes as many
times as the axis occurs.

Note that unless the ``auto_align_input`` argument to the function
is set to ``True``, the ``FFTW_UNALIGNED`` :ref:`flag<FFTW_flags>` is 
set in the returned FFTW object. This disables some of the FFTW
optimisations that rely on aligned arrays. Also woth noting is that
the ``auto_align_input`` flag only results in a copy when calling the
resultant FFTW object if the input array is not already aligned correctly.

In addition to the arguments that are present with their complementary
functions in ``numpy.fft``, each of these functions also offers the
following additional keyword arguments:
    ``planner_effort``: A string dictating how much effort is spent 
    in planning the FFTW routines. This is passed to the creation
    of the FFTW object as an entry in the flags list. They correspond
    to constants in the underlying `C library 
    <http://www.fftw.org/doc/Planner-Flags.html>`_. 
    The valid strings, in order of their increasing impact on the time 
    to compute  are:
    ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default), ``'FFTW_PATIENT'``
    and ``'FFTW_EXHAUSTIVE'``. 

    ``threads``: The number of threads used to perform the FFT.

    ``auto_align_input``: Correctly byte align the input array for optimal
    usage of vector instructions. This can lead to a substantial speedup.
    Setting this argument to ``True`` makes sure that the input array
    is correctly aligned. It is possible to correctly byte align the array
    prior to calling this function (using, for example,
    :ref:`n_byte_align()<n_byte_align>`). If and only if a realignment is 
    necessary is a new array created. If a new array *is* created, it is 
    up to the calling code to acquire that new input array using 
    :ref:`FFTW.get_input_array()<FFTW_get_input_array>`.

    ``avoid_copy``: By default, these functions will always create a copy 
    (and sometimes more than one) of the passed in input array. This is 
    because the creation of the FFTW object generally destroys the contents 
    of the input array. Setting this argument to ``True`` will try not to 
    create a copy of the input array. This may not be possible if the shape 
    of the FFT input as dictated by ``s`` is necessarily different from the 
    shape of the passed-in array, or the dtypes are incompatible with the
    FFT routine.
'''

from _utils import _precook_1d_args, _Xfftn

__all__ = ['fft','ifft', 'rfft', 'irfft', 'rfftn',
           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']


def fft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = False

    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = False
    
    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def fft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):
    
    inverse = False
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):
    
    inverse = True
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def fftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def rfft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True
    
    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def irfft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def rfft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def irfft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def rfftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def irfftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)



