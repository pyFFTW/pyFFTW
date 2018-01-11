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
Overview
""""""""

This module contains a set of functions that return
:class:`pyfftw.FFTW` objects.

The interface to create these objects is mostly the same as
:mod:`numpy.fft`, only instead of the call returning the result of the
FFT, a :class:`pyfftw.FFTW` object is returned that performs that FFT
operation when it is called. Users should be familiar with
:mod:`!numpy.fft` before reading on.

In the case where the shape argument, ``s`` (or ``n`` in the
1-dimensional case), dictates that the passed-in input array be copied
into a different processing array, the returned object is an
instance of a child class of :class:`pyfftw.FFTW`,
:class:`~pyfftw.builders._utils._FFTWWrapper`, which wraps the call
method in order to correctly perform that copying. That is, subsequent
calls to the object (i.e. through
:meth:`~pyfftw.builders._utils._FFTWWrapper.__call__`) should occur
with an input array that can be sliced to the same size as the
expected internal array. Note that a side effect of this is that
subsequent calls to the object can be made with an array that is
*bigger* than the original (but not smaller).

Only the call method is wrapped; :meth:`~pyfftw.FFTW.update_arrays`
still expects an array with the correct size, alignment, dtype etc for
the :class:`pyfftw.FFTW` object.

When the internal input array is bigger along any axis than the input
array that is passed in (due to ``s`` dictating a larger size), then the
extra entries are padded with zeros. This is a one time action. If the
internal input array is then extracted using
:attr:`pyfftw.FFTW.input_array`, it is possible to
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

Although the array that is internal to the :class:`pyfftw.FFTW` object
will be correctly loaded with the values within the input array, it is
not necessarily the case that the internal array *is* the input array.
The actual internal input array can always be retrieved with
:attr:`pyfftw.FFTW.input_array`.

The behaviour of the ``norm`` option in all builder routines matches that of
the corresponding numpy functions.  In short, if ``norm`` is ``None``, then the
output from the forward DFT is unscaled and the inverse DFT is scaled by 1/N,
where N is the product of the lengths of input array on which the FFT is taken.
If ``norm == 'ortho'``, then the output of both forward and inverse DFT
operations are scaled by 1/sqrt(N).

**Example:**

.. doctest::

    >>> import pyfftw
    >>> a = pyfftw.empty_aligned(4, dtype='complex128')
    >>> fft = pyfftw.builders.fft(a)
    >>> a[:] = [1, 2, 3, 4]
    >>> fft() # returns the output
    array([ 10.+0.j,  -2.+2.j,  -2.+0.j,  -2.-2.j])

More examples can be found in the :doc:`tutorial </source/tutorial>`.

Supported Functions and Caveats
"""""""""""""""""""""""""""""""

The following functions are supported. They can be used with the
same calling signature as their respective functions in
:mod:`numpy.fft`.

**Standard FFTs**

* :func:`~pyfftw.builders.fft`
* :func:`~pyfftw.builders.ifft`
* :func:`~pyfftw.builders.fft2`
* :func:`~pyfftw.builders.ifft2`
* :func:`~pyfftw.builders.fftn`
* :func:`~pyfftw.builders.ifftn`

**Real FFTs**

* :func:`~pyfftw.builders.rfft`
* :func:`~pyfftw.builders.irfft`
* :func:`~pyfftw.builders.rfft2`
* :func:`~pyfftw.builders.irfft2`
* :func:`~pyfftw.builders.rfftn`
* :func:`~pyfftw.builders.irfftn`

The first caveat is that the dtype of the input array must match the
transform. For example, for ``fft`` and ``ifft``, the dtype must
be complex, for ``rfft`` it must be real, and so on. The other point
to note from this is that the precision of the transform matches the
precision of the input array. So, if a single precision input array is
passed in, then a single precision transform will be used.

The second caveat is that repeated axes are handled differently; with
the returned :class:`pyfftw.FFTW` object, axes that are repeated in the
axes argument are considered only once, as compared to :mod:`numpy.fft`
in which repeated axes results in the DFT being taken along that axes
as many times as the axis occurs (this is down to the underlying
library).

Note that unless the ``auto_align_input`` argument to the function
is set to ``True``, the ``'FFTW_UNALIGNED'`` :ref:`flag <FFTW_flags>`
is set in the returned :class:`pyfftw.FFTW` object. This disables some
of the FFTW optimisations that rely on aligned arrays. Also worth
noting is that the ``auto_align_input`` flag only results in a copy
when calling the resultant :class:`pyfftw.FFTW` object if the input
array is not already aligned correctly.

.. _builders_args:

Additional Arguments
""""""""""""""""""""

In addition to the arguments that are present with their complementary
functions in :mod:`numpy.fft`, each of these functions also offers the
following additional keyword arguments:

* ``overwrite_input``: Whether or not the input array can be
  overwritten during the transform. This sometimes results in a faster
  algorithm being made available. It causes the ``'FFTW_DESTROY_INPUT'``
  flag to be passed to the :class:`pyfftw.FFTW` object. This flag is not
  offered for the multi-dimensional inverse real transforms, as FFTW is
  unable to not overwrite the input in that case.

* ``planner_effort``: A string dictating how much effort is spent
  in planning the FFTW routines. This is passed to the creation
  of the :class:`pyfftw.FFTW` object as an entry in the flags list.
  They correspond to flags passed to the :class:`pyfftw.FFTW` object.

  The valid strings, in order of their increasing impact on the time
  to compute  are:
  ``'FFTW_ESTIMATE'``, ``config.PLANNER_EFFORT`` (default), ``'FFTW_PATIENT'``
  and ``'FFTW_EXHAUSTIVE'``.

  The `Wisdom
  <http://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>`_
  that FFTW has accumulated or has loaded (through
  :func:`pyfftw.import_wisdom`) is used during the creation of
  :class:`pyfftw.FFTW` objects.

* ``threads``: The number of threads used to perform the FFT.

* ``auto_align_input``: Correctly byte align the input array for optimal
  usage of vector instructions. This can lead to a substantial speedup.

  Setting this argument to ``True`` makes sure that the input array
  is correctly aligned. It is possible to correctly byte align the array
  prior to calling this function (using, for example,
  :func:`pyfftw.byte_align`). If and only if a realignment is
  necessary is a new array created. If a new array *is* created, it is
  up to the calling code to acquire that new input array using
  :attr:`pyfftw.FFTW.input_array`.

  The resultant :class:`pyfftw.FFTW` object that is created will be
  designed to operate on arrays that are aligned. If the object is
  called with an unaligned array, this would result in a copy. Despite
  this, it may still be faster to set the ``auto_align_input`` flag
  and incur a copy with unaligned arrays than to set up an object
  that uses aligned arrays.

  It's worth noting that just being aligned may not be sufficient to
  create the fastest possible transform. For example, if the array is not
  contiguous (i.e. certain axes have gaps in memory between slices), it
  may be faster to plan a transform for a contiguous array, and then rely
  on the array being copied in before the transform (which
  :class:`pyfftw.FFTW` will handle for you). The ``auto_contiguous``
  argument controls whether this function also takes care of making sure
  the array is contiguous or not.

* ``auto_contiguous``: Make sure the input array is contiguous in
  memory before performing the transform on it. If the array is not
  contiguous, it is copied into an interim array. This is because it
  is often faster to copy the data before the transform and then transform
  a contiguous array than it is to try to take the transform of a
  non-contiguous array. This is particularly true in conjunction with
  the ``auto_align_input`` argument which is used to make sure that the
  transform is taken of an aligned array.

  Like ``auto_align_input``, If a new array is created, it is
  up to the calling code to acquire that new input array using
  :attr:`pyfftw.FFTW.input_array`.

* ``avoid_copy``: By default, these functions will always create a copy
  (and sometimes more than one) of the passed in input array. This is
  because the creation of the :class:`pyfftw.FFTW` object generally
  destroys the contents of the input array. Setting this argument to
  ``True`` will try not to create a copy of the input array, likely
  resulting in the input array being destroyed. If it is not possible
  to create the object without a copy being made, a ``ValueError`` is
  raised.

  Example situations that require a copy, and so cause the exception
  to be raised when this flag is set:

  * The shape of the FFT input as dictated by ``s`` is
    necessarily different from the shape of the passed-in array.

  * The dtypes are incompatible with the FFT routine.

  * The ``auto_contiguous`` or ``auto_align`` flags are True and
    the input array is not already contiguous or aligned.

  This argument is distinct from ``overwrite_input`` in that it only
  influences a copy during the creation of the object. It changes no
  flags in the :class:`pyfftw.FFTW` object.

The exceptions raised by each of these functions are as per their
equivalents in :mod:`numpy.fft`, or as documented above.
'''

from ._utils import (_precook_1d_args, _Xfftn, _norm_args, _default_effort,
                     _default_threads)

__all__ = ['fft','ifft', 'fft2', 'ifft2', 'fftn',
           'ifftn', 'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn',
           'irfftn']


def fft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 1D FFT.

    The first three arguments are as per :func:`numpy.fft.fft`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''
    inverse = False
    real = False

    s, axes = _precook_1d_args(a, n, axis)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def ifft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 1D
    inverse FFT.

    The first three arguments are as per :func:`numpy.fft.ifft`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''

    inverse = True
    real = False

    s, axes = _precook_1d_args(a, n, axis)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))


def fft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 2D FFT.

    The first three arguments are as per :func:`numpy.fft.fft2`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = False
    real = False
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def ifft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a
    2D inverse FFT.

    The first three arguments are as per :func:`numpy.fft.ifft2`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = True
    real = False
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))


def fftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a n-D FFT.

    The first three arguments are as per :func:`numpy.fft.fftn`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = False
    real = False
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def ifftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing an n-D
    inverse FFT.

    The first three arguments are as per :func:`numpy.fft.ifftn`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = True
    real = False
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def rfft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 1D
    real FFT.

    The first three arguments are as per :func:`numpy.fft.rfft`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = False
    real = True

    s, axes = _precook_1d_args(a, n, axis)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def irfft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 1D
    real inverse FFT.

    The first three arguments are as per :func:`numpy.fft.irfft`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = True
    real = True

    s, axes = _precook_1d_args(a, n, axis)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def rfft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 2D
    real FFT.

    The first three arguments are as per :func:`numpy.fft.rfft2`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''

    inverse = False
    real = True
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))

def irfft2(a, s=None, axes=(-2,-1),
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing a 2D
    real inverse FFT.

    The first three arguments are as per :func:`numpy.fft.irfft2`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = True
    real = True

    overwrite_input = True
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))


def rfftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing an n-D
    real FFT.

    The first three arguments are as per :func:`numpy.fft.rfftn`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = False
    real = True
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))


def irfftn(a, s=None, axes=None,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True,
        avoid_copy=False, norm=None):
    '''Return a :class:`pyfftw.FFTW` object representing an n-D
    real inverse FFT.

    The first three arguments are as per :func:`numpy.fft.rfftn`;
    the rest of the arguments are documented
    :ref:`in the module docs <builders_args>`.
    '''


    inverse = True
    real = True

    overwrite_input = True
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            avoid_copy, inverse, real, **_norm_args(norm))
