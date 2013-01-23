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
This module is designed to be a drop-in replacement for :mod:`numpy.fft`.

It works be generating a :class:`pyfftw.FFTW` object behind the scenes
using the :mod:`pyfftw.builders` interface, which is then executed. There
is therefore a potentially substantial overhead when a new plan needs 
to be created. This overhead is down to FFTW's internal planner process.
After a specific transform has been planned once, subsequent calls in which
the input array is equivalent will be fast.

In practice, this means something like the following:

.. doctest::

    >>> import pyfftw
    >>> a = pyfftw.n_byte_align_empty(128, 16)
    >>> fft_a = pyfftw.interfaces.numpy_fft.fft(a) # Will need to plan
    >>> b = pyfftw.n_byte_align_empty(128, 16)
    >>> fft_b = pyfftw.interfaces.numpy_fft.fft(b) # Already planned, so fast
    >>> c = pyfftw.n_byte_align_empty(132, 16)
    >>> fft_c = pyfftw.interfaces.numpy_fft.fft(c) # Needs a new plan

Caching
"""""""

In addition, when a :class:`pyfftw.FFTW` object is created, it is cached for
a short period of time, set by FIXME. Subsequent calls to these methods will
inspect the cache for a suitable pre-existing :class:`pyfftw.FFTW` object. 
If one is found, the cached object is used. Although the time to create
a new :class:`pyfftw.FFTW` is short (assuming that the planner possesses the
necessary wisdom to create the plan immediately), it may still take longer
than a short transform. Recalling a cached object is substantially faster than
creating a new one from scratch. The timeout exists so that FIXME FIXME

.. _numpy_fft_interfaces_args:

Additional Arguments
""""""""""""""""""""

In addition to the equivalent arguments in :mod:`numpy.fft`, all these
functions also expose a subset of the keyword arguments in
:mod:`pyfftw.builders` with the exception that the multidimensional inverse
real FFTs (:func:`pyfftw.interfaces.numpy_fft.irfft2` and
:func:`pyfftw.interfaces.numpy_fft.irfftn`) include the missing
``overwrite_input`` argument.

* ``overwrite_input``: Whether or not the input array can be
  overwritten during the transform. This sometimes results in a faster
  algorithm being made available. It causes the ``'FFTW_DESTROY_INPUT'``
  flag to be passed to the intermediate :class:`pyfftw.FFTW` object. 

  The default is ``False`` to be consistent with :mod:`numpy.fft`.

* ``planner_effort``: A string dictating how much effort is spent 
  in planning the FFTW routines. This is passed to the creation
  of the intermediate :class:`pyfftw.FFTW` object as an entry 
  in the flags list. They correspond to flags passed to the 
  :class:`pyfftw.FFTW` object.

  The valid strings, in order of their increasing impact on the time 
  to compute  are:
  ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default), ``'FFTW_PATIENT'``
  and ``'FFTW_EXHAUSTIVE'``.

  The `Wisdom
  <http://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>`_
  that FFTW has accumulated or has loaded (through
  :func:`pyfftw.import_wisdom`) is used during the creation of
  :class:`pyfftw.FFTW` objects.

  Note that the first time planning stage can take a substantial amount
  of time. For this reason, the default is to use ``'FFTW_ESTIMATE'``, which
  potentially results in a slightly suboptimal plan being used, but with
  a substantially quicker first-time planner step.

* ``threads``: The number of threads used to perform the FFT.

  The default is ``1``.

* ``auto_align_input``: Correctly byte align the input array for optimal
  usage of vector instructions. This can lead to a substantial speedup.

  This argument being ``True`` makes sure that the input array
  is correctly aligned. It is possible to correctly byte align the array
  prior to calling this function (using, for example,
  :func:`pyfftw.n_byte_align`). If and only if a realignment is 
  necessary is a new array created.

  It's worth noting that just being aligned may not be sufficient to
  create the fastest possible transform. For example, if the array is not
  contiguous (i.e. certain axes have gaps in memory between slices), it may
  be faster to plan a transform for a contiguous array, and then rely on
  the array being copied in before the transform (which
  :class:`pyfftw.FFTW` will handle for you). The ``auto_contiguous``
  argument controls whether this function also takes care of making sure
  the array is contiguous or not.

  The default is ``True``.

* ``auto_contiguous``: Make sure the input array is contiguous in
  memory before performing the transform on it. If the array is not
  contiguous, it is copied into an interim array. This is because it
  is often faster to copy the data before the transform and then transform
  a contiguous array than it is to try to take the transform of a 
  non-contiguous array. This is particularly true in conjunction with
  the ``auto_align_input`` argument which is used to make sure that the 
  transform is taken of an aligned array.

  The default is ``True``.

* ``avoid_copy``: Generally these functions will create a copy 
  (and sometimes more than one) of the passed in input array. Setting this
  argument to ``True`` will try not to create a copy of the input array,
  possibly resulting in the input array being destroyed. If it is not 
  possible to create the object without a copy being made, a 
  ``ValueError`` is raised.

  Example situations that require a copy, and so cause the exception
  to be raised when this flag is set:

  * The shape of the FFT input as dictated by ``s`` is
    necessarily different from the shape of the passed-in array.
  * The dtypes are incompatible with the FFT routine.
  * The ``auto_contiguous`` or ``auto_align`` flags are True and 
    the input array is not already contiguous or aligned.

The exceptions raised by each of these functions are as per their
equivalents in :mod:`numpy.fft`.
'''

from ._utils import _Xfftn

__all__ = ['fft','ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn']

def fft(a, n=None, axis=-1, overwrite_input=False, 
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.
    
    The first three arguments are as per :func:`numpy.fft.fft`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''

    calling_func = 'fft'

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def ifft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.ifft`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'ifft'

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)


def fft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.
    
    The first three arguments are as per :func:`numpy.fft.fft2`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'fft2'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def ifft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.ifft2`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'ifft2'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)


def fftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.
    
    The first three arguments are as per :func:`numpy.fft.fftn`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'fftn'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def ifftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.ifftn`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'ifftn'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def rfft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.
    
    The first three arguments are as per :func:`numpy.fft.rfft`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'rfft'

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def irfft(a, n=None, axis=-1, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.irfft`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'irfft'

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def rfft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real FFT.
    
    The first three arguments are as per :func:`numpy.fft.rfft2`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'rfft2'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

def irfft2(a, s=None, axes=(-2,-1), overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.irfft2`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'irfft2'
    
    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)


def rfftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real FFT.
    
    The first three arguments are as per :func:`numpy.fft.rfftn`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'rfftn'

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)


def irfftn(a, s=None, axes=None, overwrite_input=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real inverse FFT.
    
    The first three arguments are as per :func:`numpy.fft.rfftn`; 
    the rest of the arguments are documented 
    :ref:`in the module docs <numpy_fft_interfaces_args>`.
    '''
    calling_func = 'irfftn'
    
    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous, 
            calling_func)

