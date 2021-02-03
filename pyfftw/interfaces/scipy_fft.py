#!/usr/bin/env python
#
# Copyright 2019, The pyFFTW developers
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
This module implements those functions that replace aspects of the
:mod:`scipy.fft` module. This module *provides* the entire documented namespace
of :mod:`scipy.fft`, but those functions that are not included here are
imported directly from :mod:`scipy.fft`.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`scipy.fft`, though there are some corner cases in which
this may not be true.

Some corner (mis)usages of :mod:`scipy.fft` may not transfer neatly.
For example, using :func:`scipy.fft.fft2` with a non 1D array and
a 2D `s` argument will return without exception whereas
:func:`pyfftw.interfaces.scipy_fft.fft2` will raise a `ValueError`.

'''
import os

from . import numpy_fft
from .scipy_fftpack import (_dct, _idct, _dctn, _idctn,
                            _dst, _idst, _dstn, _idstn)

# Complete the namespace (these are not actually used in this module)
from scipy.fft import (hfft2, ihfft2, hfftn, ihfftn,
                       fftshift, ifftshift, fftfreq, rfftfreq,
                       get_workers, set_workers)

# a next_fast_len specific to pyFFTW is used in place of the scipy.fft one
from ..pyfftw import next_fast_len

import scipy.fft as _fft
import numpy as np


__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
           'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
           'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'get_workers',
           'set_workers', 'next_fast_len']


# Backend support for scipy.fft

_implemented = {}

__ua_domain__ = 'numpy.scipy.fft'

_cpu_count = os.cpu_count()


def __ua_function__(method, args, kwargs):
    fn = _implemented.get(method, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)


def _implements(scipy_func):
    '''Decorator adds function to the dictionary of implemented functions'''
    def inner(func):
        _implemented[scipy_func] = func
        return func

    return inner


def _workers_to_threads(workers):
    """Handle conversion of workers to a positive number of threads in the
    same way as scipy.fft.helpers._workers.
    """
    if workers is None:
        return get_workers()

    if workers < 0:
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError("workers value out of range; got {}, must not be"
                             " less than {}".format(workers, -_cpu_count))
    elif workers == 0:
        raise ValueError("workers must not be zero")
    return workers


@_implements(_fft.fft)
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
        planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.

    The first six arguments are as per :func:`scipy.fft.fft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.fft(x, n, axis, norm, overwrite_x, planner_effort,
                         threads, auto_align_input, auto_contiguous)


@_implements(_fft.ifft)
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
         planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.

    The first six arguments are as per :func:`scipy.fft.ifft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.ifft(x, n, axis, norm, overwrite_x,
                          planner_effort, threads, auto_align_input,
                          auto_contiguous)


@_implements(_fft.fft2)
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None,
         planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.

    The first six arguments are as per :func:`scipy.fft.fft2`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.fft2(x, s, axes, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)


@_implements(_fft.ifft2)
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.

    The first six arguments are as per :func:`scipy.fft.ifft2`;
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.ifft2(x, s, axes, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.fftn)
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None,
         planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.

    The first six arguments are as per :func:`scipy.fft.fftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.fftn(x, s, axes, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)


@_implements(_fft.ifftn)
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.

    The first six arguments are as per :func:`scipy.fft.ifftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.ifftn(x, s, axes, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfft)
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
         planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.

    The first six arguments are as per :func:`scipy.fft.rfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')
    threads = _workers_to_threads(workers)
    return numpy_fft.rfft(x, n, axis, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)


@_implements(_fft.irfft)
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.

    The first six arguments are as per :func:`scipy.fft.irfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.irfft(x, n, axis, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfft2)
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real FFT.

    The first six arguments are as per :func:`scipy.fft.rfft2`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')
    threads = _workers_to_threads(workers)
    return numpy_fft.rfft2(x, s, axes, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.irfft2)
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False,
           workers=None, planner_effort=None, auto_align_input=True,
           auto_contiguous=True):
    '''Perform a 2D real inverse FFT.

    The first six arguments are as per :func:`scipy.fft.irfft2`;
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.irfft2(x, s, axes, norm, overwrite_x, planner_effort,
                            threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfftn)
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real FFT.

    The first six arguments are as per :func:`scipy.fft.rfftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')
    threads = _workers_to_threads(workers)
    return numpy_fft.rfftn(x, s, axes, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.irfftn)
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None,
           planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real inverse FFT.

    The first six arguments are as per :func:`scipy.fft.irfftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.irfftn(x, s, axes, norm, overwrite_x, planner_effort,
                            threads, auto_align_input, auto_contiguous)


@_implements(_fft.hfft)
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
         planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D Hermitian FFT.

    The first six arguments are as per :func:`scipy.fft.hfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    return numpy_fft.hfft(x, n, axis, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)


@_implements(_fft.ihfft)
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None,
          planner_effort=None, auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D Hermitian inverse FFT.

    The first six arguments are as per :func:`scipy.fft.ihfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')

    threads = _workers_to_threads(workers)
    return numpy_fft.ihfft(x, n, axis, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.dct)
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, planner_effort=None, auto_align_input=True,
        auto_contiguous=True):
    '''Perform a 1D discrete cosine transform.

    The first seven arguments are as per :func:`scipy.fft.dct`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _dct(x, type=type, n=n, axis=axis, norm=norm,
                overwrite_x=overwrite_x,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input,
                auto_contiguous=auto_contiguous)


@_implements(_fft.idct)
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, planner_effort=None, auto_align_input=True,
         auto_contiguous=True):
    '''Perform an inverse 1D discrete cosine transform.

    The first seven arguments are as per :func:`scipy.fft.idct`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _idct(x, type=type, n=n, axis=axis, norm=norm,
                 overwrite_x=overwrite_x,
                 planner_effort=planner_effort, threads=threads,
                 auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)


@_implements(_fft.dst)
def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, planner_effort=None, auto_align_input=True,
        auto_contiguous=True):
    '''Perform a 1D discrete sine transform.

    The first seven arguments are as per :func:`scipy.fft.dst`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _dst(x, type=type, n=n, axis=axis, norm=norm,
                overwrite_x=overwrite_x,
                planner_effort=planner_effort, threads=threads,
                auto_align_input=auto_align_input,
                auto_contiguous=auto_contiguous)


@_implements(_fft.idst)
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None, planner_effort=None, auto_align_input=True,
         auto_contiguous=True):
    '''Perform an inverse 1D discrete sine transform.

    The first seven arguments are as per :func:`scipy.fft.idst`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _idst(x, type=type, n=n, axis=axis, norm=norm,
                 overwrite_x=overwrite_x,
                 planner_effort=planner_effort, threads=threads,
                 auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)

@_implements(_fft.dctn)
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, planner_effort=None, auto_align_input=True,
         auto_contiguous=True):
    """Performan a multidimensional Discrete Cosine Transform.

    The first seven arguments are as per :func:`scipy.fft.dctn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _dctn(x, type=type, shape=s, axes=axes, norm=norm,
                 overwrite_x=overwrite_x,
                 planner_effort=planner_effort, threads=threads,
                 auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)


@_implements(_fft.idctn)
def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None, planner_effort=None, auto_align_input=True,
          auto_contiguous=True):
    """Performan a multidimensional inverse Discrete Cosine Transform.

    The first seven arguments are as per :func:`scipy.fft.idctn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _idctn(x, type=type, shape=s, axes=axes, norm=norm,
                  overwrite_x=overwrite_x,
                  planner_effort=planner_effort, threads=threads,
                  auto_align_input=auto_align_input,
                  auto_contiguous=auto_contiguous)


@_implements(_fft.dstn)
def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, planner_effort=None, auto_align_input=True,
         auto_contiguous=True):
    """Performan a multidimensional Discrete Sine Transform.

    The first seven arguments are as per :func:`scipy.fft.dstn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _dstn(x, type=type, shape=s, axes=axes, norm=norm,
                 overwrite_x=overwrite_x,
                 planner_effort=planner_effort, threads=threads,
                 auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)


@_implements(_fft.idstn)
def idstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None, planner_effort=None, auto_align_input=True,
          auto_contiguous=True):
    """Performan a multidimensional inverse Discrete Sine Transform.

    The first seven arguments are as per :func:`scipy.fft.idstn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    threads = _workers_to_threads(workers)
    if norm is None:
        norm = 'backward'
    return _idstn(x, type=type, shape=s, axes=axes, norm=norm,
                  overwrite_x=overwrite_x,
                  planner_effort=planner_effort, threads=threads,
                  auto_align_input=auto_align_input,
                  auto_contiguous=auto_contiguous)
