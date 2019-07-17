#!/usr/bin/env python
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
This module implements those functions that replace aspects of the
:mod:`scipy.fft` module. This module *provides* the entire documented namespace
of :mod:`scipy.fft`, but those functions that are not included here are
imported directly from :mod:`scipy.fft`.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`scipy.fft`, though there are some corner cases in which
this may not be true.

Some corner (mis)usages of :mod:`scipy.fft` may not transfer neatly.
For example, using :func:`scipy.fft.fft2` with a non 1D array and
a 2D `shape` argument will return without exception whereas
:func:`pyfftw.interfaces.scipy_fft.fft2` will raise a `ValueError`.

'''

from . import numpy_fft

# Complete the namespace (these are not actually used in this module)
from scipy.fft import (dct, idct, dst, idst, dctn, idctn, dstn, idstn,
                       hfft2, ihfft2, hfftn, ihfftn,
                       fftshift, ifftshift, fftfreq, rfftfreq)

# a next_fast_len specific to pyFFTW is used in place of the scipy.fft one
from ..pyfftw import next_fast_len

import scipy.fft as _fft


__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'hfft2', 'ihfft2', 'hfftn', 'ihfftn',
           'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn',
           'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']


# Backend support for scipy.fft

_implemented = {}

__ua_domain__ = 'numpy.scipy.fft'


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


@_implements(_fft.fft)
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.

    The first five arguments are as per :func:`scipy.fft.fft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.fft(x, n, axis, norm, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)

@_implements(_fft.ifft)
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.

    The first five arguments are as per :func:`scipy.fft.ifft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.ifft(x, n, axis, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.fft2)
def fft2(x, shape=None, axes=(-2,-1), norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.

    The first three arguments are as per :func:`scipy.fft.fft2`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.fft2(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.ifft2)
def ifft2(x, shape=None, axes=(-2,-1), norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.

    The first five arguments are as per :func:`scipy.fft.ifft2`;
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''
    return numpy_fft.ifft2(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.fftn)
def fftn(x, shape=None, axes=None, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.

    The first five arguments are as per :func:`scipy.fft.fftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fft.fftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')
    return numpy_fft.fftn(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.ifftn)
def ifftn(x, shape=None, axes=None, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.

    The first five arguments are as per :func:`scipy.fft.ifftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fft.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')

    return numpy_fft.ifftn(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfft)
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.

    The first five arguments are as per :func:`scipy.fft.rfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.rfft(x, n, axis, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)

@_implements(_fft.irfft)
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.

    The first five arguments are as per :func:`scipy.fft.irfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.irfft(x, n, axis, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfft2)
def rfft2(x, shape=None, axes=(-2,-1), norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real FFT.

    The first five arguments are as per :func:`scipy.fft.rfft2`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.rfft2(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.irfft2)
def irfft2(x, shape=None, axes=(-2,-1), norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real inverse FFT.

    The first five arguments are as per :func:`scipy.fft.irfft2`;
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''
    return numpy_fft.irfft2(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.rfftn)
def rfftn(x, shape=None, axes=None, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real FFT.

    The first five arguments are as per :func:`scipy.fft.rfftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fft.rfftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')
    return numpy_fft.rfftn(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.irfftn)
def irfftn(x, shape=None, axes=None, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real inverse FFT.

    The first five arguments are as per :func:`scipy.fft.irfftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fft.irfftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')

    return numpy_fft.irfftn(x, shape, axes, norm, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


@_implements(_fft.hfft)
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D Hermitian FFT.

    The first five arguments are as per :func:`scipy.fft.hfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.hfft(x, n, axis, norm, overwrite_x, planner_effort,
                          threads, auto_align_input, auto_contiguous)

@_implements(_fft.ihfft)
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D Hermitian inverse FFT.

    The first five arguments are as per :func:`scipy.fft.ihfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    return numpy_fft.ihfft(x, n, axis, norm, overwrite_x, planner_effort,
                           threads, auto_align_input, auto_contiguous)
