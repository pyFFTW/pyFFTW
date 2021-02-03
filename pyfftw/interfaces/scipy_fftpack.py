#!/usr/bin/env python
#
# Copyright 2014 Knowledge Economy Developments Ltd
# Copyright 2014 David Wells
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
:mod:`scipy.fftpack` module. This module *provides* the entire documented
namespace of :mod:`scipy.fftpack`, but those functions that are not included
here are imported directly from :mod:`scipy.fftpack`.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`scipy.fftpack`, though there are some corner cases in
which this may not be true.

Some corner (mis)usages of :mod:`scipy.fftpack` may not transfer neatly.
For example, using :func:`scipy.fftpack.fft2` with a non 1D array and
a 2D `shape` argument will return without exception whereas
:func:`pyfftw.interfaces.scipy_fftpack.fft2` will raise a `ValueError`.
'''

import itertools as it
import math
from . import numpy_fft

from ..builders._utils import _default_effort, _default_threads, _cook_nd_args
from ._utils import _Xfftn
import numpy

# Complete the namespace (these are not actually used in this module)
from scipy.fftpack import (diff, tilbert, itilbert,
        hilbert, ihilbert, cs_diff, sc_diff, ss_diff, cc_diff,
        shift, fftshift, ifftshift, fftfreq, rfftfreq,
        convolve)

# a next_fast_len specific to pyFFTW is used in place of the scipy.fftpack one
from ..pyfftw import next_fast_len

try:
    # scipy 1.2.0 introduced helpers for validating shape and axes
    from scipy.fftpack.helper import (
        _init_nd_shape_and_axes_sorted, _init_nd_shape_and_axes_sorted)
except ImportError:
    _init_nd_shape_and_axes_sorted = None
    _init_nd_shape_and_axes = None


__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2',
           'dct', 'idct', 'dst', 'idst', 'diff', 'tilbert', 'itilbert',
           'hilbert', 'ihilbert', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff',
           'shift', 'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'convolve',
           'next_fast_len', 'dctn', 'idctn', 'dstn', 'idstn']

_NORM_MAP = {None: 0, 'backward': 0, 'ortho': 1, 'forward': 2}


def _normalization(norm, forward):
    """Returns the pypocketfft normalization mode from the norm argument"""
    try:
        inorm = _NORM_MAP[norm]
        return inorm if forward else (2 - inorm)
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm!r}, should '
             'be "backward", "ortho" or "forward"') from None


def _norm_factor(inorm, n):
    if inorm == 0:
        return 1
    elif inorm == 2:
        return 1 / n
    elif inorm == 1:
        return 1 / math.sqrt(n)


def _norm_factor_nd(inorm, shape, axes, fct=1, delta=0):
    if inorm == 0:
        return 1
    n = 1
    for ax in axes:
        n *= fct * (shape[ax] + delta)
    return _norm_factor(inorm, n)


def _dct_norm_factor(norm, forward, type, shape, axes):
    inorm = _normalization(norm, forward)
    if type == 1:
        return _norm_factor_nd(inorm, shape, axes, fct=2, delta=-1)
    else:
        return _norm_factor_nd(inorm, shape, axes, fct=2)


def _dst_norm_factor(norm, forward, type, shape, axes):
    inorm = _normalization(norm, forward)
    if type == 1:
        return _norm_factor_nd(inorm, shape, axes, fct=2, delta=1)
    else:
        return _norm_factor_nd(inorm, shape, axes, fct=2)


def fft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.

    The first three arguments are as per :func:`scipy.fftpack.fft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    return numpy_fft.fft(x, n, axis, None, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)

def ifft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.

    The first three arguments are as per :func:`scipy.fftpack.ifft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    return numpy_fft.ifft(x, n, axis, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.

    The first three arguments are as per :func:`scipy.fftpack.fft2`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    return numpy_fft.fft2(x, shape, axes, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.

    The first three arguments are as per :func:`scipy.fftpack.ifft2`;
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    return numpy_fft.ifft2(x, shape, axes, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


def fftn(x, shape=None, axes=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.

    The first three arguments are as per :func:`scipy.fftpack.fftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    if _init_nd_shape_and_axes_sorted is not None:
        shape, axes = _init_nd_shape_and_axes_sorted(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.fftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    return numpy_fft.fftn(x, shape, axes, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


def ifftn(x, shape=None, axes=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.

    The first three arguments are as per :func:`scipy.fftpack.ifftn`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)
    if _init_nd_shape_and_axes_sorted is not None:
        shape, axes = _init_nd_shape_and_axes_sorted(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')

    return numpy_fft.ifftn(x, shape, axes, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)

def _complex_to_rfft_output(complex_output, output_shape, axis):
    '''Convert the complex output from pyfftw to the real output expected
    from :func:`scipy.fftpack.rfft`.
    '''

    rfft_output = numpy.empty(output_shape, dtype=complex_output.real.dtype)
    source_slicer = [slice(None)] * complex_output.ndim
    target_slicer = [slice(None)] * complex_output.ndim

    # First element
    source_slicer[axis] = slice(0, 1)
    target_slicer[axis] = slice(0, 1)
    rfft_output[tuple(target_slicer)] = complex_output[tuple(source_slicer)].real

    # Real part
    source_slicer[axis] = slice(1, None)
    target_slicer[axis] = slice(1, None, 2)
    rfft_output[tuple(target_slicer)] = complex_output[tuple(source_slicer)].real

    # Imaginary part
    if output_shape[axis] % 2 == 0:
        end_val = -1
    else:
        end_val = None

    source_slicer[axis] = slice(1, end_val, None)
    target_slicer[axis] = slice(2, None, 2)
    rfft_output[tuple(target_slicer)] = complex_output[tuple(source_slicer)].imag

    return rfft_output


def _irfft_input_to_complex(irfft_input, axis):
    '''Convert the expected real input to :func:`scipy.fftpack.irfft` to
    the complex input needed by pyfftw.
    '''
    complex_dtype = numpy.result_type(irfft_input, 1j)

    input_shape = list(irfft_input.shape)
    input_shape[axis] = input_shape[axis]//2 + 1

    complex_input = numpy.empty(input_shape, dtype=complex_dtype)
    source_slicer = [slice(None)] * len(input_shape)
    target_slicer = [slice(None)] * len(input_shape)

    # First element
    source_slicer[axis] = slice(0, 1)
    target_slicer[axis] = slice(0, 1)
    complex_input[tuple(target_slicer)] = irfft_input[tuple(source_slicer)]

    # Real part
    source_slicer[axis] = slice(1, None, 2)
    target_slicer[axis] = slice(1, None)
    complex_input[tuple(target_slicer)].real = irfft_input[tuple(source_slicer)]

    # Imaginary part
    if irfft_input.shape[axis] % 2 == 0:
        end_val = -1
        target_slicer[axis] = slice(-1, None)
        complex_input[tuple(target_slicer)].imag = 0.0
    else:
        end_val = None

    source_slicer[axis] = slice(2, None, 2)
    target_slicer[axis] = slice(1, end_val)
    complex_input[tuple(target_slicer)].imag = irfft_input[tuple(source_slicer)]

    return complex_input


def rfft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.

    The first three arguments are as per :func:`scipy.fftpack.rfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    if not numpy.isrealobj(x):
        raise TypeError('Input array must be real to maintain '
                'compatibility with scipy.fftpack.rfft.')

    x = numpy.asanyarray(x)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    complex_output = numpy_fft.rfft(x, n, axis, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)

    output_shape = list(x.shape)
    if n is not None:
        output_shape[axis] = n

    return _complex_to_rfft_output(complex_output, output_shape, axis)

def irfft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.

    The first three arguments are as per :func:`scipy.fftpack.irfft`;
    the rest of the arguments are documented
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''
    if not numpy.isrealobj(x):
        raise TypeError('Input array must be real to maintain '
                'compatibility with scipy.fftpack.irfft.')

    x = numpy.asanyarray(x)
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    if n is None:
        n = x.shape[axis]

    complex_input = _irfft_input_to_complex(x, axis)

    return numpy_fft.irfft(complex_input, n, axis, None, overwrite_x,
            planner_effort, threads, auto_align_input, auto_contiguous)


def _dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         planner_effort=None, threads=None, auto_align_input=True,
         auto_contiguous=True):
    if not numpy.isrealobj(x):
        raise TypeError("1st argument must be real sequence")

    x = numpy.asanyarray(x)
    if n is None:
        n = x.shape[axis]
    elif n != x.shape[axis]:
        raise NotImplementedError("Padding/truncating not yet implemented")

    if norm not in [None, 'forward', 'backward', 'ortho']:
        raise ValueError("Unknown normalize mode %s" % norm)

    if norm == 'ortho':
        if type == 1:
            x = numpy.copy(x)
            sp = list(it.repeat(slice(None), len(x.shape)))
            sp[axis] = 0
            x[tuple(sp)] *= math.sqrt(2)
            sp[axis] = -1
            x[tuple(sp)] *= math.sqrt(2)
        elif type == 3:
            x = numpy.copy(x)
            sp = list(it.repeat(slice(None), len(x.shape)))
            sp[axis] = 0
            x[tuple(sp)] /= math.sqrt(x.shape[axis])
            sp[axis] = slice(1, None, None)
            x[tuple(sp)] /= math.sqrt(2*x.shape[axis])

    type_flag_lookup = {
        1: 'FFTW_REDFT00',
        2: 'FFTW_REDFT10',
        3: 'FFTW_REDFT01',
        4: 'FFTW_REDFT11',
    }
    try:
        type_flag = type_flag_lookup[type]
    except KeyError:
        raise ValueError("Type %d not understood" % type)

    calling_func = 'dct'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    result_unnormalized = _Xfftn(x, n, axis, overwrite_x, planner_effort,
                                 threads, auto_align_input, auto_contiguous,
                                 calling_func, real_direction_flag=type_flag)
    if norm is None:
        return result_unnormalized
    result = result_unnormalized
    if norm == 'ortho':
        if type == 1:
            sp = list(it.repeat(slice(None), len(x.shape)))
            sf_ends = 1 / math.sqrt(2)
            sp[axis] = 0
            result[tuple(sp)] *= sf_ends
            sp[axis] = -1
            result[tuple(sp)] *= sf_ends
            result *= 1 / math.sqrt(2 * (x.shape[axis] - 1))
        elif type == 2:
            sp = list(it.repeat(slice(None), len(x.shape)))
            sp[axis] = 0
            result[tuple(sp)] /= math.sqrt(2)
            result *= 1 / math.sqrt(2 * x.shape[axis])
        elif type == 4:
            result *= 1 / math.sqrt(2 * x.shape[axis])
    elif norm == 'forward':
        if type == 1:
            result *= 1 / (2 * (x.shape[axis] - 1))
        else:
            result *= 1 / (2 * x.shape[axis])
    return result


def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D discrete cosine transform.

    The first three arguments are as per :func:`scipy.fftpack.dct`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _dct(x, type, n, axis, norm, overwrite_x, planner_effort, threads,
                auto_align_input, auto_contiguous)


inverse_dct_norm = {None: None, 'ortho': 'ortho', 'backward': 'forward',
                   'forward': 'backward'}


def _idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
          planner_effort=None, threads=None, auto_align_input=True,
          auto_contiguous=True):
    try:
        inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    except KeyError:
        raise ValueError("Type %d not understood" % type)

    try:
        norm = inverse_dct_norm[norm]
    except KeyError:
        raise ValueError("Norm type {} not understood".format(norm))

    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _dct(x, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x,
                type=inverse_type, planner_effort=planner_effort,
                threads=threads, auto_align_input=auto_align_input,
                auto_contiguous=auto_contiguous)


def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         planner_effort=None, threads=None,
         auto_align_input=True, auto_contiguous=True):
    '''Perform an inverse 1D discrete cosine transform.

    The first three arguments are as per :func:`scipy.fftpack.idct`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _idct(x, type, n, axis, norm, overwrite_x, planner_effort, threads,
                 auto_align_input, auto_contiguous)


def _dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         planner_effort=None, threads=None,
         auto_align_input=True, auto_contiguous=True):
    if not numpy.isrealobj(x):
        raise TypeError("1st argument must be real sequence")

    x = numpy.asanyarray(x)
    if n is None:
        n = x.shape[axis]
    elif n != x.shape[axis]:
        raise NotImplementedError("Padding/truncating not yet implemented")

    if norm not in [None, 'forward', 'backward', 'ortho']:
        raise ValueError("Unknown normalize mode %s" % norm)

    if type == 3 and norm == 'ortho':
        x = numpy.copy(x)
        sp = list(it.repeat(Ellipsis, len(x.shape)))
        sp[axis] = 0
        x[tuple(sp)] /= math.sqrt(x.shape[axis])
        sp[axis] = slice(1, None, None)
        x[tuple(sp)] /= math.sqrt(2*x.shape[axis])

    type_flag_lookup = {
        1: 'FFTW_RODFT00',
        2: 'FFTW_RODFT10',
        3: 'FFTW_RODFT01',
        4: 'FFTW_RODFT11',
    }
    try:
        type_flag = type_flag_lookup[type]
    except KeyError:
        raise ValueError("Type %d not understood" % type)

    calling_func = 'dst'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    result_unnormalized = _Xfftn(x, n, axis, overwrite_x, planner_effort,
                                 threads, auto_align_input, auto_contiguous,
                                 calling_func, real_direction_flag=type_flag)
    if norm is None:
        return result_unnormalized
    result = result_unnormalized
    if norm == 'ortho':
        if type == 1:
            result *= 1 / math.sqrt(2 * (x.shape[axis] + 1))
        elif type == 2:
            sp = list(it.repeat(Ellipsis, len(x.shape)))
            sp[axis] = 0
            result[tuple(sp)] *= 1 / math.sqrt(2)
            result *= 1 / math.sqrt(2 * x.shape[axis])
        elif type == 4:
            result *= 1 / math.sqrt(2 * x.shape[axis])
    elif norm == 'forward':
        if type == 1:
            result *= 1 / (2 * (x.shape[axis] + 1))
        else:
            result *= 1 / (2 * x.shape[axis])
    return result


def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D discrete sine transform.

    The first three arguments are as per :func:`scipy.fftpack.dst`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _dst(x, type, n, axis, norm, overwrite_x, planner_effort, threads,
                auto_align_input, auto_contiguous)


def _idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
          planner_effort=None, threads=None,
          auto_align_input=True, auto_contiguous=True):

    try:
        inverse_type = {1: 1, 2: 3, 3:2, 4: 4}[type]
    except KeyError:
        raise ValueError("Type %d not understood" % type)

    try:
        norm = inverse_dct_norm[norm]
    except KeyError:
        raise ValueError("Norm type {} not understood".format(norm))

    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _dst(x, n=n, axis=axis, norm=norm, overwrite_x=overwrite_x,
                type=inverse_type, planner_effort=planner_effort,
                threads=threads, auto_align_input=auto_align_input,
                auto_contiguous=auto_contiguous)


def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an inverse 1D discrete sine transform.

    The first three arguments are as per :func:`scipy.fftpack.idst`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _idst(x, type, n, axis, norm, overwrite_x, planner_effort, threads,
                auto_align_input, auto_contiguous)


def _dctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
          planner_effort=None, threads=None,
          auto_align_input=True, auto_contiguous=True):
    x = numpy.asanyarray(x)
    if _init_nd_shape_and_axes is not None:
        shape, axes = _init_nd_shape_and_axes(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')
        if numpy.isscalar(shape):
            shape = (shape, )
        if numpy.isscalar(axes):
            axes = (axes, )
        shape, axes = _cook_nd_args(x, s=shape, axes=axes, invreal=False)
    for n, ax in zip(shape, axes):
        x = _dct(x, type=type, n=n, axis=ax, norm=norm,
                 overwrite_x=overwrite_x, planner_effort=planner_effort,
                 threads=threads, auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)
    return x


def dctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
         planner_effort=None, threads=None, auto_align_input=True,
         auto_contiguous=True):
    """Perform a multidimensional Discrete Cosine Transform.

    The first six arguments are as per :func:`scipy.fftpack.dctn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _dctn(x, type, shape, axes, norm, overwrite_x, planner_effort,
                 threads, auto_align_input, auto_contiguous)


def _idctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
           planner_effort=None, threads=None,
           auto_align_input=True, auto_contiguous=True):
    x = numpy.asanyarray(x)
    if _init_nd_shape_and_axes is not None:
        shape, axes = _init_nd_shape_and_axes(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')
        if numpy.isscalar(shape):
            shape = (shape, )
        if numpy.isscalar(axes):
            axes = (axes, )
        shape, axes = _cook_nd_args(x, s=shape, axes=axes, invreal=False)
    for n, ax in zip(shape, axes):
        x = _idct(x, type=type, n=n, axis=ax, norm=norm,
                  overwrite_x=overwrite_x, planner_effort=planner_effort,
                  threads=threads, auto_align_input=auto_align_input,
                  auto_contiguous=auto_contiguous)
    return x


def idctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
          planner_effort=None, threads=None, auto_align_input=True,
          auto_contiguous=True):
    """Perform a multidimensional inverse Discrete Cosine Transform.

    The first six arguments are as per :func:`scipy.fftpack.idctn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _idctn(x, type, shape, axes, norm, overwrite_x, planner_effort,
                  threads, auto_align_input, auto_contiguous)


def _dstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
          planner_effort=None, threads=None,
          auto_align_input=True, auto_contiguous=True):
    x = numpy.asanyarray(x)
    if _init_nd_shape_and_axes is not None:
        shape, axes = _init_nd_shape_and_axes(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')
        if numpy.isscalar(shape):
            shape = (shape, )
        if numpy.isscalar(axes):
            axes = (axes, )
        shape, axes = _cook_nd_args(x, s=shape, axes=axes, invreal=False)
    for n, ax in zip(shape, axes):
        x = _dst(x, type=type, n=n, axis=ax, norm=norm,
                 overwrite_x=overwrite_x, planner_effort=planner_effort,
                 threads=threads, auto_align_input=auto_align_input,
                 auto_contiguous=auto_contiguous)
    return x



def dstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
         planner_effort=None, threads=None, auto_align_input=True,
         auto_contiguous=True):
    """Perform a multidimensional Discrete Sine Transform.

    The first six arguments are as per :func:`scipy.fftpack.dstn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _dstn(x, type, shape, axes, norm, overwrite_x, planner_effort,
                 threads, auto_align_input, auto_contiguous)


def _idstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
           planner_effort=None, threads=None,
           auto_align_input=True, auto_contiguous=True):
    x = numpy.asanyarray(x)
    if _init_nd_shape_and_axes is not None:
        shape, axes = _init_nd_shape_and_axes(x, shape, axes)
    else:
        if shape is not None:
            if ((axes is not None and len(shape) != len(axes)) or
                    (axes is None and len(shape) != x.ndim)):
                raise ValueError(
                    'Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, '
                    'consider using the numpy interface.')
        if numpy.isscalar(shape):
            shape = (shape, )
        if numpy.isscalar(axes):
            axes = (axes, )
        shape, axes = _cook_nd_args(x, s=shape, axes=axes, invreal=False)
    for n, ax in zip(shape, axes):
        x = _idst(x, type=type, n=n, axis=ax, norm=norm,
                  overwrite_x=overwrite_x, planner_effort=planner_effort,
                  threads=threads, auto_align_input=auto_align_input,
                  auto_contiguous=auto_contiguous)
    return x


def idstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False,
          planner_effort=None, threads=None, auto_align_input=True,
          auto_contiguous=True):
    """Perform a multidimensional inverse Discrete Sine Transform.

    The first six arguments are as per :func:`scipy.fftpack.idstn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    """
    if norm not in [None, 'ortho']:
        raise ValueError("unrecognized norm: {}".format(norm))
    return _idstn(x, type, shape, axes, norm, overwrite_x, planner_effort,
                  threads, auto_align_input, auto_contiguous)
