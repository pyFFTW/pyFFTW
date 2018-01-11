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
:mod:`scipy.fftpack` module. This module *provides* the entire documented
namespace of :mod:`scipy.fftpack`, but those functions that are not included
here are imported directly from :mod:`scipy.fftpack`.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`scipy.fftpack`, though there are some corner cases in
which this may not be true.

It is notable that unlike :mod:`scipy.fftpack`, these functions will
generally return an output array with the same precision as the input
array, and the transform that is chosen is chosen based on the precision
of the input array. That is, if the input array is 32-bit floating point,
then the transform will be 32-bit floating point and so will the returned
array. Half precision input will be converted to single precision.  Otherwise,
if any type conversion is required, the default will be double precision.

Some corner (mis)usages of :mod:`scipy.fftpack` may not transfer neatly.
For example, using :func:`scipy.fftpack.fft2` with a non 1D array and
a 2D `shape` argument will return without exception whereas
:func:`pyfftw.interfaces.scipy_fftpack.fft2` will raise a `ValueError`.
'''

from . import numpy_fft

from ..builders._utils import _default_effort, _default_threads
import numpy

# Complete the namespace (these are not actually used in this module)
from scipy.fftpack import (dct, idct, dst, idst, diff, tilbert, itilbert,
        hilbert, ihilbert, cs_diff, sc_diff, ss_diff, cc_diff,
        shift, fftshift, ifftshift, fftfreq, rfftfreq,
        convolve, _fftpack)

# a next_fast_len specific to pyFFTW is used in place of the scipy.fftpack one
from ..pyfftw import next_fast_len


__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2',
           'dct', 'idct', 'dst', 'idst', 'diff', 'tilbert', 'itilbert',
           'hilbert', 'ihilbert', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff',
           'shift', 'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'convolve',
           'next_fast_len', ]

try:
    from scipy.fftpack import dctn, idctn, dstn, idstn
    __all__ += ['dctn', 'idctn', 'dstn', 'idstn']
except ImportError:
    pass


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

    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.fftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')
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
    if shape is not None:
        if ((axes is not None and len(shape) != len(axes)) or
                (axes is None and len(shape) != x.ndim)):
            raise ValueError('Shape error: In order to maintain better '
                    'compatibility with scipy.fftpack.ifftn, a ValueError '
                    'is raised when the length of the shape argument is '
                    'not the same as x.ndim if axes is None or the length '
                    'of axes if it is not. If this is problematic, consider '
                    'using the numpy interface.')

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
