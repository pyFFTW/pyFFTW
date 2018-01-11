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
:mod:`numpy.fft` module. This module *provides* the entire documented namespace
of :mod:`numpy.fft`, but those functions that are not included here are imported
directly from :mod:`numpy.fft`.


It is notable that unlike :mod:`numpy.fftpack`, these functions will generally
return an output array with the same precision as the input array, and the
transform that is chosen is chosen based on the precision of the input array.
That is, if the input array is 32-bit floating point, then the transform will
be 32-bit floating point and so will the returned array. Half precision input
will be converted to single precision. Otherwise, if any type conversion is
required, the default will be double precision. If pyFFTW was not built with
support for double precision, the default is long double precision. If that is not
available, it defaults to single precision.

One known caveat is that repeated axes are handled differently to
:mod:`numpy.fft`; axes that are repeated in the axes argument are considered
only once, as compared to :mod:`numpy.fft` in which repeated axes results in
the DFT being taken along that axes as many times as the axis occurs.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`numpy.fft`, though there are some corner cases in
which this may not be true.
'''

from ._utils import _Xfftn
from ..builders._utils import (_norm_args, _unitary, _default_effort,
                               _default_threads)

# Complete the namespace (these are not actually used in this module)
from numpy.fft import fftfreq, fftshift, ifftshift

import numpy

__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'hfft', 'ihfft', 'fftfreq', 'fftshift', 'ifftshift']

try:
    # rfftfreq was added to the namespace in numpy 1.8
    from numpy.fft import rfftfreq
    __all__ += ['rfftfreq', ]
except ImportError:
    pass


def fft(a, n=None, axis=-1, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.

    The first four arguments are as per :func:`numpy.fft.fft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''

    calling_func = 'fft'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))

def ifft(a, n=None, axis=-1, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.

    The first four arguments are as per :func:`numpy.fft.ifft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'ifft'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def fft2(a, s=None, axes=(-2,-1), norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.

    The first four arguments are as per :func:`numpy.fft.fft2`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'fft2'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))

def ifft2(a, s=None, axes=(-2,-1), norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.

    The first four arguments are as per :func:`numpy.fft.ifft2`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'ifft2'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def fftn(a, s=None, axes=None, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.

    The first four arguments are as per :func:`numpy.fft.fftn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'fftn'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def ifftn(a, s=None, axes=None, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.

    The first four arguments are as per :func:`numpy.fft.ifftn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'ifftn'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def rfft(a, n=None, axis=-1, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.

    The first four arguments are as per :func:`numpy.fft.rfft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'rfft'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def irfft(a, n=None, axis=-1, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.

    The first four arguments are as per :func:`numpy.fft.irfft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'irfft'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def rfft2(a, s=None, axes=(-2,-1), norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real FFT.

    The first four arguments are as per :func:`numpy.fft.rfft2`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'rfft2'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def irfft2(a, s=None, axes=(-2,-1), norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D real inverse FFT.

    The first four arguments are as per :func:`numpy.fft.irfft2`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'irfft2'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def rfftn(a, s=None, axes=None, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real FFT.

    The first four arguments are as per :func:`numpy.fft.rfftn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'rfftn'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def irfftn(a, s=None, axes=None, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D real inverse FFT.

    The first four arguments are as per :func:`numpy.fft.rfftn`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''
    calling_func = 'irfftn'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return _Xfftn(a, s, axes, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, **_norm_args(norm))


def hfft(a, n=None, axis=-1, norm=None, overwrite_input=False,
         planner_effort=None, threads=None,
         auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT of a signal with hermitian symmetry.
    This yields a real output spectrum. See :func:`numpy.fft.hfft`
    for more information.

    The first four arguments are as per :func:`numpy.fft.hfft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''

    # The hermitian symmetric transform is equivalent to the
    # irfft of the conjugate of the input (do the maths!) without
    # any normalisation of the result (so normalise_idft is set to
    # False).
    a = numpy.conjugate(a)
    if a.size < 2:
        raise ValueError("hfft requires input with length >= 2")

    calling_func = 'irfft'
    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    if _unitary(norm):
        if n is None:
            if not isinstance(a, numpy.ndarray):
                a = numpy.asarray(a)
            n = (a.shape[axis] - 1)*2
        if a.dtype == numpy.float16:
            # FFT will be in float32 so perform normalization in this dtype too
            a = a.astype(numpy.float32)
        a /= numpy.sqrt(n)

    return _Xfftn(a, n, axis, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous,
            calling_func, normalise_idft=False)


def ihfft(a, n=None, axis=-1, norm=None, overwrite_input=False,
        planner_effort=None, threads=None,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT of a real-spectrum, yielding
    a signal with hermitian symmetry. See :func:`numpy.fft.ihfft`
    for more information.

    The first four arguments are as per :func:`numpy.fft.ihfft`;
    the rest of the arguments are documented
    in the :ref:`additional arguments docs<interfaces_additional_args>`.
    '''

    # Result is equivalent to the conjugate of the output of
    # the rfft of a.
    # It is necessary to perform the inverse scaling, as this
    # is not done by rfft.
    if n is None:
        if not isinstance(a, numpy.ndarray):
            a = numpy.asarray(a)
        n = a.shape[axis]

    if _unitary(norm):
        scaling = 1.0/numpy.sqrt(n)
    else:
        scaling = 1.0/n

    planner_effort = _default_effort(planner_effort)
    threads = _default_threads(threads)

    return scaling * rfft(a, n, axis, None, overwrite_input, planner_effort,
            threads, auto_align_input, auto_contiguous).conj()
