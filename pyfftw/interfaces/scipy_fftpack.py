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
This module implements those functions that replace aspects of the
:mod:`scipy.fftpack` module. This module *provides* the entire documented
namespace of :mod:`scipy.fftpack`, but those functions that are not included
here are imported directly from :mod:`scipy.fftpack`.
'''

from . import numpy_fft

# Complete the namespace (these are not actually used in this module)
from scipy.fftpack import (dct, idct, diff, tilbert, itilbert, 
        hilbert, ihilbert, cs_diff, sc_diff, ss_diff, cc_diff, 
        shift, fftshift, ifftshift, fftfreq, rfftfreq, 
        convolve, _fftpack)

__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft', 'fft2','ifft2', 
        'diff', 'tilbert','itilbert','hilbert','ihilbert', 'sc_diff',
        'cs_diff','cc_diff','ss_diff', 'shift', 'rfftfreq']

def fft(x, n=None, axis=-1, overwrite_x=False, 
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.fft`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.fft(x, n, axis, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)

def ifft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D inverse FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.ifft`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.ifft(x, n, axis, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.fft2`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.fft2(x, shape, axes, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 2D inverse FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.ifft2`; 
    the rest of the arguments are documented in the
    :ref:`additional argument docs <interfaces_additional_args>`.
    '''

    return numpy_fft.ifft2(x, shape, axes, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)


def fftn(x, shape=None, axes=None, overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.fftn`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.fftn(x, shape, axes, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)


def ifftn(x, shape=None, axes=None, overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform an n-D inverse FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.ifftn`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.ifftn(x, shape, axes, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)


def rfft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.rfft`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.rfft(x, n, axis, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)

def irfft(x, n=None, axis=-1, overwrite_x=False,
        planner_effort='FFTW_MEASURE', threads=1,
        auto_align_input=True, auto_contiguous=True):
    '''Perform a 1D real inverse FFT.
    
    The first three arguments are as per :func:`scipy.fftpack.irfft`; 
    the rest of the arguments are documented 
    in the :ref:`additional argument docs<interfaces_additional_args>`.
    '''

    return numpy_fft.irfft(x, n, axis, overwrite_x, planner_effort,
            threads, auto_align_input, auto_contiguous)

