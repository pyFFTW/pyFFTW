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
:mod:`dask.fft` module. This module *provides* the entire documented
namespace of :mod:`dask.fft`, but those functions that are not included
here are imported directly from :mod:`dask.fft`.


It is notable that unlike :mod:`numpy.fftpack`, which :mod:`dask.fft`
wraps, these functions will generally return an output array with the
same precision as the input array, and the transform that is chosen is
chosen based on the precision of the input array. That is, if the input
array is 32-bit floating point, then the transform will be 32-bit floating
point and so will the returned array. Half precision input will be
converted to single precision.  Otherwise, if any type conversion is
required, the default will be double precision.

The exceptions raised by each of these functions are mostly as per their
equivalents in :mod:`dask.fft`, though there are some corner cases in
which this may not be true.
'''

from . import numpy_fft as _numpy_fft
from dask.array.fft import (
    fft_wrap,
    fftfreq,
    rfftfreq,
    fftshift,
    ifftshift,
)

fft = fft_wrap(_numpy_fft.fft)
fft2 = fft_wrap(_numpy_fft.fft2)
fftn = fft_wrap(_numpy_fft.fftn)
ifft = fft_wrap(_numpy_fft.ifft)
ifft2 = fft_wrap(_numpy_fft.ifft2)
ifftn = fft_wrap(_numpy_fft.ifftn)
rfft = fft_wrap(_numpy_fft.rfft)
rfft2 = fft_wrap(_numpy_fft.rfft2)
rfftn = fft_wrap(_numpy_fft.rfftn)
irfft = fft_wrap(_numpy_fft.irfft)
irfft2 = fft_wrap(_numpy_fft.irfft2)
irfftn = fft_wrap(_numpy_fft.irfftn)
hfft = fft_wrap(_numpy_fft.hfft)
ihfft = fft_wrap(_numpy_fft.ihfft)
