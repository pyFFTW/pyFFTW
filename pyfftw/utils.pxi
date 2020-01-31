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

from bisect import bisect_left
cimport numpy as np
from . cimport cpu
from libc.stdint cimport intptr_t
import warnings


cdef int _simd_alignment = cpu.simd_alignment()

#: The optimum SIMD alignment in bytes, found by inspecting the CPU.
simd_alignment = _simd_alignment

#: A tuple of simd alignments that make sense for this cpu
if _simd_alignment == 16:
    _valid_simd_alignments = (16,)

elif _simd_alignment == 32:
    _valid_simd_alignments = (16, 32)

else:
    _valid_simd_alignments = ()

cpdef n_byte_align_empty(shape, n, dtype='float64', order='C'):
    '''n_byte_align_empty(shape, n, dtype='float64', order='C')
    **This function is deprecated:** ``empty_aligned`` **should be used
    instead.**

    Function that returns an empty numpy array that is n-byte aligned.

    The alignment is given by the first optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    warnings.warn('This function is deprecated in favour of'
    '``empty_aligned``.', DeprecationWarning)
    return empty_aligned(shape, dtype=dtype, order=order, n=n)


cpdef n_byte_align(array, n, dtype=None):
    '''n_byte_align(array, n, dtype=None)

    **This function is deprecated:** ``byte_align`` **should be used instead.**

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is an optional parameter. If ``n`` is not provided
    then this function will inspect the CPU to determine alignment. If the
    array is aligned then it is returned without further ado.  If it is not
    aligned then a new array is created and the data copied in, but aligned
    on the n-byte boundary.

    ``dtype`` is an optional argument that forces the resultant array to be
    of that dtype.
    '''
    warnings.warn('This function is deprecated in favour of'
    '``byte_align``.', DeprecationWarning)
    return byte_align(array, n=n, dtype=dtype)


cpdef byte_align(array, n=None, dtype=None):
    '''byte_align(array, n=None, dtype=None)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is an optional parameter. If ``n`` is not provided
    then this function will inspect the CPU to determine alignment. If the
    array is aligned then it is returned without further ado.  If it is not
    aligned then a new array is created and the data copied in, but aligned
    on the n-byte boundary.

    ``dtype`` is an optional argument that forces the resultant array to be
    of that dtype.
    '''

    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: byte_align requires a subclass '
                'of ndarray')

    if n is None:
        n = _simd_alignment

    if dtype is not None:
        if not array.dtype == dtype:
            update_dtype = True

    else:
        dtype = array.dtype
        update_dtype = False

    # See if we're already n byte aligned. If so, do nothing.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    if offset is not 0 or update_dtype:

        _array_aligned = empty_aligned(array.shape, dtype, n=n)

        _array_aligned[:] = array

        array = _array_aligned.view(type=array.__class__)

    return array


cpdef is_byte_aligned(array, n=None):
    ''' is_n_byte_aligned(array, n=None)

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is an optional parameter, returning ``True`` if it is,
    and ``False`` if it is not. If ``n`` is not provided then this function will
    inspect the CPU to determine alignment.
    '''
    if not isinstance(array, np.ndarray):
        raise TypeError('Invalid array: is_n_byte_aligned requires a subclass '
                'of ndarray')

    if n is None:
        n = _simd_alignment

    # See if we're n byte aligned.
    offset = <intptr_t>np.PyArray_DATA(array) %n

    return not bool(offset)


cpdef is_n_byte_aligned(array, n):
    ''' is_n_byte_aligned(array, n)
    **This function is deprecated:** ``is_byte_aligned`` **should be used
    instead.**

    Function that takes a numpy array and checks it is aligned on an n-byte
    boundary, where ``n`` is a passed parameter, returning ``True`` if it is,
    and ``False`` if it is not.
    '''
    return is_byte_aligned(array, n=n)


cpdef empty_aligned(shape, dtype='float64', order='C', n=None):
    '''empty_aligned(shape, dtype='float64', order='C', n=None)

    Function that returns an empty numpy array that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.empty`.
    '''
    cdef long long array_length

    if n is None:
        n = _simd_alignment

    itemsize = np.dtype(dtype).itemsize

    # Apparently there is an issue with numpy.prod wrapping around on 32-bits
    # on Windows 64-bit. This shouldn't happen, but the following code
    # alleviates the problem.
    if not isinstance(shape, (int, np.integer)):
        array_length = 1
        for each_dimension in shape:
            array_length *= each_dimension

    else:
        array_length = shape

    # Allocate a new array that will contain the aligned data
    _array_aligned = np.empty(array_length*itemsize+n, dtype='int8')

    # We now need to know how to offset _array_aligned
    # so it is correctly aligned
    _array_aligned_offset = (n-<intptr_t>np.PyArray_DATA(_array_aligned))%n

    array = np.frombuffer(
            _array_aligned[_array_aligned_offset:_array_aligned_offset-n].data,
            dtype=dtype).reshape(shape, order=order)

    return array


cpdef zeros_aligned(shape, dtype='float64', order='C', n=None):
    '''zeros_aligned(shape, dtype='float64', order='C', n=None)

    Function that returns a numpy array of zeros that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.zeros`.
    '''
    array = empty_aligned(shape, dtype=dtype, order=order, n=n)
    array.fill(0)
    return array


cpdef ones_aligned(shape, dtype='float64', order='C', n=None):
    '''ones_aligned(shape, dtype='float64', order='C', n=None)

    Function that returns a numpy array of ones that is n-byte aligned,
    where ``n`` is determined by inspecting the CPU if it is not
    provided.

    The alignment is given by the final optional argument, ``n``. If
    ``n`` is not provided then this function will inspect the CPU to
    determine alignment. The rest of the arguments are as per
    :func:`numpy.ones`.
    '''
    array = empty_aligned(shape, dtype=dtype, order=order, n=n)
    array.fill(1)
    return array


cpdef next_fast_len(target):
    '''next_fast_len(target)

    Find the next fast transform length for FFTW.

    FFTW has efficient functions for transforms of length
    2**a * 3**b * 5**c * 7**d * 11**e * 13**f, where e + f is either 0 or 1.

    Parameters
    ----------
    target : int
        Length to start searching from.  Must be a positive integer.

    Returns
    -------
    out : int
        The first fast length greater than or equal to `target`.

    Examples
    --------
    On a particular machine, an FFT of prime length takes 2.1 ms:

    >>> from pyfftw.interfaces import scipy_fftpack
    >>> min_len = 10007  # prime length is worst case for speed
    >>> a = numpy.random.randn(min_len)
    >>> b = scipy_fftpack.fft(a)

    Zero-padding to the next fast length reduces computation time to
    406 us, a speedup of ~5 times:

    >>> next_fast_len(min_len)
    10080
    >>> b = scipy_fftpack.fft(a, 10080)

    Rounding up to the next power of 2 is not optimal, taking 598 us to
    compute, 1.5 times as long as the size selected by next_fast_len.

    >>> b = fftpack.fft(a, 16384)

    Similar speedups will occur for pre-planned FFTs as generated via
    pyfftw.builders.

    '''
    lpre = (18,    20,    21,    22,    24,    25,    26,    27,    28,    30,
            32,    33,    35,    36,    39,    40,    42,    44,    45,    48,
            49,    50,    52,    54,    55,    56,    60,    63,    64,
            65,    66,    70,    72,    75,    77,    78,    80,    81,
            84,    88,    90,    91,    96,    98,    99,    100,   104,
            105,   108,   110,   112,   117,   120,   125,   126,   128,
            130,   132,   135,   140,   144,   147,   150,   154,   156,
            160,   162,   165,   168,   175,   176,   180,   182,   189,
            192,   195,   196,   198,   200,   208,   210,   216,   220,
            224,   225,   231,   234,   240,   243,   245,   250,   252,
            256,   260,   264,   270,   273,   275,   280,   288,   294,
            297,   300,   308,   312,   315,   320,   324,   325,   330,
            336,   343,   350,   351,   352,   360,   364,   375,   378,
            384,   385,   390,   392,   396,   400,   405,   416,   420,
            432,   440,   441,   448,   450,   455,   462,   468,   480,
            486,   490,   495,   500,   504,   512,   520,   525,   528,
            539,   540,   546,   550,   560,   567,   576,   585,   588,
            594,   600,   616,   624,   625,   630,   637,   640,   648,
            650,   660,   672,   675,   686,   693,   700,   702,   704,
            720,   728,   729,   735,   750,   756,   768,   770,   780,
            784,   792,   800,   810,   819,   825,   832,   840,   864,
            875,   880,   882,   891,   896,   900,   910,   924,   936,
            945,   960,   972,   975,   980,   990,   1000,  1008,  1024,
            1029,  1040,  1050,  1053,  1056,  1078,  1080,  1092,  1100,
            1120,  1125,  1134,  1152,  1155,  1170,  1176,  1188,  1200,
            1215,  1225,  1232,  1248,  1250,  1260,  1274,  1280,  1296,
            1300,  1320,  1323,  1344,  1350,  1365,  1372,  1375,  1386,
            1400,  1404,  1408,  1440,  1456,  1458,  1470,  1485,  1500,
            1512,  1536,  1540,  1560,  1568,  1575,  1584,  1600,  1617,
            1620,  1625,  1638,  1650,  1664,  1680,  1701,  1715,  1728,
            1750,  1755,  1760,  1764,  1782,  1792,  1800,  1820,  1848,
            1872,  1875,  1890,  1911,  1920,  1925,  1944,  1950,  1960,
            1980,  2000,  2016,  2025,  2048,  2058,  2079,  2080,  2100,
            2106,  2112,  2156,  2160,  2184,  2187,  2200,  2205,  2240,
            2250,  2268,  2275,  2304,  2310,  2340,  2352,  2376,  2400,
            2401,  2430,  2450,  2457,  2464,  2475,  2496,  2500,  2520,
            2548,  2560,  2592,  2600,  2625,  2640,  2646,  2673,  2688,
            2695,  2700,  2730,  2744,  2750,  2772,  2800,  2808,  2816,
            2835,  2880,  2912,  2916,  2925,  2940,  2970,  3000,  3024,
            3072,  3080,  3087,  3120,  3125,  3136,  3150,  3159,  3168,
            3185,  3200,  3234,  3240,  3250,  3276,  3300,  3328,  3360,
            3375,  3402,  3430,  3456,  3465,  3500,  3510,  3520,  3528,
            3564,  3584,  3600,  3640,  3645,  3675,  3696,  3744,  3750,
            3773,  3780,  3822,  3840,  3850,  3888,  3900,  3920,  3960,
            3969,  4000,  4032,  4050,  4095,  4096,  4116,  4125,  4158,
            4160,  4200,  4212,  4224,  4312,  4320,  4368,  4374,  4375,
            4400,  4410,  4455,  4459,  4480,  4500,  4536,  4550,  4608,
            4620,  4680,  4704,  4725,  4752,  4800,  4802,  4851,  4860,
            4875,  4900,  4914,  4928,  4950,  4992,  5000,  5040,  5096,
            5103,  5120,  5145,  5184,  5200,  5250,  5265,  5280,  5292,
            5346,  5376,  5390,  5400,  5460,  5488,  5500,  5544,  5600,
            5616,  5625,  5632,  5670,  5733,  5760,  5775,  5824,  5832,
            5850,  5880,  5940,  6000,  6048,  6075,  6125,  6144,  6160,
            6174,  6237,  6240,  6250,  6272,  6300,  6318,  6336,  6370,
            6400,  6468,  6480,  6500,  6552,  6561,  6600,  6615,  6656,
            6720,  6750,  6804,  6825,  6860,  6875,  6912,  6930,  7000,
            7020,  7040,  7056,  7128,  7168,  7200,  7203,  7280,  7290,
            7350,  7371,  7392,  7425,  7488,  7500,  7546,  7560,  7644,
            7680,  7700,  7776,  7800,  7840,  7875,  7920,  7938,  8000,
            8019,  8064,  8085,  8100,  8125,  8190,  8192,  8232,  8250,
            8316,  8320,  8400,  8424,  8448,  8505,  8575,  8624,  8640,
            8736,  8748,  8750,  8775,  8800,  8820,  8910,  8918,  8960,
            9000,  9072,  9100,  9216,  9240,  9261,  9360,  9375,  9408,
            9450,  9477,  9504,  9555,  9600,  9604,  9625,  9702,  9720,
            9750,  9800,  9828,  9856,  9900,  9984,  10000)

    if target <= 16:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    # Get result quickly for small sizes, since FFT itself is similarly fast.
    if target <= lpre[-1]:
        return lpre[bisect_left(lpre, target)]

    # check if 13 or 11 is a factor first
    if target % 13 == 0:
        p11_13 = 13
        e_f_cases = [13, ]  # e=0, f=1
    elif target % 11 == 0:
        p11_13 = 11
        e_f_cases = [11, ]  # e=1, f=0
    else:
        p11_13 = 1
        # try all three cases where e + f <= 1 (see docstring)
        e_f_cases = [13, 11, 1]

    best_match = float('inf')  # Anything found will be smaller

    # outer loop is for the cases where e + f <= 1 (see docstring)
    for p11_13 in e_f_cases:
        match = float('inf')
        # allow any integer powers of 2, 3, 5 or 7
        p7_11_13 = p11_13
        while p7_11_13 < target:
            p5_7_11_13 = p7_11_13
            while p5_7_11_13 < target:
                p3_5_7_11_13 = p5_7_11_13
                while p3_5_7_11_13 < target:
                    # Ceiling integer division, avoiding conversion to
                    # float.
                    # (quotient = ceil(target / p35))
                    quotient = -(-target // p3_5_7_11_13)

                    # Quickly find next power of 2 >= quotient
                    p2 = 2**((quotient - 1).bit_length())

                    N = p2 * p3_5_7_11_13
                    if N == target:
                        return N
                    elif N < match:
                        match = N
                    p3_5_7_11_13 *= 3
                    if p3_5_7_11_13 == target:
                        return p3_5_7_11_13
                if p3_5_7_11_13 < match:
                    match = p3_5_7_11_13
                p5_7_11_13 *= 5
                if p5_7_11_13 == target:
                    return p5_7_11_13
            if p5_7_11_13 < match:
                match = p5_7_11_13
            p7_11_13 *= 7
            if p7_11_13 == target:
                return p7_11_13
        if p7_11_13 < match:
            match = p7_11_13
        if match < best_match:
            best_match = match
    return best_match
