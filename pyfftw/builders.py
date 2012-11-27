#!/usr/bin/env python

import pyfftw
import numpy

'''A set of builder functions that return FFTW objects. The interface
to create these objects is mostly the same as `numpy.fft
<http://docs.scipy.org/doc/numpy/reference/routines.fft.html>`_, only
instead of the call returning the result of the FFT, an FFTW object is
returned that performs that FFT operation when it is called.

In the case where ``s`` dictates that the passed-in input array be
copied into a different processing array, the returned FFTW object 
is a child class of ``pyfftw.FFTW``, _FFTWWrapper, which wraps the 
call method in order to correctly perform that copying. Only the call 
method is wrapped: ``update_arrays`` still expects an array with the 
correct size, alignment, dtype etc for the underlying FFTW object.

The precision of the FFT operation is acquired from the input array.
If an array is passed in that is not of float type, or is of an
unknown float type, an attempt is made to convert the array to a
double precision array. This results in a copy being made.

If an array of the incorrect complexity is passed in (e.g. a complex
array is passed to a real transform routine, or vice-versa), then an
attempt is made to convert the array to an array of the correct
complexity. This results in a copy being made.

The following functions are supported. They can be used with the
same calling signature as their respective functions in ``numpy.fft``.

'fft','ifft', 'rfft', 'irfft', 'rfftn',
'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn'

The first caveat is that the dtype of the input array must match the
transform. For example, for ``fft`` and ``ifft``, the dtype must
be complex, for ``rfft`` it must be real, and so on. The other point to
note from this is that the precision of the transform matches the 
precision of the input array. So, if a single precision input array
is passed in, then a single precision transform will be used.

The second caveat is that repeated axes are handled differently: with 
the returned FFTW object, axes that are repeated in the axes argument
are considered only once, as compared to ``numpy.fft`` in which
repeated axes results in the DFT being taken along that axes as many
times as the axis occurs.

Note that unless the ``auto_align_input`` argument to the function
is set to ``True``, the ``FFTW_UNALIGNED`` :ref:`flag<FFTW_flags>` is 
set in the returned FFTW object. This disables some of the FFTW
optimisations that rely on aligned arrays. Also woth noting is that
the ``auto_align_input`` flag only results in a copy when calling the
resultant FFTW object if the input array is not already aligned correctly.

In addition to the arguments that are present with their complementary
functions in ``numpy.fft``, each of these functions also offers the
following additional keyword arguments:
    ``planner_effort``: A string dictating how much effort is spent 
    in planning the FFTW routines. This is passed to the creation
    of the FFTW object as an entry in the flags list. They correspond
    to constants in the underlying `C library 
    <http://www.fftw.org/doc/Planner-Flags.html>`_. 
    The valid strings, in order of their increasing impact on the time 
    to compute  are:
    ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default), ``'FFTW_PATIENT'``
    and ``'FFTW_EXHAUSTIVE'``. 

    ``threads``: The number of threads used to perform the FFT.

    ``auto_align_input``: Correctly byte align the input array for optimal
    usage of vector instructions. This can lead to a substantial speedup.
    Setting this argument to ``True`` makes sure that the input array
    is correctly aligned. It is possible to correctly byte align the array
    prior to calling this function (using, for example,
    :ref:`n_byte_align()<n_byte_align>`). If and only if a realignment is 
    necessary is a new array created. If a new array *is* created, it is 
    up to the calling code to acquire that new input array using 
    :ref:`FFTW.get_input_array()<FFTW_get_input_array>`.

    ``avoid_copy``: By default, these functions will always create a copy 
    (and sometimes more than one) of the passed in input array. This is 
    because the creation of the FFTW object generally destroys the contents 
    of the input array. Setting this argument to ``True`` will try not to 
    create a copy of the input array. This may not be possible if the shape 
    of the FFT input as dictated by ``s`` is necessarily different from the 
    shape of the passed-in array, or the dtypes are incompatible with the
    FFT routine.
'''

__all__ = ['fft','ifft', 'rfft', 'irfft', 'rfftn',
           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']

_valid_efforts = ('FFTW_ESTIMATE', 'FFTW_MEASURE', 
        'FFTW_PATIENT', 'FFTW_EXHAUSTIVE')

# Looking up a dtype in here returns the complex complement of the same
# precision.
_rc_dtype_pairs = {'float32': 'complex64',
        'float64': 'complex128',
        'longdouble': 'clongdouble',
        'complex64': 'float32',
        'complex128': 'float64',
        'clongdouble': 'longdouble'}

_default_dtype = numpy.dtype('float64')


def fft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = False

    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = False
    
    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def fft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):
    
    inverse = False
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):
    
    inverse = True
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def fftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def ifftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = False

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def rfft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True
    
    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def irfft(a, n=None, axis=-1, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    s, axes = _precook_1d_args(a, n, axis)

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def rfft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)

def irfft2(a, s=None, axes=(-2,-1), planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def rfftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = False
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def irfftn(a, s=None, axes=None, planner_effort='FFTW_MEASURE', 
        threads=1, auto_align_input=False, avoid_copy=False):

    inverse = True
    real = True

    return _Xfftn(a, s, axes, planner_effort, threads, 
            auto_align_input, avoid_copy, inverse, real)


def _Xfftn(a, s, axes, planner_effort, threads, auto_align_input, 
        avoid_copy, inverse, real):
    '''Generic transform interface for all the transforms. No
    defaults exist. The transform must be specified exactly.
    '''
    invreal = inverse and real

    if inverse:
        direction = 'FFTW_BACKWARD'
    else:
        direction = 'FFTW_FORWARD'

    if planner_effort not in _valid_efforts:
        raise ValueError('Invalid planner effort: ', planner_effort)

    s, axes = _cook_nd_args(a, s, axes, invreal)
    
    input_shape, output_shape = _compute_array_shapes(
            a, s, axes, inverse, real)

    # Set the align_flag default
    align_flag = 'FFTW_UNALIGNED'

    a_is_complex = numpy.iscomplexobj(a)

    # Make the input dtype correct
    if str(a.dtype) not in _rc_dtype_pairs:
        # We make it the default dtype
        if not real or inverse:
            a = numpy.asarray(a, dtype=_rc_dtype_pairs[str(_default_dtype)])
        else:
            a = numpy.asarray(a, dtype=_default_dtype)
    
    elif not (real and not inverse) and not a_is_complex:
        # We need to make it a complex dtype
        a = numpy.asarray(a, dtype=_rc_dtype_pairs[str(a.dtype)])

    elif (real and not inverse) and a_is_complex:
        # It should be real
        a = numpy.asarray(a, dtype=_rc_dtype_pairs[str(a.dtype)])

    # Make the output dtype correct
    if not real:
        output_dtype = a.dtype
    
    else:
        output_dtype = _rc_dtype_pairs[str(a.dtype)]

    if not avoid_copy:
        a_copy = a.copy()

    output_array = pyfftw.n_byte_align_empty(output_shape, 16, output_dtype)

    if auto_align_input:
        align_flag = 'FFTW_ALIGNED'

    flags = [align_flag, planner_effort]

    if not a.shape == input_shape:
        # This means we need to use an _FFTWWrapper object
        # and so need to create slicers.
        update_input_array_slicer, FFTW_array_slicer = (
                _setup_input_slicers(a.shape, input_shape))

        # Also, the input array will be a different shape to the shape of 
        # `a`, so we need to create a new array.
        input_array = pyfftw.n_byte_align_empty(input_shape, 16, a.dtype)

        FFTW_object = _FFTWWrapper(input_array, output_array, axes, direction,
                flags, threads, input_array_slicer=update_input_array_slicer,
                FFTW_array_slicer=FFTW_array_slicer)

        if not avoid_copy:
            # We copy the data back into the internal FFTW object array
            internal_array = FFTW_object.get_input_array()
            internal_array[:] = 0
            internal_array[FFTW_array_slicer] = (
                    a_copy[update_input_array_slicer])

    else:
        # Otherwise we can use `a` as-is
        if auto_align_input:
            input_array = pyfftw.n_byte_align(a, 16)
        else:
            input_array = a

        FFTW_object = pyfftw.FFTW(input_array, output_array, axes, direction,
                flags, threads)

        # Copy the data back into the (likely) destroyed array
        FFTW_object.get_input_array()[:] = a_copy

    return FFTW_object


class _FFTWWrapper(pyfftw.FFTW):

    def __init__(self, input_array, output_array, axes=[-1], 
            direction='FFTW_FORWARD', flags=['FFTW_MEASURE'], 
            threads=1, *args, **kwargs):

        self.__input_array_slicer = kwargs.pop('input_array_slicer')
        self.__FFTW_array_slicer = kwargs.pop('FFTW_array_slicer')

        super(_FFTWWrapper, self).__init__(input_array, output_array, 
                axes, direction, flags, threads, *args, **kwargs)

    def __call__(self, input_array=None, output_array=None, 
            normalise_idft=True):

        if input_array is not None:
            # Do the update here (which is a copy, so it's alignment
            # safe etc).

            internal_input_array = self.get_input_array()
            input_array = numpy.asanyarray(input_array)

            sliced_internal = internal_input_array[self.__FFTW_array_slicer]
            sliced_input = input_array[self.__input_array_slicer]
            
            if not sliced_internal.shape == sliced_input.shape:
                raise ValueError('Invalid input shape: '
                        'The new input array should be the same shape '
                        'as the input array used to instantiate the '
                        'object.')

            sliced_internal[:] = sliced_input

        return super(_FFTWWrapper, self).__call__(input_array=None,
                output_array=output_array, normalise_idft=normalise_idft)


def _setup_input_slicers(a_shape, input_shape):
    ''' This function returns two slicers that are to be used to
    copy the data from the input array to the FFTW object internal
    array, which can then be passed to _FFTWWrapper.

    These are:
    update_input_array_slicer
    FFTW_array_slicer

    On calls to _FFTWWrapper objects, the input array is copied in
    as:
    FFTW_array[FFTW_array_slicer] = input_array[update_input_array_slicer]
    '''

    # default the slicers to include everything
    update_input_array_slicer = (
            [slice(None)]*len(a_shape))
    FFTW_array_slicer = [slice(None)]*len(a_shape)

    # iterate over each dimension and modify the slicer and FFTW dimension
    for axis in xrange(len(a_shape)):

        if a_shape[axis] > input_shape[axis]:
            update_input_array_slicer[axis] = (
                    slice(0, input_shape[axis]))

        elif a_shape[axis] < input_shape[axis]:
            FFTW_array_slicer[axis] = (
                    slice(0, a_shape[axis]))

        else:
            # If neither of these, we use the whole dimension.
            pass

    return update_input_array_slicer, FFTW_array_slicer

def _compute_array_shapes(a, s, axes, inverse, real):
    '''Given a passed in array a, and the rest of the arguments
    (that have been fleshed out with _cook_nd_args), compute
    the shape the input and output arrays needs to be in order 
    to satisfy all the requirements for the transform. The input
    shape *may* be different to the shape of a.

    returns:
    (input_shape, output_shape)
    '''
    # Start with the shape of a
    orig_domain_shape = list(a.shape)
    fft_domain_shape = list(a.shape)
    
    try:
        for n, axis in enumerate(axes):
            orig_domain_shape[axis] = s[n]
            fft_domain_shape[axis] = s[n]

        if real:
            fft_domain_shape[axes[-1]] = s[-1]//2 + 1

    except IndexError:
        raise IndexError('Invalid axes: '
                'At least one of the passed axes is invalid.')

    if inverse:
        input_shape = fft_domain_shape
        output_shape = orig_domain_shape
    else:
        input_shape = orig_domain_shape
        output_shape = fft_domain_shape

    return tuple(input_shape), tuple(output_shape)

def _precook_1d_args(a, n, axis):
    '''Turn *(n, axis) into (s, axes)
    '''
    if n is not None:
        s = [n]
    else:
        s = None

    # Force an error with an invalid axis
    a.shape[axis]

    return s, (axis,)

def _cook_nd_args(a, s=None, axes=None, invreal=False):
    '''Similar to _cook_nd_args in numpy's fftpack
    '''

    if axes is None:
        if s is None:
            len_s = len(a.shape)
        else:
            len_s = len(s)

        axes = range(-len_s, 0)

    if s is None:
        s = list(numpy.take(a.shape, axes))

        if invreal:
            s[-1] = (a.shape[axes[-1]] - 1) * 2


    if len(s) != len(axes):
        raise ValueError('Shape error: '
                'Shape and axes have different lengths.')

    if len(s) > len(a.shape):
        raise ValueError('Shape error: '
                'The length of s or axes cannot exceed the dimensionality '
                'of the input array, a.')

    return tuple(s), tuple(axes)

