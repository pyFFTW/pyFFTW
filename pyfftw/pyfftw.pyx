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

import numpy as np
cimport numpy as np
from libc.stdlib cimport calloc, malloc, free
from libc.stdint cimport intptr_t, int64_t
from libc cimport limits

include 'utils.pxi'

cdef extern from *:
    int Py_AtExit(void (*callback)()) 

cdef object directions
directions = {'FFTW_FORWARD': FFTW_FORWARD,
        'FFTW_BACKWARD': FFTW_BACKWARD}

cdef object flag_dict
flag_dict = {'FFTW_MEASURE': FFTW_MEASURE,
        'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
        'FFTW_PATIENT': FFTW_PATIENT,
        'FFTW_ESTIMATE': FFTW_ESTIMATE,
        'FFTW_UNALIGNED': FFTW_UNALIGNED,
        'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT}

_flag_dict = flag_dict.copy()

# Function wrappers
# =================
# All of these have the same signature as the fftw_generic functions
# defined in the .pxd file. The arguments and return values are
# cast as required in order to call the actual fftw functions.
# 
# The wrapper function names are simply the fftw names prefixed
# with a single underscore.

#     Planners
#     ========
#
# Complex double precision
cdef void* _fftw_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftw_plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims,
            <double complex *>_in, <double complex *>_out,
            sign, flags)

# Complex single precision
cdef void* _fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims,
            <float complex *>_in, <float complex *>_out,
            sign, flags)

# Complex long double precision
cdef void* _fftwl_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims,
            <long double complex *>_in, <long double complex *>_out,
            sign, flags)

# real to complex double precision
cdef void* _fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftw_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <double *>_in, <double complex *>_out,
            flags)

# real to complex single precision
cdef void* _fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <float *>_in, <float complex *>_out,
            flags)

# real to complex long double precision
cdef void* _fftwl_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <long double *>_in, <long double complex *>_out,
            flags)

# complex to real double precision
cdef void* _fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftw_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <double complex *>_in, <double *>_out,
            flags)

# complex to real single precision
cdef void* _fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <float complex *>_in, <float *>_out,
            flags)

# complex to real long double precision
cdef void* _fftwl_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <long double complex *>_in, <long double *>_out,
            flags)

#    Executors
#    =========
#
# Complex double precision
cdef void _fftw_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft(<fftw_plan>_plan, 
            <double complex *>_in, <double complex *>_out)

# Complex single precision
cdef void _fftwf_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft(<fftwf_plan>_plan, 
            <float complex *>_in, <float complex *>_out)

# Complex long double precision
cdef void _fftwl_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft(<fftwl_plan>_plan, 
            <long double complex *>_in, <long double complex *>_out)

# real to complex double precision
cdef void _fftw_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft_r2c(<fftw_plan>_plan, 
            <double *>_in, <double complex *>_out)

# real to complex single precision
cdef void _fftwf_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft_r2c(<fftwf_plan>_plan, 
            <float *>_in, <float complex *>_out)

# real to complex long double precision
cdef void _fftwl_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft_r2c(<fftwl_plan>_plan, 
            <long double *>_in, <long double complex *>_out)

# complex to real double precision
cdef void _fftw_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft_c2r(<fftw_plan>_plan, 
            <double complex *>_in, <double *>_out)

# complex to real single precision
cdef void _fftwf_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft_c2r(<fftwf_plan>_plan, 
            <float complex *>_in, <float *>_out)

# complex to real long double precision
cdef void _fftwl_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft_c2r(<fftwl_plan>_plan, 
            <long double complex *>_in, <long double *>_out)

#    Destroyers
#    ==========
#
# Double precision
cdef void _fftw_destroy_plan(void *_plan):

    fftw_destroy_plan(<fftw_plan>_plan)

# Single precision
cdef void _fftwf_destroy_plan(void *_plan):

    fftwf_destroy_plan(<fftwf_plan>_plan)

# Long double precision
cdef void _fftwl_destroy_plan(void *_plan):

    fftwl_destroy_plan(<fftwl_plan>_plan)


# Function lookup tables
# ======================

# Planner table (of side the number of planners).
cdef fftw_generic_plan_guru planners[9]

cdef fftw_generic_plan_guru * _build_planner_list():

    planners[0] = <fftw_generic_plan_guru>&_fftw_plan_guru_dft
    planners[1] = <fftw_generic_plan_guru>&_fftwf_plan_guru_dft
    planners[2] = <fftw_generic_plan_guru>&_fftwl_plan_guru_dft
    planners[3] = <fftw_generic_plan_guru>&_fftw_plan_guru_dft_r2c
    planners[4] = <fftw_generic_plan_guru>&_fftwf_plan_guru_dft_r2c
    planners[5] = <fftw_generic_plan_guru>&_fftwl_plan_guru_dft_r2c
    planners[6] = <fftw_generic_plan_guru>&_fftw_plan_guru_dft_c2r
    planners[7] = <fftw_generic_plan_guru>&_fftwf_plan_guru_dft_c2r
    planners[8] = <fftw_generic_plan_guru>&_fftwl_plan_guru_dft_c2r

# Executor table (of size the number of executors)
cdef fftw_generic_execute executors[9]

cdef fftw_generic_execute * _build_executor_list():

    executors[0] = <fftw_generic_execute>&_fftw_execute_dft
    executors[1] = <fftw_generic_execute>&_fftwf_execute_dft
    executors[2] = <fftw_generic_execute>&_fftwl_execute_dft
    executors[3] = <fftw_generic_execute>&_fftw_execute_dft_r2c
    executors[4] = <fftw_generic_execute>&_fftwf_execute_dft_r2c
    executors[5] = <fftw_generic_execute>&_fftwl_execute_dft_r2c
    executors[6] = <fftw_generic_execute>&_fftw_execute_dft_c2r
    executors[7] = <fftw_generic_execute>&_fftwf_execute_dft_c2r
    executors[8] = <fftw_generic_execute>&_fftwl_execute_dft_c2r

# Destroyer table (of size the number of destroyers)
cdef fftw_generic_destroy_plan destroyers[3]

cdef fftw_generic_destroy_plan * _build_destroyer_list():

    destroyers[0] = <fftw_generic_destroy_plan>&_fftw_destroy_plan
    destroyers[1] = <fftw_generic_destroy_plan>&_fftwf_destroy_plan
    destroyers[2] = <fftw_generic_destroy_plan>&_fftwl_destroy_plan


# nthreads plan setters table
cdef fftw_generic_plan_with_nthreads nthreads_plan_setters[3]

cdef fftw_generic_plan_with_nthreads * _build_nthreads_plan_setters_list():
    nthreads_plan_setters[0] = (
            <fftw_generic_plan_with_nthreads>&fftw_plan_with_nthreads)
    nthreads_plan_setters[1] = (
            <fftw_generic_plan_with_nthreads>&fftwf_plan_with_nthreads)
    nthreads_plan_setters[2] = (
            <fftw_generic_plan_with_nthreads>&fftwl_plan_with_nthreads)

# Set planner timelimits
cdef fftw_generic_set_timelimit set_timelimit_funcs[3]

cdef fftw_generic_set_timelimit * _build_set_timelimit_funcs_list():
    set_timelimit_funcs[0] = (
            <fftw_generic_set_timelimit>&fftw_set_timelimit)
    set_timelimit_funcs[1] = (
            <fftw_generic_set_timelimit>&fftwf_set_timelimit)
    set_timelimit_funcs[2] = (
            <fftw_generic_set_timelimit>&fftwl_set_timelimit)


# Data validators table
cdef validator validators[2]

cdef validator * _build_validators_list():
    validators[0] = &_validate_r2c_arrays
    validators[1] = &_validate_c2r_arrays

# Validator functions
# ===================
cdef bint _validate_r2c_arrays(np.ndarray input_array, 
        np.ndarray output_array, int64_t *axes, int64_t *not_axes, 
        int64_t axes_length):
    ''' Validates the input and output array to check for
    a valid real to complex transform.
    '''
    # We firstly need to confirm that the dimenions of the arrays
    # are the same
    if not (input_array.ndim == output_array.ndim):
        return False

    in_shape = input_array.shape
    out_shape = output_array.shape

    for n in range(axes_length - 1):
        if not out_shape[axes[n]] == in_shape[axes[n]]:
            return False
    
    # The critical axis is the last of those over which the 
    # FFT is taken.
    if not (out_shape[axes[axes_length-1]]
            == in_shape[axes[axes_length-1]]//2 + 1):
        return False

    for n in range(input_array.ndim - axes_length):
        if not out_shape[not_axes[n]] == in_shape[not_axes[n]]:
            return False

    return True


cdef bint _validate_c2r_arrays(np.ndarray input_array, 
        np.ndarray output_array, int64_t *axes, int64_t *not_axes, 
        int64_t axes_length):
    ''' Validates the input and output array to check for
    a valid complex to real transform.
    '''

    # We firstly need to confirm that the dimenions of the arrays
    # are the same
    if not (input_array.ndim == output_array.ndim):
        return False

    in_shape = input_array.shape
    out_shape = output_array.shape

    for n in range(axes_length - 1):
        if not in_shape[axes[n]] == out_shape[axes[n]]:
            return False
    
    # The critical axis is the last of those over which the 
    # FFT is taken.
    if not (in_shape[axes[axes_length-1]] 
            == out_shape[axes[axes_length-1]]//2 + 1):
        return False

    for n in range(input_array.ndim - axes_length):
        if not in_shape[not_axes[n]] == out_shape[not_axes[n]]:
            return False

    return True


# Shape lookup functions
# ======================
def _lookup_shape_r2c_arrays(input_array, output_array):
    return input_array.shape

def _lookup_shape_c2r_arrays(input_array, output_array):
    return output_array.shape

# fftw_schemes is a dictionary with a mapping from a keys,
# which are a tuple of the string representation of numpy
# dtypes to a scheme name.
#
# scheme_functions is a dictionary of functions, either 
# an index to the array of functions in the case of 
# 'planner', 'executor' and 'generic_precision' or a callable
# in the case of 'validator' (generic_precision is a catchall for
# functions that only change based on the precision changing - 
# i.e the prefix fftw, fftwl and fftwf is the only bit that changes).
#
# The array indices refer to the relevant functions for each scheme,
# the tables to which are defined above.
#
# The 'validator' function is a callable for validating the arrays
# that has the following signature:
# bool callable(ndarray in_array, ndarray out_array, axes, not_axes)
# and checks that the arrays are a valid pair. If it is set to None,
# then the default check is applied, which confirms that the arrays
# have the same shape.
# 
# The 'fft_shape_lookup' function is a callable for returning the
# FFT shape - that is, an array that describes the length of the 
# fft along each axis. It has the following signature:
# fft_shape = fft_shape_lookup(in_array, out_array) 
# (note that this does not correspond to the lengths of the FFT that is 
# actually taken, it's the lengths of the FFT that *could* be taken
# along each axis. It's necessary because the real FFT has a length
# that is different to the length of the input array).

cdef object fftw_schemes
fftw_schemes = {
        (np.dtype('complex128'), np.dtype('complex128')): ('c2c', '64'),
        (np.dtype('complex64'), np.dtype('complex64')): ('c2c', '32'),
        (np.dtype('clongdouble'), np.dtype('clongdouble')): ('c2c', 'ld'),
        (np.dtype('float64'), np.dtype('complex128')): ('r2c', '64'),
        (np.dtype('float32'), np.dtype('complex64')): ('r2c', '32'),
        (np.dtype('longdouble'), np.dtype('clongdouble')): ('r2c', 'ld'),
        (np.dtype('complex128'), np.dtype('float64')): ('c2r', '64'),
        (np.dtype('complex64'), np.dtype('float32')): ('c2r', '32'),
        (np.dtype('clongdouble'), np.dtype('longdouble')): ('c2r', 'ld')}

cdef object scheme_directions
scheme_directions = {
        ('c2c', '64'): ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        ('c2c', '32'): ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        ('c2c', 'ld'): ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        ('r2c', '64'): ['FFTW_FORWARD'],
        ('r2c', '32'): ['FFTW_FORWARD'],
        ('r2c', 'ld'): ['FFTW_FORWARD'],
        ('c2r', '64'): ['FFTW_BACKWARD'],
        ('c2r', '32'): ['FFTW_BACKWARD'],
        ('c2r', 'ld'): ['FFTW_BACKWARD']}

# In the following, -1 denotes using the default. A segfault has been
# reported on some systems when this is set to None. It seems 
# sufficiently trivial to use -1 in place of None, especially given 
# that scheme_functions is an internal cdef object.
cdef object scheme_functions
scheme_functions = {
    ('c2c', '64'): {'planner': 0, 'executor':0, 'generic_precision':0,
        'validator': -1, 'fft_shape_lookup': -1},
    ('c2c', '32'): {'planner':1, 'executor':1, 'generic_precision':1,
        'validator': -1, 'fft_shape_lookup': -1},
    ('c2c', 'ld'): {'planner':2, 'executor':2, 'generic_precision':2,
        'validator': -1, 'fft_shape_lookup': -1},
    ('r2c', '64'): {'planner':3, 'executor':3, 'generic_precision':0,
        'validator': 0, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    ('r2c', '32'): {'planner':4, 'executor':4, 'generic_precision':1,
        'validator': 0, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    ('r2c', 'ld'): {'planner':5, 'executor':5, 'generic_precision':2,
        'validator': 0, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    ('c2r', '64'): {'planner':6, 'executor':6, 'generic_precision':0, 
        'validator': 1, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays},
    ('c2r', '32'): {'planner':7, 'executor':7, 'generic_precision':1, 
        'validator': 1, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays},
    ('c2r', 'ld'): {'planner':8, 'executor':8, 'generic_precision':2,
        'validator': 1, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays}}

# Initialize the module

# Define the functions        
_build_planner_list()
_build_destroyer_list()
_build_executor_list()
_build_nthreads_plan_setters_list()
_build_validators_list()
_build_set_timelimit_funcs_list()

fftw_init_threads()
fftwf_init_threads()
fftwl_init_threads()

# Set the cleanup routine
cdef void _cleanup():
    fftw_cleanup()
    fftwf_cleanup()
    fftwl_cleanup()
    fftw_cleanup_threads()
    fftwf_cleanup_threads()
    fftwl_cleanup_threads()

Py_AtExit(_cleanup)

# Helper functions
cdef void make_axes_unique(int64_t *axes, int64_t axes_length, 
        int64_t **unique_axes, int64_t **not_axes, int64_t dimensions, 
        int64_t *unique_axes_length):
    ''' Takes an array of axes and makes that array unique, returning
    the unique array in unique_axes. It also creates and fills another 
    array, not_axes, with those axes that are not included in unique_axes.

    unique_axes_length is updated with the length of unique_axes.

    dimensions is the number of dimensions to which the axes array
    might refer.

    It is the responsibility of the caller to free unique_axes and not_axes.
    '''

    cdef int64_t unique_axes_count = 0
    cdef int64_t holding_offset = 0

    cdef int64_t *axes_holding = (
            <int64_t *>calloc(dimensions, sizeof(int64_t)))
    cdef int64_t *axes_holding_offset = (
            <int64_t *>calloc(dimensions, sizeof(int64_t)))

    for n in range(dimensions):
        axes_holding[n] = -1

    # Iterate over all the axes and store each index if it hasn't already
    # been stored (this keeps one and only one and the first index to axes
    # i.e. storing the unique set of entries).
    #
    # axes_holding_offset holds the shift due to repeated axes
    for n in range(axes_length):
        if axes_holding[axes[n]] == -1:
            axes_holding[axes[n]] = n
            axes_holding_offset[axes[n]] = holding_offset
            unique_axes_count += 1
        else:
            holding_offset += 1

    unique_axes[0] = <int64_t *>malloc(
            unique_axes_count * sizeof(int64_t))

    not_axes[0] = <int64_t *>malloc(
            (dimensions - unique_axes_count) * sizeof(int64_t))

    # Now we need to write back the unique axes to a tmp axes
    cdef int64_t not_axes_count = 0

    for n in range(dimensions):
        if axes_holding[n] != -1:
            unique_axes[0][axes_holding[n] - axes_holding_offset[n]] = (
                    axes[axes_holding[n]])

        else:
            not_axes[0][not_axes_count] = n
            not_axes_count += 1

    free(axes_holding)
    free(axes_holding_offset)

    unique_axes_length[0] = unique_axes_count

    return

# The External Interface
# ======================
#
cdef class FFTW:
    '''
    FFTW is a class for computing the complex N-Dimensional DFT or
    inverse DFT of an array using the FFTW library. The interface is 
    designed to be somewhat pythonic, with the correct transform being 
    inferred from the dtypes of the passed arrays.

    On instantiation, the dtypes and relative shapes of the input array and
    output arrays are compared to the set of valid (and implemented)
    :ref:`FFTW schemes <scheme_table>`.  If a match is found, the plan that
    corresponds to that scheme is created, operating on the arrays that are
    passed in. If no scheme can be created, then ``ValueError`` is raised.

    The actual FFT or iFFT is performed by calling the 
    :meth:`~pyfftw.FFTW.execute` method.
    
    The arrays can be updated by calling the 
    :meth:`~pyfftw.FFTW.update_arrays` method.

    The created instance of the class is itself callable, and can perform
    the execution of the FFT, both with or without array updates, returning
    the result of the FFT. Unlike calling the :meth:`~pyfftw.FFTW.execute`
    method, calling the class instance will also optionally normalise the
    output as necessary. Additionally, calling with an input array update
    will also coerce that array to be the correct dtype. 
    
    See the documentation on the :meth:`~pyfftw.FFTW.__call__` method 
    for more information.
    '''
    # Each of these function pointers simply
    # points to a chosen fftw wrapper function
    cdef fftw_generic_plan_guru __fftw_planner
    cdef fftw_generic_execute __fftw_execute
    cdef fftw_generic_destroy_plan __fftw_destroy
    cdef fftw_generic_plan_with_nthreads __nthreads_plan_setter

    # The plan is typecast when it is created or used
    # within the wrapper functions
    cdef void *__plan

    cdef np.ndarray __input_array
    cdef np.ndarray __output_array
    cdef int __direction
    cdef int __flags

    cdef bint __simd_allowed
    cdef int __input_array_alignment
    cdef int __output_array_alignment    
    cdef bint __use_threads

    cdef object __input_strides
    cdef object __input_byte_strides
    cdef object __output_strides
    cdef object __output_byte_strides
    cdef object __input_shape
    cdef object __output_shape
    cdef object __input_dtype
    cdef object __output_dtype
    cdef object __flags_used

    cdef float __normalisation_scaling

    cdef int __rank
    cdef _fftw_iodim *__dims
    cdef int __howmany_rank
    cdef _fftw_iodim *__howmany_dims

    cdef int64_t *__axes
    cdef int64_t *__not_axes

    cdef int __N
    def __get_N(self):
        '''
        The product of the lengths of the DFT over all DFT axes.
        1/N is the normalisation constant. For any input array A, 
        and for any set of axes, 1/N * ifft(fft(A)) = A
        '''
        return self.__N

    N = property(__get_N)

    def __get_simd_aligned(self):
        '''
        Return whether or not this FFTW object requires simd aligned
        input and output data.
        '''
        return self.__simd_allowed

    simd_aligned = property(__get_simd_aligned)

    def __get_input_alignment(self):
        '''
        Returns the byte alignment of the input arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Input array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised, or
        a copy being made if the :meth:`~pyfftw.FFTW.__call__` 
        interface is used.
        '''
        return self.__input_array_alignment

    input_alignment = property(__get_input_alignment)

    def __get_output_alignment(self):
        '''
        Returns the byte alignment of the output arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Output array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised.
        '''
        return self.__output_array_alignment

    output_alignment = property(__get_output_alignment)

    def __get_flags_used(self):
        '''
        Return which flags were used to construct the FFTW object.
        
        This includes flags that were added during initialisation.
        '''
        return tuple(self.__flags_used)

    flags = property(__get_flags_used)

    def __cinit__(self, input_array, output_array, axes=(-1,),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), 
            unsigned int threads=1, planning_timelimit=None,
            *args, **kwargs):
        
        # Initialise the pointers that need to be freed
        self.__plan = NULL
        self.__dims = NULL
        self.__howmany_dims = NULL

        self.__axes = NULL
        self.__not_axes = NULL

        flags = list(flags)

        cdef double _planning_timelimit
        if planning_timelimit is None:
            _planning_timelimit = FFTW_NO_TIMELIMIT
        else:
            try:
                _planning_timelimit = planning_timelimit
            except TypeError:
                raise TypeError('Invalid planning timelimit: '
                        'The planning timelimit needs to be a float.')

        if not isinstance(input_array, np.ndarray):
            raise ValueError('Invalid input array: '
                    'The input array needs to be an instance '
                    'of numpy.ndarray')

        if not isinstance(output_array, np.ndarray):
            raise ValueError('Invalid output array: '
                    'The output array needs to be an instance '
                    'of numpy.ndarray')

        try:
            input_dtype = input_array.dtype
            output_dtype = output_array.dtype
            scheme = fftw_schemes[(input_dtype, output_dtype)]
        except KeyError:
            raise ValueError('Invalid scheme: '
                    'The output array and input array dtypes '
                    'do not correspond to a valid fftw scheme.')

        self.__input_dtype = input_dtype
        self.__output_dtype = output_dtype
        
        functions = scheme_functions[scheme]
        
        self.__fftw_planner = planners[functions['planner']]
        self.__fftw_execute = executors[functions['executor']]
        self.__fftw_destroy = destroyers[functions['generic_precision']]

        self.__nthreads_plan_setter = (
                nthreads_plan_setters[functions['generic_precision']])

        cdef fftw_generic_set_timelimit set_timelimit_func = (
                set_timelimit_funcs[functions['generic_precision']])

        # If either of the arrays is not aligned on a 16-byte boundary,
        # we set the FFTW_UNALIGNED flag. This disables SIMD.
        # (16 bytes is assumed to be the minimal alignment)
        if 'FFTW_UNALIGNED' in flags:
            self.__simd_allowed = False
            self.__input_array_alignment = self.__input_dtype.alignment
            self.__output_array_alignment = self.__output_dtype.alignment

        else:

            self.__input_array_alignment = -1
            self.__output_array_alignment = -1

            for each_alignment in _valid_simd_alignments:
                if (<intptr_t>np.PyArray_DATA(input_array) % 
                        each_alignment == 0 and
                        <intptr_t>np.PyArray_DATA(output_array) % 
                        each_alignment == 0):

                    self.__simd_allowed = True

                    self.__input_array_alignment = each_alignment
                    self.__output_array_alignment = each_alignment

                    break

            if (self.__input_array_alignment == -1 or
                    self.__output_array_alignment == -1):

                self.__simd_allowed = False

                self.__input_array_alignment = (
                        self.__input_dtype.alignment)
                self.__output_array_alignment = (
                        self.__output_dtype.alignment)
                flags.append('FFTW_UNALIGNED')

        if (not (<intptr_t>np.PyArray_DATA(input_array)
            % self.__input_array_alignment == 0)):
            raise ValueError('Invalid input alignment: '
                    'The input array is expected to lie on a %d '
                    'byte boundary.' % self.__input_array_alignment)

        if (not (<intptr_t>np.PyArray_DATA(output_array)
            % self.__output_array_alignment == 0)):
            raise ValueError('Invalid output alignment: '
                    'The output array is expected to lie on a %d '
                    'byte boundary.' % self.__output_array_alignment)

        if not direction in scheme_directions[scheme]:
            raise ValueError('Invalid direction: '
                    'The direction is not valid for the scheme. '
                    'Try setting it explicitly if it is not already.')

        self.__direction = directions[direction]
        self.__input_shape = input_array.shape
        self.__output_shape = output_array.shape
        
        self.__input_array = input_array
        self.__output_array = output_array

        self.__axes = <int64_t *>malloc(len(axes)*sizeof(int64_t))
        for n in range(len(axes)):
            self.__axes[n] = axes[n]

        # Set the negative entries to their actual index (use the size
        # of the shape array for this)
        cdef int64_t array_dimension = len(self.__input_shape)

        for n in range(len(axes)):
            if self.__axes[n] < 0:
                self.__axes[n] = self.__axes[n] + array_dimension

            if self.__axes[n] >= array_dimension or self.__axes[n] < 0:
                raise IndexError('Invalid axes: '
                    'The axes list cannot contain invalid axes.')

        cdef int64_t unique_axes_length
        cdef int64_t *unique_axes
        cdef int64_t *not_axes
        
        make_axes_unique(self.__axes, len(axes), &unique_axes,
                &not_axes, array_dimension, &unique_axes_length)

        # and assign axes and not_axes to the filled arrays
        free(self.__axes)
        self.__axes = unique_axes
        self.__not_axes = not_axes

        total_N = 1
        for n in range(unique_axes_length):
            if self.__input_shape[self.__axes[n]] == 0:
                raise ValueError('Zero length array: '
                    'The input array should have no zero length'
                    'axes over which the FFT is to be taken')

            if self.__direction == FFTW_FORWARD:
                total_N *= self.__input_shape[self.__axes[n]]
            else:
                total_N *= self.__output_shape[self.__axes[n]]

        self.__N = total_N
        self.__normalisation_scaling = 1/float(self.N)

        # Now we can validate the array shapes
        cdef validator _validator

        if functions['validator'] == -1:
            if not (output_array.shape == input_array.shape):
                raise ValueError('Invalid shapes: '
                        'The output array should be the same shape as the '
                        'input array for the given array dtypes.')
        else:
            _validator = validators[functions['validator']]
            if not _validator(input_array, output_array, 
                    self.__axes, self.__not_axes, unique_axes_length):
                raise ValueError('Invalid shapes: '
                        'The input array and output array are invalid '
                        'complementary shapes for their dtypes.')

        self.__rank = unique_axes_length
        self.__howmany_rank = self.__input_array.ndim - unique_axes_length
        
        self.__flags = 0
        self.__flags_used = []
        for each_flag in flags:
            try:
                self.__flags |= flag_dict[each_flag]
                self.__flags_used.append(each_flag)
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' + 
                        each_flag + '\' is not a valid planner flag.')

        
        if ('FFTW_DESTROY_INPUT' not in flags) and (
                (scheme[0] != 'c2r') or not self.__rank > 1):
            # The default in all possible cases is to preserve the input
            # This is not possible for r2c arrays with rank > 1
            self.__flags |= FFTW_PRESERVE_INPUT

        # Set up the arrays of structs for holding the stride shape 
        # information
        self.__dims = <_fftw_iodim *>malloc(
                self.__rank * sizeof(_fftw_iodim))
        self.__howmany_dims = <_fftw_iodim *>malloc(
                self.__howmany_rank * sizeof(_fftw_iodim))

        if self.__dims == NULL or self.__howmany_dims == NULL:
            # Not much else to do than raise an exception
            raise MemoryError

        # Find the strides for all the axes of both arrays in terms of the 
        # number of elements (as opposed to the number of bytes).
        self.__input_byte_strides = input_array.strides        
        self.__input_strides = tuple([stride/input_array.itemsize 
            for stride in input_array.strides])
        self.__output_byte_strides = output_array.strides
        self.__output_strides = tuple([stride/output_array.itemsize 
            for stride in output_array.strides])

        # Make sure that the arrays are not too big for fftw
        # This is hard to test, so we cross our fingers and hope for the 
        # best (any suggestions, please get in touch).
        cdef int i
        for i in range(0, len(self.__input_shape)):
            if self.__input_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the input array must be ' +
                        'less than ', str(limits.INT_MAX))

            if self.__input_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the input array must be ' +
                        'less than ', str(limits.INT_MAX))

        for i in range(0, len(self.__output_shape)):
            if self.__output_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the output array must be ' +
                        'less than ', str(limits.INT_MAX))

            if self.__output_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the output array must be ' +
                        'less than ', str(limits.INT_MAX))

        fft_shape_lookup = functions['fft_shape_lookup']
        if fft_shape_lookup == -1:
            fft_shape = self.__input_shape
        else:
            fft_shape = fft_shape_lookup(input_array, output_array)

        # Fill in the stride and shape information
        input_strides_array = self.__input_strides
        output_strides_array = self.__output_strides
        for i in range(0, self.__rank):
            self.__dims[i]._n = fft_shape[self.__axes[i]]
            self.__dims[i]._is = input_strides_array[self.__axes[i]]
            self.__dims[i]._os = output_strides_array[self.__axes[i]]

        for i in range(0, self.__howmany_rank):
            self.__howmany_dims[i]._n = fft_shape[self.__not_axes[i]]
            self.__howmany_dims[i]._is = input_strides_array[self.__not_axes[i]]
            self.__howmany_dims[i]._os = output_strides_array[self.__not_axes[i]]

        ## Point at which FFTW calls are made
        ## (and none should be made before this)
        if threads > 1:
            self.__use_threads = True
            self.__nthreads_plan_setter(threads)
        else:
            self.__use_threads = False
            self.__nthreads_plan_setter(1)

        # Set the timelimit
        set_timelimit_func(_planning_timelimit)

        # Finally, construct the plan
        self.__plan = self.__fftw_planner(
            self.__rank, <fftw_iodim *>self.__dims,
            self.__howmany_rank, <fftw_iodim *>self.__howmany_dims,
            <void *>np.PyArray_DATA(self.__input_array),
            <void *>np.PyArray_DATA(self.__output_array),
            self.__direction, self.__flags)

        if self.__plan == NULL:
            raise RuntimeError('The data has an uncaught error that led '+
                    'to the planner returning NULL. This is a bug.')

    def __init__(self, input_array, output_array, axes=(-1,), 
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), 
            int threads=1, planning_timelimit=None, 
            *args, **kwargs):
        '''
        **Arguments**:

        * ``input_array`` and ``output_array`` should be numpy arrays.
          The contents of these arrays will be destroyed by the planning 
          process during initialisation. Information on supported 
          dtypes for the arrays is :ref:`given below <scheme_table>`.
        
        * ``axes`` describes along which axes the DFT should be taken.
          This should be a valid list of axes. Repeated axes are 
          only transformed once. Invalid axes will raise an ``IndexError`` 
          exception. This argument is equivalent to the same
          argument in :func:`numpy.fft.fftn`, except for the fact that
          the behaviour of repeated axes is different (``numpy.fft``
          will happily take the fft of the same axis if it is repeated
          in the ``axes`` argument). Rudimentary testing has suggested
          this is down to the underlying FFTW library and so unlikely 
          to be fixed in these wrappers.

        * ``direction`` should be a string and one of ``'FFTW_FORWARD'`` 
          or ``'FFTW_BACKWARD'``, which dictate whether to take the
          DFT (forwards) or the inverse DFT (backwards) respectively 
          (specifically, it dictates the sign of the exponent in the 
          DFT formulation).

          Note that only the Complex schemes allow a free choice
          for ``direction``. The direction *must* agree with the 
          the :ref:`table below <scheme_table>` if a Real scheme 
          is used, otherwise a ``ValueError`` is raised.

        .. _FFTW_flags:

        * ``flags`` is a list of strings and is a subset of the 
          flags that FFTW allows for the planners:

          * ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'``, ``'FFTW_PATIENT'`` and 
            ``'FFTW_EXHAUSTIVE'`` are supported. These describe the 
            increasing amount of effort spent during the planning 
            stage to create the fastest possible transform. 
            Usually ``'FFTW_MEASURE'`` is a good compromise. If no flag
            is passed, the default ``'FFTW_MEASURE'`` is used.
          * ``'FFTW_UNALIGNED'`` is supported. 
            This tells FFTW not to assume anything about the 
            alignment of the data and disabling any SIMD capability 
            (see below).
          * ``'FFTW_DESTROY_INPUT'`` is supported.
            This tells FFTW that the input array can be destroyed during
            the transform, sometimes allowing a faster algorithm to be
            used. The default behaviour is, if possible, to preserve the
            input. In the case of the 1D Backwards Real transform, this 
            may result in a performance hit. In the case of a backwards
            real transform for greater than one dimension, it is not
            possible to preserve the input, making this flag implicit
            in that case. A little more on this is given 
            :ref:`below<scheme_table>`.

          The `FFTW planner flags documentation 
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          has more information about the various flags and their impact.
          Note that only the flags documented here are supported.

        * ``threads`` tells the wrapper how many threads to use
          when invoking FFTW, with a default of 1. If the number
          of threads is greater than 1, then the GIL is released
          by necessity.

        * ``planning_timelimit`` is a floating point number that 
          indicates to the underlying FFTW planner the maximum number of
          seconds it should spend planning the FFT. This is a rough
          estimate and corresponds to calling of ``fftw_set_timelimit()``
          (or an equivalent dependent on type) in the underlying FFTW
          library. If ``None`` is set, the planner will run indefinitely
          until all the planning modes allowed by the flags have been
          tried. See the `FFTW planner flags page
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          for more information on this.

        .. _fftw_schemes:

        **Schemes**

        The currently supported schemes are as follows:

        .. _scheme_table:

        +----------------+-----------------------+------------------------+-----------+
        | Type           | ``input_array.dtype`` | ``output_array.dtype`` | Direction |
        +================+=======================+========================+===========+
        | Complex        | ``complex64``         | ``complex64``          | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Complex        | ``complex128``        | ``complex128``         | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Complex        | ``clongdouble``       | ``clongdouble``        | Both      |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | ``float32``           | ``complex64``          | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | ``float64``           | ``complex128``         | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real           | ``longdouble``        | ``clongdouble``        | Forwards  |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | ``complex64``         | ``float32``            | Backwards |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | ``complex128``        | ``float64``            | Backwards |
        +----------------+-----------------------+------------------------+-----------+
        | Real\ :sup:`1` | ``clongdouble``       | ``longdouble``         | Backwards |
        +----------------+-----------------------+------------------------+-----------+

        \ :sup:`1`  Note that the Backwards Real transform for the case
        in which the dimensionality of the transform is greater than 1
        will destroy the input array. This is inherent to FFTW and the only
        general work-around for this is to copy the array prior to
        performing the transform. In the case where the dimensionality
        of the transform is 1, the default is to preserve the input array.
        This is different from the default in the underlying library, and
        some speed gain may be achieved by allowing the input array to
        be destroyed by passing the ``'FFTW_DESTROY_INPUT'`` 
        :ref:`flag <FFTW_flags>`.

        ``clongdouble`` typically maps directly to ``complex256``
        or ``complex192``, and ``longdouble`` to ``float128`` or
        ``float96``, dependent on platform.

        The relative shapes of the arrays should be as follows:

        * For a Complex transform, ``output_array.shape == input_array.shape``
        * For a Real transform in the Forwards direction, both the following 
          should be true:

          * ``output_array.shape[axes][-1] == input_array.shape[axes][-1]//2 + 1``
          * All the other axes should be equal in length.

        * For a Real transform in the Backwards direction, both the following 
          should be true:

          * ``input_array.shape[axes][-1] == output_array.shape[axes][-1]//2 + 1``
          * All the other axes should be equal in length.

        In the above expressions for the Real transform, the ``axes`` 
        arguments denotes the unique set of axes on which we are taking
        the FFT, in the order passed. It is the last of these axes that 
        is subject to the special case shown.

        The shapes for the real transforms corresponds to those
        stipulated by the FFTW library. Further information can be
        found in the FFTW documentation on the `real DFT
        <http://www.fftw.org/fftw3_doc/Guru-Real_002ddata-DFTs.html>`_.

        The actual arrangement in memory is arbitrary and the scheme
        can be planned for any set of strides on either the input
        or the output. The user should not have to worry about this
        and any valid numpy array should work just fine.

        What is calculated is exactly what FFTW calculates. 
        Notably, this is an unnormalized transform so should 
        be scaled as necessary (fft followed by ifft will scale 
        the input by N, the product of the dimensions along which
        the DFT is taken). For further information, see the 
        `FFTW documentation
        <http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html>`_.

        The FFTW library benefits greatly from the beginning of each
        DFT axes being aligned on the correct byte boundary, enabling
        SIMD instructions. By default, if the data begins on such a
        boundary, then FFTW will be allowed to try and enable
        SIMD instructions. This means that all future changes to
        the data arrays will be checked for similar alignment. SIMD
        instructions can be explicitly disabled by setting the
        FFTW_UNALIGNED flags, to allow for updates with unaligned
        data.
        
        :func:`~pyfftw.n_byte_align` and 
        :func:`~pyfftw.n_byte_align_empty` are two methods
        included with this module for producing aligned arrays.

        The optimum alignment for the running platform is provided
        by :data:`pyfftw.simd_alignment`, though a different alignment
        may still result in some performance improvement. For example,
        if the processor supports AVX (requiring 32-byte alignment) as
        well as SSE (requiring 16-byte alignment), then if the array
        is 16-byte aligned, SSE will still be used.

        It's worth noting that just being aligned may not be sufficient
        to create the fastest possible transform. For example, if the
        array is not contiguous (i.e. certain axes are displaced in
        memory), it may be faster to plan a transform for a contiguous
        array, and then rely on the array being copied in before the
        transform (which :class:`pyfftw.FFTW` will handle for you when
        accessed through :meth:`~pyfftw.FFTW.__call__`).
        '''
        pass

    def __dealloc__(self):

        if not self.__axes == NULL:
            free(self.__axes)

        if not self.__not_axes == NULL:
            free(self.__not_axes)

        if not self.__plan == NULL:
            self.__fftw_destroy(self.__plan)

        if not self.__dims == NULL:
            free(self.__dims)

        if not self.__howmany_dims == NULL:
            free(self.__howmany_dims)

    def __call__(self, input_array=None, output_array=None, 
            normalise_idft=True):
        '''__call__(input_array=None, output_array=None, normalise_idft=True)

        Calling the class instance (optionally) updates the arrays, then
        calls :meth:`~pyfftw.FFTW.execute`, before optionally normalising 
        the output and returning the output array.

        It has some built-in helpers to make life simpler for the calling
        functions (as distinct from manually updating the arrays and
        calling :meth:`~pyfftw.FFTW.execute`).

        If ``normalise_idft`` is ``True`` (the default), then the output from 
        an inverse DFT (i.e. when the direction flag is ``'FFTW_BACKWARD'``) is
        scaled by 1/N, where N is the product of the lengths of input array on
        which the FFT is taken. If the direction is ``'FFTW_FORWARD'``, this
        flag makes no difference to the output array.
        
        When ``input_array`` is something other than None, then the passed in
        array is coerced to be the same dtype as the input array used when the
        class was instantiated, the byte-alignment of the passed in array is
        made consistent with the expected byte-alignment and the striding is 
        made consistent with the expected striding. All this may, but not 
        necessarily, require a copy to be made.

        As noted in the :ref:`scheme table<scheme_table>`, if the FFTW 
        instance describes a backwards real transform of more than one
        dimension, the contents of the input array will be destroyed. It is
        up to the calling function to make a copy if it is necessary to
        maintain the input array.

        ``output_array`` is always used as-is if possible. If the dtype, the 
        alignment or the striding is incorrect for the FFTW object, then a
        ``ValueError`` is raised.
        
        The coerced input array and the output array (as appropriate) are 
        then passed as arguments to
        :meth:`~pyfftw.FFTW.update_arrays`, after which
        :meth:`~pyfftw.FFTW.execute` is called, and then normalisation
        is applied to the output array if that is desired.
        
        Note that it is possible to pass some data structure that can be
        converted to an array, such as a list, so long as it fits the data
        requirements of the class instance, such as array shape.

        Other than the dtype and the alignment of the passed in arrays, the 
        rest of the requirements on the arrays mandated by
        :meth:`~pyfftw.FFTW.update_arrays` are enforced.

        A ``None`` argument to either keyword means that that array is not 
        updated.

        The result of the FFT is returned. This is the same array that is used
        internally and will be overwritten again on subsequent calls. If you
        need the data to persist longer than a subsequent call, you should
        copy the returned array.
        '''

        if input_array is not None or output_array is not None:

            if input_array is None:
                input_array = self.__input_array

            if output_array is None:
                output_array = self.__output_array

            if not isinstance(input_array, np.ndarray):
                copy_needed = True
            elif (not input_array.dtype == self.__input_dtype):
                copy_needed = True
            elif (not input_array.strides == self.__input_byte_strides):
                copy_needed = True
            elif not (<intptr_t>np.PyArray_DATA(input_array) 
                    % self.input_alignment == 0):
                copy_needed = True
            else:
                copy_needed = False

            if copy_needed:

                if not isinstance(input_array, np.ndarray):
                    input_array = np.asanyarray(input_array)

                if not input_array.shape == self.__input_shape:
                    raise ValueError('Invalid input shape: '
                            'The new input array should be the same shape '
                            'as the input array used to instantiate the '
                            'object.')
                
                self.__input_array[:] = input_array
                
                if output_array is not None:
                    # No point wasting time if no update is necessary
                    # (which the copy above may have avoided)
                    input_array = self.__input_array
                    self.update_arrays(input_array, output_array)

            else:
                self.update_arrays(input_array, output_array)

        self.execute()

        if self.__direction == FFTW_BACKWARD and normalise_idft:
            self.__output_array *= self.__normalisation_scaling

        return self.__output_array

    cpdef update_arrays(self, 
            new_input_array, new_output_array):
        '''update_arrays(new_input_array, new_output_array)

        Update the arrays upon which the DFT is taken.

        The new arrays should be of the same dtypes as the originals, the same
        shapes as the originals and should have the same strides between axes.
        If the original data was aligned so as to allow SIMD instructions
        (e.g. by being aligned on a 16-byte boundary), then the new array must
        also be aligned so as to allow SIMD instructions (assuming, of
        course, that the ``FFTW_UNALIGNED`` flag was not enabled).
        
        The byte alignment requirement extends to requiring natural
        alignment in the non-SIMD cases as well, but this is much less
        stringent as it simply means avoiding arrays shifted by, say,
        a single byte (which invariably takes some effort to
        achieve!).

        If all these conditions are not met, a ``ValueError`` will
        be raised and the data will *not* be updated (though the 
        object will still be in a sane state).
        '''
        if not isinstance(new_input_array, np.ndarray):
            raise ValueError('Invalid input array: '
                    'The new input array needs to be an instance '
                    'of numpy.ndarray')

        if not isinstance(new_output_array, np.ndarray):
            raise ValueError('Invalid output array '
                    'The new output array needs to be an instance '
                    'of numpy.ndarray')

        if not (<intptr_t>np.PyArray_DATA(new_input_array) % 
                self.input_alignment == 0):
            raise ValueError('Invalid input alignment: '
                    'The original arrays were %d-byte aligned. It is '
                    'necessary that the update input array is similarly '
                    'aligned.' % self.input_alignment)

        if not (<intptr_t>np.PyArray_DATA(new_output_array) % 
                self.output_alignment == 0):
            raise ValueError('Invalid output alignment: '
                    'The original arrays were %d-byte aligned. It is '
                    'necessary that the update output array is similarly '
                    'aligned.' % self.output_alignment)

        if not new_input_array.dtype == self.__input_dtype:
            raise ValueError('Invalid input dtype: '
                    'The new input array is not of the same '
                    'dtype as was originally planned for.')

        if not new_output_array.dtype == self.__output_dtype:
            raise ValueError('Invalid output dtype: '
                    'The new output array is not of the same '
                    'dtype as was originally planned for.')

        new_input_shape = new_input_array.shape
        new_output_shape = new_output_array.shape

        new_input_strides = new_input_array.strides
        new_output_strides = new_output_array.strides

        if not new_input_shape == self.__input_shape:
            raise ValueError('Invalid input shape: '
                    'The new input array should be the same shape as '
                    'the input array used to instantiate the object.')

        if not new_output_shape == self.__output_shape:
            raise ValueError('Invalid output shape: '
                    'The new output array should be the same shape as '
                    'the output array used to instantiate the object.')
        
        if not new_input_strides == self.__input_byte_strides:
            raise ValueError('Invalid input striding: '
                    'The strides should be identical for the new '
                    'input array as for the old.')
        
        if not new_output_strides == self.__output_byte_strides:
            raise ValueError('Invalid output striding: '
                    'The strides should be identical for the new '
                    'output array as for the old.')

        self._update_arrays(new_input_array, new_output_array)

    cdef _update_arrays(self, 
            np.ndarray new_input_array, np.ndarray new_output_array):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self.__input_array = new_input_array
        self.__output_array = new_output_array

    def get_input_array(self):
        '''get_input_array()

        Return the input array that is associated with the FFTW 
        instance.
        '''
        return self.__input_array

    def get_output_array(self):
        '''get_output_array()

        Return the output array that is associated with the FFTW
        instance.
        '''
        return self.__output_array

    cpdef execute(self):
        '''execute()

        Execute the planned operation, taking the correct kind of FFT of
        the input array (what is returned by :meth:`get_input_array`), 
        and putting the result in the output array (what is returned by
        :meth:`get_output_array`).
        '''
        cdef void *input_pointer = (
                <void *>np.PyArray_DATA(self.__input_array))
        cdef void *output_pointer = (
                <void *>np.PyArray_DATA(self.__output_array))
        
        cdef void *plan = self.__plan
        cdef fftw_generic_execute fftw_execute = self.__fftw_execute
        
        if self.__use_threads:
            with nogil:
                fftw_execute(plan, input_pointer, output_pointer)
        else:
            fftw_execute(self.__plan, input_pointer, output_pointer)

cdef void count_char(char c, void *counter_ptr):
    '''
    On every call, increment the derefenced counter_ptr.
    '''
    (<int *>counter_ptr)[0] += 1


cdef void write_char_to_string(char c, void *string_location_ptr):
    '''
    Write the passed character c to the memory location
    pointed to by the contents of string_location_ptr (i.e. a pointer
    to a pointer), then increment the contents of string_location_ptr 
    (i.e. move to the next byte in memory).

    In other words, for every character that is passed, we write that
    to a string that is referenced by the dereferenced value of 
    string_location_ptr.

    If the derefenced value of string_location points to an
    unallocated piece of memory, a segfault will likely occur.
    '''
    cdef char *write_location = <char *>((<intptr_t *>string_location_ptr)[0])
    write_location[0] = c

    (<intptr_t *>string_location_ptr)[0] += 1


def export_wisdom():
    '''export_wisdom()

    Return the FFTW wisdom as a tuple of strings.

    The first string in the tuple is the string for the double
    precision wisdom. The second string in the tuple is the string 
    for the single precision wisdom. The third string in the tuple 
    is the string for the long double precision wisdom.

    The tuple that is returned from this function can be used as the
    argument to :func:`~pyfftw.import_wisdom`.
    '''

    cdef bytes py_wisdom
    cdef bytes py_wisdomf
    cdef bytes py_wisdoml

    cdef int counter = 0
    cdef int counterf = 0
    cdef int counterl = 0

    fftw_export_wisdom(&count_char, <void *>&counter)
    fftwf_export_wisdom(&count_char, <void *>&counterf)
    fftwl_export_wisdom(&count_char, <void *>&counterl)

    cdef char* c_wisdom = <char *>malloc(sizeof(char)*(counter + 1))
    cdef char* c_wisdomf = <char *>malloc(sizeof(char)*(counterf + 1))
    cdef char* c_wisdoml = <char *>malloc(sizeof(char)*(counterl + 1))

    if c_wisdom == NULL or c_wisdomf == NULL or c_wisdoml == NULL:
        raise MemoryError

    # Set the pointers to the string pointers
    cdef intptr_t c_wisdom_ptr = <intptr_t>c_wisdom
    cdef intptr_t c_wisdomf_ptr = <intptr_t>c_wisdomf
    cdef intptr_t c_wisdoml_ptr = <intptr_t>c_wisdoml

    fftw_export_wisdom(&write_char_to_string, <void *>&c_wisdom_ptr)
    fftwf_export_wisdom(&write_char_to_string, <void *>&c_wisdomf_ptr)
    fftwl_export_wisdom(&write_char_to_string, <void *>&c_wisdoml_ptr)

    # Write the last byte as the null byte
    c_wisdom[counter] = 0
    c_wisdomf[counterf] = 0
    c_wisdoml[counterl] = 0

    try:
        py_wisdom = c_wisdom
        py_wisdomf = c_wisdomf
        py_wisdoml = c_wisdoml

    finally:
        free(c_wisdom)
        free(c_wisdomf)
        free(c_wisdoml)

    return (py_wisdom, py_wisdomf, py_wisdoml)

def import_wisdom(wisdom):
    '''import_wisdom(wisdom)

    Function that imports wisdom from the passed tuple
    of strings.

    The first string in the tuple is the string for the double
    precision wisdom. The second string in the tuple is the string 
    for the single precision wisdom. The third string in the tuple 
    is the string for the long double precision wisdom.

    The tuple that is returned from :func:`~pyfftw.export_wisdom`
    can be used as the argument to this function.

    This function returns a tuple of boolean values indicating
    the success of loading each of the wisdom types (double, float 
    and long double, in that order).
    '''

    cdef char* c_wisdom = wisdom[0]
    cdef char* c_wisdomf = wisdom[1]
    cdef char* c_wisdoml = wisdom[2]

    cdef bint success = fftw_import_wisdom_from_string(c_wisdom)
    cdef bint successf = fftwf_import_wisdom_from_string(c_wisdomf)
    cdef bint successl = fftwl_import_wisdom_from_string(c_wisdoml)

    return (success, successf, successl)

#def export_wisdom_to_files(
#        double_wisdom_file=None,
#        single_wisdom_file=None, 
#        long_double_wisdom_file=None):
#    '''export_wisdom_to_file(double_wisdom_file=None, single_wisdom_file=None, long_double_wisdom_file=None)
#
#    Export the wisdom to the passed files.
#
#    The double precision wisdom is written to double_wisdom_file. 
#    The single precision wisdom is written to single_wisdom_file.
#    The long double precision wisdom is written to 
#    long_double_wisdom_file.
#
#    If any of the arguments are None, then nothing is done for that
#    file.
#
#    This function returns a tuple of boolean values indicating
#    the success of storing each of the wisdom types (double, float 
#    and long double, in that order).
#    '''
#    cdef bint success = True
#    cdef bint successf = True
#    cdef bint successl = True
#
#    cdef char *_double_wisdom_file
#    cdef char *_single_wisdom_file
#    cdef char *_long_double_wisdom_file
#
#
#    if double_wisdom_file is not None:
#        _double_wisdom_file = double_wisdom_file
#        success = fftw_export_wisdom_to_filename(_double_wisdom_file)
#
#    if single_wisdom_file is not None:
#        _single_wisdom_file = single_wisdom_file
#        successf = fftwf_export_wisdom_to_filename(_single_wisdom_file)
#
#    if long_double_wisdom_file is not None:
#        _long_double_wisdom_file = long_double_wisdom_file
#        successl = fftwl_export_wisdom_to_filename(
#                _long_double_wisdom_file)
#
#    return (success, successf, successl)
#
#def import_wisdom_to_files(
#        double_wisdom_file=None,
#        single_wisdom_file=None, 
#        long_double_wisdom_file=None):
#    '''import_wisdom_to_file(double_wisdom_file=None, single_wisdom_file=None, long_double_wisdom_file=None)
#
#    import the wisdom to the passed files.
#
#    The double precision wisdom is imported from double_wisdom_file. 
#    The single precision wisdom is imported from single_wisdom_file.
#    The long double precision wisdom is imported from 
#    long_double_wisdom_file.
#
#    If any of the arguments are None, then nothing is done for that
#    file.
#
#    This function returns a tuple of boolean values indicating
#    the success of loading each of the wisdom types (double, float 
#    and long double, in that order).
#    '''
#    cdef bint success = True
#    cdef bint successf = True
#    cdef bint successl = True
#
#    cdef char *_double_wisdom_file
#    cdef char *_single_wisdom_file
#    cdef char *_long_double_wisdom_file
#
#    if double_wisdom_file is not None:
#        _double_wisdom_file = double_wisdom_file
#        success = fftw_import_wisdom_from_filename(_double_wisdom_file)
#
#    if single_wisdom_file is not None:
#        _single_wisdom_file = single_wisdom_file
#        successf = fftwf_import_wisdom_from_filename(_single_wisdom_file)
#
#    if long_double_wisdom_file is not None:
#        _long_double_wisdom_file = long_double_wisdom_file
#        successl = fftwl_import_wisdom_from_filename(
#                _long_double_wisdom_file)
#
#    return (success, successf, successl)

def forget_wisdom():
    '''forget_wisdom()

    Forget all the accumulated wisdom.
    '''
    fftw_forget_wisdom()
    fftwf_forget_wisdom()
    fftwl_forget_wisdom()


