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
from libc.stdlib cimport malloc, free
from libc cimport limits

from pyfftw cimport _fftw_iodim, fftw_iodim, fftwf_plan
cimport pyfftw


directions = {'FFTW_FORWARD': FFTW_FORWARD,
        'FFTW_BACKWARD': FFTW_BACKWARD}

flag_dict = {'FFTW_MEASURE': FFTW_MEASURE,
        'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
        'FFTW_PATIENT': FFTW_PATIENT,
        'FFTW_ESTIMATE': FFTW_ESTIMATE,
        'FFTW_UNALIGNED': FFTW_UNALIGNED}

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

#    Init threads
#    ============
#
# Double precision
cdef void _fftw_init_threads():

    fftw_init_threads()

# Single precision
cdef void _fftwf_init_threads():

    fftwf_init_threads()

# Long double precision
cdef void _fftwl_init_threads():

    fftwl_init_threads()

#    Plan with n threads
#    ===================
#
# Double precision
cdef void _fftw_plan_with_nthreads(int n):

    fftw_plan_with_nthreads(n)

# Single precision
cdef void _fftwf_plan_with_nthreads(int n):

    fftwf_plan_with_nthreads(n)

# Long double precision
cdef void _fftwl_plan_with_nthreads(int n):

    fftwl_plan_with_nthreads(n)

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


# Threads init table
cdef fftw_generic_init_threads thread_initializers[3]

cdef fftw_generic_init_threads * _build_init_threads_list():
    thread_initializers[0] = <fftw_generic_init_threads>&_fftw_init_threads
    thread_initializers[1] = <fftw_generic_init_threads>&_fftwf_init_threads
    thread_initializers[2] = <fftw_generic_init_threads>&_fftwl_init_threads

# nthreads plan setters table
cdef fftw_generic_plan_with_nthreads nthreads_plan_setters[3]

cdef fftw_generic_plan_with_nthreads * _build_nthreads_plan_setters_list():
    nthreads_plan_setters[0] = \
            <fftw_generic_plan_with_nthreads>&_fftw_plan_with_nthreads
    nthreads_plan_setters[1] = \
            <fftw_generic_plan_with_nthreads>&_fftwf_plan_with_nthreads
    nthreads_plan_setters[2] = \
            <fftw_generic_plan_with_nthreads>&_fftwl_plan_with_nthreads

# Validator functions
# ===================
def _validate_r2c_arrays(input_array, output_array, axes, not_axes):
    ''' Validates the input and output array to check for
    a valid real to complex transform.
    '''

    # We firstly need to confirm that the dimenions of the arrays
    # are the same
    if not (input_array.ndim == output_array.ndim):
        return False

    # The critical axis is the last of those over which the 
    # FFT is taken. The following shape variables contain
    # the shapes of the FFT and shapes of the other axes.
    in_fft_shape = np.array(input_array.shape)[axes]
    out_fft_shape = np.array(output_array.shape)[axes]

    in_not_fft_shape = np.array(input_array.shape)[not_axes]
    out_not_fft_shape = np.array(output_array.shape)[not_axes]

    if np.alltrue(out_fft_shape[:-1] == in_fft_shape[:-1]) and \
            np.alltrue(out_fft_shape[-1] == in_fft_shape[-1]//2 + 1) and \
            np.alltrue(in_not_fft_shape == out_not_fft_shape):
        return True
    else:
        return False

def _validate_c2r_arrays(input_array, output_array, axes, not_axes):
    ''' Validates the input and output array to check for
    a valid complex to real transform.
    '''
    # We firstly need to confirm that the dimenions of the arrays
    # are the same
    if not (input_array.ndim == output_array.ndim):
        return False

    # The critical axis is the last of those over which the 
    # FFT is taken. The following shape variables contain
    # the shapes of the FFT and shapes of the other axes.
    in_fft_shape = np.array(input_array.shape)[axes]
    out_fft_shape = np.array(output_array.shape)[axes]

    in_not_fft_shape = np.array(input_array.shape)[not_axes]
    out_not_fft_shape = np.array(output_array.shape)[not_axes]

    if np.alltrue(in_fft_shape[:-1] == out_fft_shape[:-1]) and \
            np.alltrue(in_fft_shape[-1] == out_fft_shape[-1]//2 + 1) and \
            np.alltrue(in_not_fft_shape == out_not_fft_shape):
        return True
    else:
        return False

# Shape lookup functions
# ======================
def _lookup_shape_r2c_arrays(input_array, output_array):
    return np.array(input_array.shape)

def _lookup_shape_c2r_arrays(input_array, output_array):
    return np.array(output_array.shape)

# fftw_schemes is a dictionary with a mapping from a keys,
# which are a tuple of the string representation of numpy
# dtypes to a scheme name.
#
# scheme_functions is a dictionary of functions, either 
# an index to the array of functions in the case of 
# 'planner', 'executor' and 'destroyer' or a callable
# in the case of 'validator'
#
# The array indices refer to the relevant functions for each scheme,
# the tables to which are defined above.
#
# The 'validator' function is a callable for validating the arrays
# that has the following signature:
# bool callable(in_array, out_array, axes)
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

fftw_schemes = {
        (np.dtype('complex128'), np.dtype('complex128')): 'c128',
        (np.dtype('complex64'), np.dtype('complex64')): 'c64',
        (np.dtype('clongdouble'), np.dtype('clongdouble')): 'cld',
        (np.dtype('float64'), np.dtype('complex128')): 'r64_to_c128',
        (np.dtype('float32'), np.dtype('complex64')): 'r32_to_c64',
        (np.dtype('longdouble'), np.dtype('clongdouble')): 'rld_to_cld',
        (np.dtype('complex128'), np.dtype('float64')): 'c128_to_r64',
        (np.dtype('complex64'), np.dtype('float32')): 'c64_to_r32',
        (np.dtype('clongdouble'), np.dtype('longdouble')): 'cld_to_rld'}

scheme_directions = {
        'c128': ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        'c64': ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        'cld': ['FFTW_FORWARD', 'FFTW_BACKWARD'],
        'r64_to_c128': ['FFTW_FORWARD'],
        'r32_to_c64': ['FFTW_FORWARD'],
        'rld_to_cld': ['FFTW_FORWARD'],
        'c128_to_r64': ['FFTW_BACKWARD'],
        'c64_to_r32': ['FFTW_BACKWARD'],
        'cld_to_rld': ['FFTW_BACKWARD']}

scheme_functions = {
    'c128': {'planner': 0, 'executor':0, 'destroyer':0,
        't_init': 0, 't_plan': 0,
        'validator':None, 'fft_shape_lookup': None},
    'c64': {'planner':1, 'executor':1, 'destroyer':1,
        't_init': 1, 't_plan': 1,        
        'validator':None, 'fft_shape_lookup': None},
    'cld': {'planner':2, 'executor':2, 'destroyer':2,
        't_init': 2, 't_plan': 2,
        'validator':None, 'fft_shape_lookup': None},
    'r64_to_c128': {'planner':3, 'executor':3, 'destroyer':0,
        't_init': 0, 't_plan': 0,
        'validator':_validate_r2c_arrays, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    'r32_to_c64': {'planner':4, 'executor':4, 'destroyer':1,
        't_init': 1, 't_plan': 1,
        'validator':_validate_r2c_arrays, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    'rld_to_cld': {'planner':5, 'executor':5, 'destroyer':2,
        't_init': 2, 't_plan': 2,
        'validator':_validate_r2c_arrays, 
        'fft_shape_lookup': _lookup_shape_r2c_arrays},
    'c128_to_r64': {'planner':6, 'executor':6, 'destroyer':0, 
        't_init': 0, 't_plan': 0,
        'validator':_validate_c2r_arrays, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays},
    'c64_to_r32': {'planner':7, 'executor':7, 'destroyer':1, 
        't_init': 1, 't_plan': 1,
        'validator':_validate_c2r_arrays, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays},
    'cld_to_rld': {'planner':8, 'executor':8, 'destroyer':2,
        't_init': 2, 't_plan': 2,
        'validator':_validate_c2r_arrays, 
        'fft_shape_lookup': _lookup_shape_c2r_arrays}}

# Initialize the module

# Define the functions        
_build_planner_list()
_build_destroyer_list()
_build_executor_list()
_build_init_threads_list()
_build_nthreads_plan_setters_list()

# Set the cleanup routine
import atexit
@atexit.register
def cleanup():
    fftw_cleanup()
    fftwf_cleanup()
    fftwl_cleanup()
    fftw_cleanup_threads()
    fftwf_cleanup_threads()
    fftwl_cleanup_threads()

# The External Interface
# ======================
#
cdef class FFTW:
    '''
    FFTW is a class for computing the complex N-Dimensional DFT or
    inverse DFT of an array using the FFTW library. The interface is 
    designed to be somewhat pythonic, with the correct transform being 
    inferred from the dtypes of the passed arrays.

    On instantiation, the dtypes of the input arrays are compared to the 
    set of valid (and implemented) FFTW schemes. If a match is found,
    the plan that corresponds to that scheme is created, operating on the
    arrays that are passed in. If no scheme can be created, then 
    ``ValueError`` is raised.

    The actual FFT or iFFT is performed by calling the 
    :ref:`execute()<FFTW_execute>` method.
    
    The arrays can be updated by calling the 
    :ref:`update_arrays()<FFTW_update_arrays>` method.
    '''
    # Each of these function pointers simply
    # points to a chosen fftw wrapper function
    cdef fftw_generic_plan_guru __fftw_planner
    cdef fftw_generic_execute __fftw_execute
    cdef fftw_generic_destroy_plan __fftw_destroy
    cdef fftw_generic_init_threads __thread_initializer
    cdef fftw_generic_plan_with_nthreads __nthreads_plan_setter


    # The plan is typecast when it is created or used
    # within the wrapper functions
    cdef void *__plan

    cdef np.ndarray __input_array
    cdef np.ndarray __output_array
    cdef int __direction
    cdef int __flags
    cdef bint __simd_allowed
    cdef bint __use_threads

    cdef object __input_strides
    cdef object __output_strides
    cdef object __input_shape
    cdef object __output_shape
    cdef object __input_dtype
    cdef object __output_dtype

    cdef int __rank
    cdef _fftw_iodim *__dims
    cdef int __howmany_rank
    cdef _fftw_iodim *__howmany_dims

    def __cinit__(self, input_array, output_array, axes=(-1,),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), 
            unsigned int threads=1):
        
        # Initialise the pointers that need to be freed
        self.__plan = NULL
        self.__dims = NULL
        self.__howmany_dims = NULL

        flags = list(flags)

        try:
            scheme = fftw_schemes[
                    (input_array.dtype, output_array.dtype)]
        except KeyError:
            raise ValueError('The output array and input array dtypes '+\
                    'do not correspond to a valid fftw scheme.')

        self.__input_dtype = input_array.dtype
        self.__output_dtype = output_array.dtype

        # If either of the arrays is not aligned on a 16-byte boundary,
        # we set the FFTW_UNALIGNED flag. This disables SIMD.
        if 'FFTW_UNALIGNED' in flags:
            self.__simd_allowed = False
        elif input_array.ctypes.data%16 == 0 and \
                input_array.ctypes.data%16 == 0:
            self.__simd_allowed = True
        else:
            flags.append('FFTW_UNALIGNED')
            self.__simd_allowed = False

        if not direction in scheme_directions[scheme]:
            raise ValueError('The direction is not valid for the scheme. '\
                    'Try setting it explicitly if it is not already.')

        self.__direction = directions[direction]
        self.__input_shape = np.array(input_array.shape)
        self.__output_shape = np.array(output_array.shape)
        
        functions = scheme_functions[scheme]
        self.__fftw_planner = planners[functions['planner']]
        self.__fftw_execute = executors[functions['executor']]
        self.__fftw_destroy = destroyers[functions['destroyer']]

        self.__thread_initializer = \
                thread_initializers[functions['t_init']]
        self.__nthreads_plan_setter = \
                nthreads_plan_setters[functions['t_plan']]
        
        self.__flags = 0 
        for each_flag in flags:
            self.__flags |= flag_dict[each_flag]

        self.__input_array = input_array
        self.__output_array = output_array
        
        _axes = np.array(axes)
        
        # Set the negative entries to their actual index (use the size
        # of the shape array for this)
        _axes[_axes<0] = _axes[_axes<0] + len(self.__input_shape)

        if (_axes >= len(self.__input_shape)).any() or (_axes < 0).any():
            raise ValueError('The axes list cannot contain invalid axes.')

        # We want to make sure that the axes list contains unique entries
        _axes = np.unique(_axes)

        # Now get the axes along which the FFT is *not* taken
        _not_axes = np.setdiff1d(np.arange(0,len(self.__input_shape)), _axes)
        
        # Now we can validate the array shapes
        if functions['validator'] == None:
            if not (output_array.shape == input_array.shape):
                raise ValueError('The output array should be the same '+\
                        'shape as the input array for the given array '+\
                        'dtypes.')
        else:
            if not functions['validator'](input_array, output_array,
                    _axes, _not_axes):
                raise ValueError('The input array and output array are '+\
                        'invalid complementary shapes for their dtypes.')

        self.__rank = len(_axes)
        self.__howmany_rank = len(_not_axes)

        # Set up the arrays of structs for holding the stride shape 
        # information
        self.__dims = <_fftw_iodim *>malloc(
                self.__rank * sizeof(_fftw_iodim))
        self.__howmany_dims = <_fftw_iodim *>malloc(
                self.__howmany_rank * sizeof(_fftw_iodim))

        # Find the strides for all the axes of both arrays in terms of the 
        # number of elements (as opposed to the number of bytes).
        self.__input_strides = \
                np.array(input_array.strides)/input_array.itemsize
        self.__output_strides = \
                np.array(output_array.strides)/output_array.itemsize

        # Make sure that the arrays are not too big for fftw
        cdef int i
        for i in range(0, len(self.__input_shape)):
            if self.__input_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the input array must be ' +\
                        'less than ', str(limits.INT_MAX))

            if self.__input_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the input array must be ' +\
                        'less than ', str(limits.INT_MAX))

        for i in range(0, len(self.__output_shape)):
            if self.__output_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the output array must be ' +\
                        'less than ', str(limits.INT_MAX))

            if self.__output_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the output array must be ' +\
                        'less than ', str(limits.INT_MAX))
        
        fft_shape_lookup = functions['fft_shape_lookup']
        if fft_shape_lookup == None:
            fft_shape = self.__input_shape
        else:
            fft_shape = fft_shape_lookup(input_array, output_array)

        # Fill in the stride and shape information
        for i in range(0, self.__rank):
            self.__dims[i]._n = fft_shape[_axes][i]
            self.__dims[i]._is = self.__input_strides[_axes][i]
            self.__dims[i]._os = self.__output_strides[_axes][i]

        for i in range(0, self.__howmany_rank):
            self.__howmany_dims[i]._n = fft_shape[_not_axes][i]
            self.__howmany_dims[i]._is = self.__input_strides[_not_axes][i]
            self.__howmany_dims[i]._os = self.__output_strides[_not_axes][i]

        ## Point at which FFTW calls are made
        ## (and none should be made before this)
        if threads > 1:
            self.__use_threads = True
            self.__thread_initializer()
            self.__nthreads_plan_setter(threads)
        else:
            self.__use_threads = False
            self.__thread_initializer()            
            self.__nthreads_plan_setter(1)

        # Finally, construct the plan
        self.__plan = self.__fftw_planner(
            self.__rank, <fftw_iodim *>self.__dims,
            self.__howmany_rank, <fftw_iodim *>self.__howmany_dims,
            <void *>np.PyArray_DATA(self.__input_array),
            <void *>np.PyArray_DATA(self.__output_array),
            self.__direction, self.__flags)

        if self.__plan == NULL:
            raise ValueError('The data has an uncaught error that led '+\
                    'to the planner returning NULL. This is a bug.')

    def __init__(self, input_array, output_array, axes=[-1], 
            direction='FFTW_FORWARD', flags=['FFTW_MEASURE'], 
            int threads=1):
        '''
        ``input_array`` and ``output_array`` should be numpy arrays.
        The contents of these arrays will be destroyed by the planning 
        process during initialisation. Information on supported 
        dtypes for the arrays is given below.
        
        ``axes`` describes along which axes the DFT should be taken.
        This should be a valid list of axes. Repeated axes are 
        only transformed once. Invalid axes will raise an 
        exception. This argument is equivalent to the same
        argument in ``numpy.fft.fftn``.

        ``direction`` should be a string and one of FFTW_FORWARD 
        or FFTW_BACKWARD, which dictate whether to take the
        DFT (forwards) or the inverse DFT (backwards) respectively 
        (specifically, it dictates the sign of the exponent in the 
        DFT formulation).

        Note that only the Complex schemes allow a free choice
        for ``direction``. The direction *must* agree with the 
        the table below if a Real scheme is used, otherwise a 
        ``ValueError`` is raised.

        ``flags`` is a list of strings and is a subset of the 
        flags that FFTW allows for the planners. Specifically, 
        FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT and 
        FFTW_EXHAUSTIVE are supported. These describe the 
        increasing amount of effort spent during the planning 
        stage to create the fastest possible transform. 
        Usually, FFTW_MEASURE is a good compromise and is the 
        default.
        
        In addition the FFTW_UNALIGNED flag is supported. 
        This tells FFTW not to assume anything about the 
        alignment of the data and disabling any SIMD capability 
        (see below).

        ``threads`` tells the wrapper how many threads to use
        when invoking FFTW, with a default of 1. If the number
        of threads is greater than 1, then the GIL is released
        by necessity.

        The currently supported schemes are as follows:

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

        \ :sup:`1`  Note that the Backwards Real transform will destroy the 
        input array. This is inherent to FFTW and the only general 
        work-around for this is to copy the array prior to performing the 
        transform.

        ``clongdouble`` typically maps directly to ``complex256``
        or ``complex192``, and ``longdouble`` to ``float128`` or ``float96``, 
        dependent on platform.

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
        arguments denotes the the unique set of axes on which we are taking
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
        DFT axes being aligned on a 16-byte boundary, which enables
        SIMD instructions. By default, if the data begins on a 16-byte 
        boundary, then FFTW will be allowed to try and enable
        SIMD instructions. This means that all future changes to
        the data arrays will be checked for similar alignment. SIMD
        instructions can be explicitly disabled by setting the
        FFTW_UNALIGNED flags, to allow for updates with unaligned
        data.

        :ref:`n_byte_align()<n_byte_align>` and 
        :ref:`n_byte_align_empty()<n_byte_align_empty>` are two methods
        included with this module for producing aligned arrays.
        '''
        pass

    def __dealloc__(self):

        if not self.__plan == NULL:
            self.__fftw_destroy(self.__plan)

        if not self.__dims == NULL:
            free(self.__dims)

        if not self.__howmany_dims == NULL:
            free(self.__howmany_dims)

    cpdef update_arrays(self, 
            new_input_array, new_output_array):
        ''' 
        Update the arrays upon which the DFT is taken.

        The new arrays should be of the same dtypes as the originals, the
        same shapes as the originals and
        should have the same strides between axes. If the original
        data was aligned so as to allow SIMD instructions (by being 
        aligned on a 16-byte boundary), then the new array
        must also be aligned in the same way.

        If all these conditions are not met, a ``ValueError`` will
        be raised and the data will *not* be updated (though the 
        object will still be in a sane state).
        
        Note that if the original array was not aligned on a 16-byte
        boundary, then SIMD is disabled and the alignment of the new
        array can be arbitrary.
        '''
        if self.__simd_allowed:
            if not (new_input_array.ctypes.data%16 == 0 and \
                    new_input_array.ctypes.data%16 == 0):
                
                raise ValueError('The original arrays were 16-byte '+\
                        'aligned. It is necessary that the update arrays '+\
                        'are similarly aligned.')

        if not new_input_array.dtype == self.__input_dtype:
            raise ValueError('The new input array is not of the same '+\
                    'dtype as was originally planned for.')

        if not new_output_array.dtype == self.__output_dtype:
            raise ValueError('The new output array is not of the same '+\
                    'dtype as was originally planned for.')

        new_input_shape = np.array(new_input_array.shape)
        new_output_shape = np.array(new_output_array.shape)
        new_input_strides = \
                np.array(new_input_array.strides)/new_input_array.itemsize
        new_output_strides = \
                np.array(new_output_array.strides)/new_output_array.itemsize

        if not (new_input_shape == self.__input_shape).all():
            raise ValueError('The new input array should be the same '+\
                    'shape as the input array used to instantiate the object.')

        if not (new_output_shape == self.__output_shape).all():
            raise ValueError('The new output array should be the same '+\
                    'shape as the output array used to instantiate the object.')
        
        if not (new_input_strides == self.__input_strides).all():
            raise ValueError('The strides should be identical for the new '+\
                    'input array as for the old.')
        
        if not (new_output_strides == self.__output_strides).all():
            raise ValueError('The strides should be identical for the new '+\
                    'output array as for the old.')

        self._update_arrays(new_input_array, new_output_array)

    cdef _update_arrays(self, 
            np.ndarray new_input_array, np.ndarray new_output_array):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self.__input_array = new_input_array
        self.__output_array = new_output_array

    cpdef execute(self):
        '''
        Execute the planned operation.
        '''
        cdef void *input_pointer = \
                <void *>np.PyArray_DATA(self.__input_array)
        cdef void *output_pointer = \
                <void *>np.PyArray_DATA(self.__output_array)
        
        cdef void *plan = self.__plan
        cdef fftw_generic_execute fftw_execute = self.__fftw_execute
        
        if self.__use_threads:
            with nogil:
                fftw_execute(plan, input_pointer, output_pointer)
        else:
            fftw_execute(self.__plan, input_pointer, output_pointer)

def export_wisdom():
    ''' export_wisdom()

    Return the FFTW wisdom as a tuple of strings.

    The first string in the tuple is the string for the double
    precision wisdom. The second string in the tuple is the string 
    for the single precision wisdom. The third string in the tuple 
    is the string for the long double precision wisdom.

    The tuple that is returned from this function can be used as the
    argument to :ref:`import_wisdom()<import_wisdom>`.
    '''

    cdef bytes py_wisdom
    cdef bytes py_wisdomf
    cdef bytes py_wisdoml

    cdef char* c_wisdom = fftw_export_wisdom_to_string()
    cdef char* c_wisdomf = fftwf_export_wisdom_to_string()
    cdef char* c_wisdoml = fftwl_export_wisdom_to_string()
    
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

    The tuple that is returned from :ref:`export_wisdom()<export_wisdom>`
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

cpdef n_byte_align_empty(shape, n, dtype='float64', order='C'):
    '''n_byte_align_empty(shape, n, dtype='float64', order='C')

    Function that returns an empty numpy array
    that is n-byte aligned.

    The alignment is given by the second argument, ``n``.
    The rest of the arguments are as per ``numpy.empty``.
    '''
    
    itemsize = np.dtype(dtype).itemsize

    # Allocate a new array that will contain the aligned data
    _array_aligned = np.empty(\
            np.prod(shape)*itemsize+n,\
            dtype='int8')
    
    # We now need to know how to offset _array_aligned 
    # so it is correctly aligned
    _array_aligned_offset = (n-_array_aligned.ctypes.data)%n

    array = np.frombuffer(\
            _array_aligned[_array_aligned_offset:_array_aligned_offset-n].data,\
            dtype=dtype).reshape(shape, order=order)
    
    return array

cpdef n_byte_align(array, n):
    ''' n_byte_align(array, n)
    Function that takes a numpy array and 
    checks it is aligned on an n-byte boundary, 
    where ``n`` is a passed parameter. If it is, 
    the array is returned without further ado. 
    If it is not, a new array is created and 
    the data copied in, but aligned on the
    n-byte boundary.
    '''
    
    if not isinstance(array, np.ndarray):
        raise TypeError('n_byte_align requires a subclass of ndarray')
    
    # See if we're already n byte aligned. If so, do nothing.
    offset = array.ctypes.data%n
    
    if offset is not 0:
        # Allocate a new array that will contain the aligned data
        _array_aligned = np.empty(\
                np.prod(array.shape)*array.itemsize+n,\
                dtype='int8')
        
        # We now need to know how to offset _array_aligned 
        # so it is correctly aligned
        _array_aligned_offset = (n-_array_aligned.ctypes.data)%n

        #if _array_aligned_offset == n:
        #    _array_aligned_offset = 0

        # Copy the data in with the correct alignment.
        # The frombuffer method was found to be the fastest
        # in various tests using the timeit module. (see
        # the blog post, 4/8/11)
        np.frombuffer(
                _array_aligned.data, dtype='int8')\
                [_array_aligned_offset:_array_aligned_offset-n]\
                = np.frombuffer(array.data, dtype='int8')[:]
        
        array = np.frombuffer(\
                _array_aligned[_array_aligned_offset:_array_aligned_offset-n].data, \
                dtype=array.dtype).reshape(array.shape).view(type=array.__class__)
    
    return array
