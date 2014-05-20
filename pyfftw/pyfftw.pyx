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
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py.MPI cimport Comm
from mpi4py cimport libmpi

import warnings

include 'utils.pxi'

# To avoid
#
# error: unknown type name ‘MPI_Message’
#    MPI_Message ob_mpi;
#
# see also https://bitbucket.org/mpi4py/mpi4py/issue/1
cdef extern from 'mpi-compat.h': pass

cdef extern from *:
    int Py_AtExit(void (*callback)())

cdef object directions
directions = {'FFTW_FORWARD': FFTW_FORWARD,
        'FFTW_BACKWARD': FFTW_BACKWARD}

cdef object directions_lookup
directions_lookup = {FFTW_FORWARD: 'FFTW_FORWARD',
        FFTW_BACKWARD: 'FFTW_BACKWARD'}

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
            <cdouble *>_in, <cdouble *>_out,
            sign, flags)

# Complex single precision
cdef void* _fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims,
            <cfloat *>_in, <cfloat *>_out,
            sign, flags)

# Complex long double precision
cdef void* _fftwl_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft(rank, dims,
            howmany_rank, howmany_dims,
            <clongdouble *>_in, <clongdouble *>_out,
            sign, flags)

# real to complex double precision
cdef void* _fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftw_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <double *>_in, <cdouble *>_out,
            flags)

# real to complex single precision
cdef void* _fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <float *>_in, <cfloat *>_out,
            flags)

# real to complex long double precision
cdef void* _fftwl_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft_r2c(rank, dims,
            howmany_rank, howmany_dims,
            <long double *>_in, <clongdouble *>_out,
            flags)

# complex to real double precision
cdef void* _fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftw_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <cdouble *>_in, <double *>_out,
            flags)

# complex to real single precision
cdef void* _fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwf_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <cfloat *>_in, <float *>_out,
            flags)

# complex to real long double precision
cdef void* _fftwl_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            void *_in, void *_out,
            int sign, int flags):

    return <void *>fftwl_plan_guru_dft_c2r(rank, dims,
            howmany_rank, howmany_dims,
            <clongdouble *>_in, <long double *>_out,
            flags)

#    Executors
#    =========
#
# Complex double precision
cdef void _fftw_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft(<fftw_plan>_plan,
            <cdouble *>_in, <cdouble *>_out)

# Complex single precision
cdef void _fftwf_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft(<fftwf_plan>_plan,
            <cfloat *>_in, <cfloat *>_out)

# Complex long double precision
cdef void _fftwl_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft(<fftwl_plan>_plan,
            <clongdouble *>_in, <clongdouble *>_out)

# real to complex double precision
cdef void _fftw_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft_r2c(<fftw_plan>_plan,
            <double *>_in, <cdouble *>_out)

# real to complex single precision
cdef void _fftwf_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft_r2c(<fftwf_plan>_plan,
            <float *>_in, <cfloat *>_out)

# real to complex long double precision
cdef void _fftwl_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft_r2c(<fftwl_plan>_plan,
            <long double *>_in, <clongdouble *>_out)

# complex to real double precision
cdef void _fftw_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftw_execute_dft_c2r(<fftw_plan>_plan,
            <cdouble *>_in, <double *>_out)

# complex to real single precision
cdef void _fftwf_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwf_execute_dft_c2r(<fftwf_plan>_plan,
            <cfloat *>_in, <float *>_out)

# complex to real long double precision
cdef void _fftwl_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwl_execute_dft_c2r(<fftwl_plan>_plan,
            <clongdouble *>_in, <long double *>_out)

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

# Planner table (of size the number of planners).
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
        (np.dtype('float64'), np.dtype('complex128')): ('r2c', '64'),
        (np.dtype('float32'), np.dtype('complex64')): ('r2c', '32'),
        (np.dtype('complex128'), np.dtype('float64')): ('c2r', '64'),
        (np.dtype('complex64'), np.dtype('float32')): ('c2r', '32')}

cdef object fftw_default_output
fftw_default_output = {
    np.dtype('float32'): np.dtype('complex64'),
    np.dtype('float64'): np.dtype('complex128'),
    np.dtype('complex64'): np.dtype('complex64'),
    np.dtype('complex128'): np.dtype('complex128')}

if np.dtype('longdouble') != np.dtype('float64'):
    fftw_schemes.update({
        (np.dtype('clongdouble'), np.dtype('clongdouble')): ('c2c', 'ld'),
        (np.dtype('longdouble'), np.dtype('clongdouble')): ('r2c', 'ld'),
        (np.dtype('clongdouble'), np.dtype('longdouble')): ('c2r', 'ld')})

    fftw_default_output.update({
        np.dtype('longdouble'): np.dtype('clongdouble'),
        np.dtype('clongdouble'): np.dtype('clongdouble')})

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

cdef mpi_init():
    fftw_mpi_init()
    fftwf_mpi_init()
    fftwl_mpi_init()

    _build_distributor_list()
    _build_mpi_executor_list()
    _build_mpi_planner_list()
    _build_mpi_wisdom_list()

# Set the cleanup routine
cdef void _cleanup():
    # TODO tight coupling with non-MPI code
    # mpi_cleanup() includes serial clean up
    fftw_mpi_cleanup()
    fftwf_mpi_cleanup()
    fftwl_mpi_cleanup()

    fftw_cleanup()
    fftwf_cleanup()
    fftwl_cleanup()
    fftw_cleanup_threads()
    fftwf_cleanup_threads()
    fftwl_cleanup_threads()

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

# TODO tight coupling with non-MPI code
mpi_init()

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
    cdef fftw_generic_plan_guru _fftw_planner
    cdef fftw_generic_execute _fftw_execute
    cdef fftw_generic_destroy_plan _fftw_destroy
    cdef fftw_generic_plan_with_nthreads _nthreads_plan_setter

    # The plan is typecast when it is created or used
    # within the wrapper functions
    cdef void *_plan

    cdef np.ndarray _input_array
    cdef np.ndarray _output_array
    cdef int _direction
    cdef int _flags

    cdef bint _simd_allowed
    cdef int _input_array_alignment
    cdef int _output_array_alignment
    cdef bint _use_threads

    cdef object _input_item_strides
    cdef object _input_strides
    cdef object _output_item_strides
    cdef object _output_strides
    cdef object _input_shape
    cdef object _output_shape
    cdef object _input_dtype
    cdef object _output_dtype
    cdef object _flags_used

    cdef double _normalisation_scaling

    cdef int _rank
    cdef _fftw_iodim *_dims
    cdef int _howmany_rank
    cdef _fftw_iodim *_howmany_dims

    cdef int64_t *_axes
    cdef int64_t *_not_axes

    cdef int64_t _N
    def _get_N(self):
        '''
        The product of the lengths of the DFT over all DFT axes.
        1/N is the normalisation constant. For any input array A,
        and for any set of axes, 1/N * ifft(fft(A)) = A
        '''
        return self._N

    N = property(_get_N)

    def _get_simd_aligned(self):
        '''
        Return whether or not this FFTW object requires simd aligned
        input and output data.
        '''
        return self._simd_allowed

    simd_aligned = property(_get_simd_aligned)

    def _get_input_alignment(self):
        '''
        Returns the byte alignment of the input arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Input array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised, or
        a copy being made if the :meth:`~pyfftw.FFTW.__call__`
        interface is used.
        '''
        return self._input_array_alignment

    input_alignment = property(_get_input_alignment)

    def _get_output_alignment(self):
        '''
        Returns the byte alignment of the output arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Output array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised.
        '''
        return self._output_array_alignment

    output_alignment = property(_get_output_alignment)

    def _get_flags_used(self):
        '''
        Return which flags were used to construct the FFTW object.

        This includes flags that were added during initialisation.
        '''
        return tuple(self._flags_used)

    flags = property(_get_flags_used)

    def _get_input_array(self):
        '''
        Return the input array that is associated with the FFTW
        instance.
        '''
        return self._input_array

    input_array = property(_get_input_array)

    def _get_output_array(self):
        '''
        Return the output array that is associated with the FFTW
        instance.
        '''
        return self._output_array

    output_array = property(_get_output_array)

    def _get_input_strides(self):
        '''
        Return the strides of the input array for which the FFT is planned.
        '''
        return self._input_strides

    input_strides = property(_get_input_strides)

    def _get_output_strides(self):
        '''
        Return the strides of the output array for which the FFT is planned.
        '''
        return self._output_strides

    output_strides = property(_get_output_strides)

    def _get_input_shape(self):
        '''
        Return the shape of the input array for which the FFT is planned.
        '''
        return self._input_shape

    input_shape = property(_get_input_shape)

    def _get_output_shape(self):
        '''
        Return the shape of the output array for which the FFT is planned.
        '''
        return self._output_shape

    output_shape = property(_get_output_shape)

    def _get_input_dtype(self):
        '''
        Return the dtype of the input array for which the FFT is planned.
        '''
        return self._input_dtype

    input_dtype = property(_get_input_dtype)

    def _get_output_dtype(self):
        '''
        Return the shape of the output array for which the FFT is planned.
        '''
        return self._output_dtype

    output_dtype = property(_get_output_dtype)

    def _get_direction(self):
        '''
        Return the planned FFT direction. Either `'FFTW_FORWARD'` or
        `'FFTW_BACKWARD'`.
        '''
        return directions_lookup[self._direction]

    direction = property(_get_direction)

    def _get_axes(self):
        '''
        Return the axes for the planned FFT in canonical form. That is, as
        a tuple of positive integers. The order in which they were passed
        is maintained.
        '''
        axes = []
        for i in range(self._rank):
            axes.append(self._axes[i])

        return tuple(axes)

    axes = property(_get_axes)

    def __cinit__(self, input_array, output_array, axes=(-1,),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
            unsigned int threads=1, planning_timelimit=None, comm=None,
            *args, **kwargs):

        # Initialise the pointers that need to be freed
        self._plan = NULL
        self._dims = NULL
        self._howmany_dims = NULL

        self._axes = NULL
        self._not_axes = NULL

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

        self._input_dtype = input_dtype
        self._output_dtype = output_dtype

        functions = scheme_functions[scheme]

        self._fftw_planner = planners[functions['planner']]
        self._fftw_execute = executors[functions['executor']]
        self._fftw_destroy = destroyers[functions['generic_precision']]

        self._nthreads_plan_setter = (
                nthreads_plan_setters[functions['generic_precision']])

        cdef fftw_generic_set_timelimit set_timelimit_func = (
                set_timelimit_funcs[functions['generic_precision']])

        # We're interested in the natural alignment on the real type, not
        # necessarily on the complex type At least one bug was found where
        # numpy reported an alignment on a complex dtype that was different
        # to that on the real type.
        cdef int natural_input_alignment = input_array.real.dtype.alignment
        cdef int natural_output_alignment = output_array.real.dtype.alignment

        # If either of the arrays is not aligned on a 16-byte boundary,
        # we set the FFTW_UNALIGNED flag. This disables SIMD.
        # (16 bytes is assumed to be the minimal alignment)
        if 'FFTW_UNALIGNED' in flags:
            self._simd_allowed = False
            self._input_array_alignment = natural_input_alignment
            self._output_array_alignment = natural_output_alignment

        else:

            self._input_array_alignment = -1
            self._output_array_alignment = -1

            for each_alignment in _valid_simd_alignments:
                if (<intptr_t>np.PyArray_DATA(input_array) %
                        each_alignment == 0 and
                        <intptr_t>np.PyArray_DATA(output_array) %
                        each_alignment == 0):

                    self._simd_allowed = True

                    self._input_array_alignment = each_alignment
                    self._output_array_alignment = each_alignment

                    break

            if (self._input_array_alignment == -1 or
                    self._output_array_alignment == -1):

                self._simd_allowed = False

                self._input_array_alignment = (
                        natural_input_alignment)
                self._output_array_alignment = (
                        natural_output_alignment)
                flags.append('FFTW_UNALIGNED')

        if (not (<intptr_t>np.PyArray_DATA(input_array)
            % self._input_array_alignment == 0)):
            raise ValueError('Invalid input alignment: '
                    'The input array is expected to lie on a %d '
                    'byte boundary.' % self._input_array_alignment)

        if (not (<intptr_t>np.PyArray_DATA(output_array)
            % self._output_array_alignment == 0)):
            raise ValueError('Invalid output alignment: '
                    'The output array is expected to lie on a %d '
                    'byte boundary.' % self._output_array_alignment)

        if not direction in scheme_directions[scheme]:
            raise ValueError('Invalid direction: '
                    'The direction is not valid for the scheme. '
                    'Try setting it explicitly if it is not already.')

        self._direction = directions[direction]
        self._input_shape = input_array.shape
        self._output_shape = output_array.shape

        self._input_array = input_array
        self._output_array = output_array

        self._axes = <int64_t *>malloc(len(axes)*sizeof(int64_t))
        for n in range(len(axes)):
            self._axes[n] = axes[n]

        # Set the negative entries to their actual index (use the size
        # of the shape array for this)
        cdef int64_t array_dimension = len(self._input_shape)

        for n in range(len(axes)):
            if self._axes[n] < 0:
                self._axes[n] = self._axes[n] + array_dimension

            if self._axes[n] >= array_dimension or self._axes[n] < 0:
                raise IndexError('Invalid axes: '
                    'The axes list cannot contain invalid axes.')

        cdef int64_t unique_axes_length
        cdef int64_t *unique_axes
        cdef int64_t *not_axes

        make_axes_unique(self._axes, len(axes), &unique_axes,
                &not_axes, array_dimension, &unique_axes_length)

        # and assign axes and not_axes to the filled arrays
        free(self._axes)
        self._axes = unique_axes
        self._not_axes = not_axes

        total_N = 1
        for n in range(unique_axes_length):
            if self._input_shape[self._axes[n]] == 0:
                raise ValueError('Zero length array: '
                    'The input array should have no zero length'
                    'axes over which the FFT is to be taken')

            if self._direction == FFTW_FORWARD:
                total_N *= self._input_shape[self._axes[n]]
            else:
                total_N *= self._output_shape[self._axes[n]]

        self._N = total_N
        self._normalisation_scaling = 1/float(self.N)

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
                    self._axes, self._not_axes, unique_axes_length):
                raise ValueError('Invalid shapes: '
                        'The input array and output array are invalid '
                        'complementary shapes for their dtypes.')

        self._rank = unique_axes_length
        self._howmany_rank = self._input_array.ndim - unique_axes_length

        self._flags = 0
        self._flags_used = []
        for each_flag in flags:
            try:
                self._flags |= flag_dict[each_flag]
                self._flags_used.append(each_flag)
            except KeyError:
                raise ValueError('Invalid flag: ' + '\'' +
                        each_flag + '\' is not a valid planner flag.')


        if ('FFTW_DESTROY_INPUT' not in flags) and (
                (scheme[0] != 'c2r') or not self._rank > 1):
            # The default in all possible cases is to preserve the input
            # This is not possible for r2c arrays with rank > 1
            self._flags |= FFTW_PRESERVE_INPUT

        # Set up the arrays of structs for holding the stride shape
        # information
        self._dims = <_fftw_iodim *>malloc(
                self._rank * sizeof(_fftw_iodim))
        self._howmany_dims = <_fftw_iodim *>malloc(
                self._howmany_rank * sizeof(_fftw_iodim))

        if self._dims == NULL or self._howmany_dims == NULL:
            # Not much else to do than raise an exception
            raise MemoryError

        # Find the strides for all the axes of both arrays in terms of the
        # number of items (as opposed to the number of bytes).
        self._input_strides = input_array.strides
        self._input_item_strides = tuple([stride/input_array.itemsize
            for stride in input_array.strides])
        self._output_strides = output_array.strides
        self._output_item_strides = tuple([stride/output_array.itemsize
            for stride in output_array.strides])

        # Make sure that the arrays are not too big for fftw
        # This is hard to test, so we cross our fingers and hope for the
        # best (any suggestions, please get in touch).
        cdef int i
        for i in range(0, len(self._input_shape)):
            if self._input_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the input array must be ' +
                        'less than ', str(limits.INT_MAX))

            if self._input_item_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the input array must be ' +
                        'less than ', str(limits.INT_MAX))

        for i in range(0, len(self._output_shape)):
            if self._output_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the output array must be ' +
                        'less than ', str(limits.INT_MAX))

            if self._output_item_strides[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Strides of the output array must be ' +
                        'less than ', str(limits.INT_MAX))

        fft_shape_lookup = functions['fft_shape_lookup']
        if fft_shape_lookup == -1:
            fft_shape = self._input_shape
        else:
            fft_shape = fft_shape_lookup(input_array, output_array)

        # Fill in the stride and shape information
        input_strides_array = self._input_item_strides
        output_strides_array = self._output_item_strides
        for i in range(0, self._rank):
            self._dims[i]._n = fft_shape[self._axes[i]]
            self._dims[i]._is = input_strides_array[self._axes[i]]
            self._dims[i]._os = output_strides_array[self._axes[i]]

        for i in range(0, self._howmany_rank):
            self._howmany_dims[i]._n = fft_shape[self._not_axes[i]]
            self._howmany_dims[i]._is = input_strides_array[self._not_axes[i]]
            self._howmany_dims[i]._os = output_strides_array[self._not_axes[i]]

        # parallel execution
        self._use_threads = (threads > 1)

        ## Point at which FFTW calls are made
        ## (and none should be made before this)
        self._nthreads_plan_setter(threads)

        # Set the timelimit
        set_timelimit_func(_planning_timelimit)

        # Finally, construct the plan
        self._plan = self._fftw_planner(
            self._rank, <fftw_iodim *>self._dims,
            self._howmany_rank, <fftw_iodim *>self._howmany_dims,
            <void *>np.PyArray_DATA(self._input_array),
            <void *>np.PyArray_DATA(self._output_array),
            self._direction, self._flags)

        if self._plan == NULL:
            raise RuntimeError('The data has an uncaught error that led '+
                    'to the planner returning NULL. This is a bug.')

    def __init__(self, input_array, output_array, axes=(-1,),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
            int threads=1, planning_timelimit=None, comm=None,
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

        if not self._axes == NULL:
            free(self._axes)

        if not self._not_axes == NULL:
            free(self._not_axes)

        if not self._plan == NULL:
            self._fftw_destroy(self._plan)

        if not self._dims == NULL:
            free(self._dims)

        if not self._howmany_dims == NULL:
            free(self._howmany_dims)

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
                input_array = self._input_array

            if output_array is None:
                output_array = self._output_array

            if not isinstance(input_array, np.ndarray):
                copy_needed = True
            elif (not input_array.dtype == self._input_dtype):
                copy_needed = True
            elif (not input_array.strides == self._input_strides):
                copy_needed = True
            elif not (<intptr_t>np.PyArray_DATA(input_array)
                    % self.input_alignment == 0):
                copy_needed = True
            else:
                copy_needed = False

            if copy_needed:

                if not isinstance(input_array, np.ndarray):
                    input_array = np.asanyarray(input_array)

                if not input_array.shape == self._input_shape:
                    raise ValueError('Invalid input shape: '
                            'The new input array should be the same shape '
                            'as the input array used to instantiate the '
                            'object.')

                self._input_array[:] = input_array

                if output_array is not None:
                    # No point wasting time if no update is necessary
                    # (which the copy above may have avoided)
                    input_array = self._input_array
                    self.update_arrays(input_array, output_array)

            else:
                self.update_arrays(input_array, output_array)

        self.execute()

        if self._direction == FFTW_BACKWARD and normalise_idft:
            self._output_array *= self._normalisation_scaling

        return self._output_array

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

        if not new_input_array.dtype == self._input_dtype:
            raise ValueError('Invalid input dtype: '
                    'The new input array is not of the same '
                    'dtype as was originally planned for.')

        if not new_output_array.dtype == self._output_dtype:
            raise ValueError('Invalid output dtype: '
                    'The new output array is not of the same '
                    'dtype as was originally planned for.')

        new_input_shape = new_input_array.shape
        new_output_shape = new_output_array.shape

        new_input_strides = new_input_array.strides
        new_output_strides = new_output_array.strides

        if not new_input_shape == self._input_shape:
            raise ValueError('Invalid input shape: '
                    'The new input array should be the same shape as '
                    'the input array used to instantiate the object.')

        if not new_output_shape == self._output_shape:
            raise ValueError('Invalid output shape: '
                    'The new output array should be the same shape as '
                    'the output array used to instantiate the object.')

        if not new_input_strides == self._input_strides:
            raise ValueError('Invalid input striding: '
                    'The strides should be identical for the new '
                    'input array as for the old.')

        if not new_output_strides == self._output_strides:
            raise ValueError('Invalid output striding: '
                    'The strides should be identical for the new '
                    'output array as for the old.')

        self._update_arrays(new_input_array, new_output_array)

    cdef _update_arrays(self,
            np.ndarray new_input_array, np.ndarray new_output_array):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self._input_array = new_input_array
        self._output_array = new_output_array

    def get_input_array(self):
        '''get_input_array()

        Return the input array that is associated with the FFTW
        instance.

        *Deprecated since 0.10. Consider using the* :attr:`FFTW.input_array`
        *property instead.*
        '''
        warnings.warn('get_input_array is deprecated. '
                'Consider using the input_array property instead.',
                DeprecationWarning)

        return self._input_array

    def get_output_array(self):
        '''get_output_array()

        Return the output array that is associated with the FFTW
        instance.

        *Deprecated since 0.10. Consider using the* :attr:`FFTW.output_array`
        *property instead.*
        '''
        warnings.warn('get_output_array is deprecated. '
                'Consider using the output_array property instead.',
                DeprecationWarning)

        return self._output_array

    cpdef execute(self):
        '''execute()

        Execute the planned operation, taking the correct kind of FFT of
        the input array (i.e. :attr:`FFTW.input_array`),
        and putting the result in the output array (i.e.
        :attr:`FFTW.output_array`).
        '''
        cdef void *input_pointer = (
                <void *>np.PyArray_DATA(self._input_array))
        cdef void *output_pointer = (
                <void *>np.PyArray_DATA(self._output_array))

        cdef void *plan = self._plan
        cdef fftw_generic_execute fftw_execute = self._fftw_execute

        if self._use_threads:
            with nogil:
                fftw_execute(plan, input_pointer, output_pointer)
        else:
            fftw_execute(self._plan, input_pointer, output_pointer)

# MPI
# ======================
#
import mpi4py

# adapter functions with uniform set of arguments
# but covariant return type

# (d>1) dimensional
cdef  _fftw_mpi_local_size_many(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
            ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size = fftw_mpi_local_size_many(rank, n, howmany,
                                                         block0, comm,
                                                         local_n0, local_0_start)

    return local_size, local_n0[0], local_0_start[0]

# (d>1) dimensional transposed
cdef _fftw_mpi_local_size_many_transposed(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
            ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size = fftw_mpi_local_size_many_transposed(
           rank, n, howmany,
           block0, block1, comm,
           local_n0, local_0_start,
           local_n1, local_1_start)
    return local_size, local_n0[0], local_0_start[0], local_n1[0], local_1_start[0]

# d=1
cdef object _fftw_mpi_local_size_many_1d(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
            ptrdiff_t *local_no, ptrdiff_t *local_o_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size = fftw_mpi_local_size_many_1d(
           n[0], howmany,
           comm, sign, flags,
           local_ni, local_i_start,
           local_no, local_o_start)

    return local_size, local_ni[0], local_i_start[0], local_no[0], local_o_start[0]

#     Planners
#     ========
#
# Complex double precision
cdef void* _fftw_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                   ptrdiff_t block0, ptrdiff_t block1,
                                   void * _in, void * _out,
                                   MPI_Comm comm,
                                   int sign, unsigned int flags):

    return <void *> fftw_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                           <cdouble *> _in, <cdouble *> _out,
                                           comm, sign, flags)

# Complex single precision
cdef void* _fftwf_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                    ptrdiff_t block0, ptrdiff_t block1,
                                    void * _in, void * _out,
                                    MPI_Comm comm,
                                    int sign, unsigned int flags):

    return <void *> fftwf_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                            <cfloat *> _in, <cfloat *> _out,
                                            comm, sign, flags)

# Complex long double precision
cdef void* _fftwl_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                    ptrdiff_t block0, ptrdiff_t block1,
                                    void * _in, void * _out,
                                    MPI_Comm comm,
                                    int sign, unsigned int flags):

    return <void *> fftwl_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                            <clongdouble *> _in, <clongdouble *> _out,
                                            comm, sign, flags)

# real to complex double precision
cdef void* _fftw_mpi_plan_many_dft_r2c(int rank, const ptrdiff_t *n,
                                       ptrdiff_t howmany,
                                       ptrdiff_t block0, ptrdiff_t block1,
                                       void * _in, void * _out,
                                       MPI_Comm comm,
                                       int sign, unsigned int flags):

    return <void *> fftw_mpi_plan_many_dft_r2c(rank, n, howmany, block0, block1,
                                               <double *> _in, <cdouble *> _out,
                                               comm, flags)

# real to complex single precision
cdef void* _fftwf_mpi_plan_many_dft_r2c(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwf_mpi_plan_many_dft_r2c(rank, n, howmany, block0, block1,
                                                <float *> _in, <cfloat *> _out,
                                                comm, flags)

# real to complex long double precision
cdef void* _fftwl_mpi_plan_many_dft_r2c(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwl_mpi_plan_many_dft_r2c(rank, n, howmany, block0, block1,
                                                <long double *> _in,
                                                <clongdouble *> _out,
                                                comm, flags)

# complex to real double precision
cdef void* _fftw_mpi_plan_many_dft_c2r(int rank, const ptrdiff_t *n,
                                       ptrdiff_t howmany,
                                       ptrdiff_t block0, ptrdiff_t block1,
                                       void * _in, void * _out,
                                       MPI_Comm comm,
                                       int sign, unsigned int flags):

    return <void *> fftw_mpi_plan_many_dft_c2r(rank, n, howmany, block0, block1,
                                               <cdouble *> _in, <double *> _out,
                                               comm, flags)

# complex to real single precision
cdef void* _fftwf_mpi_plan_many_dft_c2r(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwf_mpi_plan_many_dft_c2r(rank, n, howmany, block0, block1,
                                                <cfloat *> _in, <float *> _out,
                                                comm, flags)

# complex to real long double precision
cdef void* _fftwl_mpi_plan_many_dft_c2r(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwl_mpi_plan_many_dft_c2r(rank, n, howmany, block0, block1,
                                                <clongdouble *> _in,
                                                <long double *> _out,
                                                comm, flags)

# transpose double
cdef void* _fftw_mpi_plan_many_transpose(int rank, const ptrdiff_t *n,
                                         ptrdiff_t howmany,
                                         ptrdiff_t block0, ptrdiff_t block1,
                                         void * _in, void * _out,
                                         MPI_Comm comm,
                                         int sign, unsigned int flags):

    return <void *> fftw_mpi_plan_many_transpose(n[0], n[1], howmany,
                                                 block0, block1,
                                                 <double *> _in, <double *> _out,
                                                 comm, flags)

# transpose float
cdef void* _fftwf_mpi_plan_many_transpose(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwf_mpi_plan_many_transpose(n[0], n[1], howmany,
                                                 block0, block1,
                                                 <float *> _in, <float *> _out,
                                                 comm, flags)

# transpose long double
cdef void* _fftwl_mpi_plan_many_transpose(int rank, const ptrdiff_t *n,
                                        ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

    return <void *> fftwl_mpi_plan_many_transpose(n[0], n[1], howmany,
                                                  block0, block1,
                                                  <long double *> _in,
                                                  <long double *> _out,
                                                  comm, flags)

#    Executors
#    =========
#
# Complex double precision
cdef void _fftw_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftw_mpi_execute_dft(<fftw_plan>_plan,
            <cdouble *>_in, <cdouble *>_out)

# Complex single precision
cdef void _fftwf_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwf_mpi_execute_dft(<fftwf_plan>_plan,
            <cfloat *>_in, <cfloat *>_out)

# Complex long double precision
cdef void _fftwl_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

    fftwl_mpi_execute_dft(<fftwl_plan>_plan,
            <clongdouble *>_in, <clongdouble *>_out)

# real to complex double precision
cdef void _fftw_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftw_mpi_execute_dft_r2c(<fftw_plan>_plan,
            <double *>_in, <cdouble *>_out)

# real to complex single precision
cdef void _fftwf_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwf_mpi_execute_dft_r2c(<fftwf_plan>_plan,
            <float *>_in, <cfloat *>_out)

# real to complex long double precision
cdef void _fftwl_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

    fftwl_mpi_execute_dft_r2c(<fftwl_plan>_plan,
            <long double *>_in, <clongdouble *>_out)

# complex to real double precision
cdef void _fftw_mpi_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftw_mpi_execute_dft_c2r(<fftw_plan>_plan,
            <cdouble *>_in, <double *>_out)

# complex to real single precision
cdef void _fftwf_mpi_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwf_mpi_execute_dft_c2r(<fftwf_plan>_plan,
            <cfloat *>_in, <float *>_out)

# complex to real long double precision
cdef void _fftwl_mpi_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

    fftwl_mpi_execute_dft_c2r(<fftwl_plan>_plan,
            <clongdouble *>_in, <long double *>_out)

# Validator functions
# ===================
def _mpi_validate_array(local_array, size, name):
    '''Validate that ``local_array`` (input or output) has the right size and is contiguous.

    '''
    # arrays must have a minimum size, whether it is in-place or not
    if local_array.size < size:
        raise ValueError('Invalid %s array size. Expected at least %d, got %d' %
                         (name, size, local_array.size))

    # needs to be contiguous, FFTW imposes its own ordering
    if not (local_array.flags['C_CONTIGUOUS'] or local_array.flags['F_CONTIGUOUS']):
        raise ValueError('%d array is not contiguous' % name)

def fftw_mpi_1D_unsupported(msg):
    raise NotImplementedError('FFTW with MPI does not support ' + msg +
                              ' in 1D. Use a complex-to-complex transform'
                              ' and ignore the relevant imaginary part!')

# Shape lookup functions
# ======================
def _mpi_local_size_input_shape(input_shape, scheme):
    '''Return the shape of (complex) input dimensions to feed into ``local_size`` as
    a tuple. It coincides with ``input_shape`` except if the ``scheme`` is
    ``r2c``, in which the last dimension is converted from ``N`` to ``N / 2 +
    1`` to account for the Hermitian symmetry.

    '''
    # need padding for r2c; cf. FFTW manual 6.5 'Multi-dimensional MPI DFTs of Real Data'
    if scheme == 'r2c' or scheme == 'c2r':
        res = np.array(input_shape)
        res[-1] = res[-1] // 2 + 1
        return tuple(res)
    else:
        return input_shape

def tuple_or_None(myfunc):
    '''Decorator to a function that returns two iterables.

    It return two values, either a tuple or ``None``. Both of
    ``myfunc``'s return values are checked for their first element, if
    it is 0, the iterable is converted to ``None``, else it is turned
    into a tuple.

    '''
    def new_function(*args, **kwargs):
        i,o = myfunc(*args, **kwargs)
        i = None if i[0] == 0 else tuple(i)
        o = None if o[0] == 0 else tuple(o)
        return i, o

    return new_function

def _mpi_local_input_output_shape_nD(input_shape, local_size_result, last_in, last_out, flags):
    '''Generic function to compute both the input and output shape of the local
    arrays on this MPI rank as arrays. Note that this may give zero-sized arrays
    if no input/output on this rank.

    '''
    shapes = (np.array(input_shape), np.array(input_shape))
    for s, last in zip(shapes, (last_in, last_out)):
        # local portion along first dimension
        s[0] = local_size_result[1]
        s[-1] = last

    # heed transposition
    for i, flag in enumerate(('FFTW_MPI_TRANSPOSED_IN', 'FFTW_MPI_TRANSPOSED_OUT')):
        if flag in flags:
            assert len(local_size_result) == 5, 'Internal error: call local_size_transposed'
            shapes[i][0] = local_size_result[3]
            shapes[i][1] = input_shape[0]

    # no output on this rank. Possible if work small and not divisible
    # by number of processors
    # if shapes[i][0] == 0:
    #     shapes[i] = None

    return shapes

@tuple_or_None
def _mpi_local_input_output_shape_r2c(input_shape, local_size_result, flags):
    if len(input_shape) > 1:
        # Hermitian symmetry in last dimension
        last_in  = input_shape[-1]
        last_out = input_shape[-1] // 2 + 1
        return _mpi_local_input_output_shape_nD(input_shape, local_size_result,
                                                last_in, last_out, flags)
    else:
        fftw_mpi_1D_unsupported('r2c')

@tuple_or_None
def _mpi_local_input_output_shape_c2r(input_shape, local_size_result, flags):
    if len(input_shape) > 1:
        # Hermitian symmetry in last dimension
        last_in  = input_shape[-1] // 2 + 1
        last_out = input_shape[-1]
        return _mpi_local_input_output_shape_nD(input_shape, local_size_result,
                                                last_in, last_out, flags)
    else:
         fftw_mpi_1D_unsupported('c2r')

@tuple_or_None
def _mpi_local_input_output_shape_c2c(input_shape, local_size_result, flags):
    if len(input_shape) > 1:
        return _mpi_local_input_output_shape_nD(input_shape, local_size_result,
                                                input_shape[-1], input_shape[-1], flags)
    else:
        return (local_size_result[1],), (local_size_result[3],)

def _mpi_output_shape_r2c(input_shape):
    '''Conceptual output shape for global array.'''
    output_shape = np.array(input_shape)
    output_shape[-1] = input_shape[-1] // 2 + 1
    return tuple(output_shape)

def _mpi_output_shape_c2c(input_shape):
    '''Conceptual output shape for global array. Simply the input for c2r and c2c.

    '''
    return input_shape

@tuple_or_None
def _mpi_local_input_output_shape_padded_r2c(input_shape, local_size_result, flags):
    if len(input_shape) > 1:
        last_in  = input_shape[-1]
        last_out = input_shape[-1] // 2 + 1
        i,o = _mpi_local_input_output_shape_nD(input_shape, local_size_result,
                                               last_in, last_out, flags)
        # always twice the size to allow in-place transform
        if 'FFTW_MPI_TRANSPOSED_OUT' in flags:
            i[-1] = 2 * last_out
        else:
            i[-1] = 2 * o[-1]
        return i, o
    else:
        fftw_mpi_1D_unsupported('r2c')

@tuple_or_None
def _mpi_local_input_output_shape_padded_c2r(input_shape, local_size_result, flags):
    if len(input_shape) > 1:
        # padding in last dimension
        last_in  = input_shape[-1] // 2 + 1
        last_out = input_shape[-1]
        # assume padded input, so output just same storage but real instead of complex
        i, o = _mpi_local_input_output_shape_nD(input_shape, local_size_result,
                                             last_in, last_out, flags)

        # always twice the size to allow in-place transform
        o[-1] = 2 * i[-1]
        return i, o
    else:
        fftw_mpi_1D_unsupported('c2r')

# Function lookup tables
# ======================

cdef fftw_mpi_generic_local_size distributors[3]

cdef fftw_mpi_generic_local_size * _build_distributor_list():

    distributors[0] = <fftw_mpi_generic_local_size> &_fftw_mpi_local_size_many
    distributors[1] = <fftw_mpi_generic_local_size> &_fftw_mpi_local_size_many_transposed
    distributors[2] = <fftw_mpi_generic_local_size> &_fftw_mpi_local_size_many_1d

# Planner table (of size the number of planners).
cdef fftw_mpi_generic_plan mpi_planners[12]

# TODO When to call the builder?
cdef fftw_mpi_generic_plan * _build_mpi_planner_list():
    mpi_planners[0]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft
    mpi_planners[1]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft
    mpi_planners[2]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft
    mpi_planners[3]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft_r2c
    mpi_planners[4]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft_r2c
    mpi_planners[5]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft_r2c
    mpi_planners[6]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft_c2r
    mpi_planners[7]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft_c2r
    mpi_planners[8]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft_c2r
    mpi_planners[9]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_transpose
    mpi_planners[10] = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_transpose
    mpi_planners[11] = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_transpose

# Executor table (of size the number of executors)
cdef fftw_generic_execute mpi_executors[9]

cdef fftw_generic_execute * _build_mpi_executor_list():

    mpi_executors[0] = <fftw_generic_execute>&_fftw_mpi_execute_dft
    mpi_executors[1] = <fftw_generic_execute>&_fftwf_mpi_execute_dft
    mpi_executors[2] = <fftw_generic_execute>&_fftwl_mpi_execute_dft
    mpi_executors[3] = <fftw_generic_execute>&_fftw_mpi_execute_dft_r2c
    mpi_executors[4] = <fftw_generic_execute>&_fftwf_mpi_execute_dft_r2c
    mpi_executors[5] = <fftw_generic_execute>&_fftwl_mpi_execute_dft_r2c
    mpi_executors[6] = <fftw_generic_execute>&_fftw_mpi_execute_dft_c2r
    mpi_executors[7] = <fftw_generic_execute>&_fftwf_mpi_execute_dft_c2r
    mpi_executors[8] = <fftw_generic_execute>&_fftwl_mpi_execute_dft_c2r

cdef fftw_mpi_generic_wisdom mpi_wisdom[6]

# TODO doesn't have a return value?
cdef fftw_mpi_generic_wisdom * _build_mpi_wisdom_list():

    mpi_wisdom[0] = <fftw_mpi_generic_wisdom> &fftw_mpi_gather_wisdom
    mpi_wisdom[1] = <fftw_mpi_generic_wisdom> &fftwf_mpi_gather_wisdom
    mpi_wisdom[2] = <fftw_mpi_generic_wisdom> &fftwl_mpi_gather_wisdom
    mpi_wisdom[3] = <fftw_mpi_generic_wisdom> &fftw_mpi_broadcast_wisdom
    mpi_wisdom[4] = <fftw_mpi_generic_wisdom> &fftwf_mpi_broadcast_wisdom
    mpi_wisdom[5] = <fftw_mpi_generic_wisdom> &fftwl_mpi_broadcast_wisdom

cdef object mpi_flag_dict
mpi_flag_dict = {'FFTW_MPI_DEFAULT_BLOCK': FFTW_MPI_DEFAULT_BLOCK,
                 'FFTW_MPI_SCRAMBLED_IN': FFTW_MPI_SCRAMBLED_IN,
                 'FFTW_MPI_SCRAMBLED_OUT': FFTW_MPI_SCRAMBLED_OUT,
                 'FFTW_MPI_TRANSPOSED_IN': FFTW_MPI_TRANSPOSED_IN,
                 'FFTW_MPI_TRANSPOSED_OUT': FFTW_MPI_TRANSPOSED_OUT}

_mpi_flag_dict = mpi_flag_dict.copy()

# look up shapes of local IO arrays
cdef object mpi_local_shapes
mpi_local_shapes = [_mpi_local_input_output_shape_r2c,
                    _mpi_local_input_output_shape_c2r,
                    _mpi_local_input_output_shape_c2c]

cdef object mpi_local_shapes_padded
mpi_local_shapes_padded = [_mpi_local_input_output_shape_padded_r2c,
                           _mpi_local_input_output_shape_padded_c2r,
                           _mpi_local_input_output_shape_c2c]

# look up conceptual global output shape
cdef object mpi_output_shape
mpi_output_shape = [_mpi_output_shape_r2c,
                    # yes, c2r shape just input shape
                    _mpi_output_shape_c2c,
                    _mpi_output_shape_c2c]

def n_elements_in_out(n, scheme):
    '''Return the number of elements to allocate for input and output given ``n``,
    the number of complex elements to allocate, and the transformation
    ``scheme``.

    '''
    nin = nout =  n
    # nout = n
    if scheme == 'r2c':
        nin *= 2
    elif scheme == 'c2r':
        nout *= 2

    return nin, nout

cdef object mpi_scheme_functions
mpi_scheme_functions = {
    ('c2c', '64'): {'planner': 0, 'executor':0, 'generic_precision':0,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('c2c', '32'): {'planner':1, 'executor':1, 'generic_precision':1,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('c2c', 'ld'): {'planner':2, 'executor':2, 'generic_precision':2,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('r2c', '64'): {'planner':3, 'executor':3, 'generic_precision':0,
                    'validator': 0, 'fft_shape_lookup': 0,
                    'output_shape': 0},
    ('r2c', '32'): {'planner':4, 'executor':4, 'generic_precision':1,
                    'validator': 0, 'fft_shape_lookup':  0,
                    'output_shape': 0},
    ('r2c', 'ld'): {'planner':5, 'executor':5, 'generic_precision':2,
                    'validator': 0, 'fft_shape_lookup':  0,
                    'output_shape': 0},
    ('c2r', '64'): {'planner':6, 'executor':6, 'generic_precision':0,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1},
    ('c2r', '32'): {'planner':7, 'executor':7, 'generic_precision':1,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1},
    ('c2r', 'ld'): {'planner':8, 'executor':8, 'generic_precision':2,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1}}

cdef class IntegerArray:
    '''Wrapper around a small chunk of memory.'''
    cdef ptrdiff_t* data

    def __cinit__(self, number):
        # allocate some memory (filled with random data)
        self.data = <ptrdiff_t*> PyMem_Malloc(number * sizeof(ptrdiff_t))
        if not self.data:
            raise MemoryError()

    def resize(self, new_number):
        # Allocates new_number * sizeof(ptrdiff_t) bytes,
        # preserving the contents and making a best-effort to
        # re-use the original data location.
        mem = <ptrdiff_t*> PyMem_Realloc(self.data, new_number * sizeof(ptrdiff_t))
        if not mem:
            raise MemoryError()
        # Only overwrite the pointer if the memory was really reallocated.
        # On error (mem is NULL), the original memory has not been freed.
        self.data = mem

    def __dealloc__(self):
        PyMem_Free(self.data)     # no-op if self.data is NULL

cdef libmpi.MPI_Comm extract_communicator(Comm comm=None):
    '''Extract the C struct from a mpi4py.MPI.Comm object. Default to
    COMM_WORLD.

    '''
    if comm is None:
        return libmpi.MPI_COMM_WORLD

    # see mpi4py/src/MPI/Comm.pyx and mpi4py/demo/cython/helloworld.pyx
    cdef libmpi.MPI_Comm _comm = comm.ob_mpi
    if _comm is NULL:
        raise ValueError('Invalid MPI communicator')
    return _comm

cdef object validate_mpi_flags(flags):
    '''Check that only valid flags that can be passed to
    ``fftw_mpi_local_size_many_1d`` and ``fftw_mpi_plan_many_dft_*``
    are contained in ``flags``, a list of strings.

    Return the combination of flags in binary format.

    '''
    cdef:
        unsigned int _flags = 0
        object _flags_used = []

    for each_flag in flags:
           try:
               flag = flag_dict.get(each_flag)
               if flag is None:
                   flag = mpi_flag_dict[each_flag]
               _flags |= flag
               _flags_used.append(each_flag)
           except KeyError:
               raise ValueError('Invalid flag: ' + '\'' +
                                each_flag + '\' is not a valid planner flag.')

    return _flags, _flags_used

cdef ptrdiff_t validate_block(block) except -1:
    '''Validate the block size assigned to an MPI rank.

    ``block`` can be either the string ``DEFAULT_BLOCK`` or a
    non-negative number.

    '''
    cdef ptrdiff_t iblock

    if block == 'DEFAULT_BLOCK':
        return FFTW_MPI_DEFAULT_BLOCK

    v = ValueError('Invalid block size: %s' % block)
    try:
        iblock = block
    except TypeError:
        raise v
    # block could have been a float
    if int(block) != float(block):
        raise v

    # a valid integer value?
    if iblock >= 0:
        return iblock

    raise v

cdef int validate_direction(direction) except 0:
    '''Wrap dict access for a more telling error message'''
    try:
        return directions[direction]
    except KeyError:
        raise ValueError('Invalid direction: %s' % direction)

cdef int validate_direction_scheme(direction, scheme) except 0:
    if not direction in scheme_directions[scheme]:
        raise ValueError("Invalid direction: '%s'. " % direction +
                         'The direction is not valid for the %s scheme. ' % scheme[0] +
                         'Try setting it explicitly if it is not already.')
    return directions[direction]

cdef object dtype_offset
dtype_offset = {np.dtype('complex128'):0,
               np.dtype('complex64'):1,
               np.dtype('clongdouble'):2,
               np.dtype('float64'):0,
               np.dtype('float32'):1,
               np.dtype('longdouble'):2,
              }

cdef int validate_dtype(dtype) except -1:
    '''Extract offset corresponding to a numpy.dtype object:
    0: double precision,
    1: single precision,
    2: long double precision.

    '''
    try:
        return dtype_offset[dtype]
    except KeyError:
        raise TypeError('Invalid dtype: %s' % dtype)

def validate_input_shape(input_shape):
    '''Turn ``input_shape`` into array, make sure only positive integers present.

    '''
    dtype = 'uint64'
    if np.iterable(input_shape):
        i = np.array(input_shape, dtype)
    else:
        i = input_shape * np.ones(1, dtype)
    if not len(i) or (i <= 0).any():
        raise ValueError('Invalid input shape: %s' % input_shape)
    return i

# TODO same order of arguments as in FFTW_MPI
# TODO Default efficiency options: inplace, transpose out
def create_mpi_plan(input_shape, input_chunk=None, input_dtype=None,
                    output_chunk=None, output_dtype=None,
                    ptrdiff_t howmany=1,
                    block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK',
                    flags=tuple(), direction=None, unsigned int threads=1,
                    Comm comm=None):

    '''Create a plan to use FFTW with MPI.

    Specify the global shape of the input array as ``input_shape``. This is the
    shape of entire array that may span multiple processors. Then specify either
    the ``input_dtype`` or an input array ``input_chunk``. If the latter
    is given, the dtype can extracted. The conversion scheme is inferred from
    the data type of either ``output_chunk`` or ``output_dtype``.

    If the ``local_*put_array`` arguments are omitted [recommended], arrays of
    the proper size to this MPI rank are allocated with SIMD alignment when
    possible. Note that these arrays may need to be slightly larger than what
    one might expect due to padding or extra memory needed in intermediate FFTW
    steps. This depends on the transform, dimensions, fftw implementation
    etc. Resist to guess! The arrays are accessible from the returned plan as
    attributes ``plan.*put_array``. Note that the physical size of a memory
    chunk may be slightly larger than what say ``plan.output_array.shape``
    indicates. For more details, see the FFTW manual on Distributed-memory FFTW
    with MPI.

    Construct an in-place transform with ``output_chunk='INPUT'``.

    All other arguments are used to allocate byte-aligned input/output arrays
    and the appropriate FFT plan, as described in :py:func:`local_size` and
    :py:class:`FFTW_MPI`.

    The ``direction`` argument is ignored for r2c and c2r transforms, and should
    be one of 'FFTW_FORWARD', or 'FFTW_BACKWARD' for c2c.

    Return a :class:`FFTW_MPI` object ``plan``. Execute the plan as ``plan(*args,
    *kwargs)``.

    '''
    # common arguments in function calls are checked there
    kwargs = dict(n_transforms=howmany,
                  block0=block0, block1=block1, flags=flags,
                  comm=comm, threads=threads)

    ###
    # determine input and output dtype
    ###
    err = ValueError("Invalid input specification: use either "
                     "'input_chunk' or 'input_dtype'")
    if input_chunk is not None:
        if input_dtype is not None:
            raise err
        input_dtype = input_chunk.dtype

    if input_dtype is None:
        raise err
    input_dtype = np.dtype(input_dtype)

    err = ValueError("Invalid output specification: use either "
                     "'output_chunk' or 'output_dtype'")
    if output_chunk is not None:
        if output_chunk == 'INPUT':
            if output_dtype is None:
                output_dtype = fftw_default_output[input_dtype]
        elif output_dtype is not None:
            raise err
        else:
            output_dtype = output_chunk.dtype

    if output_dtype is None:
        raise err

    # create dtype objects
    input_dtype = np.dtype(input_dtype)
    output_dtype = np.dtype(output_dtype)

    try:
        scheme = fftw_schemes[(input_dtype, output_dtype)]
    except KeyError:
        raise TypeError('Invalid scheme: '
                         'The output array and input array dtypes '
                         'do not correspond to a valid fftw scheme.')

    # ignore argument ``direction`` for r2c, c2r and infer it from dtypes
    if scheme[0] == 'c2c':
        validate_direction_scheme(direction, scheme)
    else:
        # only one choice
        direction = scheme_directions[scheme][0]
    kwargs['direction'] = direction

    input_shape = validate_input_shape(input_shape)

    # leave most of validation up to local_size
    local_size_input_shape = _mpi_local_size_input_shape(input_shape, scheme[0])
    res = local_size(local_size_input_shape, howmany, block0, block1,
                     flags, direction, comm)

    # number of elements to allocate
    n_elements_in, n_elements_out = n_elements_in_out(res[0], scheme[0])

    functions = mpi_scheme_functions[scheme]

    # need padding for r2c; cf. FFTW manual 6.5 'Multi-dimensional MPI DFTs of Real Data'
    local_input_shape, local_output_shape = mpi_local_shapes[functions['fft_shape_lookup']](input_shape, res, flags)
    # print 'shapes in plan', input_shape, local_size_input_shape, local_input_shape, local_output_shape
    # print 'scheme', scheme, 'n_in', n_elements_in, 'howmany', howmany, 'n_out', n_elements_out

    ###
    # Check or allocate input array
    ###
    # TODO replace np.prod with 64 bit version, see bug comment above for windows
    if input_chunk is None:
        # due to extra bytes for intermediate step, the conceptual size may
        # differ from the actual memory accessed. The extra bytes come after the last
        # input element
        input_chunk = n_byte_align_empty(n_elements_in, simd_alignment, dtype=input_dtype)

    if output_chunk is 'INPUT':
        output_chunk = np.frombuffer(input_chunk.data, output_dtype)
    elif output_chunk is None:
        # consider this is a 1D chunk of memory
        output_chunk = n_byte_align_empty(n_elements_out, simd_alignment, dtype=output_dtype)

    # check both input and output
    for s in ('in', 'out'):
        name = s + 'put'
        _mpi_validate_array(locals()[name + '_chunk'],
                            locals()['n_elements_' + s],
                            name)

    # print 'Leaving create_mpi_plan: ', n_elements_in, n_elements_out, \
    #     input_shape, input_chunk.shape, output_chunk.shape

    fftw_object = FFTW_MPI(input_shape, input_chunk, output_chunk,
                           **kwargs)
    return fftw_object

def local_size(input_shape, ptrdiff_t howmany=1,
               block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK',
               flags=tuple(), direction='FFTW_FORWARD',
               Comm comm=None):

    '''Determine number of elements (float, double...) needed for the output array
    and possibly extra elements for transposition. This is a generic wrapper
    around all ``fftw_mpi_local_size_*`` functions described in detail in
    sec. 6.12.4 of the FFTW manual.

    :param input_shape:

        tuple; The full shape of the input data as it exists scattered over
        multiple MPI ranks.

    :param howmany:

        To perform multiple transform of the same dimensions at once, interleave
        the elements one after another and pass ``how_many`` to indicate the
        stride. Example: Transform ``x`` and ``y``, then store ``x[0], y[0],
        x[1], y[1]`` contiguosly in memory and pass ``how_many=2``.

    :param block0:

        The block size in the first dimension is the number of elements
        that this MPI rank operates on during the transform. Applies only to
        multidimensional transforms.

    :param block1:

        The block size in the second dimension. Useful in conjunction with the
        flags 'FFTW_MPI_SCRAMBLED_*' and 'FFTW_MPI_TRANSPOSED_*' passed to the
        plan. Applies only to multidimensional transforms.

    :param flags:

        list; Flags need to match those given when creating a plan.

    :param direction:

        One of ('FFTW_FORWARD', 'FFTW_BACKWARD'); Applies only to 1D
        complex transforms. Needs to match the direction given when
        creating a plan.

    :param comm:

        mpi4py.libmpi.MPI_Comm; default to the world communicator.

    Return a tuple; In all cases, the first three values are
    1. the total number of elements to allocate for the output
    2. the number of elements along the first input dimension that this MPI rank operates on
    3. the starting index of the local MPI rank in the first input dimension

    With the 'FFTW_MPI_TRANSPOSED_OUT' flag, return additionally the number of
    elements and the starting index local to this MPI rank of the first
    dimension in the transposed output, which would be the second without
    'FFTW_MPI_TRANSPOSED_OUT'.

    In one dimension, return additionally the output size and starting index
    local to this MPI rank.

    '''
    cdef:
        unsigned int _flags
        int _direction
        int _rank
        ptrdiff_t _local_n0, _local_0_start, _local_n1, _local_1_start
        ptrdiff_t _block0, _block1, _howmany, i
        np.ndarray[ptrdiff_t, ndim=1] _n
        libmpi.MPI_Comm _comm

    ###
    # argument validation and initialisation
    ###

    # input data dimension
    try:
        _rank = len(input_shape)
    except TypeError:
        raise ValueError('Invalid dimension: '
                         'The input shape needs to be an iterable')
    if _rank <= 0:
        raise ValueError('Invalid dimension: '
                         'The input shape needs to be at least 1D')

    if howmany <= 0:
        raise ValueError('Invalid howmany: %d. Needs to be >= 1' % howmany)

    _howmany = howmany

    _block0 = validate_block(block0)
    _block1 = validate_block(block1)

    try:
        flags = list(flags)
    except TypeError:
        raise ValueError("Expect an iterable for 'flags'")
    _flags, _ = validate_mpi_flags(flags)
    _direction = validate_direction(direction)
    _comm = extract_communicator(comm)

    # copy over the shape, rely on np.intp == ptrdiff_t
    _n = np.empty(_rank, dtype=np.intp)
    _n[:] = input_shape

    # compute index of data-distribution function in table
    cdef unsigned int type_offset = 0
    if _rank == 1:
        type_offset = 2
    # only useful for d > 1
    elif 'FFTW_MPI_TRANSPOSED_IN'  in flags or \
         'FFTW_MPI_TRANSPOSED_OUT' in flags:
        type_offset = 1

    f = distributors[type_offset]

    return f(_rank, <ptrdiff_t *>np.PyArray_DATA(_n), _howmany,
             _block0, _block1, _comm,
             &_local_n0, &_local_0_start,
             &_local_n1, &_local_1_start,
             _direction, _flags)

# TODO store scrambled or transpose flags so user can query afterwards
# TODO check alignment
cdef class FFTW_MPI:
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

    cdef:
        # Each of these function pointers simply
        # points to a chosen fftw wrapper function
        fftw_mpi_generic_plan _fftw_planner
        fftw_generic_execute _fftw_execute
        fftw_generic_destroy_plan _fftw_destroy
        fftw_generic_plan_with_nthreads _nthreads_plan_setter

        # The plan is typecast when it is created or used
        # within the wrapper functions
        void *_plan

        # despite the appearance, these are C pointers to ndarrays
        np.ndarray _input_chunk
        np.ndarray _output_chunk

        # the sign: -1 for forward, +1 for backward transform
        int _direction
        int _flags

        bint _simd_allowed
        int _input_array_alignment
        int _output_array_alignment
        unsigned int _threads

        # global shapes defined on all MPI ranks
        object _input_shape
        object _output_shape

        object _local_input_shape
        object _local_output_shape

        object _local_input_shape_padded
        object _local_output_shape_padded

        object _input_dtype
        object _output_dtype
        object _flags_used

        # TODO keep MPI stuff? Then need property
        libmpi.MPI_Comm _comm
        int _MPI_rank

        double _normalisation_scaling

        # FFTW: rnk
        int _rank
        # FFTW: howmany
        # The howmany parameter specifies that the
        # transform is of contiguous howmany-tuples rather than
        # individual complex numbers
        ptrdiff_t _howmany,

        # variables assigned in local_size
        ptrdiff_t _local_n_elements, _local_n0, _local_0_start, _local_n1, _local_1_start

        # desired block sizes for this rank
        ptrdiff_t _block0, _block1

        # FFTW: n
        IntegerArray _dims

        # the offset in the output dimension
        # Is the same as _local_n0 unless FFTW_MPI_TRANSPOSED_OUT
        size_t _local_out_start

        # total number of elements
        int64_t _N

    @property
    def N(self):
        '''
        The product of the lengths of the DFT over all DFT axes.
        1/N is the normalisation constant. For any input array A,
        and for any set of axes, 1/N * ifft(fft(A)) = A
        '''
        return self._N

    def _get_simd_aligned(self):
        '''
        Return whether or not this FFTW object requires simd aligned
        input and output data.
        '''
        return self._simd_allowed

    simd_aligned = property(_get_simd_aligned)

    def _get_input_alignment(self):
        '''
        Returns the byte alignment of the input arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Input array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised, or
        a copy being made if the :meth:`~pyfftw.FFTW.__call__`
        interface is used.
        '''
        return self._input_array_alignment

    input_alignment = property(_get_input_alignment)

    def _get_output_alignment(self):
        '''
        Returns the byte alignment of the output arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Output array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised.
        '''
        return self._output_array_alignment

    output_alignment = property(_get_output_alignment)

    def _get_flags_used(self):
        '''
        Return which flags were used to construct the FFTW object.

        This includes flags that were added during initialisation.
        '''
        return tuple(self._flags_used)

    flags = property(_get_flags_used)

    # TODO Rename to input_chunk or input_data, have fixed attribute for
    # first array so it need not be recreated, then conceptual_input_array --> input_array.
    # drop local_input_shape, add bool has_input
    @property
    def input_chunk(self):
        '''Return the ndarray wrapping the local chunk of memory for the input
        of the Fourier transform on this MPI rank.

        '''
        return self._input_chunk

    @property
    def output_chunk(self):
        '''Return the ndarray wrapping the local chunk of memory for the
        output of the Fourier transform on this MPI rank.

        '''
        return self._output_chunk

    def _get_input_shape(self):
        '''Return the global shape of the input array that spans multiple MPI ranks for
        which the FFT is planned. Note that this usually differs from the shape
        of ``input_array`` local to this MPI rank.

        '''
        return self._input_shape

    input_shape = property(_get_input_shape)

    # TODO merge input/output, they do the same trick
    def get_input_array(self, transform=0):
        '''Return a view of the input buffer with the right conceptual dimensions; i.e.,
        the padding elements are hidden. If multiple transforms are done at
        once, select the array using ``transform``. Default: the first array.

        '''
        if transform + 1 > self._howmany:
            raise IndexError('Invalid index %d exceeds number of transforms %d' % (transform, self._howmany))
        if not self._local_input_shape:
            raise AttributeError('MPI rank %d does not have any input data' % self._MPI_rank)
        # transform from single to many transforms
        shape = np.array(self._local_input_shape_padded)
        shape[-1] *= self._howmany

        # print 'rank', self._MPI_rank, 'padded input shape', shape, 'size', self._input_chunk.size
        # print 'dtype', self._input_chunk.dtype

        # first select only as many elements as needed,
        # then create a view with right dimensions for many transforms,
        # then pick out only one transform and ignore padding elements in last dimension
        arr = self._input_chunk[0:np.prod(shape)].reshape(shape)
        return arr[..., transform:self._local_input_shape[-1] * self._howmany:self._howmany]

    def get_output_array(self, transform=0):
        '''Return a view of the output buffer with the right conceptual dimensions; i.e.,
        the padding elements are hidden. If multiple transforms are done at
        once, select the array using ``transform``. Default: first array.

        '''
        if transform + 1 > self._howmany:
            raise IndexError('Invalid index %d exceeds number of transforms %d' % (transform, self._howmany))
        if not self._local_output_shape:
            raise AttributeError('MPI rank %d does not have any output data' % self._MPI_rank)
        # transform from single to many transforms
        shape = np.array(self._local_output_shape_padded)
        shape[-1] *= self._howmany

        # print 'rank', self._MPI_rank, 'padded output shape', shape, 'size', self._input_chunk.size

        # first select only as many elements as needed,
        # then create a view with right dimensions for many transforms,
        # then pick out only one transform and ignore padding elements in last dimension
        arr = self._output_chunk[0:np.prod(shape)].reshape(shape)
        return arr[..., transform:self._local_output_shape[-1] * self._howmany:self._howmany]

    @property
    def input_array(self):
        return self.get_input_array()

    @property
    def output_array(self):
        return self.get_output_array()

    def _get_output_shape(self):

        '''Return the global shape of the output array that spans multiple MPI ranks for
        which the FFT is planned. Note that this usually differs from the shape
        of ``output_chunk`` local to this MPI rank.

        '''
        return self._output_shape

    output_shape = property(_get_output_shape)

    def _get_input_dtype(self):
        '''
        Return the dtype of the input array for which the FFT is planned.
        '''
        return self._input_dtype

    input_dtype = property(_get_input_dtype)

    def _get_output_dtype(self):
        '''
        Return the shape of the output array for which the FFT is planned.
        '''
        return self._output_dtype

    output_dtype = property(_get_output_dtype)

    def _get_direction(self):
        '''
        Return the planned FFT direction. Either `'FFTW_FORWARD'` or
        `'FFTW_BACKWARD'`.
        '''
        return directions_lookup[self._direction]

    direction = property(_get_direction)

    def _get_local_n_elements(self):
        '''Return the total number of elements, including padding and possibly extra
        bytes for intermediate steps, that need to be allocated for input/output
        on this MPI rank.

        '''
        return self._local_n_elements

    local_n_elements = property(_get_local_n_elements)

    def _get_local_n0(self):
        '''Return the number of elements in the first dimension this MPI rank operates
        on.

        '''
        return self._local_n0

    local_n0 = property(_get_local_n0)

    def _get_local_0_start(self):
        '''Return the offset in the first dimension this MPI rank operates on.

        '''
        return self._local_0_start

    local_0_start = property(_get_local_0_start)

    def _get_local_n1(self):
        '''Return the number of elements in the second dimension this MPI rank operates
        on.

        '''
        return self._local_n1

    local_n1 = property(_get_local_n1)

    def _get_local_1_start(self):
        '''Return the offset in the first dimension this MPI rank operates on.

        '''
        return self._local_1_start

    local_1_start = property(_get_local_1_start)

    @property
    def input_slice(self):
        '''Return range of elements in the first input dimension of this MPI rank. Return None if
        no input data on this rank.

        Example: myplan.get_input_array(1)[:] = global_data[myplan.input_slice]

        '''
        if self._local_input_shape:
            return slice(self._local_0_start, self._local_0_start + self._local_n0)
        else:
            return None

    @property
    def output_slice(self):
        '''Return range of elements in the first output dimension of this MPI rank. Return None if
        no output data on this rank.

        Example: myplan.get_output_array(1)[:] = global_data[myplan.output_slice]

        '''

        if self._local_output_shape:
            return slice(self._local_out_start, self._local_out_start + int(self._local_output_shape[0]))

    @property
    def local_input_shape(self):
        '''Return shape of a single local input array without any padding. Return None
        if no input data on this rank. The product over dimensions gives the
        number of input data elements that need be present on this MPI rank for the
        transform to work.

        '''
        return self._local_input_shape

    @property
    def local_output_shape(self):
        '''Return shape of a single local output array without any padding. Return None
        if no output data on this rank. The product over dimensions gives the
        number of data elements that are the output of the transform on this MPI rank.

        '''
        return self._local_output_shape

    @property
    def has_input(self):
        '''Evaluate to true if this MPI rank has input data.'''
        return bool(self._local_input_shape)

    @property
    def has_output(self):
        '''Evaluate to true if this MPI rank has input data.'''
        return bool(self._local_output_shape)

    @property
    def threads(self):
        '''The number of threads to use for the execution of the plan.'''
        return self._threads

    # TODO accept single flag and convert into sequence
    # TODO same args for __init__, add signature to pyfft.rst once it is stable
    # TODO not NULL tests for chunks
    def __cinit__(self, input_shape,  input_chunk, output_chunk,
                  block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK',
                  direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
                  unsigned int threads=1, planning_timelimit=None,
                  n_transforms=1, comm=None,
                  *args, **kwargs):

        # TODO Check or warn about prime n0
        # TODO document blocks

        # Initialise the pointers that need to be freed
        self._plan = NULL

        ###
        # argument parsing and checking
        ###
        flags = list(flags)

        _block0 = validate_block(block0)
        _block1 = validate_block(block1)

        cdef double _planning_timelimit
        if planning_timelimit is None:
            _planning_timelimit = FFTW_NO_TIMELIMIT
        else:
            try:
                _planning_timelimit = planning_timelimit
            except TypeError:
                raise TypeError('Invalid planning timelimit: '
                        'The planning timelimit needs to be a float.')

        # TODO document in-place
        # if output_chunk is None:
        #     output_chunk = input_chunk

        # save communicator and this process' rank
        cdef int ierr = 0
        cdef int comm_size = 0
        self._comm = extract_communicator(comm)
        ierr = libmpi.MPI_Comm_rank(self._comm, &self._MPI_rank)
        if ierr:
            raise RuntimeError('MPI_Comm_rank returned %d' % ierr)
        ierr = libmpi.MPI_Comm_size(self._comm, &comm_size)
        if ierr:
            raise RuntimeError('MPI_Comm_size returned %d' % ierr)

        # TODO still needed? ducktyping!
        # if not isinstance(input_chunk, np.ndarray):
        #     raise TypeError('Invalid input array: '
        #

        # 'The input array needs to be an instance '
        #             'of numpy.ndarray')

        # if not isinstance(output_chunk, np.ndarray):
        #     raise TypeError('Invalid output array: '
        #             'The output array needs to be an instance '
        #             'of numpy.ndarray')


        input_dtype = input_chunk.dtype
        output_dtype = output_chunk.dtype

        # # what if in place r2c? would assume r2r transform, so guess the type if not given
        #     if <intptr_t>np.PyArray_DATA(input_chunk) == <intptr_t>np.PyArray_DATA(output_chunk):
        #         # output_dtype = np.dtype(
        #         pass

        try:
            scheme = fftw_schemes[(input_dtype, output_dtype)]
        except KeyError:
            raise TypeError('Invalid scheme: '
                    'The output array and input array dtypes '
                    'do not correspond to a valid fftw scheme.')

        self._input_dtype = input_dtype
        self._output_dtype = output_dtype

        functions = mpi_scheme_functions[scheme]

        self._fftw_planner = mpi_planners[functions['planner']]
        self._fftw_execute = mpi_executors[functions['executor']]
        self._fftw_destroy = destroyers[functions['generic_precision']]

        self._nthreads_plan_setter = (
                nthreads_plan_setters[functions['generic_precision']])

        cdef fftw_generic_set_timelimit set_timelimit_func = (
                set_timelimit_funcs[functions['generic_precision']])

        # We're interested in the natural alignment on the real type, not
        # necessarily on the complex type. At least one bug was found where
        # numpy reported an alignment on a complex dtype that was different
        # to that on the real type.
        cdef int natural_input_alignment = input_chunk.real.dtype.alignment
        cdef int natural_output_alignment = output_chunk.real.dtype.alignment

        # If either of the arrays is not aligned on a 16-byte boundary,
        # we set the FFTW_UNALIGNED flag. This disables SIMD.
        # (16 bytes is assumed to be the minimal alignment)
        if 'FFTW_UNALIGNED' in flags:
            self._simd_allowed = False
            self._input_array_alignment = natural_input_alignment
            self._output_array_alignment = natural_output_alignment

        else:

            self._input_array_alignment = -1
            self._output_array_alignment = -1

            for each_alignment in _valid_simd_alignments:
                if (<intptr_t>np.PyArray_DATA(input_chunk) %
                        each_alignment == 0 and
                        <intptr_t>np.PyArray_DATA(output_chunk) %
                        each_alignment == 0):

                    self._simd_allowed = True

                    self._input_array_alignment = each_alignment
                    self._output_array_alignment = each_alignment

                    break

            if (self._input_array_alignment == -1 or
                    self._output_array_alignment == -1):

                self._simd_allowed = False

                self._input_array_alignment = (
                        natural_input_alignment)
                self._output_array_alignment = (
                        natural_output_alignment)
                flags.append('FFTW_UNALIGNED')

        if (not (<intptr_t>np.PyArray_DATA(input_chunk)
            % self._input_array_alignment == 0)):
            raise ValueError('Invalid input alignment: '
                    'The input array is expected to lie on a %d '
                    'byte boundary.' % self._input_array_alignment)

        if (not (<intptr_t>np.PyArray_DATA(output_chunk)
            % self._output_array_alignment == 0)):
            raise ValueError('Invalid output alignment: '
                    'The output array is expected to lie on a %d '
                    'byte boundary.' % self._output_array_alignment)

        self._direction = validate_direction_scheme(direction, scheme)

        ###
        # validate arrays
        ###
        input_shape = validate_input_shape(input_shape)

        # passing FFTW_MPI_TRANSPOSED_IN and FFTW_MPI_TRANSPOSED_OUT
        # just swaps first two dimensions
        if len(input_shape) > 1 and \
           'FFTW_MPI_TRANSPOSED_IN'  in flags and \
           'FFTW_MPI_TRANSPOSED_OUT' in flags:
            input_shape[0], input_shape[1] = input_shape[1], input_shape[0]

        # TODO issue warning if FFTW_MPI_TRANSPOSED_* for 1D?

        # need local_size to tell us how many elements in first dimension are
        # processed on this MPI rank to determine the right shape, and to let
        # the user know at what offset in the global this rank is in the first
        # dimension
        local_size_res = local_size(_mpi_local_size_input_shape(input_shape, scheme[0]),
                                    n_transforms, block0, block1, flags, direction, comm)
        self._local_input_shape, self._local_output_shape = mpi_local_shapes[functions['fft_shape_lookup']] \
                                                                (input_shape, local_size_res, flags)

        # print 'In rank', self._MPI_rank, 'local_size_res', local_size_res

        # Now we can validate the arrays
        for name in ('input', 'output'):
            _mpi_validate_array(locals()[name + '_chunk'],
                                local_size_res[0], name)

        self._rank = len(input_shape)
        self._howmany = n_transforms

        self._input_shape = input_shape
        self._output_shape = mpi_output_shape[functions['output_shape']](input_shape)

        self._local_input_shape_padded, self._local_output_shape_padded = \
        mpi_local_shapes_padded[functions['fft_shape_lookup']] (input_shape, local_size_res, flags)

        # print 'padded shapes', self._local_input_shape_padded, self._local_output_shape_padded

        # now that we know the shape is right, save only a view with the right
        # conceptual dimensions; i.e., without padded elements
        self._input_chunk = input_chunk
        self._output_chunk = output_chunk

        # remember local size for user attribute access
        self._local_n_elements, self._local_n0, self._local_0_start = local_size_res[:3]
        if len(local_size_res) > 3:
            self._local_n1, self._local_1_start = local_size_res[3:5]

        self._local_out_start = self._local_0_start
        if 'FFTW_MPI_TRANSPOSED_OUT' in flags or self._rank == 1:
            self._local_out_start = self._local_1_start

        # copy shape into right format for fftw
        self._dims = IntegerArray(self._rank)
        for n in range(self._rank):
            self._dims.data[n] = input_shape[n]

        ###
        # total number of elements for FFT normalization (independent of howmany!)
        ###
        self._N = 1
        for n in input_shape:
            self._N *= n

        self._normalisation_scaling = 1 / float(self.N)

        # valid flags extended by MPI flags
        self._flags = 0
        self._flags_used = []
        for each_flag in flags:
           try:
               flag = flag_dict.get(each_flag)
               if flag is None:
                   flag = mpi_flag_dict[each_flag]
               self._flags |= flag
               self._flags_used.append(each_flag)
           except KeyError:
               raise ValueError('Invalid flag: ' + '\'' +
                                each_flag + '\' is not a valid planner flag.')

        if self._rank == 1:
            if 'FFTW_MPI_TRANSPOSED_IN'  in flags or \
               'FFTW_MPI_TRANSPOSED_OUT' in flags:
                raise ValueError('Invalid flag: FFTW_MPI_TRANSPOSED_* does not  apply in 1d')

        if ('FFTW_MPI_SCRAMBLED_IN'  in flags or \
            'FFTW_MPI_SCRAMBLED_OUT' in flags):
            if self._rank > 1:
                raise ValueError('Invalid flag: FFTW_MPI_SCRAMBLED_* applies only in 1d')
            if comm_size <= 1:
                raise NotImplementedError('Invalid flag: FFTW_MPI_SCRAMBLED_* requires at least two processes')

        if ('FFTW_DESTROY_INPUT' not in flags) and (
                (scheme[0] != 'c2r') or not self._rank > 1):
            # The default in all possible cases is to preserve the input
            # This is not possible for r2c arrays with rank > 1
            self._flags |= FFTW_PRESERVE_INPUT

        # Make sure that the arrays are not too big for fftw
        # This is hard to test, so we cross our fingers and hope for the
        # best (any suggestions, please get in touch).
        cdef int i
        for i in range(0, len(self._input_shape)):
            if self._input_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the input array must be ' +
                        'less than ', str(limits.INT_MAX))

        for i in range(0, len(self._output_shape)):
            if self._output_shape[i] >= <Py_ssize_t> limits.INT_MAX:
                raise ValueError('Dimensions of the output array must be ' +
                        'less than ', str(limits.INT_MAX))

        # parallel execution
        self._threads = threads

        if self._threads > 1 and (not mpi4py.rc.threaded) or (mpi4py.rc.thread_level == 'single'):
            warnings.warn('MPI was not initialized with proper support for threads. '
                          'FFTW needs at least MPI_THREAD_FUNNELED. Proceeding with a single thread.')
            threads = 1
        self._nthreads_plan_setter(threads)

        # print('Before constructing the plan')
        # print self._rank, '(',
        # for i in range(self._rank):
        #     print self._dims.data[i],
        # print ')'
        # print self._howmany, self._block0, self._block1, \
        #       self._direction, self._flags
        # print self._input_chunk.size, self._output_chunk.size

        # Set the timelimit
        set_timelimit_func(_planning_timelimit)

        # Finally, construct the plan
        self._plan = self._fftw_planner(
            self._rank, self._dims.data, self._howmany,
            self._block0, self._block1,
            <void *>np.PyArray_DATA(self._input_chunk),
            <void *>np.PyArray_DATA(self._output_chunk),
            self._comm, self._direction, self._flags)

        if self._plan is NULL:
            raise RuntimeError('The data configuration has an uncaught error that led '+
                               'to the planner returning NULL in MPI rank %d. This is a bug.' % self._MPI_rank)

    def __init__(self, input_array, output_chunk, axes=(-1,),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
            int threads=1, planning_timelimit=None, comm=None,
            *args, **kwargs):
        '''
        **Arguments**:

        * ``input_array`` and ``output_chunk`` should be numpy arrays.
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
        | Type           | ``input_array.dtype`` | ``output_chunk.dtype`` | Direction |
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

        * For a Complex transform, ``output_chunk.shape == input_array.shape``
        * For a Real transform in the Forwards direction, both the following
          should be true:

          * ``output_chunk.shape[axes][-1] == input_array.shape[axes][-1]//2 + 1``
          * All the other axes should be equal in length.

        * For a Real transform in the Backwards direction, both the following
          should be true:

          * ``input_array.shape[axes][-1] == output_chunk.shape[axes][-1]//2 + 1``
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

        ``n_transforms`` is the number of same-size arrays to
        transform simultaneously in one go. Suppose you have three
        arrays x,y,z you want to Fourier transform. Then you can lay
        them out in interleaved format such that in memory they are
        ordered as x[0], y[0], z[0], x[1], y[1], z[1]... and specify
        ``n_transforms=3``. So ``n_transforms`` is the number of
        elements between the first and second element of each
        individual array. All arrays must be of the same data type and
        length.

        '''
        pass

    def __dealloc__(self):

        if not self._plan == NULL:
            self._fftw_destroy(self._plan)

    def __call__(self, input_array=None, output_chunk=None,
            normalise_idft=True):
        '''__call__(input_array=None, output_chunk=None, normalise_idft=True)

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

        ``output_chunk`` is always used as-is if possible. If the dtype, the
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

        if input_array is not None or output_chunk is not None:

            if input_array is None:
                input_array = self._input_chunk

            if output_chunk is None:
                output_chunk = self._output_chunk

            if not isinstance(input_array, np.ndarray):
                copy_needed = True
            elif (not input_array.dtype == self._input_dtype):
                copy_needed = True
            elif (not input_array.strides == self._input_strides):
                copy_needed = True
            elif not (<intptr_t>np.PyArray_DATA(input_array)
                    % self.input_alignment == 0):
                copy_needed = True
            else:
                copy_needed = False

            if copy_needed:

                if not isinstance(input_array, np.ndarray):
                    input_array = np.asanyarray(input_array)

                if not input_array.shape == self._input_shape:
                    raise ValueError('Invalid input shape: '
                            'The new input array should be the same shape '
                            'as the input array used to instantiate the '
                            'object.')

                self._input_chunk[:] = input_array

                if output_chunk is not None:
                    # No point wasting time if no update is necessary
                    # (which the copy above may have avoided)
                    input_array = self._input_chunk
                    self.update_arrays(input_array, output_chunk)

            else:
                self.update_arrays(input_array, output_chunk)

        self.execute()

        if self._direction == FFTW_BACKWARD and normalise_idft:
            self._output_chunk *= self._normalisation_scaling

        # TODO should we still return it? Just a part if howmany > 1
        return self._output_chunk

    # TODO update to MPI
    cpdef update_arrays(self,
            new_input_array, new_output_chunk):
        '''update_arrays(new_input_array, new_output_chunk)

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

        if not isinstance(new_output_chunk, np.ndarray):
            raise ValueError('Invalid output array '
                    'The new output array needs to be an instance '
                    'of numpy.ndarray')

        if not (<intptr_t>np.PyArray_DATA(new_input_array) %
                self.input_alignment == 0):
            raise ValueError('Invalid input alignment: '
                    'The original arrays were %d-byte aligned. It is '
                    'necessary that the update input array is similarly '
                    'aligned.' % self.input_alignment)

        if not (<intptr_t>np.PyArray_DATA(new_output_chunk) %
                self.output_alignment == 0):
            raise ValueError('Invalid output alignment: '
                    'The original arrays were %d-byte aligned. It is '
                    'necessary that the update output array is similarly '
                    'aligned.' % self.output_alignment)

        if not new_input_array.dtype == self._input_dtype:
            raise ValueError('Invalid input dtype: '
                    'The new input array is not of the same '
                    'dtype as was originally planned for.')

        if not new_output_chunk.dtype == self._output_dtype:
            raise ValueError('Invalid output dtype: '
                    'The new output array is not of the same '
                    'dtype as was originally planned for.')

        new_input_shape = new_input_array.shape
        new_output_shape = new_output_chunk.shape

        new_input_strides = new_input_array.strides
        new_output_strides = new_output_chunk.strides

        if not new_input_shape == self._input_shape:
            raise ValueError('Invalid input shape: '
                    'The new input array should be the same shape as '
                    'the input array used to instantiate the object.')

        if not new_output_shape == self._output_shape:
            raise ValueError('Invalid output shape: '
                    'The new output array should be the same shape as '
                    'the output array used to instantiate the object.')

        if not new_input_strides == self._input_strides:
            raise ValueError('Invalid input striding: '
                    'The strides should be identical for the new '
                    'input array as for the old.')

        if not new_output_strides == self._output_strides:
            raise ValueError('Invalid output striding: '
                    'The strides should be identical for the new '
                    'output array as for the old.')

        self._update_arrays(new_input_array, new_output_chunk)

    cdef _update_arrays(self,
            np.ndarray new_input_array, np.ndarray new_output_chunk):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self._input_chunk = new_input_array
        self._output_chunk = new_output_chunk

    cpdef execute(self):
        '''execute()

        Execute the planned operation, taking the correct kind of FFT of
        the input array (i.e. :attr:`FFTW.input_array`),
        and putting the result in the output array (i.e.
        :attr:`FFTW.output_chunk`).
        '''
        cdef void *input_pointer = (
                <void *>np.PyArray_DATA(self._input_chunk))
        cdef void *output_pointer = (
                <void *>np.PyArray_DATA(self._output_chunk))

        cdef void *plan = self._plan
        cdef fftw_generic_execute fftw_execute = self._fftw_execute

        if self._threads > 1:
            with nogil:
                fftw_execute(plan, input_pointer, output_pointer)
        else:
            fftw_execute(self._plan, input_pointer, output_pointer)

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
