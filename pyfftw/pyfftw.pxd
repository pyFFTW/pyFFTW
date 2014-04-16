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

cimport numpy as np
from libc.stdint cimport int64_t

ctypedef struct _fftw_iodim:
    int _n
    int _is
    int _os

cdef extern from 'pyfftw_complex.h':
    
    ctypedef float cfloat[2]
    ctypedef double cdouble[2]
    ctypedef long double clongdouble[2]

cdef extern from 'fftw3.h':
    
    # Double precision plans
    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftw_plan

    # Single precision plans
    ctypedef struct fftwf_plan_struct:
        pass

    ctypedef fftwf_plan_struct *fftwf_plan

    # Long double precision plans
    ctypedef struct fftwl_plan_struct:
        pass

    ctypedef fftwl_plan_struct *fftwl_plan

    # The stride info structure. I think that strictly
    # speaking, this should be defined with a type suffix
    # on fftw (ie fftw, fftwf or fftwl), but since the
    # definition is transparent and is defined as _fftw_iodim,
    # we ignore the distinction in order to simplify the code.
    ctypedef struct fftw_iodim:
        pass
    
    # Double precision complex planner
    fftw_plan fftw_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cdouble *_in, cdouble *_out,
            int sign, unsigned flags)
    
    # Single precision complex planner
    fftwf_plan fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, cfloat *_out,
            int sign, unsigned flags)

    # Single precision complex planner
    fftwl_plan fftwl_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            clongdouble *_in, clongdouble *_out,
            int sign, unsigned flags)
    
    # Double precision real to complex planner
    fftw_plan fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, cdouble *_out,
            unsigned flags)
    
    # Single precision real to complex planner
    fftwf_plan fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, cfloat *_out,
            unsigned flags)

    # Single precision real to complex planner
    fftwl_plan fftwl_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double *_in, clongdouble *_out,
            unsigned flags)

    # Double precision complex to real planner
    fftw_plan fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cdouble *_in, double *_out,
            unsigned flags)
    
    # Single precision complex to real planner
    fftwf_plan fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, float *_out,
            unsigned flags)

    # Single precision complex to real planner
    fftwl_plan fftwl_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            clongdouble *_in, long double *_out,
            unsigned flags)

    # Double precision complex new array execute
    void fftw_execute_dft(fftw_plan,
          cdouble *_in, cdouble *_out) nogil
    
    # Single precision complex new array execute    
    void fftwf_execute_dft(fftwf_plan,
          cfloat *_in, cfloat *_out) nogil

    # Long double precision complex new array execute    
    void fftwl_execute_dft(fftwl_plan,
          clongdouble *_in, clongdouble *_out) nogil
   
    # Double precision real to complex new array execute
    void fftw_execute_dft_r2c(fftw_plan,
          double *_in, cdouble *_out) nogil
    
    # Single precision real to complex new array execute    
    void fftwf_execute_dft_r2c(fftwf_plan,
          float *_in, cfloat *_out) nogil

    # Long double precision real to complex new array execute    
    void fftwl_execute_dft_r2c(fftwl_plan,
          long double *_in, clongdouble *_out) nogil

    # Double precision complex to real new array execute
    void fftw_execute_dft_c2r(fftw_plan,
          cdouble *_in, double *_out) nogil
    
    # Single precision complex to real new array execute    
    void fftwf_execute_dft_c2r(fftwf_plan,
          cfloat *_in, float *_out) nogil

    # Long double precision complex to real new array execute    
    void fftwl_execute_dft_c2r(fftwl_plan,
          clongdouble *_in, long double *_out) nogil

    # Double precision plan destroyer
    void fftw_destroy_plan(fftw_plan)

    # Single precision plan destroyer
    void fftwf_destroy_plan(fftwf_plan)

    # Long double precision plan destroyer
    void fftwl_destroy_plan(fftwl_plan)

    # Double precision set timelimit
    void fftw_set_timelimit(double seconds)

    # Single precision set timelimit
    void fftwf_set_timelimit(double seconds)

    # Long double precision set timelimit
    void fftwl_set_timelimit(double seconds)

    # Threading routines
    # Double precision
    void fftw_init_threads()
    void fftw_plan_with_nthreads(int n)

    # Single precision
    void fftwf_init_threads()
    void fftwf_plan_with_nthreads(int n)

    # Long double precision
    void fftwl_init_threads()
    void fftwl_plan_with_nthreads(int n)

    # cleanup routines
    void fftw_cleanup()
    void fftwf_cleanup()
    void fftwl_cleanup()
    void fftw_cleanup_threads()
    void fftwf_cleanup_threads()
    void fftwl_cleanup_threads()

    # wisdom functions
    void fftw_export_wisdom(void (*write_char)(char c, void *), void *data)
    void fftwf_export_wisdom(void (*write_char)(char c, void *), void *data)
    void fftwl_export_wisdom(void (*write_char)(char c, void *), void *data)

    int fftw_import_wisdom_from_string(char *input_string)
    int fftwf_import_wisdom_from_string(char *input_string)
    int fftwl_import_wisdom_from_string(char *input_string)

    #int fftw_export_wisdom_to_filename(char *filename)
    #int fftwf_export_wisdom_to_filename(char *filename)
    #int fftwl_export_wisdom_to_filename(char *filename)
    #
    #int fftw_import_wisdom_from_filename(char *filename)
    #int fftwf_import_wisdom_from_filename(char *filename)
    #int fftwl_import_wisdom_from_filename(char *filename)

    void fftw_forget_wisdom()
    void fftwf_forget_wisdom()
    void fftwl_forget_wisdom()

    double FFTW_NO_TIMELIMIT

# Define function pointers that can act as a placeholder
# for whichever dtype is used (the problem being that fftw
# has different function names and signatures for all the 
# different precisions and dft types).
ctypedef void * (*fftw_generic_plan_guru)(
        int rank, fftw_iodim *dims,
        int howmany_rank, fftw_iodim *howmany_dims,
        void *_in, void *_out,
        int sign, int flags)

ctypedef void (*fftw_generic_execute)(void *_plan, void *_in, void *_out) nogil

ctypedef void (*fftw_generic_destroy_plan)(void *_plan)

ctypedef void (*fftw_generic_init_threads)()

ctypedef void (*fftw_generic_plan_with_nthreads)(int n)

ctypedef void (*fftw_generic_set_timelimit)(double seconds)

ctypedef bint (*validator)(np.ndarray input_array, 
        np.ndarray output_array, int64_t *axes, int64_t *not_axes, 
        int64_t axes_length)

# Direction enum
cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1

# Documented flags
cdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64

# Declare external interface here so that it is accessible from other cython
# modules using cimport
cdef class FFTW:
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

    cdef int __rank
    cdef _fftw_iodim *__dims
    cdef int __howmany_rank
    cdef _fftw_iodim *__howmany_dims

    cdef int64_t *_axes
    cdef int64_t *_not_axes

    cdef int64_t _N

    cpdef update_arrays(self, new_input_array, new_output_array)

    cdef _update_arrays(self, np.ndarray new_input_array, np.ndarray new_output_array)

    cpdef execute(self)
