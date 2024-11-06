# cython: language_level=3
#
# Copyright 2014 Knowledge Economy Developments Ltd
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

cimport numpy as np
from libc.stdint cimport int64_t

cdef extern from *:
    bint PYFFTW_HAVE_SINGLE
    bint PYFFTW_HAVE_DOUBLE
    bint PYFFTW_HAVE_LONG
    bint PYFFTW_HAVE_SINGLE_OMP
    bint PYFFTW_HAVE_DOUBLE_OMP
    bint PYFFTW_HAVE_LONG_OMP
    bint PYFFTW_HAVE_SINGLE_THREADS
    bint PYFFTW_HAVE_DOUBLE_THREADS
    bint PYFFTW_HAVE_LONG_THREADS
    bint PYFFTW_HAVE_SINGLE_MULTITHREADING
    bint PYFFTW_HAVE_DOUBLE_MULTITHREADING
    bint PYFFTW_HAVE_LONG_MULTITHREADING

ctypedef struct _fftw_iodim:
    int _n
    int _is
    int _os

cdef extern from *:
    '''
    /* Defines complex types that are bit compatible with C99's complex.h
    * and (crucially) the same type as expected by fftw3.h.
    * Note, don't use this with complex.h. fftw3.h checks to see whether
    * complex.h is included and then uses that to set the interface.
    * Since MSVC doesn't support C99, by using the following types we
    * have a cross platform/compiler solution.
    *
    * */

    #ifndef PYFFTW_COMPLEX_H
    #define PYFFTW_COMPLEX_H

    typedef float cfloat[2];
    typedef double cdouble[2];
    typedef long double clongdouble[2];

    #endif /* Header guard */
    '''

    ctypedef float cfloat[2]
    ctypedef double cdouble[2]
    ctypedef long double clongdouble[2]

cdef extern from 'fftw3.h':
    """
    #if !PYFFTW_HAVE_DOUBLE
    #define fftw_plan_guru_dft(...) (NULL)
    #define fftw_plan_guru_dft_r2c(...) (NULL)
    #define fftw_plan_guru_dft_c2r(...) (NULL)
    #define fftw_plan_guru_r2r(...) (NULL)
    #define fftw_execute_dft(...) ((void)0)
    #define fftw_execute_dft_r2c(...) ((void)0)
    #define fftw_execute_dft_c2r(...) ((void)0)
    #define fftw_execute_r2r(...) ((void)0)
    #define fftw_destroy_plan(plan) ((void)0)
    #define fftw_cleanup() ((void)0)
    #define fftw_export_wisdom(...) ((void)0)
    #define fftw_import_wisdom_from_string(wisdom) (0)
    #define fftw_forget_wisdom() ((void)0)
    #define fftw_set_timelimit(...) ((void)0)
    #define fftw_version ""
    #define fftw_cc ""
    #define fftw_codelet_optim ""
    #endif

    #if !PYFFTW_HAVE_SINGLE
    #define fftwf_plan_guru_dft(...) (NULL)
    #define fftwf_plan_guru_dft_r2c(...) (NULL)
    #define fftwf_plan_guru_dft_c2r(...) (NULL)
    #define fftwf_plan_guru_r2r(...) (NULL)
    #define fftwf_execute_dft(...) ((void)0)
    #define fftwf_execute_dft_r2c(...) ((void)0)
    #define fftwf_execute_dft_c2r(...) ((void)0)
    #define fftwf_execute_r2r(...) ((void)0)
    #define fftwf_destroy_plan(plan) ((void)0)
    #define fftwf_cleanup() ((void)0)
    #define fftwf_export_wisdom(...) ((void)0)
    #define fftwf_import_wisdom_from_string(wisdom) (0)
    #define fftwf_forget_wisdom() ((void)0)
    #define fftwf_set_timelimit(...) ((void)0)
    #define fftwf_version ""
    #define fftwf_cc ""
    #define fftwf_codelet_optim ""
    #endif

    #if !PYFFTW_HAVE_LONG
    #define fftwl_plan_guru_dft(...) (NULL)
    #define fftwl_plan_guru_dft_r2c(...) (NULL)
    #define fftwl_plan_guru_dft_c2r(...) (NULL)
    #define fftwl_plan_guru_r2r(...) (NULL)
    #define fftwl_execute_dft(...) ((void)0)
    #define fftwl_execute_dft_r2c(...) ((void)0)
    #define fftwl_execute_dft_c2r(...) ((void)0)
    #define fftwl_execute_r2r(...) ((void)0)
    #define fftwl_destroy_plan(plan) ((void)0)
    #define fftwl_cleanup() ((void)0)
    #define fftwl_export_wisdom(...) ((void)0)
    #define fftwl_import_wisdom_from_string(wisdom) (0)
    #define fftwl_forget_wisdom() ((void)0)
    #define fftwl_set_timelimit(...) ((void)0)
    #define fftwl_version ""
    #define fftwl_cc ""
    #define fftwl_codelet_optim ""
    #endif

    #if !PYFFTW_HAVE_DOUBLE_MULTITHREADING
    #define fftw_cleanup_threads() ((void)0)
    #define fftw_init_threads() ((void)0)
    #define fftw_plan_with_nthreads(...) ((void)0)
    #endif

    #if !PYFFTW_HAVE_SINGLE_MULTITHREADING
    #define fftwf_cleanup_threads() ((void)0)
    #define fftwf_init_threads() ((void)0)
    #define fftwf_plan_with_nthreads(...) ((void)0)
    #endif

    #if !PYFFTW_HAVE_LONG_MULTITHREADING
    #define fftwl_cleanup_threads() ((void)0)
    #define fftwl_init_threads() ((void)0)
    #define fftwl_plan_with_nthreads(...) ((void)0)
    #endif

    /* FFTW Windows' DLL (as of 3.3.5) doesn't export these symbols */
    #if defined(_WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
    #if PYFFTW_HAVE_DOUBLE
    #define fftw_version ""
    #define fftw_cc ""
    #define fftw_codelet_optim ""
    #endif
    #if PYFFTW_HAVE_SINGLE
    #define fftwf_version ""
    #define fftwf_cc ""
    #define fftwf_codelet_optim ""
    #endif
    #if PYFFTW_HAVE_LONG
    #define fftwl_version ""
    #define fftwl_cc ""
    #define fftwl_codelet_optim ""
    #endif
    #endif
    """

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
            int sign, unsigned flags) nogil

    # Single precision complex planner
    fftwf_plan fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, cfloat *_out,
            int sign, unsigned flags) nogil

    # Single precision complex planner
    fftwl_plan fftwl_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            clongdouble *_in, clongdouble *_out,
            int sign, unsigned flags) nogil

    # Double precision real to complex planner
    fftw_plan fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, cdouble *_out,
            unsigned flags) nogil

    # Single precision real to complex planner
    fftwf_plan fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, cfloat *_out,
            unsigned flags) nogil

    # Single precision real to complex planner
    fftwl_plan fftwl_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double *_in, clongdouble *_out,
            unsigned flags) nogil

    # Double precision complex to real planner
    fftw_plan fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cdouble *_in, double *_out,
            unsigned flags) nogil

    # Single precision complex to real planner
    fftwf_plan fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            cfloat *_in, float *_out,
            unsigned flags) nogil

    # Single precision complex to real planner
    fftwl_plan fftwl_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            clongdouble *_in, long double *_out,
            unsigned flags) nogil

    # Double precision real planner
    fftw_plan fftw_plan_guru_r2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, double *_out,
            fftw_r2r_kind *kind, unsigned flags) nogil

    # Single precision real planner
    fftwf_plan fftwf_plan_guru_r2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, float *_out,
            fftwf_r2r_kind *kind, unsigned flags) nogil

    # Long double precision real planner
    fftwl_plan fftwl_plan_guru_r2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double *_in, long double *_out,
            fftwl_r2r_kind *kind, unsigned flags) nogil

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

    # Double precision real new array execute
    void fftw_execute_r2r(fftw_plan,
          double *_in, double *_out) nogil

    # Single precision real new array execute
    void fftwf_execute_r2r(fftwf_plan,
          float *_in, float *_out) nogil

    # Long double precision real new array execute
    void fftwl_execute_r2r(fftwl_plan,
          long double *_in, long double *_out) nogil

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
    void fftw_cleanup() nogil
    void fftwf_cleanup() nogil
    void fftwl_cleanup() nogil
    void fftw_cleanup_threads() nogil
    void fftwf_cleanup_threads() nogil
    void fftwl_cleanup_threads() nogil

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

    const char fftw_version[]
    const char fftwf_version[]
    const char fftwl_version[]

    const char fftw_cc[]
    const char fftwf_cc[]
    const char fftwl_cc[]

    const char fftw_codelet_optim[]
    const char fftwf_codelet_optim[]
    const char fftwl_codelet_optim[]

    double FFTW_NO_TIMELIMIT

    ctypedef enum fftw_r2r_kind:
        FFTW_R2HC = 0
        FFTW_HC2R = 1
        FFTW_DHT = 2
        # from fftw3.f 3.3.3; may not be valid for different versions of FFTW.
        FFTW_REDFT00  = 3
        FFTW_REDFT01  = 4
        FFTW_REDFT10  = 5
        FFTW_REDFT11  = 6
        FFTW_RODFT00  = 7
        FFTW_RODFT01  = 8
        FFTW_RODFT10  = 9
        FFTW_RODFT11  = 10

    ctypedef enum fftwf_r2r_kind:
        FFTW_R2HC = 0
        FFTW_HC2R = 1
        FFTW_DHT = 2
        # from fftw3.f 3.3.3; may not be valid for different versions of FFTW.
        FFTW_REDFT00  = 3
        FFTW_REDFT01  = 4
        FFTW_REDFT10  = 5
        FFTW_REDFT11  = 6
        FFTW_RODFT00  = 7
        FFTW_RODFT01  = 8
        FFTW_RODFT10  = 9
        FFTW_RODFT11  = 10

    ctypedef enum fftwl_r2r_kind:
        FFTW_R2HC = 0
        FFTW_HC2R = 1
        FFTW_DHT = 2
        # from fftw3.f 3.3.3; may not be valid for different versions of FFTW.
        FFTW_REDFT00  = 3
        FFTW_REDFT01  = 4
        FFTW_REDFT10  = 5
        FFTW_REDFT11  = 6
        FFTW_RODFT00  = 7
        FFTW_RODFT01  = 8
        FFTW_RODFT10  = 9
        FFTW_RODFT11  = 10

# Define function pointers that can act as a placeholder
# for whichever dtype is used (the problem being that fftw
# has different function names and signatures for all the
# different precisions and dft types).
ctypedef void * (*fftw_generic_plan_guru)(
        int rank, fftw_iodim *dims,
        int howmany_rank, fftw_iodim *howmany_dims,
        void *_in, void *_out,
        int *directions, unsigned flags) nogil

ctypedef void (*fftw_generic_execute)(void *_plan, void *_in, void *_out) noexcept nogil

ctypedef struct fftw_exe:

    fftw_generic_execute _fftw_execute
    void* _plan
    void* _input_pointer
    void* _output_pointer

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
    FFTW_WISDOM_ONLY = 2097152


cdef class FFTW:
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
    cdef void* _input_pointer
    cdef np.ndarray _output_array
    cdef void* _output_pointer

    cdef int *_direction
    cdef unsigned _flags

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
    cdef double _sqrt_normalisation_scaling

    cdef int _rank
    cdef _fftw_iodim *_dims
    cdef int _howmany_rank
    cdef _fftw_iodim *_howmany_dims

    cdef int64_t *_axes
    cdef int64_t *_not_axes

    cdef int64_t _total_size

    cdef bint _normalise_idft
    cdef bint _ortho

    cpdef update_arrays(self,
            new_input_array, new_output_array)

    cdef _update_arrays(self,
            np.ndarray new_input_array, np.ndarray new_output_array)

    cpdef execute(self)

    cdef fftw_exe get_fftw_exe(self)

    cdef void execute_nogil(self) noexcept nogil

cdef void execute_in_nogil(fftw_exe* exe_ptr) noexcept nogil
