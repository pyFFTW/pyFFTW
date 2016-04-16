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
        int sign, unsigned flags) nogil

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
    FFTW_WISDOM_ONLY = 2097152
