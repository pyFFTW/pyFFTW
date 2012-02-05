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

cdef extern from "complex.h":
    pass

ctypedef struct _fftw_iodim:
    int _n
    int _is
    int _os

cdef extern from "fftw3.h":
    
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
            double complex *_in, double complex *_out,
            int sign, unsigned flags)
    
    # Single precision complex planner
    fftwf_plan fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float complex *_in, float complex *_out,
            int sign, unsigned flags)

    # Single precision complex planner
    fftwl_plan fftwl_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double complex *_in, long double complex *_out,
            int sign, unsigned flags)
    
    # Double precision real to complex planner
    fftw_plan fftw_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double *_in, double complex *_out,
            unsigned flags)
    
    # Single precision real to complex planner
    fftwf_plan fftwf_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float *_in, float complex *_out,
            unsigned flags)

    # Single precision real to complex planner
    fftwl_plan fftwl_plan_guru_dft_r2c(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double *_in, long double complex *_out,
            unsigned flags)

    # Double precision complex to real planner
    fftw_plan fftw_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            double complex *_in, double *_out,
            unsigned flags)
    
    # Single precision complex to real planner
    fftwf_plan fftwf_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float complex *_in, float *_out,
            unsigned flags)

    # Single precision complex to real planner
    fftwl_plan fftwl_plan_guru_dft_c2r(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            long double complex *_in, long double *_out,
            unsigned flags)

    # Double precision complex new array execute
    void fftw_execute_dft(fftw_plan,
          double complex *_in, double complex *_out)
    
    # Single precision complex new array execute    
    void fftwf_execute_dft(fftwf_plan,
          float complex *_in, float complex *_out)

    # Long double precision complex new array execute    
    void fftwl_execute_dft(fftwl_plan,
          long double complex *_in, long double complex *_out)
   
    # Double precision real to complex new array execute
    void fftw_execute_dft_r2c(fftw_plan,
          double *_in, double complex *_out)
    
    # Single precision real to complex new array execute    
    void fftwf_execute_dft_r2c(fftwf_plan,
          float *_in, float complex *_out)

    # Long double precision real to complex new array execute    
    void fftwl_execute_dft_r2c(fftwl_plan,
          long double *_in, long double complex *_out)

    # Double precision complex to real new array execute
    void fftw_execute_dft_c2r(fftw_plan,
          double complex *_in, double *_out)
    
    # Single precision complex to real new array execute    
    void fftwf_execute_dft_c2r(fftwf_plan,
          float complex *_in, float *_out)

    # Long double precision complex to real new array execute    
    void fftwl_execute_dft_c2r(fftwl_plan,
          long double complex *_in, long double *_out)

    # Double precision plan destroyer
    void fftw_destroy_plan(fftw_plan)

    # Single precision plan destroyer
    void fftwf_destroy_plan(fftwf_plan)

    # Long double precision plan destroyer
    void fftwl_destroy_plan(fftwl_plan)

# Define function pointers that can act as a placeholder
# for whichever dtype is used (the problem being that fftw
# has different function names and signatures for all the 
# different precisions and dft types).
ctypedef void * (*fftw_generic_plan_guru)(
        int rank, fftw_iodim *dims,
        int howmany_rank, fftw_iodim *howmany_dims,
        void *_in, void *_out,
        int sign, int flags)

ctypedef void (*fftw_generic_execute)(void *_plan, void *_in, void *_out)

ctypedef void (*fftw_generic_destroy_plan)(void *_plan)

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
