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

from libc.stddef cimport ptrdiff_t
# cdef extern from 'stddef.h':
#     ptrdiff_t

# ctypedef struct _fftw_mpi_ddim:
#     int _n
#     int _ib
#     int _ob

# mpi4py is a cython wrapper of MPI. This helps a lot
from mpi4py.libmpi cimport MPI_Comm

cdef extern from 'fftw3-mpi.h':
    # ctypedef struct fftw_iodim:
    #     pass


     # Double precision complex new array execute
    void fftw_mpi_execute_dft(fftw_plan,
          cdouble *_in, cdouble *_out)

    # Single precision complex new array execute
    void fftwf_mpi_execute_dft(fftwf_plan,
          cfloat *_in, cfloat *_out)

    # Long double precision complex new array execute
    void fftwl_mpi_execute_dft(fftwl_plan,
          clongdouble *_in, clongdouble *_out)

    # Double precision real to complex new array execute
    void fftw_mpi_execute_dft_r2c(fftw_plan,
          double *_in, cdouble *_out)

    # Single precision real to complex new array execute
    void fftwf_mpi_execute_dft_r2c(fftwf_plan,
          float *_in, cfloat *_out)

    # Long double precision real to complex new array execute
    void fftwl_mpi_execute_dft_r2c(fftwl_plan,
          long double *_in, clongdouble *_out)

    # Double precision complex to real new array execute
    void fftw_mpi_execute_dft_c2r(fftw_plan,
          cdouble *_in, double *_out)

    # Single precision complex to real new array execute
    void fftwf_mpi_execute_dft_c2r(fftwf_plan,
          cfloat *_in, float *_out)

    # Long double precision complex to real new array execute
    void fftwl_mpi_execute_dft_c2r(fftwl_plan,
          clongdouble *_in, long double *_out)

    # Initialization

    # Double precision
    void fftw_mpi_init()

    # Single precision
    void fftwf_mpi_init()

    # Long double precision
    void fftwl_mpi_init()

    # cleanup routines
    void fftw_mpi_cleanup()
    void fftwf_mpi_cleanup()
    void fftwl_mpi_cleanup()

    # data distribution
    # todo support FFTW_MPI_TRANSPOSED_[IN,OUT] 6.4.3
    ptrdiff_t fftw_mpi_local_size_2d(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftw_mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                     MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftw_mpi_local_size(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start)

    ptrdiff_t fftw_mpi_local_size_2d_transposed(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftw_mpi_local_size_3d_transposed(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                                MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftw_mpi_local_size_transposed(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                             ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                             ptrdiff_t *local_n1, ptrdiff_t *local_1_start)

    ptrdiff_t fftwf_mpi_local_size_2d(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwf_mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                     MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwf_mpi_local_size(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start)

    ptrdiff_t fftwf_mpi_local_size_2d_transposed(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftwf_mpi_local_size_3d_transposed(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                                MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftwf_mpi_local_size_transposed(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                             ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                             ptrdiff_t *local_n1, ptrdiff_t *local_1_start)

    ptrdiff_t fftwl_mpi_local_size_2d(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwl_mpi_local_size_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                     MPI_Comm comm,
                                     ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwl_mpi_local_size(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start)

    ptrdiff_t fftwl_mpi_local_size_2d_transposed(ptrdiff_t n0, ptrdiff_t n1, MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftwl_mpi_local_size_3d_transposed(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                                MPI_Comm comm,
                                                ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                ptrdiff_t *local_n1, ptrdiff_t *local_1_start)
    ptrdiff_t fftwl_mpi_local_size_transposed(int rnk, const ptrdiff_t *n, MPI_Comm comm,
                                             ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                             ptrdiff_t *local_n1, ptrdiff_t *local_1_start)

    # todo support FFTW_MPI_DEFAULT_BLOCK
    ptrdiff_t fftw_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                       ptrdiff_t block0, MPI_Comm comm,
                                       ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftw_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                                  ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
                                                  ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                                                  ptrdiff_t *local_n1, ptrdiff_t *local_1_start)

    # todo support FFTW_MPI_SCRAMBLED_[IN,OUT] 6.4.4
    ptrdiff_t fftw_mpi_local_size_1d(ptrdiff_t n0, MPI_Comm comm, int sign, unsigned flags,
                                     ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                     ptrdiff_t *local_no, ptrdiff_t *local_o_start)
    ptrdiff_t fftw_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany,
                                          MPI_Comm comm, int sign, unsigned flags,
                                          ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                          ptrdiff_t *local_no, ptrdiff_t *local_o_start)

    # plan creation

    # complex to complex
    fftw_plan fftw_mpi_plan_dft_1d(ptrdiff_t n0, cdouble *_in, cdouble *out,
                               MPI_Comm comm, int sign, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1,
                                   cdouble *_in, cdouble *out,
                                   MPI_Comm comm, int sign, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                   cdouble *_in, cdouble *out,
                                   MPI_Comm comm, int sign, unsigned flags)
    fftw_plan fftw_mpi_plan_dft(int rnk, const ptrdiff_t *n,
                                cdouble *_in, cdouble *out,
                                MPI_Comm comm, int sign, unsigned flags)
    fftw_plan fftw_mpi_plan_many_dft(int rnk, const ptrdiff_t *n,
                                     ptrdiff_t howmany, ptrdiff_t block, ptrdiff_t tblock,
                                     cdouble *_in, cdouble *out,
                                     MPI_Comm comm, int sign, unsigned flags)

    # real to complex
    fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       double *_in, cdouble *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       double *_in, cdouble *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                       double *_in, cdouble *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_r2c(int rnk, const ptrdiff_t *n,
                                    double *_in, cdouble *out,
                                    MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       cdouble *_in, double *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                       cdouble *_in, double *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                       cdouble *_in, double *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r(int rnk, const ptrdiff_t *n,
                                    cdouble *_in, double *out,
                                    MPI_Comm comm, unsigned flags)

    # real to real not supported

    # wisdom functions
    void fftw_mpi_gather_wisdom(MPI_Comm comm)
    void fftw_mpi_broadcast_wisdom(MPI_Comm comm)

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

ctypedef object (*fftw_mpi_generic_local_size)(
                    int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                    ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
                    ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                    ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
                    int sign, unsigned int flags)

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

# MPI-specific flags
cdef enum:
    FFTW_MPI_DEFAULT_BLOCK = 0

cdef enum:
    FFTW_MPI_SCRAMBLED_IN   =  67108864
    FFTW_MPI_SCRAMBLED_OUT  = 134217728
    FFTW_MPI_TRANSPOSED_IN  = 268435456
    FFTW_MPI_TRANSPOSED_OUT = 536870912
