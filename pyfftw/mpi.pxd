# mpi4py is a cython wrapper of MPI. This helps a lot
from mpi4py.libmpi cimport MPI_Comm

from libc.stddef cimport ptrdiff_t

cdef extern from 'fftw3-mpi.h':
    # ctypedef struct fftw_iodim:
    #     pass

    # TODO Are fftw_mpi_execute_* even needed? Can call fftw_execute_* instead

    #  # Double precision complex new array execute
    # void fftw_mpi_execute_dft(fftw_plan,
    #       cdouble *_in, cdouble *_out)

    # # Single precision complex new array execute
    # void fftwf_mpi_execute_dft(fftwf_plan,
    #       cfloat *_in, cfloat *_out)

    # # Long double precision complex new array execute
    # void fftwl_mpi_execute_dft(fftwl_plan,
    #       clongdouble *_in, clongdouble *_out)

    # # Double precision real to complex new array execute
    # void fftw_mpi_execute_dft_r2c(fftw_plan,
    #       double *_in, cdouble *_out)

    # # Single precision real to complex new array execute
    # void fftwf_mpi_execute_dft_r2c(fftwf_plan,
    #       float *_in, cfloat *_out)

    # # Long double precision real to complex new array execute
    # void fftwl_mpi_execute_dft_r2c(fftwl_plan,
    #       long double *_in, clongdouble *_out)

    # # Double precision complex to real new array execute
    # void fftw_mpi_execute_dft_c2r(fftw_plan,
    #       cdouble *_in, double *_out)

    # # Single precision complex to real new array execute
    # void fftwf_mpi_execute_dft_c2r(fftwf_plan,
    #       cfloat *_in, float *_out)

    # # Long double precision complex to real new array execute
    # void fftwl_mpi_execute_dft_c2r(fftwl_plan,
    #       clongdouble *_in, long double *_out)

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

    ptrdiff_t fftw_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                       ptrdiff_t block0, MPI_Comm comm,
                                       ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftw_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *n,
                                                  ptrdiff_t howmany,
                                                  ptrdiff_t block0, ptrdiff_t block1,
                                                  MPI_Comm comm,
                                                  ptrdiff_t *local_n0,
                                                  ptrdiff_t *local_0_start,
                                                  ptrdiff_t *local_n1,
                                                  ptrdiff_t *local_1_start)

    ptrdiff_t fftwf_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                       ptrdiff_t block0, MPI_Comm comm,
                                       ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwf_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *n,
                                                  ptrdiff_t howmany,
                                                  ptrdiff_t block0, ptrdiff_t block1,
                                                  MPI_Comm comm,
                                                  ptrdiff_t *local_n0,
                                                  ptrdiff_t *local_0_start,
                                                  ptrdiff_t *local_n1,
                                                  ptrdiff_t *local_1_start)

    ptrdiff_t fftwl_mpi_local_size_many(int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                                       ptrdiff_t block0, MPI_Comm comm,
                                       ptrdiff_t *local_n0, ptrdiff_t *local_0_start)
    ptrdiff_t fftwl_mpi_local_size_many_transposed(int rnk, const ptrdiff_t *n,
                                                  ptrdiff_t howmany,
                                                  ptrdiff_t block0, ptrdiff_t block1,
                                                  MPI_Comm comm,
                                                  ptrdiff_t *local_n0,
                                                  ptrdiff_t *local_0_start,
                                                  ptrdiff_t *local_n1,
                                                  ptrdiff_t *local_1_start)

    ptrdiff_t fftw_mpi_local_size_1d(ptrdiff_t n0, MPI_Comm comm, int sign,
                                     unsigned flags,
                                     ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                     ptrdiff_t *local_no, ptrdiff_t *local_o_start)
    ptrdiff_t fftw_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany,
                                          MPI_Comm comm, int sign, unsigned flags,
                                          ptrdiff_t *local_ni,
                                          ptrdiff_t *local_i_start,
                                          ptrdiff_t *local_no,
                                          ptrdiff_t *local_o_start)

    ptrdiff_t fftwf_mpi_local_size_1d(ptrdiff_t n0, MPI_Comm comm, int sign,
                                     unsigned flags,
                                     ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                     ptrdiff_t *local_no, ptrdiff_t *local_o_start)
    ptrdiff_t fftwf_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany,
                                          MPI_Comm comm, int sign, unsigned flags,
                                          ptrdiff_t *local_ni,
                                          ptrdiff_t *local_i_start,
                                          ptrdiff_t *local_no,
                                          ptrdiff_t *local_o_start)

    ptrdiff_t fftwl_mpi_local_size_1d(ptrdiff_t n0, MPI_Comm comm, int sign,
                                     unsigned flags,
                                     ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
                                     ptrdiff_t *local_no, ptrdiff_t *local_o_start)
    ptrdiff_t fftwl_mpi_local_size_many_1d(ptrdiff_t n0, ptrdiff_t howmany,
                                          MPI_Comm comm, int sign, unsigned flags,
                                          ptrdiff_t *local_ni,
                                          ptrdiff_t *local_i_start,
                                          ptrdiff_t *local_no,
                                          ptrdiff_t *local_o_start)

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
                                     ptrdiff_t howmany, ptrdiff_t block,
                                     ptrdiff_t tblock,
                                     cdouble *_in, cdouble *out,
                                     MPI_Comm comm, int sign, unsigned flags)

    fftwf_plan fftwf_mpi_plan_dft_1d(ptrdiff_t n0, cfloat *_in, cfloat *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1,
                                    cfloat *_in, cfloat *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                    cfloat *_in, cfloat *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft(int rnk, const ptrdiff_t *n,
                                 cfloat *_in, cfloat *out,
                                 MPI_Comm comm, int sign, unsigned flags)
    fftwf_plan fftwf_mpi_plan_many_dft(int rnk, const ptrdiff_t *n,
                                      ptrdiff_t howmany, ptrdiff_t block,
                                      ptrdiff_t tblock,
                                      cfloat *_in, cfloat *out,
                                      MPI_Comm comm, int sign, unsigned flags)

    fftwl_plan fftwl_mpi_plan_dft_1d(ptrdiff_t n0, clongdouble *_in, clongdouble *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_2d(ptrdiff_t n0, ptrdiff_t n1,
                                    clongdouble *_in, clongdouble *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                    clongdouble *_in, clongdouble *out,
                                    MPI_Comm comm, int sign, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft(int rnk, const ptrdiff_t *n,
                                 clongdouble *_in, clongdouble *out,
                                 MPI_Comm comm, int sign, unsigned flags)
    fftwl_plan fftwl_mpi_plan_many_dft(int rnk, const ptrdiff_t *n,
                                      ptrdiff_t howmany, ptrdiff_t block, ptrdiff_t tblock,
                                      clongdouble *_in, clongdouble *out,
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
    fftw_plan fftw_mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n,
                                         ptrdiff_t howmany,
                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                         double *_in, cdouble *out,
                                         MPI_Comm comm, unsigned flags)

    fftwf_plan fftwf_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        float *_in, cfloat *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        float *_in, cfloat *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                        float *_in, cfloat *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_r2c(int rnk, const ptrdiff_t *n,
                                     float *_in, cfloat *out,
                                     MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n,
                                          ptrdiff_t howmany,
                                          ptrdiff_t iblock, ptrdiff_t oblock,
                                          float *_in, cfloat *out,
                                          MPI_Comm comm, unsigned flags)

    fftwl_plan fftwl_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        long double *_in, clongdouble *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        long double *_in, clongdouble *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_r2c_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                        long double *_in, clongdouble *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_r2c(int rnk, const ptrdiff_t *n,
                                     long double *_in, clongdouble *out,
                                     MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_many_dft_r2c(int rnk, const ptrdiff_t *n,
                                          ptrdiff_t howmany,
                                          ptrdiff_t iblock, ptrdiff_t oblock,
                                          long double *_in, clongdouble *out,
                                          MPI_Comm comm, unsigned flags)

    # complex to real
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
    fftw_plan fftw_mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n,
                                         ptrdiff_t howmany,
                                         ptrdiff_t iblock, ptrdiff_t oblock,
                                         cdouble *_in, double *out,
                                         MPI_Comm comm, unsigned flags)

    fftwf_plan fftwf_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        cfloat *_in, float *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        cfloat *_in, float *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                        cfloat *_in, float *out,
                                        MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_dft_c2r(int rnk, const ptrdiff_t *n,
                                     cfloat *_in, float *out,
                                     MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n,
                                          ptrdiff_t howmany,
                                          ptrdiff_t iblock, ptrdiff_t oblock,
                                          cfloat *_in, float *out,
                                          MPI_Comm comm, unsigned flags)

    fftwl_plan fftwl_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        clongdouble *_in, long double *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_c2r_2d(ptrdiff_t n0, ptrdiff_t n1,
                                        clongdouble *_in, long double *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_c2r_3d(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2,
                                        clongdouble *_in, long double *out,
                                        MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_dft_c2r(int rnk, const ptrdiff_t *n,
                                     clongdouble *_in, long double *out,
                                     MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_many_dft_c2r(int rnk, const ptrdiff_t *n,
                                          ptrdiff_t howmany,
                                          ptrdiff_t iblock, ptrdiff_t oblock,
                                          clongdouble *_in, long double *out,
                                          MPI_Comm comm, unsigned flags)

    # real to real not supported

    # transposition
    fftw_plan fftw_mpi_plan_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                       double *_in, double *out,
                                       MPI_Comm comm, unsigned flags)
    fftw_plan fftw_mpi_plan_many_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                           ptrdiff_t howmany,
                                           ptrdiff_t block0, ptrdiff_t block1,
                                           double *_in, double *out,
                                           MPI_Comm comm, unsigned flags)

    fftwf_plan fftwf_mpi_plan_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                       float *_in, float *out,
                                       MPI_Comm comm, unsigned flags)
    fftwf_plan fftwf_mpi_plan_many_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                           ptrdiff_t howmany,
                                           ptrdiff_t block0, ptrdiff_t block1,
                                           float *_in, float *out,
                                           MPI_Comm comm, unsigned flags)

    fftwl_plan fftwl_mpi_plan_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                       long double *_in, long double *out,
                                       MPI_Comm comm, unsigned flags)
    fftwl_plan fftwl_mpi_plan_many_transpose(ptrdiff_t n0, ptrdiff_t n1,
                                           ptrdiff_t howmany,
                                           ptrdiff_t block0, ptrdiff_t block1,
                                           long double *_in, long double *out,
                                           MPI_Comm comm, unsigned flags)

    # ordinary execute shares serial interface
    # but new new-array execute() is specialized

    # Double precision complex new array execute
    void fftw_mpi_execute_dft(fftw_plan,
          cdouble *_in, cdouble *_out) nogil

    # Single precision complex new array execute
    void fftwf_mpi_execute_dft(fftwf_plan,
          cfloat *_in, cfloat *_out) nogil

    # Long double precision complex new array execute
    void fftwl_mpi_execute_dft(fftwl_plan,
          clongdouble *_in, clongdouble *_out) nogil

    # Double precision real to complex new array execute
    void fftw_mpi_execute_dft_r2c(fftw_plan,
          double *_in, cdouble *_out) nogil

    # Single precision real to complex new array execute
    void fftwf_mpi_execute_dft_r2c(fftwf_plan,
          float *_in, cfloat *_out) nogil

    # Long double precision real to complex new array execute
    void fftwl_mpi_execute_dft_r2c(fftwl_plan,
          long double *_in, clongdouble *_out) nogil

    # Double precision complex to real new array execute
    void fftw_mpi_execute_dft_c2r(fftw_plan,
          cdouble *_in, double *_out) nogil

    # Single precision complex to real new array execute
    void fftwf_mpi_execute_dft_c2r(fftwf_plan,
          cfloat *_in, float *_out) nogil

    # Long double precision complex to real new array execute
    void fftwl_mpi_execute_dft_c2r(fftwl_plan,
          clongdouble *_in, long double *_out) nogil

    # wisdom functions
    void fftw_mpi_gather_wisdom(MPI_Comm comm)
    void fftwf_mpi_gather_wisdom(MPI_Comm comm)
    void fftwl_mpi_gather_wisdom(MPI_Comm comm)

    void fftw_mpi_broadcast_wisdom(MPI_Comm comm)
    void fftwf_mpi_broadcast_wisdom(MPI_Comm comm)
    void fftwl_mpi_broadcast_wisdom(MPI_Comm comm)

ctypedef ptrdiff_t (*fftw_mpi_generic_local_size_many)(
    int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
    ptrdiff_t block0, MPI_Comm comm,
    ptrdiff_t *local_n0, ptrdiff_t *local_0_start)

ctypedef ptrdiff_t (*fftw_mpi_generic_local_size_many_transposed)(
    int rnk, const ptrdiff_t *n,
    ptrdiff_t howmany,
    ptrdiff_t block0, ptrdiff_t block1,
    MPI_Comm comm,
    ptrdiff_t *local_n0,
    ptrdiff_t *local_0_start,
    ptrdiff_t *local_n1,
    ptrdiff_t *local_1_start)

ctypedef ptrdiff_t (*fftw_mpi_generic_local_size_many_1d)(
    ptrdiff_t n0, ptrdiff_t howmany,
    MPI_Comm comm, int sign, unsigned flags,
    ptrdiff_t *local_ni,
    ptrdiff_t *local_i_start,
    ptrdiff_t *local_no,
    ptrdiff_t *local_o_start)

ctypedef object (*fftw_mpi_generic_local_size)(
                    int rnk, const ptrdiff_t *n, ptrdiff_t howmany,
                    ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
                    ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
                    ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
                    int sign, unsigned int flags)

ctypedef int (*mpi_validator)(np.ndarray input_array, np.ndarray output_array) except -1

ctypedef void (*fftw_mpi_generic_wisdom)(MPI_Comm comm)

ctypedef fftw_plan (*fftw_mpi_generic_plan)(
                    int rank, ptrdiff_t *n,
                    ptrdiff_t howmany,
                    ptrdiff_t block0, ptrdiff_t block1,
                    void *_in, void *_out,
                    MPI_Comm comm,
                    int sign, unsigned int flags)

# MPI-specific flags
cdef enum:
    FFTW_MPI_DEFAULT_BLOCK = 0

cdef enum:
    FFTW_MPI_SCRAMBLED_IN   =  134217728
    FFTW_MPI_SCRAMBLED_OUT  =  268435456
    FFTW_MPI_TRANSPOSED_IN  =  536870912
    FFTW_MPI_TRANSPOSED_OUT = 1073741824
