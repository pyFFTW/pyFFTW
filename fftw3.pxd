
cimport numpy as np

cdef extern from "complex.h":
    pass

ctypedef struct _fftw_iodim:
    int _n
    int _is
    int _os

cdef extern from "fftw3.h":

    ctypedef struct fftw_plan_struct:
        pass

    ctypedef fftw_plan_struct *fftwf_plan

    ctypedef struct fftw_iodim:
        pass

    fftwf_plan fftwf_plan_guru_dft(
            int rank, fftw_iodim *dims,
            int howmany_rank, fftw_iodim *howmany_dims,
            float complex *_in, float complex *_out,
            int sign, unsigned flags)
    
    void fftwf_execute_dft(fftwf_plan,
          float complex *_in, float complex *out)

    void fftwf_execute(fftwf_plan)

    void fftwf_destroy_plan(fftwf_plan)

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
