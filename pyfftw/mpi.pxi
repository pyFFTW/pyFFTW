import mpi4py
from mpi4py.MPI cimport Comm
cimport mpi4py.libmpi as libmpi

# To avoid
#
# error: unknown type name ‘MPI_Message’
#    MPI_Message ob_mpi;
#
# see also https://bitbucket.org/mpi4py/mpi4py/issue/1
cdef extern from 'mpi-compat.h': pass

include 'mpi.pxd'

# Initialize the module
supported_mpi_types = []
IF HAVE_DOUBLE_MPI:
    fftw_mpi_init()
    supported_mpi_types.append('64')
IF HAVE_SINGLE_MPI:
    fftwf_mpi_init()
    supported_mpi_types.append('32')
IF HAVE_LONG_MPI:
    fftwl_mpi_init()
    supported_mpi_types.append('ld')

_build_distributor_list()
_build_mpi_executor_list()
_build_mpi_planner_list()
_build_mpi_wisdom_list()

# adapter functions with uniform set of arguments
# but covariant return type

# (d>1) dimensional
cdef  _fftw_mpi_local_size_many(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
            ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size
    IF HAVE_DOUBLE_MPI:
        local_size = fftw_mpi_local_size_many(rank, n, howmany,
                                                         block0, comm,
                                                         local_n0, local_0_start)
    ELIF HAVE_SINGLE_MPI:
        local_size = fftwf_mpi_local_size_many(rank, n, howmany,
                                                          block0, comm,
                                                          local_n0, local_0_start)
    ELIF HAVE_LONG_MPI:
        local_size = fftwl_mpi_local_size_many(rank, n, howmany,
                                                          block0, comm,
                                                          local_n0, local_0_start)
    ELSE:
        raise NotImplementedError('Could not find any FFTW library that implements fftw*_mpi_local_size_many')


    return local_size, local_n0[0], local_0_start[0]

# (d>1) dimensional transposed
cdef _fftw_mpi_local_size_many_transposed(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_n0, ptrdiff_t *local_0_start,
            ptrdiff_t *local_n1, ptrdiff_t *local_1_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size
    IF HAVE_DOUBLE_MPI:
        local_size = fftw_mpi_local_size_many_transposed(rank, n, howmany, block0, block1,
                                                         comm, local_n0, local_0_start,
                                                         local_n1, local_1_start)
    ELIF HAVE_SINGLE_MPI:
        local_size = fftwf_mpi_local_size_many_transposed(rank, n, howmany, block0, block1,
                                                          comm, local_n0, local_0_start,
                                                          local_n1, local_1_start)
    ELIF HAVE_LONG_MPI:
        local_size = fftwl_mpi_local_size_many_transposed(rank, n, howmany, block0, block1,
                                                          comm, local_n0, local_0_start,
                                                          local_n1, local_1_start)
    ELSE:
        raise NotImplementedError('Could not find any FFTW library that implements fftw*_mpi_local_size_many_transposed')

    return local_size, local_n0[0], local_0_start[0], local_n1[0], local_1_start[0]

# d=1
cdef object _fftw_mpi_local_size_many_1d(
            int rank, const ptrdiff_t *n, ptrdiff_t howmany,
            ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm,
            ptrdiff_t *local_ni, ptrdiff_t *local_i_start,
            ptrdiff_t *local_no, ptrdiff_t *local_o_start,
            int sign, unsigned int flags):

    cdef ptrdiff_t local_size
    IF HAVE_DOUBLE_MPI:
        local_size = fftw_mpi_local_size_many_1d(n[0], howmany, comm, sign, flags,
                                                 local_ni, local_i_start,
                                                 local_no, local_o_start)
    ELIF HAVE_SINGLE_MPI:
        local_size = fftwf_mpi_local_size_many_1d(n[0], howmany, comm, sign, flags,
                                                  local_ni, local_i_start,
                                                  local_no, local_o_start)
    ELIF HAVE_LONG_MPI:
        local_size = fftwl_mpi_local_size_many_1d(n[0], howmany, comm, sign, flags,
                                                  local_ni, local_i_start,
                                                  local_no, local_o_start)
    ELSE:
        raise NotImplementedError('Could not find any FFTW library that implements fftw*_mpi_local_size_many_1d')

    return local_size, local_ni[0], local_i_start[0], local_no[0], local_o_start[0]

#     Planners
#     ========
#
IF HAVE_DOUBLE_MPI:
    # Complex double precision
    cdef void* _fftw_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                       ptrdiff_t block0, ptrdiff_t block1,
                                       void * _in, void * _out,
                                       MPI_Comm comm,
                                       int sign, unsigned int flags):

        return <void *> fftw_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                               <cdouble *> _in, <cdouble *> _out,
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

IF HAVE_SINGLE_MPI:
    # Complex single precision
    cdef void* _fftwf_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

        return <void *> fftwf_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                                <cfloat *> _in, <cfloat *> _out,
                                                comm, sign, flags)

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

IF HAVE_LONG_MPI:
    # Complex long double precision
    cdef void* _fftwl_mpi_plan_many_dft(int rank, const ptrdiff_t *n, ptrdiff_t howmany,
                                        ptrdiff_t block0, ptrdiff_t block1,
                                        void * _in, void * _out,
                                        MPI_Comm comm,
                                        int sign, unsigned int flags):

        return <void *> fftwl_mpi_plan_many_dft(rank, n, howmany, block0, block1,
                                                <clongdouble *> _in, <clongdouble *> _out,
                                                comm, sign, flags)

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
IF HAVE_DOUBLE_MPI:
    # Complex double precision
    cdef void _fftw_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

        fftw_mpi_execute_dft(<fftw_plan>_plan,
                <cdouble *>_in, <cdouble *>_out)

    # real to complex double precision
    cdef void _fftw_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

        fftw_mpi_execute_dft_r2c(<fftw_plan>_plan,
                <double *>_in, <cdouble *>_out)

    # complex to real double precision
    cdef void _fftw_mpi_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

        fftw_mpi_execute_dft_c2r(<fftw_plan>_plan,
                <cdouble *>_in, <double *>_out)

IF HAVE_SINGLE_MPI:
    # Complex single precision
    cdef void _fftwf_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

        fftwf_mpi_execute_dft(<fftwf_plan>_plan,
                <cfloat *>_in, <cfloat *>_out)

    # real to complex single precision
    cdef void _fftwf_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

        fftwf_mpi_execute_dft_r2c(<fftwf_plan>_plan,
                <float *>_in, <cfloat *>_out)

    # complex to real single precision
    cdef void _fftwf_mpi_execute_dft_c2r(void *_plan, void *_in, void *_out) nogil:

        fftwf_mpi_execute_dft_c2r(<fftwf_plan>_plan,
                <cfloat *>_in, <float *>_out)

IF HAVE_LONG_MPI:
    # Complex long double precision
    cdef void _fftwl_mpi_execute_dft(void *_plan, void *_in, void *_out) nogil:

        fftwl_mpi_execute_dft(<fftwl_plan>_plan,
                <clongdouble *>_in, <clongdouble *>_out)

    # real to complex long double precision
    cdef void _fftwl_mpi_execute_dft_r2c(void *_plan, void *_in, void *_out) nogil:

        fftwl_mpi_execute_dft_r2c(<fftwl_plan>_plan,
                <long double *>_in, <clongdouble *>_out)

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
    # Shapes should be integers. numpy < 1.12 created an array of ints for a
    # tuple of ints but newer version always create an array of floats, which
    # causes trouble when used as an index in cython.
    dtype = 'uint64'
    shapes = (np.array(input_shape, dtype), np.array(input_shape, dtype))
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
    for i in range(12):
        mpi_planners[i] = NULL

    IF HAVE_DOUBLE_MPI:
        mpi_planners[0]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft
        mpi_planners[3]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft_r2c
        mpi_planners[6]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_dft_c2r
        mpi_planners[9]  = <fftw_mpi_generic_plan> &_fftw_mpi_plan_many_transpose
    IF HAVE_SINGLE_MPI:
        mpi_planners[1]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft
        mpi_planners[4]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft_r2c
        mpi_planners[7]  = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_dft_c2r
        mpi_planners[10] = <fftw_mpi_generic_plan> &_fftwf_mpi_plan_many_transpose
    IF HAVE_LONG_MPI:
        mpi_planners[2]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft
        mpi_planners[5]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft_r2c
        mpi_planners[8]  = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_dft_c2r
        mpi_planners[11] = <fftw_mpi_generic_plan> &_fftwl_mpi_plan_many_transpose

# Executor table (of size the number of executors)
cdef fftw_generic_execute mpi_executors[9]

cdef fftw_generic_execute * _build_mpi_executor_list():

    # TODO don't repeat 9, compute it at compile time
    for i in range(9):
        mpi_executors[i] = NULL

    IF HAVE_DOUBLE_MPI:
        mpi_executors[0] = <fftw_generic_execute>&_fftw_mpi_execute_dft
        mpi_executors[3] = <fftw_generic_execute>&_fftw_mpi_execute_dft_r2c
        mpi_executors[6] = <fftw_generic_execute>&_fftw_mpi_execute_dft_c2r

    IF HAVE_SINGLE_MPI:
        mpi_executors[1] = <fftw_generic_execute>&_fftwf_mpi_execute_dft
        mpi_executors[4] = <fftw_generic_execute>&_fftwf_mpi_execute_dft_r2c
        mpi_executors[7] = <fftw_generic_execute>&_fftwf_mpi_execute_dft_c2r

    IF HAVE_LONG_MPI:
        mpi_executors[2] = <fftw_generic_execute>&_fftwl_mpi_execute_dft
        mpi_executors[5] = <fftw_generic_execute>&_fftwl_mpi_execute_dft_r2c
        mpi_executors[8] = <fftw_generic_execute>&_fftwl_mpi_execute_dft_c2r

cdef fftw_mpi_generic_wisdom mpi_wisdom[6]
cdef fftw_mpi_generic_wisdom * _build_mpi_wisdom_list():
    # TODO don't repeat 6, compute it at compile time
    for i in range(6):
        mpi_wisdom[i] = NULL

    IF HAVE_DOUBLE_MPI:
        mpi_wisdom[0] = <fftw_mpi_generic_wisdom> &fftw_mpi_gather_wisdom
        mpi_wisdom[3] = <fftw_mpi_generic_wisdom> &fftw_mpi_broadcast_wisdom

    IF HAVE_SINGLE_MPI:
        mpi_wisdom[1] = <fftw_mpi_generic_wisdom> &fftwf_mpi_gather_wisdom
        mpi_wisdom[4] = <fftw_mpi_generic_wisdom> &fftwf_mpi_broadcast_wisdom

    IF HAVE_LONG_MPI:
        mpi_wisdom[2] = <fftw_mpi_generic_wisdom> &fftwl_mpi_gather_wisdom
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

# Every access to MPI functions should go via this dictionary
# any library that's missing gives rise to a KeyError instead of segfault
# if a NULL function is called
cdef object _mpi_scheme_functions = {}
IF HAVE_DOUBLE_MPI:
    _mpi_scheme_functions.update({
    ('c2c', '64'): {'planner': 0, 'executor':0, 'generic_precision':0,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('r2c', '64'): {'planner':3, 'executor':3, 'generic_precision':0,
                    'validator': 0, 'fft_shape_lookup': 0,
                    'output_shape': 0},
    ('c2r', '64'): {'planner':6, 'executor':6, 'generic_precision':0,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1}})
IF HAVE_SINGLE_MPI:
    _mpi_scheme_functions.update({
    ('c2c', '32'): {'planner':1, 'executor':1, 'generic_precision':1,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('r2c', '32'): {'planner':4, 'executor':4, 'generic_precision':1,
                    'validator': 0, 'fft_shape_lookup':  0,
                    'output_shape': 0},
    ('c2r', '32'): {'planner':7, 'executor':7, 'generic_precision':1,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1}})
IF HAVE_LONG_MPI:
    _mpi_scheme_functions.update({
    ('c2c', 'ld'): {'planner':2, 'executor':2, 'generic_precision':2,
                    'validator': 2, 'fft_shape_lookup': 2,
                    'output_shape': 2},
    ('r2c', 'ld'): {'planner':5, 'executor':5, 'generic_precision':2,
                    'validator': 0, 'fft_shape_lookup':  0,
                    'output_shape': 0},
    ('c2r', 'ld'): {'planner':8, 'executor':8, 'generic_precision':2,
                    'validator': 1, 'fft_shape_lookup': 1,
                    'output_shape': 1}})

def mpi_scheme_functions(scheme):
    try:
        return _mpi_scheme_functions[scheme]
    except KeyError:
        msg = "The scheme '%s' is not supported." % str(scheme)
        if scheme[1] in _all_types:
            msg += "\nRebuild pyfftw with support for the data type '%s'!" % scheme[1]
        raise NotImplementedError(msg)

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
    '''Turn ``input_shape`` into integers, then ensure only positive integers present.'''
    dtype = 'uint64'
    if np.iterable(input_shape):
        i = np.array(input_shape, dtype)
    else:
        i = input_shape * np.ones(1, dtype)
    if not len(i):
        raise ValueError('Empty input shape')
    if (i <= 0).any():
        raise ValueError('Invalid input shape: %s' % input_shape)
    return tuple(i)

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

    If the ``*put_array`` arguments are set to ``None``, arrays of the
    proper size to this MPI rank are allocated with SIMD alignment
    when possible. Note that these arrays may need to be slightly
    larger than what one might expect due to padding or extra memory
    needed in intermediate FFTW steps. This depends on the transform,
    dimensions, fftw implementation etc. Resist to guess!  The arrays
    are accessible from the returned plan as attributes
    ``plan.*put_chunk``. Note that the physical size of a memory chunk
    is given by ``plan.output_chunk.size``, which can be larger than
    than ``plan.output_array.size`` indicates. For more details, see
    the FFTW manual on Distributed-memory FFTW with MPI at
    `<http://fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html>`_

    Construct an in-place transform with ``output_chunk='INPUT'`` This
    saves about half the memory and can significantly speed up the
    transform. Think twice before *not* using this option.

    All other arguments are passed forward to func:`~pyfftw.local_size`
    and :class:`~pyfftw.FFTW_MPI` in order to allocate byte-aligned
    input/output arrays and to create the appropriate FFT plan.

    The ``direction`` argument is ignored for r2c and c2r transforms,
    and should be either 'FFTW_FORWARD' or 'FFTW_BACKWARD' for c2c.

    Return a :class:`~pyfftw.FFTW_MPI` object ``plan``. Execute the
    ``plan`` as ``plan(*args, *kwargs)``;
    cf. :meth:`~pyfftw.FFTW_MPI.__call__`.

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

    functions = mpi_scheme_functions(scheme)

    # need padding for r2c; cf. FFTW manual 6.5 'Multi-dimensional MPI DFTs of Real Data'
    local_input_shape, local_output_shape = mpi_local_shapes[functions['fft_shape_lookup']](input_shape, res, flags)

    ###
    # Check or allocate input array
    ###
    if input_chunk is None:
        # due to extra bytes for intermediate step, the conceptual size may
        # differ from the actual memory accessed. The extra bytes come after the last
        # input element
        input_chunk = n_byte_align_empty(n_elements_in, simd_alignment, dtype=input_dtype)

    if output_chunk is 'INPUT':
        output_chunk = np.frombuffer(input_chunk.data, output_dtype)
    elif output_chunk is None:
        # treat this as a 1D chunk of memory
        output_chunk = n_byte_align_empty(n_elements_out, simd_alignment, dtype=output_dtype)

    # check both input and output
    for s in ('in', 'out'):
        name = s + 'put'
        _mpi_validate_array(locals()[name + '_chunk'],
                            locals()['n_elements_' + s],
                            name)

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

    if (np.array(input_shape) <= 0).any():
        raise ValueError('Invalid input shape. '
                         'Need at least one element in every dimension: ' + str(input_shape))

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

        # MPI stuff
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

    # todo is this still useful with MPI?
    @property
    def simd_aligned(self):
        '''
        Return whether or not this FFTW object requires simd aligned
        input and output data.
        '''
        return self._simd_allowed

    @property
    def input_alignment(self):
        '''
        Returns the byte alignment of the input arrays for which the
        :class:`~pyfftw.FFTW_MPI` object was created.

        Input array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised, or
        a copy being made if the :meth:`~pyfftw.FFTW_MPI.__call__`
        interface is used.
        '''
        return self._input_array_alignment

    @property
    def output_alignment(self):
        '''
        Returns the byte alignment of the output arrays for which the
        :class:`~pyfftw.FFTW` object was created.

        Output array updates with arrays that are not aligned on this
        byte boundary will result in a ValueError being raised.
        '''
        return self._output_array_alignment

    @property
    def flags(self):
        '''
        Return flags used to construct the FFTW object.

        This includes flags that were added during initialisation.
        '''
        return tuple(self._flags_used)

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

    @property
    def input_shape(self):
        '''Return the global shape of the input array that spans multiple MPI ranks for
        which the FFT is planned. Note that this usually differs from the shape
        of ``input_array`` local to this MPI rank because the data is distributed, and every rank has only a portion of the data.

        '''
        return self._input_shape

    @property
    def output_shape(self):

        '''Return the global shape of the output array that spans multiple MPI ranks for
        which the FFT is planned. Note that this usually differs from the shape
        of ``output_chunk`` local to this MPI rank.

        '''
        return self._output_shape

    def _get_array(self, mode, transform):
        '''Return a view of the input/output buffer with the right conceptual dimensions; i.e.,
        the padding elements are hidden.

        mode: Either input or output

        transform: Index of the transform in the range [0:howmany]. 0 always
        points to the first array, higher indices only make sense if multiple
        transforms are computed at once.
        '''
        if mode == 'input':
            local_shape_padded = self._local_input_shape_padded
            local_shape = self._local_input_shape
            chunk = self._input_chunk
        elif mode == 'output':
            local_shape_padded = self._local_output_shape_padded
            local_shape = self._local_output_shape
            chunk = self._output_chunk
        else:
            raise KeyError('Unknown mode' + str(mode))

        if transform + 1 > self._howmany:
            raise IndexError('Invalid index %d exceeds number of transforms %d' % (transform, self._howmany))
        if not local_shape:
            raise AttributeError('MPI rank %d does not have any %s data' % (self._MPI_rank, mode))

        # transform from single to many transforms
        shape = np.array(local_shape_padded)
        shape[-1] *= self._howmany

        # first select only as many elements as needed,
        # then create a view with right dimensions for many transforms,
        # then pick out only one transform and ignore padding elements in last dimension
        arr = chunk[0:np.prod(shape)].reshape(shape)
        return arr[..., transform:local_shape[-1] * self._howmany:self._howmany]

    def get_input_array(self, transform=0):
        '''Return a view of the input buffer with the right conceptual dimensions; i.e.,
        the padding elements are hidden. If multiple transforms are done at
        once, select the array using ``transform``.

        Default: return the first array.

        '''
        return self._get_array(mode='input', transform=transform)

    def get_output_array(self, transform=0):
        '''Return a view of the output buffer with the right conceptual dimensions; i.e.,
        the padding elements are hidden. If multiple transforms are done at
        once, select the array using ``transform``.

        Default: return the first array.

        '''
        return self._get_array(mode='output', transform=transform)

    @property
    def input_array(self):
        '''Short cut to the first input array.'''
        return self.get_input_array(0)

    @property
    def output_array(self):
        '''Short cut to the first output array.'''
        return self.get_output_array(0)

    @property
    def input_dtype(self):
        '''
        Return the dtype of the input array for which the FFT is planned.
        '''
        return self._input_dtype

    @property
    def output_dtype(self):
        '''
        Return the shape of the output array for which the FFT is planned.
        '''
        return self._output_dtype

    @property
    def direction(self):
        '''
        Return the planned FFT direction. Either `'FFTW_FORWARD'` or
        `'FFTW_BACKWARD'`.
        '''
        return directions_lookup[self._direction]

    @property
    def local_n_elements(self):
        '''Return the total number of elements, including padding and possibly extra
        bytes for intermediate steps, that need to be allocated for input/output
        on this MPI rank.

        '''
        return self._local_n_elements

    @property
    def local_n0(self):
        '''Return the number of elements in the first dimension this MPI rank operates
        on.

        '''
        return self._local_n0

    @property
    def local_0_start(self):
        '''Return the offset in the first dimension this MPI rank operates on.

        '''
        return self._local_0_start

    @property
    def local_n1(self):
        '''Return the number of elements in the second dimension this MPI rank operates
        on. The return value is 0 unless the flag 'FFTW_MPI_TRANSPOSED_OUT' has been set.

        '''
        return self._local_n1

    @property
    def local_1_start(self):
        '''Return the offset in the first dimension this MPI rank operates on.

        '''
        return self._local_1_start

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
        else:
            return None

    @property
    def threads(self):
        '''The number of threads to use for the execution of the plan.'''
        return self._threads

    # TODO why `*args, **kwargs`? They are not used at all
    # TODO Tune default flags for high performance:
    def __cinit__(self, input_shape, input_chunk, output_chunk,
                  block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK',
                  direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
                  unsigned int threads=1, planning_timelimit=None,
                  n_transforms=1, comm=None,
                  *args, **kwargs):

        # TODO Check or warn about prime n0

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

        # save communicator
        cdef int ierr = 0
        self._comm = extract_communicator(comm)
        ierr = libmpi.MPI_Comm_rank(self._comm, &self._MPI_rank)
        if ierr:
            raise RuntimeError('MPI_Comm_rank returned %d' % ierr)

        # and the rank of this process for debugging
        cdef int comm_size = 0
        ierr = libmpi.MPI_Comm_size(self._comm, &comm_size)
        if ierr:
            raise RuntimeError('MPI_Comm_size returned %d' % ierr)

        input_dtype = input_chunk.dtype
        output_dtype = output_chunk.dtype

        try:
            scheme = fftw_schemes[(input_dtype, output_dtype)]
        except KeyError:
            raise TypeError('Invalid scheme: '
                    'The output array and input array dtypes '
                    'do not correspond to a valid fftw scheme.')

        self._input_dtype = input_dtype
        self._output_dtype = output_dtype

        functions = mpi_scheme_functions(scheme)

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
        self._rank = len(input_shape)
        self._howmany = n_transforms

        # passing FFTW_MPI_TRANSPOSED_IN and FFTW_MPI_TRANSPOSED_OUT
        # just swaps first two dimensions
        if self._rank > 1 and \
           'FFTW_MPI_TRANSPOSED_IN'  in flags and \
           'FFTW_MPI_TRANSPOSED_OUT' in flags:
            input_shape[0], input_shape[1] = input_shape[1], input_shape[0]

        if self._rank == 1 and \
           ('FFTW_MPI_TRANSPOSED_IN' in flags or 'FFTW_MPI_TRANSPOSED_OUT' in flags):
                raise ValueError('Invalid flag: FFTW_MPI_TRANSPOSED_* does not  apply in 1d')

        # need local_size to tell us how many elements in first dimension are
        # processed on this MPI rank to determine the right shape, and to let
        # the user know at what offset in the global this rank is in the first
        # dimension
        local_size_res = local_size(_mpi_local_size_input_shape(input_shape, scheme[0]),
                                    n_transforms, block0, block1, flags, direction, comm)
        self._local_input_shape, self._local_output_shape = mpi_local_shapes[functions['fft_shape_lookup']] \
                                                                (input_shape, local_size_res, flags)

        # Now we can validate the arrays
        for name in ('input', 'output'):
            _mpi_validate_array(locals()[name + '_chunk'],
                                local_size_res[0], name)

        self._input_shape = input_shape
        self._output_shape = mpi_output_shape[functions['output_shape']](input_shape)

        self._local_input_shape_padded, self._local_output_shape_padded = \
        mpi_local_shapes_padded[functions['fft_shape_lookup']] (input_shape, local_size_res, flags)

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

        # thread-parallel execution
        self._threads = threads

        try:
            mpi4py_rc_threads = mpi4py.rc.threads
        except AttributeError:
            mpi4py_rc_threads = mpi4py.rc.threaded

        if self._threads > 1 and (not mpi4py_rc_threads) or (mpi4py.rc.thread_level == 'single'):
            warnings.warn('MPI was not initialized with proper support for threads. '
                          'FFTW needs at least MPI_THREAD_FUNNELED. Proceeding with a single thread.')
            threads = 1
        self._nthreads_plan_setter(threads)

        set_timelimit_func(_planning_timelimit)

        # Finally, construct the plan
        self._plan = self._fftw_planner(
            self._rank, self._dims.data, self._howmany,
            self._block0, self._block1,
            <void *>np.PyArray_DATA(self._input_chunk),
            <void *>np.PyArray_DATA(self._output_chunk),
            self._comm, self._direction, self._flags)

        if self._plan is NULL:
            raise RuntimeError('The data configuration has an uncaught error that led ' +
                               'to FFTW returning an invalid plan in MPI rank %d.' % self._MPI_rank +
                               'Please report this as a bug.')

    def __init__(self, input_shape, input_chunk, output_chunk,
                  block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK',
                  direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
                  unsigned int threads=1, planning_timelimit=None,
                  n_transforms=1, comm=None,
                  *args, **kwargs):
        '''**Arguments**:

        * ``input_shape`` is the *global* shape of the input array,
          that is the shape of the array that one would transform with
          the serial interface on a single machine. This shape is
          identical on every MPI rank participating in the transform.

        * ``input_chunk`` and ``output_chunk`` should be numpy arrays.
          The contents of these arrays could be destroyed by the
          planning process during initialisation. Information on
          supported dtypes for the arrays is :ref:`given below
          <scheme_table>`. The chunks may very well point to the same
          memory to perform an in-place transform.

        * ``block0, block1`` are the block sizes in the first/second
          dimension. It is the number of elements that this MPI rank
          operates on during the transform. Applies only to
          multidimensional transforms.

        * ``direction`` should be a string and one of ``'FFTW_FORWARD'``
          or ``'FFTW_BACKWARD'``, which dictate whether to take the
          DFT (forwards) or the inverse DFT (backwards) respectively
          (specifically, it dictates the sign of the exponent in the
          DFT formulation).

          Note that only the Complex schemes allow a free choice
          for ``direction``. The direction *must* agree with the
          the :ref:`table below <scheme_table>` if a Real scheme
          is used, otherwise a ``ValueError`` is raised.

        .. _FFTW_MPI_flags:

        * ``flags`` is a list of strings corresponding to the flags
          that FFTW allows for the planners. In addition to the the
          flags described in :ref:`below<FFTW_flags>`, these
          MPI-specific flags are supported:

          * ``FFTW_MPI_SCRAMBLED_OUT, FFTW_MPI_SCRAMBLED_IN``: valid
            for 1d transforms only, these flags indicate that the
            output/input of the transform are in an undocumented
            “scrambled” order. A forward ``FFTW_MPI_SCRAMBLED_OUT``
            transform can be inverted by a backward
            ``FFTW_MPI_SCRAMBLED_IN`` (times the usual 1/N
            normalization). See `FFTW doc <http://fftw.org/fftw3_doc/One_002ddimensional-distributions.html#One_002ddimensional-distributions>`_

          * ``FFTW_MPI_TRANSPOSED_OUT, FFTW_MPI_TRANSPOSED_IN``: valid
            for multidimensional (rnk > 1) transforms only, these
            flags specify that the output or input of an n0 × n1 × n2
            × … × nd-1 transform is transposed to n1 × n0 × n2 ×…×
            nd-1. This speeds up the calculation as one less
            synchronization between all MPI ranks is needed. See `FFTW
            doc
            <http://fftw.org/fftw3_doc/Transposed-distributions.html#Transposed-distributions>`_

          The `FFTW planner flags documentation
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          has more information about the various flags and their impact.
          Note that only the flags documented here are supported.

        * ``threads`` tells the wrapper how many threads to use
          when invoking FFTW, with a default of 1. If the number
          of threads is greater than 1, then the GIL is released
          by necessity.

        * ``planning_timelimit`` is a float that indicates to the
          underlying FFTW planner the maximum number of seconds it
          should spend planning the FFT. This is a rough estimate and
          corresponds to calling of ``fftw_set_timelimit()`` (or an
          equivalent dependent on type) in the underlying FFTW
          library. If ``None`` is set, the planner will run
          indefinitely until all the planning modes allowed by the
          flags have been tried. See the `FFTW planner flags page
          <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_
          for more information on this.

        * ``n_transforms`` is the number of same-size arrays to
          transform simultaneously in one go. Suppose you have three
          arrays ``x,y,z`` you want to Fourier transform. Then you can
          save some MPI communication overhead by doing the
          ``n_transforms=3`` transforms at once; the arrays are
          expected to reside in an interleaved format such that in
          memory they are ordered as ``x[0], y[0], z[0], x[1], y[1],
          z[1]...``. So ``n_transforms`` is the number of elements
          between the first and second element of each individual
          array. All arrays must be of the same data type and
          length. Access to array ``n`` is available via
          ``get_input_array(n)`` and ``get_output_array(n)``.

        * ``comm`` is expected to be an instance of
          ``mpi4py.libmpi.MPI_Comm``. If ``None``, default to the
          world communicator. ``comm`` is used in all communication
          between MPI ranks within FFTW.

        .. _fftw_mpi_schemes:

        **Schemes**

        The currently supported schemes are as follows:

        .. _scheme_mpi_table:

        +----------------+-----------------------+------------------------+-----------+
        | Type           | ``input_chunk.dtype`` | ``output_chunk.dtype`` | Direction |
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

        ``clongdouble`` typically maps directly to ``complex256`` or
        ``complex192``, and ``longdouble`` to ``float128`` or
        ``float96``, dependent on platform.

        FFTW MPI supports and encourages in-place transforms. In that
        case, the sizes (in units of the respective data type) of the
        ``*chunk``s should be as follows:

        * For a complex transform, ``output_chunk.size ==
          input_chunk.size``
        * For a real-to-complex transform in the
          forward direction ``2 * output_chunk.size ==
          input_chunk.size``
        * For a complex-to-real transform in the backward direction
          ``output_chunk.size == 2 * input_chunk.size``

        Only transforms over all dimensions of ``input_shape`` are
        considered. Hence the memory in ``*chunk`` must be contiguous.

        The shapes for the real transforms corresponds to those
        stipulated by the FFTW library. Further information can be
        found in the FFTW documentation on the `real DFT
        <http://www.fftw.org/fftw3_doc/Guru-Real_002ddata-DFTs.html>`_.

        The data is expected to lie at certain positions of the
        array. There is padding and possibly extra bytes in the memory
        chunk, so before manipulating the memory directly, consult the
        FFTW MPI docs. It is strongly recommended to use the plan's
        attributes ``input_array`` and ``output_array`` to get a view
        on the memory chunk that allows natural read/write access to
        the right locations. Note that due to padding in the last
        column, the view's memory need *not* be contiguous.

        What is calculated is exactly what FFTW calculates.
        Notably, this is an unnormalized transform so should
        be scaled as necessary (fft followed by ifft will scale
        the input by N, the product of the dimensions along which
        the DFT is taken). For further information, see the
        `FFTW documentation
        <http://www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html>`_.

        The FFTW library benefits greatly from the beginning of
        ``input_chunk`` being aligned on the correct byte boundary,
        enabling SIMD instructions. By default, if the data begins on
        such a boundary, then FFTW will be allowed to try and enable
        SIMD instructions. This means that all future changes to the
        data arrays will be checked for similar alignment. SIMD
        instructions can be explicitly disabled by setting the
        FFTW_UNALIGNED flags, to allow for updates with unaligned
        data. Usually this should not be necessary.

        :func:`~pyfftw.n_byte_align` and
        :func:`~pyfftw.n_byte_align_empty` are two methods
        included with this module for producing aligned arrays.

        The optimum alignment for the running platform is provided
        by :data:`pyfftw.simd_alignment`, though a different alignment
        may still result in some performance improvement. For example,
        if the processor supports AVX (requiring 32-byte alignment) as
        well as SSE (requiring 16-byte alignment), then if the array
        is 16-byte aligned, SSE will still be used.

        '''
        pass

    def __dealloc__(self):
        if not self._plan == NULL:
            self._fftw_destroy(self._plan)

    def __call__(self, input_chunk=None, output_chunk=None,
            normalise_idft=True):
        '''__call__(input_chunk=None, output_chunk=None, normalise_idft=True)

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
        flag has no effect.

        When ``input_chunk`` is something other than None, then the passed in
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

        Check ``self.has_output()``, then obtain the result of the FFT
        computed on this rank via
        :attr:`~pyfftw.FFTW_MPI.get_output_array`. If you need the
        data to persist longer than a subsequent call, you should make
        a copy before the next execution of the plan.

        '''

        if input_chunk is not None or output_chunk is not None:

            if input_chunk is None:
                input_chunk = self._input_chunk

            if output_chunk is None:
                output_chunk = self._output_chunk

            if not isinstance(input_chunk, np.ndarray):
                copy_needed = True
            elif (not input_chunk.dtype == self._input_dtype):
                copy_needed = True
            elif (not input_chunk.strides == self._input_strides):
                copy_needed = True
            elif not (<intptr_t>np.PyArray_DATA(input_chunk)
                    % self.input_alignment == 0):
                copy_needed = True
            else:
                copy_needed = False

            if copy_needed:

                if not isinstance(input_chunk, np.ndarray):
                    input_chunk = np.asanyarray(input_chunk)

                if not input_chunk.shape == self._input_shape:
                    raise ValueError('Invalid input shape: '
                            'The new input array should be the same shape '
                            'as the input array used to instantiate the '
                            'object.')

                self._input_chunk[:] = input_chunk

                if output_chunk is not None:
                    # No point wasting time if no update is necessary
                    # (which the copy above may have avoided)
                    input_chunk = self._input_chunk
                    self.update_arrays(input_chunk, output_chunk)

            else:
                self.update_arrays(input_chunk, output_chunk)

        self.execute()

        if self._direction == FFTW_BACKWARD and normalise_idft:
            self._output_chunk *= self._normalisation_scaling

    cpdef update_arrays(self,
            new_input_chunk, new_output_chunk):
        '''update_arrays(new_input_chunk, new_output_chunk)

        Update the arrays upon which the DFT is taken.

        The new arrays should be of the same sizes and dtypes as the
        originals. If the original data was aligned so as to allow
        SIMD instructions (e.g. by being aligned on a 16-byte
        boundary), then the new array must also be aligned so as to
        allow SIMD instructions (assuming, of course, that the
        ``FFTW_UNALIGNED`` flag was not enabled).

        The byte alignment requirement extends to requiring natural
        alignment in the non-SIMD cases as well, but this is much less
        stringent as it simply means avoiding arrays shifted by, say,
        a single byte (which invariably takes some effort to
        achieve!).

        If all these conditions are not met, a ``ValueError`` will
        be raised and the data will *not* be updated (though the
        object will still be in a sane state).

        '''
        if not isinstance(new_input_chunk, np.ndarray):
            raise ValueError('Invalid input array: '
                    'The new input array needs to be an instance '
                    'of numpy.ndarray')

        if not isinstance(new_output_chunk, np.ndarray):
            raise ValueError('Invalid output array '
                    'The new output array needs to be an instance '
                    'of numpy.ndarray')

        if not (<intptr_t>np.PyArray_DATA(new_input_chunk) %
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

        if not new_input_chunk.dtype == self._input_dtype:
            raise ValueError('Invalid input dtype: '
                    'The new input array is not of the same '
                    'dtype as was originally planned for.')

        if not new_output_chunk.dtype == self._output_dtype:
            raise ValueError('Invalid output dtype: '
                    'The new output array is not of the same '
                    'dtype as was originally planned for.')

        for name in ('input', 'output'):
            _mpi_validate_array(locals()['new_' + name + '_chunk'],
                                self._local_n_elements, name)

        self._update_arrays(new_input_chunk, new_output_chunk)

    cdef _update_arrays(self, np.ndarray new_input_chunk, np.ndarray new_output_chunk):
        ''' A C interface to the update_arrays method that does not
        perform any checks on strides being correct and so on.
        '''
        self._input_chunk = new_input_chunk
        self._output_chunk = new_output_chunk

    cpdef execute(self):
        '''Execute the plan.'''
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

def broadcast_wisdom(Comm comm=None):
    IF HAVE_DOUBLE:
        fftw_mpi_broadcast_wisdom(extract_communicator(comm))
    IF HAVE_SINGLE:
        fftwf_mpi_broadcast_wisdom(extract_communicator(comm))
    IF HAVE_LONG:
        fftwl_mpi_broadcast_wisdom(extract_communicator(comm))

def gather_wisdom(Comm comm=None):
    IF HAVE_DOUBLE:
        fftw_mpi_gather_wisdom(extract_communicator(comm))
    IF HAVE_SINGLE:
        fftwf_mpi_gather_wisdom(extract_communicator(comm))
    IF HAVE_LONG:
        fftwl_mpi_gather_wisdom(extract_communicator(comm))
