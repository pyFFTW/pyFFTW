
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from pyfftw3 cimport _fftw_iodim, fftw_iodim, fftwf_plan
cimport pyfftw3

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
            <float complex *>_in, <float complex *>_out,
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

#    Execute
#    =======
#
# Double precision
cdef void _fftw_execute_dft(void *_plan, void *_in, void *_out):

    fftw_execute_dft(<fftw_plan>_plan, 
            <float complex *>_in, <float complex *>_out)

# Single precision
cdef void _fftwf_execute_dft(void *_plan, void *_in, void *_out):

    fftwf_execute_dft(<fftwf_plan>_plan, 
            <float complex *>_in, <float complex *>_out)


#    Destroy
#    =======
#
# Double precision
cdef void _fftw_destroy_plan(void *_plan):

    fftw_destroy_plan(<fftw_plan>_plan)

# Single precision
cdef void _fftwf_destroy_plan(void *_plan):

    fftwf_destroy_plan(<fftwf_plan>_plan)


# The External Interface
# ======================
#
cdef class ComplexFFTW:
    ''' Class for computing the complex N-Dimensional FFT 
    of an array using FFTW, along arbitrary axes and with
    arbitrary data striding.
    '''

    # Each of these function pointers simply
    # points to a chosen fftw wrapper function
    cdef fftw_generic_plan_guru __fftw_planner
    cdef fftw_generic_execute __fftw_execute
    cdef fftw_generic_destroy_plan __fftw_destroy

    # The plan is typecast when it is created or used
    # within the wrapper functions
    cdef void *__plan

    cdef np.ndarray __input_array
    cdef np.ndarray __output_array
    cdef int __direction
    cdef int __flags
    cdef bint __simd_allowed

    cdef object __input_strides
    cdef object __output_strides
    cdef object __shape

    cdef int __rank
    cdef _fftw_iodim *__dims
    cdef int __howmany_rank
    cdef _fftw_iodim *__howmany_dims

    def __cinit__(self, input_array, output_array, axes=[-1],
            direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        
        # Initialise the pointers that need to be freed
        self.__plan = NULL
        self.__dims = NULL
        self.__howmany_dims = NULL
        
        if not (output_array.shape == input_array.shape):
            raise ValueError('The output array should be the same shape as '+\
                    'the input array.')

        if not input_array.dtype == 'complex64':
            raise ValueError('The input array should be of type complex64')

        if not output_array.dtype == 'complex64':
            raise ValueError('The output array should be of type complex64')

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

        self.__direction = directions[direction]
        self.__shape = np.array(input_array.shape)
        
        self.__fftw_planner = \
                <fftw_generic_plan_guru>&_fftwf_plan_guru_dft
        
        self.__fftw_destroy = \
                <fftw_generic_destroy_plan>&_fftwf_destroy_plan
        
        self.__fftw_execute = \
                <fftw_generic_execute>&_fftwf_execute_dft

        self.__flags = 0 
        for each_flag in flags:
            self.__flags |= flag_dict[each_flag]

        self.__input_array = input_array
        self.__output_array = output_array
        
        _axes = np.array(axes)
        
        # in_shape = out_shape (or else an exception was raised)
        in_shape = np.array(input_array.shape)
        out_shape = np.array(output_array.shape)
        
        # Set the negative entries to their actual index (use the size
        # of the shape array for this)
        _axes[_axes<0] = _axes[_axes<0] + len(in_shape)

        if (_axes >= len(in_shape)).any() or (_axes < 0).any():
            raise ValueError('The axes list cannot contain invalid axes.')

        # We want to make sure that the axes list contains unique entries
        _axes = np.unique(_axes)

        # Now get the axes along which the FFT is *not* taken
        _not_axes = np.setdiff1d(np.arange(0,len(in_shape)), _axes)

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

        # Fill in the stride and shape information
        cdef int i
        for i in range(0, self.__rank):
            self.__dims[i]._n = in_shape[_axes][i]
            self.__dims[i]._is = self.__input_strides[_axes][i]
            self.__dims[i]._os = self.__output_strides[_axes][i]

        for i in range(0, self.__howmany_rank):
            self.__howmany_dims[i]._n = in_shape[_not_axes][i]
            self.__howmany_dims[i]._is = self.__input_strides[_not_axes][i]
            self.__howmany_dims[i]._os = self.__output_strides[_not_axes][i]

        # Finally, construct the plan
        self.__plan = self.__fftw_planner(
            self.__rank, <fftw_iodim *>self.__dims,
            self.__howmany_rank, <fftw_iodim *>self.__howmany_dims,
            <void *>self.__input_array.data,
            <void *>self.__output_array.data,
            self.__direction, self.__flags)

    def __init__(self, input_array, output_array, axes=[-1], 
            direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        '''
        Instantiate and return a class representing a planned
        FFTW operation.

        Run the FFT (or the iFFT) by calling the execute() method
        (with no arguments) on the returned class.

        `input_array' and `output_array' are assumed to be numpy
        arrays, and it is checked that they are both of the
        same shape (though they need not share the same alignment
        or striding in memory). Once this object has been created,
        the arrays can be changed by calling the
        update_arrays(new_input_array=None, new_output_array=None)
        method. See the documentation on that method for more
        information.

        `axes' describes along which axes the FFT should be taken.
        This should be a valid list of axes. Repeated axes are 
        only transformed once. Invalid axes will raise an 
        exception. This argument is equivalent to the same
        argument in numpy.fft.fftn.

        `direction' should be a string and one of FFTW_FORWARD 
        or FFTW_BACKWARD, which dictate whether to take the
        FFT or the inverse FFT respectively (specifically, it
        dictates the sign of the exponent in the DFT formulation).

        `flags' is a list of strings and is a subset of the 
        flags that FFTW allows for the planners. Specifically, 
        FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT and 
        FFTW_EXHAUSTIVE are supported. These describe the 
        increasing amount of effort spent during the planning 
        stage to create the fastest possible transform. 
        Usually, FFTW_MEASURE is a good compromise and is the 
        default. In addition FFTW_UNALIGNED is supported. 
        This tells FFTW not to assume anything about the 
        alignment of the data and disabling any SIMD capability 
        (see below).

        What is calculated is exactly what FFTW calculates. 
        Notably, this is an unnormalized transform so should 
        be scaled as necessary (fft followed by ifft will scale 
        the input by N, the product of the dimensions along which
        the FFT is taken).

        The content of the arrays that are passed in will be 
        destroyed by the planning process during initialisation.

        The FFTW library benefits greatly from the beginning of each
        FFT axes being aligned on a 16-byte boundary, which enables
        SIMD instructions. By default, if the data begins on a 16-
        byte boundary, then FFTW will be allowed to try and enable
        SIMD instructions. This means that all future changes to
        the data arrays will be checked for similar alignment. SIMD
        instructions can be explicitly disabled by setting the
        FFTW_UNALIGNED flags, to allow for updates with unaligned
        data.
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
        ''' Update the arrays upon which the FFT is taken.

        The new arrays should be of the same dtype as the original, and
        should have the same strides between axes. If the original
        data was aligned so as to allow SIMD instructions (by being 
        aligned on a 16-byte boundary), then the new array
        must also be aligned in the same way.
        
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

        new_input_shape = np.array(new_input_array.shape)
        new_output_shape = np.array(new_output_array.shape)
        new_input_strides = \
                np.array(new_input_array.strides)/new_input_array.itemsize
        new_output_strides = \
                np.array(new_output_array.strides)/new_input_array.itemsize

        if not (new_input_shape == new_output_shape).all():
            raise ValueError('The output array should be the same shape as '+\
                    'the input array.')

        if not (new_input_shape == self.__shape).all():
            raise ValueError('The new arrays should be the same shape as '+\
                    'the arrays used to instantiate the object.')

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
        ''' Execute the FFTW plan.
        '''
        self.__fftw_execute(self.__plan,
                <void *>self.__input_array.data,
                <void *>self.__output_array.data)


cpdef n_byte_align_empty(shape, n, dtype='float64', order='C'):
    ''' Function at returns an empty numpy array
    that is n-byte aligned.

    The alignment is given by the second argument, n.
    The rest of the arguments are as per numpy.empty.
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
    ''' Function that takes a numpy array and 
    checks it is aligned on an n-byte boundary, 
    where n is a passed parameter. If it is, 
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
