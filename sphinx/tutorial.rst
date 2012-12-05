A Short Tutorial
================

A quick (and very much non-comprehensive) usage example:

.. doctest::

    >>> import pyfftw, numpy
    >>> # Create 3 16-byte aligned arays using the aligned array creation functions
    >>> # They are cleared during the object creation, so there is no point filling them.
    >>> a = pyfftw.n_byte_align_empty((1,4), 16, dtype=numpy.complex128) 
    >>> b = pyfftw.n_byte_align_empty(a.shape, 16, dtype=a.dtype)
    >>> c = pyfftw.n_byte_align_empty(a.shape, 16, dtype=a.dtype)
    >>> # Create the DFT and inverse DFT objects
    >>> fft = pyfftw.FFTW(a, b)
    >>> ifft = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')
    >>> # Fill a with data
    >>> a[:] = [1, 2, 3, 4]
    >>> print a
    [[ 1.+0.j  2.+0.j  3.+0.j  4.+0.j]]
    >>> # perform the fft
    >>> fft.execute()
    >>> print b
    [[ 10.+0.j  -2.+2.j  -2.+0.j  -2.-2.j]]
    >>> # perform the inverse fft
    >>> ifft.execute()
    >>> print c/a.size
    [[ 1.+0.j  2.+0.j  3.+0.j  4.+0.j]]

