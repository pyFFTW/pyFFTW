.. pyFFTW documentation master file, created by
   sphinx-quickstart on Mon Jan 30 14:37:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyFFTW's documentation!
==================================

* :ref:`FFTW Class <FFTW_class>`
* :ref:`Utility Functions <UtilityFunctions>`

pyFFTW is an attempt to produce a pythonic wrapper around 
`FFTW <http://www.fftw.org/>`_. The ultimate aim is to present a unified
interface for all the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft`` 
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not). 
See the ``numpy.fft`` `documentation
<http://docs.scipy.org/doc/numpy/reference/routines.fft.html>`_ 
for more information on that module.

The source can be found in `github <https://github.com/hgomersall/pyFFTW>`_ 
and it's page in the python package index is 
`here <http://pypi.python.org/pypi/pyFFTW>`_.

A comprehensive unittest suite is included with the source on the repository.

A quick (and very much non-comprehensive) usage example:

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

**Note that what was previously the ComplexFFTW class is now just called
FFTW. Simply renaming the class should be sufficient to migrate**

.. _FFTW_class:

FFTW Class
-----------------

.. autoclass:: pyfftw.FFTW(input_array, output_array, axes=(-1,), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',))

   .. _FFTW_update_arrays:

   .. automethod:: pyfftw.FFTW.update_arrays(new_input_array, new_output_array)

   .. _FFTW_execute:

   .. automethod:: pyfftw.FFTW.execute()

.. _UtilityFunctions:

Utility Functions
-----------------

.. _n_byte_align:

.. autofunction:: pyfftw.n_byte_align

.. _n_byte_align_empty:

.. autofunction:: pyfftw.n_byte_align_empty

