.. pyFFTW documentation master file, created by
   sphinx-quickstart on Mon Jan 30 14:37:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyFFTW's documentation!
==================================

pyFFTW is an attempt to produce a pythonic wrapper around 
`FFTW <http://www.fftw.org/>`_. The ultimate aim is to present a unified
interface for all the possible transforms that FFTW can perform.

Currently, only the complex DFT is supported, though on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard FFT functions of ``numpy.fft`` (indeed, 
it supports the ``clongdouble`` dtype which ``numpy.fft`` does not). 
It shouldn't be too much work to extend it to other schemes such as 
the real DFT.

A comprehensive unittest suite is included with the source.

* :ref:`ComplexFFTW Class <ComplexFFTW_class>`
* :ref:`Utility Functions <UtilityFunctions>`

.. _ComplexFFTW_class:

ComplexFFTW Class
-----------------

.. autoclass:: pyfftw3.ComplexFFTW(input_array, output_array, axes=[-1], direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])

   .. _ComplexFFTW_update_arrays:

   .. automethod:: pyfftw3.ComplexFFTW.update_arrays(new_input_array, new_output_array)

   .. _ComplexFFTW_execute:

   .. automethod:: pyfftw3.ComplexFFTW.execute()

.. _UtilityFunctions:

Utility Functions
-----------------

.. _n_byte_align:

.. autofunction:: pyfftw3.n_byte_align

.. _n_byte_align_empty:

.. autofunction:: pyfftw3.n_byte_align_empty

