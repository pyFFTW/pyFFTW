.. pyFFTW documentation master file, created by
   sphinx-quickstart on Mon Jan 30 14:37:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyFFTW's documentation!
==================================

Introduction
------------

pyFFTW is a pythonic wrapper around `FFTW <http://www.fftw.org/>`_, the
speedy FFT library.  The ultimate aim is to present a unified interface for all
the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of :mod:`numpy.fft`
(indeed, it supports the :attr:`~numpy.clongdouble` dtype which
:mod:`!numpy.fft` does not).

Operating FFTW in multithreaded mode is supported.

The core interface is provided by a unified class, :class:`pyfftw.FFTW`.
This core interface can be accessed directly, or through a series of helper
functions, provided by the :mod:`pyfftw.builders` module. These helper
functions provide an interface similar to :mod:`numpy.fft` for ease of use.

In addition to using :class:`pyfftw.FFTW`, a convenient series of functions
are included through :mod:`pyfftw.interfaces` that make using :mod:`pyfftw`
almost equivalent to :mod:`numpy.fft` or :mod:`scipy.fftpack`.


The source can be found in `github <https://github.com/hgomersall/pyFFTW>`_ and
its page in the python package index is `here
<http://pypi.python.org/pypi/pyFFTW>`_.

A comprehensive unittest suite is included with the source on the repository.
If any aspect of this library is not covered by the test suite, that is a bug
(please report it!).

Contents
--------

.. toctree::
   :maxdepth: 2

   /sphinx/tutorial
   /sphinx/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
