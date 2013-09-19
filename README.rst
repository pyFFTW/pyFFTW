PyFFTW
======

pyFFTW is a pythonic wrapper around `FFTW 3 <http://www.fftw.org/>`_, the
speedy FFT library.  The ultimate aim is to present a unified interface for all the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft`` 
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

Wisdom import and export now works fairly reliably.

Operating FFTW in multithreaded mode is supported.

pyFFTW implements the numpy and scipy fft interfaces in order for users to
take advantage of the speed of FFTW with minimal code modifications.

A comprehensive unittest suite can be found with the source on the github 
repository or with the source distribution on PyPI.

The documentation can be found on 
`github pages <http://hgomersall.github.com/pyFFTW>`_, the source is 
on `github <https://github.com/hgomersall/pyFFTW>`_ and the python package 
index page is `here <http://pypi.python.org/pypi/pyFFTW>`_.

Requirements (i.e. what it was designed for)
--------------------------------------------
- Python 2.7 or greater (Python 3 is supported)
- Numpy 1.6
- FFTW 3.2 or higher (lower versions *may* work)
- Cython 0.15 or higher (though the source release on PyPI loses this 
  dependency)

(install these as much as possible with your preferred package manager).

Installation
------------

We recommend *not* building from github, but using the release on 
the python package index with tools such as easy_install or pip::

  pip install pyfftw

or::

  easy_install pyfftw

Success has been reported on building on Linux, 32-bit Windows and Mac OSX.
It doesn't mean it won't work anywhere else, just we don't have any information
on it.

64-bit windows is possible but a bit of fiddling is required (see below).

Read on if you do want to build from source...

Building
--------

To build in place::

  python setup.py build_ext --inplace

That cythons the python extension and builds it into a shared library
which is placed in ``pyfftw/``. The directory can then be treated as a python
package.

After you've run ``setup.py`` with cython available, you then have a 
normal C extension in the ``pyfftw`` directory. 
Further building does not depend on cython (as long as the .c file remains).

For more ways of building and installing, see the 
`distutils documentation <http://docs.python.org/distutils/builtdist.html>`_

Platform specific build info
----------------------------

Windows
~~~~~~~

To build for windows from source, download the fftw dlls for your system
and the header file from `here <http://www.fftw.org/install/windows.html>`_ 
(they're in a zip file) and place them in the pyfftw
directory. The files are ``libfftw3-3.dll``, ``libfftw3l-3.dll``, 
``libfftw3f-3.dll``. If you're using a version of FFTW other than 3.3, it may
be necessary to copy ``fftw3.h`` into ``include\win``.

The setup scripts are designed for using with MinGW. They don't work as is
with MSVC. If you want to build for 64-bit windows, you will *have* to use
MSVC as building python extensions for 64-bit Windows with MinGW is currently
badly supported. This is a work in progress...

Mac OSX
~~~~~~~

It has been suggested that FFTW should be installed from `macports <http://www.macports.org/>`_.

