+---------------+-----------------+
| Service       | Master branch   |
+===============+=================+
| Travis        | |travis_ci|     |
+---------------+-----------------+
| Appveyor      | |appveyor_ci|   |
+---------------+-----------------+
| Read the Docs | |read_the_docs| |
+---------------+-----------------+

.. |travis_ci| image:: https://travis-ci.org/pyFFTW/pyFFTW.svg?branch=master
   :align: middle
   :target: https://travis-ci.org/pyFFTW/pyFFTW

.. |appveyor_ci| image:: https://ci.appveyor.com/api/projects/status/uf854abck4x1qsjj/branch/master?svg=true
   :align: middle
   :target: https://ci.appveyor.com/project/hgomersall/pyfftw

.. |read_the_docs| image:: https://readthedocs.org/projects/pyfftw/badge/?version=latest
   :align: middle
   :target: http://pyfftw.readthedocs.io/en/latest/?badge=latest

PyFFTW
======

pyFFTW is a pythonic wrapper around FFTW_ 3, the speedy FFT library.  The
ultimate aim is to present a unified interface for all the possible transforms
that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft``
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

Wisdom import and export now works fairly reliably.

Operating FFTW in multithreaded mode is supported.

pyFFTW implements the numpy and scipy fft interfaces in order for users to
take advantage of the speed of FFTW with minimal code modifications.

A comprehensive unittest suite can be found with the source on the GitHub_
repository or with the source distribution on PyPI_.

The documentation can be found on `Read the Docs`_, the source is on GitHub_
and the python package index page (PyPI_).  Issues and questions can be
raised at the `GitHub Issues`_ page.

Requirements (i.e. what it was designed for)
--------------------------------------------
- Python_ 2.7 or >= 3.4
- Numpy_ >= 1.10.4  (lower versions *may* work)
- FFTW_ >= 3.3 (lower versions *may* work) libraries for single, double,
  and long double precision in serial and multithreading (pthreads or openMP)
  versions.
- Cython_ >= 0.23 (lower versions *may* work)

(install these as much as possible with your preferred package manager).

In practice, pyFFTW *may* work with older versions of these dependencies, but
it is not tested against them.

Optional Dependencies
---------------------
- Scipy_ >= 0.16
- Dask_ >= 0.14.2

Scipy and Dask are only required in order to use their respective interfaces.

Installation
------------

We recommend *not* building from github, but using the release on
the python package index with tools such as easy_install or pip::

  pip install pyfftw

or::

  easy_install pyfftw

Installers are on the PyPI_ page for both 32- and 64-bit Windows, which include
all the necessary DLLs.

With FFTW installed, the PyPI release should install fine on Linux and Mac OSX. It doesn't mean it won't work anywhere else, just we don't have any information on it.

Windows development builds are also automatically uploaded to bintray_ as
wheels (which are built against numpy 1.9), from where they can be downloaded
and installed with something like::

  pip install pyFFTW-0.10.0.dev0+79ec589-cp35-none-win_amd64.whl

where the version and the revision hash are set accordingly.

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

During configuration the available FFTW libraries are detected, so pay attention
to the output when running ``setup.py``. On certain platforms, for example the
long double precision is not available. pyFFTW still builds fine but will fail
at runtime if asked to perform a transform involving long double precision.

Regarding multithreading, if both posix and openMP FFTW libs are available, the
openMP libs are preferred. This preference can be reversed by defining the
environment variable ``PYFFTW_USE_PTHREADS`` prior to building. If neither
option is available, pyFFTW works in serial mode only.

For more ways of building and installing, see the
`distutils documentation <http://docs.python.org/distutils/builtdist.html>`_
and `setuptools documentation <https://setuptools.readthedocs.io>`_.

Platform specific build info
----------------------------

Windows
~~~~~~~

To build for windows from source, download the fftw dlls for your system and the
header file from `here <http://www.fftw.org/install/windows.html>`_ (they're in
a zip file) and place them in the pyfftw directory. The files are
``libfftw3-3.dll``, ``libfftw3l-3.dll``, ``libfftw3f-3.dll``. These libs use
pthreads for multithreading. If you're using a version of FFTW other than 3.3,
it may be necessary to copy ``fftw3.h`` into ``include\win``.

The builds on PyPI use mingw for the 32-bit release and the Windows SDK
C++ compiler for the 64-bit release. The scripts should handle this
automatically. If you want to compile for 64-bit Windows, you have to use
the MS Visual C++ compiler. Set up your environment as described
`here <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>`_ and then
run ``setup.py`` with the version of python you wish to target and a suitable
build command.

For using the MS Visual C++ compiler, you'll need to create a set of
suitable ``.lib`` files as described on the
`FFTW page <http://www.fftw.org/install/windows.html>`_.

Mac OSX
~~~~~~~
Install FFTW from `homebrew <http://brew.sh>`_::

  brew install fftw

Set temporary environmental variables, such that pyfftw finds fftw::

  export DYLD_LIBRARY_PATH=/usr/local/lib
  export LDFLAGS="-L/usr/local/lib"
  export CFLAGS="-I/usr/local/include"

Now install pyfftw from pip::

  pip install pyfftw

It has been suggested that macports_ might also work fine. You should then
replace the LD environmental variables above with the right ones.

- DYLD - path for libfftw3.dylib etc - ``find /usr -name libfftw3.dylib``
- LDFLAGS - path for fftw3.h - ``find /usr -name fftw3.h``

FreeBSD
~~~~~~~

Install FFTW from ports tree or ``pkg``:

    - math/fftw3
    - math/fftw3-float
    - math/fftw3-long

Please install all of them, if possible.

Contributions
-------------

Contributions are always welcome and valued. The primary restriction on
accepting contributions is that they are exhaustively tested. The bulk of
pyFFTW has been developed in a test-driven way (i.e. the test to be
satisfied is written before the code). I strongly encourage potential
contributors to adopt such an approach.

See some of my philosophy on testing in development `here
<https://hgomersall.wordpress.com/2014/10/03/from-test-driven-development-and-specifications/>`_.
If you want to argue with the philosophy, there is probably a good place to
do it.

New contributions should adhere to `PEP 8`_, but this is only weakly enforced
(there is loads of legacy stuff that breaks it, and things like a single
trailing whitespace is not a big deal).

The best place to start with contributing is by raising an issue detailing the
specifics of what you wish to achieve (there should be a clear use-case for
any new functionality). I tend to respond pretty quickly and am happy to help
where I can with any conceptual issues.

I suggest reading the issues already open in order that you know where things
might be heading, or what others are working on.

.. _Python: https://python.org
.. _FFTW: https://www.fftw.org
.. _NumPy: https://www.numpy.org
.. _Cython: https://cython.org
.. _SciPy: https://www.scipy.org
.. _Dask: https://dask.pydata.org
.. _GitHub: https://github.com/PyFFTW/PyFFTW
.. _GitHub Issues: https://github.com/PyFFTW/PyFFTW/issues
.. _PyPI: https://pypi.python.org
.. _Read the Docs: https://pyfftw.readthedocs.io
.. _bintray: https://bintray.com/hgomersall/generic/PyFFTW-development-builds/view
.. _PEP 8: https://www.python.org/dev/peps/pep-0008
.. _macports:  https://www.macports.org
