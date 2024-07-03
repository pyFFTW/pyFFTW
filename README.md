### Current Build Status

| GitHub Actions                                                                                                                                                                                                | Read the Docs                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [![Build Wheels, Test and Release](https://github.com/pyFFTW/pyFFTW/actions/workflows/wheel_tests_and_release.yml/badge.svg)](https://github.com/pyFFTW/pyFFTW/actions/workflows/wheel_tests_and_release.yml) | [![read_the_docs](https://readthedocs.org/projects/pyfftw/badge/?version=latest)](http://pyfftw.readthedocs.io/en/latest/?badge=latest) |

### Conda-forge Status

[![Linux](https://img.shields.io/circleci/project/github/conda-forge/pyfftw-feedstock/master.svg?label=Linux)](https://circleci.com/gh/conda-forge/pyfftw-feedstock) [![OSX](https://img.shields.io/travis/conda-forge/pyfftw-feedstock/master.svg?label=macOS)](https://travis-ci.org/conda-forge/pyfftw-feedstock) [![Windows](https://img.shields.io/appveyor/ci/conda-forge/pyfftw-feedstock/master.svg?label=Windows)](https://ci.appveyor.com/project/conda-forge/pyfftw-feedstock/branch/master)


### Conda-forge Info

| Name                                                                                                             | Downloads                                                                                                             | Version                                                                                                             | Platforms                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-pyfftw-green.svg)](https://anaconda.org/conda-forge/pyfftw) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pyfftw.svg)](https://anaconda.org/conda-forge/pyfftw) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyfftw.svg)](https://anaconda.org/conda-forge/pyfftw) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pyfftw.svg)](https://anaconda.org/conda-forge/pyfftw) |

# PyFFTW

pyFFTW is a pythonic wrapper around [FFTW 3](https://www.fftw.org), the speedy
FFT library.  The ultimate aim is to present a unified interface for all the
possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of arbitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft``
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

Wisdom import and export now works fairly reliably.

Operating FFTW in multithreaded mode is supported.

pyFFTW implements the numpy and scipy fft interfaces in order for users to
take advantage of the speed of FFTW with minimal code modifications.

A comprehensive unittest suite can be found with the source on the
[GitHub](https://github.com/PyFFTW/PyFFTW) repository or with the source
distribution on [PyPI](https://pypi.org/).

The documentation can be found on
[Read the Docs](https://pyfftw.readthedocs.io) the source is on
[GitHub](https://github.com/PyFFTW/PyFFTW) and the python package index page
[PyPI](https://pypi.org/). Issues and questions can be raised at the
[GitHub Issues](https://github.com/PyFFTW/PyFFTW/issues) page.

## Requirements (i.e. what it was designed for)

- [Python](https://python.org) >= 3.9 (lower versions *may* work)
- [Numpy](https://www.numpy.org) >= 1.20 (lower versions *may* work)
- [FFTW](https://www.fftw.org) >= 3.3 (lower versions *may* work) libraries for
  single, double, and long double precision in serial and multithreading
  (pthreads or openMP) versions.
- [Cython](https://cython.org) >= 0.29

(install these as much as possible with your preferred package manager).

In practice, pyFFTW *may* work with older versions of these dependencies, but
it is not tested against them.

We build wheels for PyPy 3.9, but this platform has not been tested.

## Optional Dependencies

- [Scipy](https://www.scipy.org) >= 1.8
- [Dask](https://dask.pydata.org) >= 1.0

Scipy and Dask are only required in order to use their respective interfaces.
In practice, older versions may work, but they are not tested against.

## Installation

We recommend *not* building from github, but using the release on the python
package index with tools such as pip:

    pip install pyfftw

Pre-built binary wheels for 64-bit Python on Linux, Mac OS X and Windows are
available on the [PyPI](https://pypi.org/) page for all supported Python versions.
Note that we only support binaries for 64-bit Python. 32-bit and ARM architectures have
prebuilt wheels for some configurations - see below.

Note that prior to Python 3.9, the Windows installation defaulted to being 32-bit
even on 64-bit Windows, so if you are having problems installing using pip
(typically with an error message like `ERROR: Failed building wheel for pyfftw`)
then please check your Python version.

Installation from PyPI may also work on other systems when the FFTW libraries
are available, but other platforms have not been tested.

Alternatively, users of the [conda](https://conda.io/docs/) package manager can
install from the [conda-forge](https://conda-forge.org/) channel via:

    conda install -c conda-forge pyfftw

Read on if you do want to build from source...

## Wheels

Prebuilt wheels are available for the following configurations:

|          Python version          | Windows (32 bit) | Windows (64 bit) | Windows ARM (64 bit) | MacOS | MacOS ARM | Linux (64 bit) | Linux ARM (64 bit) |
| :------------------------------: | :--------------: | :--------------: | :------------------: | :---: | :-------- | :------------: | :----------------: |
|   CPython < 3.9 (unsupported)    |        ❌        |        ❌        |          ❌          |  ❌   | ❌        |       ❌       |         ❌         |
|           CPython 3.9            |        ✔        |        ✔        |          ❌          |  ✔   | ✔        |       ✔       |         ✔         |
|           CPython 3.10           |        ✔        |        ✔        |          ❌          |  ✔   | ✔        |       ✔       |         ✔         |
|           CPython 3.11           |        ✔        |        ✔        |          ❌          |  ✔   | ✔        |       ✔       |         ✔         |
|           CPython 3.12           |        ✔        |        ✔        |          ❌          |  ✔   | ✔        |       ✔       |         ✔         |
|     PyPy < 3.9 (unsupported)     |        ❌        |        ❌        |          ❌          |  ❌   | ❌        |      ❌       |         ❌         |
|             PyPy 3.9             |        ❌        |        ✔        |          ❌          |  ✔   | ✔        |       ✔       |         ❌         |
| PyPy > 3.9 (unsupported for now) |        ❌        |        ❌        |          ❌          |  ❌   | ❌        |        ❌       |         ❌         |

Note that Linux 32-bit wheels are available only for pyFFTW <= 0.13.1.

If your configuration does not match one of these you will have to build `pyfft` from source yourself.
See instructions below.

## Building

To build in place:

    pip install -e . -v

That cythonizes the python extension and builds it into a shared library
which is placed in ``pyfftw/``. The directory can then be treated as a python
package.

After you've run ``setup.py`` with cython available, you then have a
normal C extension in the ``pyfftw`` directory.
Further building does not depend on cython (as long as the .c file remains).

During configuration the available FFTW libraries are detected, so pay attention
to the output when running ``setup.py``. On certain platforms, for example the
long double precision is not available. pyFFTW still builds fine but will fail
at runtime if asked to perform a transform involving long double precision.

To build against FFTW libraries at non standard location, [some compilers are
sensitive to the environment
variables](https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html)
`CPATH` and `LIBRARY_PATH`. Moreover, you can also use `PYFFTW_INCLUDE` and
`PYFFTW_LIB_DIR`. If the FFTW libraries still cannot be found, you might also
need to set the environment variable `CC` to build with the compiler used to
compile the libraries.

Regarding multithreading, if both posix and openMP FFTW libs are available, the
openMP libs are preferred. This preference can be reversed by defining the
environment variable ``PYFFTW_USE_PTHREADS`` prior to building. If neither
option is available, pyFFTW works in serial mode only.

For more ways of building and installing, see the
[distutils documentation](http://docs.python.org/distutils/builtdist.html)
and [setuptools documentation](https://setuptools.readthedocs.io).

### Platform specific build info

#### Windows

To build for windows from source, download the fftw dlls for your system and the
header file from [here](http://www.fftw.org/install/windows.html) (they're in
a zip file) and place them in the pyfftw directory. The files are
``libfftw3-3.dll``, ``libfftw3l-3.dll``, ``libfftw3f-3.dll``. These libs use
pthreads for multithreading. If you're using a version of FFTW other than 3.3,
it may be necessary to copy ``fftw3.h`` into ``include\win``.

The builds on PyPI use mingw for the 32-bit release and the Windows SDK
C++ compiler for the 64-bit release. The scripts should handle this
automatically. If you want to compile for 64-bit Windows, you have to use
the MS Visual C++ compiler. Set up your environment as described
[here](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows) and then
run ``setup.py`` with the version of python you wish to target and a suitable
build command.

For using the MS Visual C++ compiler, you'll need to create a set of
suitable ``.lib`` files as described on the
[FFTW page](http://www.fftw.org/install/windows.html).

#### Mac OSX

Install FFTW from [homebrew](http://brew.sh):

    brew install fftw

Set temporary environmental variables, such that pyfftw finds fftw:

    export DYLD_LIBRARY_PATH=/usr/local/lib
    export LDFLAGS="-L/usr/local/lib"
    export CFLAGS="-I/usr/local/include"

Now install pyfftw from pip:

    pip install pyfftw

It has been suggested that [macports](https://www.macports.org) might also work
fine. You should then replace the LD environmental variables above with the
right ones.

- DYLD - path for libfftw3.dylib etc - ``find /usr -name libfftw3.dylib``
- LDFLAGS - path for fftw3.h - ``find /usr -name fftw3.h``

#### FreeBSD

Install FFTW from ports tree or ``pkg``:

    - math/fftw3
    - math/fftw3-float
    - math/fftw3-long

Please install all of them, if possible.

## Testing

Tests should be run using `pytest`. Install using:

```sh
pip install pytest
```

To run tests against the installed (compiled) `pyFFTW` wheel, use:

```sh
pytest --import-mode=append tests/
```

**Note**: `--import-mode=append` is needed to prevent `pytest` patching `sys.path`
in a way that resolves the local installation over the wheel installation.

## Contributions

Contributions are always welcome and valued. The primary restriction on
accepting contributions is that they are exhaustively tested. The bulk of
pyFFTW has been developed in a test-driven way (i.e. the test to be
satisfied is written before the code). I strongly encourage potential
contributors to adopt such an approach.

See some of my philosophy on testing in development [here]
(https://hgomersall.wordpress.com/2014/10/03/from-test-driven-development-and-specifications).
If you want to argue with the philosophy, there is probably a good place to
do it.

New contributions should adhere to
[PEP 8](https://www.python.org/dev/peps/pep-0008), but this is only weakly
enforced (there is loads of legacy stuff that breaks it, and things like a
single trailing whitespace is not a big deal).

The best place to start with contributing is by raising an issue detailing the
specifics of what you wish to achieve (there should be a clear use-case for
any new functionality). I tend to respond pretty quickly and am happy to help
where I can with any conceptual issues.

I suggest reading the issues already open in order that you know where things
might be heading, or what others are working on.

### Use PDM to setup a virtual environment for the development of pyFFTW

[PDM], which can be for example installed with [Pipx], can be used to setup a virtual
environment `.venv` for the development of pyFFTW and install pyFFTW in editable mode.

```sh
pdm sync --clean -v
```

This environment is described in the file `pyproject.toml` and the exact versions of
the packages are listed in a lock file `pdm.lock`. Maintainers can update the lockfile
by running (typically in a PR before each release)

```sh
pdm lock
```

[pdm]: https://pdm-project.org
[pipx]: https://github.com/pypa/pipx