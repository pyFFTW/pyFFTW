pyFFTW is an attempt to produce a pythonic wrapper around 
FFTW ( http://www.fftw.org/ ). The ultimate aim is to present a unified
interface for all the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft`` 
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

A comprehensive unittest suite can be found with the source on the github 
repository.

To build for windows from source, download the fftw dlls for your system
and the header file from here (they're in a zip file):
http://www.fftw.org/install/windows.html and place them in the pyfftw
directory. The files are libfftw3-3.dll, libfftw3l-3.dll, libfftw3f-3.dll 
and libfftw3.h.

Under linux, to build from source, the FFTW library must be installed already.
This should probably work for OSX, though I've not tried it.

Numpy is a dependency for both.

The documentation can be found at http://hgomersall.github.com/pyFFTW/ , the source is on github: https://github.com/hgomersall/pyFFTW and the python package index page is here: http://pypi.python.org/pypi/pyFFTW .

If you want to build the code that is here, use the cython_setup script like:

python ./cython_setup.py build_ext --inplace

That will build the cython code into a .c file (and should compile that too).

setup.py is designed for after the .c file has been created (as in the source distribution).
