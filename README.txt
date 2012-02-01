pyFFTW is an attempt to produce a pythonic wrapper around 
FFTW ( http://www.fftw.org/ ). The ultimate aim is to present a unified
interface for all the possible transforms that FFTW can perform.

Currently, only the complex DFT is supported, though on arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard FFT functions of ``numpy.fft`` (indeed, 
it supports the ``clongdouble`` dtype which ``numpy.fft`` does not). 
It shouldn't be too much work to extend it to other schemes such as 
the real DFT.

A comprehensive unittest suite is included with the source.

The documentation can be found at http://hgomersall.github.com/pyFFTW/ , the source is on github: https://github.com/hgomersall/pyFFTW and the python package index page is here: http://pypi.python.org/pypi/pyFFTW .

If you want to build the code that is here, use the cython_setup script like:

python ./cython_setup.py build_ext --inplace

That will build the cython code into a .c file (and should compile that too).

setup.py is designed for after the .c file has been created (as in the source distribution).
