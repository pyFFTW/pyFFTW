from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('pyfftw3', ['pyfftw3.pyx'], 
    libraries=['fftw3', 'fftw3f','m'])]

setup(
  name = 'pyFFTW',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
