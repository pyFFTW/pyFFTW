from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('fftw3', ['fftw3.pyx'], 
    libraries=['fftw3f','m'])]

setup(
  name = 'FFTW wrapper',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
