from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('pyfftw3', ['pyfftw3.pyx'], 
    libraries=['fftw3', 'fftw3f', 'fftw3l', 'm'])]

description = ''

setup(
  name = 'pyFFTW',
  version = '0.5.0',
  author = 'Henry Gomersall',
  author_email = 'heng@kedevelopments.co.uk',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
