# Copyright 2012 Knowledge Economy Developments Ltd
# 
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup
from distutils.extension import Extension

libraries = ['fftw3', 'fftw3f', 'fftw3l', 'm']
ext_modules = [Extension('pyfftw', ['pyfftw.c'], 
    libraries=libraries)]

long_description = '''
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

The documentation can be found 
`here <http://hgomersall.github.com/pyFFTW/>`_, and the source
is on `github <https://github.com/hgomersall/pyFFTW>`_.
'''

setup_args = {
        'name': 'pyFFTW',
        'version': '0.5.0',
        'author': 'Henry Gomersall',
        'author_email': 'heng@kedevelopments.co.uk',
        'description': 'A pythonic wrapper around FFTW, presenting a unified interface for all the supported transforms.',
        'url': 'http://hgomersall.github.com/pyFFTW/',
        'long_description': long_description,
        'classifiers': [
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: GNU General Public License (GPL)',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            ],
        'ext_modules': ext_modules,
  }

if __name__ == '__main__':
    setup(**setup_args)
