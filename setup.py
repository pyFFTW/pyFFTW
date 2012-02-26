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
from distutils.util import get_platform

import os
import numpy


include_dirs = [numpy.get_include()]
library_dirs = []
package_data = {}

if get_platform() == 'win32':
    libraries = ['fftw3-3', 'fftw3f-3', 'fftw3l-3', 'm']
    include_dirs.append('pyfftw')
    library_dirs.append(os.path.join(os.getcwd(),'pyfftw'))
    package_data['pyfftw'] = \
            ['libfftw3-3.dll', 'libfftw3l-3.dll', 'libfftw3f-3.dll']
else:
    libraries = ['fftw3', 'fftw3f', 'fftw3l', 'm']


ext_modules = [Extension('pyfftw.pyfftw',
    sources=[os.path.join('pyfftw', 'pyfftw.c')],
    libraries=libraries,
    library_dirs=library_dirs)]

version = '0.6.1'

long_description = '''
pyFFTW is an attempt to produce a pythonic wrapper around 
`FFTW <http://www.fftw.org/>`_. The ultimate aim is to present a unified
interface for all the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as arbitrary
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

The documentation can be found 
`here <http://hgomersall.github.com/pyFFTW/>`_, and the source
is on `github <https://github.com/hgomersall/pyFFTW>`_.
'''

setup_args = {
        'name': 'pyFFTW',
        'version': version,
        'author': 'Henry Gomersall',
        'author_email': 'heng@kedevelopments.co.uk',
        'description': 'A pythonic wrapper around FFTW, the FFT library, presenting a unified interface for all the supported transforms.',
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
        'packages':['pyfftw'],
        'ext_modules': ext_modules,
        'include_dirs': include_dirs,
        'package_data': package_data,
  }

if __name__ == '__main__':
    setup(**setup_args)
