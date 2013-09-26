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

from distutils.core import setup, Command
from distutils.extension import Extension
from distutils.util import get_platform
from distutils.ccompiler import get_default_compiler

import os
import numpy
import sys

# Get the version string in rather a roundabout way.
# We can't import it directly as the module may not yet be
# built in pyfftw.
import imp
ver_file, ver_pathname, ver_description = imp.find_module(
            '_version', ['pyfftw'])
try:
    _version = imp.load_module('version', ver_file, ver_pathname, 
            ver_description)
finally:
    ver_file.close()

version = _version.version

try:
    from Cython.Distutils import build_ext as build_ext
    sources = [os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.pyx')]
except ImportError as e:
    sources = [os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.c')]
    if not os.path.exists(sources[0]):
        raise ImportError(str(e) + '. ' +
                'Cython is required to build the initial .c file.')

    # We can't cythonize, but that's ok as it's been done already.
    from distutils.command.build_ext import build_ext

include_dirs = [os.path.join(os.getcwd(), 'include'), 
        os.path.join(os.getcwd(), 'pyfftw'),
        numpy.get_include()]
library_dirs = []
package_data = {}

if get_platform() in ('win32', 'win-amd64'):
    libraries = ['libfftw3-3', 'libfftw3f-3', 'libfftw3l-3']
    include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))
    library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))
    package_data['pyfftw'] = [
            'libfftw3-3.dll', 'libfftw3l-3.dll', 'libfftw3f-3.dll']
else:
    libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 
            'fftw3f_threads', 'fftw3l_threads']

class custom_build_ext(build_ext):
    def finalize_options(self):

        build_ext.finalize_options(self)

        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

        if compiler == 'msvc':
            # Add msvc specific hacks
            
            if (sys.version_info.major, sys.version_info.minor) < (3, 3):
                # The check above is a nasty hack. We're using the python
                # version as a proxy for the MSVC version. 2008 doesn't
                # have stdint.h, so is needed. 2010 does.
                #
                # We need to add the path to msvc includes
                include_dirs.append(os.path.join(os.getcwd(), 
                    'include', 'msvc_2008'))

            # We need to prepend lib to all the library names
            _libraries = []
            for each_lib in self.libraries:
                _libraries.append('lib' + each_lib)

            self.libraries = _libraries

ext_modules = [Extension('pyfftw.pyfftw',
    sources=sources,
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=include_dirs)]

long_description = '''
pyFFTW is a pythonic wrapper around `FFTW <http://www.fftw.org/>`_, the
speedy FFT library. The ultimate aim is to present a unified interface for all
the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as arbitrary
axes of abitrary shaped and strided arrays, which makes it almost
feature equivalent to standard and real FFT functions of ``numpy.fft`` 
(indeed, it supports the ``clongdouble`` dtype which ``numpy.fft`` does not).

Operating FFTW in multithreaded mode is supported.

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

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys, subprocess
        errno = subprocess.call([sys.executable, '-m', 'unittest', 
            'discover'])
        raise SystemExit(errno)

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
        'packages':['pyfftw', 'pyfftw.builders', 'pyfftw.interfaces'],
        'ext_modules': ext_modules,
        'include_dirs': include_dirs,
        'package_data': package_data,
        'cmdclass': {'test': TestCommand,
            'build_ext': custom_build_ext},
  }

if __name__ == '__main__':
    setup(**setup_args)
