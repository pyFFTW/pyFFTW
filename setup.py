# Copyright 2014 Knowledge Economy Developments Ltd
# 
# Henry Gomersall
# heng@kedevelopments.co.uk
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

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
        errno = subprocess.call([sys.executable, '-m', 
            'unittest', 'discover'])
        raise SystemExit(errno)

class QuickTestCommand(Command):
    '''Runs a set of test cases that covers a limited set of the 
    functionality. It is intended that this class be used as a sanity check
    that everything is loaded and basically working as expected. It is not
    meant to replace the comprehensive test suite.
    '''
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        quick_test_cases = [
            'test.test_pyfftw_complex.Complex64FFTWTest',
            'test.test_pyfftw_complex.Complex128FFTWTest.test_2d',
            'test.test_pyfftw_complex.ComplexLongDoubleFFTWTest.test_2d',
            'test.test_pyfftw_real_forward.RealForwardSingleFFTWTest',
            'test.test_pyfftw_real_forward.RealForwardDoubleFFTWTest.test_2d',
            'test.test_pyfftw_real_forward.RealForwardLongDoubleFFTWTest.test_2d',
            'test.test_pyfftw_real_backward.RealBackwardSingleFFTWTest',
            'test.test_pyfftw_real_backward.RealBackwardDoubleFFTWTest.test_2d',
            'test.test_pyfftw_real_backward.RealBackwardLongDoubleFFTWTest.test_2d',
            'test.test_pyfftw_wisdom',
            'test.test_pyfftw_utils',
            'test.test_pyfftw_call',
            'test.test_pyfftw_class_misc',
            'test.test_pyfftw_nbyte_align',
            'test.test_pyfftw_interfaces_cache',
            'test.test_pyfftw_multithreaded',
            'test.test_pyfftw_numpy_interface.InterfacesNumpyFFTTestModule',
            'test.test_pyfftw_numpy_interface.InterfacesNumpyFFTTestFFT2',
            'test.test_pyfftw_numpy_interface.InterfacesNumpyFFTTestIFFT2',            
            'test.test_pyfftw_builders.BuildersTestFFTWWrapper',
            'test.test_pyfftw_builders.BuildersTestFFT2',
            'test.test_pyfftw_builders.BuildersTestIRFFT2',
        ]

        import sys, subprocess
        errno = subprocess.call([sys.executable, '-m', 
            'unittest'] + quick_test_cases)
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
            'License :: OSI Approved :: BSD License',
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
                     'quick_test': QuickTestCommand,
                     'build_ext': custom_build_ext},
  }

if __name__ == '__main__':
    setup(**setup_args)
