#! /usr/bin/env python

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
from distutils.ccompiler import get_default_compiler, new_compiler
from distutils.errors import CompileError, LinkError

import os
import mpi4py
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

# todo check latest cython doc at sage server on best practice for
#      building. I think cythonize is the way to go; it seems to
#      be only way to pass macros from setup.py into cython for
#      conditional compilation

# TODO read option from command line
#      http://stackoverflow.com/questions/677577/distutils-how-to-pass-a-user-defined-parameter-to-setup-py
use_mpi = 0

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

data_types = ('DOUBLE', 'SINGLE', 'LONG', 'QUAD')
data_types_short = ('', 'f', 'l', 'q')

class LibraryChecker:
    '''Container for libraries that checks their existence.

    :param exclude:

        Iterable of strings; pass packages to ignore in here. Example: exclude=('DOUBLE_MPI', 'SINGLE_MPI', 'LONG_MPI', 'QUAD_MPI')

    '''
    def __init__(self, exclude=None):
        self.include_dirs = [os.path.join(os.getcwd(), 'include'),
                             os.path.join(os.getcwd(), 'pyfftw'),
                             numpy.get_include()]
        if use_mpi:
            self.include_dirs.append(mpi4py.get_include())

        self.libraries = []
        self.library_dirs = []
        self.package_data = {} # why package data only updated for windows?
        self.compile_time_env = {}

        # construct checks
        # self.data['HAVE_SINGLE_THREADS'] = ['fftw3f_threads', 'fftwf_init_threads']
        self.data = {}
        lib_types = ('', '_MPI', '_THREADS', '_OMP')
        functions = ('plan_dft', 'mpi_init', 'init_threads', 'init_threads')
        for f, l in zip(functions, lib_types):
            for d, s in zip(data_types, data_types_short):
                self.data['HAVE_' + d + l] = ['fftw3' + s + l.lower(), 'fftw' + s + '_' + f]

        if get_platform() in ('win32', 'win-amd64'):
            # self.libraries = ['libfftw3-3', 'libfftw3f-3', 'libfftw3l-3']
            for k, v in self.data.iteritems():
                # fftw3 -> libfftw3-3
                v[0] = 'lib' + v[0] + '-3'
            self.include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))
            self.library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))
            # TODO fix package data *after* we know which libraries exist
            # self.package_data['pyfftw'] = [lib + '.dll' for lib in self.libraries]
            # TODO What about thread libraries on windows?
            # TODO mpi support missing and untested on windows

        # now check library existence by linking. If the linker can't find it, we don't have it
        self.compiler = new_compiler(verbose=1)

        for macro, (lib, function) in self.data.iteritems():
            if exclude is not None and macro[5:] in exclude:
                exists = False
            else:
                # print macro, lib, function
                with stdchannel_redirected(sys.stderr, os.devnull):
                    exists = self.has_function(function, libraries=(lib,),
                                                   include_dirs=self.include_dirs,
                                                   library_dirs=self.library_dirs)
            if exists:
                self.libraries.append(lib)
            self.compile_time_env[macro] = exists

        have_mpi = False
        for d in data_types:
            have_mpi |= self.compile_time_env['HAVE_' + d + '_MPI']

    def has_function(self, function, includes=None, libraries=None, include_dirs=None, library_dirs=None):
        '''Alternative implementation of distutils.ccompiler.has_function that deletes the output and works reliably.'''

        if includes is None:
            includes = []
        if libraries is None:
            libraries = []
        if include_dirs is None:
            include_dirs = []
        if library_dirs is None:
            library_dirs = []

        print "Checking for", function, "in", libraries, "..",

        import tempfile, shutil
        tmpdir = tempfile.mkdtemp(prefix='pyfftw-')
        devnull = oldstderr = None
        try:
            try:
                fname = os.path.join(tmpdir, '%s.c' % function)
                f = open(fname, 'w')
                for inc in includes:
                    f.write('#include <%s>\n' % inc)
                f.write('int main(void) {\n')
                f.write('    %s();\n' % function)
                f.write('}\n')
            finally:
                f.close()
            try:
                objects = self.compiler.compile([fname], output_dir=tmpdir, include_dirs=include_dirs)
            except CompileError:
                print 'no'
                return False
            except Exception as e:
                print 'Compilation failed'
                print e
                return False
            try:
                self.compiler.link_executable(objects,
                                              os.path.join(tmpdir, "a.out"),
                                              libraries=libraries,
                                              library_dirs=library_dirs)
            except LinkError:
                print 'no'
                return False
            except Exception as e:
                print 'Linking failed'
                print e
                return False
            # no error, seems to work
            print 'ok'
            return True
        finally:
            shutil.rmtree(tmpdir)

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

# check what's available
import contextlib

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')

    Taken from http://stackoverflow.com/a/17752729/987623
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()

checker = LibraryChecker()
checker.compile_time_env["HAVE_MPI"] = use_mpi

print checker.compile_time_env

# recompile if any of these files are modified
dependencies = [os.path.join('pyfftw', f) for f in ('pyfftw.pxd', 'mpi.pxd', 'mpi.pxi', 'utils.pxi')]

ext_modules = [Extension('pyfftw.pyfftw',
                         sources=sources,
                         libraries=checker.libraries,
                         library_dirs=checker.library_dirs,
                         include_dirs=checker.include_dirs,
                         extra_compile_args=['-Wno-maybe-uninitialized'],
                         depends=dependencies)]

from Cython.Build import cythonize
ext_modules = cythonize(ext_modules, compile_time_env=checker.compile_time_env)

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

# todo how is TestCommand used by setup?
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
        'include_dirs': checker.include_dirs,
        'package_data': checker.package_data,
        'cmdclass': {'test': TestCommand,
            'build_ext': custom_build_ext},
  }

if __name__ == '__main__':
    setup(**setup_args)
