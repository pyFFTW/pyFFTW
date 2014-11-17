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
from distutils import log
from distutils.extension import Extension
from distutils.util import get_platform
from distutils.ccompiler import get_default_compiler, new_compiler
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler

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

# todo infos still printed, even with error threshold
# 0=minimal output, 2=maximum debug
# log.set_verbosity(0)
log.set_threshold(log.ERROR)

# we require cython to know which part of wrapper to build depending on available libraries
from Cython.Distutils import build_ext as build_ext
sources = [os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.pyx')]

# check what's available
import contextlib

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout and stderr to a file.

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')

    Taken from http://stackoverflow.com/a/17752729/987623
    """
    dest_file = None
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

class EnvironmentSniffer:
    '''Check for availability of headers and libraries of FFTW and MPI.

    :param compiler:

        Distutils.ccompiler; The compiler should preferably be the compiler
        that is used for actual compilation to ensure that include directories etc are identical.

    :param exclude:

        Iterable of strings; pass packages to ignore in here. Example: exclude=('DOUBLE_MPI', 'SINGLE_MPI', 'LONG_MPI', 'QUAD_MPI')

    '''
    def __init__(self, compiler, exclude=None):
        self.include_dirs = [os.path.join(os.getcwd(), 'include'),
                             os.path.join(os.getcwd(), 'pyfftw'),
                             numpy.get_include()]
        self.libraries = []
        self.library_dirs = []
        self.package_data = {} # todo why package data only updated for windows?
        self.compile_time_env = {}

        self.compiler = compiler

        have_mpi_h = self.has_header(['mpi.h'], include_dirs=self.include_dirs)
        if have_mpi_h:
            try:
                import mpi4py
                self.include_dirs.append(mpi4py.get_include())
            except ImportError:
                print("Could not import mpi4py. Skipping support for FFTW MPI.")
                have_mpi_h = False

        # construct checks
        self.data = {}
        data_types = ['DOUBLE', 'SINGLE', 'LONG', 'QUAD']
        data_types_short = ['', 'f', 'l', 'q']
        lib_types = ['', '_THREADS', '_OMP']
        functions = ['plan_dft', 'init_threads', 'init_threads']
        if have_mpi_h:
            lib_types.append('_MPI')
            functions.append('mpi_init')

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

        # first check if headers are there
        if not self.has_header(['fftw3.h'], include_dirs=self.include_dirs):
            raise CompileError("Could not find the FFTW header 'fftw3.h'")

        log.debug(self.data)
        for macro, (lib, function) in self.data.iteritems():
            if exclude is not None and macro[5:] in exclude:
                exists = False
            else:
                exists = self.has_function(function, libraries=(lib,),
                                           include_dirs=self.include_dirs,
                                           library_dirs=self.library_dirs)
            if exists:
                self.libraries.append(lib)
            self.compile_time_env[macro] = exists

        # optional packages summary: True if exists for any of the data types
        for l in lib_types[1:]:
            self.compile_time_env['HAVE' + l] = False
            for d in data_types:
                self.compile_time_env['HAVE' + l] |= self.compile_time_env['HAVE_' + d + l]
        # compile only if mpi.h *and* one of the fftw mpi libraries are found
        if have_mpi_h and self.has_header(['fftw3-mpi.h'], include_dirs=self.include_dirs) and self.compile_time_env['HAVE_MPI']:
            found_mpi_types = []
            for d in data_types:
                if self.compile_time_env['HAVE_' + d + '_MPI']:
                    found_mpi_types.append(d)

            print("Enabling mpi support for " + str(found_mpi_types))
        else:
            self.compile_time_env['HAVE_MPI'] = False

        # required package: FFTW itself
        have_fftw = False
        for d in data_types:
            have_fftw |= self.compile_time_env['HAVE_' + d]

        if not have_fftw:
            raise LinkError("Could not find any of the FFTW libraries")

    def has_function(self, function, includes=None, libraries=None, include_dirs=None, library_dirs=None):
        '''Alternative implementation of distutils.ccompiler.has_function that deletes the output and works reliably.'''

        if includes is None:
            includes = []
        if libraries is None:
            libraries = self.libraries
        if include_dirs is None:
            include_dirs = self.include_dirs
        if library_dirs is None:
            library_dirs = self.library_dirs

        msg = "Checking"
        if function:
            msg += " for %s" % function
        if libraries:
            msg += " in " + str(libraries)
        if includes:
            msg += " with includes " + str(includes)
        msg += "..."
        status = "no"

        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix='pyfftw-')
        try:
            try:
                fname = os.path.join(tmpdir, '%s.c' % function)
                f = open(fname, 'w')
                for inc in includes:
                    f.write('#include <%s>\n' % inc)
                if function:
                    f.write('void %s(void);\n' % function)
                # f.write('int main(void) {\n')
                # f.write('    %s();\n' % function)
                # f.write('}\n')
            finally:
                f.close()
                # the root directory
                file_root = os.path.abspath(os.sep)
            try:
                # output file is stored relative to input file since
                # the output has the full directory, joining with the
                # file root gives the right directory
                with stdchannel_redirected(sys.stdout, os.devnull), stdchannel_redirected(sys.stderr, os.devnull):
                    objects = self.compiler.compile([fname], output_dir=file_root, include_dirs=include_dirs)
            except CompileError:
                return False
            except Exception as e:
                log.info(e)
                return False
            try:
                with stdchannel_redirected(sys.stdout, os.devnull), stdchannel_redirected(sys.stderr, os.devnull):
                    # using link_executable, LDFLAGS that the user can modify are ignored
                    self.compiler.link_shared_object(objects,
                                                     os.path.join(tmpdir, 'a.out'),
                                                     libraries=libraries,
                                                     library_dirs=library_dirs)
            except LinkError:
                return False
            except Exception as e:
                print e
                return False
            # no error, seems to work
            status = "ok"
            return True
        finally:
            shutil.rmtree(tmpdir)
            log.info(msg + status)

    def has_header(self, headers, include_dirs=None):
        '''Check for existence and usability of header files by compiling a test file.'''
        return self.has_function(None, includes=headers, include_dirs=include_dirs)

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

    def build_extensions(self):
        '''Check for availability of fftw libraries before building the wrapper.

        Do it here to make sure we use the exact same compiler for checking includes/linking as for building the libraries.'''
        sniffer = EnvironmentSniffer(self.compiler)

        # read out information and modify compiler

        # define macros, that is which part of wrapper is built
        self.cython_compile_time_env = sniffer.compile_time_env

        # prepend automatically generated info to whatever the user specified
        include_dirs = sniffer.include_dirs or []
        if self.include_dirs is not None:
            include_dirs += self.include_dirs
        self.compiler.set_include_dirs(include_dirs)

        libraries = sniffer.libraries or None
        if self.libraries is not None:
            libraries += self.libraries
        self.compiler.set_libraries(libraries)

        library_dirs = sniffer.library_dirs
        if self.library_dirs is not None:
            library_dirs += self.library_dirs
        self.compiler.set_library_dirs(self.library_dirs)

        # delegate actual work to standard implementation
        build_ext.build_extensions(self)

# recompile if any of these files are modified
dependencies = [os.path.join('pyfftw', f) for f in ('pyfftw.pxd', 'mpi.pxd', 'mpi.pxi', 'utils.pxi')]

ext_modules = [Extension('pyfftw.pyfftw',
                         sources=sources,
                         extra_compile_args=['-Wno-maybe-uninitialized'],
                         depends=dependencies)]

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
#         'include_dirs': checker.include_dirs,
# todo package data from checker?
#         'package_data': checker.package_data,
        'cmdclass': {'test': TestCommand,
                     'build_ext': custom_build_ext},
  }

if __name__ == '__main__':
    setup(**setup_args)

# Local variables:
# compile-command: "CC=mpicc python setup.py build_ext -i"
# End:
