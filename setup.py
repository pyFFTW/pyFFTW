# Copyright 2017 Henry Gomersall and the PyFFTW contributors
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

# import only from standard library so dependencies can be installed
try:
    # use setuptools if we can
    from setuptools import setup, Command
    from setuptools.command.build_ext import build_ext
    using_setuptools = True
except ImportError:
    from distutils.core import setup, Command
    from distutils.command.build_ext import build_ext
    using_setuptools = False

from distutils import log
from distutils.ccompiler import get_default_compiler, new_compiler
from distutils.errors import CompileError, LinkError
from distutils.extension import Extension
from distutils.sysconfig import customize_compiler
from distutils.util import get_platform

import contextlib
import os
import sys
import versioneer


if os.environ.get("READTHEDOCS") == "True":
    try:
        environ = os.environb
    except AttributeError:
        environ = os.environ

    environ[b"CC"] = b"x86_64-linux-gnu-gcc"
    environ[b"LD"] = b"x86_64-linux-gnu-ld"
    environ[b"AR"] = b"x86_64-linux-gnu-ar"


def get_include_dirs():
    import numpy
    from pkg_resources import get_build_platform

    include_dirs = [os.path.join(os.getcwd(), 'include'),
                    os.path.join(os.getcwd(), 'pyfftw'),
                    numpy.get_include(),
                    os.path.join(sys.prefix, 'include')]

    if 'PYFFTW_INCLUDE' in os.environ:
        include_dirs.append(os.environ['PYFFTW_INCLUDE'])

    if get_build_platform() in ('win32', 'win-amd64'):
        include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))

    if get_build_platform().startswith('freebsd'):
        include_dirs.append('/usr/local/include')

    return include_dirs

# TODO Do we need to determine package data dynamically? If so, should
# take the output from Sniffer but it's only available when the
# extension is build and not when setup() is called.
def get_package_data():
    from pkg_resources import get_build_platform

    package_data = {}

    if get_build_platform() in ('win32', 'win-amd64'):
        if 'PYFFTW_WIN_CONDAFORGE' in os.environ:
            # fftw3.dll, fftw3f.dll will already be on the path (via the
            # conda environment's \bin subfolder)
            pass
        else:
            # as download from http://www.fftw.org/install/windows.html
            package_data['pyfftw'] = [
                'libfftw3-3.dll', 'libfftw3l-3.dll', 'libfftw3f-3.dll']

    return package_data


def get_library_dirs():
    from pkg_resources import get_build_platform

    library_dirs = []
    if get_build_platform() in ('win32', 'win-amd64'):
        library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))
        library_dirs.append(os.path.join(sys.prefix, 'bin'))

    if 'PYFFTW_LIB_DIR' in os.environ:
        library_dirs.append(os.environ['PYFFTW_LIB_DIR'])

    library_dirs.append(os.path.join(sys.prefix, 'lib'))
    if get_build_platform().startswith('freebsd'):
        library_dirs.append('/usr/local/lib')

    return library_dirs


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


class EnvironmentSniffer(object):
    '''Check for availability of headers and libraries of FFTW and MPI.

    :param compiler:

        Distutils.ccompiler; The compiler should be the compiler that
        is used for actual compilation to ensure that include
        directories, linker flags etc. are identical.

    '''
    def __init__(self, compiler):
        log.debug("Compiler include_dirs: %s" % compiler.include_dirs)
        if hasattr(compiler, "initialize"):
            compiler.initialize() # to set all variables
            log.debug("Compiler include_dirs after initialize: %s" % compiler.include_dirs)
        self.compiler = compiler

        log.debug(sys.version) # contains the compiler used to build this python

        # members with the info for the outside world
        self.include_dirs = get_include_dirs()
        self.objects = []
        self.libraries = []
        self.library_dirs = get_library_dirs()
        self.linker_flags = []
        self.compile_time_env = {}

        if self.compiler.compiler_type == 'msvc':
            if (sys.version_info.major, sys.version_info.minor) < (3, 3):
                # The check above is a nasty hack. We're using the python
                # version as a proxy for the MSVC version. 2008 doesn't
                # have stdint.h, so is needed. 2010 does.
                #
                # We need to add the path to msvc includes
                msvc_2008_path = (os.path.join(os.getcwd(), 'include', 'msvc_2008'))
                self.include_dirs.append(msvc_2008_path)
            elif (sys.version_info.major, sys.version_info.minor) < (3, 5):
                # Actually, it seems that appveyor doesn't have a stdint that
                # works, so even for 2010 we use our own (hacked) version
                # of stdint.
                # This should be pretty safe in whatever case.
                msvc_2010_path = (os.path.join(os.getcwd(), 'include', 'msvc_2010'))
                self.include_dirs.append(msvc_2010_path)

                # To avoid http://bugs.python.org/issue4431
                #
                # C:\Program Files\Microsoft
                # SDKs\Windows\v7.1\Bin\x64\mt.exe -nologo -manifest
                # C:\Users\appveyor\AppData\Local\Temp\1\pyfftw-9in6l66u\a.out.exe.manifest
                # -outputresource:C:\Users\appveyor\AppData\Local\Temp\1\pyfftw-9in6l66u\a.out.exe;1
                # C:\Users\appveyor\AppData\Local\Temp\1\pyfftw-9in6l66u\a.out.exe.manifest
                # : general error c1010070: Failed to load and parse
                # the manifest. The system cannot find the file
                # specified.
                self.compiler.ldflags_shared.append('/MANIFEST')

        if get_platform().startswith('linux'):
            # needed at least libm for linker checks to succeed
            self.libraries.append('m')

        # main fftw3 header is required
        if not self.has_header(['fftw3.h'], include_dirs=self.include_dirs):
            raise CompileError("Could not find the FFTW header 'fftw3.h'")

        # mpi is optional
        # self.support_mpi = self.has_header(['mpi.h', 'fftw3-mpi.h'])
        # TODO enable check when wrappers are included in Pyfftw
        self.support_mpi = False

        if self.support_mpi:
            try:
                import mpi4py
                self.include_dirs.append(mpi4py.get_include())
            except ImportError:
                log.error("Could not import mpi4py. Skipping support for FFTW MPI.")
                self.support_mpi = False

        self.search_dependencies()

    def search_dependencies(self):

        # lib_checks = {}
        data_types = ['DOUBLE', 'SINGLE', 'LONG']
        data_types_short = ['', 'f', 'l']
        lib_types = ['', 'THREADS', 'OMP']
        functions = ['plan_dft', 'init_threads', 'init_threads']
        if self.support_mpi:
            lib_types.append('MPI')
            functions.append('mpi_init')

        for d, s in zip(data_types, data_types_short):
            # first check for serial library...
            basic_lib = self.check('', 'plan_dft', d, s, True)
            self.add_library(basic_lib)

            # ...then multithreading: link check with threads requires
            # the serial library. Both omp and posix define the same
            # function names. Prefer openmp if linking dynamically,
            # else fall back to pthreads.  pthreads can be prioritized over
            # openmp by defining the environment variable PYFFTW_USE_PTHREADS
            if 'PYFFTW_USE_PTHREADS' not in os.environ:
                # openmp requires special linker treatment
                self.linker_flags.append(self.openmp_linker_flag())
                lib_omp = self.check('OMP', 'init_threads', d, s,
                                     basic_lib and not hasattr(self, 'static_fftw_dir'))
                if lib_omp:
                    self.add_library(lib_omp)
                    # manually set flag because it won't be checked below
                    self.compile_time_env[self.HAVE(d, 'THREADS')] = False
                else:
                    self.linker_flags.pop()
            else:
                lib_omp = False
                self.compile_time_env[self.HAVE(d, 'OMP')] = False

            if lib_omp:
                self.compile_time_env[self.HAVE(d, 'THREADS')] = False

            if not lib_omp:
                # -pthread added for gcc/clang when checking for threads
                self.linker_flags.append(self.pthread_linker_flag())
                lib_pthread = self.check('THREADS', 'init_threads', d, s,
                                         basic_lib)
                if lib_pthread:
                    self.add_library(lib_pthread)
                else:
                    self.linker_flags.pop()

            # On windows, the serial and posix threading functions are
            # build into one library as released on fftw.org. openMP
            # and MPI are not supported in the releases
            if get_platform() in ('win32', 'win-amd64'):
                if basic_lib:
                    self.compile_time_env[self.HAVE(d, 'THREADS')] = True

            # check whatever multithreading is available
            self.compile_time_env[self.HAVE(d, 'MULTITHREADING')] = self.compile_time_env[self.HAVE(d, 'OMP')] or self.compile_time_env[self.HAVE(d, 'THREADS')]

            # check MPI only if headers were found
            self.add_library(self.check('MPI', 'mpi_init', d, s, basic_lib and self.support_mpi))

        # compile only if mpi.h *and* one of the fftw mpi libraries are found
        if self.support_mpi:
            found_mpi_types = []
            for d in data_types:
                if self.compile_time_env['HAVE_' + d + '_MPI']:
                    found_mpi_types.append(d)
        else:
            self.compile_time_env['HAVE_MPI'] = False

        # Pretend FFTW precision not available, regardless if it was found or
        # not. Useful for testing that pyfftw still works without requiring all
        # precisions
        if 'PYFFTW_IGNORE_SINGLE' in os.environ:
            self.compile_time_env['HAVE_SINGLE'] = False
        if 'PYFFTW_IGNORE_DOUBLE' in os.environ:
            self.compile_time_env['HAVE_DOUBLE'] = False
        if 'PYFFTW_IGNORE_LONG' in os.environ:
            self.compile_time_env['HAVE_LONG'] = False

        log.debug(repr(self.compile_time_env))

        # required package: FFTW itself
        have_fftw = False
        for d in data_types:
            have_fftw |= self.compile_time_env['HAVE_' + d]

        if not have_fftw:
            raise LinkError("Could not find any of the FFTW libraries")

        log.info('Build pyFFTW with support for FFTW with')
        for d in data_types:
            if not self.compile_time_env[self.HAVE(d)]:
                continue
            s = d.lower() + ' precision'
            if self.compile_time_env[self.HAVE(d, 'OMP')]:
                s += ' + openMP'
            elif self.compile_time_env[self.HAVE(d, 'THREADS')]:
                s += ' + pthreads'
            if self.compile_time_env[self.HAVE(d, 'MPI')]:
                s += ' + MPI'
            log.info(s)

    def check(self, lib_type, function, data_type, data_type_short, do_check):
        m = self.HAVE(data_type, lib_type)
        exists = False
        lib = ''
        if do_check:
            lib = self.lib_root_name(
                'fftw3' + data_type_short +
                ('_' + lib_type.lower() if lib_type else ''))
            function = 'fftw' + data_type_short + '_' + function
            exists = self.has_library(lib, function)

        self.compile_time_env[m] = exists
        return lib if exists else ''

    def HAVE(self, data_type, lib_type=''):
        s = 'HAVE_' + data_type
        if lib_type:
            return s + '_' + lib_type
        else:
            return s

    def lib_root_name(self, lib):
        '''Build the name of the lib to pass to the python compiler
        interface.

        Examples:

        With a unix compiler, `fftw3` is unchanged but the interface
        passes `-lfftw3` to the linker.

        On windows, `fftw3l` -> `libfftw3l-3` and the interfaces
        passes `libfftw3l-3.libf` to the linker.

        '''
        if get_platform() in ('win32', 'win-amd64'):
            if 'PYFFTW_WIN_CONDAFORGE' in os.environ:
                return '%s' % lib
            else:
                # for download from http://www.fftw.org/install/windows.html
                return 'lib%s-3' % lib

        else:
            return lib

    def add_library(self, lib):
        raise NotImplementedError

    def has_function(self, function, includes=None, objects=None, libraries=None,
                     include_dirs=None, library_dirs=None, linker_flags=None):
        '''Alternative implementation of distutils.ccompiler.has_function that
deletes the output and hides calls to the compiler and linker.'''
        if includes is None:
            includes = []
        if objects is None:
            objects = self.objects
        if libraries is None:
            libraries = self.libraries
        if include_dirs is None:
            include_dirs = self.include_dirs
        if library_dirs is None:
            library_dirs = self.library_dirs
        if linker_flags is None:
            linker_flags = self.linker_flags

        msg = "Checking"
        if function:
            msg += " for %s" % function
        if includes:
            msg += " with includes " + str(includes)
        msg += "..."
        status = "no"

        log.debug("objects: %s" % objects)
        log.debug("libraries: %s" % libraries)
        log.debug("include dirs: %s" % include_dirs)

        import tempfile, shutil

        tmpdir = tempfile.mkdtemp(prefix='pyfftw-')
        try:
            try:
                fname = os.path.join(tmpdir, '%s.c' % function)
                f = open(fname, 'w')
                for inc in includes:
                    f.write('#include <%s>\n' % inc)
                f.write("""\
                int main() {
                """)
                if function:
                    f.write('%s();\n' % function)
                f.write("""\
                return 0;
                }""")
            finally:
                f.close()
                # the root directory
                file_root = os.path.abspath(os.sep)
            try:
                # output file is stored relative to input file since
                # the output has the full directory, joining with the
                # file root gives the right directory
                stdout = os.path.join(tmpdir, "compile-stdout")
                stderr = os.path.join(tmpdir, "compile-stderr")
                with stdchannel_redirected(sys.stdout, stdout), stdchannel_redirected(sys.stderr, stderr):
                    tmp_objects = self.compiler.compile([fname], output_dir=file_root, include_dirs=include_dirs)
                with open(stdout, 'r') as f: log.debug(f.read())
                with open(stderr, 'r') as f: log.debug(f.read())
            except CompileError:
                with open(stdout, 'r') as f: log.debug(f.read())
                with open(stderr, 'r') as f: log.debug(f.read())
                return False
            except Exception as e:
                log.error(e)
                return False
            try:
                # additional objects should come last to resolve symbols, linker order matters
                tmp_objects.extend(objects)
                stdout = os.path.join(tmpdir, "link-stdout")
                stderr = os.path.join(tmpdir, "link-stderr")
                with stdchannel_redirected(sys.stdout, stdout), stdchannel_redirected(sys.stderr, stderr):
                    # TODO using link_executable, LDFLAGS that the
                    # user can modify are ignored
                    self.compiler.link_executable(tmp_objects, 'a.out',
                                                  output_dir=tmpdir,
                                                  libraries=libraries,
                                                  extra_preargs=linker_flags,
                                                  library_dirs=library_dirs)
                with open(stdout, 'r') as f: log.debug(f.read())
                with open(stderr, 'r') as f: log.debug(f.read())
            except (LinkError, TypeError):
                with open(stdout, 'r') as f: log.debug(f.read())
                with open(stderr, 'r') as f: log.debug(f.read())
                return False
            except Exception as e:
                log.error(e)
                return False
            # no error, seems to work
            status = "ok"
            return True
        finally:
            shutil.rmtree(tmpdir)
            log.debug(msg + status)

    def has_header(self, headers, include_dirs=None):
        '''Check for existence and usability of header files by compiling a test file.'''
        return self.has_function(None, includes=headers, include_dirs=include_dirs)

    def has_library(function, lib):
        raise NotImplementedError

    def openmp_linker_flag(self):
        # gcc and newer clang support openmp
        if self.compiler.compiler_type == 'unix':
            return '-fopenmp'
        # TODO support other compilers
        else:
            return ''

    def pthread_linker_flag(self):
        # gcc and clang
        if self.compiler.compiler_type == 'unix':
            return '-pthread'
        else:
            # TODO support other compilers
            return ''

class StaticSniffer(EnvironmentSniffer):
    def __init__(self, compiler):
        self.static_fftw_dir = os.environ.get('STATIC_FFTW_DIR', None)
        if not os.path.exists(self.static_fftw_dir):
            raise LinkError('STATIC_FFTW_DIR="%s" was specified but does not exist' % self.static_fftw_dir)

        # call parent init
        super(self.__class__, self).__init__(compiler)

    def has_library(self, root_name, function):
        '''Expect library in root form'''
        # get full name of lib
        objects = [os.path.join(self.static_fftw_dir, self.lib_full_name(root_name))]
        objects.extend(self.objects)
        return self.has_function(function, objects=objects)

    def lib_full_name(self, root_lib):
        # TODO use self.compiler.library_filename
        from pkg_resources import get_build_platform
        if get_build_platform() in ('win32', 'win-amd64'):
            lib_pre = ''
            lib_ext = '.lib'
        else:
            lib_pre = 'lib'
            lib_ext = '.a'
        return os.path.join(self.static_fftw_dir, lib_pre + root_lib + lib_ext)

    def add_library(self, lib):
        full_name = self.lib_full_name(self.lib_root_name(lib))
        if lib:
            self.objects.insert(0, full_name)

class DynamicSniffer(EnvironmentSniffer):
    def __init__(self, compiler):
        super(self.__class__, self).__init__(compiler)

    def has_library(self, lib, function):
        '''Expect lib in root name so it can be passed to compiler'''
        libraries = [lib]
        libraries.extend(self.libraries)
        return self.has_function(function, libraries=libraries)

    def add_library(self, lib):
        if lib:
            self.libraries.insert(0, lib)

def make_sniffer(compiler):
    if os.environ.get('STATIC_FFTW_DIR', None) is None:
        log.debug("Link FFTW dynamically")
        return DynamicSniffer(compiler)
    else:
        log.debug("Link FFTW statically")
        return StaticSniffer(compiler)

def get_extensions():
    ext_modules = [Extension('pyfftw.pyfftw',
                             sources=[os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.pyx')])]
    return ext_modules


long_description = '''
pyFFTW is a pythonic wrapper around `FFTW <http://www.fftw.org/>`_, the
speedy FFT library. The ultimate aim is to present a unified interface for all
the possible transforms that FFTW can perform.

Both the complex DFT and the real DFT are supported, as well as arbitrary
axes of arbitrary shaped and strided arrays, which makes it almost
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
`here <http://pyfftw.readthedocs.io>`_, and the source
is on `github <https://github.com/pyFFTW/pyFFTW>`_.
'''


class custom_build_ext(build_ext):
    def finalize_options(self):

        build_ext.finalize_options(self)

        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

    def build_extensions(self):
        '''Check for availability of fftw libraries before building the wrapper.

        Do it here to make sure we use the exact same compiler for checking includes/linking as for building the libraries.'''
        sniffer = make_sniffer(self.compiler)

        # read out information and modify compiler

        # define macros, that is which part of wrapper is built
        self.cython_compile_time_env = sniffer.compile_time_env

        # call `extend()` to keep argument set neither by sniffer nor by
        # user. On windows there are includes set automatically, we
        # must not lose them.

        # prepend automatically generated info to whatever the user specified
        include_dirs = sniffer.include_dirs or []
        if self.include_dirs is not None:
            include_dirs += self.include_dirs
        self.compiler.include_dirs.extend(include_dirs)

        libraries = sniffer.libraries or None
        if self.libraries is not None:
            if libraries is None:
                libraries = self.libraries
            else:
                libraries += self.libraries
        self.compiler.libraries.extend(libraries)

        library_dirs = sniffer.library_dirs
        if self.library_dirs is not None:
            library_dirs += self.library_dirs
        self.compiler.library_dirs.extend(library_dirs)

        objects = sniffer.objects
        if self.link_objects is not None:
            objects += self.objects
        self.compiler.set_link_objects(objects)

        # delegate actual work to standard implementation
        build_ext.build_extensions(self)


class CreateChangelogCommand(Command):
    '''Depends on the ruby program github_changelog_generator. Install with
    gem install gihub_changelog_generator.
    '''
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        github_token_file = 'github_changelog_generator_token'

        with open(github_token_file) as f:
            github_token = f.readline().strip()

        subprocess.call(['github_changelog_generator', '-t', github_token])


class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
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

        import subprocess
        subprocess.check_call([sys.executable, '-m',
                                 'unittest'] + quick_test_cases)


cmdclass = {'test': TestCommand,
            'quick_test': QuickTestCommand,
            'build_ext': custom_build_ext,
            'create_changelog': CreateChangelogCommand}
cmdclass.update(versioneer.get_cmdclass())


def setup_package():

    # Figure out whether to add ``*_requires = ['numpy']``.
    build_requires = []
    numpy_requirement = 'numpy>=1.10, <2.0'
    try:
        import numpy
    except ImportError:
        build_requires = [numpy_requirement]

    # we require cython because we need to know which part of the wrapper
    # to build to avoid missing symbols at runtime. But if this script is
    # called without building pyfftw, for example to install the
    # dependencies, then we have to hide the cython dependency.
    try:
        import cython
    except ImportError:
        build_requires.append('cython>=0.23, <1.0')

    install_requires = [numpy_requirement]

    opt_requires = {
        'dask': ['numpy>=1.10, <2.0', 'dask[array]>=0.15.0'],
        'scipy': ['scipy>=0.12.0']
    }

    setup_args = {
        'name': 'pyFFTW',
        'version': versioneer.get_version(),
        'author': 'Henry Gomersall',
        'author_email': 'heng@kedevelopments.co.uk',
        'description': (
            'A pythonic wrapper around FFTW, the FFT library, presenting a '
            'unified interface for all the supported transforms.'),
        'url': 'https://github.com/pyFFTW/pyFFTW',
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
            'Topic :: Multimedia :: Sound/Audio :: Analysis'],
        'cmdclass': cmdclass
    }

    if using_setuptools:
        setup_args['setup_requires'] = build_requires
        setup_args['install_requires'] = install_requires
        setup_args['extras_require'] = opt_requires

    if len(sys.argv) >= 2 and (
        '--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean')):
        # For these actions, NumPy is not required.
        pass
    else:
        setup_args['packages'] = [
            'pyfftw', 'pyfftw.builders', 'pyfftw.interfaces']
        setup_args['ext_modules'] = get_extensions()
        setup_args['package_data'] = get_package_data()

    setup(**setup_args)

if __name__ == '__main__':
    setup_package()
