# Copyright 2015 Knowledge Economy Developments Ltd
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
from __future__ import print_function

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

import os, sys

# we require cython because we need to know which part of wrapper to build to
# avoid missing symbols at run time. But if this script is called without
# building pyfftw, then we may hide cython dependency.
# TODO Drop zig-zag to avoid cython dependency below
# from Cython.Distutils import build_ext

MAJOR = 0
MINOR = 10
MICRO = 5
ISRELEASED = False

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

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

class EnvironmentSniffer(object):
    '''Check for availability of headers and libraries of FFTW and MPI.

    :param compiler:

        Distutils.ccompiler; The compiler should be the compiler that
        is used for actual compilation to ensure that include
        directories, linker flags etc. are identical.

    '''
    def __init__(self, compiler):
        self.compiler = compiler

        import numpy
        # members with the info for the outside world
        self.include_dirs = [os.path.join(os.getcwd(), 'include'),
                             os.path.join(os.getcwd(), 'pyfftw'),
                             numpy.get_include()]
        self.objects = []
        self.libraries = []
        self.library_dirs = []
        self.linker_flags = []
        self.package_data = {} # TODO why package data only updated for windows?
        self.compile_time_env = {}

        # main fftw3 header is required
        if not self.has_header(['fftw3.h'], include_dirs=self.include_dirs):
            raise CompileError("Could not find the FFTW header 'fftw3.h'")

        # mpi is optional
        self.support_mpi = self.has_header(['mpi.h', 'fftw3-mpi.h'])

        if self.support_mpi:
            try:
                import mpi4py
                self.include_dirs.append(mpi4py.get_include())
            except ImportError:
                print("Could not import mpi4py. Skipping support for FFTW MPI.")
                support_mpi = False

        platform = get_platform()

        if platform in ('win32', 'win-amd64'):
            self.include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))
            self.library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))
            # TODO fix package data *after* we know which libraries exist
            # self.package_data['pyfftw'] = [lib + '.dll' for lib in self.libraries]
            # TODO What about thread libraries on windows?
            # TODO mpi support missing and untested on windows
        elif platform.startswith('linux'):
            # needed at least for linker checks to succeed
            self.libraries.insert(0, 'm')

        self.search_dependencies()

    def search_dependencies(self):

        # lib_checks = {}
        data_types = ['DOUBLE', 'SINGLE', 'LONG', 'QUAD']
        data_types_short = ['', 'f', 'l', 'q']
        lib_types = ['', 'THREADS', 'OMP']
        functions = ['plan_dft', 'init_threads', 'init_threads']
        if self.support_mpi:
            lib_types.append('_MPI')
            functions.append('mpi_init')

        for d, s in zip(data_types, data_types_short):
            # first check for serial library...
            basic_lib = self.check('', 'plan_dft', d, s, True)
            self.add_library(basic_lib)

            # ...then multithreading: link check with threads requires
            # the serial library. Both omp and posix define the same
            # function names. Prefer openmp if it works, fall back to
            # pthreads.

            # openmp requires special linker treatment
            self.linker_flags.append(self.openmp_linker_flag())
            lib_omp = self.check('OMP', 'init_threads', d, s, basic_lib)
            if lib_omp:
                self.add_library(lib_omp)
            else:
                self.linker_flags.pop()

            self.add_library(self.check('THREADS', 'init_threads', d, s,
                                        basic_lib and not lib_omp))

            # check MPI only if headers were found
            self.add_library(self.check('MPI', 'mpi_init', d, s, basic_lib and self.support_mpi))

        # optional packages summary: True if exists for any of the data types
        for l in lib_types[1:]:
            self.compile_time_env['HAVE_' + l] = False
            for d in data_types:
                self.compile_time_env['HAVE_' + l] |= self.compile_time_env[self.HAVE(d, l)]

        # compile only if mpi.h *and* one of the fftw mpi libraries are found
        if self.support_mpi:
            found_mpi_types = []
            for d in data_types:
                if self.compile_time_env['HAVE_' + d + '_MPI']:
                    found_mpi_types.append(d)

            print("Enabling mpi support for " + str(found_mpi_types))
        else:
            self.compile_time_env['HAVE_MPI'] = False

        log.debug(self.compile_time_env)
        # required package: FFTW itself
        have_fftw = False
        for d in data_types:
            have_fftw |= self.compile_time_env['HAVE_' + d]

        if not have_fftw:
            raise LinkError("Could not find any of the FFTW libraries")

        log.info('Supporting FFTW with')
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
            lib = self.lib_root_name('fftw3' + data_type_short + ('_' + lib_type.lower() if lib_type else ''))
            function = 'fftw' + data_type_short + '_' + function
            exists =  self.has_library(lib, function)

        self.compile_time_env[m] = exists
        return lib if exists else ''

    def HAVE(self, data_type, lib_type=''):
        s = 'HAVE_' + data_type
        if lib_type:
            return s + '_' + lib_type
        else:
            return s

    def lib_root_name(self, lib):
        '''Build the name of the lib w/o prefix and suffix.

        Example: fftw3l -> fftw3l-3 (windows)
        '''
        if get_platform() in ('win32', 'win-amd64'):
            return lib + '-3'
        else:
            return lib

    def lib_name(self, lib):
        '''Name of the library with prefix and suffix but w/o directory component'''
        raise NotImplementedError

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
                with stdchannel_redirected(sys.stdout, os.devnull), stdchannel_redirected(sys.stderr, os.devnull):
                    tmp_objects = self.compiler.compile([fname], output_dir=file_root, include_dirs=include_dirs)
            except CompileError:
                return False
            except Exception as e:
                log.error(e)
                return False
            try:
                # additional objects should come last to resolve symbols, linker order matters
                tmp_objects.extend(objects)
                # TODO using link_executable, LDFLAGS that the user can modify are ignored
                # but with link_executable, it doesn't fail if library exists but doesn't actually define the symbol we are looking for
                with stdchannel_redirected(sys.stdout, os.devnull), stdchannel_redirected(sys.stderr, os.devnull):
                # if True:
                    self.compiler.link_executable(tmp_objects, 'a.out',
                                                  output_dir=tmpdir,
                                                  libraries=libraries,
                                                  extra_preargs=linker_flags,
                                                  library_dirs=library_dirs)
            except (LinkError, TypeError):
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

class StaticSniffer(EnvironmentSniffer):
    def __init__(self, compiler):
        # TODO check if STATIC_FFTW_DIR exists
        self.static_fftw_dir = os.environ.get('STATIC_FFTW_DIR', None)

        # call parent init
        super(self.__class__, self).__init__(compiler)

    def has_library(self, lib, function):
        root_name = self.lib_root_name(lib)
        # get full name of lib
        objects = [os.path.join(self.static_fftw_dir, self.lib_full_name(root_name))]
        objects.extend(self.objects)
        return self.has_function(function, objects=objects)

    def lib_full_name(self, root_lib):
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
        lib = self.lib_root_name(lib)
        libraries = [lib]
        libraries.extend(self.libraries)
        return self.has_function(function, libraries=libraries)

    def add_library(self, lib):
        if lib:
            self.libraries.insert(0, lib)

def make_sniffer(compiler):
    if os.environ.get('STATIC_FFTW_DIR', None) is None:
        return DynamicSniffer(compiler)
    else:
        return StaticSniffer(compiler)

if os.environ.get("READTHEDOCS") == "True":
    try:
        environ = os.environb
    except AttributeError:
        environ = os.environ

    environ[b"CC"] = b"x86_64-linux-gnu-gcc"
    environ[b"LD"] = b"x86_64-linux-gnu-ld"
    environ[b"AR"] = b"x86_64-linux-gnu-ar"

# TODO need to determine package data dynamically
def get_package_data():
    from pkg_resources import get_build_platform

    package_data = {}

    if get_build_platform() in ('win32', 'win-amd64'):
        package_data['pyfftw'] = [
            'libfftw3-3.dll', 'libfftw3l-3.dll', 'libfftw3f-3.dll']

    return package_data

# TODO integrate into sniffer
def get_include_dirs():
    import numpy
    from pkg_resources import get_build_platform

    include_dirs = [os.path.join(os.getcwd(), 'include'),
                    os.path.join(os.getcwd(), 'pyfftw'),
                    numpy.get_include(),
                    os.path.join(sys.prefix, 'include')]

    if get_build_platform() in ('win32', 'win-amd64'):
        include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))

    if get_build_platform().startswith('freebsd'):
        include_dirs.append('/usr/local/include')

    return include_dirs

# TODO integrate into or call from sniffer
def get_library_dirs():
    from pkg_resources import get_build_platform

    library_dirs = []
    if get_build_platform() in ('win32', 'win-amd64'):
        library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))
        library_dirs.append(os.path.join(sys.prefix, 'bin'))

    library_dirs.append(os.path.join(sys.prefix, 'lib'))
    if get_build_platform().startswith('freebsd'):
        library_dirs.append('/usr/local/lib')

    return library_dirs

# TODO integrate or call from sniffer
def get_libraries():
    from pkg_resources import get_build_platform

    if get_build_platform() in ('win32', 'win-amd64'):
        libraries = ['libfftw3-3', 'libfftw3f-3', 'libfftw3l-3']

    else:
        libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
                     'fftw3f_threads', 'fftw3l_threads']

    return libraries

def get_extensions():
    from Cython.Build import cythonize

    ext_modules = [Extension('pyfftw.pyfftw',
                             sources=[os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.pyx')],
                             extra_compile_args=['-Wno-maybe-uninitialized'])]
    return cythonize(ext_modules)

# TODO get_extensions changes lib dependencies. Make it work with our custom_build_ext
def get_extensions2():
    # will use static linking if STATIC_FFTW_DIR defined
    static_fftw_path = os.environ.get('STATIC_FFTW_DIR', None)
    link_static_fftw = static_fftw_path is not None

    common_extension_args = {
        'include_dirs': get_include_dirs(),
        'library_dirs': get_library_dirs(),
        'extra_compile_args': ['-Wno-maybe-uninitialized'],
        }

    try:
        from Cython.Build import cythonize
        sources = [os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.pyx')]
        have_cython = True

    except ImportError as e:
        # no cython
        sources = [os.path.join(os.getcwd(), 'pyfftw', 'pyfftw.c')]
        if not os.path.exists(sources[0]):
            raise ImportError(
                str(e) + '. ' +
                'Cython is required to build the initial .c file.')

        have_cython = False

    libraries = get_libraries()
    if link_static_fftw:
        from pkg_resources import get_build_platform
        if get_build_platform() in ('win32', 'win-amd64'):
            lib_pre = ''
            lib_ext = '.lib'
        else:
            lib_pre = 'lib'
            lib_ext = '.a'
        extra_link_args = []
        for lib in libraries:
            extra_link_args.append(
                os.path.join(static_fftw_path, lib_pre + lib + lib_ext))

        common_extension_args['extra_link_args'] = extra_link_args
        common_extension_args['libraries'] = []
    else:
        # otherwise we use dynamic libraries
        common_extension_args['extra_link_args'] = []
        common_extension_args['libraries'] = libraries

    ext_modules = [
        Extension('pyfftw.pyfftw', sources=sources,
                  **common_extension_args)]

    if have_cython:
        return cythonize(ext_modules)

    else:
        return ext_modules

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

        if compiler == 'msvc':
            # Add msvc specific hacks

            if (sys.version_info.major, sys.version_info.minor) < (3, 3):
                # The check above is a nasty hack. We're using the python
                # version as a proxy for the MSVC version. 2008 doesn't
                # have stdint.h, so is needed. 2010 does.
                #
                # We need to add the path to msvc includes

                msvc_2008_path = (
                    os.path.join(os.getcwd(), 'include', 'msvc_2008'))

                if self.include_dirs is not None:
                    self.include_dirs.append(msvc_2008_path)
                else:
                    self.include_dirs = [msvc_2008_path]

            elif (sys.version_info.major, sys.version_info.minor) < (3, 5):
                # Actually, it seems that appveyor doesn't have a stdint that
                # works, so even for 2010 we use our own (hacked) version
                # of stdint.
                # This should be pretty safe in whatever case.
                msvc_2010_path = (
                    os.path.join(os.getcwd(), 'include', 'msvc_2010'))

                if self.include_dirs is not None:
                    self.include_dirs.append(msvc_2010_path)
                else:
                    self.include_dirs = [msvc_2010_path]

            # We need to prepend lib to all the library names
            _libraries = []
            for each_lib in self.libraries:
                _libraries.append('lib' + each_lib)

            self.libraries = _libraries

    def build_extensions(self):
        '''Check for availability of fftw libraries before building the wrapper.

        Do it here to make sure we use the exact same compiler for checking includes/linking as for building the libraries.'''
        sniffer = make_sniffer(self.compiler)

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
        self.compiler.set_library_dirs(library_dirs)

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
        errno = subprocess.call([sys.executable, '-m',
            'unittest'] + quick_test_cases)
        raise SystemExit(errno)


# borrowed from scipy via pyNFFT
def git_version():

    import subprocess

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# borrowed from scipy via pyNFFT
def get_version_info():
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(os.path.join('pyfftw', 'version.py')):
        # must be a source distribution, use existing version file
        # load it as a separate module in order not to load __init__.py
        import imp
        version = imp.load_source(
            'pyfftw.version', os.path.join('pyfftw', 'version.py'))
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION

# borrowed from scipy via pyNFFT
def write_version_py(filename=None):

    if filename is None:
        filename = os.path.join('pyfftw', 'version.py')

    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version

if __name__ == "__main__":
    print(short_version)
    print(version)
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    f = open(filename, 'w')
    try:
        f.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        f.close()

def setup_package():

    # Get current version
    FULLVERSION, GIT_REVISION = get_version_info()

    # Refresh version file if we're not a source release
    if ISRELEASED and os.path.exists(os.path.join('pyfftw', 'version.py')):
        pass
    else:
        write_version_py()

    # Figure out whether to add ``*_requires = ['numpy']``.
    build_requires = []
    try:
        import numpy
    except ImportError:
        build_requires = ['numpy>=1.6, <2.0']

    try:
        import cython
    except ImportError:
        build_requires.append('cython>=0.23, <1.0')

    install_requires = []
    install_requires.extend(build_requires)

    setup_args = {
        'name': 'pyFFTW',
        'version': FULLVERSION,
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
        'cmdclass': {'test': TestCommand,
                     'quick_test': QuickTestCommand,
                     'build_ext': custom_build_ext,
                     'create_changelog': CreateChangelogCommand}
    }

    if using_setuptools:
        setup_args['setup_requires'] = build_requires
        setup_args['install_requires'] = install_requires

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
