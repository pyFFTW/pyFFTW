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

from __future__ import print_function

# TODO distutils or setuptools? see http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2
# from distutils.core import setup, Command
# from distutils import log
# from distutils.extension import Extension
# from distutils.util import get_platform
# from distutils.ccompiler import get_default_compiler, new_compiler
# from distutils.errors import CompileError, LinkError
# from distutils.sysconfig import customize_compiler

try:
    # use setuptools if we can
    from setuptools import setup, Command
    # from setuptools.command.build_ext import build_ext
    using_setuptools = True
except ImportError:
    from distutils.core import setup, Command
    # from distutils.command.build_ext import build_ext
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
from Cython.Distutils import build_ext

# todo infos still printed, even with error threshold
# 0=minimal output, 2=maximum debug
# log.set_verbosity(0)
log.set_threshold(log.ERROR)

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

class EnvironmentSniffer:
    '''Check for availability of headers and libraries of FFTW and MPI.

    :param compiler:

        Distutils.ccompiler; The compiler should preferably be the compiler
        that is used for actual compilation to ensure that include directories etc are identical.

    :param exclude:

        Iterable of strings; pass packages to ignore in here. Example: exclude=('DOUBLE_MPI', 'SINGLE_MPI', 'LONG_MPI', 'QUAD_MPI')

    '''
    def __init__(self, compiler, exclude=None):
        import numpy

        self.include_dirs = [os.path.join(os.getcwd(), 'include'),
                             os.path.join(os.getcwd(), 'pyfftw'),
                             numpy.get_include()]
        self.libraries = []
        self.library_dirs = []
        self.package_data = {} # TODO why package data only updated for windows?
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
                print(e)
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

if os.environ.get("READTHEDOCS") == "True":
    try:
        environ = os.environb
    except AttributeError:
        environ = os.environ

    environ[b"CC"] = b"x86_64-linux-gnu-gcc"
    environ[b"LD"] = b"x86_64-linux-gnu-ld"
    environ[b"AR"] = b"x86_64-linux-gnu-ar"

def get_package_data():
    from pkg_resources import get_build_platform

    package_data = {}

    if get_build_platform() in ('win32', 'win-amd64'):
        package_data['pyfftw'] = [
            'libfftw3-3.dll', 'libfftw3l-3.dll', 'libfftw3f-3.dll']

    return package_data

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

def get_libraries():
    from pkg_resources import get_build_platform

    if get_build_platform() in ('win32', 'win-amd64'):
        libraries = ['libfftw3-3', 'libfftw3f-3', 'libfftw3l-3']

    else:
        libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
                     'fftw3f_threads', 'fftw3l_threads']

    return libraries

# TODO get_extensions changes lib dependencies. Make it work with our custom_build_ext
def get_extensions():
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
