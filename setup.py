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

try:
    # use setuptools if we can
    from setuptools import setup, Command, Extension
    from setuptools.command.build_ext import build_ext
    using_setuptools = True
except ImportError:
    from distutils.core import setup, Command, Extension
    from distutils.command.build_ext import build_ext
    using_setuptools = False

from distutils.ccompiler import get_default_compiler

import os
import sys

#if os.environ.get('TRAVIS_TAG') != '':
#    git_tag = os.environ.get('TRAVIS_TAG')
#elif os.environ.get('APPVEYOR_REPO_TAG_NAME') is not None:
#    git_tag = os.environ.get('APPVEYOR_REPO_TAG_NAME')
#else:
#    git_tag = None

MAJOR = 0
MINOR = 10
MICRO = 4
ISRELEASED = True

#if git_tag is not None:
#    # Check the tag is properly formed and in agreement with the
#    # expected versio number before declaring a release.
#    import re
#    version_re = re.compile(r'v[0-9]+\.[0-9]+\.[0-9]+')
#    if version_re.match(git_tag) is not None:
#        tag_major, tag_minor, tag_micro = [
#            int(each) for each in git_tag[1:].split('.')]
#        
#        assert tag_major == MAJOR
#        assert tag_minor == MINOR
#        assert tag_micro == MICRO
#        
#        ISRELEASED = True
#    else:
#        raise ValueError("Malformed version tag for release")
#
#else:
#    ISRELEASED = False

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

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
                    numpy.get_include()]

    if get_build_platform() in ('win32', 'win-amd64'):
        include_dirs.append(os.path.join(os.getcwd(), 'include', 'win'))

    return include_dirs

def get_library_dirs():
    from pkg_resources import get_build_platform

    library_dirs = []
    if get_build_platform() in ('win32', 'win-amd64'):
        library_dirs.append(os.path.join(os.getcwd(), 'pyfftw'))

    return library_dirs

def get_libraries():
    from pkg_resources import get_build_platform

    if get_build_platform() in ('win32', 'win-amd64'):
        libraries = ['libfftw3-3', 'libfftw3f-3', 'libfftw3l-3']

    else:
        libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
                     'fftw3f_threads', 'fftw3l_threads']

    return libraries

def get_extensions():
    from distutils.extension import Extension

    # will use static linking if STATIC_FFTW_DIR defined
    static_fftw_path = os.environ.get('STATIC_FFTW_DIR', None)
    link_static_fftw = static_fftw_path is not None

    common_extension_args = {
        'include_dirs': get_include_dirs(),
        'library_dirs': get_library_dirs()}

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
`here <http://hgomersall.github.com/pyFFTW/>`_, and the source
is on `github <https://github.com/hgomersall/pyFFTW>`_.
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

    setup_args = {
        'name': 'pyFFTW',
        'version': FULLVERSION,
        'author': 'Henry Gomersall',
        'author_email': 'heng@kedevelopments.co.uk',
        'description': (
            'A pythonic wrapper around FFTW, the FFT library, presenting a '
            'unified interface for all the supported transforms.'),
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
            'Topic :: Multimedia :: Sound/Audio :: Analysis'],
        'cmdclass': {'test': TestCommand,
                     'quick_test': QuickTestCommand,
                     'build_ext': custom_build_ext,
                     'create_changelog': CreateChangelogCommand}
    }

    if using_setuptools:
        setup_args['setup_requires'] = build_requires
        setup_args['install_requires'] = build_requires

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
