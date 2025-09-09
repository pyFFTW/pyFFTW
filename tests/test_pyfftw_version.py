import re

import pyfftw


def test_fftw_version_pattern():
    fv = pyfftw.fftw_version
    if fv != "":
        match = re.search(r"(\d+).(\d+).(\d+)", fv)
        assert match is not None
    else:
        # cannot get version from fftw
        assert True


def test_fftw_version_tuple():
    fvt = pyfftw.fftw_version_tuple
    if fvt:
        # major, minor, release
        assert fvt >= (3, 0, 0)
    else:
        # cannot get version from fftw
        assert True


def test_fftw_cc():
    # compiler flags cannot be empty (at least there is a C compiler)
    assert len(pyfftw.fftw_cc) > 0
