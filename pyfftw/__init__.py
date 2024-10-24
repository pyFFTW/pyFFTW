#!/usr/bin/env python
# The pyfftw namespace

''' The core of ``pyfftw`` consists of the :class:`FFTW` class,
:ref:`wisdom functions <wisdom_functions>` and a couple of
:ref:`utility functions <utility_functions>` for dealing with aligned
arrays.

This module represents the full interface to the underlying `FFTW
library <http://www.fftw.org/>`_. However, users may find it easier to
use the helper routines provided in :mod:`pyfftw.builders`. Default values
used by the helper routines can be controlled as via
:ref:`configuration variables <configuration_variables>`.
'''

import os
import re

from .pyfftw import (
        FFTW,
        export_wisdom,
        import_wisdom,
        forget_wisdom,
        simd_alignment,
        n_byte_align_empty,
        n_byte_align,
        is_n_byte_aligned,
        byte_align,
        is_byte_aligned,
        empty_aligned,
        ones_aligned,
        zeros_aligned,
        next_fast_len,
        _supported_types,
        _supported_nptypes_complex,
        _supported_nptypes_real,
        _fftw_version_dict,
        _fftw_cc_dict,
        _all_types_human_readable,
        _all_types_np,
        _threading_type,
)

from . import config
from . import builders
from . import interfaces


# clean up the namespace
del builders.builders

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# retrieve fftw library version (only numbers) from the 'double' API
fv_match = re.search(r'(\d+).(\d+).(\d+)', _fftw_version_dict.get('64', ''))
if fv_match is not None:
    fftw_version = fv_match.group()
    fftw_version_tuple = tuple(int(n) for n in fv_match.groups())
else:
    fftw_version = ''
    fftw_version_tuple = ()
del fv_match

# retrieve compiler flags from the 'double' API
fftw_cc = _fftw_cc_dict.get('64', '')
