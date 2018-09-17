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
        _all_types_human_readable,
        _all_types_np,
        _threading_type
)

from . import config
from . import builders
from . import interfaces


# clean up the namespace
del builders.builders

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
