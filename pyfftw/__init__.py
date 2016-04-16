#!/usr/bin/env python
# The pyfftw namespace

''' The core of ``pyfftw`` consists of the :class:`FFTW` class,
:ref:`wisdom functions <wisdom_functions>` and a couple of
:ref:`utility functions <utility_functions>` for dealing with aligned
arrays.

This module represents the full interface to the underlying `FFTW
library <http://www.fftw.org/>`_. However, users may find it easier to
use the helper routines provided in :mod:`pyfftw.builders`.
'''

from .version import version as __version__

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
)

from . import builders
from . import interfaces

# clean up the namespace
del builders.builders
