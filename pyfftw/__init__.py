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

from ._version import version

from .pyfftw import (
        FFTW,
        export_wisdom,
        import_wisdom,
        forget_wisdom,
        simd_alignment,
        n_byte_align_empty,
        n_byte_align,
        is_n_byte_aligned,)

try:
    from .pyfftw import (
        FFTW_MPI,
        broadcast_wisdom,
        create_mpi_plan,
        gather_wisdom,
        local_size,
        supported_mpi_types)
except ImportError:
    pass

from . import builders
from . import interfaces

# clean up the namespace
del builders.builders
