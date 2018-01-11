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

import os

# All planners and interfaces default to single threaded unless otherwise
# specified via PYFFTW_NUM_THREADS.
default_num_threads = int(os.environ.get('PYFFTW_NUM_THREADS', 1))
if default_num_threads <= 0:
    import multiprocessing
    default_num_threads = multiprocessing.cpu_count()

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
)

from . import builders
from . import interfaces

# clean up the namespace
del builders.builders

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
