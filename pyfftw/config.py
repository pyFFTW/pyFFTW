"""Global configuration variables for PyFFTW.

The approach taken here was adapated from Numba's config.py.
"""
from __future__ import print_function, division, absolute_import

import os
import multiprocessing
import warnings

from .pyfftw import _threading_type


class _EnvReloader(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.old_environ = {}
        self.update(force=True)

    def update(self, force=False):
        new_environ = {}

        # read local env var OMP_NUM_THREADS and any starting with PYFFTW_
        for name, value in os.environ.items():
            if name.startswith('PYFFTW_') or name == 'OMP_NUM_THREADS':
                new_environ[name] = value
        # We update the config variables if at least one PYFFTW environment
        # variable was modified.  This lets the user modify values
        # directly in the config module without having them when
        # reload_config() is called by the compiler.
        if force or self.old_environ != new_environ:
            self.process_environ(new_environ)
            # Store a copy
            self.old_environ = dict(new_environ)

    def process_environ(self, environ):
        def _readenv(name, ctor, default):
            value = environ.get(name)
            if value is None:
                return default() if callable(default) else default
            try:
                return ctor(value)
            except Exception:
                warnings.warn("environ %s defined but failed to parse '%s'" %
                              (name, value), RuntimeWarning)
                return default

        def optional_str(x):
            return str(x) if x is not None else None

        if _threading_type is None:
            NUM_THREADS = 1
        else:
            if (_threading_type == "OMP" and
                    "PYFFTW_NUM_THREADS" not in environ):
                # fallback to OMP_NUM_THREADS if PYFFTW_NUM_THREADS undefined
                NUM_THREADS = _readenv("OMP_NUM_THREADS", int, 1)
            else:
                NUM_THREADS = _readenv("PYFFTW_NUM_THREADS", int, 1)
            # if user requested <= 0 threads, use the maximum available
            if NUM_THREADS <= 0:
                NUM_THREADS = multiprocessing.cpu_count()

        PLANNER_EFFORT = _readenv(
            "PYFFTW_PLANNER_EFFORT", str, "FFTW_ESTIMATE")

        # Inject the configuration values into the module globals
        for name, value in locals().copy().items():
            if name.isupper():
                globals()[name] = value

_env_reloader = _EnvReloader()


def _reload_config():
    """
    Reload the configuration from environment variables, if necessary.
    """
    _env_reloader.update()
