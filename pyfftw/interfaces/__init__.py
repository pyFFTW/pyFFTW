#!/usr/bin/env python

'''The :mod:`pyfftw.interfaces` package provides interfaces to :mod:`pyfftw`
that implement the API of other, more commonly used FFT libraries; specifically
:mod:`numpy.fft`, :mod:`scipy.fft` and :mod:`scipy.fftpack`. The intention is
to satisfy two clear use cases:

1. Simple, clean and well established interfaces to using :mod:`pyfftw`,
   removing the requirement for users to know or understand about creating and
   using :class:`pyfftw.FFTW` objects, whilst still benefiting from most of the
   speed benefits of FFTW.
2. A library that can be dropped into code that is already written to
   use a supported FFT library, with no significant change to the existing
   code. The power of python allows this to be done at runtime to a third
   party library, without changing any of that library's code.

The :mod:`pyfftw.interfaces` implementation is designed to sacrifice a small
amount of the flexibility compared to accessing the :class:`pyfftw.FFTW`
object directly, but implements a reasonable set of defaults and optional
tweaks that should satisfy most situations.

The precision of the transform that is used is selected from the array that
is passed in, defaulting to double precision if any type conversion is
required.

This module works by generating a :class:`pyfftw.FFTW` object behind the
scenes using the :mod:`pyfftw.builders` interface, which is then executed.
There is therefore a potentially substantial overhead when a new plan needs
to be created. This is down to FFTW's internal planner process.
After a specific transform has been planned once, subsequent calls in which
the input array is equivalent will be much faster, though still not without
potentially significant overhead. *This* overhead can be largely alleviated by
enabling the :mod:`pyfftw.interfaces.cache` functionality. However, even when
the cache is used, very small transforms may suffer a significant relative
slow-down not present when accessing :mod:`pyfftw.FFTW` directly (because the
transform time can be negligibly small compared to the fixed
:mod:`pyfftw.interfaces` overhead).

In addition, potentially extra copies of the input array might be made.

If speed or memory conservation is of absolutely paramount importance, the
suggestion is to use :mod:`pyfftw.FFTW` (which provides better control over
copies and so on), either directly or through :mod:`pyfftw.builders`. As
always, experimentation is the best guide to optimisation.

In practice, this means something like the following (taking
:mod:`~pyfftw.interfaces.numpy_fft` as an example):

.. doctest::

    >>> import pyfftw, numpy
    >>> a = pyfftw.empty_aligned((128, 64), dtype='complex64', n=16)
    >>> a[:] = numpy.random.randn(*a.shape) + 1j*numpy.random.randn(*a.shape)
    >>> fft_a = pyfftw.interfaces.numpy_fft.fft2(a) # Will need to plan

.. doctest::

    >>> b = pyfftw.empty_aligned((128, 64), dtype='complex64', n=16)
    >>> b[:] = a
    >>> fft_b = pyfftw.interfaces.numpy_fft.fft2(b) # Already planned, so faster

.. doctest::

    >>> c = pyfftw.empty_aligned(132, dtype='complex128', n=16)
    >>> fft_c = pyfftw.interfaces.numpy_fft.fft(c) # Needs a new plan
    >>> c[:] = numpy.random.randn(*c.shape) + 1j*numpy.random.randn(*c.shape)

.. doctest::

    >>> pyfftw.interfaces.cache.enable()
    >>> fft_a = pyfftw.interfaces.numpy_fft.fft2(a) # still planned
    >>> fft_b = pyfftw.interfaces.numpy_fft.fft2(b) # much faster, from the cache

The usual wisdom import and export functions work well for the case where
the initial plan might be prohibitively expensive. Just use
:func:`pyfftw.export_wisdom` and :func:`pyfftw.import_wisdom` as needed after
having performed the transform once.

Implemented Functions
---------------------

The implemented functions are listed below. :mod:`numpy.fft` is implemented by
:mod:`pyfftw.interfaces.numpy_fft`, :mod:`scipy.fftpack` by
:mod:`pyfftw.interfaces.scipy_fftpack` and :mod:`scipy.fft` by
:mod:`pyfftw.interfaces.scipy_fft`. All the implemented functions are extended
by the use of additional arguments, which are
:ref:`documented below<interfaces_additional_args>`.

Not all the functions provided by :mod:`numpy.fft`, :mod:`scipy.fft` and
:mod:`scipy.fftpack` are implemented by :mod:`pyfftw.interfaces`. In the case
where a function is not implemented, the function is imported into the
namespace from the corresponding library. This means that all the documented
functionality of the library *is* provided through :mod:`pyfftw.interfaces`.

One known caveat is that repeated axes are handled differently. Axes that are
repeated in the ``axes`` argument are considered only once and without error;
as compared to :mod:`numpy.fft` in which repeated axes results in the DFT being
taken along that axes as many times as the axis occurs, or to :mod:`scipy`
where an error is raised.

:mod:`~pyfftw.interfaces.numpy_fft`
"""""""""""""""""""""""""""""""""""

* :func:`pyfftw.interfaces.numpy_fft.fft`
* :func:`pyfftw.interfaces.numpy_fft.ifft`
* :func:`pyfftw.interfaces.numpy_fft.fft2`
* :func:`pyfftw.interfaces.numpy_fft.ifft2`
* :func:`pyfftw.interfaces.numpy_fft.fftn`
* :func:`pyfftw.interfaces.numpy_fft.ifftn`
* :func:`pyfftw.interfaces.numpy_fft.rfft`
* :func:`pyfftw.interfaces.numpy_fft.irfft`
* :func:`pyfftw.interfaces.numpy_fft.rfft2`
* :func:`pyfftw.interfaces.numpy_fft.irfft2`
* :func:`pyfftw.interfaces.numpy_fft.rfftn`
* :func:`pyfftw.interfaces.numpy_fft.irfftn`
* :func:`pyfftw.interfaces.numpy_fft.hfft`
* :func:`pyfftw.interfaces.numpy_fft.ihfft`

:mod:`~pyfftw.interfaces.scipy_fft`
"""""""""""""""""""""""""""""""""""""""

* :func:`pyfftw.interfaces.scipy_fft.fft`
* :func:`pyfftw.interfaces.scipy_fft.ifft`
* :func:`pyfftw.interfaces.scipy_fft.fft2`
* :func:`pyfftw.interfaces.scipy_fft.ifft2`
* :func:`pyfftw.interfaces.scipy_fft.fftn`
* :func:`pyfftw.interfaces.scipy_fft.ifftn`
* :func:`pyfftw.interfaces.scipy_fft.rfft`
* :func:`pyfftw.interfaces.scipy_fft.irfft`
* :func:`pyfftw.interfaces.scipy_fft.rfft2`
* :func:`pyfftw.interfaces.scipy_fft.irfft2`
* :func:`pyfftw.interfaces.scipy_fft.rfftn`
* :func:`pyfftw.interfaces.scipy_fft.irfftn`
* :func:`pyfftw.interfaces.scipy_fft.hfft`
* :func:`pyfftw.interfaces.scipy_fft.ihfft`
* :func:`pyfftw.interfaces.scipy_fft.next_fast_len`

:mod:`~pyfftw.interfaces.scipy_fftpack`
"""""""""""""""""""""""""""""""""""""""

* :func:`pyfftw.interfaces.scipy_fftpack.fft`
* :func:`pyfftw.interfaces.scipy_fftpack.ifft`
* :func:`pyfftw.interfaces.scipy_fftpack.fft2`
* :func:`pyfftw.interfaces.scipy_fftpack.ifft2`
* :func:`pyfftw.interfaces.scipy_fftpack.fftn`
* :func:`pyfftw.interfaces.scipy_fftpack.ifftn`
* :func:`pyfftw.interfaces.scipy_fftpack.rfft`
* :func:`pyfftw.interfaces.scipy_fftpack.irfft`
* :func:`pyfftw.interfaces.scipy_fftpack.next_fast_len`

:mod:`~pyfftw.interfaces.dask_fft`
"""""""""""""""""""""""""""""""""""

* :func:`pyfftw.interfaces.dask_fft.fft`
* :func:`pyfftw.interfaces.dask_fft.ifft`
* :func:`pyfftw.interfaces.dask_fft.fft2`
* :func:`pyfftw.interfaces.dask_fft.ifft2`
* :func:`pyfftw.interfaces.dask_fft.fftn`
* :func:`pyfftw.interfaces.dask_fft.ifftn`
* :func:`pyfftw.interfaces.dask_fft.rfft`
* :func:`pyfftw.interfaces.dask_fft.irfft`
* :func:`pyfftw.interfaces.dask_fft.rfft2`
* :func:`pyfftw.interfaces.dask_fft.irfft2`
* :func:`pyfftw.interfaces.dask_fft.rfftn`
* :func:`pyfftw.interfaces.dask_fft.irfftn`
* :func:`pyfftw.interfaces.dask_fft.hfft`
* :func:`pyfftw.interfaces.dask_fft.ihfft`


.. _interfaces_additional_args:

Additional Arguments
--------------------

In addition to the equivalent arguments in :mod:`numpy.fft`, :mod:`scipy.fft`
and :mod:`scipy.fftpack`, all these functions also add several additional
arguments for finer control over the FFT. These additional arguments are
largely a subset of the keyword arguments in :mod:`pyfftw.builders` with a few
exceptions and with different defaults.

* ``overwrite_input``: Whether or not the input array can be
  overwritten during the transform. This sometimes results in a faster
  algorithm being made available. It causes the ``'FFTW_DESTROY_INPUT'``
  flag to be passed to the intermediate :class:`pyfftw.FFTW` object.
  Unlike with :mod:`pyfftw.builders`, this argument is included with
  *every* function in this package.

  In :mod:`~pyfftw.interfaces.scipy_fftpack` and
  :mod:`~pyfftw.interfaces.scipy_fft`, this argument is replaced by
  ``overwrite_x``, to which it is equivalent (albeit at the same position).

  The default is ``False`` to be consistent with :mod:`numpy.fft`.

* ``planner_effort``: A string dictating how much effort is spent
  in planning the FFTW routines. This is passed to the creation
  of the intermediate :class:`pyfftw.FFTW` object as an entry
  in the flags list. They correspond to flags passed to the
  :class:`pyfftw.FFTW` object.

  The valid strings, in order of their increasing impact on the time
  to compute  are:
  ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default), ``'FFTW_PATIENT'``
  and ``'FFTW_EXHAUSTIVE'``.

  The `Wisdom
  <http://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>`_
  that FFTW has accumulated or has loaded (through
  :func:`pyfftw.import_wisdom`) is used during the creation of
  :class:`pyfftw.FFTW` objects.

  Note that the first time planning stage can take a substantial amount
  of time. For this reason, the default is to use ``'FFTW_ESTIMATE'``, which
  potentially results in a slightly suboptimal plan being used, but with
  a substantially quicker first-time planner step.

* ``threads``: The number of threads used to perform the FFT.

  In :mod:`~pyfftw.interfaces.scipy_fft`, this argument is replaced by
  ``workers``, which serves the same purpose, but is also compatible with the
  :func:`scipy.fft.set_workers` context manager.

  The default is ``1``.

* ``auto_align_input``: Correctly byte align the input array for optimal
  usage of vector instructions. This can lead to a substantial speedup.

  This argument being ``True`` makes sure that the input array
  is correctly aligned. It is possible to correctly byte align the array
  prior to calling this function (using, for example,
  :func:`pyfftw.byte_align`). If and only if a realignment is
  necessary is a new array created.

  It's worth noting that just being aligned may not be sufficient to
  create the fastest possible transform. For example, if the array is not
  contiguous (i.e. certain axes have gaps in memory between slices), it may
  be faster to plan a transform for a contiguous array, and then rely on
  the array being copied in before the transform (which
  :class:`pyfftw.FFTW` will handle for you). The ``auto_contiguous``
  argument controls whether this function also takes care of making sure
  the array is contiguous or not.

  The default is ``True``.

* ``auto_contiguous``: Make sure the input array is contiguous in
  memory before performing the transform on it. If the array is not
  contiguous, it is copied into an interim array. This is because it
  is often faster to copy the data before the transform and then transform
  a contiguous array than it is to try to take the transform of a
  non-contiguous array. This is particularly true in conjunction with
  the ``auto_align_input`` argument which is used to make sure that the
  transform is taken of an aligned array.

  The default is ``True``.

'''

from . import (
        numpy_fft,
        cache,)

try:
    import scipy.fftpack
except ImportError:
    pass
else:
    from numpy.lib import NumpyVersion

    has_scipy_fft = NumpyVersion(scipy.__version__) >= NumpyVersion('1.4.0')
    del NumpyVersion
    del scipy

    from . import scipy_fftpack
    if has_scipy_fft:
        from . import scipy_fft


fft_wrap = None
try:
    from dask.array.fft import fft_wrap
except ImportError:
    pass

if fft_wrap:
    from . import dask_fft
del fft_wrap
