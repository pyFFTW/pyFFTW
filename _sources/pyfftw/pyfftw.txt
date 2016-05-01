``pyfftw`` - The core
=====================

.. automodule:: pyfftw

FFTW Class
----------

.. autoclass:: pyfftw.FFTW(input_array, output_array, axes=(-1,), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), threads=1, planning_timelimit=None)

   .. autoattribute:: pyfftw.FFTW.N

   .. autoattribute:: pyfftw.FFTW.simd_aligned

   .. autoattribute:: pyfftw.FFTW.input_alignment

   .. autoattribute:: pyfftw.FFTW.output_alignment

   .. autoattribute:: pyfftw.FFTW.flags

   .. autoattribute:: pyfftw.FFTW.input_array

   .. autoattribute:: pyfftw.FFTW.output_array

   .. autoattribute:: pyfftw.FFTW.input_shape

   .. autoattribute:: pyfftw.FFTW.output_shape

   .. autoattribute:: pyfftw.FFTW.input_strides

   .. autoattribute:: pyfftw.FFTW.output_strides

   .. autoattribute:: pyfftw.FFTW.input_dtype

   .. autoattribute:: pyfftw.FFTW.output_dtype

   .. autoattribute:: pyfftw.FFTW.direction

   .. autoattribute:: pyfftw.FFTW.axes

   .. automethod:: pyfftw.FFTW.__call__

   .. automethod:: pyfftw.FFTW.update_arrays

   .. automethod:: pyfftw.FFTW.execute

   .. automethod:: pyfftw.FFTW.get_input_array

   .. automethod:: pyfftw.FFTW.get_output_array

.. _wisdom_functions:

Wisdom Functions
----------------

Functions for dealing with FFTW's ability to export and restore plans,
referred to as *wisdom*. For further information, refer to the `FFTW
wisdom documentation <http://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html#Words-of-Wisdom_002dSaving-Plans>`_.

.. autofunction:: pyfftw.export_wisdom

.. autofunction:: pyfftw.import_wisdom

.. autofunction:: pyfftw.forget_wisdom

.. _utility_functions:

Utility Functions
-----------------

.. data:: pyfftw.simd_alignment

   An integer giving the optimum SIMD alignment in bytes, found by
   inspecting the CPU (e.g. if AVX is supported, its value will be 32).

   This can be used as ``n`` in the arguments for :func:`byte_align`,
   :func:`empty_aligned`, :func:`zeros_aligned`, and :func:`ones_aligned` to
   create optimally aligned arrays for the running platform.

.. autofunction:: pyfftw.byte_align

.. autofunction:: pyfftw.empty_aligned

.. autofunction:: pyfftw.zeros_aligned

.. autofunction:: pyfftw.ones_aligned

.. autofunction:: pyfftw.is_byte_aligned

.. autofunction:: pyfftw.n_byte_align

.. autofunction:: pyfftw.n_byte_align_empty

.. autofunction:: pyfftw.is_n_byte_aligned
