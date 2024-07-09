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

   .. autoattribute:: pyfftw.FFTW.ortho

   .. autoattribute:: pyfftw.FFTW.normalise_idft

   .. automethod:: pyfftw.FFTW.__call__

   .. automethod:: pyfftw.FFTW.update_arrays

   .. automethod:: pyfftw.FFTW.execute

   .. automethod:: pyfftw.FFTW.get_input_array

   .. automethod:: pyfftw.FFTW.get_output_array

   .. method:: execute_nogil

      Same as :func:`pyfftw.FFTW.execute`, but should be called from Cython directly within a
      nogil block.

      **For Cython use only.**

      Warning: This method is **NOT** thread-safe. Concurrent calls
      to :func:`pyfftw.FFTW.execute_nogil` will lead to race conditions and ultimately
      wrong FFT results.

   .. method:: get_fftw_exe

      Returns a C struct :data:`pyfftw.fftw_exe` that is associated with the FFTW
      instance.

      **For Cython use only.**

      This is really only useful if you want to 
      bundle a few :data:`pyfftw.fftw_exe` in a C array, and then call them all from
      within a nogil block.

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

.. autofunction:: pyfftw.next_fast_len

.. data:: pyfftw.fftw_exe

   C struct for executing configured plans in a nogil block.

   **For Cython use only.**

.. function:: pyfftw.execute_in_nogil(fftw_exe* exe_ptr)

   Runs the FFT as defined by the pointed :data:`pyfftw.fftw_exe`.

   **For Cython use only.**

   Warning: This method is **NOT** thread-safe. Concurrent calls
   to :func:`pyfftw.execute_in_nogil` with an aliased :data:`pyfftw.fftw_exe` will lead 
   to wrong FFT results.

.. _configuration_variables:

FFTW Configuration
------------------

.. data:: pyfftw.config.NUM_THREADS

   This variable controls the default number of threads used by the functions
   in :mod:`pyfftw.builders` and :mod:`pyfftw.interfaces`.

   The default value is read from the environment variable
   ``PYFFTW_NUM_THREADS``. If this variable is undefined and the user's
   underlying FFTW library was built using OpenMP threading, the number of
   threads will be read from the environment variable ``OMP_NUM_THREADS``
   instead. If neither environment variable is defined, the default value is 1.

   If the specified value is ``<= 0``, the library will use
   :func:`multiprocessing.cpu_count` to determine the number of threads.

   The user can modify the value at run time by assigning to this variable.

.. data:: pyfftw.config.PLANNER_EFFORT

   This variable controls the default planning effort used by the functions
   in :mod:`pyfftw.builders` and :mod:`pyfftw.interfaces`.

   The default value of is determined by reading from the environment variable
   ``PYFFTW_PLANNER_EFFORT``. If this environment variable is undefined, it
   defaults to ``'FFTW_ESTIMATE'``.

   The user can modify the value at run time by assigning to this variable.
