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

FFTW MPI
--------
.. autofunction:: pyfftw.create_mpi_plan(input_shape, input_chunk=None, input_dtype=None, output_chunk=None, output_dtype=None, ptrdiff_t howmany=1, block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK', flags=tuple(), direction=None, unsigned int threads=1, Comm comm=None)

.. autofunction:: pyfftw.local_size(input_shape, ptrdiff_t howmany=1, block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK', flags=tuple(), direction='FFTW_FORWARD', Comm comm=None)

.. autoclass:: pyfftw.FFTW_MPI(input_shape, input_chunk, output_chunk, block0='DEFAULT_BLOCK', block1='DEFAULT_BLOCK', direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), unsigned int threads=1, planning_timelimit=None, n_transforms=1, comm=None, *args, **kwargs)

   .. autoattribute:: pyfftw.FFTW_MPI.N

   .. autoattribute:: pyfftw.FFTW_MPI.simd_aligned

   .. autoattribute:: pyfftw.FFTW_MPI.input_alignment

   .. autoattribute:: pyfftw.FFTW_MPI.output_alignment

   .. autoattribute:: pyfftw.FFTW_MPI.flags

   .. autoattribute:: pyfftw.FFTW_MPI.input_chunk

   .. autoattribute:: pyfftw.FFTW_MPI.output_chunk

   .. autoattribute:: pyfftw.FFTW_MPI.input_array

   .. autoattribute:: pyfftw.FFTW_MPI.output_array

   .. autoattribute:: pyfftw.FFTW_MPI.input_shape

   .. autoattribute:: pyfftw.FFTW_MPI.output_shape

   .. autoattribute:: pyfftw.FFTW_MPI.input_dtype

   .. autoattribute:: pyfftw.FFTW_MPI.output_dtype

   .. autoattribute:: pyfftw.FFTW_MPI.direction

   .. autoattribute:: pyfftw.FFTW_MPI.local_n_elements

   .. autoattribute:: pyfftw.FFTW_MPI.local_n0

   .. autoattribute:: pyfftw.FFTW_MPI.local_0_start

   .. autoattribute:: pyfftw.FFTW_MPI.local_n1

   .. autoattribute:: pyfftw.FFTW_MPI.local_1_start

   .. autoattribute:: pyfftw.FFTW_MPI.input_slice

   .. autoattribute:: pyfftw.FFTW_MPI.output_slice

   .. autoattribute:: pyfftw.FFTW_MPI.local_input_shape

   .. autoattribute:: pyfftw.FFTW_MPI.local_output_shape

   .. autoattribute:: pyfftw.FFTW_MPI.has_input

   .. autoattribute:: pyfftw.FFTW_MPI.has_output

   .. autoattribute:: pyfftw.FFTW_MPI.threads

   .. automethod:: pyfftw.FFTW_MPI.__call__

   .. automethod:: pyfftw.FFTW_MPI.update_arrays

   .. automethod:: pyfftw.FFTW_MPI.execute

   .. automethod:: pyfftw.FFTW_MPI.get_input_array(transform=0)

   .. automethod:: pyfftw.FFTW_MPI.get_output_array(transform=0)

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

   This can be used as ``n`` in the arguments for :func:`n_byte_align` and
   :func:`n_byte_align_empty` to create optimally aligned arrays for
   the running platform.

.. autofunction:: pyfftw.n_byte_align

.. autofunction:: pyfftw.n_byte_align_empty

.. autofunction:: pyfftw.is_n_byte_aligned
