``pyfftw`` - The core
=====================

.. automodule:: pyfftw

FFTW Class
----------

.. autoclass:: pyfftw.FFTW(input_array, output_array, axes=(-1,), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), threads=1, planning_time_limit=None)

   .. autoattribute:: pyfftw.FFTW.N

   .. autoattribute:: pyfftw.FFTW.aligned

   .. automethod:: pyfftw.FFTW.__call__(input_array=None, output_array=None, normalise_idft=True)

   .. automethod:: pyfftw.FFTW.update_arrays(new_input_array, new_output_array)

   .. automethod:: pyfftw.FFTW.execute()

   .. automethod:: pyfftw.FFTW.get_input_array()

   .. automethod:: pyfftw.FFTW.get_output_array()

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

.. autofunction:: pyfftw.n_byte_align

.. autofunction:: pyfftw.n_byte_align_empty

