Overview and A Short Tutorial on MPI
====================================

Before we begin, we assume that you are already familiar with the
serial/threaded `FFTW <http://www.fftw.org/>`_ and you require FFTW's
MPI support for your project. The basic concept of FFTW remains: it is
designed to work by planning *in advance* the fastest way to perform a
particular transform.  It does this by trying lots of different
techniques and measuring the fastest way, so called *planning*. Once
the plan is created, it can be executed as often as desired.

Using MPI typically requires you to (re)write your program from the
ground up keeping in mind how the individual MPI ranks (processes)
communicate with each other. This wrapper is designed to render the
interaction with FFTW as easy, error-avoiding, and pythonic as
possible. But it cannot and must not obscure the fact that MPI is used
at the basis.

The major differences to the non-MPI interface include:

#. Only transforms of entires arrays are supported, transforms over
   *individual* axes are not. Hence the MPI wrapper can *not* work as a
   dropin replacement for numpy's or scipy's FFT.
#. Each MPI rank only has a part of the data. Depending on the size
   and number of MPI ranks, it may happen that some process don't even
   have any input or output. The user is completely responsible for
   communicating the data that are input to the transform as only the
   user knows where and on which MPI rank a particular element of the
   array resides. FFTW then takes care of all the communication during
   the transform.
#. MPI communication is the most expensive part of the execution, so
   any optimization should first target to reduce the amount of
   (collective) MPI calls.
#. 1D transforms are supported only for the complex-to-complex
   case. If input or output are real, one has to plan for a
   complex-to-complex transform, and set the relevant imaginary parts
   to zero.

As in the serial interface, the user needs to specify in advance
exactly what transform is needed, including things like the data type,
and the *global* input array shape. To avoid MPI deadlocks, it is
crucial to note that some operations such as executing a plan are
collective calls. If some MPI ranks do not make that call, all other
ranks will wait indefinitely.

.. _create_mpi_plan_tutorial:

Quick and easy: using :func:`pyfftw.create_mpi_plan`
----------------------------------------------------

The easiest way to create an FFTW MPI plan is to use the factory
function :func:`pyfftw.create_mpi_plan`. The only mandatory argument
is the *global* shape of the input array, that is the shape of the
array that one would transform with the serial interface on a single
machine. This shape is identical on every MPI rank participating in
the transform. Besides that, only the data type of the input and
possibly the output are required. :func:`pyfftw.create_mpi_plan` takes
care of allocating sufficient (aligned!) memory, then creates and
returns the FFTW MPI plan.

Let's do a simple real-to-complex transform. We transform an array of
ones to a delta distribution. The minimal example is

 .. testcode::
    from mpi4py import MPI
    from pyfftw import create_mpi_plan

    p = create_mpi_plan(input_shape=(5, 6),
                        input_dtype='float64',
                        output_dtype='complex128')
    # fill local input data with constant
    if p.has_input:
        p.input_array[:] = 1.0

    # execute the plan
    p()

    # result is delta function
    if p.has_output:
        print 'output on rank', MPI.COMM_WORLD.Get_rank(), p.output_array

Save the script as ``simple_r2c.py``. Running with 2 MPI processes as
``mpirun -n 2 python simple_r2c.py`` may produce this output::

  output on rank 1 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
   [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
  output on rank 0 [[ 30.+0.j   0.+0.j   0.+0.j   0.+0.j]
   [  0.+0.j   0.+0.j   0.+0.j   0.+0.j]
   [  0.+0.j   0.+0.j   0.+0.j   0.+0.j]]

The Fourier transform of a distribution of ones is a delta
distribution, hence the ``[0,0]`` element of the output should equal
the number of input elements ``5 * 6 = 30``. Note how we assigned the
constant value 1.0 on every process that has input. By default, FFTW
distributes the work by splitting the array into chunks along the
*first* dimension. Since 5 is not divisible by 2, the work load is not
balanced, and the first process gets the first 3, and the second gets
the last 2 of the 5 input rows. Each plan knows which part of the
global array it is responsible for. In the example, you can obtain
this info via ``p.input_slice`` and ``p.output_slice``.

If you run the example again with more than 5 processes, all but 5
processes have no input. On the idle processes, ``p.input_slice is
None``, ``p.has_input is False``, and calling ``p.input_array`` yields
an ``AttributeError``. Internally, FFTW operates on a chunk of memory
that may be larger than expected from ``input_array`` due to
intermediate steps and other implementation details. Even the idle
processes need to reserve a small amount of memory.

Note that the output arrays have only 4 columns. Due to the Hermitian
symmetry of the output given the real input, 4 columns represent the
minimum nonredundant part of the output, so only that is stored. These
subtleties are taken care of by the wrapper, so the user needs to only
supply the ``input_shape``. In this example, it is the shape of the
real array.

Improving the speed
-------------------

This is based on http://fftw.org/fftw3_doc/FFTW-MPI-Performance-Tips.html.

#. Ensure that FFTW can balance the work by choosing dimensions that
   are divisible by the number of processes.
#. Remove one MPI communication step by using ``create_mpi_plan(..,
   flags=('FFTW_MPI_TRANSPOSED_OUT',))``. For an input shape ``(X, Y,
   Z)``, the output is not sliced in the ``X`` direction but in the ``Y``
   direction.
#. Perform the transform in place; i.e., overwrite the input with the
   output to save memory. In most cases the data layout of your
   simulation, computation etc. is not the one that FFTW expects. Your
   data is not sliced along the first dimension, so you have to create
   a *new* array on each MPI rank just to hold the input for FFTW and
   shuffle the data yourself. Consider this extra memory just scratch,
   and all you want is the output of the Fourier transform. In that
   case, you will not need to make a copy of the input, and probably
   need the output array only until you have communicated the results
   among the MPI ranks.
   ``create_mpi_plan(..., output_chunk='INPUT')``
#. Use multiple threads on a shared-memory machine with ``create_mpi_plan(..., threads=4)``.




An example with explicit communication
--------------------------------------

From Margarita
