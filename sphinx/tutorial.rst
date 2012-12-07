Overview and A Short Tutorial
=============================

Before we begin, we assume that you are already familiar with the 
`discrete Fourier transform 
<http://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_, 
and why you want a faster library to perform your FFTs for you.

`FFTW <http://www.fftw.org/>`_ is a very fast FFT C library. The way it
is designed to work is by planning *in advance* the fastest way to
perform a particular transform.  It does this by trying lots of
different techniques and measuring the fastest way, so called 
*planning*.

One consequence of this is that the user needs to specify in advance
exactly what transform is needed, including things like the data type,
the array shapes and strides and the precision. This is quite
different to how one uses, for example, the :mod:`numpy.fft` module.

The purpose of this library is to provide a simple and pythonic way
to, firstly, handle the necessary logic of working out what kind of
FFTW plan should be created for a given data type, and secondly, 
provide a simple way of generating those plans which is akin, from
the users perspective, to the way in which the widely used 
:mod:`numpy.fft` module is used.

This tutorial is split into two parts. Firstly an 
:ref:`overview <FFTW_tutorial>` is given of :class:`pyfftw.FFTW`, 
the core of the library. Secondly, the :mod:`pyfftw.builders` helper
functions are :ref:`introduced <builders_tutorial>`.

.. _FFTW_tutorial:

The workhorse :class:`pyfftw.FFTW` class
----------------------------------------

The core of this library is provided through the :class:`pyfftw.FFTW`
class. FFTW is fully encapsulated within this class.

The following gives an overview of the :class:`pyfftw.FFTW` class, but
the easiest way to of dealing with it is through the
:mod:`pyfftw.builders` helper functions, also
:ref:`discussed in this tutorial <builders_tutorial>`.

For users that already have some experience of FFTW, there is no
interface distinction between any of the supported data types, shapes
or transforms, and operating on arbitrarily strided arrays (which are
common when using :mod:`numpy`) is fully supported with no copies 
necessary.

In its simplest form, a :class:`pyfftw.FFTW` object is created with
a pair of complementary :mod:`numpy` arrays: an input array and an
output array.  They are complementary insomuch as the data types and the
array sizes together define exactly what transform should be performed.
We refer to a valid transform as a :ref:`scheme <fftw_schemes>`.

Internally, three precisions of FFT are supported. These correspond
to 32-bit floating point, 64-bit floating point and 128-bit floating
point, which correspond to :mod:`numpy`'s ``float32``, ``float64``
and ``longdouble`` dtypes respectively (and the corresponding
complex types). The precision is decided by the relevant scheme, 
which is specified by the dtype of the input array.

Various schemes are supported by :class:`pyfftw.FFTW`. The scheme
that is used depends on the data types of the input array and output
arrays, the shape of the arrays and the direction flag. For a full
discussion of the schemes available, see the API documentation for
:class:`pyfftw.FFTW`.

**One-Dimensional Transforms**

We will first consider creating a simple one-dimensional transform of
a one-dimensional complex array:

.. testcode::

   import pyfftw
   
   a = pyfftw.n_byte_align_empty(128, 16, 'complex128')
   b = pyfftw.n_byte_align_empty(128, 16, 'complex128')

   fft_object = pyfftw.FFTW(a, b)

In this case, we create 2 complex arrays, ``a`` and ``b`` each of
length 128. :func:`pyfftw.n_byte_align_empty` is a helper function 
that works like :func:`numpy.empty` but returns the array aligned to
a particular number of bytes in memory, in this case 16. Having byte
aligned arrays allows FFTW to performed vector operations, potentially
speeding up the FFT (a similar :func:`pyfftw.n_byte_align` exists to
align a pre-existing array as necessary).

Given these 2 arrays, the only transform that makes sense is a 
1D complex DFT. The direction in this case is the default, which is
forward, and so that is the transform that is *planned*. The 
returned ``fft_object`` represents such a transform.

In general, the creation of the :class:`pyfftw.FFTW` object clears the
contents of the arrays, so the arrays should be filled or updated 
after creation.

Similarly, to plan the inverse:

.. testcode::
   
   c = pyfftw.n_byte_align_empty(128, 16, 'complex128')
   ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')

In this case, the direction argument is given as ``'FFTW_BACKWARD'`` 
(to override the default of ``'FFTW_FORWARD'``).

The actual FFT is performed by calling the returned objects:

.. testcode::
   
   import numpy

   # Generate some data
   ar, ai = numpy.random.randn(2, 128)
   a[:] = ar + 1j*ai

   fft_a = fft_object()

Note that calling the object like this performs the FFT and returns
the result in an array. This is the *same* array as ``b``:

.. doctest::

   >>> fft_a is b
   True

This is particularly useful when using :mod:`pyfftw.builders` to 
generate the :class:`pyfftw.FFTW` objects.

Calling the FFT object followed by the inverse FFT object yields
an output that is numerically the same as the original ``a`` 
(within numerical accuracy).

.. doctest::
   
   >>> fft_a = fft_object()
   >>> ifft_b = ifft_object()
   >>> ifft_b is c
   True
   >>> numpy.allclose(a, c)
   True
   >>> a is c
   False

In this case, the normalisation of the DFT is performed automatically
by the inverse FFTW object (``ifft_object``). This can be disabled
by setting the ``normalise_idft=False`` argument.

It is possible to change the data on which a :class:`pyfftw.FFTW` 
operates. The :meth:`pyfftw.FFTW.__call__` accepts both an 
``input_array`` and and ``output_array`` argument to update the
arrays. The arrays should be compatible with the arrays with which
the :class:`pyfftw.FFTW` object was originally created. Please read the
API docs on :meth:`pyfftw.FFTW.__call__` to fully understand the
requirements for updating the array.

.. doctest::

   >>> d = pyfftw.n_byte_align_empty(4, 16, 'complex128')
   >>> e = pyfftw.n_byte_align_empty(4, 16, 'complex128')   
   >>> f = pyfftw.n_byte_align_empty(4, 16, 'complex128')
   >>> fft_object = pyfftw.FFTW(d, e)
   >>> fft_object.get_input_array() is d # get the input array from the object
   True
   >>> f[:] = [1, 2, 3, 4] # Add some data to f
   >>> fft_object(f)
   array([ 10.+0.j,  -2.+2.j,  -2.+0.j,  -2.-2.j])
   >>> fft_object.get_input_array() is d # No longer true!
   False
   >>> fft_object.get_input_array() is f # It has been updated with f :)
   True

If the new input array is of the wrong dtype or wrongly strided, 
:meth:`pyfftw.FFTW.__call__` method will copy the new array into the
internal array, if necessary changing it's dtype in the process.

It should be made clear that the :meth:`pyfftw.FFTW.__call__` method
is simply a helper routine around the other methods of the object.
Though it is expected that most of the time 
:meth:`pyfftw.FFTW.__call__` will be sufficient, all the FFTW 
functionality can be accessed through other methods at a slightly
lower level.

**Multi-Dimensional Transforms**

Arrays of more than one dimension are easily supported as well.
In this case, the ``axes`` argument specifies over which axes the
transform is to be taken.

.. testcode::

   import pyfftw
   
   a = pyfftw.n_byte_align_empty((128, 64), 16, 'complex128')
   b = pyfftw.n_byte_align_empty((128, 64), 16, 'complex128')

   # Plan an fft over the last axis
   fft_object_a = pyfftw.FFTW(a, b)

   # Over the first axis
   fft_object_b = pyfftw.FFTW(a, b, axes=(0,))

   # Over the both axes
   fft_object_c = pyfftw.FFTW(a, b, axes=(0,1))

For further information on all the supported transforms, including
real transforms, as well as full documentaion on all the 
instantiation arguments, see the :class:`pyfftw.FFTW` documentation.

**Wisdom**

When creating a :class:`pyfftw.FFTW` object, it is possible to instruct
FFTW how much effort it should put into finding the fastest possible 
method for computing the DFT. This is done by specifying a suitable
planner flag in ``flags`` argument to :class:`pyfftw.FFTW`. Some
of the planner flags can take a very long time to complete which can
be problematic.

When the a particular transform has been created, distinguished by
things like the data type, the shape, the stridings and the flags, 
FFTW keeps a record of the fastest way to compute such a transform in
future. This is referred to as 
`wisdom <http://www.fftw.org/fftw3_doc/Wisdom.html>`_. When 
the program is completed, the wisdom that has been accumulated is
forgotten.

It is possible to output the accumulated wisdom using the 
:ref:`wisdom output routines <wisdom_functions>`. 
:func:`pyfftw.export_wisdom` exports and returns the wisdom as a tuple
of strings that can be easily written to file. To load the wisdom back
in, use the :func:`pyfftw.import_wisdom` function which takes as its
argument that same tuple of strings that was returned from
:func:`pyfftw.export_wisdom`.

If for some reason you wish to forget the accumulated wisdom, call
:func:`pyfftw.forget_wisdom`.

.. _builders_tutorial:

The :mod:`pyfftw.builders` functions
------------------------------------

The easiest way to begin working with this package is by using the 
convenient :mod:`pyfftw.builders` package. These functions take care
of much of the difficulty in specifying the exact size and dtype
requirements to produce a valid scheme.

The :mod:`pyfftw.builders` functions are a series of helper functions
that provide an interface very much like that provided by 
:mod:`numpy.fft`, only instead of returning the result of the
transform, a :class:`pyfftw.FFTW` object (or in some cases a wrapper
around :class:`pyfftw.FFTW`) is returned.

.. testcode::

   import pyfftw

   a = pyfftw.n_byte_align_empty((128, 64), 16, 'complex128')
   
   # Generate some data
   ar, ai = numpy.random.randn(2, 128, 64)
   a[:] = ar + 1j*ai

   fft_object = pyfftw.builders.fft(a)

   b = fft_object()

``fft_object`` is an instance of :class:`pyfftw.FFTW`, ``b`` is 
the result of the DFT.

Note that in this example, unlike creating a :class:`pyfftw.FFTW` 
object using the direct interface, we can fill the array in advance.
This is because by default all the functions in :mod:`pyfftw.builders`
keep a copy of the input array during creation (though this can
be disabled).

The :mod:`pyfftw.builders` functions construct an output array of
the correct size and type. In the case of the regular DFTs, this
always creates an output array of the same size as the input array.
In the case of the real transform, the output array is the right
shape to satisfy the scheme requirements.

The precision of the transform is determined by the dtype of the
input array. If the input array is a floating point array, then
the precision of the floating point is used. If the input array 
is not a float point array then a double precision transform is used.
Any calls made to the resultant object with an array of the same 
size will then be copied into the internal array of the object, 
changing the dtype in the process.

Like :mod:`numpy.fft`, it is possible to specify a length (in the 
one-dimensional case) or a shape (in the multi-dimensional case) that
may be different to the array that is passed in. In such a case,
a wrapper object of type 
:class:`pyfftw.builders._utils._FFTWWrapper` is returned. From an
interface perspective, this is identical to :class:`pyfftw.FFTW`. The
difference is in the way calls to the object are handled. With
:class:`pyfftw.builders._utils._FFTWWrapper` objects, an array that
is passed as an argument when calling the object is *copied* into the
internal array. This is done by a suitable slicing of the new
passed-in array and the internal array and is done precisely because
the shape of the transform is different to the shape of the input
array.

.. testcode::

   a = pyfftw.n_byte_align_empty((128, 64), 16, 'complex128')
   
   fft_wrapper_object = pyfftw.builders.fftn(a, s=(32, 256))
   
   b = fft_wrapper_object()

Inspecting these objects gives us their shapes:

.. doctest::
   
   >>> b.shape
   (32, 256)
   >>> fft_wrapper_object.get_input_array().shape
   (32, 256)
   >>> a.shape
   (128, 64)

It is only possible to call ``fft_wrapper_object`` with an array
that is the same shape as ``a``. In this case, the first axis of ``a``
is sliced to include only the first 32 elements, and the second axis
of the internal array is sliced to include only the last 64 elements.
This way, shapes are made consistent for copying.

Understanding :mod:`numpy.fft`, these functions are largely
self-explanatory. We point the reader to the :mod:`API docs <pyfftw.builders>` for more information.

