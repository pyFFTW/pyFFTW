# Copyright 2014
#
# Frederik.Beaujean@lmu.de
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# TODO skip all tests if FFTW_MPI doesn't exist but FFTW does
from pyfftw import FFTW, FFTW_MPI, create_mpi_plan, local_size, n_byte_align_empty, simd_alignment, supported_mpi_types
from mpi4py import MPI
import numpy as np
import unittest
from timeit import Timer

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = (rank == 0)
msg = dict(err_msg='Error on rank %d' % rank)

np.set_printoptions(precision=3, linewidth=120)

def my_assert_allclose(x, y, **kwargs):
    '''Update tolerance and error message default arguments passed to
    numpy.testing.assert_allclose.

    '''
    kwargs['err_msg'] = kwargs.get('err_msg', 'Error on rank %d' % rank)
    kwargs['atol'] = kwargs.get('atol', 1e-5)

    np.testing.assert_allclose(x, y, **kwargs)

def assert_pointers(x, y):
    '''Compare pointers of two numpy arrays for equality.

    Based on
    http://www.gossamer-threads.com/lists/python/dev/1011232

    '''
    from ctypes import addressof, c_char

    assert addressof(c_char.from_buffer(x.data)) == addressof(c_char.from_buffer(y.data)), \
    'buffer addresses differ'

def create_data(data):
    '''Integer sequence 0...4095 in row-major order.'''

    nx, ny, nz = data.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                data[i,j,k] = (i + j * nx) * ny + k

global_N = 16

class MPITest(unittest.TestCase):
    '''
    Recreate the C example: Fourier transform 3D data back and forth
    using the MPI version of FFTW
    '''
    def setUp(self):
        # data shape definition
        self.dim = 3
        self.N = global_N

        self.share = self.N // comm.Get_size()

    def test_local_size_invalid(self):
        args = ((self.N, self.N, self.N),)

        ###
        # invalid arguments
        ###
        d = dict
        kwargs = [d(block0='servus'), d(block1='fred'),
                  d(block0=-12), d(block1=-152),
                  d(block0=1.1), d(block1=15.2),
                  d(howmany=0), d(howmany=-1),
                  d(flags=('NONEXISTING_FLAG',)), d(flags=1),
                  d(direction='NONEXISTING_DIRECTION'),
                 ]

        for kw in kwargs:
            with self.assertRaises(ValueError) as cm:
                local_size(*args, **kw)

        kwargs = [d(comm=3), # need proper communicator
                 ]

        for kw in kwargs:
            with self.assertRaises(TypeError) as cm:
                local_size(*args, **kw)

    def validate_local_size_result(self, res, nres, dim=3):
        self.assertEqual(len(res), nres)

        # assume equal distribution if no remainder
        total_size = self.N ** dim
        if total_size % comm.Get_size() == 0:
            # integer division
            balanced = total_size // comm.Get_size()
            self.assertEqual(res[0], balanced)
        if nres == 1:
            return

        self.assertEqual(res[1], self.N // comm.Get_size())

        self.assertEqual(res[2], 0)

        if nres == 3:
            return

    @unittest.skipIf(comm.Get_size() > 1, 'only for a single process')
    def test_local_size(self):
        ###
        # valid arguments
        ###
        args  = ((self.N, self.N, self.N),)
        # fargs = (self.input_shape, 'float32')
        # largs = (self.input_shape, 'longdouble')

        # fftw_mpi_local_size_many
        self.validate_local_size_result(local_size(*args), 3)
        self.validate_local_size_result(local_size(*args, comm=comm), 3)
        self.validate_local_size_result(local_size(*args, direction='FFTW_BACKWARD'), 3)
        self.validate_local_size_result(local_size(*args, direction='FFTW_BACKWARD'), 3)
        self.validate_local_size_result(local_size(*args, block0=0), 3)
        self.validate_local_size_result(local_size(*args, block1=0), 3)
        self.validate_local_size_result(local_size(*args, howmany=1), 3)


        # fftw_mpi_local_size_many_transposed
        self.validate_local_size_result(local_size(*args, flags=('FFTW_MPI_TRANSPOSED_OUT',)), 5)

        # fftw_mpi_local_size_many_1d
        self.validate_local_size_result(local_size(input_shape=(self.N,)), 5, dim=1)

    @unittest.skipIf(global_N % comm.Get_size(), 'Predict FFTW MPI share only if %d divisible by number of processes' % global_N)
    def test_create_mpi_plan(self):
        # 3d r2c
        plan = create_mpi_plan(input_shape=(self.N, self.N, self.N), input_dtype='float64', output_dtype='complex128')
        n_out = self.N // 2 + 1
        self.assertEqual(plan.input_array.shape, (self.share, self.N, self.N))
        self.assertEqual(plan.get_output_array().shape, (self.share, self.N, n_out))

        # 3d c2r
        plan = create_mpi_plan(input_shape=(self.N, self.N, self.N), input_dtype='complex128',
                               output_dtype='float64', direction='FFTW_BACKWARD')

        self.assertEqual(plan.input_array.shape,  (self.share, self.N, n_out))
        self.assertEqual(plan.get_output_array().shape, (self.share, self.N, self.N))

        # 2d c2c example
        plan = create_mpi_plan(input_shape=(self.N, self.N), input_dtype='complex128',
                               output_dtype='complex128', direction='FFTW_FORWARD')
        self.assertEqual(plan.input_array.shape, (self.share, self.N))
        self.assertEqual(plan.get_output_array().shape, (self.share, self.N))

        # 1d c2c
        plan = create_mpi_plan(input_shape=(self.N,), input_dtype='complex128',
                               output_dtype='complex128', direction='FFTW_BACKWARD')
        # fftw 6.4.2 says: For a one-dimensional complex DFT, the
        # length N of the data should be divisible by P squared to be
        # able to divide the problem equally among the processes.
        if self.N % comm.Get_size()**2 == 0:
            self.assertEqual(plan.input_array.shape, (self.share,))
            self.assertEqual(plan.get_output_array().shape, (self.share,))

        # invalid complex type
        with self.assertRaises(TypeError):
            create_mpi_plan(input_shape=(self.N, self.N, self.N), input_dtype='float64', output_dtype='complex124')

        # 1d r2c / c2r not supported
        for i, o in (('float64', 'complex128'), ('complex128', 'float64')):
            with self.assertRaises(NotImplementedError):
                plan = create_mpi_plan(input_shape=(self.N,), input_dtype=i, output_dtype=o)

    @unittest.skip('test serial only for communication with Margarita')
    def test_serial(self):
        b = n_byte_align_empty((self.N, self.N, self.N // 2 + 1), simd_alignment, dtype=self.output_dtype)
        fft_object = FFTW(self.a, b, direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',), axes=range(self.a.ndim))
        create_data(self.a)
        fft_object()

        with file('transform.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(b.shape))

            for slice in b:
                np.savetxt(outfile, slice, fmt='%.4f')
                outfile.write('# New slice\n')
        print(b[0,0:2,0])

        data = np.ones((2,2))
        res = np.fft.rfftn(data)
        print(data)
        print(res)

    # compute forward and backward complex transform of constant array
    @unittest.skipIf(comm.Get_size() > 1, 'only for a single process')
    def test_basic(self):
        input_shape = tuple([self.N] * 3)
        input_dtype = 'complex128'
        output_dtype = 'complex128'
        a = n_byte_align_empty(input_shape, simd_alignment, dtype=input_dtype)

        a[:] = 1.0

        # get local size
        res = local_size(a.shape, flags=('FFTW_MPI_TRANSPOSED_OUT',), comm=comm);
        self.validate_local_size_result(res, 5)

        # allocate aligned data of the right size
        chunk = n_byte_align_empty(res[0], simd_alignment, dtype=output_dtype)
        b = np.frombuffer(chunk.data, dtype=output_dtype,
                          count=np.prod(a.shape)).reshape(a.shape)
        c = a.copy()

        forward_flags = ('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_OUT')

        fft_object = FFTW_MPI(a.shape, a, b, direction='FFTW_FORWARD', comm=comm,
                              flags=forward_flags)

        fft_object()
        backward_flags = ('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_IN')
        fft_back   = FFTW_MPI(a.shape, b, c, direction='FFTW_BACKWARD', comm=comm,
                                     flags=backward_flags)

        fft_object()
        fft_back(normalise_idft=True)
        # forward and normalized backward should be identical
        np.testing.assert_allclose(a, c)

        # only DC component at (0,0,0), everything else vanishes
        target_b = np.zeros(input_shape, dtype=input_dtype)
        target_b[0,0,0] = self.N**3
        np.testing.assert_equal(b, target_b)

        # now repeat the same with less manual intervention
        forward_plan = create_mpi_plan(input_shape, input_dtype=input_dtype,
                                       output_dtype=output_dtype, direction='FFTW_FORWARD',
                                       flags=forward_flags)
        forward_plan.input_array[:] = 1.0
        backward_plan = create_mpi_plan(input_shape, input_chunk=forward_plan.output_chunk,
                                        output_dtype=input_dtype, direction='FFTW_BACKWARD',
                                        flags=backward_flags)

        forward_plan()
        backward_plan()
        np.testing.assert_equal(forward_plan.get_output_array(), target_b)

        # no unnecessary reallocation of arrays
        assert_pointers(forward_plan.output_chunk, backward_plan.input_chunk)

        # now do it in-place
        forward_plan = create_mpi_plan(input_shape, input_dtype=input_dtype,
                                       output_chunk='INPUT', direction='FFTW_FORWARD',
                                       flags=forward_flags)
        forward_plan.input_array[:] = 1.0
        forward_plan()
        np.testing.assert_equal(forward_plan.get_output_array(), target_b)
        # no unnecessary reallocation of arrays
        assert_pointers(forward_plan.input_chunk, forward_plan.output_chunk)

    # test r2c and c2r in 2D on multiple processes with
    # constant -> delta fct. -> constant
    def test_real(self):
        # 2d
        forward_flags = ('FFTW_ESTIMATE',)
        input_shape  = (6, 4)
        fplan = create_mpi_plan(input_shape, input_dtype='float64',
                                       output_dtype='complex128', flags=forward_flags,
                                       comm=comm)

        local_n0 = fplan.local_n0

        target = np.zeros((6,3), dtype='complex128')

        # both commands are equivalent
        # 0-3 data columns; 4,5 padding
        if fplan.local_input_shape:
            input_data = fplan.input_array
            input_data[:] = 1.0
            padded_shape = (local_n0, 6)
            np.testing.assert_equal(input_data, fplan.input_chunk[0:np.prod(padded_shape)].reshape(padded_shape)[..., 0:4:1],**msg)

            # only one rank has the delta component;
            # everything else is zero
            if fplan.local_0_start == 0:
                target[0,0] = np.prod(input_shape) + 0.j


        fplan()

        if fplan.local_output_shape:
            np.testing.assert_equal(fplan.get_output_array(), target[fplan.output_slice])

        backward_flags = forward_flags
        backward_plan = create_mpi_plan(input_shape,
                                        input_chunk=fplan.output_chunk,
                                        output_dtype='float64',
                                        flags=backward_flags)

        backward_plan()

        if backward_plan.local_output_shape:
            np.testing.assert_equal(backward_plan.get_output_array(), input_data)

        # TODO hangs with 7 processes

    @unittest.skipIf(comm.Get_size() > 1, 'only for a single process')
    def test_c2c(self):
        forward_flags = ('FFTW_ESTIMATE',)
        input_shape  = (6, 3)
        output_shape = input_shape
        forward_plan = create_mpi_plan(input_shape, input_dtype='complex128',
                                       output_dtype='complex128', direction='FFTW_FORWARD',
                                       flags=forward_flags)

        for m in range(input_shape[0]):
            for n in range(input_shape[1]):
                forward_plan.input_array[m,n] = float(m) - 2.j * n

        # compare with numpy
        target = np.fft.fftn(forward_plan.input_array)
        forward_plan()

        my_assert_allclose(forward_plan.get_output_array(), target)

        backward_flags = forward_flags
        backward_plan = create_mpi_plan(output_shape, input_chunk=forward_plan.output_chunk,
                                        output_dtype='complex128', direction='FFTW_BACKWARD',
                                        flags=backward_flags)

        backward_plan.execute()
        # unnormalized * FFT
        target = np.prod(input_shape) * forward_plan.input_array
        # print backward_plan.output_chunk
        # print target
        my_assert_allclose(backward_plan.get_output_array(), target)

        # automatic normalization
        backward_plan()
        my_assert_allclose(backward_plan.get_output_array(), forward_plan.input_array)

    def test_howmany(self):
        input_shape = (100, 200, 6)
        dtype = 'complex128'

        # complex data
        global_data = np.array([i - 3.j * i for i in range(np.prod(input_shape))]).reshape(*input_shape)
        global_target = np.fft.fftn(global_data)
        # if master:
        #     print global_target

        kwargs = dict(input_dtype=dtype, output_dtype=dtype, howmany=2,
                      direction='FFTW_FORWARD',
                      flags=('FFTW_ESTIMATE',))
        forward_plan = create_mpi_plan(input_shape, **kwargs)

        ###
        # copy local part
        ###
        if forward_plan.has_input:
            local_data = forward_plan.input_array
            # print local_data.shape
            # print forward_plan.input_array.shape
            # print forward_plan.input_shape

            local_n0 = forward_plan.local_n0
            local_0_start = forward_plan.local_0_start
            local_data[:] = global_data[local_0_start:local_0_start + local_n0]

            # second transform just a constant
            forward_plan.get_input_array(1)[:] = 1.

        if forward_plan.local_output_shape:
            print 'rank', rank, forward_plan.local_output_shape
            delta_fct = np.zeros_like(forward_plan.get_output_array(1))
            # only one rank has the delta component;
            # everything else is zero
            if forward_plan.local_0_start == 0:
                delta_fct[0,0,0] = np.prod(input_shape) + 0.j

        ###
        # transform and check
        ###
        forward_plan()
        if forward_plan.has_output:
            local_target = global_target[local_0_start:local_0_start + local_n0]
            local_result = forward_plan.get_output_array(0)

            # lots of near-zero numbers => fix atol
            self.assertEqual(local_result.shape, local_target.shape)
            my_assert_allclose(local_result, local_target, atol=1e-2)

            # second transform a delta function
            self.assertEqual(forward_plan.get_output_array(1).shape, delta_fct.shape)
            np.testing.assert_allclose(forward_plan.get_output_array(1), delta_fct, atol=1e-10, **msg)

    @unittest.skipIf('32' not in supported_mpi_types, 'only single precision')
    def test_r2c_inplace(self):
        # large but random data with *equal* dimensions
        input_shape = [100] * 2 #[20, 60]
        input_dtype = np.dtype('float32')
        output_dtype = np.dtype('complex64')
        # flags = ('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_OUT')
        flags = ['FFTW_ESTIMATE']
        kwargs = dict(input_dtype=input_dtype,
                  output_chunk='INPUT',
                  direction='FFTW_FORWARD', flags=['FFTW_ESTIMATE',])

        fplan = create_mpi_plan(input_shape, **kwargs)
        # fplan()
        self.assertEqual(fplan.input_array.dtype, input_dtype)
        self.assertEqual(fplan.output_array.dtype, output_dtype)
        if fplan.has_input:
            self.assertEqual(fplan.local_input_shape[-1], 100)
        if fplan.has_output:
            self.assertEqual(fplan.local_output_shape[-1], 51)

        # transpose should change the output shape only
        kwargs['flags'].append('FFTW_MPI_TRANSPOSED_OUT')
        fplan_tr = create_mpi_plan(fplan.input_shape, **kwargs)
        if fplan_tr.has_input:
            self.assertEqual(fplan_tr.local_input_shape, fplan.local_input_shape)
        if fplan.has_output:
            self.assertEqual(fplan_tr.local_output_shape[-1], 100)

    # TODO fails with shape [700] * 3
    @unittest.skipIf('32' not in supported_mpi_types, 'only single precision')
    def test_threads(self):
        input_shape = [200] * 3
        input_dtype = np.dtype('float32')
        output_dtype = np.dtype('complex64')
        kwargs = dict(input_dtype=input_dtype,
                      output_chunk='INPUT',
                      direction='FFTW_FORWARD', flags=['FFTW_ESTIMATE'],
                      threads=1)
        fplan = create_mpi_plan(input_shape, **kwargs)

        # constant array --> delta fct.
        if fplan.has_input:
            fplan.input_array[:] = 1.

        print 'start serial'
        t = Timer(lambda: fplan())
        print 'serial time', t.timeit(number=1)
        # print 'end serial'

        target = np.zeros_like(fplan.output_array)
        if fplan.has_output:
            if fplan.local_0_start == 0:
                # use itemset to access first memory location
                # regardless of dimensionality
                target.itemset(0, np.prod(input_shape) + 0.j)
                # pass

            difference = np.transpose(np.nonzero(abs(fplan.output_array - target) > 1e-6))

            # msg ='x: %s\n, y: %s' % (str(fplan.output_array[difference]), str(target[difference]))
            # msg = str(difference)
            # msg ='x: %f\n, y: %f' % (fplan.output_array[difference[0]].sum(), target[difference[1]].sum())
            diff = abs(fplan.output_array - target) > 1e-6
            imax = np.argmax(diff)
            msg = str(fplan.output_array.item(imax)) + ', ' + str(target.item(imax))
            np.testing.assert_array_almost_equal(fplan.output_array, target, decimal=2, err_msg=msg)

        # now use multiple threads to get same result
        # reuse local memory this time
        kwargs['threads'] = 2
        kwargs.pop('input_dtype')
        fplan_threads = create_mpi_plan(input_shape, input_chunk=fplan.input_chunk, **kwargs)
        if fplan_threads.has_input:
            fplan_threads.input_array[:] = 1.

        print 'start parallel'
        t = Timer(lambda: fplan_threads())
        print 'time using %d threads' % fplan_threads.threads, t.timeit(number=1)

        if fplan_threads.has_output:
            np.testing.assert_allclose(fplan_threads.output_array, target, atol=1e-2)

    def test_transposed(self):
        input_shape = (6,4)
        flags=('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_OUT')
        # res = local_size(input_shape, flags=flags)
        # print res
        fplan = create_mpi_plan(input_shape, input_dtype='complex128', output_dtype='complex128',
                                direction='FFTW_FORWARD', flags=flags)

        # 4 not divisible by 3, so last rank with get no output data
        if (comm.Get_size() == 3 and rank == 2) or \
           (comm.Get_size() > 4 and rank > 3):
            with self.assertRaises(AttributeError) as cm:
                fplan.get_output_array()
        else:
            shape = fplan.get_output_array().shape
            # print 'rank', rank, 'output shape', shape

            # get at least one slab, two only if comm size <= 4
            out = 4 // comm.Get_size()
            if comm.Get_size() == 3 or comm.Get_size() > 4:
                out += 1

            self.assertEqual(shape[0], out)
            # transpose, so x dimension *not* partitioned
            self.assertEqual(shape[1], 6)

        # NOTE: If run with 4 or more processes, there are extra bytes
        # on some processors as 6 not divisible by 4(?)

        ###
        # backward transform
        ###
        bflags = ('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_IN')
        bplan = create_mpi_plan(input_shape, input_chunk=fplan.output_chunk,
                                output_dtype='complex128',
                                direction='FFTW_BACKWARD', flags=bflags)

        ###
        # enter data and do the transforms
        ###
        data = np.empty((input_shape), fplan.input_chunk.dtype)
        for m in range(input_shape[0]):
            for n in range(input_shape[1]):
                data[m,n] = float(m) - 2.j * n

        # cumbersome and manual attribute handling
        try:
            i = fplan.input_array
        except AttributeError:
            i = None
        if i is not None:
            # i[:] = data[fplan.local_0_start:fplan.local_0_start + fplan.local_n0]
            i[:] = data[fplan.input_slice]

        # compare with numpy
        target = np.fft.fftn(data)
        fplan()
        bplan()

        # get local part of data,
        # check there is some at all
        if fplan.local_output_shape:
            o = fplan.get_output_array()

            # output is transposed, and from the input we have to get
            # a selection of the columns, not the rows
            # np.testing.assert_allclose(o.transpose(), target[..., fplan.local_1_start:fplan.local_1_start + o.shape[0]])
            np.testing.assert_allclose(o.transpose(), target[..., fplan.output_slice], **msg)

        if bplan.local_output_shape:
            o = bplan.get_output_array()

            # now the output is back to the usual format, and *not* transposed
            np.testing.assert_allclose(o, data[bplan.output_slice], **msg)

    def test_1d(self):
        # only c2c implemented
        N = 50
        p = create_mpi_plan(N, input_dtype='complex128',
                        output_dtype='complex128',
                        direction='FFTW_FORWARD')

        global_data = np.array([float(n) - 2.j * n for n in range(N)])
        global_target = np.fft.fft(global_data)

        if p.local_input_shape:
            p.input_array[:] = global_data[p.input_slice]

        p()

        if p.local_output_shape:
            np.testing.assert_allclose(p.get_output_array(), global_target[p.output_slice], **msg)

        ###
        # backward transform
        ###
        b = create_mpi_plan(N, input_chunk=p.output_chunk,
                            output_dtype='complex128',
                            direction='FFTW_BACKWARD',
                            flags=('FFTW_ESTIMATE',))

        b()

        # comparing zeros => nonzero atol
        if b.local_output_shape:
            np.testing.assert_allclose(b.get_output_array(),
                                       global_data[b.output_slice],
                                       atol=1e-13, **msg)

        ###
        # SCRAMBLED only works with more than one process
        ###
        if comm.Get_size() == 1:
            kwargs = dict(input_dtype='complex128',
                                output_dtype='complex128',
                                direction='FFTW_FORWARD')

            for f in ('FFTW_MPI_SCRAMBLED_OUT', 'FFTW_MPI_SCRAMBLED_IN'):
                with self.assertRaises(NotImplementedError) as cm:
                    create_mpi_plan(N, flags=('FFTW_ESTIMATE', 'FFTW_MPI_SCRAMBLED_IN'), **kwargs)

        return

        p = create_mpi_plan(N, input_dtype='complex128',
                            output_dtype='complex128',
                            direction='FFTW_FORWARD',
                            flags=('FFTW_ESTIMATE', 'FFTW_MPI_SCRAMBLED_OUT'))

        p.input_array[:] = global_data[p.input_slice]

        p()

        # if master:
        #     print p.get_output_array()

        b = create_mpi_plan(N, input_chunk=p.output_chunk,
                            output_dtype='complex128',
                            direction='FFTW_BACKWARD',
                            flags=('FFTW_ESTIMATE', 'FFTW_MPI_SCRAMBLED_IN'))
        b()

        if b.local_output_shape:
            np.testing.assert_allclose(b.get_output_array(),
                                       global_data[b.output_slice],
                                       atol=1e-13, **msg)


if __name__ == '__main__':
    '''Start as mpirun -n 4 python test_pyfftw_mpi.py'''
    unittest.main()

# compile-command: "cd ../ && CC=mpicc python setup.py build_ext --inplace && cd test && mpirun -n 1 python -m unittest test_pyfftw_mpi.MPITest.test_c2c"
# Local Variables:
# compile-command: "cd ../ && rm pyfftw/pyfftw.c; CC=mpicc python setup.py build_ext --inplace && mpirun -n 2 python test/test_pyfftw_mpi.py"
# End:
