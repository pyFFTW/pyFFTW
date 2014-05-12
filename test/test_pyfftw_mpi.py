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

from pyfftw import FFTW, FFTW_MPI, create_mpi_plan, local_size, n_byte_align_empty, simd_alignment
from mpi4py import MPI
import numpy as np
import unittest

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = (rank == 0)
msg = dict(err_msg='Error on rank %d' % rank)

np.set_printoptions(precision=3, linewidth=120)

def create_data(data):
    '''Integer sequence 0...4095 in row-major order.'''

    nx, ny, nz = data.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                data[i,j,k] = (i + j * nx) * ny + k

class MPITest(unittest.TestCase):
    '''
    Recreate the C example: Fourier transform 3D data back and forth
    using the MPI version of FFTW
    '''
    def setUp(self):
        # data shape definition
        self.dim = 3
        self.N = 16

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

    def test_create_mpi_plan(self):
        # 3d r2c
        plan = create_mpi_plan(input_shape=(self.N, self.N, self.N), input_dtype='float64', output_dtype='complex128')
        n_out = self.N // 2 + 1
        self.assertEqual(plan.conceptual_input_array().shape, (self.share, self.N, self.N))
        self.assertEqual(plan.conceptual_output_array().shape, (self.share, self.N, n_out))

        # 3d c2r
        plan = create_mpi_plan(input_shape=(self.N, self.N, self.N), input_dtype='complex128',
                               output_dtype='float64', direction='FFTW_BACKWARD')

        self.assertEqual(plan.conceptual_input_array().shape,  (self.share, self.N, n_out))
        self.assertEqual(plan.conceptual_output_array().shape, (self.share, self.N, self.N))

        # 2d c2c example
        plan = create_mpi_plan(input_shape=(self.N, self.N), input_dtype='complex128',
                               output_dtype='complex128', direction='FFTW_FORWARD')
        self.assertEqual(plan.conceptual_input_array().shape, (self.share, self.N))
        self.assertEqual(plan.conceptual_output_array().shape, (self.share, self.N))

        # 1d c2c
        plan = create_mpi_plan(input_shape=(self.N,), input_dtype='complex128',
                               output_dtype='complex128', direction='FFTW_BACKWARD')
        # fftw 6.4.2 says: For a one-dimensional complex DFT, the
        # length N of the data should be divisible by P squared to be
        # able to divide the problem equally among the processes.
        if self.N % comm.Get_size()**2 == 0:
            self.assertEqual(plan.conceptual_input_array().shape, (self.share,))
            self.assertEqual(plan.conceptual_output_array().shape, (self.share,))

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
        backward_plan = create_mpi_plan(input_shape, local_input_array=forward_plan.output_array,
                                        output_dtype=input_dtype, direction='FFTW_BACKWARD',
                                        flags=backward_flags)

        forward_plan()
        backward_plan()
        np.testing.assert_equal(forward_plan.conceptual_output_array(), target_b)

        # no unnecessary reallocation of arrays
        self.assert_(forward_plan.output_array is backward_plan.input_array)

        # now do it in-place
        forward_plan = create_mpi_plan(input_shape, input_dtype=input_dtype,
                                       local_output_array='INPUT', direction='FFTW_FORWARD',
                                       flags=forward_flags)
        forward_plan.input_array[:] = 1.0
        forward_plan()
        np.testing.assert_equal(forward_plan.conceptual_output_array(), target_b)
        # no unnecessary reallocation of arrays
        self.assert_(forward_plan.input_array is forward_plan.output_array)

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
            input_data = fplan.conceptual_input_array()
            input_data[:] = 1.0
            print 'rank', rank, fplan.input_array.shape
            padded_shape = (local_n0, 6)
            np.testing.assert_equal(input_data, fplan.input_array[0:np.prod(padded_shape)].reshape(padded_shape)[..., 0:4:1],**msg)

            # only one rank has the delta component;
            # everything else is zero
            if fplan.local_0_start == 0:
                target[0,0] = np.prod(input_shape) + 0.j


        fplan()

        if fplan.local_output_shape:
            np.testing.assert_equal(fplan.conceptual_output_array(), target[fplan.output_slice])

        backward_flags = forward_flags
        backward_plan = create_mpi_plan(input_shape,
                                        local_input_array=fplan.output_array,
                                        output_dtype='float64',
                                        flags=backward_flags)

        backward_plan()

        if backward_plan.local_output_shape:
            np.testing.assert_equal(backward_plan.conceptual_output_array(), input_data)

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
                forward_plan.conceptual_input_array()[m,n] = float(m) - 2.j * n

        # compare with numpy
        target = np.fft.fftn(forward_plan.conceptual_input_array())
        forward_plan()
        np.testing.assert_allclose(forward_plan.conceptual_output_array(), target)

        backward_flags = forward_flags
        backward_plan = create_mpi_plan(output_shape, local_input_array=forward_plan.output_array,
                                        output_dtype='complex128', direction='FFTW_BACKWARD',
                                        flags=backward_flags)

        backward_plan.execute()
        # unnormalized * FFT
        target = np.prod(input_shape) * forward_plan.conceptual_input_array()
        # print backward_plan.output_array
        # print target
        np.testing.assert_allclose(backward_plan.conceptual_output_array(), target)

        # automatic normalization
        backward_plan()
        np.testing.assert_allclose(backward_plan.conceptual_output_array(), forward_plan.conceptual_input_array())

    def test_howmany(self):
        input_shape = (10, 5, 6)
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
        local_data = forward_plan.conceptual_input_array()
        # print local_data.shape
        # print forward_plan.input_array.shape
        # print forward_plan.input_shape

        local_n0 = forward_plan.local_n0
        local_0_start = forward_plan.local_0_start
        local_data[:] = global_data[local_0_start:local_0_start + local_n0]

        # second transform just a constant
        forward_plan.conceptual_input_array(1)[:] = 1.

        delta_fct = np.zeros_like(forward_plan.conceptual_output_array(1))
        # only one rank has the delta component;
        # everything else is zero
        if forward_plan.local_0_start == 0:
            delta_fct[0,0,0] = np.prod(input_shape) + 0.j

        ###
        # transform and check
        ###
        forward_plan()
        local_target = global_target[local_0_start:local_0_start + local_n0]

        local_result = forward_plan.conceptual_output_array(0)

        if master:
            print local_result[0,1]
            print local_target[1,0]

        # lots of near-zero numbers => fix atol
        self.assertEqual(local_result.shape, local_target.shape)
        np.testing.assert_allclose(local_result, local_target, atol=1e-10, **msg)

        # second transform a delta function
        self.assertEqual(forward_plan.conceptual_output_array(1).shape, delta_fct.shape)
        np.testing.assert_allclose(forward_plan.conceptual_output_array(1), delta_fct, atol=1e-10, **msg)

        ###
        # assign second

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
                fplan.conceptual_output_array()
        else:
            shape = fplan.conceptual_output_array().shape
            # print 'rank', rank, 'output shape', shape

            # get at least one slab, two only if comm size <= 4
            out = 4 // comm.Get_size()
            if comm.Get_size() == 3 or comm.Get_size() > 4:
                out += 1
            print 'transposed output shape', shape
            self.assertEqual(shape[0], out)
            # transpose, so x dimension *not* partitioned
            self.assertEqual(shape[1], 6)

        # NOTE: If run with 4 or more processes, there are extra bytes
        # on some processors as 6 not divisible by 4(?)

        ###
        # backward transform
        ###
        bflags = ('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_IN')
        bplan = create_mpi_plan(input_shape, local_input_array=fplan.output_array,
                                output_dtype='complex128',
                                direction='FFTW_BACKWARD', flags=bflags)
        if master:
            print bplan.conceptual_input_array().shape
            print bplan.conceptual_output_array().shape
            print bplan.flags

        ###
        # enter data and do the transforms
        ###
        data = np.empty((input_shape), fplan.input_array.dtype)
        for m in range(input_shape[0]):
            for n in range(input_shape[1]):
                data[m,n] = float(m) - 2.j * n

        # cumbersome and manual attribute handling
        try:
            i = fplan.conceptual_input_array()
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
            o = fplan.conceptual_output_array()

            # output is transposed, and from the input we have to get
            # a selection of the columns, not the rows
            # np.testing.assert_allclose(o.transpose(), target[..., fplan.local_1_start:fplan.local_1_start + o.shape[0]])
            np.testing.assert_allclose(o.transpose(), target[..., fplan.output_slice], **msg)

        if bplan.local_output_shape:
            o = bplan.conceptual_output_array()

            # now the output is back to the usual format, and *not* transposed
            np.testing.assert_allclose(o, data[bplan.output_slice], **msg)

if __name__ == '__main__':
    '''Start as mpirun -n 4 python test_pyfftw_mpi.py'''
    unittest.main()

# Local Variables:
# compile-command: "cd ../ && CC=mpicc python setup.py build_ext --inplace && mpirun -n 4 python test/test_pyfftw_mpi.py"
# End:
