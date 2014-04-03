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

from pyfftw import FFTW_MPI, local_size, n_byte_align_empty, simd_alignment
from mpi4py import MPI
import numpy as np
import unittest

# TODO check if 16 byte alignment used on AVX machine

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
master = (rank == 0)

def create_data(data):
    nx, ny, nz = data.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                data[i,j,k] = rank + (i + j * nx) * ny + k + 1

class MPITest(unittest.TestCase):
    '''
    Recreate the C example: Fourier transform 3D data back and forthz
    using the MPI version of FFTW
    '''

    # data shape definition
    dim = 3
    N = 16
    input_shape = (N, N, N)
    input_dtype = 'float64'
    output_dtype = 'complex128'
    a = n_byte_align_empty(input_shape, simd_alignment, dtype=input_dtype)

    def test_local_size_invalid(self):
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
                local_size(self.a, **kw)

        kwargs = [d(comm=3), # need proper communicator
                 ]

        for kw in kwargs:
            with self.assertRaises(TypeError) as cm:
                local_size(self.a, **kw)

        # FFT only defined for float, double, and long double
        with self.assertRaises(TypeError) as cm:
            local_size(np.empty(12, dtype='int'))

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

        if self.N % comm.Get_size() == 0:
            self.assertEqual(res[1], self.N // comm.Get_size())

        self.assertEqual(res[2], 0)

        if nres == 3:
            return

    def test_local_size(self):
        ###
        # valid arguments
        ###

        # fftw_mpi_local_size_many
        self.validate_local_size_result(local_size(self.a), 3)
        self.validate_local_size_result(local_size(self.a, comm=comm), 3)
        self.validate_local_size_result(local_size(self.a, direction='FFTW_BACKWARD'), 3)
        self.validate_local_size_result(local_size(self.a, direction='FFTW_BACKWARD'), 3)
        self.validate_local_size_result(local_size(self.a, block0=0), 3)
        self.validate_local_size_result(local_size(self.a, block1=0), 3)
        self.validate_local_size_result(local_size(self.a, howmany=1), 3)

        f = np.empty_like(self.a, dtype='float32')
        self.validate_local_size_result(local_size(f), 3)

        l = np.empty_like(self.a, dtype='longdouble')
        self.validate_local_size_result(local_size(f), 3)

        # now complex
        a = n_byte_align_empty(self.input_shape, simd_alignment, dtype='complex128')
        f = n_byte_align_empty(self.input_shape, simd_alignment, dtype='complex64')
        l = n_byte_align_empty(self.input_shape, simd_alignment, dtype='clongdouble')

        self.validate_local_size_result(local_size(a), 3)
        self.validate_local_size_result(local_size(f), 3)
        self.validate_local_size_result(local_size(l), 3)

        # fftw_mpi_local_size_many_transposed
        self.validate_local_size_result(local_size(self.a, transposed=True), 5)
        self.validate_local_size_result(local_size(f, transposed=True), 5)
        self.validate_local_size_result(local_size(l, transposed=True), 5)

        # fftw_mpi_local_size_many_1d
        a = n_byte_align_empty(self.N, simd_alignment, dtype='float64')
        f = n_byte_align_empty(self.N, simd_alignment, dtype='float32')
        l = n_byte_align_empty(self.N, simd_alignment, dtype='longdouble')

        self.validate_local_size_result(local_size(a), 5, dim=1)
        self.validate_local_size_result(local_size(f), 5, dim=1)
        self.validate_local_size_result(local_size(l), 5, dim=1)

    def test_basic(self):

        create_data(self.a)

        # get local size
        res = local_size(self.a, transposed=True, comm=comm);
        self.validate_local_size_result(res, 5)

        if master:
            print('local_size returns')
            print(res)
        return

        # allocate aligned data of the right size
        b = n_byte_align_empty(total_size, simd_alignment, dtype=self.output_dtype)
        c = self.a.copy()

        fft_object = FFTW_MPI(a, b, direction='FFTW_FORWARD', comm=comm,
                              flags=('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_OUT'))
        fft_back   = pyfftw.FFTW_MPI(b, c, direction='FFTW_BACKWARD', comm=comm,
                                     flags=('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_IN'))

        fft_object()
        fft_back(normalise_idft=True)
        np.testing.assert_allclose(a, c)

if __name__ == '__main__':
    '''Start as mpirun -n 4 python test_pyfftw_mpi.py'''
    unittest.main()
