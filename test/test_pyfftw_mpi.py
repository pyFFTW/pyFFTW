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

from pyfftw import FFTW, FFTW_MPI, n_byte_align_empty, simd_alignment
from mpi4py import MPI
import numpy
import unittest

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
    Recreate the C example: Fourier transform 3D data back and forth
    using the MPI version of FFTW
    '''

    # data shape definition
    dim = 3
    N = 16
    input_shape = (N, N, N)
    input_dtype = 'float64'
    output_dtype = 'complex128'

    def test_basic(self):

        a = n_byte_align_empty(self.input_shape, simd_alignment, dtype=self.input_dtype)
        create_data(a)

        # get local size
        local_size, local_n0, local_0_start = FFTW_MPI.local_size(a, comm=comm)
        b = n_byte_align_empty(local_size, simd_alignment, dtype=output_dtype)
        c = a.copy()

        fft_object = pyfftw.FFTW(a, b, direction='FFTW_FORWARD',  flags=('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_OUT'), comm=comm)
        fft_back   = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD', flags=('FFTW_ESTIMATE', 'FFTW_MPI_TRANSPOSED_IN'), comm=comm)

        fft_object()
        fft_back(normalise_idft=True)
        np.testing.assert_allclose(a, c)
