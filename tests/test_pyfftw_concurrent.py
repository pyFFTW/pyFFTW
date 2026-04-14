# Copyright 2025 the PyFFTW contributors
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

"""Tests for concurrent pyFFTW operations from multiple Python threads.

These tests verify that plan creation, execution, and wisdom operations
are thread-safe. On free-threaded Python (3.13t+), the threads run truly
in parallel; on regular Python, the GIL serializes Python code but the
locking logic (plan_lock) is still exercised.
"""

import concurrent.futures
import unittest

import numpy as np

import pyfftw

from .test_pyfftw_base import run_test_suites


class ConcurrentPlanCreationTest(unittest.TestCase):
    """Multiple threads creating FFTW plans simultaneously."""

    def test_concurrent_plan_creation(self):
        def create_plan(size):
            a = pyfftw.empty_aligned(size, dtype='complex128')
            b = pyfftw.empty_aligned(size, dtype='complex128')
            return pyfftw.FFTW(a, b, flags=('FFTW_ESTIMATE',))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            sizes = [32, 64, 128, 256] * 10
            futures = [pool.submit(create_plan, s) for s in sizes]
            plans = [f.result() for f in futures]

        self.assertEqual(len(plans), 40)

    def test_concurrent_plan_creation_varied_threads(self):
        """Plans with different FFTW thread counts created concurrently.

        Verifies that fftw_plan_with_nthreads is protected by plan_lock
        so one thread's nthreads setting doesn't leak into another's plan.
        """
        results = {}

        def create_plan_with_threads(n_fftw_threads):
            a = pyfftw.empty_aligned(128, dtype='complex128')
            b = pyfftw.empty_aligned(128, dtype='complex128')
            return pyfftw.FFTW(
                a, b, threads=n_fftw_threads, flags=('FFTW_ESTIMATE',)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for t in [1, 2, 4, 1, 2, 4] * 3:
                futures[pool.submit(create_plan_with_threads, t)] = t
            for f in concurrent.futures.as_completed(futures):
                f.result()  # raises if plan creation failed


class ConcurrentExecutionTest(unittest.TestCase):
    """Multiple threads executing FFTs simultaneously."""

    def test_concurrent_execution_correctness(self):
        np.random.seed(42)
        n_workers = 8
        n_tasks = 32

        tasks = []
        for i in range(n_tasks):
            size = 64 * (i % 4 + 1)
            a = pyfftw.empty_aligned(size, dtype='complex128')
            b = pyfftw.empty_aligned(size, dtype='complex128')
            fft = pyfftw.FFTW(a, b, flags=('FFTW_ESTIMATE',))
            data = np.random.randn(size) + 1j * np.random.randn(size)
            expected = np.fft.fft(data)
            tasks.append((fft, data, expected))

        def execute_and_verify(task):
            fft, data, expected = task
            result = fft(data.copy())
            return np.allclose(result, expected, atol=1e-10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(execute_and_verify, t) for t in tasks]
            results = [f.result() for f in futures]

        self.assertTrue(all(results))


class ConcurrentWisdomTest(unittest.TestCase):
    """Wisdom operations from multiple threads simultaneously."""

    def test_concurrent_wisdom_roundtrip(self):
        def wisdom_roundtrip(_):
            wisdom = pyfftw.export_wisdom()
            pyfftw.import_wisdom(wisdom)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(wisdom_roundtrip, i) for i in range(20)]
            for f in futures:
                f.result()

    def test_concurrent_plan_and_wisdom(self):
        """Plan creation and wisdom operations interleaved."""
        def create_plan(_):
            a = pyfftw.empty_aligned(64, dtype='complex128')
            b = pyfftw.empty_aligned(64, dtype='complex128')
            pyfftw.FFTW(a, b, flags=('FFTW_ESTIMATE',))

        def wisdom_op(_):
            wisdom = pyfftw.export_wisdom()
            pyfftw.import_wisdom(wisdom)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for i in range(30):
                if i % 2 == 0:
                    futures.append(pool.submit(create_plan, i))
                else:
                    futures.append(pool.submit(wisdom_op, i))
            for f in futures:
                f.result()


test_cases = (
    ConcurrentPlanCreationTest,
    ConcurrentExecutionTest,
    ConcurrentWisdomTest,
)

test_set = None

if __name__ == '__main__':
    run_test_suites(test_cases, test_set)
