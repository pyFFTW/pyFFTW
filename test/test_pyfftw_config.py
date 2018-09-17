
from pyfftw import config, _threading_type


from .test_pyfftw_base import run_test_suites

import unittest
import os
from numpy.testing import assert_equal


class ConfigTest(unittest.TestCase):

    env_keys = ['PYFFTW_NUM_THREADS', 'OMP_NUM_THREADS',
                'PYFFTW_PLANNER_EFFORT']
    orig_env = {}

    def setUp(self):
        # store environment variables prior to testing
        for key in self.env_keys:
            self.orig_env[key] = os.environ.get(key, None)
        return

    def tearDown(self):
        # resstore original environment variables values
        for key in self.env_keys:
            val = self.orig_env[key]
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        return

    def test_default_config(self):
        # unset environment variables if they were defined
        os.environ.pop('PYFFTW_NUM_THREADS', None)
        os.environ.pop('OMP_NUM_THREADS', None)
        os.environ.pop('PYFFTW_PLANNER_EFFORT', None)
        # defaults to single-threaded and FFTW_ESTIMATE
        config._reload_config()
        assert_equal(config.NUM_THREADS, 1)
        assert_equal(config.PLANNER_EFFORT, 'FFTW_ESTIMATE')

    @unittest.skipIf(_threading_type != 'OMP', reason='non-OpenMP build')
    def test_default_threads_OpenMP(self):
        # unset environment variables if they were defined
        os.environ.pop('PYFFTW_NUM_THREADS', None)
        os.environ.pop('OMP_NUM_THREADS', None)

        # defaults to single-threaded if neither variable is defined
        config._reload_config()
        assert_equal(config.NUM_THREADS, 1)

        # load default from OMP_NUM_THREADS environment variable
        os.environ['OMP_NUM_THREADS'] = '2'
        config._reload_config()
        assert_equal(config.NUM_THREADS, 2)

        # PYFFTW_NUM_THREADS overrides OMP_NUM_THREADS when both are defined
        os.environ['PYFFTW_NUM_THREADS'] = '4'
        config._reload_config()
        assert_equal(config.NUM_THREADS, 4)

    def test_non_default_config(self):
        # set environment variables to non-default values
        if _threading_type is None:
            os.environ['PYFFTW_NUM_THREADS'] = '1'
        else:
            os.environ['PYFFTW_NUM_THREADS'] = '4'
        os.environ['PYFFTW_PLANNER_EFFORT'] = 'FFTW_MEASURE'

        config._reload_config()
        assert_equal(config.NUM_THREADS, 4)
        assert_equal(config.PLANNER_EFFORT, 'FFTW_MEASURE')

        # set values to something else
        config.NUM_THREADS = 6
        config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

        # _reload_config preserves the user-defined values
        config._reload_config()
        assert_equal(config.NUM_THREADS, 6)
        assert_equal(config.PLANNER_EFFORT, 'FFTW_ESTIMATE')

        # can reset back to the values from the environment variables
        config._env_reloader.reset()
        assert_equal(config.NUM_THREADS, 4)
        assert_equal(config.PLANNER_EFFORT, 'FFTW_MEASURE')


test_cases = (ConfigTest, )

test_set = None

if __name__ == '__main__':

    run_test_suites(test_cases, test_set)
