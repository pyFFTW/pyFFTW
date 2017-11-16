#!/bin/bash

# build pyfftw while ignoring one precision, then run tests.

# fail if any command below fails
set -e

# move relative to directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/..

for v in PYFFTW_IGNORE_DOUBLE PYFFTW_IGNORE_SINGLE PYFFTW_IGNORE_LONG
do
    # declare variable based on the value of `v`
    declare "${v}"=1
    export "${v}"
    # test shouldn't fail because pyfftw.c is not there yet
    rm pyfftw/pyfftw.c || true
    python setup.py build_ext -i
    python setup.py test
    unset "${v}"
done
