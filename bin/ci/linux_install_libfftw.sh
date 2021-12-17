#!/bin/bash
if [ uname -m = "aarch64" ]
then
    apk add fftw-dev
else
    yum install -y fftw-devel
fi
