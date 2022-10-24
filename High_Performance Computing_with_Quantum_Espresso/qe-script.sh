#!/bin/bash

# clone the project from GitHub and unzip
wget https://github.com/QEF/q-e/releases/download/qe-7.0/qe-7.0-ReleasePack.tgz
tar -zxvf ./qe-7.0-ReleasePack.tgz
cd ./qe-7.0

# module load the essential dependencies
module load intel-compiler/2021.4.0
module load intel-mpi/2021.4.0
module load intel-mkl/2021.4.0

# configure
./configure FC=mpiifort CC=mpiicc CFLAGS="-fast" --enable-openmp --with-scalapack=intel

# make
make pw