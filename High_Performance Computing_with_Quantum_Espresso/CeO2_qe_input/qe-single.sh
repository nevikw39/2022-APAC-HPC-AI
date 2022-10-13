#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l ncpus=40
#PBS -l mem=190GB
#PBS -l software=qe
#PBS -l other=hyperthread
#PBS -l wd
#PBS -P jx00
#PBS -N QE-single

module load qe

export OMP_NUM_THREADS=1

mpirun -np 40 pw.x -npool 20 -ndiag 4 -inp CeO2.in > qe-single.txt 

