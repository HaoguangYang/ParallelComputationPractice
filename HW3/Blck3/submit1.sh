#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -n 27
#SBATCH -N 5
#SBATCH -c 4
#SBATCH -w cn02
#SBATCH -t 00:30:00
#SBATCH -o out
#SBATCH -p batch
unset I_MPI_PMI_LIBRARY
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 1000 1000 1000 20
