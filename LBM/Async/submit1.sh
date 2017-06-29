#!/bin/bash
#SBATCH -J mpi_gpu_test
#SBATCH -n 2
#SBATCH -N 2
#SBATCH -c 24
#SBATCH -w cn15
#SBATCH -t 00:30:00
#SBATCH -o out
#SBATCH -p gpu
unset I_MPI_PMI_LIBRARY
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./LBM
