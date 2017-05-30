#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -w cn02
#SBATCH -t 00:30:00
#SBATCH -o out-1proc-1node-8thrd
#SBATCH -p batch
out=out-1proc-1node-8thrd
unset I_MPI_PMI_LIBRARY
echo "========== TEST 1 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 80 80 80 1000
echo "========== TEST 2 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 150 150 150 500
echo "========== TEST 3 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 500 500 500 200
echo "========== TEST 4 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 266 798 2394 200
echo "========== TEST 5 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 100 20000 256 200
echo "========== TEST 6 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 20000 100 256 200
echo "========== TEST 7 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 2372 2372 91 200
echo "========== TEST 8 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 81 81 78000 200
echo "========== TEST 9 ==========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 800 800 800 200
echo "========== TEST 10 =========">>$out
mpiexec.hydra -bootstrap slurm -l \
  -genv KMP_AFFINITY compact ./jacobi 1200 1200 1200 100
