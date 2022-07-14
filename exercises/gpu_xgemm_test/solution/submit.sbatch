#!/bin/bash
#SBATCH -A stf007uanofn 
#SBATCH -J gpu_xgemm_hip
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1

module load rocm openblas/0.3.17-omp

srun -N1 -G1 -n1 -c1 --gpus-per-task 1 ./gpu_xgemm_hip
