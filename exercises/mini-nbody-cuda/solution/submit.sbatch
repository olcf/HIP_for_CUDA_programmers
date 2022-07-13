#!/bin/bash
#SBATCH -A stf007uanofn 
#SBATCH -J mini_nbody_hip
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1


srun -N1 -G1 -n1 -c1 --gpus-per-task 1 ./mini-nbody_hip 
