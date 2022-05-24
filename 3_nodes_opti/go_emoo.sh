#!/bin/bash
#SBATCH -t2-12
#SBATCH -c 1
export OMP_NUM_THREADS=%SLURM_CPUS_PER_TASK
mpirun -np $SLURM_NTASKS python3 $1 $2