#!/bin/sh
#SBATCH -t2-12
#SBATCH --mail-user=f.lehue@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT_80
#SBATCH --output=slurm-%A.out
#SBATCH --mem-per-cpu=12G
#SBATCH -c 1
#SBATCH --partition=epyc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 $1 

