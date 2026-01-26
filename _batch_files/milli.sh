#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=CPU
#SBATCH --exclude=node47
set -x

# t=$SLURM_ARRAY_TASK_ID
python process_millionsong.py