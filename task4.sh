#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --account=teaching
#SBATCH --job-name=ph510_task4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1

echo "--- 1 rank ---"
time mpirun -n 1 python3 vmc_bosons_parallel.py

echo "--- 2 ranks ---"
time mpirun -n 2 python3 vmc_bosons_parallel.py

echo "--- 4 ranks ---"
time mpirun -n 4 python3 vmc_bosons_parallel.py

echo "--- 8 ranks ---"
time mpirun -n 8 python3 vmc_bosons_parallel.py
