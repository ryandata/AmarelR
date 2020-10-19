#!/bin/bash
#SBATCH --partition=main          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=zipx001a       # Assign an short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=8                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --mem=2000                # Real memory (RAM) required (MB)
#SBATCH --time=00:30:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out  # STDOUT output file
#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)
cd /scratch/$USER
module purge
srun Rscript importdata.R
