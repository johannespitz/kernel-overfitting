#!/bin/bash
#SBATCH --job-name=MyTraining
#SBATCH --nodes=1
#SBATCH --cpus=3
#SBATCH --gres=gpu:1,VRAM:20G
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH -o "slurm_jobs/slurm-%j.out"

python -u Figure1.py
