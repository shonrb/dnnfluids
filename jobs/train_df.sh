#!/bin/bash
#SBATCH --job-name=df
#SBATCH --output=df.log
#SBATCH --partition=teaching-gpu --gres=gpu
#SBATCH --account=teaching
#SBATCH --time=48:00:00

module load nvidia/sdk
module load nvidia/cudnn
module load anaconda/python-3.10.9/2023.03
conda activate proj2
srun python3 train.py scenes/double_jet128 data/dj128data.npz data/dj128df.pt

