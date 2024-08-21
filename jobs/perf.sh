#!/bin/bash
#SBATCH --job-name=perf
#SBATCH --output=perf.log
#SBATCH --partition=teaching-gpu --gres=gpu
#SBATCH --account=teaching
#SBATCH --time=4:00:00

module load init_opencl
module load nvidia/sdk
module load nvidia/cudnn
module load anaconda/python-3.10.9/2023.03
conda activate proj2
srun python3 analysis/solvertodataset.py 

