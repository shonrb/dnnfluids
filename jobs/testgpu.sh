#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.log
#SBATCH --partition=teaching-gpu --gres=gpu
#SBATCH --account=teaching
#SBATCH --time=48:00:00

module load nvidia/sdk
module load nvidia/cudnn
module load anaconda/python-3.10.9/2023.03
conda activate proj2
nvcc --version
nvidia-smi
python3 -c "import torch; print(torch.cuda.device_count())"
