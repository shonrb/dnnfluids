#!/bin/bash
#SBATCH --job-name=env
#SBATCH --output=env.log 
#SBATCH --partition=teaching-gpu --gres=gpu
#SBATCH --account=teaching
#SBATCH --time=1:00:00

module load init_opencl
module load nvidia/sdk/23.3
module load anaconda/python-3.10.9/2023.03
conda create -n proj2 python=3.10
conda config --add channels conda-forge
conda install -n base conda-forge::mamba
conda activate proj2
mamba install numpy
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install pyopencl
mamba install ocl-icd-system
mamba install reikna
mamba install matplotlib
