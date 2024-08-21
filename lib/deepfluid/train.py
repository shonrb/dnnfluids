# deepfluid/train.py
# - Training code for DF based solver
from lib.deepfluid.model import DeepFluidNet
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

EPOCHS = 50000

def train_deep_fluid(scene, data, save_path):
    vx = data.velocity_x
    vy = data.velocity_y

    # Reshape for training data
    frames     = np.linspace(-1.0, 1.0, scene.step_count).reshape(-1, 1)
    velocities = np.array([vx, vy]).transpose((1, 0, 2, 3))

    # Normalise  
    velocities /= data.max_abs
    
    frames = torch.Tensor(frames)
    velocities = torch.Tensor(velocities)
    dataset = TensorDataset(frames, velocities)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DeepFluidNet(scene)
    model.train(loader, EPOCHS)
    model.save(save_path)

