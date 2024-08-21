# deepfluid/solver.py
# - Fluid solver based on Deep Fluids Generative Networks
from lib.deepfluid.model import DeepFluidNet
import torch
import numpy as np

class DeepFluidSolver:

    def __init__(self, scene, path):
        self.model = DeepFluidNet(scene, path)
        self.scene = scene
        self.dt = 2.0 / self.scene.step_count
        self.time = -1.0
        self.solve_current()

    @property
    def current_x_velocity(self):
        return self.vx.detach().cpu().numpy().T

    @property
    def current_y_velocity(self):
        return self.vy.detach().cpu().numpy().T

    def step(self):
        self.time += self.dt
        self.solve_current()

    def reset(self):
        self.time = -1.0

    def solve_current(self):
        label = torch.tensor([self.time])
        r = self.model.solve(label)
        self.vx = r[0, 0, ...]
        self.vy = r[0, 1, ...]

