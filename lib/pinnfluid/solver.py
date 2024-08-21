from lib.pinnfluid.model import Pinn
import numpy as np
import torch

class PinnFluidSolver:
    
    def __init__(self, scene, model_path):
        self.scene = scene
        self.pinn = Pinn(scene, model_path)
       
        self.time_step = 0
        self.x  = self.pinn.tensor(np.tile(scene.points.x, scene.resolution.y).reshape(-1,1))  
        self.y  = self.pinn.tensor(np.repeat(scene.points.y, scene.resolution.x).reshape(-1,1))
        self.t  = self.pinn.tensor(np.ones((scene.volume_count, 1), dtype=np.float32))
        self.vx, self.vy, self.p, _, _ = self.pinn.solve(self.x, self.y, self.t*0)
        self.shape = (scene.resolution.y, scene.resolution.x)
        self.step()

    def get_field(self, f):
        return f.cpu().detach().numpy().reshape(self.shape)

    @property
    def current_x_velocity(self):
        return self.get_field(self.vx)

    @property
    def current_y_velocity(self):
        return self.get_field(self.vy)

    @property
    def current_pressure(self):
        return self.get_field(self.p)

    def step(self):
        self.time_step += self.scene.delta_time
        self.vx, self.vy, self.p, _, _ = self.pinn.solve(self.x, self.y, self.t*self.time_step)

    def reset(self):
        self.time_step = 0

