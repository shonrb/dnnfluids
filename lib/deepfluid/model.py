# deepfluid/model.py
# - Model code for DF based solver
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
import datetime

SB_COUNT = 5 
BETA1 = 0.5
BETA2 = 0.999
LR_MAX = 0.0001 
LR_MIN = 0.0000025
PRINT_EACH = 1

class Net(nn.Module):
    
    def __init__(self, scene):
        super().__init__()

        def layer_init(m):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # Feature map dimensions
        w = scene.resolution.x
        h = scene.resolution.y
        dmax = max(w, h)
        self.q = int(np.log2(dmax) - 3)
        self.q2 = 2 ** self.q
        self.fw = int(w / self.q2)
        self.fh = int(h / self.q2)
        
        # Projection layer
        m = self.fw * self.fh * 128
        self.project = nn.Linear(1, m).apply(layer_init)
        
        # BB x q layer
        def sb():
            return nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1).apply(layer_init),
                nn.LeakyReLU(0.2)
            )
        def bb():
            return nn.Sequential(*[
                sb() for _ in range(SB_COUNT)
            ])
        self.bb = nn.ModuleList([
            bb() for _ in range(self.q)
        ])
        self.x2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output Layer
        self.out_conv = nn.Conv2d(128, 1, 3, padding=1).apply(layer_init)

    def forward(self, t):
        curr = torch.reshape(
            self.project(t), 
            (-1, 128, self.fw, self.fh)
        )

        for bb in self.bb:
            curr = bb(curr) + curr
            curr = self.x2(curr)
        
        return self.out_conv(curr)

class DeepFluidNet:
    
    def __init__(self, scene, path=None):
        """ Load a pretrained Deep Fluid network if a path is given, 
            otherwise make a blank one for training 
        """
        self.net = Net(scene)

        # Try to use GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Running on GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        # Load or init
        if path is not None:
            w = torch.load(path, map_location=self.device)
            self.net.load_state_dict(w)
            self.net.eval()
        else:
            self.net.train()

        self.net.to(self.device)

        self.opt = optim.Adam(
            self.net.parameters(), 
            lr    = LR_MAX,
            betas = (BETA1, BETA2)
        )
        
    @staticmethod
    def jacobian(v):
        """ Jacobian matrix of velocity field """
        vx = v[:, 0, ...]
        vy = v[:, 1, ...]
        
        vx_x = vx[..., 1:, :] - vx[..., :-1, :] 
        vy_x = vy[..., 1:, :] - vy[..., :-1, :] 
        vx_y = vx[..., 1:]    - vx[..., :-1] 
        vy_y = vy[..., 1:]    - vy[..., :-1] 

        # Extend the final index
        vx_x = torch.cat((vx_x, torch.unsqueeze(vx_x[..., -1, :], 1)), dim=1)
        vy_x = torch.cat((vy_x, torch.unsqueeze(vy_x[..., -1, :], 1)), dim=1)
        vx_y = torch.cat((vx_y, torch.unsqueeze(vx_y[..., -1], 2)), dim=2)
        vy_y = torch.cat((vy_y, torch.unsqueeze(vy_y[..., -1], 2)), dim=2)

        return torch.cat((vx_x, vx_y, vy_x, vy_y))

    def tensor(self, x):
        return torch.tensor(x).to(self.device)

    def solve(self, t):
        """ Given a time, solve for a velocity field 2 x W x H  """
        stream = self.net(self.tensor(t))
        
        # vx = d(stream) / dy
        # vy = -d(stream) / dx
        vx = stream[..., 1:]     - stream[..., :-1]
        vy = stream[..., :-1, :] - stream[..., 1:, :] 
        
        # Extend the final index
        vx = torch.cat([vx, torch.unsqueeze(vx[..., -1],    axis=3)], axis=3)
        vy = torch.cat([vy, torch.unsqueeze(vy[..., -1, :], axis=2)], axis=2)

        return torch.cat([vx, vy], axis=1)

    def train_step(self, input, ground):
        self.opt.zero_grad()

        ground_v = self.tensor(ground)
        ground_j = DeepFluidNet.jacobian(ground_v)

        pred_v = self.solve(input)
        pred_j = DeepFluidNet.jacobian(pred_v)

        loss_v = torch.mean(torch.abs(pred_v - ground_v))
        loss_j = torch.mean(torch.abs(pred_j - ground_j))

        loss = loss_v + loss_j
        loss.backward()
        self.opt.step()

        return loss

    def train(self, data_loader, epochs):
        lr = CosineAnnealingLR(self.opt, epochs, LR_MIN)

        for i in range(epochs):
            total_loss = 0
            for (input, label) in data_loader:
                total_loss += self.train_step(input, label)
            lr.step()
            
            if (i+1) % PRINT_EACH == 0:
                msg1 = f"Step {i+1}/{epochs}"
                msg2 = f" with loss {total_loss}: {datetime.datetime.now()}"
                print(msg1 + msg2, flush=True)

    def save(self, path):
        p = self.net.state_dict()
        torch.save(p, path)

