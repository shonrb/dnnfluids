import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

IN_LAYER_SIZE = 3
OUT_LAYER_SIZE = 2
HIDDEN_LAYER_SIZE = 20
HIDDEN_LAYER_COUNT = 9

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        def layer(i, o):
            return nn.Sequential(
                nn.Linear(i, o),
                nn.Tanh()
            )

        self.i = layer(IN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.h = nn.Sequential(*[ 
            layer(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) 
            for _ in range(HIDDEN_LAYER_COUNT) 
        ])
        self.o = nn.Linear(HIDDEN_LAYER_SIZE, OUT_LAYER_SIZE)

    def forward(self, x):
        return self.o(self.h(self.i(x)))

class Pinn:

    def __init__(self, scene, path=None):
        """ Load a pretrained PINN if a path is given, 
            otherwise make a blank one for training 
        """
        self.scene = scene
        self.net = Net()

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

        self.opt = torch.optim.LBFGS(
            self.net.parameters(),
            lr               = 1,
            max_iter         = 50000,
            max_eval         = 50000,
            tolerance_grad   = 1e-05,
            tolerance_change = np.finfo(float).eps,
            history_size     = 50,
            line_search_fn   = "strong_wolfe"
        )

    @staticmethod
    def derivative(f, x):
        """ Helper function. Differentiate f WRT. x """
        val, *_ = torch.autograd.grad(
            f, x,
            grad_outputs=torch.ones_like(f),
            create_graph=True
        )
        return val

    def tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)

    def solve(self, x, y, t):
        """ Given a position <x,y> and time t, fine the velocity and pressure
        """
        input = torch.hstack((x, y, t))
        res   = self.net(input)
        psi   = res[..., 0:1]
        p     = res[..., 1:2]

        vx =  Pinn.derivative(psi,  y)
        vy = -Pinn.derivative(psi,  x)

        return vx, vy, p

    def loss(self, x, y, t, gvx, gvy):
        self.opt.zero_grad()

        vx, vy, p = self.solve(x, y, t)

        # Derivatives
        vx_x  = Pinn.derivative(vx,   x)
        vx_y  = Pinn.derivative(vx,   y)
        vx_x2 = Pinn.derivative(vx_x, x)
        vx_y2 = Pinn.derivative(vx_y, y)
        vx_t  = Pinn.derivative(vx,   t)

        vy_x  = Pinn.derivative(vy,   x)
        vy_y  = Pinn.derivative(vy,   y)
        vy_x2 = Pinn.derivative(vy_x, x)
        vy_y2 = Pinn.derivative(vy_y, y)
        vy_t  = Pinn.derivative(vy,   t)

        p_x = Pinn.derivative(p, x)
        p_y = Pinn.derivative(p, y)

        # du/dt + N[u,p]
        resx = vx_t + vx*vx_x+vy*vx_y + p_x - self.scene.viscosity*(vx_x2+vx_y2) 
        resy = vy_t + vx*vy_x+vy*vy_y + p_y - self.scene.viscosity*(vy_x2+vy_y2) 

        loss_data = (
            torch.sum(torch.square(gvx - vx)) +
            torch.sum(torch.square(gvy - vy))
        )
        loss_physics = (
            torch.sum(torch.square(resx)) +
            torch.sum(torch.square(resy))
        )

        loss_total = loss_data + loss_physics
        loss_total.backward()
        return loss_total

    def train(self, gx, gy, gt, gvx, gvy):
        SHOW_EVERY = 1
        steps = 0

        def step():
            nonlocal steps
            steps += 1
            loss = self.loss(gx, gy, gt, gvx, gvy)
            if steps % SHOW_EVERY == 0:
                print(f"Step {steps} with loss {loss} : {datetime.now()}", flush=True)
            return loss

        self.opt.step(step)

    def save(self, path):
        params = self.net.state_dict()
        torch.save(params, path)

