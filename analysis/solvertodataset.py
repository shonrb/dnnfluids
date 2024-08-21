import parent
import sys
import os
from lib import FftFluidSolver, PinnFluidSolver, DeepFluidSolver, SimulationData, save_simulation_data, generate_dataset, parse_scene
import numpy as np

scene = parse_scene(open(parent.PATH + "/scenes/double_jet128.json").read())
N = 250

def gen(arg, solve):
    vx, vy, p = (
        np.zeros(
            (N, scene.resolution.x, scene.resolution.y),
            np.float32
        ) for _ in range(3)
    )

    for i in range(N):
        print(f"{arg} step {i}")
        vx[i] = solve.current_x_velocity
        vy[i] = solve.current_y_velocity
        if hasattr(solve, "current_pressure"):
            p[i] = solve.current_pressure
        solve.step()

    ma = np.max(np.abs([vx, vy]))
    ds = SimulationData(vx, vy, p, ma)
    save_simulation_data(ds, path.HERE + f"/data/dataset{arg}.npz")

gen("fftfluid",  FftFluidSolver(scene))
gen("pinnfluid", PinnFluidSolver(scene, path.PARENT + "/data/dj128pinn.pt"))
gen("deepfluid", DeepFluidSolver(scene, path.PARENT + "/data/dj128df.pt"))
    
ds = generate_dataset(scene, override_frames=N, verbose=True)
save_simulation_data(ds, path.HERE + f"/data/datasetground.npz")

