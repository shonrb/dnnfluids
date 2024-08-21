import path
import sys
import os
import timeit
import numpy as np
from lib import DeepFluidSolver, FftFluidSolver, PinnFluidSolver, parse_scene_json

RUNS = 100

for solver in ["fftfluid", "pinnfluid", "deepfluid"]:
    per_res = np.ones((5,), dtype=float) * np.inf
    for i, res in enumerate(["16", "32", "64", "128", "256"]):
        json = open(path.PARENT + f"/scenes/double_jet{res}.json").read()
        scene = parse_scene_json(json)

        match solver:
            case "fftfluid":
                solve = FftFluidSolver(scene)
            case "pinnfluid":
                solve = PinnFluidSolver(scene, path.PARENT + f"/data/dj{res}pinn.pt")
            case "deepfluid":
                solve = DeepFluidSolver(scene, path.PARENT + f"/data/dj{res}df.pt")

        print(f"{solver} : {i}", flush=True)

        for j in range(RUNS):
            t = 0.0
            for k in range(150):
                t += timeit.timeit(lambda: solve.step(), number=1)
            per_res[i] = min(per_res[i], t)
            solve.reset()
            print(f"run {j} done", flush=True)

        del solve

    np.savetxt(path.HERE + f"/data/{solver}growth", per_res)
    print("done")

