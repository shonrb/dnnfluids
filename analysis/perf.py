import path
import sys
import os
import cProfile
import timeit
import numpy as np
from lib import DeepFluidSolver, FftFluidSolver, PinnFluidSolver, parse_scene_json

FULL_RUNS = 200

scene = parse_scene_json(open(path.PARENT + "/scenes/double_jet128.json").read())

print("starting profiling", flush=True)

def profile(arg, solve):
    prof = cProfile.Profile()
    print("starting profile")

    for i in range(FULL_RUNS):
        print(f"run {i}", flush=True)
        for _ in range(scene.step_count):
            prof.enable()
            solve.step()
            prof.disable()
        solve.reset()

    prof.dump_stats(path.HERE + f"/data/{arg}full")
    print("done")

profile("fftfluid",  FftFluidSolver(scene))
profile("pinnfluid", PinnFluidSolver(scene, path.PARENT + "/data/dj128/pinn.pt"))
profile("deepfluid", DeepFluidSolver(scene, path.PARENT + "/data/dj128/df.pt"))


