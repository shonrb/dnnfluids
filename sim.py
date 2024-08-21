#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from lib import FftFluidSolver, PinnFluidSolver, DeepFluidSolver, parse_scene_json

def main():
    parser = ArgumentParser(prog="sim.py")
    parser.add_argument("scene_path", type=str)
    parser.add_argument("model", type=str, choices=["fftfluid", "pinnfluid", "deepfluid"])
    parser.add_argument("-w", "--weights_path", type=str, default=None)

    args = parser.parse_args()

    with open(args.scene_path) as f:
        scene = parse_scene_json(f.read())

    solver = None
    if args.model == "fftfluid":
        solver = FftFluidSolver(scene)
    elif args.weights_path is not None:
        if args.model == "pinnfluid":
            solver = PinnFluidSolver(scene, args.weights_path)
        if args.model == "deepfluid":
            solver = DeepFluidSolver(scene, args.weights_path)
    else:
        print(f"Need a weights file for {args.model}")
        return

    fig, (ax_v, ax_x, ax_y) = plt.subplots(1, 3)
    ax_v.set_aspect('equal', adjustable='box')
    ax, ay, bx, by = scene.bounds
    extent = [ax, bx, ay, by]

    xs = np.tile(scene.points.x, scene.resolution.y)
    ys = np.repeat(scene.points.y, scene.resolution.x)

    def closure(*_):
        for ax in [ax_v, ax_x, ax_y]:
            ax.clear()
        vx = solver.current_x_velocity.T
        vy = solver.current_y_velocity.T
        ax_v.quiver(xs, ys, vx.flatten(), vy.flatten(), color="darkblue")
        ax_x.imshow(vx, extent=extent, cmap="seismic")
        ax_y.imshow(vy, extent=extent, cmap="seismic")
        solver.step()

    _ = anim.FuncAnimation(fig, closure, scene.step_count-1, interval=0, repeat=False)
    plt.show()
    
if __name__ == "__main__":
    main()

