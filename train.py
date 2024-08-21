#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import pickle
from lib import load_simulation_data, train_pinn_fluid, train_deep_fluid, parse_scene_json

def main():
    parser = ArgumentParser(prog="train.py")
    parser.add_argument("scene_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("model", type=str, choices=["pinnfluid", "deepfluid"])
    parser.add_argument("save_path", type=str)

    args = parser.parse_args()

    with open(args.scene_path) as f:
        scene = parse_scene_json(f.read())

    data = load_simulation_data(args.data_path)

    if args.model == "pinnfluid":
        train_pinn_fluid(scene, data, args.save_path)
    if args.model == "deepfluid":
        train_deep_fluid(scene, data, args.save_path)

if __name__ == "__main__":
    main()

