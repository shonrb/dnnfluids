#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from lib import load_simulation_data, generate_dataset, save_simulation_data, parse_scene_json

def main():
    parser = ArgumentParser(prog="dataset.py")
    parser.add_argument("scene_path", type=str)
    parser.add_argument("save_path", type=str)

    args = parser.parse_args()

    with open(args.scene_path) as f:
        scene = parse_scene_json(f.read())

    dataset = generate_dataset(scene, verbose=True)
    save_simulation_data(dataset, args.save_path)

if __name__ == "__main__":
    main()

