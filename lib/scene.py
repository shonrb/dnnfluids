# scene.py 
# - Data structure for flow problems
import numpy as np
from collections import namedtuple 
import pickle
import os
import json

Pair = namedtuple("Pair", [ 
    "x", "y" 
])

Rectangle = namedtuple("Rectangle", [
    "ax", "ay", "bx", "by"
])

Jet = namedtuple("Jet", [
    "bounds",
    "bounds_volumes",
    "force",
    "force_volumes",
])

Scene = namedtuple("Scene", [
    "key",
    "step_count",
    "delta_time",
    "bounds",
    "points",
    "time_points",
    "resolution",
    "volume_count",
    "viscosity",
    "jets"
])

def parse_pair(obj):
    return Pair(obj["x"], obj["y"])

def parse_rect(obj):
    return Rectangle(obj["x1"], obj["y1"], obj["x2"], obj["y2"])

def parse_scene_json(string, name="scene"):
    kv = json.loads(string)

    # Extract mesh
    resolution = parse_pair(kv["resolution"])
    bounds = parse_rect(kv["bounds"])
    width = bounds.bx - bounds.ax
    height = bounds.by - bounds.ay
    points = Pair(
        np.linspace(bounds.ax, bounds.bx, resolution.x, dtype=np.float32),
        np.linspace(bounds.ay, bounds.by, resolution.y, dtype=np.float32)    
    )

    # Extract time domain
    time_points = np.fromfunction(
        lambda i: (i + 1) * kv["delta_time"],
        (kv["step_count"],),
        dtype=np.float32
    )

    # Extract force addition
    def make_jet(j):
        b = parse_rect(j["bounds"])
        f = parse_pair(j["force"])
        bv = Rectangle(
            int(resolution.x * (b.ax - bounds.ax) / width ),
            int(resolution.y * (b.ay - bounds.ay) / height ),
            int(resolution.x * (b.bx - bounds.ax) / width ),
            int(resolution.y * (b.by - bounds.ay) / height )
        )
        vf = Pair(
            f.x * resolution.x / width,
            f.y * resolution.y / height,
        )
        return Jet(b, bv, f, vf)

    jets = [make_jet(j) for j in kv["jets"]]
 
    return Scene(
        key          = name,
        step_count   = kv["step_count"],
        delta_time   = kv["delta_time"],
        bounds       = bounds,
        points       = points,
        time_points  = time_points,
        resolution   = resolution,
        volume_count = resolution.x * resolution.y,
        viscosity    = kv["viscosity"],
        jets         = jets
    )

