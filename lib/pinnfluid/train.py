from lib.pinnfluid.model import Pinn
import torch
import numpy as np
import datetime

def boundary_condition_indices(scene):
    """ Extract the indices across a flat time*width*height 
        array corresponding to positions which lie on BCs
    """
    EXTENT   = 0.1 
    points   = [(x, y) for x in scene.points.x for y in scene.points.y]
    x_bounds = {scene.bounds.ax, scene.bounds.bx}
    y_bounds = {scene.bounds.ay, scene.bounds.by}

    # Extract indices which lie on the boundary conditions row major
    indices = []
    for i, (x, y) in enumerate(points):
        if x in x_bounds or y in y_bounds:
            indices.append(i)
            continue
        here = np.array([x, y])
        for jet in scene.jets:
            ax, ay, bx, by = jet.bounds
            x_in = x >= ax - EXTENT and x <= bx + EXTENT
            y_in = y >= ay - EXTENT and y <= by + EXTENT
            if x_in and y_in:
                indices.append(i)
                break

    # Extrapolate across time
    over_time = np.array([
        np.array(indices) + scene.volume_count * i
        for i in range(scene.step_count)
    ])

    return over_time.flatten()

def train_pinn_fluid(scene, data, save_path):
    """ Train a PINN on a given scene, and save it as a trained model """
    SAMPLING_RATE          = 0.01
    BOUNDARY_SAMPLING_RATE = 0.1
    torch.autograd.set_detect_anomaly(True) 

    # Sample over bcs
    bcs            = boundary_condition_indices(scene)
    bc_count       = len(bcs)
    resample_count = int(bc_count * BOUNDARY_SAMPLING_RATE)
    samples        = np.random.choice(bc_count, resample_count, replace=False)
    bc_indices     = bcs[samples]
    
    # Sample over rest of domain
    point_count     = scene.volume_count*scene.step_count
    resample_count  = int(point_count * SAMPLING_RATE)
    uniform_indices = np.random.choice(point_count, resample_count, replace=False)

    # Combine
    union            = np.concatenate([bc_indices, uniform_indices])
    training_indices = np.unique(union)
    np.random.shuffle(training_indices)
 
    # Transform training dataset into tensors
    pinn = Pinn(scene)
    
    def training_data(arr):
        return pinn.tensor(
            arr
            .flatten()
            .reshape((-1, 1))
            [training_indices]
        )
    
    # Positions
    xs       = np.tile(scene.points.x, scene.resolution.y)
    ys       = np.repeat(scene.points.y, scene.resolution.x)
    xs_t     = np.tile(xs, (scene.step_count))
    ys_t     = np.tile(ys, (scene.step_count))
    ground_x = training_data(xs_t)
    ground_y = training_data(ys_t)

    # Time
    ts       = np.repeat(scene.time_points, (scene.volume_count))
    ground_t = training_data(ts)

    # Velocities
    vx_t      = data.velocity_x 
    vy_t      = data.velocity_y
    ground_vx = training_data(vx_t)
    ground_vy = training_data(vy_t)

    # Train in one batch
    pinn.train(
        ground_x,
        ground_y,
        ground_t,
        ground_vx,
        ground_vy
    )
    
    print("Done")
    pinn.save(save_path)

