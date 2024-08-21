# generate.py
# - Ground truth dataset generation
from collections import namedtuple
import numpy as np

SimulationData = namedtuple("SimulationData", [
    "velocity_x",
    "velocity_y",
    "pressure",
    "max_abs"
])

def generate_dataset(scene, override_frames=None, verbose=False):
    import phi.flow as pf

    frames = scene.step_count if override_frames is None else override_frames

    # Init fields
    ax, ay, bx, by = scene.bounds
    scene_bounds = pf.Box(x=(ax, bx), y=(ay, by))
    bounds = {
        "x"             : scene.resolution.x, 
        "y"             : scene.resolution.y,
        "bounds"        : scene_bounds,
        "extrapolation" : pf.extrapolation.PERIODIC
    }
    velocity = pf.CenteredGrid((0.0, 0.0), **bounds)
    pressure = pf.CenteredGrid(0.0, **bounds)

    # External force buffers
    external = []
    for jet in scene.jets:
        ax, ay, bx, by = jet.bounds
        bound = pf.Box(x=(ax, bx), y=(ay, by))
        force = (jet.force_volumes.x, jet.force_volumes.y)
        source = force * pf.resample(bound, to=velocity, soft=True) * scene.delta_time 
        external.append(source)

    # Field buffers
    def new_buffer():
        return np.zeros(
            (frames, scene.resolution.x, scene.resolution.y),
            np.float32
        )
    velocity_x_buffer = new_buffer()
    velocity_y_buffer = new_buffer()
    pressure_buffer   = new_buffer()

    # Absolute max values
    ma = 0

    for step in range(frames):
        if verbose and step % 10 == 0:
            print(f"Step {step} done")

        # Solve NS
        for force in external:
            velocity += force
        velocity = pf.advect.mac_cormack(
            velocity, velocity, scene.delta_time
        )
        velocity = pf.diffuse.implicit(
            velocity, scene.viscosity, scene.delta_time
        )
        velocity, pressure = pf.fluid.make_incompressible(
            velocity, (), pf.Solve(x0=pressure)
        )

        # Save to np buffers
        v = velocity.values.numpy("x,y,vector")
        ma = max(ma, np.max(np.abs(v)))
        velocity_x_buffer[step] = v[..., 0]
        velocity_y_buffer[step] = v[..., 1]
        pressure_buffer[step]   = pressure.uniform_values().numpy("x,y")

    return SimulationData(
        velocity_x_buffer, 
        velocity_y_buffer, 
        pressure_buffer,
        ma
    )

def save_simulation_data(data, path):
    np.savez_compressed(
        path,
        vx = data.velocity_x,
        vy = data.velocity_y,
        p  = data.pressure,
        ma = data.max_abs
    )

def load_simulation_data(path):
    d = np.load(path)
    return SimulationData(d["vx"], d["vy"], d["p"], d["ma"])

