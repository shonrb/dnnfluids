import pyopencl as cl
import pyopencl.array as cla
from reikna.fft import FFT 
from reikna.core import Annotation, Type, Transformation, Parameter
from reikna.cluda import ocl_api
import numpy as np
from os import path

import matplotlib.pyplot as plt

SCALING_FACTOR = 0.2

class FftFluidSolver:

    def __init__(self, scene):
        self.scene = scene
        self.context = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        self.time = 0.0

        print(self.context.get_info(cl.context_info.DEVICES))

        # Kernels
        here = path.dirname(path.abspath(__file__))
        p = path.join(here, "device.cl")
        with open(p) as f:
            src = f.read()
            self.dev = cl.Program(self.context, src).build()
        
        # Fields
        self.shape = (self.scene.resolution.x, self.scene.resolution.y)
        f = np.zeros(self.shape, dtype=np.float32)
        c = np.zeros(self.shape, dtype=np.complex64)
        
        self.velocity_x      = cla.to_device(self.queue, f)
        self.velocity_y      = cla.to_device(self.queue, f)
        self.pressure        = cla.to_device(self.queue, f)

        self.velocity_x_fourier = cla.to_device(self.queue, c)
        self.velocity_y_fourier = cla.to_device(self.queue, c)
        self.pressure_fourier   = cla.to_device(self.queue, c)
        
        # External force from jets as a single matrix
        ex_x = f
        ex_y = f.copy()
        for jet in self.scene.jets:
            ax, ay, bx, by = jet.bounds_volumes
            ex_x[ax:bx, ay:by] += jet.force_volumes.x * SCALING_FACTOR 
            ex_y[ax:bx, ay:by] += jet.force_volumes.y * SCALING_FACTOR

        self.external_x = cla.to_device(self.queue, ex_x)
        self.external_y = cla.to_device(self.queue, ex_y)

        # Fast Fourier Transform
        self.fft_api         = ocl_api()
        self.fft_thread      = self.fft_api.Thread(self.queue)
        self.fft_calculation = FFT(Type(np.complex64, self.shape))
        self.fft             = self.fft_calculation.compile(self.fft_thread)

        # Properties in 32 bits
        self.delta_time = np.float32(self.scene.delta_time)
        self.viscosity  = np.float32(self.scene.viscosity)

        # Preliminary step
        self.step()

    
    @property
    def current_x_velocity(self):
        return self.velocity_x.get()

    @property
    def current_y_velocity(self):
        return self.velocity_y.get()

    @property
    def current_pressure(self):
        return self.pressure.get()

    def step(self):
        # External force
        self.dev.add_force(
            self.queue, self.shape, None,
            self.external_x.data, self.external_y.data, 
            self.velocity_x.data, self.velocity_y.data,
            self.delta_time
        )
        # Advection 
        self.dev.advect(
            self.queue, self.shape, None,
            self.velocity_x.data, self.velocity_y.data,
            self.velocity_x_fourier.data, self.velocity_y_fourier.data,
            self.delta_time
        )
        # Forward FFT
        self.fft(self.velocity_x_fourier, self.velocity_x_fourier)
        self.fft(self.velocity_y_fourier, self.velocity_y_fourier)
        # Diffusion + mass conservation
        self.dev.diffuse_project(
            self.queue, self.shape, None,
            self.velocity_x_fourier.data, self.velocity_y_fourier.data,
            self.pressure_fourier.data,
            self.viscosity, self.delta_time
        )
        # Inverse FFT
        self.fft(self.velocity_x_fourier, self.velocity_x_fourier, inverse=True)
        self.fft(self.velocity_y_fourier, self.velocity_y_fourier, inverse=True)
        self.dev.to_real(
            self.queue, self.shape, None,
            self.velocity_x_fourier.data, self.velocity_y_fourier.data,
            self.velocity_x.data, self.velocity_y.data,
        )
        self.queue.finish()
        self.time += self.delta_time

    def reset(self):
        self.velocity_x.fill(np.float32(0.0))
        self.velocity_y.fill(np.float32(0.0))
        self.time = 0.0
