# Deep Learning for Real-Time Navier-Stokes Fluid Simulation
Code for a university research project which aims to test and analyse statistical methods for real-time fluid simulation.

![](./images/combined.gif)

Features:
- PyTorch implementations of two statistical fluid solvers.
- A PyOpenCL implementation of a numerical fluid solver for comparison.
- A PhiFlow training dataset generator.
- Slurm job scripts for training and running models.
- Various analysis/test scripts and example flow problems.

## Dependencies
- Python 3.10
- NumPy
- MatPlotLib
- PyTorch
- PyOpenCL
- Reikna
- PhiFlow

## Resources
[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations - M. Raissi, P. Perdikaris, G.E. Karniadakis](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)

Reference Implementation: https://github.com/maziarraissi/PINNs

[Deep Fluids: A Generative Network for Parameterized Fluid Simulations - B. Kim, V.C. Azevedo, N. Thuerey, T. Kim, M. Gross, B. Solenthaler](https://arxiv.org/pdf/1806.02071)

Reference Implementation: https://github.com/byungsook/deep-fluids

[A Simple Fluid Solver based on the FFT - J. Stam](https://www.researchgate.net/publication/2377059_A_Simple_Fluid_Solver_based_on_the_FFT)
