# Data Analysis and Numerical Simulations

Welcome to my data analysis and numerical simulations repository. This collection contains projects and homework assignments focusing on advanced numerical methods, partial differential equations (PDEs), and physical systems. Below is an overview of the key projects.

## Contents

1. **Boundary Value Problems and Eigenfunction Analysis** (Homework 1 and 2)
   - **Homework 1**:
     - Solves boundary value problems for second-order differential equations using finite-difference methods.
     - Explores shooting methods to compute eigenvalues and eigenfunctions for various boundary conditions.
     - Conducts error analysis and convergence studies for numerical solutions.
   - **Homework 2**:
     - Analyzes physical systems using eigenfunction expansions.
     - Implements numerical techniques to compute energy eigenvalues and eigenfunctions of quantum systems.
     - Extends solutions to non-linear systems, exploring parameter dependencies and stability.

2. **Quantum Harmonic Oscillator** (Homework 3)
   - Solves the Schrödinger equation to compute eigenfunctions and eigenvalues for the quantum harmonic oscillator.
   - Explores nonlinear effects on eigenfunctions and eigenvalues using shooting methods.
   - Conducts convergence studies of numerical solvers and error analysis against exact solutions.

3. **Vorticity-Streamfunction Dynamics** (Homework 5)
   - Solves vorticity-streamfunction equations for fluid flow with periodic boundary conditions.
   - Implements numerical solvers like LU decomposition, BICGSTAB, GMRES, and FFT-based methods for the streamline equation.
   - Includes computational performance comparisons and dynamics visualizations for various initial vorticity configurations.

4. **Reaction-Diffusion Systems** (Homework 6)
   - Simulates the λ-ω reaction-diffusion system with periodic and no-flux boundary conditions.
   - Implements numerical methods such as FFTs for periodic boundaries and Chebyshev polynomials for no-flux boundaries.
   - Visualizes spiral wave solutions and explores stability and chaos for various initial conditions and system parameters.

## Technologies Used
- Python (NumPy, SciPy, Matplotlib)
- Fast Fourier Transform (FFT)
- Chebyshev Polynomials
- Iterative Solvers (LU Decomposition, BICGSTAB, GMRES)
- ODE Integration (solve_ivp)
- Shooting Methods for Boundary Value Problems

## Features
- Visualization of solutions for PDEs and dynamical systems.
- Comparative analysis of numerical methods in terms of accuracy and efficiency.
- Simulation of physical phenomena including fluid dynamics and quantum mechanics.
- Error and convergence studies for numerical solvers.
