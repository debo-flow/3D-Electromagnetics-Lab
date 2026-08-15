# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

## Milestone 1 — 3D FDTD Electromagnetic Wave Propagation
Implemented a robust 3D FDTD solver on a Yee-grid. Includes basic numerical boundaries (PEC), Gaussian source injection, CFL stability control, memory protection, and PyVista-based 3D visualization.

## Milestone 2 — PML Absorbing Boundary (Virtual Anechoic Environment)
Replaced basic PEC boundaries with Convolutional Perfectly Matched Layers (CPML). This numerical boundary minimizes artificial electromagnetic reflections.

## Milestone 3 — Half-Wave Dipole Antenna
Introduces the first physical antenna into the electromagnetic laboratory: a center-fed half-wave dipole.

## Milestone 4 — Near-Field Electromagnetic Analysis
Introduces quantitative near-field sampling, frequency-domain analysis, and energy-flow calculations directly from the FDTD grid.

## Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
Implements a scientifically rigorous transformation of sampled near-field data into an estimated far-field electromagnetic solution based on the equivalence principle.

## Milestone 6 — 3D Antenna Radiation Pattern
Converts the validated NF2FF transformation data into a complete 3D angular antenna radiation pattern overlaying the 3D PyVista computational grid.

## Milestone 7 — Gain, Directivity & Efficiency
Introduces quantitative antenna performance metrics (Total Radiated Power, Radiation Intensity, Directivity) integrated natively via 2D spherical solid-angle evaluations over the Far-Field limits.

## Milestone 8 — Rectangular Microstrip Patch Antenna
Replaces wire-antenna modeling with complex planar structures featuring dielectrics, vertical lumped feeds, and finite ground boundaries conforming to standard Microstrip Transmission Line theory.

## Milestone 9 — Advanced Electromagnetic Materials
Generalizes the FDTD kernel to evaluate wave propagation through spatially varying environments featuring independent permittivity ($\epsilon_r$), permeability ($\mu_r$), and conductivity ($\sigma$).

## Milestone 10 — GPU Accelerated FDTD Engine
Accelerates the immensely computation-heavy 3D FDTD loop using array-based vectorization strictly parallelized across GPU hardware frameworks.
* **CuPy Implementation:** Employs the `cupy` library as a transparent, drop-in acceleration layer bypassing Python GIL constraints and serial CPU execution times.
* **Numerical Equivalency Preserved:** The original Numba CPU solver was definitively preserved as the strict mathematical baseline. GPU operations execute the exact same discrete Maxwell differences natively on VRAM, yielding validation RMS errors consistently under $\approx 10^{-7}$. 
* **Dynamic Backend Architecture:** Features Auto, CPU, and GPU switching modes. A failure to locate CUDA dependencies natively aborts safely to the CPU backend instead of terminating the laboratory instance. 
* **Hardware Benchmarking System:** Introduces an isolated CPU vs GPU performance dashboard logging the precise time-deltas to calculate hardware-specific execution speedups.
* **Requirements Update:** GPU execution now intrinsically requires `cupy-cuda11x` (or relative toolkit versions) installed inside the backend environment. If not provided, CPU fallback engages seamlessly.

## Project Roadmap
- [x] Milestone 1 — 3D FDTD electromagnetic wave propagation
- [x] Milestone 2 — PML absorbing boundary
- [x] Milestone 3 — Half-wave dipole antenna
- [x] Milestone 4 — Near-field analysis
- [x] Milestone 5 — Near-to-far-field transformation
- [x] Milestone 6 — 3D radiation pattern
- [x] Milestone 7 — Gain/directivity/efficiency
- [x] Milestone 8 — Rectangular microstrip patch antenna
- [x] Milestone 9 — Advanced electromagnetic materials
- [x] Milestone 10 — GPU acceleration
