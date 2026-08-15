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
Accelerates the immensely computation-heavy 3D FDTD loop using array-based vectorization strictly parallelized across GPU hardware frameworks using the `cupy` library as a transparent acceleration layer.

## Milestone 11 — Advanced Numerical Validation & Benchmarking
Transforms the existing validation system into a rigorous, reproducible numerical verification framework to prove the mathematical integrity of the FDTD kernel.
* **Analytical Plane-Wave Validation:** Evaluates a uniform 1D plane wave passing through probes to numerically measure phase velocity ($v = \Delta z / \Delta t$) and intrinsic wave impedance ($\eta = |E_x| / |H_y|$). Assesses discretization-induced offsets natively compared to vacuum constants ($c$ and $Z_0$).
* **Boundary & Material Validation:** Verifies exact reflection mechanics natively processed by the localized integration coefficients. Accurately duplicates the theoretical Fresnel reflection coefficient ($R_{th} \approx 0.333$ for $\epsilon_r = 4.0$) entirely dynamically without hard-coded material math.
* **CPML Reflection Verification:** Isolates post-interface transmitted waves traveling into the boundary, actively verifying that boundary reflection decay coefficients exceed $-60$ dB, proving true anechoic absorption.
* **Precision & Architectural Consistency:** Computes strict Maximum Absolute and RMS Error constraints bounding Numba CPU computations (float64) against CuPy GPU executions (float32), documenting error constraints safely within negligible boundaries ($< 10^{-5}$). 

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
- [x] Milestone 11 — Advanced numerical validation & benchmarking
