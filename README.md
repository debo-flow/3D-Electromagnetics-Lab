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
Transforms the existing validation system into a rigorous, reproducible numerical verification framework to prove the mathematical integrity of the FDTD kernel against free-space and interface reflections.

## Milestone 12 — Dispersive Electromagnetic Materials
Extends the material system so that electromagnetic properties can depend heavily on frequency.
* **Debye Model Integration:** Fully implements the Debye dispersion relation $\epsilon(\omega) = \epsilon_\infty + \frac{\epsilon_s - \epsilon_\infty}{1 + j\omega\tau}$ dynamically.
* **Auxiliary Differential Equation (ADE):** Time-domain equations were entirely restructured to track internal polarization arrays ($P_x, P_y, P_z$). The Yee-grid couples these variables continuously into Ampere's law, meaning fields evolve natively against dispersive friction without requiring offline fake constants.
* **Material Dispersion Analyzer:** New UI tab calculates the analytical Real ($\epsilon'$) and Imaginary ($\epsilon''$) permittivity constraints across wideband frequency sweeps before executing simulations.
* **GPU & PML Compatibility:** The heavy memory overhead of storing previous field states and 3D polarization states is handled implicitly by the GPU CuPy execution. CPML naturally absorbs heavily attenuated dispersive waves exactly as it does free-space waves.

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
- [x] Milestone 12 — Dispersive electromagnetic materials
