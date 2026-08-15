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
Extends the material system so that electromagnetic properties can depend heavily on frequency using an Auxiliary Differential Equation (ADE) Debye model integrated continuously into Ampere's law.

## Milestone 13 — Anisotropic Electromagnetic Materials
Extends the material system to support diagonal tensor permittivity matrices ($\varepsilon̿$), enabling mathematically rigorous direction-dependent wave propagation (Birefringence).
* **Diagonal Tensor Permittivity:** The Maxwell curl-solver was expanded to execute three fundamentally independent electric field updates per cell. $\epsilon_x, \epsilon_y,$ and $\epsilon_z$ variables are isolated to compute their respective $E_x, E_y, E_z$ fields aligned exactly with the physical staggering of the Yee-grid.
* **Birefringence Validation:** An advanced material testing module was implemented verifying that orthogonally polarized waves ($E_x$ vs $E_y$) correctly exhibit discrete phase velocity variations ($v_x = c / \sqrt{\varepsilon_x} \neq v_y = c / \sqrt{\varepsilon_y}$) when passing through anisotropic media.
* **Patch Substrate Compatibility:** Microstrip patch antennas now natively support anisotropic tensor substrates (e.g., Sapphire or composite weaves), dynamically modifying fringing fields and resonance dependent on geometric planar alignment.
* **Scientific Limitations:** Implementing full off-diagonal tensor components ($\epsilon_{xy}$, $\epsilon_{xz}$) requires complex multi-point spatial interpolations across the staggered Yee-grid. To preserve broad computational stability and strictly avoid sub-cell artifacting, the tensor currently mandates diagonal alignment.

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
- [x] Milestone 13 — Anisotropic electromagnetic materials
