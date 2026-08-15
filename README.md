# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

## Milestone 1 — 3D FDTD Electromagnetic Wave Propagation
Implemented a robust 3D FDTD solver on a Yee-grid. Includes basic numerical boundaries (PEC), Gaussian source injection, CFL stability control, memory protection, and PyVista-based 3D visualization.

## Milestone 2 — PML Absorbing Boundary (Virtual Anechoic Environment)
Replaced basic PEC boundaries with Convolutional Perfectly Matched Layers (CPML). This numerical boundary minimizes artificial electromagnetic reflections, effectively creating a virtual anechoic chamber for accurate near-field analysis. Includes dynamic reflection coefficient calculation and energy absorption tracking.

## Milestone 3 — Half-Wave Dipole Antenna
Introduces the first physical antenna into the electromagnetic laboratory: a center-fed half-wave dipole.
* **Overview:** A half-wave dipole is a fundamental reference antenna used widely in RF engineering. It serves as the perfect baseline for understanding electromagnetic radiation.
* **FDTD Representation:** The dipole is modeled consistently on the Yee-grid as perfectly conducting (PEC) cells aligned along the Z-axis.
* **Center Feed Implementation:** A small feed gap at the center is excited using a time-dependent "soft source" (additive current-density approximation) with configurable waveforms to simulate actual antenna current.

## Milestone 4 — Near-Field Electromagnetic Analysis
Introduces quantitative near-field sampling, frequency-domain analysis, and energy-flow calculations directly from the FDTD grid.
* **Near-Field Data Acquisition:** Evaluates the spatial distribution of E and H components without synthetic interpolation.
* **Frequency Analysis (FFT):** Calculates the continuous-time frequency spectrum E(f) and phase using a rectangular window optimal for FDTD transient pulses.
* **Poynting Vector (S):** Implements both the instantaneous ($S = E \times H$) and the strictly time-averaged ($\langle S \rangle$) Poynting vector over a configurable later-time window to analyze actual energy-flow density.

## Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
Implements a scientifically rigorous transformation of sampled near-field data into an estimated far-field electromagnetic solution.
* **Electromagnetic Equivalence Principle:** A virtual closed bounding box (Huygens surface) is drawn around the antenna. Equivalent electric ($J_s = \hat{n} \times H$) and magnetic ($M_s = -\hat{n} \times E$) surface currents are calculated on this boundary.
* **Frequency-Domain Transformation:** Time-domain tangential fields over the entire equivalence surface are converted to the frequency domain (via windowed FFT). Maintaining correct spatial *phase* relationships is critical for accurate constructive and destructive spatial interference calculations in the far field.

## Milestone 6 — 3D Antenna Radiation Pattern
Converts the validated NF2FF transformation data into a complete 3D angular antenna radiation pattern.
* **Spherical Angular Grid:** Fully configurable spatial mapping across $\theta$ (0°–180°) and $\phi$ (0°–360°), yielding high-resolution angular datasets directly from the numerical simulation.
* **Field Computations:** Computes the total far-field electric magnitude $|E_{far}| = \sqrt{|E_\theta|^2 + |E_\phi|^2}$ supporting both absolute scaling, global normalization, and dB logarithmic representations (with configurable numerical limits to prevent asymptotic singularities).
* **3D Visualization:** Maps the calculated normalized or dB data onto a true spherical coordinate geometry overlaying the physical computational grid using PyVista.
* **Polar Analytics:** Implements 2D principal-plane cuts, generating quantitative polar charts for evaluating E-plane and H-plane characteristics inherently tied to the Z-axis dipole rotation. 
* **Scientific Limitations:** Results describe the FDTD-derived angular distribution, heavily constrained by Yee-grid phase numerical dispersion, bounding box proximity to reactive near fields, and discrete temporal simulation lengths limiting precise FFT evaluation. Metrics like Directivity and absolute Antenna Efficiency are intentionally bypassed here.

## Project Roadmap

- [x] Milestone 1 — 3D FDTD electromagnetic wave propagation
- [x] Milestone 2 — PML absorbing boundary
- [x] Milestone 3 — Half-wave dipole antenna
- [x] Milestone 4 — Near-field analysis
- [x] Milestone 5 — Near-to-far-field transformation
- [x] Milestone 6 — 3D radiation pattern
- [ ] Milestone 7 — Gain/directivity/efficiency
- [ ] Milestone 8 — Patch antenna
- [ ] Milestone 9 — Advanced materials
- [ ] Milestone 10 — GPU acceleration
