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
* **Wavelength & Dipole Length:** The physical dipole length is dynamically calculated based on the chosen operating frequency to approximate L = λ/2.
* **Grid Resolution Importance:** Antenna behavior is highly sensitive to spatial discretization. The system automatically calculates the *Cells per Wavelength* ratio and warns against severe numerical dispersion if the resolution drops below safe limits.
* **PML Integration:** The antenna acts as an active energy injector, and the CPML boundaries perfectly absorb the outgoing radiated waves, demonstrating actual propagation.
* **Scientific Limitations:** This numerical half-wave dipole is an FDTD approximation. Real-world exact resonance may shift due to grid discretization (staircasing of the conductor radius), feed gap size, finite computational domain limits, and inherent numerical dispersion.

## Project Roadmap

- [x] Milestone 1 — 3D FDTD electromagnetic wave propagation
- [x] Milestone 2 — PML absorbing boundary
- [x] Milestone 3 — Half-wave dipole antenna
- [ ] Milestone 4 — Near-field analysis
- [ ] Milestone 5 — Far-field transformation
- [ ] Milestone 6 — 3D radiation pattern
- [ ] Milestone 7 — Gain/directivity/efficiency
- [ ] Milestone 8 — Patch antenna
- [ ] Milestone 9 — Advanced materials
- [ ] Milestone 10 — GPU acceleration
