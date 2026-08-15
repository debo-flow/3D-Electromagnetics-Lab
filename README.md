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

## Milestone 4 — Near-Field Electromagnetic Analysis
Introduces quantitative near-field sampling, frequency-domain analysis, and energy-flow calculations directly from the FDTD grid.
* **Near-Field Data Acquisition:** Evaluates the spatial distribution of E and H components without synthetic interpolation.
* **Probes & Time-Domain:** Multi-point probes strictly track the transient and later-time field evolution at specific geometrical locations (Feed, Axis, Perpendicular).
* **Frequency Analysis (FFT):** Calculates the continuous-time frequency spectrum E(f) and phase using a rectangular window optimal for FDTD transient pulses.
* **Electromagnetic Energy Density:** Calculates and maps the instantaneous energy density $u = \frac{1}{2}(\epsilon|E|^2 + \mu|H|^2)$ across 2D slices.
* **Poynting Vector (S):** Implements both the instantaneous ($S = E \times H$) and the strictly time-averaged ($<S>$) Poynting vector over a configurable later-time window to analyze actual energy-flow density.
* **Scientific Limitations:** Numerical dispersion, Yee-cell offsets (half-step spatial differences), and finite simulation duration mean FFT and phase results are computational approximations, not exact infinite-time continuous solutions.

## Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
Implements a scientifically rigorous transformation of sampled near-field data into an estimated far-field electromagnetic solution.
* **Electromagnetic Equivalence Principle:** A virtual closed bounding box (Huygens surface) is drawn around the antenna. Equivalent electric ($J_s = \hat{n} \times H$) and magnetic ($M_s = -\hat{n} \times E$) surface currents are calculated on this boundary.
* **Frequency-Domain Transformation:** Time-domain tangential fields over the entire equivalence surface are converted to the frequency domain (via windowed FFT). Maintaining correct spatial *phase* relationships is critical for accurate constructive and destructive spatial interference calculations in the far field.
* **Observation Directions & Distance:** Solves asymptotic far-field integrals for specific spherical coordinates ($\theta, \phi$) and user-defined observation distance ($r$). The data is purely calculated, not synthetically extrapolated. 
* **Scientific Limitations:** Because E and H fields are evaluated at slightly different spatial locations by half a cell on the standard Yee-grid, raw field extraction introduces a minor geometric phase error at very high frequencies. The numerical FFT frequency bin resolution is also limited by the finite simulation duration, avoiding "exact" analytical assumptions.

## Project Roadmap

- [x] Milestone 1 — 3D FDTD electromagnetic wave propagation
- [x] Milestone 2 — PML absorbing boundary
- [x] Milestone 3 — Half-wave dipole antenna
- [x] Milestone 4 — Near-field analysis
- [x] Milestone 5 — Far-field transformation
- [ ] Milestone 6 — 3D radiation pattern
- [ ] Milestone 7 — Gain/directivity/efficiency
- [ ] Milestone 8 — Patch antenna
- [ ] Milestone 9 — Advanced materials
- [ ] Milestone 10 — GPU acceleration
