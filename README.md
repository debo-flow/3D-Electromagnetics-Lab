# 3D Electromagnetics & Antenna Radiation Laboratory

Welcome to the 3D Electromagnetics & Antenna Radiation Laboratory. 

**This repository currently contains up to Milestone 2**, which provides a foundational 3D Finite-Difference Time-Domain (FDTD) electromagnetic wave propagation simulator, complete with a Convolutional Perfectly Matched Layer (CPML) absorbing boundary.

## Overview
This laboratory is built to model the physical behavior of electromagnetic fields traveling through space. Rather than relying on simple visualizations, this application solves the differential equations governing electromagnetism directly in the time domain.

## Milestone 2 — PML Absorbing Boundary
When computing FDTD in a finite discrete grid, waves that hit the edges of the computational array will artificially reflect back into the center (a Perfect Electric Conductor or PEC boundary). To simulate infinite free space (a "Virtual Anechoic Environment"), we implemented a **Convolutional Perfectly Matched Layer (CPML)**.

**How it works:**
* CPML introduces localized auxiliary variables ($\psi$) into Maxwell's curl equations strictly near the domain boundaries.
* It applies a polynomially graded artificial conductivity profile that smoothly absorbs incident electromagnetic energy.
* This is a **numerical absorbing boundary** designed to significantly reduce artificial reflection, making it highly suitable for future antenna modeling.
* *Note: While it reduces reflections drastically (often > 99% absorption), it is a numerical approximation and not a physical anechoic chamber.*

## What is FDTD?
The Finite-Difference Time-Domain (FDTD) method is a robust grid-based numerical modeling technique. It allows for broad-band simulation of electromagnetic interactions by stepping forward through time, updating the electric (E) and magnetic (H) fields alternately on a standard staggered **Yee lattice**.

## CFL Condition
To ensure the numerical simulation remains stable, the timestep (Δt) is rigidly controlled by the Courant-Friedrichs-Lewy (CFL) criterion.

---

## How to Run locally
1. Ensure you have **Python 3.11+** installed.
2. Clone the repository and install requirements:
   `pip install -r requirements.txt`
3. Execute the application:
   `streamlit run app.py`

## Example Usage: The PML Test
1. Set grid to `60 x 60 x 60`.
2. Under BOUNDARY CONDITIONS, ensure `CPML (Absorbing)` is selected.
3. Click **Run PML Validation Test**.
4. Observe the comparison graph. The red line (PEC) will show a massive secondary spike as the wave bounces off the wall. The blue line (CPML) will remain flat, demonstrating a severely reduced reflection coefficient.

---

## Future Roadmap

- [x] **Milestone 1:** 3D FDTD free-space wave propagation
- [x] **Milestone 2:** PML absorbing boundary
- [ ] **Milestone 3:** Half-wave dipole antenna
- [ ] **Milestone 4:** Near-field analysis
- [ ] **Milestone 5:** Far-field transformation
- [ ] **Milestone 6:** 3D radiation pattern
- [ ] **Milestone 7:** Gain/directivity/efficiency
- [ ] **Milestone 8:** Patch antenna
- [ ] **Milestone 9:** Advanced materials
- [ ] **Milestone 10:** GPU acceleration
