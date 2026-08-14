# 3D Electromagnetics & Antenna Radiation Laboratory

Welcome to the 3D Electromagnetics & Antenna Radiation Laboratory. 

**This repository currently contains Milestone 1**, which provides a foundational, scientifically correct 3D Finite-Difference Time-Domain (FDTD) electromagnetic wave propagation simulator.

## Overview
This laboratory is built to model the physical behavior of electromagnetic fields traveling through space. Rather than relying on simple visualizations or analytical approximations, this application solves the differential equations governing electromagnetism directly in the time domain.

## What is FDTD?
The Finite-Difference Time-Domain (FDTD) method is a highly robust grid-based numerical modeling technique used in computational electrodynamics. It allows for broad-band simulation of electromagnetic interactions by stepping forward through time, updating the electric (E) and magnetic (H) fields alternately.

## Maxwell Equations
The core engine implements the source-free, time-dependent Maxwell's curl equations in free space:
- ∇ × E = -μ_0 (∂H/∂t)
- ∇ × H = ε_0 (∂E/∂t)

## Yee Grid
The spatial domain is discretized using the standard **Yee lattice**. In this staggered arrangement, the E field components are located on the edges of the cell, while the H field components are evaluated on the faces. This maintains the physical curl properties of the equations exactly within the discrete domain.

## CFL Condition
To ensure the numerical simulation does not spiral into infinity (blow up), the timestep (Δt) is rigidly controlled by the Courant-Friedrichs-Lewy (CFL) stability criterion:
Δt ≤ 1 / (c * √(1/Δx² + 1/Δy² + 1/Δz²))
A configurable safety factor dictates the final temporal step width.

## Gaussian Source
A configurable "soft" source injects a Gaussian pulse into the grid. Users can alter the polarization, physical location, delay, and temporal width of the excitation.

---

## How to Run locally
1. Ensure you have **Python 3.11+** installed.
2. Clone the repository and navigate to the folder.
3. Install the minimal scientific requirements:
   `pip install -r requirements.txt`
4. Execute the application via Streamlit:
   `streamlit run app.py`

## Example Usage
1. Leave the grid at `50 x 50 x 50` cells.
2. Set Pulse Delay to `45` and Pulse Width to `15`.
3. Click **Run Simulation**.
4. Observe the `|E|` component under **Orthogonal Slices** mode to see a volumetric cross-section of the spherical wave propagating outward from the center.

---

## Current Limitations
* Milestone 1 currently simulates free-space electromagnetic wave propagation only.
* **Boundaries:** The simulation uses basic numerical truncation (equivalent to Perfect Electric Conductor / PEC at the edges). This causes reflections if the wave hits the boundary.
* Complex materials are not implemented.

## Future Roadmap

- **Milestone 1:** 3D FDTD free-space wave propagation (Current)
- **Milestone 2:** PML / absorbing boundary
- **Milestone 3:** Half-wave dipole antenna
- **Milestone 4:** Near-field analysis
- **Milestone 5:** Far-field transformation
- **Milestone 6:** 3D radiation pattern
- **Milestone 7:** Gain/directivity/efficiency
- **Milestone 8:** Patch antenna
- **Milestone 9:** Advanced materials
- **Milestone 10:** GPU acceleration

