# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

*(Milestones 1 through 34 are fully implemented, encompassing fundamental 3D FDTD, GPU Acceleration, Inverse Design, Digital Twins, hardware SCPI abstractions, NF2FF geometry transformations, and rigorous metrology calibration workflows. See previously documented roadmaps for specifics).*

## Milestone 35 — Advanced Antenna Characterization & Automated Measurement Validation Laboratory
Completes the end-to-end Antenna Metrology pipeline by consolidating raw RF and Spatial transforms into an Automated Characterization Scorecard.
* **Rigorous Analytical Math Engine:** Derives exact Directivity ($D = 4\pi U_{max} / P_{rad}$) through complete Spherical solid-angle integration. Calculates true Total Efficiency by unifying the spatial Radiation Efficiency limits with scalar Mismatch Efficiency equations ($\eta = 1 - |S_{11}|^2$).
* **Automated Anomaly Diagnostics:** Scans extracted dataset properties heuristically, flagging critical warnings (e.g., resonance shifting away from the designed Operating Frequency, or Side-Lobe Levels breaching the $-10$ dB threshold) to proactively block faulty measurement reporting.
* **Unified Scorecard & Provenance Export:** Binds the 3D Radiation metrics, Ludwig Polarization ratios (Axial Ratio), VSWR Bandwidth computations, and 1D Front-to-Back Ratios directly into a printable JSON manifest certifying strict scientific isolation between `MOCK`, `SIMULATED`, and `MEASURED` data sources.

## Project Roadmap
- [x] Milestone 1-10: Fundamental FDTD Physics & GPU Acceleration
- [x] Milestone 11-20: Advanced Materials, Inverse Design & Pareto MOO
- [x] Milestone 21-25: Surrogate ML Modeling, UQ, and Automated Experiment Tracking
- [x] Milestone 26-29: Digital Twins, SCPI VNA Interfaces, and 3D Positioner Abstractions
- [x] Milestone 30-33: Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
- [x] Milestone 34 — Advanced Antenna Measurement, Calibration & Reference Correction Laboratory
- [x] Milestone 35 — Advanced Antenna Characterization & Automated Measurement Validation Laboratory
