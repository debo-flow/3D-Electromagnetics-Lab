# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

*(Milestones 1 through 35 are fully implemented, encompassing fundamental 3D FDTD, Inverse Design, Digital Twins, hardware SCPI abstractions, NF2FF geometry transformations, and rigorous automated Metrology Scorecards. See previously documented roadmaps for specifics).*

## Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
Transitions the laboratory from single-element analytics to fully coupled, **Multi-Antenna Phased Array Systems**.
* **Analytical Array Factor (AF) Engine:** Dynamically generates complex $N$-element cartesian geometries (ULA, UPA, UCA) and computes the exact $AF(\hat{r})$ integrations based on continuous Phase-Steering offsets ($\phi_n = -k \mathbf{r}_n \cdot \hat{\mathbf{r}}_{target}$) and Side-Lobe suppression algorithms (Binomial Tapering).
* **Active Mutual Coupling ($S_{ij}$):** Extends S-Parameter matrix diagnostics to measure structural load limits across N-ports. Identifies how dynamic excitation matrices ($w_n = A_n e^{j\phi_n}$) actively alter real-time element matching boundaries (Active VSWR).
* **FDTD Full-Wave Integrity vs Degradation Analysis:** Explicitly cross-correlates the mathematical Array Factor predictions directly against a massive Full-Wave Physical FDTD simulation of interacting elements. Deploys structural Tolerance Checks evaluating the beam's survivability under specific element failure states and cumulative phase-jitter Gaussian distributions.

## Project Roadmap
- [x] Milestone 1-10: Fundamental FDTD Physics & GPU Acceleration
- [x] Milestone 11-20: Advanced Materials, Inverse Design & Pareto MOO
- [x] Milestone 21-25: Surrogate ML Modeling, UQ, and Automated Experiment Tracking
- [x] Milestone 26-29: Digital Twins, SCPI VNA Interfaces, and 3D Positioner Abstractions
- [x] Milestone 30-33: Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
- [x] Milestone 34-35: Calibration Math, Reference Planes, and Automated Characterization Scorecards
- [x] Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
