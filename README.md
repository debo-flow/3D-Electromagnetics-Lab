# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

*(Milestones 1 through 36 are fully implemented, encompassing fundamental 3D FDTD, Inverse Design, Digital Twins, hardware SCPI abstractions, NF2FF geometry transformations, and Multi-Antenna Phased Array couplings. See previously documented roadmaps for specifics).*

## Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
Upgrades the Phased Array modeling architecture with deterministic Mathematical Projections to execute advanced Interference Mitigation and hardware-degradation analytics.
* **Exact Null-Steering Orthogonal Projections:** Deploys continuous linear algebra constraints to compute Ideal Beamforming Weights natively. Synthesizes a projection matrix ($P = I - C(C^H C)^{-1} C^H$) targeting exact environmental coordinates to forcefully generate deep radiation nulls alongside standard Main-Beam phase steering requirements.
* **Hardware Quantization Modeling:** Validates the discrepancy introduced by real-world manufacturing limits by quantizing mathematically infinite Ideal Weights directly into constrained $N$-bit Phase Shifters and Voltage Variable Attenuators (e.g., locking phases to $22.5^\circ$ discrete blocks for 4-Bit chips).
* **Monte Carlo Robustness & Scan Loss Bounds:** Tracks exact peak gain destruction mathematically over a full physical $-90^\circ \to 90^\circ$ angular sweep to define Broadside-Scan Loss parameters. Runs intense 50-iteration Monte Carlo tolerance permutations tracking 95th Percentile Side-Lobe blooming and Null-Filling effects induced by structural hardware jitters (Gaussian Noise arrays).

## Project Roadmap
- [x] Milestone 1-10: Fundamental FDTD Physics & GPU Acceleration
- [x] Milestone 11-20: Advanced Materials, Inverse Design & Pareto MOO
- [x] Milestone 21-25: Surrogate ML Modeling, UQ, and Automated Experiment Tracking
- [x] Milestone 26-29: Digital Twins, SCPI VNA Interfaces, and 3D Positioner Abstractions
- [x] Milestone 30-33: Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
- [x] Milestone 34-35: Calibration Math, Reference Planes, and Automated Characterization Scorecards
- [x] Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
- [x] Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
