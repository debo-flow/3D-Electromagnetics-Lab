# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

*(Milestones 1 through 37 are fully implemented, encompassing fundamental 3D FDTD, Inverse Design, Digital Twins, hardware SCPI abstractions, NF2FF geometry transformations, and Advanced Phased Array Beamforming. See previously documented roadmaps for specifics).*

## Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
Evolves the laboratory beyond single-source Array Factor steering into complete spatial communication layers mapping full $N_T \times N_R$ Multi-Input Multi-Output (MIMO) channel matrices.
* **Broadband SVD & Ergodic Capacity:** Computes sophisticated spatial channel states ($H \in \mathbb{C}^{N_R \times N_T \times N_f}$) via deterministic geometric scattering algorithms. Resolves the exact Numerical Rank and Condition Numbers via Singular Value Decomposition (SVD) to rigorously output Shannon Spatial Capacity bounds ($C = \log_2 \det(I + \rho H H^H)$) evaluating multiplexing gains against 1x1 SISO.
* **Time-Domain Multipath Profiling:** Analyzes $200$ MHz broadband $H(f)$ tensor sweeps by executing Inverse Fast Fourier Transforms (IFFT) automatically. Formats the output natively into physical Power Delay Profiles (PDP) explicitly identifying the RMS Delay Spread boundaries ($\tau_{rms}$) introduced by Urban Non-Line-of-Sight (NLoS) configurations.
* **Precoding Integrity under Imperfect CSI:** Installs analytical Maximum Ratio Transmission (MRT) and Zero-Forcing (ZF) spatial algorithms ($H^H(HH^H)^{-1}$). Executes Monte Carlo fault simulations injecting Channel State Information (CSI) estimation errors ($\epsilon$) to definitively monitor catastrophic numerical inversion instabilities inside ZF precoders operating on ill-conditioned channel states.

## Project Roadmap
- [x] Milestone 1-10: Fundamental FDTD Physics & GPU Acceleration
- [x] Milestone 11-20: Advanced Materials, Inverse Design & Pareto MOO
- [x] Milestone 21-25: Surrogate ML Modeling, UQ, and Automated Experiment Tracking
- [x] Milestone 26-29: Digital Twins, SCPI VNA Interfaces, and 3D Positioner Abstractions
- [x] Milestone 30-33: Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
- [x] Milestone 34-35: Calibration Math, Reference Planes, and Automated Characterization Scorecards
- [x] Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
- [x] Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
- [x] Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
