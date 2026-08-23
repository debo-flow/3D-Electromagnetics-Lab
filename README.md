# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive, GPU-accelerated 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This computational laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, near-field/far-field antenna radiation, hardware SCPI abstractions, phased-array adaptive beamforming, and multi-antenna spatial communication layers.

## Core Capabilities

* **3D FDTD Maxwell Solver:** GPU/CPU accelerated Yee grid with CFL stability control and CPML boundaries.
* **Advanced Material Modeling:** Support for dispersive materials, metamaterials, and topology optimization.
* **Array Metrology & Beamforming:** Full-wave array simulation, NF-to-FF transformations, null steering, and S-parameter mutual coupling analysis.
* **Advanced MIMO Channel Modeling:** Extraction of $H \in \mathbb{C}^{N_R \times N_T}$ matrices, SVD, spatial rank, condition numbers, and Shannon Spatial Capacity.
* **Digital Twin & Provenance:** Automated validation, experiment tracking, and rigorous separation of simulated vs. measured data.

---

## Milestone 39 — Advanced Wideband Frequency-Selective MIMO & OFDM Electromagnetic Channel Laboratory

This milestone evolves the laboratory into the frequency domain, mapping fully frequency-selective wideband environments and OFDM subcarrier configurations. 

### Wideband & OFDM Features
* **Wideband MIMO:** Computes frequency-dependent channel matrices ($H(f)$) across configurable bandwidths.
* **Time-Domain Analytics:** Generates Channel Impulse Responses (CIR), Power Delay Profiles (PDP), and RMS Delay Spreads via IFFT.
* **Frequency Selectivity:** Analyzes frequency correlation and Coherence Bandwidth bounds.
* **OFDM Subcarrier Modeling:** Maps $H(f)$ to active subcarriers, evaluating Cyclic Prefix sufficiency and Inter-Symbol Interference (ISI).
* **Wideband Equalization:** Implements per-subcarrier Zero-Forcing (ZF) and MMSE equalizers.
* **Precoding & Combining:** Extends Maximum Ratio Transmission (MRT), Zero-Forcing, and SVD precoding across the wideband spectrum.
* **Hardware Non-Idealities:** Compares narrowband phase-shifter Beam Squint against ideal True-Time-Delay (TTD) models.
* **Wideband Coupling & Polarization:** Integrates frequency-dependent S-parameters, Active VSWR, and cross-polarization mismatches.
* **Robustness & Monte Carlo:** Analyzes system degradation under Imperfect CSI and channel estimation errors through stochastic distributions.

*The default environment is computational and simulation-based. No real-world wireless-network throughput, RF measurement, experimental OFDM performance, or channel-sounding measurement is claimed unless appropriate hardware and measured data are provided.*

---

## Installation & Setup

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the computational laboratory: `streamlit run app.py`

### Requirements
* Python 3.8+
* Streamlit
* NumPy & Numba (CPU Acceleration)
* CuPy (Optional, for GPU Acceleration)
* PyVista & Plotly (3D & 2D Visualization)
* Pandas

---

## Project Roadmap

* [x] Milestone 1-10: Fundamental FDTD Physics & GPU Acceleration
* [x] Milestone 11-20: Advanced Materials, Inverse Design & Pareto MOO
* [x] Milestone 21-25: Surrogate ML Modeling, UQ, and Automated Experiment Tracking
* [x] Milestone 26-29: Digital Twins, SCPI VNA Interfaces, and 3D Positioner Abstractions
* [x] Milestone 30-33: Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
* [x] Milestone 34-35: Calibration Math, Reference Planes, and Automated Characterization Scorecards
* [x] Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
* [x] Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
* [x] Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
* [x] Milestone 39 — Advanced Wideband Frequency-Selective MIMO & OFDM Electromagnetic Channel Laboratory
