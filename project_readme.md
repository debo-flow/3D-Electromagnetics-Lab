# 3D Electromagnetics & Antenna Radiation Laboratory

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Numba](https://img.shields.io/badge/Numba-JIT_Accelerated-orange)
![CUDA](https://img.shields.io/badge/CuPy-GPU_Ready-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Lab-red)
![Status](https://img.shields.io/badge/Milestones-1_to_40_Complete-brightgreen)

A comprehensive, GPU-accelerated 3D Electromagnetic wave propagation and spatial communications simulator built with Python, Numba, CuPy, PyVista, Plotly, and Streamlit. This computational laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations on a 3D Yee grid, evolving from fundamental full-wave physics into phased-array adaptive beamforming, wideband MIMO-OFDM channel characterization, and Reconfigurable Intelligent Surface (RIS) programmable electromagnetic environments.

---

## Core Architecture & Capabilities (Milestones 1–37)

* **3D FDTD Maxwell Solver:** Full-wave time-domain simulation with Courant-Friedrichs-Lewy (CFL) stability control, Convolutional Perfectly Matched Layers (CPML), and automatic CPU/GPU (`Numba`/`CuPy`) backend switching.
* **Advanced Materials & Metamaterials:** Support for lossy dielectrics, Perfect Electric Conductors (PEC), and multi-pole dispersive Debye/Lorentz media ($\epsilon(f)$, $\mu(f)$, $\sigma(f)$).
* **Inverse Design & Surrogate ML:** Adjoint sensitivity analysis, topology optimization, multi-objective Pareto front exploration, and reduced-order machine learning surrogates with Uncertainty Quantification (UQ).
* **Near-Field to Far-Field (NF2FF) & Metrology:** Planar, cylindrical, and spherical NF2FF transformations, S-parameter network analysis, reference-plane de-embedding, calibration math, and virtual SCPI hardware abstraction layers (HAL).
* **Phased-Array & Adaptive Beamforming:** Full-wave array simulation, mutual coupling characterization, active reflection coefficients, active VSWR, electronic beam steering, null steering, and adaptive weight optimization.

---

## Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory

Transforms standalone phased arrays into complete $N_T \times N_R$ spatial communication layers:

* **Broadband SVD & Ergodic Capacity:** Synthesizes spatial channel states ($H \in \mathbb{C}^{N_R \times N_T}$) via deterministic geometric scattering. Resolves singular values ($\Sigma$), numerical rank, and condition numbers ($\kappa(H)$) to evaluate theoretical Shannon spatial capacity bounds:
  $$C = \log_2 \det\left(I_{N_R} + \frac{\rho}{N_T} H H^H\right)$$
* **Time-Domain Multipath Profiling:** Executes Inverse Fast Fourier Transforms (IFFT) across broadband frequency sweeps to reconstruct Channel Impulse Responses ($h(\tau)$), Power Delay Profiles (PDP), and RMS Delay Spreads ($\tau_{\text{rms}}$).
* **Precoding Integrity under Imperfect CSI:** Implements Maximum Ratio Transmission (MRT), SVD precoding, and Zero-Forcing (ZF) spatial equalization ($H^H(HH^H)^{-1}$), evaluating numerical inversion stability under Channel State Information (CSI) estimation errors ($\epsilon$).

---

## Milestone 39 — Advanced Wideband Frequency-Selective MIMO & OFDM Electromagnetic Channel Laboratory

Evolves the laboratory from static narrowband MIMO characterization into a wideband, frequency-selective, and computational MIMO-OFDM analysis framework:

* **Wideband Channel Tensor $H(f)$:** Configurable frequency-grid engine computing complex channel matrices $H(f_k) \in \mathbb{C}^{N_R \times N_T}$ across arbitrary bandwidths without overwriting narrowband states.
* **Frequency Selectivity & Coherence:** Computes the frequency-domain correlation function $R(\Delta f)$ to extract Coherence Bandwidth ($B_c$) and classify channels into flat, mildly frequency-selective, or strongly frequency-selective regimes.
* **Wideband CIR, PDP & Delay Resolution:** Windowed IFFT reconstruction reporting physical delay resolution ($\Delta \tau = 1/B$), maximum unambiguous delay windows, mean excess delay, and $\tau_{\text{rms}}$.
* **Wideband SVD & Water-Filling Capacity:** Per-subcarrier singular value decomposition $H(f) = U(f)\Sigma(f)V^H(f)$, frequency-dependent numerical rank, condition number monitoring, and optimal frequency-domain water-filling power allocation.
* **MIMO-OFDM Computational Framework:** Models configurable subcarriers ($H_0, H_1, \dots, H_{N-1}$), useful symbol durations, and Cyclic Prefix (CP) sufficiency analysis ($T_{\text{CP}}$ vs. $\tau_{\text{rms}}$) to demonstrate computational Inter-Symbol Interference (ISI).
* **Subcarrier Equalization, Beam Squint & TTD:** Implements per-subcarrier Zero-Forcing and MMSE equalization alongside wideband phased-array Beam Squint analysis ($\theta(f)$ displacement) compared against ideal True-Time-Delay (TTD) beamforming models.

> *Disclaimer: The default environment is computational and simulation-based. No real-world wireless-network throughput, RF measurement, experimental OFDM performance, or channel-sounding measurement is claimed unless appropriate hardware and measured data are provided.*

---

## Milestone 40 — Advanced Reconfigurable Intelligent Surface & Programmable Electromagnetic Environment Laboratory

Extends the wideband MIMO-OFDM laboratory into **Reconfigurable Intelligent Surfaces (RIS)** and programmable electromagnetic environments to study how configurable metasurfaces modify spatial and frequency-selective channel behavior:

* **RIS Geometry & Element State Models:** Supports 1D, 2D planar, rectangular, and custom RIS topologies with configurable element spacing (normalized to $\lambda$), orientations, and reflection/transmission coefficients.
* **Continuous vs. Quantized Phase & Amplitude Control:** Models ideal continuous phase profiles alongside discrete $1\text{-bit}$, $2\text{-bit}$, $3\text{-bit}$, and $4\text{-bit}$ phase quantization states and configurable amplitude reflection responses.
* **Cascaded & Composite MIMO Channels:** Decomposes propagation into the direct path ($H_{\text{direct}}$), TX-to-RIS link ($H_{\text{RIS,TX}}$), and RIS-to-RX link ($H_{\text{RX,RIS}}$), constructing the composite programmable channel:
  $$H_{\text{total}}(f) = H_{\text{direct}}(f) + H_{\text{RX,RIS}}(f) \, \Phi \, H_{\text{RIS,TX}}(f)$$
  where $\Phi = \text{diag}(\alpha_1 e^{j\phi_1}, \dots, \alpha_M e^{j\phi_M})$ represents the RIS configuration matrix.
* **RIS Phase Optimization & Beam Steering:** Implements heuristic alignment, random search, coordinate/alternating optimization, and quantized search to maximize channel gain, SNR, or MIMO capacity, enabling anomalous reflection and RIS-assisted beam steering.
* **Spatial Multiplexing, SVD & Correlation:** Evaluates how optimized vs. random RIS configurations alter singular value distributions, effective spatial rank, condition numbers, and TX/RX spatial correlation matrices.
* **Wideband RIS, OFDM & Beam Squint:** Extends RIS modeling across frequency ($H_{\text{total}}(f)$) to analyze subcarrier capacity, RIS-induced delay spread impacts on OFDM cyclic prefix margins, and metasurface beam squint vs. frequency.
* **Parametric Studies, Multi-RIS & Multi-User:** Supports position, orientation, aperture size, and $\lambda$-spacing sweeps (identifying grating-lobe regimes), alongside multi-RIS cascaded interactions and multi-user interference modeling.
* **Robustness, Failure Analysis & Digital Twin:** Models RIS phase/amplitude configuration errors, imperfect CSI estimation, random/cluster element failures, and Monte Carlo uncertainty propagation, storing full state provenance inside the Electromagnetic Digital Twin.

> *Disclaimer: The default environment is computational and simulation-based. No real-world RIS hardware measurement, wireless-network throughput, experimental propagation measurement, or metasurface measurement is claimed unless appropriate hardware and measured data are provided.*

---

## Scientific Integrity & Hardware Safety

1. **Strict Data Separation:** All laboratory outputs are explicitly categorized as `SIMULATION`, `MOCK`, or `MEASURED`. Theoretical Shannon capacity and computational EVM-like metrics are never presented as physical RF instrument throughput or real-world network measurements.
2. **Hardware Safety Lock:** Real RF transmission, amplifier activation, antenna positioners, and RIS hardware controllers remain **strictly disabled by default**.
3. **Single-File Architecture:** The entire computational engine, data models, validation suites, and Streamlit UI are encapsulated within `app.py` for maximum portability and zero-dependency module resolution.

---

## Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/debo-flow/3D-Electromagnetics-Laboratory.git
cd 3D-Electromagnetics-Laboratory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Core packages: `streamlit`, `numpy`, `numba`, `pyvista`, `plotly`, `pandas`, and optionally `cupy` for NVIDIA GPU acceleration).*

### 3. Run the Laboratory
```bash
streamlit run app.py
```

---

## Project Roadmap

* [x] **Milestone 1–10:** Fundamental 3D FDTD Physics, Yee Grid, CPML & GPU Acceleration
* [x] **Milestone 11–20:** Advanced Dispersive Materials, Metamaterials, Inverse Design & Pareto MOO
* [x] **Milestone 21–25:** Surrogate ML Modeling, Uncertainty Quantification (UQ) & Automated Experiment Tracking
* [x] **Milestone 26–29:** Electromagnetic Digital Twins, SCPI VNA Interfaces & 3D Positioner Abstractions
* [x] **Milestone 30–33:** Planar, Cylindrical, and Spherical NF2FF Metrology Algorithms
* [x] **Milestone 34–35:** Calibration Math, Reference Planes & Automated Characterization Scorecards
* [x] **Milestone 36** — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
* [x] **Milestone 37** — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
* [x] **Milestone 38** — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
* [x] **Milestone 39** — Advanced Wideband Frequency-Selective MIMO & OFDM Electromagnetic Channel Laboratory
* [x] **Milestone 40** — Advanced Reconfigurable Intelligent Surface & Programmable Electromagnetic Environment Laboratory

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.