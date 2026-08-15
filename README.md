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

## Milestone 4 — Near-Field Electromagnetic Analysis
Introduces quantitative near-field sampling, frequency-domain analysis, and energy-flow calculations directly from the FDTD grid.
* **Near-Field Data Acquisition:** Evaluates the spatial distribution of E and H components without synthetic interpolation.
* **Frequency Analysis (FFT):** Calculates the continuous-time frequency spectrum E(f) and phase using a rectangular window optimal for FDTD transient pulses.
* **Poynting Vector (S):** Implements both the instantaneous ($S = E \times H$) and the strictly time-averaged ($\langle S \rangle$) Poynting vector over a configurable later-time window to analyze actual energy-flow density.

## Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
Implements a scientifically rigorous transformation of sampled near-field data into an estimated far-field electromagnetic solution.
* **Electromagnetic Equivalence Principle:** A virtual closed bounding box (Huygens surface) is drawn around the antenna. Equivalent electric ($J_s = \hat{n} \times H$) and magnetic ($M_s = -\hat{n} \times E$) surface currents are calculated on this boundary.
* **Frequency-Domain Transformation:** Time-domain tangential fields over the entire equivalence surface are converted to the frequency domain (via windowed FFT). Maintaining correct spatial *phase* relationships is critical for accurate constructive and destructive spatial interference calculations in the far field.

## Milestone 6 — 3D Antenna Radiation Pattern
Converts the validated NF2FF transformation data into a complete 3D angular antenna radiation pattern.
* **Spherical Angular Grid:** Fully configurable spatial mapping across $\theta$ (0°–180°) and $\phi$ (0°–360°), yielding high-resolution angular datasets directly from the numerical simulation.
* **Field Computations:** Computes the total far-field electric magnitude $|E_{far}| = \sqrt{|E_\theta|^2 + |E_\phi|^2}$ supporting both absolute scaling, global normalization, and dB logarithmic representations.
* **3D Visualization:** Maps the calculated normalized or dB data onto a true spherical coordinate geometry overlaying the physical computational grid using PyVista.

## Milestone 7 — Gain, Directivity & Efficiency
Introduces quantitative antenna performance metrics natively derived from the FDTD electromagnetic simulation.
* **Radiation Intensity ($U$):** Calculated directly from the far-field magnitude $U(\theta, \phi) = r_{obs}^2 \frac{|E_{far}|^2}{2Z_0}$.
* **Total Radiated Power ($P_{rad}$):** Extracted via precise 2D Trapezoidal spherical integration ($P_{rad} = \int \int U(\theta, \phi) \sin\theta d\theta d\phi$) over the entire angular computational grid.
* **Directivity ($D$):** Mapped spatially based on the angular concentration of radiation using $D = \frac{4\pi U}{P_{rad}}$. 
* **Scientific Limitations:** Extracting absolute Antenna Gain and Radiation Efficiency requires a structurally pristine extraction of accepted power $P_{in}$. The base FDTD implementation utilizes a numerical proxy soft-source to enforce CFL stability without disruptive hard boundaries. Therefore, efficiency and gain metrics are explicitly restricted and withheld from the dashboard to prevent publishing mathematically flawed or unphysical "textbook" assumptions. 

## Milestone 8 — Rectangular Microstrip Patch Antenna
Replaces wire-antenna modeling with complex planar structures featuring dielectrics and finite ground boundaries.
* **Patch Anatomy:** Implements a conductive patch, finite conductive ground plane, dielectric substrate ($\epsilon_r$), and a localized vertical feed bridging the ground to the patch directly into the FDTD Cartesian grid.
* **Analytical Design Tools:** Provides a UI interface that calculates ideal theoretical dimensions (Width, Length, Effective Dielectric Constant) via standard microstrip Transmission-Line approximations. 

## Milestone 9 — Advanced Electromagnetic Materials
Generalizes the FDTD kernel to evaluate wave propagation through spatially varying environments featuring independent permittivity ($\epsilon_r$), permeability ($\mu_r$), and conductivity ($\sigma$).
* **Material-Aware Maxwell Equations:** Overhauled the core discrete update equations to dynamically evaluate local material coefficients ($C_{e1}, C_{e2}, C_{h2}$) on a cell-by-cell basis. 
* **Dielectric & Conductive Loss:** Incorporates dielectric loss tangents ($\tan \delta$) mapped effectively to conductive loss ($\sigma = \omega \epsilon \tan \delta$). FDTD inherently suppresses wave amplitudes as currents ($J = \sigma E$) develop in real-time. Energy dissipated by Joule heating is tracked comprehensively inside the diagnostic loop.
* **Validation Lab Environments:** Features isolated 1D-like simulation modes designed strictly to prove the accuracy of Fresnel interface reflections ($R_{num}$ vs $R_{theoretical}$) and measurable conductor attenuation independent of the antenna structures.
* **Scientific Limitations:** Complex Dispersive (Drude/Lorentz), anisotropic tensor properties, and sub-cell precision conforming are withheld to safeguard basic Yee-grid broad stability constraints. The CFL condition explicitly checks velocity limits utilizing background free-space speeds ($v_{max} = c$).

## Project Roadmap

- [x] Milestone 1 — 3D FDTD electromagnetic wave propagation
- [x] Milestone 2 — PML absorbing boundary
- [x] Milestone 3 — Half-wave dipole antenna
- [x] Milestone 4 — Near-field analysis
- [x] Milestone 5 — Near-to-far-field transformation
- [x] Milestone 6 — 3D radiation pattern
- [x] Milestone 7 — Gain/directivity/efficiency
- [x] Milestone 8 — Rectangular microstrip patch antenna
- [x] Milestone 9 — Advanced electromagnetic materials
- [ ] Milestone 10 — GPU acceleration
