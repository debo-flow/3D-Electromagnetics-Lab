# 3D Electromagnetics & Antenna Radiation Laboratory

A comprehensive 3D Electromagnetic wave propagation simulator built with Python, Numba, and Streamlit. This laboratory uses the Finite-Difference Time-Domain (FDTD) method to solve Maxwell's curl equations, allowing users to visualize EM wave propagation, antenna radiation, and virtual anechoic environments.

## Milestone 1 — 3D FDTD Electromagnetic Wave Propagation
Implemented a robust 3D FDTD solver on a Yee-grid. Includes basic numerical boundaries (PEC), Gaussian source injection, CFL stability control, memory protection, and PyVista-based 3D visualization.

## Milestone 2 — PML Absorbing Boundary (Virtual Anechoic Environment)
Replaced basic PEC boundaries with Convolutional Perfectly Matched Layers (CPML). 

## Milestone 3 — Half-Wave Dipole Antenna
Introduces the first physical antenna into the electromagnetic laboratory.

## Milestone 4 — Near-Field Electromagnetic Analysis
Introduces quantitative near-field sampling, frequency-domain analysis, and energy-flow calculations directly from the FDTD grid.

## Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
Implements a scientifically rigorous transformation of sampled near-field data into an estimated far-field electromagnetic solution based on the equivalence principle.

## Milestone 6 — 3D Antenna Radiation Pattern
Converts the validated NF2FF transformation data into a complete 3D angular antenna radiation pattern.

## Milestone 7 — Gain, Directivity & Efficiency
Introduces quantitative antenna performance metrics natively via 2D spherical solid-angle evaluations over the Far-Field limits.

## Milestone 8 — Rectangular Microstrip Patch Antenna
Replaces wire-antenna modeling with complex planar structures featuring dielectrics, vertical lumped feeds, and finite ground boundaries.

## Milestone 9 — Advanced Electromagnetic Materials
Generalizes the FDTD kernel to evaluate wave propagation through spatially varying environments ($\epsilon_r, \mu_r, \sigma$).

## Milestone 10 — GPU Accelerated FDTD Engine
Accelerates the immensely computation-heavy 3D FDTD loop using array-based vectorization strictly parallelized across GPU hardware frameworks.

## Milestone 11 — Advanced Numerical Validation & Benchmarking
Transforms the existing validation system into a rigorous, reproducible numerical verification framework to prove the mathematical integrity of the FDTD kernel.

## Milestone 12 — Dispersive Electromagnetic Materials
Extends the material system utilizing an Auxiliary Differential Equation (ADE) Debye model integrated continuously into Ampere's law.

## Milestone 13 — Anisotropic Electromagnetic Materials
Extends the material system to support diagonal tensor permittivity matrices ($\varepsilon̿$), enabling mathematically rigorous direction-dependent wave propagation.

## Milestone 14 — Antenna Arrays & Beamforming
Upgrades the simulation architecture to support Uniform Linear Arrays (ULA) and electronic Beam Steering.

## Milestone 15 — Adaptive Mesh Refinement (AMR)
Introduces a block-structured static Adaptive Mesh Refinement (AMR) algorithm utilizing 2-Way spatial restriction/prolongation coupling.

## Milestone 16 — Metamaterials & Engineered Electromagnetic Media
Introduces fully numerically-defined Engineered Media supporting Dispersive Negative-Index mechanics natively within the time domain via Drude ADE models.

## Milestone 17 — Inverse Electromagnetic Design & Optimization
Evolves the laboratory from forward-simulation to a completely self-contained Inverse Design framework utilizing dynamic parameter sweeps.

## Milestone 18 — Electromagnetic Topology Optimization
Introduces genuine Topological Inverse-Design capabilities directly integrated into the unified FDTD Maxwell kernel utilizing SIMP heuristic density mapping.

## Milestone 19 — Adjoint Electromagnetic Optimization & Sensitivity Analysis
Upgrades the Topology Optimizer from a stochastic hill-climbing heuristic to a mathematically exact continuous Gradient Descent framework powered by an Adjoint Electromagnetic Solver.

## Milestone 20 — Multi-Objective Electromagnetic Optimization & Pareto Analysis
Introduces a fully realized Multi-Objective Non-dominated Sorting Genetic Algorithm (NSGA-II) natively integrated with the FDTD core.
* **Pareto Dominance Evaluation:** Systematically evaluates competing design objectives—such as Maximizing Main-Beam Directivity while simultaneously Minimizing Peak Sidelobe Levels (SLL). The algorithm strictly prevents scalarizing objectives into arbitrary weighted sums, preserving true physical trade-off relationships.
* **NSGA-II Integration:** The population iteratively progresses through explicit Fast Non-Dominated Sorting and Crowding-Distance diversity preservation mechanics to evolve generation offspring via SBX crossovers and mutations, evaluating each candidate via a rigorous Full-Wave FDTD pass. 
* **Compromise Re-Simulation:** Displays an interactive 2D Pareto Front graph charting the exact non-dominated boundary against baseline parameters. Users can manually isolate 'Compromise Solutions' from the table and explicitly re-validate the far-field pattern derivation natively without faked analytical substitutions. 

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
- [x] Milestone 10 — GPU acceleration
- [x] Milestone 11 — Advanced numerical validation & benchmarking
- [x] Milestone 12 — Dispersive electromagnetic materials
- [x] Milestone 13 — Anisotropic electromagnetic materials
- [x] Milestone 14 — Antenna arrays & beamforming
- [x] Milestone 15 — Adaptive mesh refinement
- [x] Milestone 16 — Metamaterials & Engineered Electromagnetic Media
- [x] Milestone 17 — Inverse Electromagnetic Design & Optimization
- [x] Milestone 18 — Electromagnetic Topology Optimization
- [x] Milestone 19 — Adjoint Electromagnetic Optimization & Sensitivity Analysis
- [x] Milestone 20 — Multi-Objective Electromagnetic Optimization & Pareto Analysis
