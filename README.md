3D Electromagnetics & Antenna Radiation Laboratory

An advanced Python-based computational laboratory for studying electromagnetic propagation, antenna radiation, phased arrays, MIMO channels, wideband OFDM systems, and programmable electromagnetic environments.

The project progressively combines computational electromagnetics, antenna theory, numerical methods, MIMO channel modeling, beamforming, wideband analysis, and intelligent electromagnetic surfaces into a unified research-oriented laboratory.

---

Milestone 40 — Advanced Reconfigurable Intelligent Surface & Programmable Electromagnetic Environment Laboratory

Milestone 40 extends the wideband MIMO and OFDM capabilities of Milestone 39 by introducing Reconfigurable Intelligent Surfaces (RIS) and programmable electromagnetic environments.

The new computational workflow is:

TX ARRAY
   ↓
DIRECT ELECTROMAGNETIC PATH
   ↓
RIS / METASURFACE
   ↓
REFLECTION / TRANSMISSION / SCATTERING
   ↓
RX ARRAY
   ↓
DIRECT CHANNEL + RIS CHANNEL
   ↓
COMPOSITE MIMO CHANNEL
   ↓
RIS CONFIGURATION
   ↓
PHASE / AMPLITUDE OPTIMIZATION
   ↓
WIDEBAND CHANNEL
   ↓
OFDM SUBCARRIERS
   ↓
BEAMFORMING / PRECODING
   ↓
MIMO CAPACITY
   ↓
ROBUSTNESS
   ↓
VALIDATION
   ↓
DIGITAL TWIN
   ↓
REPORT

---

Major Capabilities

RIS System Modeling

The laboratory supports computational models for:

- RIS geometry
- 1D RIS
- 2D planar RIS
- Rectangular RIS
- Circular/custom geometries
- Element positions
- Element orientations
- Polarization
- Element states
- Reflection coefficients
- Transmission coefficients where supported
- Phase control
- Amplitude control

---

RIS Phase Control

Configurable phase control supports:

- Continuous phase
- Binary phase
- 1-bit phase
- 2-bit phase
- 3-bit phase
- 4-bit phase
- Configurable phase quantization

Continuous and quantized configurations are explicitly distinguished.

---

RIS Channel Modeling

The laboratory supports the computational decomposition:

H_total = H_direct + H_RIS

and, where appropriate, a cascaded representation:

H_RIS = H_RX,RIS · Φ · H_RIS,TX

The following components can be analyzed independently:

- Direct TX → RX channel
- TX → RIS channel
- RIS → RX channel
- RIS-assisted channel
- Composite channel

This allows direct comparison between:

DIRECT ONLY
RIS ONLY
DIRECT + RIS

---

RIS Optimization

The laboratory provides configurable computational RIS optimization methods where applicable:

- Exhaustive search
- Coordinate descent
- Alternating optimization
- Gradient-based optimization where supported
- Random search
- Quantized phase search

Optimization objectives may include:

- Received power
- Channel gain
- SNR
- MIMO capacity
- Condition-number objectives where mathematically meaningful
- Multi-user objectives where implemented

Optimization parameters, stopping criteria, and random seeds are preserved for reproducibility.

---

RIS-Assisted Beamforming

Milestones 37–39 beamforming infrastructure is reused.

Supported configurations include:

- TX beamforming + RIS
- RIS optimization + RX combining
- RIS-only optimization
- TX + RIS + RX optimization

Comparisons can be made between:

TX ONLY
RX ONLY
RIS ONLY
TX + RIS
RIS + RX
TX + RIS + RX

---

MIMO Analysis

RIS-assisted MIMO channels are analyzed using:

- Channel magnitude
- Channel phase
- SVD
- Singular values
- Numerical rank
- Effective spatial rank
- Condition number
- Spatial correlation
- MIMO capacity

The laboratory does not interpret numerical rank as proof of independent physical communication channels.

---

Wideband RIS

Milestone 39's frequency-dependent MIMO framework is extended to:

H_total(f)

The laboratory can analyze:

- Channel response versus frequency
- RIS response versus frequency
- Capacity versus frequency
- Rank versus frequency
- Condition number versus frequency
- Frequency-dependent phase
- Frequency-dependent amplitude

---

RIS and OFDM

RIS-assisted OFDM analysis supports:

- Subcarrier channel matrices
- Frequency-dependent RIS channels
- Frequency-flat RIS configuration
- Frequency-dependent RIS configuration where computationally meaningful
- Per-subcarrier capacity
- Subcarrier SNR
- PDP
- Delay spread
- Cyclic-prefix analysis

The implementation remains computational and does not claim experimental OFDM performance.

---

Beam Squint

Wideband RIS systems can be analyzed for frequency-dependent beam displacement.

Comparisons include:

- Narrowband phase-based control
- Wideband phase behavior
- True-time-delay-inspired computational models where supported

Metrics include:

- Beam direction
- Beam displacement
- Gain degradation
- Capacity degradation

---

Polarization

RIS polarization analysis supports:

- Co-polar response
- Cross-polar response
- Polarization mismatch
- Polarization conversion
- Frequency-dependent polarization behavior

Vector-field information is used where available.

---

RIS Material Modeling

The existing material engine is reused for RIS studies.

Where supported, RIS models may include:

- Permittivity
- Conductivity
- Permeability
- Loss
- Frequency-dependent material properties

The laboratory distinguishes between:

IDEALIZED RIS

and:

MATERIAL-BASED RIS

---

RIS Element Coupling

Where computationally supported, coupling between RIS elements can be analyzed.

Metrics include:

- Coupling magnitude
- Coupling phase
- Frequency-dependent coupling
- Impact on channel gain
- Impact on capacity
- Impact on beamforming
- Impact on efficiency

---

RIS Efficiency

Where valid, the laboratory can analyze:

- Reflection efficiency
- Transmission efficiency
- Absorption/loss
- Element efficiency
- Overall RIS efficiency

Ideal unit-amplitude reflection is not automatically assumed for physical material models.

---

RIS Position and Geometry Studies

The laboratory supports configurable studies involving:

Position

- TX–RIS distance
- RIS–RX distance
- RIS position
- RIS height
- RIS orientation

Size

- Number of elements
- Surface dimensions
- Element spacing

Orientation

- Surface tilt
- Azimuth
- Elevation
- TX orientation
- RX orientation

Metrics include:

- Channel gain
- Capacity
- Rank
- Condition number
- Delay spread
- Beam direction
- Spatial correlation

---

Multi-RIS

Where computationally feasible, multiple RIS panels can be modeled.

Example:

TX
 ↓
RIS 1
 ↓
RIS 2
 ↓
RX

Each RIS maintains an independent configuration and provenance record.

---

Multi-User RIS

Where implemented, the laboratory supports computational multi-user RIS analysis.

Metrics may include:

- Per-user received power
- SNR
- Capacity
- Inter-user interference
- Channel correlation

Possible optimization objectives include:

- Sum capacity
- Minimum user capacity
- Weighted sum capacity
- Inter-user interference minimization

These results are computational and must not be interpreted as cellular-network measurements.

---

Channel Estimation

RIS-assisted channel estimation supports:

- Perfect CSI
- Imperfect CSI
- RIS-state-dependent estimation
- Configurable estimation error

The system explicitly distinguishes:

TRUE CHANNEL

from:

ESTIMATED CHANNEL

---

RIS Configuration Uncertainty

Uncertainty analysis can include:

- Phase errors
- Amplitude errors
- Position errors
- Orientation errors
- Material uncertainty
- Frequency uncertainty
- CSI uncertainty

Uncertainty can be propagated into:

- Channel gain
- Capacity
- Rank
- Condition number
- Beam direction
- Efficiency

---

Monte Carlo Robustness

Configurable Monte Carlo studies provide distributions for:

- Capacity
- Channel gain
- Rank
- Condition number
- Beam alignment
- Efficiency

Random seeds are preserved to maintain reproducibility.

---

RIS Failure Analysis

RIS element failures can be simulated.

Supported failure patterns include:

- Random element failures
- Cluster failures
- Configurable failure percentages
- Specific failed-element locations

The effect on:

- Beam pattern
- Channel gain
- Capacity
- Rank
- Condition number

can then be analyzed.

---

Robust RIS Optimization

Where implemented, the laboratory can compare:

IDEAL OPTIMIZATION

against:

ROBUST OPTIMIZATION

under uncertainty.

Metrics include:

- Nominal capacity
- Mean capacity
- Worst-case capacity
- Capacity variance

---

Digital Twin

Milestone 40 extends the Digital Twin to include:

- RIS geometry
- RIS element states
- RIS material configuration
- Environment
- TX array
- RX array
- Direct channel
- RIS channel
- Composite channel
- Frequency response
- OFDM state
- Beamforming
- Precoding
- Combining
- Coupling
- Calibration
- Uncertainty
- Validation
- Provenance

---

Validation

Automated validation covers:

- RIS data model
- RIS geometry
- Element states
- Phase quantization
- Reflection/transmission models
- Cascaded channels
- Composite channels
- RIS optimization
- Capacity
- SVD
- Rank
- Condition number
- Beam steering
- Beamforming
- Wideband operation
- OFDM operation
- PDP
- Delay spread
- Polarization
- Coupling
- Efficiency
- Position studies
- Orientation studies
- Size studies
- Spacing studies
- Multi-RIS
- Multi-user models
- Channel estimation
- Configuration errors
- Monte Carlo analysis
- Failure analysis
- Regression

Validation results use:

PASS
FAIL
WARNING

---

Scientific Integrity

The laboratory strictly separates:

SIMULATION
MOCK
MEASURED

data.

The project does not fabricate:

- RIS hardware measurements
- Metasurface measurements
- Experimental reflection coefficients
- Experimental propagation measurements
- Wireless-network throughput
- Cellular-network performance
- Experimental OFDM performance

Idealized or assumed parameters are explicitly identified.

Important Statement

«The default environment is computational and simulation-based. No real-world RIS hardware measurement, wireless-network throughput, experimental propagation measurement, or metasurface measurement is claimed unless appropriate hardware and measured data are provided.»

---

Hardware Safety

Real RF transmission remains disabled by default.

The software does not automatically:

- Transmit RF
- Activate amplifiers
- Configure RF transmitters
- Change RF instrument settings
- Move antennas
- Move RIS hardware
- Start physical measurements

Hardware integration remains separate from the computational laboratory.

---

Technology Stack

- Python
- NumPy
- Streamlit
- Matplotlib
- Plotly where appropriate
- PyVista where supported by the existing project
- Dataclasses
- JSON
- CSV
- Markdown

The main implementation remains in:

app.py

---

Repository Structure

3D-Electromagnetics-Lab/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

The single-file implementation requirement is preserved.

---

Roadmap

- [x] Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization Laboratory
- [x] Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
- [x] Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
- [x] Milestone 39 — Advanced Wideband Frequency-Selective MIMO & OFDM Electromagnetic Channel Laboratory
- [ ] Milestone 40 — Advanced Reconfigurable Intelligent Surface & Programmable Electromagnetic Environment Laboratory

Milestone 40 should only be marked complete after implementation and validation.

---

Development Philosophy

The project prioritizes:

SCIENTIFIC CORRECTNESS
        >
ELECTROMAGNETIC CORRECTNESS
        >
MATHEMATICAL CORRECTNESS
        >
DATA INTEGRITY
        >
NUMERICAL STABILITY
        >
VALIDATION
        >
REPRODUCIBILITY
        >
PERFORMANCE
        >
VISUALIZATION
        >
UI

The objective is to build a transparent, reproducible and scientifically defensible computational electromagnetics laboratory rather than a black-box demonstration.

---

Current Status

Milestone 40 — Advanced Reconfigurable Intelligent Surface & Programmable Electromagnetic Environment Laboratory

Status:

IMPLEMENTATION IN PROGRESS

Milestones 1–39 are intended to remain preserved throughout development.

---

License

See "LICENSE".
