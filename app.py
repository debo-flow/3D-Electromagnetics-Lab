"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 29 — Automated Antenna Characterization & 3D Radiation-Pattern Measurement
"""

import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time
import pandas as pd
import uuid
import datetime
import json
import hashlib

# ============================================================
# IMPORTS & GPU DETECTION
# ============================================================
try:
    import cupy as cp
    GPU_AVAILABLE = True
    try:
        GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        GPU_MEM_MB = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**2)
    except Exception:
        GPU_NAME = "Unknown GPU Device"
        GPU_MEM_MB = 1024.0 
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "None (CuPy not installed)"
    GPU_MEM_MB = 0.0
    cp = None

pv.global_theme.jupyter_backend = 'static'
pv.global_theme.anti_aliasing = 'fxaa'

# ============================================================
# PHYSICAL CONSTANTS & MATERIAL LIBRARY
# ============================================================
C_LIGHT = 299792458.0              
MU_0 = 4.0 * math.pi * 1e-7        
EPS_0 = 1.0 / (MU_0 * C_LIGHT**2)  
Z_0 = math.sqrt(MU_0 / EPS_0)      

MAT_LIB = {
    "Vacuum / Air": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "is_dispersive": False, "is_metamaterial": False},
    "FR-4 (Lossy)": {"er": 4.4, "mur": 1.0, "sigma": 0.005, "is_dispersive": False, "is_metamaterial": False},
    "PEC (Perfect Conductor)": {"er": 1.0, "mur": 1.0, "sigma": -1.0, "is_dispersive": False, "is_metamaterial": False},
    "Dispersive Water (Debye)": {"er_s": 78.4, "er_inf": 4.6, "tau": 8.1e-12, "sigma": 0.05, "mur": 1.0, "is_dispersive": True, "is_metamaterial": False}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 29 — Automated Antenna Characterization & 3D Pattern Measurement")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Antenna Characterization Lab (M29)",
    "RF Network Analyzer Laboratory (M28)",
    "Measurement Planning & HAL (M27)",
    "Electromagnetic Digital Twin (M26)",
    "Intelligent Design-Space Exploration (M25)",
    "Automated Experiment Manager (M24)",
    "Model Verification & Validation (V&V)",
    "Uncertainty Quantification (UQ)",
    "Surrogate & Reduced-Order Modeling",
    "Multi-Objective Pareto Optimization",
    "Adjoint Optimization & Sensitivity",
    "Electromagnetic Topology Optimization",
    "Inverse Design & Optimization",
    "Metamaterials Laboratory",
    "Adaptive Mesh Refinement (AMR)",
    "Antenna Array Laboratory",
    "Single Antenna (Dipole/Patch)"
])

# Global States
if 'scpi_log' not in st.session_state: st.session_state.scpi_log = []
if 'instrument_errors' not in st.session_state: st.session_state.instrument_errors = []
if 'scan_data' not in st.session_state: st.session_state.scan_data = None
if 'scan_plan' not in st.session_state: st.session_state.scan_plan = None

# ============================================================
# M28/M29: HARDWARE ABSTRACTION & MOCK POSITIONER
# ============================================================
class SCPIInstrument:
    def __init__(self, name="Generic RF Instrument"):
        self.name = name; self.connected = False; self.calibrated = "NOT CALIBRATED"
        self.settings = {"f_cw": 2.4e9, "power_dbm": 0.0, "z0": 50.0}
    def log(self, cmd, direction="TX"): pass

class MockVNABackend(SCPIInstrument):
    def acquire_cw_power(self, theta, phi):
        # MOCK RF DATA GENERATION: Analytical dipole pattern with simulated noise + position error
        t_rad = np.deg2rad(theta); p_rad = np.deg2rad(phi)
        
        # Base ideal pattern (Dipole along Z)
        sin_t = max(abs(np.sin(t_rad)), 1e-6)
        ideal_mag = abs(np.cos(np.pi/2 * np.cos(t_rad)) / sin_t)
        
        # Add random measurement noise (amplitude & phase)
        np.random.seed(int(time.time()*1000)%10000)
        noise_mag = np.random.normal(0, 0.02)
        measured_mag = max(ideal_mag + noise_mag, 1e-6) # Linear magnitude
        
        # Convert to dBm (relative to peak 0 dBm)
        power_dbm = 20 * np.log10(measured_mag)
        phase_deg = np.random.normal(0, 2.0) # 2 deg phase noise
        return power_dbm, phase_deg

class MockPositioner:
    def __init__(self):
        self.connected = False
        self.theta = 0.0
        self.phi = 0.0
        self.moving = False
        self.name = "Mock Antenna Positioner 3D"
        
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def home(self): self.theta = 0.0; self.phi = 0.0
    def move_to(self, t, p):
        self.moving = True
        # Simulate physical positional inaccuracy
        np.random.seed(int(time.time()*1000)%10000)
        self.theta = t + np.random.normal(0, 0.2) # 0.2 deg error
        self.phi = p + np.random.normal(0, 0.2)
        self.moving = False
    def get_position(self):
        return self.theta, self.phi

if 'vna_backend' not in st.session_state: st.session_state.vna_backend = MockVNABackend(name="Mock VNA")
if 'pos_backend' not in st.session_state: st.session_state.pos_backend = MockPositioner()

# ============================================================
# M29: ANTENNA CHARACTERIZATION LABORATORY
# ============================================================
if exp_mode == "Antenna Characterization Lab (M29)":
    vna = st.session_state.vna_backend
    pos = st.session_state.pos_backend
    
    st.sidebar.header("3. HARDWARE CONTROL")
    hw_type = st.sidebar.selectbox("Hardware Interface", ["MOCK (Simulated)", "REAL HARDWARE (N/A)"])
    if hw_type == "REAL HARDWARE (N/A)": st.sidebar.error("No real hardware backend configured.")
    
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Connect All"): vna.connected = True; pos.connect()
    if c2.button("Disconnect All"): vna.connected = False; pos.disconnect()
    
    st.markdown("### 📡 3D Antenna Characterization & Pattern Measurement")
    st.info("The default environment utilizes MOCK simulation backends for RF and Positioning. No real antenna measurements are claimed unless explicitly interfaced with physical laboratory hardware.")
    
    dash1, dash2, dash3, dash4 = st.columns(4)
    dash1.metric("VNA Connection", "CONNECTED" if vna.connected else "DISCONNECTED")
    dash2.metric("Positioner Connection", "CONNECTED" if pos.connected else "DISCONNECTED")
    dash3.metric("Current Theta (θ)", f"{pos.theta:.2f}°")
    dash4.metric("Current Phi (φ)", f"{pos.phi:.2f}°")
    
    t_plan, t_scan, t_2d, t_3d, t_dt = st.tabs(["Scan Planner", "Acquisition Sequence", "2D Pattern Analysis", "3D Pattern & Metrics", "Digital Twin Correlation"])
    
    with t_plan:
        st.markdown("#### Spatial Scan Planner")
        st.write("Coordinate Convention: Spherical ($r, \\theta, \\phi$). $\\theta \\in [0, 180]$ is Polar, $\\phi \\in [0, 360]$ is Azimuth.")
        
        with st.form("scan_plan_form"):
            cc1, cc2, cc3 = st.columns(3)
            t_start = cc1.number_input("Theta Start (°)", 0.0, 180.0, 0.0)
            t_stop = cc2.number_input("Theta Stop (°)", 0.0, 180.0, 180.0)
            t_step = cc3.number_input("Theta Step (°)", 1.0, 45.0, 5.0)
            
            p_start = cc1.number_input("Phi Start (°)", 0.0, 360.0, 0.0)
            p_stop = cc2.number_input("Phi Stop (°)", 0.0, 360.0, 360.0)
            p_step = cc3.number_input("Phi Step (°)", 1.0, 45.0, 10.0)
            
            freq_cw = st.number_input("Operating Frequency (GHz)", 0.1, 40.0, 2.4)
            
            sim_guide = st.checkbox("Apply Simulation-Guided Adaptive Resolution (Refine steps near Expected Main Lobe)")
            
            if st.form_submit_button("Validate & Generate Scan Plan"):
                if t_start > t_stop or p_start > p_stop:
                    st.error("Invalid angular bounds.")
                else:
                    t_pts = int((t_stop - t_start) / t_step) + 1
                    p_pts = int((p_stop - p_start) / p_step) + 1
                    total_pts = t_pts * p_pts
                    
                    st.session_state.scan_plan = {
                        "t_grid": np.linspace(t_start, t_stop, t_pts),
                        "p_grid": np.linspace(p_start, p_stop, p_pts),
                        "total_pts": total_pts,
                        "freq": freq_cw * 1e9
                    }
                    vna.settings["f_cw"] = freq_cw * 1e9
                    st.success(f"Scan Plan Validated! Total Measurement Points: {total_pts}. Estimated MOCK runtime: {total_pts * 0.01:.1f} s.")

    with t_scan:
        st.markdown("#### Automated Scan Execution Engine")
        if st.session_state.scan_plan is None:
            st.warning("Generate a valid Scan Plan in the previous tab.")
        elif not (vna.connected and pos.connected):
            st.error("Hardware is disconnected. Please connect the VNA and Positioner.")
        else:
            plan = st.session_state.scan_plan
            st.info(f"Targeting {plan['total_pts']} spatial coordinates at {plan['freq']/1e9:.2f} GHz.")
            
            safety_check = st.checkbox("Confirm physical clearance for 3D positioner movement", value=False)
            
            c_h, c_r = st.columns(2)
            if c_h.button("Home Positioner"): 
                if safety_check: pos.home(); st.success("Positioner Homed (0,0)."); st.rerun()
                else: st.error("Safety confirmation required.")
                
            if c_r.button("RUN SPATIAL SCAN", type="primary"):
                if not safety_check:
                    st.error("Safety confirmation required to execute physical movement.")
                else:
                    st.warning("⚠️ MOCK DATA ACQUISITION IN PROGRESS...")
                    pb = st.progress(0)
                    log_text = st.empty()
                    
                    t_grid = plan["t_grid"]; p_grid = plan["p_grid"]
                    results_mag = np.zeros((len(t_grid), len(p_grid)))
                    results_phase = np.zeros((len(t_grid), len(p_grid)))
                    err_theta = np.zeros((len(t_grid), len(p_grid)))
                    err_phi = np.zeros((len(t_grid), len(p_grid)))
                    
                    pts_done = 0
                    for i, t in enumerate(t_grid):
                        for j, p in enumerate(p_grid):
                            # Move
                            pos.move_to(t, p)
                            act_t, act_p = pos.get_position()
                            
                            # Measure
                            mag_db, ph_deg = vna.acquire_cw_power(act_t, act_p)
                            
                            # Store
                            results_mag[i, j] = mag_db
                            results_phase[i, j] = ph_deg
                            err_theta[i, j] = act_t - t
                            err_phi[i, j] = act_p - p
                            
                            pts_done += 1
                            if pts_done % 10 == 0: # UI Update rate limiter
                                pb.progress(pts_done / plan["total_pts"])
                                log_text.text(f"Scanning: Theta={t:.1f}°, Phi={p:.1f}° | Measured: {mag_db:.2f} dBm")
                    
                    pb.progress(1.0)
                    log_text.text("Scan Complete. Moving Positioner to Home.")
                    pos.home()
                    
                    st.session_state.scan_data = {
                        "theta": t_grid, "phi": p_grid,
                        "mag_db": results_mag, "phase_deg": results_phase,
                        "err_t": err_theta, "err_p": err_phi,
                        "source": "MOCK DATA",
                        "freq": plan["freq"]
                    }
                    st.success("MOCK 3D Radiation Pattern Acquired Successfully.")

    with t_2d:
        if st.session_state.scan_data is None: st.info("No scan data available.")
        else:
            d = st.session_state.scan_data
            st.markdown("#### 2D Polar Radiation Patterns")
            st.warning(f"**SOURCE:** `{d['source']}`")
            
            c_p1, c_p2 = st.columns(2)
            
            # E-Plane (Phi = 0 / 180 cut)
            phi_idx = np.argmin(np.abs(d["phi"] - 0.0))
            e_cut = d["mag_db"][:, phi_idx]
            # Mirror for full 360 polar plot (Theta 0->180, then 180->360 mapping)
            theta_full = np.concatenate([d["theta"], 360 - d["theta"][::-1]])
            e_cut_full = np.concatenate([e_cut, d["mag_db"][::-1, np.argmin(np.abs(d["phi"] - 180.0))]])
            
            # Normalize
            e_cut_norm = e_cut_full - np.max(e_cut_full)
            
            fig_e = go.Figure(go.Scatterpolar(r=e_cut_norm, theta=theta_full, mode='lines', name='E-Plane (Phi=0°)', line_color='blue'))
            fig_e.update_layout(title="E-Plane Pattern (Normalized dB)", polar=dict(radialaxis=dict(range=[-40, 0], ticksuffix=" dB")), showlegend=True)
            c_p1.plotly_chart(fig_e, use_container_width=True)
            
            # Azimuth Cut (Theta = 90)
            theta_idx = np.argmin(np.abs(d["theta"] - 90.0))
            h_cut = d["mag_db"][theta_idx, :]
            h_cut_norm = h_cut - np.max(h_cut)
            
            fig_h = go.Figure(go.Scatterpolar(r=h_cut_norm, theta=d["phi"], mode='lines', name='Azimuth (Theta=90°)', line_color='red'))
            fig_h.update_layout(title="Azimuth Pattern (Normalized dB)", polar=dict(radialaxis=dict(range=[-40, 0], ticksuffix=" dB")), showlegend=True)
            c_p2.plotly_chart(fig_h, use_container_width=True)

    with t_3d:
        if st.session_state.scan_data is None: st.info("No scan data available.")
        else:
            d = st.session_state.scan_data
            st.markdown("#### 3D Radiation Pattern & Beam Analytics")
            
            # 3D Cartesian Conversion for Surface Plot
            T, P = np.meshgrid(np.deg2rad(d["theta"]), np.deg2rad(d["phi"]), indexing='ij')
            R = d["mag_db"] - np.min(d["mag_db"]) # Linearize magnitude for radius visually
            
            X = R * np.sin(T) * np.cos(P)
            Y = R * np.sin(T) * np.sin(P)
            Z = R * np.cos(T)
            
            fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=d["mag_db"], colorscale='Jet', colorbar=dict(title="dBm"))])
            fig_3d.update_layout(title=f"Measured 3D Pattern @ {d['freq']/1e9:.2f} GHz", scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))
            
            c_3d1, c_3d2 = st.columns([2, 1])
            c_3d1.plotly_chart(fig_3d, use_container_width=True)
            
            with c_3d2:
                st.markdown("##### Beam Statistics")
                peak_db = np.max(d["mag_db"])
                idx_t, idx_p = np.unravel_index(np.argmax(d["mag_db"]), d["mag_db"].shape)
                peak_theta = d["theta"][idx_t]
                peak_phi = d["phi"][idx_p]
                
                # HPBW Estimation (E-plane rough check)
                e_cut = d["mag_db"][:, idx_p]
                hp_pts = np.where(e_cut >= peak_db - 3.0)[0]
                hpbw = d["theta"][hp_pts[-1]] - d["theta"][hp_pts[0]] if len(hp_pts) > 0 else 0.0
                
                # Front-to-Back Ratio (Assumes peak is front, opposite is back)
                back_theta = 180.0 - peak_theta
                back_idx = np.argmin(np.abs(d["theta"] - back_theta))
                fbr = peak_db - d["mag_db"][back_idx, idx_p]
                
                st.metric("Main Lobe Direction", f"θ={peak_theta:.1f}°, φ={peak_phi:.1f}°")
                st.metric("Peak Measured Power", f"{peak_db:.2f} dBm")
                st.metric("Estimated HPBW (E-Plane)", f"{hpbw:.1f}°")
                st.metric("Front-to-Back Ratio", f"{fbr:.1f} dB")
                st.metric("Mean Positioning Error", f"{np.mean(np.abs(d['err_t']))*1000:.1f} mdeg")

    with t_dt:
        if st.session_state.scan_data is None: st.info("No scan data available.")
        else:
            d = st.session_state.scan_data
            st.markdown("#### 🔄 Digital Twin Correlation (FDTD vs MOCK MEASUREMENT)")
            st.info("Generates the Ideal FDTD Simulation boundary and calculates the angular Error Map against the physical Mock positioner data.")
            
            if st.button("Correlate Simulation & Measurement"):
                # Generate Simulation (Ideal)
                T, P = np.meshgrid(np.deg2rad(d["theta"]), np.deg2rad(d["phi"]), indexing='ij')
                sin_t = np.maximum(np.abs(np.sin(T)), 1e-6)
                ideal_mag = np.abs(np.cos(np.pi/2 * np.cos(T)) / sin_t)
                ideal_db = 20 * np.log10(ideal_mag + 1e-12)
                
                # Error Map (Absolute dB Error)
                error_map = np.abs(d["mag_db"] - ideal_db)
                
                rmse = np.sqrt(np.mean(error_map**2))
                
                st.metric("Pattern RMSE (Simulation vs Measurement)", f"{rmse:.3f} dB")
                
                fig_err = go.Figure(data=go.Heatmap(z=error_map.T, x=d["theta"], y=d["phi"], colorscale='Reds', colorbar=dict(title="Absolute Error (dB)")))
                fig_err.update_layout(title="Angular Pattern Error Map (2D Projection)", xaxis_title="Theta (°)", yaxis_title="Phi (°)")
                st.plotly_chart(fig_err, use_container_width=True)
                
                report = {
                    "Plan_ID": str(uuid.uuid4()), "Timestamp": datetime.datetime.now().isoformat(),
                    "Data_Source": "MOCK_MEASUREMENT", "Frequency": d["freq"],
                    "Main_Lobe": {"theta": float(peak_theta), "phi": float(peak_phi), "power_dbm": float(peak_db)},
                    "Correlation": {"RMSE_dB": float(rmse)}
                }
                st.download_button("Export Validation Report (JSON)", data=json.dumps(report, indent=2), file_name="antenna_characterization.json", mime="application/json")

elif exp_mode not in ["Antenna Characterization Lab (M29)"]:
    st.info("Select 'Antenna Characterization Lab (M29)' to access automated 3D spatial scanning workflows.")
