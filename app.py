"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 27 — Automated Electromagnetic Measurement Planning & Hardware-Abstraction
"""

import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time
import pandas as pd
import random
import uuid
import datetime
import json
import hashlib
import sys

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
    "High-K Dielectric (Topology)": {"er": 9.0, "mur": 1.0, "sigma": 0.0, "is_dispersive": False, "is_metamaterial": False},
    "PEC (Perfect Conductor)": {"er": 1.0, "mur": 1.0, "sigma": -1.0, "is_dispersive": False, "is_metamaterial": False},
    "Dispersive Water (Debye)": {"er_s": 78.4, "er_inf": 4.6, "tau": 8.1e-12, "sigma": 0.05, "mur": 1.0, "is_dispersive": True, "is_metamaterial": False},
    "Negative Epsilon (Drude)": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "w_pe": 2*math.pi*15e9, "g_e": 2*math.pi*0.5e9, "w_pm": 0.0, "g_m": 0.0, "is_dispersive": False, "is_metamaterial": True}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 27 — Measurement Planning & Hardware-Abstraction Interface")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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
if 'exp_db' not in st.session_state: st.session_state.exp_db = []
if 'dt_meas_df' not in st.session_state: st.session_state.dt_meas_df = None
if 'dt_metadata' not in st.session_state: st.session_state.dt_metadata = {}

# ============================================================
# GRID & DOMAIN SETUP
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
Nx = Ny = Nz = 40 if exp_mode not in ["Single Antenna (Dipole/Patch)"] else 80
dx = dy = dz = 0.005 

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
pml_thickness = 10; dt_cfl = 0.9 * (1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)))
dt = dt_cfl

ce1_x = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp1_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ce1_y = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp1_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ce1_z = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp1_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ch2 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cd1_e = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cd2_e = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cd1_m = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cd2_m = np.zeros((Nx, Ny, Nz), dtype=dtype_np)

def get_mat_coeffs(er, sig, tau, eps_s, eps_inf, is_disp, step_dt):
    if is_disp:
        K1 = (2*tau - step_dt) / (2*tau + step_dt); K2 = (EPS_0 * (eps_s - eps_inf) * step_dt) / (2*tau + step_dt)
        A = (EPS_0 * eps_inf / step_dt) + (K2 / step_dt) + (sig / 2); B = (EPS_0 * eps_inf / step_dt) - (K2 / step_dt) - (sig / 2)
        return B/A, 1.0/A, (1.0 - K1)/(A*step_dt), K1, K2
    else:
        if sig < 0: return 0.0, 0.0, 0.0, 0.0, 0.0
        A = (er * EPS_0 / step_dt) + (sig / 2); B = (er * EPS_0 / step_dt) - (sig / 2)
        return B/A, 1.0/A, 0.0, 0.0, 0.0

def apply_material_block(x1, x2, y1, y2, z1, z2, mat, step_dt=dt):
    sig = mat.get("sigma", 0.0); mur = mat.get("mur", 1.0); is_disp = mat.get("is_dispersive", False)
    er_x = mat.get("er_x", mat.get("er", 1.0)); er_y = mat.get("er_y", mat.get("er", 1.0)); er_z = mat.get("er_z", mat.get("er", 1.0))
    c1x, c2x, c3x, p1x, p2x = get_mat_coeffs(er_x, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, step_dt)
    ce1_x[x1:x2+1, y1:y2+1, z1:z2+1] = c1x; ce2_x[x1:x2+1, y1:y2+1, z1:z2+1] = c2x; ce3_x[x1:x2+1, y1:y2+1, z1:z2+1] = c3x; cp1_x[x1:x2+1, y1:y2+1, z1:z2+1] = p1x; cp2_x[x1:x2+1, y1:y2+1, z1:z2+1] = p2x
    c1y, c2y, c3y, p1y, p2y = get_mat_coeffs(er_y, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, step_dt)
    ce1_y[x1:x2+1, y1:y2+1, z1:z2+1] = c1y; ce2_y[x1:x2+1, y1:y2+1, z1:z2+1] = c2y; ce3_y[x1:x2+1, y1:y2+1, z1:z2+1] = c3y; cp1_y[x1:x2+1, y1:y2+1, z1:z2+1] = p1y; cp2_y[x1:x2+1, y1:y2+1, z1:z2+1] = p2y
    c1z, c2z, c3z, p1z, p2z = get_mat_coeffs(er_z, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, step_dt)
    ce1_z[x1:x2+1, y1:y2+1, z1:z2+1] = c1z; ce2_z[x1:x2+1, y1:y2+1, z1:z2+1] = c2z; ce3_z[x1:x2+1, y1:y2+1, z1:z2+1] = c3z; cp1_z[x1:x2+1, y1:y2+1, z1:z2+1] = p1z; cp2_z[x1:x2+1, y1:y2+1, z1:z2+1] = p2z
    ch2[x1:x2+1, y1:y2+1, z1:z2+1] = step_dt / (mur * MU_0)

def reset_materials(step_dt=dt):
    ce1_x.fill(1.0); ce2_x.fill(0.0); ce3_x.fill(0.0); cp1_x.fill(0.0); cp2_x.fill(0.0)
    ce1_y.fill(1.0); ce2_y.fill(0.0); ce3_y.fill(0.0); cp1_y.fill(0.0); cp2_y.fill(0.0)
    ce1_z.fill(1.0); ce2_z.fill(0.0); ce3_z.fill(0.0); cp1_z.fill(0.0); cp2_z.fill(0.0)
    ch2.fill(0.0); cd1_e.fill(0.0); cd2_e.fill(0.0); cd1_m.fill(0.0); cd2_m.fill(0.0)
    apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"], step_dt=step_dt)

reset_materials()

# Variables
num_steps = 300; freq_hz = 0.0
nf2ff_active = False; num_elements = 1
i_min = j_min = k_min = pml_thickness + 2
i_max = Nx - 1 - pml_thickness - 2; j_max = Ny - 1 - pml_thickness - 2; k_max = Nz - 1 - pml_thickness - 2

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
memory_mb = mem_base_bytes / (1024 * 1024)
st.sidebar.markdown(f"**Est. Memory Req:** `{memory_mb:.2f} MB`")
if active_backend == "GPU" and memory_mb > (GPU_MEM_MB * 0.9): st.stop()
elif active_backend == "CPU" and memory_mb > 3000: st.stop()

# ============================================================
# M27: HARDWARE ABSTRACTION LAYER (HAL) & MOCK BACKEND
# ============================================================
class HardwareBackend:
    """Abstract conceptual interface for laboratory instruments."""
    def connect(self): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self, plan): pass
    def acquire(self): pass
    def get_status(self): pass

class MockVNABackend(HardwareBackend):
    def __init__(self):
        self.status = "DISCONNECTED"
        self.cal_state = "NOT CALIBRATED"
        self.capabilities = {"min_f": 1e9, "max_f": 18e9, "max_pts": 10001, "max_power": 10} # Hz, dBm
        self.current_plan = None
        self.name = "Mock VNA E5071C-Sim"

    def connect(self):
        time.sleep(0.5)
        self.status = "IDLE"
        return True

    def disconnect(self):
        self.status = "DISCONNECTED"
        self.cal_state = "NOT CALIBRATED"
        return True

    def calibrate(self):
        self.status = "CONFIGURING"
        time.sleep(1.0)
        self.cal_state = "CALIBRATED"
        self.status = "IDLE"

    def configure(self, plan):
        # Safety & Capability Validation
        if plan['f_start'] < self.capabilities['min_f'] or plan['f_stop'] > self.capabilities['max_f']:
            return False, "Frequency limits exceeded."
        if plan['points'] > self.capabilities['max_pts']:
            return False, "Maximum sweep points exceeded."
        if plan['power'] > self.capabilities['max_power']:
            return False, "Maximum power limit exceeded. Safety triggered."
        
        self.current_plan = plan
        return True, "Valid configuration."

    def acquire(self):
        self.status = "MEASURING"
        time.sleep(0.5) # Simulate hardware sweep time
        
        # MOCK DATA GENERATION: Simulate a realistic S11 resonance
        np.random.seed(self.current_plan.get("seed", 42) + int(time.time()*1000)%10000)
        freqs = np.linspace(self.current_plan['f_start'], self.current_plan['f_stop'], self.current_plan['points'])
        
        # Artificial resonance at 5 GHz
        f_res = 5e9
        base_mag = -3.0 - 25.0 * np.exp(-((freqs - f_res)/(0.2e9))**2) # dB
        noise_floor = -60.0
        
        # Add thermal/measurement noise
        noise = np.random.normal(0, 0.8, len(freqs))
        mag = np.maximum(base_mag + noise, noise_floor)
        
        self.status = "IDLE"
        return freqs, mag

    def get_status(self):
        return {"status": self.status, "calibration": self.cal_state, "name": self.name}

# Initialize HAL in session state
if 'hal_vna' not in st.session_state:
    st.session_state.hal_vna = MockVNABackend()

# ============================================================
# M27: MEASUREMENT PLANNING LABORATORY
# ============================================================
if exp_mode == "Measurement Planning & HAL (M27)":
    st.sidebar.header("3. MEASUREMENT CONTROL")
    vna = st.session_state.hal_vna
    
    # HARDWARE SELECTION
    hw_choice = st.sidebar.selectbox("Hardware Interface", ["MOCK (Simulated)", "REAL / PLUGIN"])
    if hw_choice == "REAL / PLUGIN":
        st.sidebar.error("No real hardware backend is currently configured.")
    
    # CONNECTION CONTROL
    c_btn1, c_btn2 = st.sidebar.columns(2)
    if c_btn1.button("Connect") and hw_choice == "MOCK (Simulated)": vna.connect()
    if c_btn2.button("Disconnect"): vna.disconnect()
    
    # STATUS DASHBOARD
    st.markdown("### 🎛️ Instrument Hardware Dashboard")
    status_info = vna.get_status()
    
    dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
    dash_col1.metric("Backend Source", "MOCK" if hw_choice == "MOCK (Simulated)" else "UNAVAILABLE")
    dash_col2.metric("Connection", status_info['status'], "Active" if status_info['status'] != "DISCONNECTED" else "Offline", delta_color="normal" if status_info['status'] != "DISCONNECTED" else "inverse")
    dash_col3.metric("Instrument Model", status_info['name'] if status_info['status'] != "DISCONNECTED" else "None")
    dash_col4.metric("Calibration State", status_info['calibration'], "Valid" if status_info['calibration'] == "CALIBRATED" else "Invalid", delta_color="normal" if status_info['calibration'] == "CALIBRATED" else "inverse")
    
    st.markdown("---")
    
    # WORKFLOW TABS
    t1, t2, t3 = st.tabs(["1. Measurement Plan", "2. Sequence Acquisition", "3. Quality & Correlation"])
    
    with t1:
        st.markdown("#### 📋 Define Measurement Plan (S-Parameter Sweep)")
        with st.form("meas_plan_form"):
            col_p1, col_p2, col_p3 = st.columns(3)
            f_start = col_p1.number_input("Start Frequency (GHz)", min_value=0.1, max_value=40.0, value=2.0, step=0.5) * 1e9
            f_stop = col_p2.number_input("Stop Frequency (GHz)", min_value=0.1, max_value=40.0, value=8.0, step=0.5) * 1e9
            pts = col_p3.number_input("Sweep Points", min_value=11, max_value=20001, value=201, step=100)
            
            pwr = st.number_input("Source Power (dBm)", min_value=-50.0, max_value=20.0, value=0.0, step=1.0)
            reps = st.number_input("Measurement Repetitions (Averaging/Quality)", min_value=1, max_value=10, value=3)
            
            submitted = st.form_submit_button("Validate & Create Plan")
            if submitted:
                plan = {
                    "plan_id": str(uuid.uuid4()), "f_start": f_start, "f_stop": f_stop, 
                    "points": pts, "power": pwr, "reps": reps, "type": "S11_Sweep", "seed": 42
                }
                valid, msg = vna.configure(plan)
                if valid:
                    st.session_state['active_plan'] = plan
                    st.success(f"Plan Validated: {msg} | Hash: {hashlib.md5(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:8]}")
                else:
                    st.error(f"Plan Rejected by Safety Limits: {msg}")

    with t2:
        st.markdown("#### ⚙️ Automated Sequence Engine")
        if 'active_plan' not in st.session_state:
            st.warning("No validated measurement plan exists. Complete Stage 1.")
        elif status_info['status'] == "DISCONNECTED":
            st.error("Hardware is disconnected. Please connect the instrument.")
        else:
            plan = st.session_state['active_plan']
            st.json(plan, expanded=False)
            
            col_seq1, col_seq2 = st.columns(2)
            if col_seq1.button("Execute Calibration Routine"):
                with st.spinner("Running OPEN/SHORT/LOAD Calibration Sequence..."):
                    vna.calibrate()
                st.success("Calibration applied successfully.")
                st.rerun()
                
            if col_seq2.button("Trigger Acquisition Sequence", type="primary"):
                if status_info['calibration'] != "CALIBRATED":
                    st.warning("Warning: Executing Uncalibrated Measurement.")
                    
                st.markdown("##### Execution Log:")
                progress_bar = st.progress(0)
                log_text = st.empty()
                
                acquisitions = []
                for i in range(plan['reps']):
                    log_text.text(f"ACQUIRE: Triggering Sweep [{i+1}/{plan['reps']}]...")
                    f_data, mag_data = vna.acquire()
                    acquisitions.append(mag_data)
                    progress_bar.progress((i+1)/plan['reps'])
                
                log_text.text("STORE: Aggregating Repetitions...")
                f_data = np.array(f_data)
                acquisitions = np.array(acquisitions)
                
                # Statistical Quality Processing
                mean_mag = np.mean(acquisitions, axis=0)
                std_mag = np.std(acquisitions, axis=0)
                max_dev = np.max(std_mag)
                snr_est = np.abs(np.mean(mean_mag)) / (np.mean(std_mag) + 1e-12)
                
                qual_score = "POOR"
                if max_dev < 1.0 and snr_est > 10: qual_score = "GOOD"
                elif max_dev < 3.0: qual_score = "LIMITED"
                
                st.session_state['meas_results'] = {
                    "f_data": f_data, "mean_mag": mean_mag, "std_mag": std_mag,
                    "max_dev": max_dev, "snr": snr_est, "quality": qual_score,
                    "source": "MOCK" if hw_choice == "MOCK (Simulated)" else "MEASURED",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                st.success(f"Acquisition Sequence Complete. MOCK DATA acquired.")

    with t3:
        st.markdown("#### 📉 Measurement Quality & Digital Twin Export")
        if 'meas_results' not in st.session_state:
            st.info("No data acquired yet.")
        else:
            res = st.session_state['meas_results']
            st.warning(f"**DATA SOURCE:** `{res['source']} MEASUREMENT`")
            
            cq1, cq2, cq3 = st.columns(3)
            cq1.metric("Data Quality Diagnostic", res['quality'], "Acceptable" if res['quality'] == "GOOD" else "Review Needed", delta_color="normal" if res['quality'] == "GOOD" else "inverse")
            cq2.metric("Max Repeatability Deviation", f"±{res['max_dev']:.2f} dB")
            cq3.metric("Estimated SNR", f"{res['snr']:.1f} dB")
            
            fig = go.Figure()
            # Mean Signal
            fig.add_trace(go.Scatter(x=res['f_data']/1e9, y=res['mean_mag'], mode='lines', name=f"{res['source']} Mean Signal", line=dict(color='blue')))
            # Confidence/Noise Band
            fig.add_trace(go.Scatter(x=np.concatenate([res['f_data']/1e9, res['f_data'][::-1]/1e9]),
                                     y=np.concatenate([res['mean_mag'] + 2*res['std_mag'], (res['mean_mag'] - 2*res['std_mag'])[::-1]]),
                                     fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name="±2σ Noise Band"))
            
            fig.update_layout(title="Acquired Signal & Noise Analysis", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### Integration")
            if st.button("Export to M26 Digital Twin Database"):
                # Format into Pandas DF as expected by M26 CSV loader
                df_export = pd.DataFrame({"Frequency_Hz": res['f_data'], "Magnitude_dB": res['mean_mag']})
                st.session_state.dt_meas_df = df_export
                st.session_state.dt_metadata = {
                    "instrument": vna.name, "cal_status": status_info['calibration'], "date": res['timestamp'], 
                    "operator": "Auto HAL", "freq_col": "Frequency_Hz", "mag_col": "Magnitude_dB", 
                    "f_mult": 1.0, "is_db": True
                }
                st.success("MOCK Dataset transferred! Switch to 'Electromagnetic Digital Twin (M26)' Mode -> Stage 2 to correlate against FDTD physics.")

elif exp_mode not in ["Measurement Planning & HAL (M27)"]:
    st.info("Select 'Measurement Planning & HAL (M27)' mode to configure automated hardware sequences.")
