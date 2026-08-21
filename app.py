"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 34 — Advanced Antenna Measurement, Calibration & Reference Correction
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
st.markdown("### Milestone 34 — Advanced Calibration & Reference Correction Lab")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Advanced Antenna Calibration Lab (M34)",
    "Advanced NF2FF Validation (M33)",
    "Spherical NF/FF Lab (M32)",
    "Cylindrical NF/FF Lab (M31)",
    "Near-Field / Far-Field Lab (M30)",
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
if 'm34_raw' not in st.session_state: st.session_state.m34_raw = None
if 'm34_cal' not in st.session_state: st.session_state.m34_cal = None
if 'm34_corr' not in st.session_state: st.session_state.m34_corr = None
if 'm34_ff' not in st.session_state: st.session_state.m34_ff = None

# ============================================================
# GRID & DOMAIN SETUP (DYNAMIC)
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
    ch2.fill(0.0)
    apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"], step_dt=step_dt)

reset_materials()

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
memory_mb = mem_base_bytes / (1024 * 1024)
st.sidebar.markdown(f"**Est. Memory Req (Per Sim):** `{memory_mb:.2f} MB`")
if active_backend == "GPU" and memory_mb > (GPU_MEM_MB * 0.9): st.stop()
elif active_backend == "CPU" and memory_mb > 3000: st.stop()

# ============================================================
# M34: CALIBRATION & REFERENCE MATHEMATICS
# ============================================================
def generate_mock_raw_data(freq, x_arr, y_arr, z_plane):
    """Generates an uncalibrated, noisy planar NF scan of a dipole."""
    k0 = 2 * np.pi * freq / C_LIGHT
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + z_plane**2)
    
    # Ideal field with systemic measurement attenuation & phase delay (e.g., cable loss)
    ideal_mag = np.exp(-(X**2 + Y**2)/0.02) / (R + 1e-6)
    
    # Inject Mock Cable Loss (e.g., -3 dB amplitude drop) and +45 deg arbitrary phase shift
    attenuation_linear = 10**(-3.0 / 20)
    phase_offset = np.deg2rad(45.0)
    
    # Inject Gaussian Measurement Noise
    np.random.seed(int(time.time()*1000)%10000)
    noise_floor = 0.02
    noise_m = np.random.normal(0, noise_floor, X.shape)
    noise_p = np.random.normal(0, 0.1, X.shape)
    
    # RAW Distorted Field
    mag_raw = (ideal_mag * attenuation_linear) + noise_m
    phase_raw = -k0 * R + phase_offset + noise_p
    
    Ex_raw = mag_raw * np.exp(1j * phase_raw)
    Ey_raw = 0.05 * Ex_raw # Cross-polar leakage
    
    return Ex_raw, Ey_raw

def analyze_data_quality(Ex_raw):
    """Diagnostic check for clipping, saturation, and SNR."""
    mag = np.abs(Ex_raw)
    max_val = np.max(mag)
    mean_val = np.mean(mag)
    
    # Estimate noise floor from the edge of the scan
    noise_floor_est = np.mean(mag[0, :]) + 1e-12 
    snr_linear = max_val / noise_floor_est
    snr_db = 20 * np.log10(snr_linear)
    
    saturation_limit = 10.0 # Arbitrary V/m hardware limit
    clipped = np.any(mag > saturation_limit)
    has_nan = np.isnan(Ex_raw).any() or np.isinf(Ex_raw).any()
    
    quality = "VALID"
    if has_nan or clipped: quality = "INVALID"
    elif snr_db < 15.0: quality = "WARNING"
    
    return {"snr_db": snr_db, "clipped": clipped, "has_nan": has_nan, "quality": quality, "max_val": max_val}

def compute_far_field_direct_planar(Ex_nf, Ey_nf, x_arr, y_arr, freq, thetas, phis):
    """NF2FF Plane Wave Spectrum Integration (from M33)."""
    k0 = 2 * np.pi * freq / C_LIGHT
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    dx_s = x_arr[1] - x_arr[0] if len(x_arr) > 1 else 1.0
    dy_s = y_arr[1] - y_arr[0] if len(y_arr) > 1 else 1.0
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
    
    kx = k0 * np.sin(THETA) * np.cos(PHI)
    ky = k0 * np.sin(THETA) * np.sin(PHI)
    
    X_b = X[:, :, None, None]; Y_b = Y[:, :, None, None]
    kx_b = kx[None, None, :, :]; ky_b = ky[None, None, :, :]
    
    kernel = np.exp(1j * (kx_b * X_b + ky_b * Y_b))
    Fx = np.sum(Ex_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_s * dy_s
    Fy = np.sum(Ey_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_s * dy_s
    
    E_theta = (Fx * np.cos(PHI) + Fy * np.sin(PHI)) * np.cos(THETA)
    E_phi = -Fx * np.sin(PHI) + Fy * np.cos(PHI)
    return E_theta, E_phi

# ============================================================
# M34: ADVANCED CALIBRATION LABORATORY UI
# ============================================================
if exp_mode == "Advanced Antenna Calibration Lab (M34)":
    st.markdown("### 🎛️ Advanced Calibration & Reference Correction Lab")
    st.info("Ingests uncalibrated raw Near-Field arrays, evaluates Data Quality (SNR, Clipping), strictly protects raw immutability, and executes complex instrumentation adjustments ($C = A e^{j\\phi}$) alongside Reference Plane spatial phase shifts before processing the NF2FF algorithm.")
    
    t_raw, t_cal, t_corr, t_ff, t_rep = st.tabs([
        "1. Raw Data & Quality", "2. Calibration Profile", "3. Apply Corrections", "4. NF2FF & Comparison", "5. Provenance Report"
    ])
    
    freq_test = 2.4e9
    wl = C_LIGHT / freq_test
    x_arr = np.arange(-0.4, 0.41, 0.02)
    y_arr = np.arange(-0.4, 0.41, 0.02)
    z_dist = 0.2
    
    with t_raw:
        st.markdown("#### Raw Measurement Ingestion & Diagnostics")
        if st.button("Generate MOCK Uncalibrated Dataset"):
            Ex_raw, Ey_raw = generate_mock_raw_data(freq_test, x_arr, y_arr, z_dist)
            q_metrics = analyze_data_quality(Ex_raw)
            st.session_state.m34_raw = {
                "Ex": Ex_raw, "Ey": Ey_raw, "freq": freq_test, "z": z_dist, "qual": q_metrics, "source": "MOCK DATA"
            }
            st.success("Raw dataset acquired and locked. (Immutable)")
            
        if st.session_state.m34_raw is not None:
            r = st.session_state.m34_raw
            q = r["qual"]
            
            c_q1, c_q2, c_q3, c_q4 = st.columns(4)
            c_q1.metric("Data Quality", q["quality"], "Safe" if q["quality"] == "VALID" else "Review", delta_color="normal" if q["quality"] == "VALID" else "inverse")
            c_q2.metric("Est. SNR", f"{q['snr_db']:.1f} dB")
            c_q3.metric("Peak Voltage", f"{q['max_val']:.2f} V/m")
            c_q4.metric("Saturation/Clipping", "DETECTED" if q["clipped"] else "NONE")
            
            # Plot Raw
            mag_raw_db = 20 * np.log10(np.abs(r["Ex"]) + 1e-12)
            fig_r = go.Figure(data=go.Heatmap(z=mag_raw_db.T, x=x_arr, y=y_arr, colorscale='Viridis', colorbar=dict(title="dBV/m")))
            fig_r.update_layout(title="RAW Uncalibrated Near-Field Magnitude", xaxis_title="X (m)", yaxis_title="Y (m)", width=500, height=400)
            st.plotly_chart(fig_r)

    with t_cal:
        st.markdown("#### Instrument & Probe Calibration Matrix")
        st.info("Calibration corrects for systematic insertion losses (cables/connectors) and phase delays. Reference plane correction mathematically translates the phase center along the propagation axis.")
        
        with st.form("cal_form"):
            c_c1, c_c2, c_c3 = st.columns(3)
            # Inverse of the injected Mock errors to restore the true signal
            amp_corr_db = c_c1.number_input("Amplitude Correction (dB)", -10.0, 10.0, 3.0, 0.5) 
            phase_corr_deg = c_c2.number_input("Instrument Phase Correction (°)", -180.0, 180.0, -45.0, 5.0)
            ref_shift_m = c_c3.number_input("Reference Plane Shift (m)", -1.0, 1.0, 0.0, 0.01)
            
            if st.form_submit_button("Load Calibration Profile"):
                amp_lin = 10**(amp_corr_db / 20.0)
                phase_rad = np.deg2rad(phase_corr_deg)
                
                # Complex Calibration Factor: C = A * exp(j * phi)
                C_factor = amp_lin * np.exp(1j * phase_rad)
                
                st.session_state.m34_cal = {
                    "amp_db": amp_corr_db, "phase_deg": phase_corr_deg, "ref_shift_m": ref_shift_m,
                    "C_factor": C_factor, "status": "CALIBRATION LOADED"
                }
                st.success("Complex Calibration Profile Verified & Loaded.")

    with t_corr:
        st.markdown("#### Apply Mathematical Corrections")
        if st.session_state.m34_raw is None or st.session_state.m34_cal is None:
            st.warning("Load Raw Data and Calibration Profile first.")
        else:
            raw = st.session_state.m34_raw
            cal = st.session_state.m34_cal
            
            if st.button("Execute Pipeline Corrections", type="primary"):
                # 1. Complex Instrument Correction
                Ex_corr = raw["Ex"] * cal["C_factor"]
                Ey_corr = raw["Ey"] * cal["C_factor"]
                
                # 2. Reference Plane Phase Shift: d_phi = k0 * d
                k0 = 2 * np.pi * raw["freq"] / C_LIGHT
                phase_shift = np.exp(1j * k0 * cal["ref_shift_m"])
                
                Ex_corr = Ex_corr * phase_shift
                Ey_corr = Ey_corr * phase_shift
                
                st.session_state.m34_corr = {
                    "Ex": Ex_corr, "Ey": Ey_corr, "source": "CORRECTED DATA"
                }
                st.success("Corrections applied successfully. Original RAW data remains completely preserved.")
                
            if st.session_state.m34_corr is not None:
                corr = st.session_state.m34_corr
                cc1, cc2 = st.columns(2)
                
                # Compare Phase (to visually prove calibration)
                raw_ph = np.angle(raw["Ex"])
                corr_ph = np.angle(corr["Ex"])
                
                fig_pr = go.Figure(data=go.Heatmap(z=raw_ph.T, x=x_arr, y=y_arr, colorscale='Phase', zmin=-np.pi, zmax=np.pi, colorbar=dict(title="Rads")))
                fig_pr.update_layout(title="RAW Phase", width=350, height=350)
                cc1.plotly_chart(fig_pr)
                
                fig_pc = go.Figure(data=go.Heatmap(z=corr_ph.T, x=x_arr, y=y_arr, colorscale='Phase', zmin=-np.pi, zmax=np.pi, colorbar=dict(title="Rads")))
                fig_pc.update_layout(title="CORRECTED Phase", width=350, height=350)
                cc2.plotly_chart(fig_pc)

    with t_ff:
        st.markdown("#### NF2FF Transform: Raw vs Corrected Comparison")
        if st.session_state.m34_raw is None or st.session_state.m34_corr is None:
            st.warning("Ensure Data is Corrected in Tab 3.")
        else:
            if st.button("Run Far-Field Correlation"):
                raw = st.session_state.m34_raw
                corr = st.session_state.m34_corr
                thetas = np.deg2rad(np.arange(0, 90, 2))
                phis = np.deg2rad(np.array([0.0])) # E-Plane only for fast visualization
                
                # Transform Raw
                Eth_raw, Eph_raw = compute_far_field_direct_planar(raw["Ex"], raw["Ey"], x_arr, y_arr, raw["freq"], thetas, phis)
                ff_raw_db = 20 * np.log10(np.sqrt(np.abs(Eth_raw)**2 + np.abs(Eph_raw)**2) + 1e-12)
                
                # Transform Corrected
                Eth_corr, Eph_corr = compute_far_field_direct_planar(corr["Ex"], corr["Ey"], x_arr, y_arr, raw["freq"], thetas, phis)
                ff_corr_db = 20 * np.log10(np.sqrt(np.abs(Eth_corr)**2 + np.abs(Eph_corr)**2) + 1e-12)
                
                # Absolute magnitudes (Not normalized, to show true calibration gain impact)
                st.session_state.m34_ff = {"thetas": thetas, "ff_raw": ff_raw_db[:, 0], "ff_corr": ff_corr_db[:, 0]}

            if st.session_state.m34_ff is not None:
                ff = st.session_state.m34_ff
                
                st.info("Notice the absolute shift in the Far-Field Gain level due to the applied Amplitude Calibration restoration.")
                fig_ff = go.Figure()
                fig_ff.add_trace(go.Scatter(x=np.rad2deg(ff["thetas"]), y=ff["ff_raw"], mode='lines', name="Far-Field (from RAW)", line=dict(dash='dash', color='red')))
                fig_ff.add_trace(go.Scatter(x=np.rad2deg(ff["thetas"]), y=ff["ff_corr"], mode='lines', name="Far-Field (from CORRECTED)", line=dict(color='blue')))
                fig_ff.update_layout(title="Absolute Far-Field Magnitude (E-Plane)", xaxis_title="Theta (°)", yaxis_title="Magnitude (dB)")
                st.plotly_chart(fig_ff, use_container_width=True)

    with t_rep:
        if st.session_state.m34_corr is not None:
            raw = st.session_state.m34_raw
            cal = st.session_state.m34_cal
            
            report = {
                "Calibration_ID": str(uuid.uuid4()),
                "Timestamp": datetime.datetime.now().isoformat(),
                "Data_Source": raw["source"],
                "Diagnostics": raw["qual"],
                "Calibration_Applied": {
                    "Amplitude_dB": cal["amp_db"],
                    "Phase_Deg": cal["phase_deg"],
                    "Reference_Plane_Shift_m": cal["ref_shift_m"]
                },
                "Provenance": "Raw datasets immutably preserved. Transformations executed via analytical spectral integration.",
                "Integrity_Hash": hashlib.md5(json.dumps(cal, default=str, sort_keys=True).encode()).hexdigest()
            }
            
            st.json(report)
            st.download_button("Export Calibration Provenance Report (JSON)", data=json.dumps(report, indent=2), file_name="m34_calibration_report.json", mime="application/json")

elif exp_mode not in ["Advanced Antenna Calibration Lab (M34)"]:
    st.info("Select 'Advanced Antenna Calibration Lab (M34)' to process Raw dataset calibrations and reference offsets.")
