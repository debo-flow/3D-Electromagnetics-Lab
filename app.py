"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 35 — Advanced Antenna Characterization & Automated Measurement Validation
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
st.markdown("### Milestone 35 — Advanced Antenna Characterization & Automated Validation")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Advanced Antenna Characterization (M35)",
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
if 'm35_dataset' not in st.session_state: st.session_state.m35_dataset = None
if 'm35_char' not in st.session_state: st.session_state.m35_char = None

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
# M35: MATH & CHARACTERIZATION ENGINE
# ============================================================
def generate_m35_mock_dataset():
    """Generates a structured Mock Far-Field & S11 dataset with intentional imperfections to trigger the Characterization Engine."""
    np.random.seed(42)
    freq = 2.4e9
    
    # 1. Mock S-Parameters
    f_arr = np.linspace(2.0e9, 3.0e9, 101)
    # Inject a slight resonance shift (2.42 GHz) to trigger correlation anomaly vs operating 2.4 GHz
    s11_db = -2.0 - 18.0 * np.exp(-((f_arr - 2.42e9)/0.05e9)**2) + np.random.normal(0, 0.5, 101)
    s11_lin = 10**(s11_db / 20.0)
    
    # 2. Mock 3D Far-Field Pattern (with artificial Side Lobes)
    thetas = np.deg2rad(np.arange(0, 181, 2))
    phis = np.deg2rad(np.arange(0, 360, 2))
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
    
    # Z-directed dipole + secondary high-frequency ripple for side lobes + structural noise
    U_ideal = np.abs(np.sin(THETA))**2
    ripple = 0.1 * np.abs(np.sin(5 * THETA))**2
    noise = np.random.uniform(0, 0.01, THETA.shape)
    
    U_mag = U_ideal + ripple + noise
    
    # Construct complex fields representing this power distribution (mostly E_theta)
    E_th = np.sqrt(U_mag) * np.exp(1j * np.random.normal(0, 0.1, THETA.shape))
    E_ph = 0.05 * E_th # Slight cross-polarization
    
    return {
        "freq": freq, "f_arr": f_arr, "s11_mag": s11_lin, "s11_db": s11_db,
        "thetas": thetas, "phis": phis, "E_th": E_th, "E_ph": E_ph, "source": "MOCK DATA"
    }

def compute_1d_beam_metrics(angles, U_cut):
    """Computes Peak, HPBW, Side Lobes, and Nulls for a 1D Radiation Cut."""
    U_db = 10 * np.log10(U_cut / np.max(U_cut) + 1e-12)
    peak_idx = np.argmax(U_cut)
    peak_angle = angles[peak_idx]
    
    # HPBW (-3dB)
    hp_idxs = np.where(U_db >= -3.0)[0]
    hpbw = 0.0
    if len(hp_idxs) > 1:
        # Check if it wraps around or is continuous
        hpbw = abs(angles[hp_idxs[-1]] - angles[hp_idxs[0]])
        if hpbw > 180: hpbw = 360 - hpbw
        
    # Simple Peak/Null Detector
    peaks = []; nulls = []
    for i in range(1, len(U_cut)-1):
        if U_db[i] > U_db[i-1] and U_db[i] > U_db[i+1] and U_db[i] > -40: peaks.append(i)
        if U_db[i] < U_db[i-1] and U_db[i] < U_db[i+1] and U_db[i] < -10: nulls.append(i)
        
    # Side Lobe Level (SLL)
    sll_db = -999.0
    for p in peaks:
        if p != peak_idx and U_db[p] > sll_db:
            if abs(angles[p] - peak_angle) > 20: # Ensure it's not just noise on the main lobe
                sll_db = U_db[p]
                
    return {"peak_angle": peak_angle, "hpbw": hpbw, "sll_db": sll_db if sll_db > -999 else None, "nulls": [angles[n] for n in nulls]}

def characterize_antenna(dataset):
    """Core Engine integrating M35 Physics analytics."""
    char = {"anomalies": [], "validation": "PASS"}
    
    # 1. S-Parameter Analytics
    s11_db = dataset["s11_db"]
    f_arr = dataset["f_arr"]
    min_idx = np.argmin(s11_db)
    f_res = f_arr[min_idx]
    s11_res = s11_db[min_idx]
    vswr_res = (1 + 10**(s11_res/20)) / (1 - 10**(s11_res/20) + 1e-12)
    
    # Bandwidth (-10dB)
    bw_idx = np.where(s11_db <= -10.0)[0]
    bw_hz = (f_arr[bw_idx[-1]] - f_arr[bw_idx[0]]) if len(bw_idx) > 1 else 0.0
    
    mismatch_eff = 1.0 - (10**(s11_res/20))**2
    
    if abs(f_res - dataset["freq"]) > 50e6:
        char["anomalies"].append(f"WARNING: Resonance ({f_res/1e9:.2f} GHz) shifted from Operating Freq ({dataset['freq']/1e9:.2f} GHz).")
        char["validation"] = "WARNING"
        
    char["s_params"] = {"f_res": f_res, "s11_res": s11_res, "vswr": vswr_res, "bw_hz": bw_hz, "mismatch_eff": mismatch_eff}
    
    # 2. 3D Radiation & Directivity Integration (D = 4 * pi * U_max / P_rad)
    E_th = dataset["E_th"]; E_ph = dataset["E_ph"]
    thetas = dataset["thetas"]; phis = dataset["phis"]
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
    
    U = np.abs(E_th)**2 + np.abs(E_ph)**2
    U_max = np.max(U)
    
    dtheta = thetas[1] - thetas[0] if len(thetas) > 1 else 1.0
    dphi = phis[1] - phis[0] if len(phis) > 1 else 1.0
    
    # Integral U * sin(theta) d_theta d_phi
    P_rad = np.sum(U * np.sin(THETA) * dtheta * dphi)
    
    directivity_lin = (4 * np.pi * U_max) / (P_rad + 1e-12)
    directivity_db = 10 * np.log10(directivity_lin + 1e-12)
    
    # Gain = Eff * D
    rad_eff = 0.95 # Assume 95% physical radiation efficiency for MOCK lossy FR4 dipole
    total_eff = rad_eff * mismatch_eff
    gain_db = 10 * np.log10(total_eff * directivity_lin + 1e-12)
    
    idx_t, idx_p = np.unravel_index(np.argmax(U), U.shape)
    peak_theta = np.rad2deg(thetas[idx_t])
    peak_phi = np.rad2deg(phis[idx_p])
    
    char["radiation"] = {
        "D_db": directivity_db, "G_db": gain_db, "rad_eff": rad_eff, "tot_eff": total_eff,
        "peak_theta": peak_theta, "peak_phi": peak_phi
    }
    
    # 3. 2D Cut Analytics (Beamwidth, SLL, FBR)
    # E-Plane (Fixed Phi)
    e_cut = U[:, idx_p]
    e_metrics = compute_1d_beam_metrics(np.rad2deg(thetas), e_cut)
    
    # Front-to-Back Ratio (Assumes peak is front, pi-theta is back)
    back_theta_idx = np.argmin(np.abs(np.rad2deg(thetas) - (180.0 - peak_theta)))
    fbr_db = 10 * np.log10(U_max / (U[back_theta_idx, idx_p] + 1e-12))
    
    if e_metrics["sll_db"] is not None and e_metrics["sll_db"] > -10.0:
        char["anomalies"].append(f"WARNING: High Side-Lobe Level detected ({e_metrics['sll_db']:.1f} dB).")
        char["validation"] = "WARNING"
        
    char["beam"] = {"e_hpbw": e_metrics["hpbw"], "e_sll": e_metrics["sll_db"], "fbr": fbr_db, "nulls": e_metrics["nulls"]}
    
    # 4. Polarization (Axial Ratio at Main Beam)
    E_th_peak = E_th[idx_t, idx_p]
    E_ph_peak = E_ph[idx_t, idx_p]
    
    # Axial ratio (Simplified ratio of orthogonal magnitudes for linear approximation)
    if np.abs(E_ph_peak) < 1e-6: ar_db = 99.9 # Linear
    else:
        ar_lin = np.abs(E_th_peak) / np.abs(E_ph_peak)
        ar_db = 20 * np.log10(max(ar_lin, 1/ar_lin))
    
    char["polarization"] = {"axial_ratio": ar_db, "type": "Linear" if ar_db > 20 else ("Circular" if ar_db <= 3 else "Elliptical")}
    
    return char

# ============================================================
# M35: ADVANCED CHARACTERIZATION LABORATORY UI
# ============================================================
if exp_mode == "Advanced Antenna Characterization (M35)":
    st.markdown("### 📊 Advanced Antenna Characterization & Validation Lab")
    st.info("Ingests validated 3D Far-Field spatial matrices and S-Parameters to compute robust, automated metrology scorecards including true Spherical Directivity Integrals ($4\\pi U_{max} / P_{rad}$), Total Efficiency limits, and Automated Anomaly detection without fabricating physics.")
    
    t_load, t_score, t_rad, t_spar, t_rep = st.tabs([
        "1. Load Dataset", "2. Performance Scorecard", "3. Beam & Polarization", "4. S-Parameter Correlation", "5. Automated Report"
    ])
    
    with t_load:
        st.markdown("#### Antenna Dataset Selection")
        if st.button("Generate Comprehensive Target Dataset (MOCK)", type="primary"):
            with st.spinner("Generating MOCK Far-Field Spatial Tensors & S-Parameters..."):
                st.session_state.m35_dataset = generate_m35_mock_dataset()
                st.session_state.m35_char = characterize_antenna(st.session_state.m35_dataset)
            st.success("MOCK Dataset Generated and Characterization Engine executed successfully.")
            
        if st.session_state.m35_dataset is not None:
            ds = st.session_state.m35_dataset
            st.write(f"**Data Source:** `{ds['source']}`")
            st.write(f"**Operating Frequency:** `{ds['freq']/1e9:.2f} GHz`")
            st.write(f"**Spatial Array:** `{len(ds['thetas'])} x {len(ds['phis'])} points`")

    with t_score:
        if st.session_state.m35_char is None: st.warning("Load a dataset first.")
        else:
            c = st.session_state.m35_char
            st.markdown("#### 🏆 Antenna Characterization Scorecard")
            
            # Validation Banner
            if c["validation"] == "PASS": st.success("OVERALL VALIDATION: PASS - No Critical Anomalies Detected.")
            else: st.warning("OVERALL VALIDATION: WARNING - Anomalies Detected in Dataset.")
            
            # Anomaly Log
            if len(c["anomalies"]) > 0:
                st.markdown("**Automated Anomaly Detection Engine:**")
                for a in c["anomalies"]: st.error(a)
                
            st.markdown("---")
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Peak Direction (θ, φ)", f"{c['radiation']['peak_theta']:.0f}°, {c['radiation']['peak_phi']:.0f}°")
            cc2.metric("Directivity (Spherical Int)", f"{c['radiation']['D_db']:.2f} dBi")
            cc3.metric("Realized Gain", f"{c['radiation']['G_db']:.2f} dB")
            cc4.metric("Total Efficiency", f"{c['radiation']['tot_eff']*100:.1f} %")
            
            cc5, cc6, cc7, cc8 = st.columns(4)
            cc5.metric("HPBW (E-Plane)", f"{c['beam']['e_hpbw']:.1f}°")
            sll_str = f"{c['beam']['e_sll']:.1f} dBc" if c['beam']['e_sll'] is not None else "N/A"
            cc6.metric("Peak Side-Lobe Level", sll_str)
            cc7.metric("Front-to-Back Ratio", f"{c['beam']['fbr']:.1f} dB")
            cc8.metric("Polarization Type", f"{c['polarization']['type']} (AR: {c['polarization']['axial_ratio']:.1f} dB)")

    with t_rad:
        if st.session_state.m35_dataset is None: st.warning("Load a dataset first.")
        else:
            ds = st.session_state.m35_dataset
            c = st.session_state.m35_char
            
            st.markdown("#### 3D Radiation Pattern & 2D E-Plane Cut")
            
            U = np.abs(ds["E_th"])**2 + np.abs(ds["E_ph"])**2
            U_db = 10 * np.log10(U / np.max(U) + 1e-12)
            
            col_r1, col_r2 = st.columns([1, 1])
            
            with col_r1:
                # 3D
                T, P = np.meshgrid(ds["thetas"], ds["phis"], indexing='ij')
                R = np.maximum(U_db + 40, 0)
                X = R * np.sin(T) * np.cos(P); Y = R * np.sin(T) * np.sin(P); Z = R * np.cos(T)
                
                fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=U_db, colorscale='Jet', colorbar=dict(title="Norm dB"))])
                fig_3d.update_layout(title="3D Normalized Far-Field", height=450)
                st.plotly_chart(fig_3d, use_container_width=True)
                
            with col_r2:
                # 2D
                idx_p = np.argmin(np.abs(ds["phis"] - np.deg2rad(c["radiation"]["peak_phi"])))
                e_cut = U_db[:, idx_p]
                
                theta_deg = np.rad2deg(ds["thetas"])
                theta_full = np.concatenate([theta_deg, 360 - theta_deg[::-1]])
                e_cut_full = np.concatenate([e_cut, U_db[::-1, np.argmin(np.abs(ds["phis"] - np.deg2rad((c["radiation"]["peak_phi"]+180)%360)))]])
                
                fig_2d = go.Figure(go.Scatterpolar(r=e_cut_full, theta=theta_full, mode='lines', line_color='blue'))
                fig_2d.update_layout(title=f"E-Plane Cut (Phi={c['radiation']['peak_phi']:.0f}°)", polar=dict(radialaxis=dict(range=[-40, 0], ticksuffix=" dB")), height=450)
                st.plotly_chart(fig_2d, use_container_width=True)

    with t_spar:
        if st.session_state.m35_dataset is None: st.warning("Load a dataset first.")
        else:
            ds = st.session_state.m35_dataset
            c = st.session_state.m35_char
            
            st.markdown("#### Frequency Domain Correlation: S11 & Impedance Match")
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Resonant Frequency", f"{c['s_params']['f_res']/1e9:.3f} GHz")
            c_s2.metric("Return Loss @ Resonance", f"{-c['s_params']['s11_res']:.2f} dB")
            c_s3.metric("-10dB Bandwidth", f"{c['s_params']['bw_hz']/1e6:.1f} MHz")
            
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=ds["f_arr"]/1e9, y=ds["s11_db"], mode='lines', name='S11 (dB)', line_color='green'))
            fig_s.add_hline(y=-10.0, line_dash="dash", line_color="red", annotation_text="-10 dB Bandwidth Limit")
            fig_s.add_vline(x=ds["freq"]/1e9, line_dash="dashdot", line_color="black", annotation_text="Operating Freq")
            fig_s.update_layout(title="S11 Reflection Coefficient Profile", xaxis_title="Frequency (GHz)", yaxis_title="S11 (dB)")
            st.plotly_chart(fig_s, use_container_width=True)

    with t_rep:
        if st.session_state.m35_char is not None:
            st.markdown("#### 🗃️ Characterization Data Export")
            ds = st.session_state.m35_dataset
            c = st.session_state.m35_char
            
            report = {
                "Characterization_ID": str(uuid.uuid4()),
                "Timestamp": datetime.datetime.now().isoformat(),
                "Data_Source": ds["source"],
                "Validation_Status": c["validation"],
                "Frequency_Hz": ds["freq"],
                "Metrics": {
                    "Directivity_dBi": float(c["radiation"]["D_db"]),
                    "Gain_dB": float(c["radiation"]["G_db"]),
                    "Total_Efficiency": float(c["radiation"]["tot_eff"]),
                    "Peak_Direction_Theta": float(c["radiation"]["peak_theta"]),
                    "Peak_Direction_Phi": float(c["radiation"]["peak_phi"]),
                    "HPBW_Deg": float(c["beam"]["e_hpbw"]),
                    "Front_To_Back_Ratio_dB": float(c["beam"]["fbr"])
                },
                "S_Parameters": {
                    "Resonance_Hz": float(c["s_params"]["f_res"]),
                    "VSWR": float(c["s_params"]["vswr"]),
                    "Bandwidth_Hz": float(c["s_params"]["bw_hz"])
                },
                "Anomalies": c["anomalies"]
            }
            
            st.json(report)
            st.download_button("Export Characterization Report (JSON)", data=json.dumps(report, indent=2), file_name="m35_antenna_report.json", mime="application/json")

elif exp_mode not in ["Advanced Antenna Characterization (M35)"]:
    st.info("Select 'Advanced Antenna Characterization (M35)' to run automated Metrology Scorecards and Anomaly Detections.")

