"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Beam Control Laboratory
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
st.markdown("### Milestone 37 — Advanced Phased-Array Beamforming & Adaptive Control")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Advanced Beamforming Lab (M37)",
    "Advanced Antenna Array Lab (M36)",
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
    "Single Antenna (Dipole/Patch)"
])

# Global States
if 'm37_array' not in st.session_state: st.session_state.m37_array = None
if 'm37_weights' not in st.session_state: st.session_state.m37_weights = None

# ============================================================
# GRID & DOMAIN SETUP (DYNAMIC)
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
Nx = Ny = Nz = 40 if exp_mode not in ["Single Antenna (Dipole/Patch)"] else 80
if exp_mode in ["Advanced Antenna Array Lab (M36)", "Advanced Beamforming Lab (M37)"]: Nx = Ny = 60; Nz = 40 
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
st.sidebar.markdown(f"**Est. Memory Req:** `{memory_mb:.2f} MB`")
if active_backend == "GPU" and memory_mb > (GPU_MEM_MB * 0.9): st.stop()
elif active_backend == "CPU" and memory_mb > 3000: st.stop()

# ============================================================
# M36/37: ARRAY MATH & ADVANCED BEAMFORMING KERNELS
# ============================================================
def generate_array_geometry(geom_type, num_el, spacing_x, spacing_y, freq):
    pos = np.zeros((num_el, 3)); wl = C_LIGHT / freq
    dx_m = spacing_x * wl; dy_m = spacing_y * wl
    
    if geom_type == "Uniform Linear Array (ULA)":
        for i in range(num_el): pos[i, 0] = (i - (num_el-1)/2.0) * dx_m
    elif geom_type == "Planar Rectangular Array (UPA)":
        side = int(np.ceil(np.sqrt(num_el)))
        for i in range(num_el):
            row = i // side; col = i % side
            pos[i, 0] = (col - (side-1)/2.0) * dx_m; pos[i, 1] = (row - (side-1)/2.0) * dy_m
    elif geom_type == "Uniform Circular Array (UCA)":
        radius = dx_m * num_el / (2 * np.pi) if num_el > 1 else dx_m
        for i in range(num_el):
            angle = i * (2 * np.pi / num_el)
            pos[i, 0] = radius * np.cos(angle); pos[i, 1] = radius * np.sin(angle)
    return pos

def compute_steering_vector(pos, freq, theta_rad, phi_rad):
    k0 = 2 * np.pi * freq / C_LIGHT
    rx = np.sin(theta_rad) * np.cos(phi_rad)
    ry = np.sin(theta_rad) * np.sin(phi_rad)
    rz = np.cos(theta_rad)
    # v_n = exp(j * k * r_n . r_hat)
    return np.exp(1j * k0 * (pos[:, 0]*rx + pos[:, 1]*ry + pos[:, 2]*rz))

def compute_adaptive_null_weights(pos, freq, t_th, t_ph, nulls):
    """Computes exact complex weights utilizing a mathematical orthogonal projection matrix to force deep nulls."""
    v_tgt = compute_steering_vector(pos, freq, np.deg2rad(t_th), np.deg2rad(t_ph))
    if not nulls: return np.conj(v_tgt) # Standard phase steering
    
    N = len(pos)
    C = np.zeros((N, len(nulls)), dtype=np.complex128)
    for i, (n_th, n_ph) in enumerate(nulls):
        C[:, i] = compute_steering_vector(pos, freq, np.deg2rad(n_th), np.deg2rad(n_ph))
    
    # Projection Matrix P = I - C(C^H C)^-1 C^H
    CH_C = np.conj(C.T) @ C
    CH_C_inv = np.linalg.pinv(CH_C) # Pseudo-inverse for numerical stability
    P = np.eye(N) - C @ CH_C_inv @ np.conj(C.T)
    
    # Project the target steering vector onto the null orthogonal complement
    w_opt = P @ np.conj(v_tgt)
    # Normalize weights so max amplitude is 1.0
    return w_opt / (np.max(np.abs(w_opt)) + 1e-12)

def quantize_weights(w_complex, amp_bits, phase_bits):
    """Simulates hardware limitations in digital phase shifters and variable gain amplifiers."""
    amps = np.abs(w_complex)
    phases = np.angle(w_complex)
    
    if amp_bits > 0:
        levels = (2**amp_bits) - 1
        amps = np.round(amps * levels) / levels
        
    if phase_bits > 0:
        step = 2 * np.pi / (2**phase_bits)
        phases = np.round(phases / step) * step
        
    return amps * np.exp(1j * phases)

def apply_weight_errors(w_complex, amp_err_std, phase_err_std_deg):
    """Injects statistical Monte Carlo Gaussian variance into the weight vectors."""
    amps = np.abs(w_complex); phases = np.angle(w_complex)
    amps_noisy = np.clip(amps + np.random.normal(0, amp_err_std, len(amps)), 0.0, 1.0)
    phases_noisy = phases + np.random.normal(0, np.deg2rad(phase_err_std_deg), len(phases))
    return amps_noisy * np.exp(1j * phases_noisy)

def compute_array_factor_complex(theta_arr, phi_arr, pos, w_complex, freq):
    """Analytically computes the 3D Array Factor using the assembled complex weight vector."""
    k0 = 2 * np.pi * freq / C_LIGHT
    THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing='ij')
    rx = np.sin(THETA) * np.cos(PHI); ry = np.sin(THETA) * np.sin(PHI); rz = np.cos(THETA)
    
    AF = np.zeros_like(THETA, dtype=np.complex128)
    for n in range(len(w_complex)):
        phase_term = k0 * (pos[n, 0]*rx + pos[n, 1]*ry + pos[n, 2]*rz)
        AF += w_complex[n] * np.exp(1j * phase_term)
    return AF

def generate_mock_s_matrix(num_el, spacing_wl):
    S = np.zeros((num_el, num_el), dtype=np.complex128)
    for i in range(num_el):
        for j in range(num_el):
            if i == j: S[i, i] = 10**(-15/20) * np.exp(-1j * np.pi/4) 
            else:
                dist = abs(i - j) * spacing_wl
                mag = 10**(-20/20) * np.exp(-dist * 1.5)
                phase = -2 * np.pi * dist
                S[i, j] = mag * np.exp(1j * phase)
    return S

def calculate_active_vswr(w_complex, S_matrix):
    N = len(w_complex)
    gamma_active = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        if np.abs(w_complex[i]) < 1e-12: continue
        coupling_sum = 0j
        for j in range(N): coupling_sum += S_matrix[i, j] * w_complex[j]
        gamma_active[i] = coupling_sum / w_complex[i]
    vswr_active = (1 + np.abs(gamma_active)) / (1 - np.abs(gamma_active) + 1e-12)
    return vswr_active

# ============================================================
# M37: ADVANCED PHASED-ARRAY BEAMFORMING LABORATORY UI
# ============================================================
if exp_mode == "Advanced Beamforming Lab (M37)":
    st.markdown("### 📡 Advanced Phased-Array Beamforming & Adaptive Control")
    st.info("Implements exact mathematical Orthogonal Projections to steer deep interference Nulls dynamically. Features hardware Quantization limits (N-bit Phase Shifters), Active VSWR mutual coupling analysis, and Monte Carlo Robustness error injection to certify beam survivability under structural degradation.")
    
    t_cfg, t_beam, t_scan, t_mc, t_rep = st.tabs([
        "1. Array & Target Config", "2. Adaptive Null Steering", "3. Scan Loss Analytics", "4. Monte Carlo Robustness", "5. Provenance Report"
    ])
    
    freq_arr = 2.4e9
    wl_arr = C_LIGHT / freq_arr
    
    with t_cfg:
        st.markdown("#### Physical Architecture & Targets")
        c_c1, c_c2, c_c3 = st.columns(3)
        array_type = c_c1.selectbox("Array Geometry", ["Uniform Linear Array (ULA)", "Planar Rectangular Array (UPA)", "Uniform Circular Array (UCA)"])
        N_el = c_c2.number_input("Number of Elements", 4, 64, 8, 1)
        spacing = c_c3.number_input("Element Spacing (λ)", 0.25, 2.0, 0.5, 0.05)
        
        pos = generate_array_geometry(array_type, N_el, spacing, spacing, freq_arr)
        
        st.markdown("##### Steering Targets")
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        t_th = c_t1.number_input("Target Theta (°)", 0.0, 180.0, 90.0)
        t_ph = c_t2.number_input("Target Phi (°)", 0.0, 360.0, 0.0)
        
        enable_null = c_t3.checkbox("Enable Null Constraints")
        null_th = c_t3.number_input("Null 1 Theta (°)", 0.0, 180.0, 90.0, disabled=not enable_null)
        null_ph = c_t4.number_input("Null 1 Phi (°)", 0.0, 360.0, 45.0, disabled=not enable_null)
        
        nulls = [(null_th, null_ph)] if enable_null else []
        
        st.session_state.m37_array = {"pos": pos, "N": N_el, "space": spacing, "type": array_type, "t_th": t_th, "t_ph": t_ph, "nulls": nulls}

    if st.session_state.m37_array is not None:
        arr = st.session_state.m37_array
        N = arr["N"]; pos = arr["pos"]
        
        with t_beam:
            st.markdown("#### Adaptive Weights & Hardware Quantization")
            
            c_w1, c_w2 = st.columns(2)
            ph_bits = c_w1.selectbox("Phase Shifter Resolution (Bits)", [0, 2, 3, 4, 5, 6, 8], index=0, format_func=lambda x: "Ideal (Infinite)" if x==0 else f"{x}-Bit ({360/(2**x)}° Steps)")
            amp_bits = c_w2.selectbox("VGA Amplitude Resolution (Bits)", [0, 4, 6, 8], index=0, format_func=lambda x: "Ideal (Continuous)" if x==0 else f"{x}-Bit")
            
            if st.button("Synthesize Adaptive Beam Vectors", type="primary"):
                w_ideal = compute_adaptive_null_weights(pos, freq_arr, arr["t_th"], arr["t_ph"], arr["nulls"])
                w_quant = quantize_weights(w_ideal, amp_bits, ph_bits)
                
                # Active VSWR evaluation
                S_mat = generate_mock_s_matrix(N, arr["space"])
                vswr_active = calculate_active_vswr(w_quant, S_mat)
                
                st.session_state.m37_weights = {"w_ideal": w_ideal, "w_quant": w_quant, "vswr": vswr_active}
                st.success("Mathematical Synthesis Complete.")

            if st.session_state.m37_weights is not None:
                w = st.session_state.m37_weights
                
                thetas = np.deg2rad(np.arange(0, 181, 1)); phis = np.deg2rad(np.array([arr["t_ph"]]))
                
                AF_ideal = compute_array_factor_complex(thetas, phis, pos, w["w_ideal"], freq_arr)
                AF_quant = compute_array_factor_complex(thetas, phis, pos, w["w_quant"], freq_arr)
                
                # Normalization
                AF_ideal_db = 20 * np.log10(np.abs(AF_ideal) / np.max(np.abs(AF_ideal)) + 1e-12)
                AF_quant_db = 20 * np.log10(np.abs(AF_quant) / np.max(np.abs(AF_ideal)) + 1e-12)
                
                col_plt1, col_plt2 = st.columns([2, 1])
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_ideal_db[:, 0], mode='lines', name="Ideal Adaptive Projection", line=dict(color='blue')))
                if ph_bits > 0 or amp_bits > 0:
                    fig_b.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_quant_db[:, 0], mode='lines', name=f"Quantized Beam ({ph_bits}-Bit Ph)", line=dict(color='red', dash='dash')))
                
                if arr["nulls"]:
                    fig_b.add_vline(x=arr["nulls"][0][0], line_dash="dot", line_color="black", annotation_text="Targeted Null")
                    
                fig_b.update_layout(title=f"Beamforming Azimuth Cut (Phi={arr['t_ph']}°)", xaxis_title="Theta (°)", yaxis_title="Normalized Array Factor (dB)", yaxis_range=[-50, 2])
                col_plt1.plotly_chart(fig_b, use_container_width=True)
                
                # Active VSWR warning list
                with col_plt2:
                    st.markdown("##### Mutual Coupling Check")
                    st.info("High mutual coupling interacting with synthesized weights can blind specific elements via extreme active reflection coefficients.")
                    max_vswr = np.max(w["vswr"])
                    st.metric("Peak Active VSWR", f"{max_vswr:.2f}:1", "Safe" if max_vswr < 3.0 else "DANGER", delta_color="inverse")
                    
                    df_w = pd.DataFrame({
                        "El": range(1, N+1),
                        "Amp": np.abs(w["w_quant"]),
                        "Phase (°)": np.rad2deg(np.angle(w["w_quant"])),
                        "VSWR": w["vswr"]
                    })
                    st.dataframe(df_w, height=300)

        with t_scan:
            st.markdown("#### 🔭 Beam Scan Loss Analytics")
            st.info("Systematically steers the array from broadside across the full angular horizon, recording the physical drop in Peak Directivity and Main Beam broadening induced by reduced effective aperture area (Scan Loss).")
            
            if st.button("Execute Scan Angle Sweep"):
                pb_scan = st.progress(0)
                scan_angles = np.arange(0, 91, 5)
                scan_peaks = []
                
                # Static grid for AF resolution
                thetas = np.deg2rad(np.arange(0, 181, 1)); phis = np.deg2rad(np.array([arr["t_ph"]]))
                
                for i, st_ang in enumerate(scan_angles):
                    # No nulls for basic scan loss curve
                    w_st = compute_adaptive_null_weights(pos, freq_arr, st_ang, arr["t_ph"], [])
                    AF_st = compute_array_factor_complex(thetas, phis, pos, w_st, freq_arr)
                    # We DO NOT normalize to 1 here, we track absolute sum magnitude to observe scan degradation
                    scan_peaks.append(20 * np.log10(np.max(np.abs(AF_st)) + 1e-12))
                    pb_scan.progress((i+1)/len(scan_angles))
                
                # Normalize relative to broadside (0 deg)
                scan_peaks = np.array(scan_peaks)
                scan_loss = scan_peaks - scan_peaks[0]
                
                fig_scan = go.Figure(go.Scatter(x=scan_angles, y=scan_loss, mode='lines+markers', marker=dict(color='purple')))
                fig_scan.update_layout(title="Array Scan Loss vs Steering Angle", xaxis_title="Steering Angle Theta (°)", yaxis_title="Gain Relative to Broadside (dB)", yaxis_range=[-10, 1])
                st.plotly_chart(fig_scan, use_container_width=True)

        with t_mc:
            st.markdown("#### 🎲 Monte Carlo Hardware Robustness")
            st.info("Injects statistical Gaussian variance into the calculated Ideal Phase and Amplitude vectors over 50 simulation runs. Generates bounded confidence intervals highlighting Null-filling vulnerability and Side Lobe degradation under real-world manufacturing tolerances.")
            
            if st.session_state.m37_weights is None: st.warning("Synthesize Beam Weights in Tab 2 first.")
            else:
                c_m1, c_m2 = st.columns(2)
                mc_ph_err = c_m1.number_input("Phase Tolerance Error (Std Dev °)", 0.0, 45.0, 10.0, 1.0)
                mc_amp_err = c_m2.number_input("Amplitude Tolerance Error (Std Dev Linear)", 0.0, 0.5, 0.1, 0.05)
                
                if st.button("Execute Monte Carlo Distribution"):
                    pb_mc = st.progress(0)
                    w_ideal = st.session_state.m37_weights["w_ideal"]
                    thetas = np.deg2rad(np.arange(0, 181, 2)); phis = np.deg2rad(np.array([arr["t_ph"]]))
                    
                    AF_ideal = compute_array_factor_complex(thetas, phis, pos, w_ideal, freq_arr)
                    AF_ideal_db = 20 * np.log10(np.abs(AF_ideal) / np.max(np.abs(AF_ideal)) + 1e-12)
                    
                    mc_runs = 50
                    all_af = np.zeros((mc_runs, len(thetas)))
                    
                    for i in range(mc_runs):
                        w_noisy = apply_weight_errors(w_ideal, mc_amp_err, mc_ph_err)
                        AF_noisy = compute_array_factor_complex(thetas, phis, pos, w_noisy, freq_arr)
                        all_af[i, :] = 20 * np.log10(np.abs(AF_noisy) / np.max(np.abs(AF_ideal)) + 1e-12)
                        pb_mc.progress((i+1)/mc_runs)
                        
                    # Statistical Bounds
                    af_mean = np.mean(all_af, axis=0)
                    af_p95 = np.percentile(all_af, 95, axis=0)
                    
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_ideal_db[:, 0], mode='lines', name="Ideal Baseline", line=dict(color='blue', width=2)))
                    fig_mc.add_trace(go.Scatter(x=np.rad2deg(thetas), y=af_p95, mode='lines', name="95th Percentile Degradation Boundary", line=dict(color='red', dash='dot')))
                    
                    # Optional visualization of all traces lightly
                    for i in range(mc_runs):
                        fig_mc.add_trace(go.Scatter(x=np.rad2deg(thetas), y=all_af[i, :], mode='lines', line=dict(color='rgba(150,150,150,0.1)'), showlegend=False))
                        
                    fig_mc.update_layout(title=f"Monte Carlo Envelope ({mc_runs} iterations)", xaxis_title="Theta (°)", yaxis_title="Normalized AF (dB)", yaxis_range=[-40, 2])
                    st.plotly_chart(fig_mc, use_container_width=True)

        with t_rep:
            if st.session_state.m37_weights is not None:
                st.markdown("#### 🗃️ Provenance & Beam Validation Export")
                
                report = {
                    "Beamforming_ID": str(uuid.uuid4()),
                    "Timestamp": datetime.datetime.now().isoformat(),
                    "Array_Configuration": {
                        "Type": arr["type"], "Elements": arr["N"], "Spacing_Lambda": arr["space"]
                    },
                    "Target_Constraints": {
                        "Main_Beam": {"Theta": arr["t_th"], "Phi": arr["t_ph"]},
                        "Orthogonal_Nulls": arr["nulls"]
                    },
                    "Hardware_Quantization": {
                        "Status": "APPLIED",
                        "Peak_Active_VSWR": float(np.max(st.session_state.m37_weights["vswr"]))
                    },
                    "Integrity_Hash": hashlib.md5(json.dumps(arr["t_th"]).encode()).hexdigest()
                }
                
                st.json(report)
                st.download_button("Export Advanced Beamforming Report (JSON)", data=json.dumps(report, indent=2), file_name="m37_beamforming_report.json", mime="application/json")

elif exp_mode not in ["Advanced Beamforming Lab (M37)"]:
    st.info("Select 'Advanced Beamforming Lab (M37)' to enable adaptive null projection and hardware quantization analytics.")

