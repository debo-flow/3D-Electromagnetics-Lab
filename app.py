"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 38 — Advanced MIMO & Multi-Antenna Electromagnetic Channel Modeling Laboratory
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
st.markdown("### Milestone 38 — Advanced MIMO & Electromagnetic Channel Laboratory")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Advanced MIMO Channel Lab (M38)",
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

if 'm38_channel' not in st.session_state: st.session_state.m38_channel = None

# ============================================================
# GRID & DOMAIN SETUP (DYNAMIC)
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
Nx = Ny = Nz = 40 if exp_mode not in ["Single Antenna (Dipole/Patch)"] else 80
if exp_mode in ["Advanced Antenna Array Lab (M36)", "Advanced Beamforming Lab (M37)", "Advanced MIMO Channel Lab (M38)"]: 
    Nx = Ny = 60; Nz = 40 
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

def compute_cpml(N, d_pml, delta, step_dt, m=3, R_err=1e-4, alpha_max=0.05):
    b_e = np.zeros(N, dtype=dtype_np); c_e = np.zeros(N, dtype=dtype_np); b_h = np.zeros(N, dtype=dtype_np); c_h = np.zeros(N, dtype=dtype_np)
    sigma_max = - (m + 1) * math.log(R_err) / (2.0 * Z_0 * (d_pml * delta)) if d_pml > 0 else 0
    for i in range(N):
        if d_pml == 0: continue
        d_e = (d_pml - i)*delta if i < d_pml else (i - (N - 1 - d_pml))*delta if i > N - 1 - d_pml else 0.0
        d_h = (d_pml - i - 0.5)*delta if i < d_pml else (i + 0.5 - (N - 1 - d_pml))*delta if i > N - 2 - d_pml else 0.0
        d_h = max(0.0, d_h)
        if d_e > 0:
            s_e = sigma_max * (d_e / (d_pml * delta))**m; a_e = alpha_max * (1.0 - d_e / (d_pml * delta))**m
            b_e[i] = math.exp(-(s_e + a_e * EPS_0 / step_dt) * (step_dt / EPS_0)); c_e[i] = s_e / (s_e + a_e * EPS_0 / step_dt) * (b_e[i] - 1.0) / delta
        if d_h > 0:
            s_h = sigma_max * (d_h / (d_pml * delta))**m; a_h = alpha_max * (1.0 - d_h / (d_pml * delta))**m
            b_h[i] = math.exp(-(s_h + a_h * EPS_0 / step_dt) * (step_dt / EPS_0)); c_h[i] = s_h / (s_h + a_h * EPS_0 / step_dt) * (b_h[i] - 1.0) / delta
    return b_e, c_e, b_h, c_h

b_e_x, c_e_x, b_h_x, c_h_x = compute_cpml(Nx, pml_thickness, dx, dt); b_e_y, c_e_y, b_h_y, c_h_y = compute_cpml(Ny, pml_thickness, dy, dt); b_e_z, c_e_z, b_h_z, c_h_z = compute_cpml(Nz, pml_thickness, dz, dt)

# ============================================================
# CORE FDTD SOLVER (PRESERVED)
# ============================================================
@nb.njit(cache=True)
def run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, obs_pts):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    omega = 2.0 * np.pi * freq_hz
    accum_steps = steps // 2 
    obs_phasors = np.zeros(obs_pts.shape[0], dtype=np.complex128)

    for n in range(steps):
        t_steps = float(n)
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                for k in range(Nz - 1):
                    dEz_dy = Ez[i, j+1, k] - Ez[i, j, k]; dEy_dz = Ey[i, j, k+1] - Ey[i, j, k]; dEx_dz = Ex[i, j, k+1] - Ex[i, j, k]
                    dEz_dx = Ez[i+1, j, k] - Ez[i, j, k]; dEy_dx = Ey[i+1, j, k] - Ey[i, j, k]; dEx_dy = Ex[i, j+1, k] - Ex[i, j, k]

                    psi_ey_hx[i,j,k] = b_h_y[j] * psi_ey_hx[i,j,k] + c_h_y[j] * dEz_dy * dy; psi_ez_hx[i,j,k] = b_h_z[k] * psi_ez_hx[i,j,k] + c_h_z[k] * dEy_dz * dz
                    psi_ez_hy[i,j,k] = b_h_x[i] * psi_ez_hy[i,j,k] + c_h_x[i] * dEx_dz * dz; psi_ex_hy[i,j,k] = b_h_z[k] * psi_ex_hy[i,j,k] + c_h_z[k] * dEz_dx * dx
                    psi_ex_hz[i,j,k] = b_h_x[i] * psi_ex_hz[i,j,k] + c_h_x[i] * dEy_dx * dx; psi_ey_hz[i,j,k] = b_h_y[j] * psi_ey_hz[i,j,k] + c_h_y[j] * dEx_dy * dy

                    Hx[i,j,k] -= ch2[i,j,k] * ( (dEz_dy/dy + psi_ey_hx[i,j,k]) - (dEy_dz/dz + psi_ez_hx[i,j,k]) )
                    Hy[i,j,k] -= ch2[i,j,k] * ( (dEx_dz/dz + psi_ex_hy[i,j,k]) - (dEz_dx/dx + psi_ez_hy[i,j,k]) )
                    Hz[i,j,k] -= ch2[i,j,k] * ( (dEy_dx/dx + psi_ex_hz[i,j,k]) - (dEx_dy/dy + psi_ey_hz[i,j,k]) )

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    dHz_dy = Hz[i, j, k] - Hz[i, j-1, k]; dHy_dz = Hy[i, j, k] - Hy[i, j, k-1]; dHx_dz = Hx[i, j, k] - Hx[i, j, k-1]
                    dHz_dx = Hz[i, j, k] - Hz[i-1, j, k]; dHy_dx = Hy[i, j, k] - Hy[i-1, j, k]; dHx_dy = Hx[i, j, k] - Hx[i, j-1, k]

                    psi_hy_ex[i,j,k] = b_e_y[j] * psi_hy_ex[i,j,k] + c_e_y[j] * dHz_dy * dy; psi_hz_ex[i,j,k] = b_e_z[k] * psi_hz_ex[i,j,k] + c_e_z[k] * dHy_dz * dz
                    psi_hx_ey[i,j,k] = b_e_z[k] * psi_hx_ey[i,j,k] + c_e_z[k] * dHx_dz * dz; psi_hz_ey[i,j,k] = b_e_x[i] * psi_hz_ey[i,j,k] + c_e_x[i] * dHz_dx * dx
                    psi_hy_ez[i,j,k] = b_e_x[i] * psi_hy_ez[i,j,k] + c_e_x[i] * dHy_dx * dx; psi_hx_ez[i,j,k] = b_e_y[j] * psi_hx_ez[i,j,k] + c_e_y[j] * dHx_dy * dy

                    ex_old = Ex[i,j,k]; ey_old = Ey[i,j,k]; ez_old = Ez[i,j,k]
                    Ex[i,j,k] = ce1_x[i,j,k]*ex_old + ce2_x[i,j,k]*((dHz_dy/dy+psi_hy_ex[i,j,k]) - (dHy_dz/dz+psi_hz_ex[i,j,k])) + ce3_x[i,j,k]*Px[i,j,k]
                    Ey[i,j,k] = ce1_y[i,j,k]*ey_old + ce2_y[i,j,k]*((dHx_dz/dz+psi_hx_ey[i,j,k]) - (dHz_dx/dx+psi_hz_ey[i,j,k])) + ce3_y[i,j,k]*Py[i,j,k]
                    Ez[i,j,k] = ce1_z[i,j,k]*ez_old + ce2_z[i,j,k]*((dHy_dx/dx+psi_hy_ez[i,j,k]) - (dHx_dy/dy+psi_hx_ez[i,j,k])) + ce3_z[i,j,k]*Pz[i,j,k]
                    
                    Px[i,j,k] = cp1_x[i,j,k]*Px[i,j,k] + cp2_x[i,j,k]*(Ex[i,j,k] + ex_old)
                    Py[i,j,k] = cp1_y[i,j,k]*Py[i,j,k] + cp2_y[i,j,k]*(Ey[i,j,k] + ey_old)
                    Pz[i,j,k] = cp1_z[i,j,k]*Pz[i,j,k] + cp2_z[i,j,k]*(Ez[i,j,k] + ez_old)

        for e in range(num_el):
            pulse = amp_arr[e] * math.sin(omega * n * dt + phase_arr[e])
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ez[fx_arr[e], fy_arr[e], k] += pulse
            
        if n >= steps - accum_steps:
            e_j_wt = np.exp(-1j * omega * n * dt)
            for r_idx in range(obs_pts.shape[0]):
                obs_phasors[r_idx] += Ez[obs_pts[r_idx, 0], obs_pts[r_idx, 1], obs_pts[r_idx, 2]] * e_j_wt

    obs_phasors /= accum_steps
    return obs_phasors

# ============================================================
# M38: ADVANCED MIMO CHANNEL MATHEMATICS
# ============================================================
def generate_mimo_channel_mock(Nt, Nr, pos_tx, pos_rx, freqs, multipath=True, env_factor=1.0):
    """Generates a geometrically consistent Broadband MIMO Channel Matrix H(f)."""
    num_f = len(freqs)
    H = np.zeros((num_f, Nr, Nt), dtype=np.complex128)
    
    np.random.seed(42) # Deterministic Mock Environment
    num_clusters = 6 if multipath else 1
    
    # Generate Environment Clusters (Scatterers)
    clusters = []
    for c in range(num_clusters):
        if c == 0: # Line of Sight (LoS)
            clusters.append({"gain": 1.0, "delay": 5e-9, "az_tx": 0, "el_tx": 90, "az_rx": 180, "el_rx": 90})
        else: # NLoS Multipath reflections
            gain = np.random.rayleigh(0.4 * env_factor)
            delay = np.random.uniform(10e-9, 100e-9 * env_factor)
            az_tx = np.random.uniform(0, 360); el_tx = np.random.uniform(60, 120)
            az_rx = np.random.uniform(0, 360); el_rx = np.random.uniform(60, 120)
            clusters.append({"gain": gain, "delay": delay, "az_tx": az_tx, "el_tx": el_tx, "az_rx": az_rx, "el_rx": el_rx})
            
    for f_idx, f in enumerate(freqs):
        k0 = 2 * np.pi * f / C_LIGHT
        H_f = np.zeros((Nr, Nt), dtype=np.complex128)
        
        for cl in clusters:
            # TX Steering Vector
            tx_rx = np.sin(np.deg2rad(cl["el_tx"])) * np.cos(np.deg2rad(cl["az_tx"]))
            tx_ry = np.sin(np.deg2rad(cl["el_tx"])) * np.sin(np.deg2rad(cl["az_tx"]))
            tx_rz = np.cos(np.deg2rad(cl["el_tx"]))
            a_tx = np.exp(1j * k0 * (pos_tx[:,0]*tx_rx + pos_tx[:,1]*tx_ry + pos_tx[:,2]*tx_rz))
            
            # RX Steering Vector
            rx_rx = np.sin(np.deg2rad(cl["el_rx"])) * np.cos(np.deg2rad(cl["az_rx"]))
            rx_ry = np.sin(np.deg2rad(cl["el_rx"])) * np.sin(np.deg2rad(cl["az_rx"]))
            rx_rz = np.cos(np.deg2rad(cl["el_rx"]))
            a_rx = np.exp(1j * k0 * (pos_rx[:,0]*rx_rx + pos_rx[:,1]*rx_ry + pos_rx[:,2]*rx_rz))
            
            # Add cluster contribution to H matrix
            phase_delay = np.exp(-1j * 2 * np.pi * f * cl["delay"])
            H_f += cl["gain"] * phase_delay * np.outer(a_rx, np.conj(a_tx))
            
        H[f_idx] = H_f
    return H, clusters

def calculate_mimo_capacity(H, snr_db):
    """Calculates Spatial Shannon Capacity: C = log2(det(I + (SNR/Nt) * H * H^H))"""
    Nr, Nt = H.shape
    snr_lin = 10**(snr_db / 10.0)
    I = np.eye(Nr)
    # H @ H.conj().T gives the spatial correlation matrix at RX
    R = H @ np.conj(H.T)
    val = I + (snr_lin / Nt) * R
    # Prevent numerical instability in det for poorly conditioned matrices
    sign, logdet = np.linalg.slogdet(val)
    return max(0.0, logdet / np.log(2))

def calculate_pdp_and_delay_spread(H_freq, freqs):
    """Converts H(f) to Channel Impulse Response (CIR) and extracts RMS Delay Spread."""
    # Mean across spatial dimensions
    H_mean = np.mean(H_freq, axis=(1, 2))
    # IFFT to Time Domain
    cir = np.fft.ifft(H_mean)
    pdp = np.abs(cir)**2
    
    df = freqs[1] - freqs[0]
    time_taps = np.fft.fftfreq(len(freqs), d=df)
    
    # Shift to positive delays
    time_taps = np.fft.fftshift(time_taps)
    pdp = np.fft.fftshift(pdp)
    
    # RMS Delay Spread Math
    valid_idx = time_taps >= 0
    t_valid = time_taps[valid_idx]
    p_valid = pdp[valid_idx]
    
    total_power = np.sum(p_valid)
    if total_power < 1e-12: return pdp, time_taps, 0.0
    
    mean_delay = np.sum(t_valid * p_valid) / total_power
    rms_delay = np.sqrt(np.sum((t_valid - mean_delay)**2 * p_valid) / total_power)
    
    return pdp, time_taps, rms_delay

def apply_precoding(H, method="SVD"):
    Nr, Nt = H.shape
    if method == "SVD":
        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        # Optimal precoder for dominant mode is the conjugate of the dominant right singular vector
        W_tx = np.conj(Vh[0, :]).reshape(-1, 1)
        W_rx = U[:, 0].reshape(-1, 1)
        gain = np.abs(np.conj(W_rx.T) @ H @ W_tx)[0,0]
        return W_tx, W_rx, gain
    elif method == "MRT":
        # Maximum Ratio Transmission (No RX combining coordination)
        W_tx = np.conj(H.T) @ np.ones((Nr, 1))
        W_tx = W_tx / np.linalg.norm(W_tx)
        gain = np.abs(np.sum(H @ W_tx))
        return W_tx, None, gain
    elif method == "Zero-Forcing":
        try:
            # Pseudo-inverse of H
            H_inv = np.linalg.pinv(H)
            # TX weights to force diagonal identity at RX (simplified for broadcast)
            W_tx = H_inv[:, 0].reshape(-1, 1) 
            W_tx = W_tx / np.linalg.norm(W_tx)
            gain = np.abs((H @ W_tx)[0, 0])
            return W_tx, None, gain
        except np.linalg.LinAlgError:
            return None, None, -1.0

def apply_imperfect_csi(H, error_variance=0.1):
    """Simulates Channel Estimation Error: H_est = sqrt(1-e^2)*H + e*Noise"""
    noise = (np.random.normal(0, 1, H.shape) + 1j * np.random.normal(0, 1, H.shape)) / np.sqrt(2)
    H_est = np.sqrt(1 - error_variance**2) * H + error_variance * noise * np.linalg.norm(H) / np.sqrt(np.prod(H.shape))
    return H_est

# ============================================================
# M38: ADVANCED MIMO CHANNEL LABORATORY UI
# ============================================================
if exp_mode == "Advanced MIMO Channel Lab (M38)":
    st.markdown("### 📡 Advanced MIMO & Electromagnetic Channel Laboratory")
    st.info("Transforms standalone Phased Arrays into complete $N_T \\times N_R$ spatial communication links. Generates geometric Multipath matrices $H(f)$, exact Shannon Spatial Capacity bounds, Power Delay Profiles (PDP) via IFFT, and explicit Zero-Forcing (ZF) precoding resilience analytics.")
    
    t_cfg, t_svd, t_cap, t_pdp, t_pre = st.tabs([
        "1. MIMO Configuration", "2. Channel Matrix & SVD", "3. Spatial Capacity", "4. Multipath Delay Profile", "5. Precoding & Imperfect CSI"
    ])
    
    with t_cfg:
        st.markdown("#### TX/RX Array Geometries & Environment")
        c_m1, c_m2, c_m3 = st.columns(3)
        Nt = c_m1.number_input("Transmit Antennas ($N_T$)", 2, 64, 4, 2)
        Nr = c_m2.number_input("Receive Antennas ($N_R$)", 2, 64, 4, 2)
        env_str = c_m3.selectbox("Environment Scattering", ["Line-of-Sight (LoS) Only", "Urban Multipath (High Delay)", "Indoor (Low Delay)"])
        
        spacing = st.slider("Array Element Spacing (λ)", 0.1, 2.0, 0.5, 0.1)
        freq_c = 2.4e9
        
        if st.button("Generate MIMO Channel Matrix", type="primary"):
            pb = st.progress(0)
            
            # Position Setup (ULA for simplicity in MOCK generation)
            wl = C_LIGHT / freq_c
            pos_tx = np.zeros((Nt, 3)); pos_rx = np.zeros((Nr, 3))
            for i in range(Nt): pos_tx[i, 1] = (i - (Nt-1)/2.0) * spacing * wl
            for i in range(Nr): pos_rx[i, 1] = (i - (Nr-1)/2.0) * spacing * wl
            # Offset RX position by 100 meters
            pos_rx[:, 0] = 100.0
            
            # Bandwidth setup for PDP evaluation
            freqs = np.linspace(2.3e9, 2.5e9, 101)
            
            multipath = False; env_factor = 1.0
            if env_str == "Urban Multipath (High Delay)": multipath = True; env_factor = 5.0
            elif env_str == "Indoor (Low Delay)": multipath = True; env_factor = 0.5
            
            H_f, clusters = generate_mimo_channel_mock(Nt, Nr, pos_tx, pos_rx, freqs, multipath, env_factor)
            
            st.session_state.m38_channel = {
                "H": H_f, "freqs": freqs, "Nt": Nt, "Nr": Nr, "clusters": clusters, "env": env_str
            }
            pb.progress(1.0)
            st.success(f"MOCK Broadband Channel Matrix $H(f) \\in \\mathbb{{C}}^{{{Nr} \\times {Nt} \\times 101}}$ generated successfully.")

    if st.session_state.m38_channel is not None:
        ch = st.session_state.m38_channel
        H_center = ch["H"][50] # Center frequency 2.4 GHz slice
        
        with t_svd:
            st.markdown("#### Narrowband Spatial Channel Properties ($f_c = 2.4$ GHz)")
            
            # Eigenanalysis
            U, S, Vh = np.linalg.svd(H_center, full_matrices=False)
            cond_num = np.max(S) / (np.min(S) + 1e-12)
            numerical_rank = np.sum(S > 1e-3 * np.max(S))
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Numerical Channel Rank", f"{numerical_rank}", f"Max possible: {min(ch['Nt'], ch['Nr'])}")
            c_s2.metric("Matrix Condition Number", f"{cond_num:.1f}", "Well-Conditioned" if cond_num < 10 else "Ill-Conditioned", delta_color="inverse")
            c_s3.metric("Dominant Singular Value (Σ₁)", f"{S[0]:.2f}")
            
            col_hm1, col_hm2 = st.columns(2)
            # Heatmaps
            fig_mag = go.Figure(data=go.Heatmap(z=20*np.log10(np.abs(H_center) + 1e-12), colorscale='Viridis', colorbar=dict(title="dB")))
            fig_mag.update_layout(title="Channel Magnitude $|H_{ij}|$ (dB)", xaxis_title="TX Antenna (j)", yaxis_title="RX Antenna (i)", width=350, height=350)
            col_hm1.plotly_chart(fig_mag)
            
            fig_ph = go.Figure(data=go.Heatmap(z=np.angle(H_center), colorscale='Phase', zmin=-np.pi, zmax=np.pi, colorbar=dict(title="Rads")))
            fig_ph.update_layout(title="Channel Phase $\\angle H_{ij}$ (Rads)", xaxis_title="TX Antenna (j)", yaxis_title="RX Antenna (i)", width=350, height=350)
            col_hm2.plotly_chart(fig_ph)

        with t_cap:
            st.markdown("#### Shannon Spatial Capacity vs SNR")
            st.info("Computes theoretical data throughput scaling bounds utilizing full spatial correlation matrices. High rank and multipath richness explicitly stack multiplexing layers, raising bounds linearly rather than logarithmically.")
            
            snr_range = np.linspace(-10, 30, 21)
            cap_mimo = [calculate_mimo_capacity(H_center, snr) for snr in snr_range]
            
            # Baseline SISO (H[0,0])
            cap_siso = [math.log2(1 + 10**(snr/10) * np.abs(H_center[0,0])**2) for snr in snr_range]
            
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(x=snr_range, y=cap_mimo, mode='lines+markers', name=f"MIMO ({ch['Nt']}x{ch['Nr']})", line=dict(color='blue', width=3)))
            fig_cap.add_trace(go.Scatter(x=snr_range, y=cap_siso, mode='lines', name="SISO (1x1 Benchmark)", line=dict(color='gray', dash='dash')))
            
            fig_cap.update_layout(title="Ergodic Channel Capacity", xaxis_title="Signal-to-Noise Ratio (dB)", yaxis_title="Spectral Efficiency (bits/s/Hz)")
            st.plotly_chart(fig_cap, use_container_width=True)

        with t_pdp:
            st.markdown("#### Time-Domain Multipath Spread (Power Delay Profile)")
            st.info("Transforms the $200$ MHz broadband $H(f)$ sweep into the time domain via Inverse Fast Fourier Transform (IFFT) to physically isolate spatial reflection clusters and compute inter-symbol interference (RMS Delay Spread) limits.")
            
            pdp, t_taps, rms_delay = calculate_pdp_and_delay_spread(ch["H"], ch["freqs"])
            
            st.metric("Calculated RMS Delay Spread ($t_{rms}$)", f"{rms_delay * 1e9:.2f} ns", "Multipath Active" if rms_delay > 1e-9 else "Clean LoS", delta_color="inverse")
            
            fig_pdp = go.Figure(go.Scatter(x=t_taps*1e9, y=10*np.log10(pdp + 1e-12), mode='lines', fill='tozeroy', line=dict(color='purple')))
            fig_pdp.update_layout(title="Power Delay Profile (PDP)", xaxis_title="Delay (ns)", yaxis_title="Relative Power (dB)", xaxis_range=[0, 100])
            st.plotly_chart(fig_pdp, use_container_width=True)

        with t_pre:
            st.markdown("#### Precoding & Channel Estimation Robustness")
            st.info("Evaluates real-time Spatial Beamforming solutions across the Matrix. Introduces `Imperfect CSI` (Channel State Information) Error variances to measure strict Zero-Forcing (ZF) instability when operating with degraded measurements.")
            
            csi_err = st.slider("CSI Estimation Error Variance ($\epsilon$)", 0.0, 0.5, 0.1, 0.05)
            
            # Perfect CSI
            _, _, gain_svd = apply_precoding(H_center, "SVD")
            _, _, gain_mrt = apply_precoding(H_center, "MRT")
            _, _, gain_zf = apply_precoding(H_center, "Zero-Forcing")
            
            # Imperfect CSI
            np.random.seed(42)
            H_est = apply_imperfect_csi(H_center, csi_err)
            
            # Precode based on H_est, but evaluate received signal through True H_center
            W_zf, _, _ = apply_precoding(H_est, "Zero-Forcing")
            gain_zf_err = np.abs((H_center @ W_zf)[0,0]) if W_zf is not None else -1.0
            
            W_mrt, _, _ = apply_precoding(H_est, "MRT")
            gain_mrt_err = np.abs(np.sum(H_center @ W_mrt))
            
            st.markdown("##### Effective Beamformed Gain (Normalized Linear)")
            db_res = pd.DataFrame({
                "Precoding Algorithm": ["Max Ratio Transmission (MRT)", "Zero-Forcing (ZF)", "Optimal SVD (Perfect Only)"],
                "Perfect CSI Gain": [gain_mrt, gain_zf if gain_zf > 0 else "UNSTABLE", gain_svd],
                f"Imperfect CSI Gain (e={csi_err})": [gain_mrt_err, gain_zf_err if gain_zf_err > 0 else "UNSTABLE", "N/A"]
            })
            st.table(db_res)
            
            if gain_zf < 0 or gain_zf_err < 0:
                st.error("ZERO-FORCING SOLUTION UNRELIABLE: Matrix inversion failed due to severe Ill-Conditioning limits.")
                
            st.markdown("---")
            if st.button("Generate MIMO Provenance Report"):
                report = {
                    "MIMO_ID": str(uuid.uuid4()), "Timestamp": datetime.datetime.now().isoformat(),
                    "System": {"Nt": ch["Nt"], "Nr": ch["Nr"], "Environment": ch["env"]},
                    "Channel_Metrics": {"Rank": int(numerical_rank), "Condition_Number": float(cond_num), "RMS_Delay_ns": float(rms_delay * 1e9)},
                    "Robustness": {"CSI_Error": csi_err, "ZF_Failure": gain_zf < 0},
                    "Data_Source": "MOCK GEOMETRIC SCATTERING MODEL"
                }
                st.download_button("Export MIMO Report (JSON)", data=json.dumps(report, indent=2), file_name="m38_mimo_report.json", mime="application/json")

elif exp_mode not in ["Advanced MIMO Channel Lab (M38)"]:
    st.info("Select 'Advanced MIMO Channel Lab (M38)' to analyze multi-antenna capacity matrices, SVD, and Power Delay Profiles.")

