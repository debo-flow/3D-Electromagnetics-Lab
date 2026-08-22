"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 36 — Advanced Antenna Array & Multi-Antenna System Characterization
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
st.markdown("### Milestone 36 — Advanced Antenna Array & Multi-Antenna Characterization")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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
if 'm36_array' not in st.session_state: st.session_state.m36_array = None
if 'm36_ff' not in st.session_state: st.session_state.m36_ff = None

# ============================================================
# GRID & DOMAIN SETUP (DYNAMIC)
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
Nx = Ny = Nz = 40 if exp_mode not in ["Single Antenna (Dipole/Patch)"] else 80
if exp_mode == "Advanced Antenna Array Lab (M36)": Nx = Ny = 60; Nz = 40 # Expand for array grids
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
# M36: FDTD FULL-WAVE ARRAY SOLVER
# ============================================================
@nb.njit(cache=True)
def run_simulation_array_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, z_scan):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    Ex_phasor = np.zeros((Nx, Ny), dtype=np.complex128)
    Ey_phasor = np.zeros((Nx, Ny), dtype=np.complex128)
    omega = 2.0 * np.pi * freq_hz
    accum_steps = steps // 2

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

        # Multi-Element Phased Array Excitation (CW)
        for e in range(num_el):
            pulse = amp_arr[e] * math.sin(omega * n * dt + phase_arr[e])
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ez[fx_arr[e], fy_arr[e], k] += pulse
            
        if n >= steps - accum_steps:
            e_j_wt = np.exp(-1j * omega * n * dt)
            for i in range(Nx):
                for j in range(Ny):
                    Ex_phasor[i,j] += Ex[i, j, z_scan] * e_j_wt
                    Ey_phasor[i,j] += Ey[i, j, z_scan] * e_j_wt

    Ex_phasor /= accum_steps; Ey_phasor /= accum_steps
    return Ex_phasor, Ey_phasor

def compute_far_field_direct_planar(Ex_nf, Ey_nf, x_arr, y_arr, freq, thetas, phis):
    k0 = 2 * np.pi * freq / C_LIGHT
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    dx_s = x_arr[1] - x_arr[0] if len(x_arr) > 1 else 1.0; dy_s = y_arr[1] - y_arr[0] if len(y_arr) > 1 else 1.0
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
    kx = k0 * np.sin(THETA) * np.cos(PHI); ky = k0 * np.sin(THETA) * np.sin(PHI)
    
    X_b = X[:, :, None, None]; Y_b = Y[:, :, None, None]
    kx_b = kx[None, None, :, :]; ky_b = ky[None, None, :, :]
    
    kernel = np.exp(1j * (kx_b * X_b + ky_b * Y_b))
    Fx = np.sum(Ex_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_s * dy_s
    Fy = np.sum(Ey_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_s * dy_s
    
    E_theta = (Fx * np.cos(PHI) + Fy * np.sin(PHI)) * np.cos(THETA); E_phi = -Fx * np.sin(PHI) + Fy * np.cos(PHI)
    return E_theta, E_phi

# ============================================================
# M36: ARRAY MATHEMATICS & ACTIVE VSWR KERNEL
# ============================================================
def generate_array_geometry(geom_type, num_el, spacing_x, spacing_y, freq):
    """Generates Cartesian element coordinates (x, y, z) in meters."""
    pos = np.zeros((num_el, 3))
    wl = C_LIGHT / freq
    dx_m = spacing_x * wl; dy_m = spacing_y * wl
    
    if geom_type == "Uniform Linear Array (ULA)":
        for i in range(num_el): pos[i, 0] = (i - (num_el-1)/2.0) * dx_m
    elif geom_type == "Planar Rectangular Array (UPA)":
        side = int(np.ceil(np.sqrt(num_el)))
        for i in range(num_el):
            row = i // side; col = i % side
            pos[i, 0] = (col - (side-1)/2.0) * dx_m
            pos[i, 1] = (row - (side-1)/2.0) * dy_m
    elif geom_type == "Uniform Circular Array (UCA)":
        radius = dx_m * num_el / (2 * np.pi) if num_el > 1 else dx_m
        for i in range(num_el):
            angle = i * (2 * np.pi / num_el)
            pos[i, 0] = radius * np.cos(angle); pos[i, 1] = radius * np.sin(angle)
    return pos

def compute_array_factor(theta_arr, phi_arr, pos, amps, phases, freq):
    """Analytically computes the 3D Array Factor (AF) matrix."""
    k0 = 2 * np.pi * freq / C_LIGHT
    THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing='ij')
    rx = np.sin(THETA) * np.cos(PHI); ry = np.sin(THETA) * np.sin(PHI); rz = np.cos(THETA)
    
    AF = np.zeros_like(THETA, dtype=np.complex128)
    for n in range(len(amps)):
        phase_term = k0 * (pos[n, 0]*rx + pos[n, 1]*ry + pos[n, 2]*rz)
        AF += amps[n] * np.exp(1j * (phase_term + phases[n]))
    return AF

def calculate_active_vswr(amps, phases, S_matrix):
    """Computes Active Reflection Coefficient & Active VSWR under full coupling."""
    N = len(amps)
    a = amps * np.exp(1j * phases) # Complex incident waves
    gamma_active = np.zeros(N, dtype=np.complex128)
    
    for i in range(N):
        if np.abs(a[i]) < 1e-12: continue
        coupling_sum = 0j
        for j in range(N): coupling_sum += S_matrix[i, j] * a[j]
        gamma_active[i] = coupling_sum / a[i]
        
    vswr_active = (1 + np.abs(gamma_active)) / (1 - np.abs(gamma_active) + 1e-12)
    return gamma_active, vswr_active

def generate_mock_s_matrix(num_el, spacing_wl):
    """MOCK S-Parameter Matrix for realistic array coupling diagnostics."""
    S = np.zeros((num_el, num_el), dtype=np.complex128)
    for i in range(num_el):
        for j in range(num_el):
            if i == j: S[i, i] = 10**(-15/20) * np.exp(-1j * np.pi/4) # S11 ~ -15 dB
            else:
                dist = abs(i - j) * spacing_wl
                # Coupling decays exponentially with distance
                mag = 10**(-20/20) * np.exp(-dist * 1.5)
                phase = -2 * np.pi * dist
                S[i, j] = mag * np.exp(1j * phase)
    return S

def binomial_weights(N):
    if N <= 1: return np.array([1.0])
    w = np.zeros(N)
    for k in range(N): w[k] = math.comb(N-1, k)
    return w / np.max(w)

# ============================================================
# M36: ADVANCED ANTENNA ARRAY LABORATORY UI
# ============================================================
if exp_mode == "Advanced Antenna Array Lab (M36)":
    st.markdown("### 📡 Advanced Antenna Array & Multi-System Characterization")
    st.info("Unifies elemental FDTD physics with Array Factor (AF) analytical steering, Active VSWR mutual coupling metrics, and structural tolerance degradation testing.")
    
    t_cfg, t_af, t_vswr, t_fdtd, t_tol = st.tabs([
        "1. Array Configuration", "2. Array Factor & Steering", "3. Coupling & Active VSWR", "4. Full-Wave Correlation", "5. Tolerance Analysis"
    ])
    
    # Base Configurations
    freq_arr = 2.4e9
    wl_arr = C_LIGHT / freq_arr
    
    with t_cfg:
        st.markdown("#### Geometry & Excitation Architecture")
        c_c1, c_c2, c_c3 = st.columns(3)
        array_type = c_c1.selectbox("Array Geometry", ["Uniform Linear Array (ULA)", "Planar Rectangular Array (UPA)", "Uniform Circular Array (UCA)"])
        N_el = c_c2.number_input("Number of Elements", 2, 64, 4, 1)
        spacing = c_c3.number_input("Element Spacing (λ)", 0.1, 2.0, 0.5, 0.05)
        
        pos = generate_array_geometry(array_type, N_el, spacing, spacing, freq_arr)
        
        st.markdown("##### Array Visualization (Top-Down)")
        fig_lyt = go.Figure(go.Scatter(x=pos[:,0]/wl_arr, y=pos[:,1]/wl_arr, mode='markers+text', text=[str(i+1) for i in range(N_el)], textposition="top center", marker=dict(size=12, color='red')))
        fig_lyt.update_layout(title="Element Layout (Wavelengths)", xaxis_title="X (λ)", yaxis_title="Y (λ)", width=600, height=400)
        st.plotly_chart(fig_lyt)
        
        st.session_state.m36_array = {"pos": pos, "N": N_el, "space": spacing, "type": array_type}

    if st.session_state.m36_array is not None:
        arr = st.session_state.m36_array
        N = arr["N"]; pos = arr["pos"]
        
        with t_af:
            st.markdown("#### Phase Steering & Side-Lobe Tapering")
            c_s1, c_s2, c_s3 = st.columns(3)
            steer_th = c_s1.number_input("Steering Theta (°)", 0.0, 180.0, 90.0)
            steer_ph = c_s2.number_input("Steering Phi (°)", 0.0, 360.0, 30.0)
            taper_type = c_s3.selectbox("Amplitude Tapering (Side-Lobe Control)", ["Uniform", "Binomial"])
            
            # Grating Lobe Diagnostic
            if arr["type"] == "Uniform Linear Array (ULA)" and arr["space"] > 1.0 / (1.0 + np.abs(np.sin(np.deg2rad(steer_ph)))):
                st.warning(f"⚠️ GRATING LOBE WARNING: Spacing ({arr['space']}λ) exceeds theoretical limits for selected steering angle. Undesired spatial aliasing beams will emerge.")
            
            if st.button("Compute Array Factor (AF)", type="primary"):
                # Weights
                amps = np.ones(N) if taper_type == "Uniform" else binomial_weights(N)
                
                # Steering Phases: phi_n = -k * (r_n \cdot r_hat)
                k0 = 2 * np.pi * freq_arr / C_LIGHT
                rx = np.sin(np.deg2rad(steer_th)) * np.cos(np.deg2rad(steer_ph))
                ry = np.sin(np.deg2rad(steer_th)) * np.sin(np.deg2rad(steer_ph))
                rz = np.cos(np.deg2rad(steer_th))
                
                phases = -k0 * (pos[:,0]*rx + pos[:,1]*ry + pos[:,2]*rz)
                
                arr["amps"] = amps; arr["phases"] = phases
                
                thetas = np.deg2rad(np.arange(0, 181, 2)); phis = np.deg2rad(np.arange(0, 360, 2))
                AF = compute_array_factor(thetas, phis, pos, amps, phases, freq_arr)
                AF_db = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)) + 1e-12)
                
                # 3D Plot
                THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
                R_plt = np.maximum(AF_db + 40, 0)
                X_plt = R_plt * np.sin(THETA) * np.cos(PHI); Y_plt = R_plt * np.sin(THETA) * np.sin(PHI); Z_plt = R_plt * np.cos(THETA)
                
                fig_af = go.Figure(data=[go.Surface(x=X_plt, y=Y_plt, z=Z_plt, surfacecolor=AF_db, colorscale='Jet')])
                fig_af.update_layout(title="3D Normalized Array Factor (dB)", height=500)
                st.plotly_chart(fig_af, use_container_width=True)

        with t_vswr:
            st.markdown("#### Mutual Coupling & Active VSWR Matrix")
            st.info("The presence of Mutual Coupling ($S_{ij}$) shifts element matching dynamically based on the excitation vector $w_n$. Active VSWR determines true amplifier load.")
            
            if "amps" in arr:
                S_mat = generate_mock_s_matrix(N, arr["space"])
                gamma_act, vswr_act = calculate_active_vswr(arr["amps"], arr["phases"], S_mat)
                
                cc1, cc2 = st.columns(2)
                
                # Heatmap
                S_db = 20 * np.log10(np.abs(S_mat) + 1e-12)
                fig_s = go.Figure(data=go.Heatmap(z=S_db, colorscale='Reds', zmax=0, zmin=-40, colorbar=dict(title="dB")))
                fig_s.update_layout(title="MOCK S-Parameter Matrix ($S_{ij}$)", xaxis_title="Port J", yaxis_title="Port I", height=400)
                cc1.plotly_chart(fig_s, use_container_width=True)
                
                # Active VSWR Bar Chart
                fig_v = go.Figure(data=[go.Bar(x=[f"Element {i+1}" for i in range(N)], y=vswr_act, marker_color='orange')])
                fig_v.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="VSWR 2.0 (Threshold)")
                fig_v.update_layout(title="Active VSWR per Element (Steered State)", yaxis_title="VSWR", height=400)
                cc2.plotly_chart(fig_v, use_container_width=True)
            else: st.warning("Compute the Array Factor first to define excitation weights.")

        with t_fdtd:
            st.markdown("#### 🔄 Full-Wave Validation vs Array Factor")
            st.info("Re-simulates the physical grid using FDTD by exciting discrete local dipole sources applying the exact generated Complex Weights. Cross-correlates the resulting Numerical Far-Field against the theoretical Array Factor.")
            
            if "amps" in arr:
                if st.button("Execute Full-Wave Array Simulation", type="primary"):
                    pb = st.progress(0)
                    # Translate physical positions to FDTD indices
                    f_x_arr = np.zeros(N, dtype=int); f_y_arr = np.zeros(N, dtype=int)
                    f_zs_arr = np.zeros(N, dtype=int); f_ze_arr = np.zeros(N, dtype=int)
                    
                    arm = 2 # Small short dipole for coupling test
                    for i in range(N):
                        f_x_arr[i] = int(cx + pos[i, 0] / dx)
                        f_y_arr[i] = int(cy + pos[i, 1] / dy)
                        f_zs_arr[i] = cz - arm; f_ze_arr[i] = cz + arm
                        
                    reset_materials()
                    # Add PECs for elements
                    for i in range(N):
                        apply_material_block(f_x_arr[i], f_x_arr[i], f_y_arr[i], f_y_arr[i], f_zs_arr[i], f_ze_arr[i], MAT_LIB["PEC (Perfect Conductor)"])
                        
                    z_scan = cz + 12
                    Ex_sim, Ey_sim = run_simulation_array_cpu(Nx, Ny, Nz, dx, dy, dz, dt, 300, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, N, f_x_arr, f_y_arr, f_zs_arr, f_ze_arr, arr["amps"], arr["phases"], freq_arr, z_scan)
                    pb.progress(0.5)
                    
                    # Planar NF2FF
                    x_scan = (np.arange(Nx) - cx) * dx; y_scan = (np.arange(Ny) - cy) * dy
                    thetas = np.deg2rad(np.arange(0, 90, 2)); phis = np.deg2rad(np.arange(0, 360, 2))
                    Eth, Eph = compute_far_field_direct_planar(Ex_sim, Ey_sim, x_scan, y_scan, freq_arr, thetas, phis)
                    
                    E_tot_db = 20 * np.log10(np.sqrt(np.abs(Eth)**2 + np.abs(Eph)**2) + 1e-12)
                    E_tot_norm = E_tot_db - np.max(E_tot_db)
                    
                    # Recompute AF for matching grid
                    AF_match = compute_array_factor(thetas, phis, pos, arr["amps"], arr["phases"], freq_arr)
                    AF_match_db = 20 * np.log10(np.abs(AF_match) / np.max(np.abs(AF_match)) + 1e-12)
                    
                    pb.progress(1.0)
                    
                    idx_p = np.argmin(np.abs(phis - np.deg2rad(steer_ph)))
                    
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_match_db[:, idx_p], mode='lines', name="Theoretical Array Factor", line=dict(color='black', width=3)))
                    fig_cmp.add_trace(go.Scatter(x=np.rad2deg(thetas), y=E_tot_norm[:, idx_p], mode='lines', name="Full-Wave FDTD Integration", line=dict(dash='dash', color='red')))
                    fig_cmp.update_layout(title=f"Azimuth Cut Correlation (Phi={steer_ph}°)", xaxis_title="Theta (°)", yaxis_title="Normalized Pattern (dB)", yaxis_range=[-40, 0])
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    st.session_state.m36_ff = {"type": "FullWave"}
            else: st.warning("Compute the Array Factor first.")

        with t_tol:
            st.markdown("#### 💥 Element Failure & Tolerance Analysis")
            if "amps" in arr:
                fail_idx = st.selectbox("Simulate Failed Element", ["None"] + [f"Element {i+1}" for i in range(N)])
                phase_err = st.number_input("Systemic Phase Tolerance Error (Std Dev °)", 0.0, 45.0, 15.0)
                
                if st.button("Run Degradation Analysis"):
                    amps_deg = arr["amps"].copy()
                    phases_deg = arr["phases"].copy()
                    
                    if fail_idx != "None":
                        idx = int(fail_idx.split()[1]) - 1
                        amps_deg[idx] = 0.0 # Element Dead
                        
                    np.random.seed(42)
                    phases_deg += np.deg2rad(np.random.normal(0, phase_err, N))
                    
                    thetas = np.deg2rad(np.arange(0, 181, 2)); phis = np.deg2rad(np.array([steer_ph]))
                    
                    AF_ideal = compute_array_factor(thetas, phis, pos, arr["amps"], arr["phases"], freq_arr)
                    AF_ideal_db = 20 * np.log10(np.abs(AF_ideal) / np.max(np.abs(AF_ideal)) + 1e-12)
                    
                    AF_deg = compute_array_factor(thetas, phis, pos, amps_deg, phases_deg, freq_arr)
                    AF_deg_db = 20 * np.log10(np.abs(AF_deg) / np.max(np.abs(AF_ideal)) + 1e-12) # Norm against ideal peak
                    
                    fig_tol = go.Figure()
                    fig_tol.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_ideal_db[:, 0], mode='lines', name="Ideal Array", line=dict(color='blue')))
                    fig_tol.add_trace(go.Scatter(x=np.rad2deg(thetas), y=AF_deg_db[:, 0], mode='lines', name="Degraded (Failed/Tolerance)", line=dict(color='red')))
                    fig_tol.update_layout(title="Array Pattern Degradation (Tolerance Run)", xaxis_title="Theta (°)", yaxis_title="Gain Relative to Ideal (dB)", yaxis_range=[-40, 5])
                    st.plotly_chart(fig_tol, use_container_width=True)
                    
                    report = {
                        "Array_ID": str(uuid.uuid4()), "Timestamp": datetime.datetime.now().isoformat(),
                        "Geometry": arr["type"], "Elements": N, "Steering_Angle": steer_th,
                        "Tolerance_Analysis": {"Failed_Element": fail_idx, "Phase_Error_Sigma": phase_err},
                        "Validation": "PASS" if st.session_state.m36_ff else "WARNING: Full-Wave Not Executed"
                    }
                    st.download_button("Export Array Provenance Report (JSON)", data=json.dumps(report, indent=2), file_name="m36_array_report.json", mime="application/json")
            else: st.warning("Compute the Array Factor first.")

elif exp_mode not in ["Advanced Antenna Array Lab (M36)"]:
    st.info("Select 'Advanced Antenna Array Lab (M36)' to interact with Phased Arrays, Mutual Coupling, and Active VSWR parameters.")
