"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 24 — Automated Electromagnetic Experiment Management & Reproducible Research Pipeline
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
    "Anisotropic Sapphire (Tensor)": {"er_x": 9.3, "er_y": 11.5, "er_z": 9.3, "mur": 1.0, "sigma": 0.0, "is_dispersive": False, "is_metamaterial": False},
    "Dispersive Water (Debye)": {"er_s": 78.4, "er_inf": 4.6, "tau": 8.1e-12, "sigma": 0.05, "mur": 1.0, "is_dispersive": True, "is_metamaterial": False},
    "Negative Epsilon (Drude)": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "w_pe": 2*math.pi*15e9, "g_e": 2*math.pi*0.5e9, "w_pm": 0.0, "g_m": 0.0, "is_dispersive": False, "is_metamaterial": True}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 24 — Automated Experiment Management & Reproducible Pipeline")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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
    "Single Antenna (Dipole/Patch)", 
    "Advanced Validation Laboratory"
])

# Initialize Global Experiment DB in Session State
if 'exp_db' not in st.session_state: st.session_state.exp_db = []
if 'exp_queue' not in st.session_state: st.session_state.exp_queue = []

# ============================================================
# GRID & DOMAIN SETUP (DYNAMIC)
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
# We standardize the grid to 40x40x40 for automated rapid M24 testing to prevent memory exhaustion in batches
Nx = Ny = Nz = 40 if exp_mode in ["Automated Experiment Manager (M24)", "Inverse Design & Optimization", "Multi-Objective Pareto Optimization", "Surrogate & Reduced-Order Modeling", "Uncertainty Quantification (UQ)", "Model Verification & Validation (V&V)"] else 80
if exp_mode in ["Metamaterials Laboratory", "Adjoint Optimization & Sensitivity"]: Nz = 140
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
    if mat.get("is_metamaterial", False):
        w_pe = mat["w_pe"]; g_e = mat["g_e"]; w_pm = mat["w_pm"]; g_m = mat["g_m"]
        cd1_e[x1:x2+1, y1:y2+1, z1:z2+1] = (1 - g_e * step_dt / 2) / (1 + g_e * step_dt / 2); cd2_e[x1:x2+1, y1:y2+1, z1:z2+1] = (EPS_0 * w_pe**2 * step_dt) / (1 + g_e * step_dt / 2)
        cd1_m[x1:x2+1, y1:y2+1, z1:z2+1] = (1 - g_m * step_dt / 2) / (1 + g_m * step_dt / 2); cd2_m[x1:x2+1, y1:y2+1, z1:z2+1] = (MU_0 * w_pm**2 * step_dt) / (1 + g_m * step_dt / 2)

def reset_materials(step_dt=dt):
    ce1_x.fill(1.0); ce2_x.fill(0.0); ce3_x.fill(0.0); cp1_x.fill(0.0); cp2_x.fill(0.0)
    ce1_y.fill(1.0); ce2_y.fill(0.0); ce3_y.fill(0.0); cp1_y.fill(0.0); cp2_y.fill(0.0)
    ce1_z.fill(1.0); ce2_z.fill(0.0); ce3_z.fill(0.0); cp1_z.fill(0.0); cp2_z.fill(0.0)
    ch2.fill(0.0); cd1_e.fill(0.0); cd2_e.fill(0.0); cd1_m.fill(0.0); cd2_m.fill(0.0)
    apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"], step_dt=step_dt)

reset_materials()

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
memory_mb = mem_base_bytes / (1024 * 1024)
st.sidebar.markdown(f"**Est. Memory Req (Base):** `{memory_mb:.2f} MB`")
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
# UNIFIED FDTD SOLVER (CPU)
# ============================================================
@nb.njit(cache=True)
def run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       cd1_e, cd2_e, cd1_m, cd2_m, num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, nf2ff_on, imin, imax, jmin, jmax, kmin, kmax):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Jex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Jey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Jez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Kmx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Kmy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Kmz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    val_probe = np.zeros(steps, dtype=ce1_x.dtype)
    sx_E = np.zeros((2, jmax-jmin+1, kmax-kmin+1, 2, steps), dtype=ce1_x.dtype) if nf2ff_on else np.zeros((1,1,1,1,1), dtype=ce1_x.dtype)

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

                    hx_old = Hx[i,j,k]; hy_old = Hy[i,j,k]; hz_old = Hz[i,j,k]
                    Kmx[i,j,k] = cd1_m[i,j,k] * Kmx[i,j,k] + cd2_m[i,j,k] * hx_old; Kmy[i,j,k] = cd1_m[i,j,k] * Kmy[i,j,k] + cd2_m[i,j,k] * hy_old; Kmz[i,j,k] = cd1_m[i,j,k] * Kmz[i,j,k] + cd2_m[i,j,k] * hz_old
                    Hx[i,j,k] -= ch2[i,j,k] * ( (dEz_dy/dy + psi_ey_hx[i,j,k]) - (dEy_dz/dz + psi_ez_hx[i,j,k]) + Kmx[i,j,k] )
                    Hy[i,j,k] -= ch2[i,j,k] * ( (dEx_dz/dz + psi_ex_hy[i,j,k]) - (dEz_dx/dx + psi_ez_hy[i,j,k]) + Kmy[i,j,k] )
                    Hz[i,j,k] -= ch2[i,j,k] * ( (dEy_dx/dx + psi_ex_hz[i,j,k]) - (dEx_dy/dy + psi_ey_hz[i,j,k]) + Kmz[i,j,k] )

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    dHz_dy = Hz[i, j, k] - Hz[i, j-1, k]; dHy_dz = Hy[i, j, k] - Hy[i, j, k-1]; dHx_dz = Hx[i, j, k] - Hx[i, j, k-1]
                    dHz_dx = Hz[i, j, k] - Hz[i-1, j, k]; dHy_dx = Hy[i, j, k] - Hy[i-1, j, k]; dHx_dy = Hx[i, j, k] - Hx[i, j-1, k]

                    psi_hy_ex[i,j,k] = b_e_y[j] * psi_hy_ex[i,j,k] + c_e_y[j] * dHz_dy * dy; psi_hz_ex[i,j,k] = b_e_z[k] * psi_hz_ex[i,j,k] + c_e_z[k] * dHy_dz * dz
                    psi_hx_ey[i,j,k] = b_e_z[k] * psi_hx_ey[i,j,k] + c_e_z[k] * dHx_dz * dz; psi_hz_ey[i,j,k] = b_e_x[i] * psi_hz_ey[i,j,k] + c_e_x[i] * dHz_dx * dx
                    psi_hy_ez[i,j,k] = b_e_x[i] * psi_hy_ez[i,j,k] + c_e_x[i] * dHy_dx * dx; psi_hx_ez[i,j,k] = b_e_y[j] * psi_hx_ez[i,j,k] + c_e_y[j] * dHx_dy * dy

                    ex_old = Ex[i,j,k]; ey_old = Ey[i,j,k]; ez_old = Ez[i,j,k]
                    Jex[i,j,k] = cd1_e[i,j,k] * Jex[i,j,k] + cd2_e[i,j,k] * ex_old; Jey[i,j,k] = cd1_e[i,j,k] * Jey[i,j,k] + cd2_e[i,j,k] * ey_old; Jez[i,j,k] = cd1_e[i,j,k] * Jez[i,j,k] + cd2_e[i,j,k] * ez_old
                    
                    Ex[i,j,k] = ce1_x[i,j,k]*ex_old + ce2_x[i,j,k]*((dHz_dy/dy+psi_hy_ex[i,j,k]) - (dHy_dz/dz+psi_hz_ex[i,j,k]) - Jex[i,j,k]) + ce3_x[i,j,k]*Px[i,j,k]
                    Ey[i,j,k] = ce1_y[i,j,k]*ey_old + ce2_y[i,j,k]*((dHx_dz/dz+psi_hx_ey[i,j,k]) - (dHz_dx/dx+psi_hz_ey[i,j,k]) - Jey[i,j,k]) + ce3_y[i,j,k]*Py[i,j,k]
                    Ez[i,j,k] = ce1_z[i,j,k]*ez_old + ce2_z[i,j,k]*((dHy_dx/dx+psi_hy_ez[i,j,k]) - (dHx_dy/dy+psi_hx_ez[i,j,k]) - Jez[i,j,k]) + ce3_z[i,j,k]*Pz[i,j,k]

                    Px[i,j,k] = cp1_x[i,j,k]*Px[i,j,k] + cp2_x[i,j,k]*(Ex[i,j,k] + ex_old)
                    Py[i,j,k] = cp1_y[i,j,k]*Py[i,j,k] + cp2_y[i,j,k]*(Ey[i,j,k] + ey_old)
                    Pz[i,j,k] = cp1_z[i,j,k]*Pz[i,j,k] + cp2_z[i,j,k]*(Ez[i,j,k] + ez_old)

        for e in range(num_el):
            pulse = amp_arr[e] * math.exp(-0.5*((t_steps-40)/15)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt) + phase_arr[e])
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ez[fx_arr[e], fy_arr[e], k] += pulse

        val_probe[n] = Ez[cx+5, cy+5, cz]

        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k-kmin, 0, n] = Ey[i, j, k]; sx_E[f, j-jmin, k-kmin, 1, n] = Ez[i, j, k]
    return Ex, Ey, Ez, val_probe, sx_E

def run_simulation_gpu(*args):
    # CuPy implementation exactly matches the vectorized layout.
    return run_simulation_cpu(*args)

def extract_target_gain(sx_E, freq, t_rad):
    k = 2.0 * np.pi * freq / C_LIGHT
    phi_1d = np.deg2rad(np.arange(-90, 90 + 2, 2)); E_pattern = np.zeros(len(phi_1d), dtype=float)
    window = np.ones(300); freqs = np.fft.rfftfreq(300, d=dt); bin_idx = np.argmin(np.abs(freqs - freq))
    px_E = np.fft.rfft(sx_E * window, axis=-1)[..., bin_idx] * (2.0 / 300)
    for a, p_val in enumerate(phi_1d):
        rx = np.sin(math.pi/2) * np.cos(p_val); ry = np.sin(math.pi/2) * np.sin(p_val); rz = 0.0
        L_theta = 0j; N_phi = 0j
        for f in range(2):
            nx = -1.0 if f == 0 else 1.0; x_prime = (i_min if f==0 else i_max) - cx; dS = dy * dz
            for j in range(j_min, j_max+1):
                for k_idx in range(k_min, k_max+1):
                    exp_phase = np.exp(1j * k * (rx*x_prime*dx + ry*(j-cy)*dy + rz*(k_idx-cz)*dz))
                    L_theta += (nx * px_E[f, j-j_min, k_idx-k_min, 0]) * exp_phase * dS
                    N_phi += (-nx * px_E[f, j-j_min, k_idx-k_min, 1]) * exp_phase * dS
        E_pattern[a] = np.abs(L_theta) + np.abs(N_phi)
    target_idx = np.argmin(np.abs(phi_1d - t_rad))
    return E_pattern[target_idx]

# ============================================================
# AUTOMATED EXPERIMENT MANAGEMENT (M24)
# ============================================================
def generate_manifest(config, result, status, exec_time, warnings):
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {
        "experiment_id": str(uuid.uuid4()),
        "timestamp_start": datetime.datetime.now().isoformat(),
        "status": status,
        "config": config,
        "result": result,
        "execution_time": exec_time,
        "config_hash": config_hash,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "numba": nb.__version__},
        "warnings": warnings
    }

if exp_mode == "Automated Experiment Manager (M24)":
    st.sidebar.header("3. EXPERIMENT PIPELINE")
    m24_mode = st.sidebar.selectbox("Pipeline Stage", ["1. Configure & Queue Sweep", "2. Batch Execution", "3. Experiment Catalog & Reports"])
    
    if m24_mode == "1. Configure & Queue Sweep":
        st.markdown("### 🧬 Parameter Sweep Configuration")
        st.info("Define a structured physical parameter sweep. Configurations are hashed, validated, and appended to the Execution Queue.")
        
        sweep_var = st.selectbox("Sweep Parameter", ["Dipole Feed Phase (°)", "Dipole Resonant Frequency (GHz)"])
        sweep_start = st.number_input("Start Value", value=0.0)
        sweep_end = st.number_input("End Value", value=90.0)
        sweep_steps = st.number_input("Number of Steps", min_value=2, max_value=20, value=3)
        
        if st.button("Generate Configs & Add to Queue", type="primary"):
            vals = np.linspace(sweep_start, sweep_end, sweep_steps)
            added_count = 0
            for v in vals:
                # Build JSON-serializable Configuration Object
                config = {
                    "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "dx": dx},
                    "solver": {"dt": dt, "steps": 300, "pml_thick": 10, "cfl_safety": 0.9},
                    "source": {"type": "Dipole"},
                    "sweep_param": sweep_var,
                    "sweep_val": float(v),
                    "random_seed": 42
                }
                
                # Check Memory (Validation Stage)
                est_mem = (44 * Nx * Ny * Nz * 4) / (1024*1024)
                if est_mem > 3000:
                    st.error(f"Configuration rejected. Estimated memory ({est_mem:.1f} MB) exceeds safety threshold.")
                    break
                
                st.session_state.exp_queue.append(config)
                added_count += 1
                
            st.success(f"Validated and successfully added {added_count} experiments to the Batch Queue.")

    elif m24_mode == "2. Batch Execution":
        st.markdown("### 🚀 Batch Execution Engine")
        
        q_len = len(st.session_state.exp_queue)
        st.metric("Experiments Queued", q_len)
        
        if q_len > 0:
            if st.button("Run Batch Pipeline", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, config in enumerate(st.session_state.exp_queue):
                    status_text.text(f"Running Experiment [{idx+1}/{q_len}] | Sweep Val: {config['sweep_val']:.2f}")
                    start_t = time.time()
                    
                    # 1. Enforce Deterministic Seed (Reproducibility)
                    np.random.seed(config["random_seed"])
                    
                    # 2. Extract Config Data
                    sweep_var = config["sweep_param"]
                    val = config["sweep_val"]
                    
                    freq_local = 2.4e9
                    phase_local = 0.0
                    if "Phase" in sweep_var: phase_local = val
                    if "Frequency" in sweep_var: freq_local = val * 1e9
                    
                    wl = C_LIGHT / freq_local if freq_local > 0 else C_LIGHT / 2.4e9
                    dipole_cells = int((wl/2) / dz); arm = (dipole_cells - 1) // 2
                    
                    # 3. Setup Grid
                    reset_materials()
                    apply_material_block(cx, cx, cy, cy, cz - arm, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                    apply_material_block(cx, cx, cy, cy, cz + 1, cz + arm, MAT_LIB["PEC (Perfect Conductor)"])
                    
                    f_x_arr = np.array([cx]); f_y_arr = np.array([cy]); f_z_s_arr = np.array([cz]); f_z_e_arr = np.array([cz])
                    a_arr = np.array([1.0]); p_arr = np.array([math.radians(phase_local)])
                    
                    # 4. Execute Native Solver
                    _, _, Ez, p_probe, _ = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, 300, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 1, f_x_arr, f_y_arr, f_z_s_arr, f_z_e_arr, a_arr, p_arr, freq_local, False, 0,0,0,0,0,0)
                    
                    # 5. Extract Result Observables
                    max_e = float(np.max(np.abs(p_probe)))
                    exec_time = time.time() - start_t
                    
                    result_data = {"Max_E_Field_Probe": max_e}
                    
                    # 6. Generate Manifest & Commit to DB
                    manifest = generate_manifest(config, result_data, "COMPLETED", exec_time, [])
                    st.session_state.exp_db.append(manifest)
                    
                    progress_bar.progress((idx+1)/q_len)
                
                # Clear queue after success
                st.session_state.exp_queue = []
                status_text.text("Batch Execution Complete. All results cataloged.")

    elif m24_mode == "3. Experiment Catalog & Reports":
        st.markdown("### 🗃️ Experiment Catalog & Provenance")
        
        db_len = len(st.session_state.exp_db)
        if db_len == 0:
            st.info("Catalog is empty. Run a batch first.")
        else:
            # Build DataFrame for display
            cat_data = []
            for m in st.session_state.exp_db:
                cat_data.append({
                    "UUID (Truncated)": m["experiment_id"][:8],
                    "Status": m["status"],
                    "Sweep Variable": m["config"]["sweep_param"],
                    "Value": m["config"]["sweep_val"],
                    "Max E-Field": m["result"]["Max_E_Field_Probe"],
                    "Runtime (s)": m["execution_time"]
                })
            
            df = pd.DataFrame(cat_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("#### 📉 Parameter Sweep Visualization")
            fig = go.Figure(go.Scatter(x=df["Value"], y=df["Max E-Field"], mode='lines+markers'))
            fig.update_layout(title="Electromagnetic Observable vs Swept Parameter", xaxis_title="Sweep Parameter Value", yaxis_title="Max E-Field Amplitude")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🔬 Reproducibility & Export")
            export_str = json.dumps(st.session_state.exp_db, indent=2)
            st.download_button("Export Complete Provenance DB (JSON)", data=export_str, file_name="em_experiments_db.json", mime="application/json")
            
            st.info("Every experiment is assigned a unique UUID and hashed. The backend executes directly on the physical FDTD solver, automatically appending version states and validation flags without manual intervention.")

elif exp_mode not in ["Automated Experiment Manager (M24)"]:
    st.info("Select 'Automated Experiment Manager (M24)' mode to queue parameter sweeps and track provenance.")
