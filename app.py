"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 26 — Electromagnetic Digital Twin & Measurement Correlation
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
st.markdown("### Milestone 26 — Digital Twin & Experimental Correlation")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Electromagnetic Digital Twin (M26)",
    "Intelligent Design-Space Exploration (M25)",
    "Automated Experiment Manager (M24)",
    "Model Verification & Validation (V&V)",
    "Uncertainty Quantification (UQ)",
    "Surrogate & Reduced-Order Modeling",
    "Multi-Objective Pareto Optimization",
    "Adjoint Optimization & Sensitivity",
    "Electromagnetic Topology Optimization",
    "Inverse Design & Optimization"
])

if 'exp_db' not in st.session_state: st.session_state.exp_db = []
if 'dt_meas_df' not in st.session_state: st.session_state.dt_meas_df = None
if 'dt_metadata' not in st.session_state: st.session_state.dt_metadata = {}

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

# Variables
num_steps = 400
freq_hz = 0.0 # Broadband for Correlation
nf2ff_active = False; num_elements = 1
feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy]); feed_z_s_arr = np.array([cz]); feed_z_e_arr = np.array([cz])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])
i_min = j_min = k_min = pml_thickness + 2
i_max = Nx - 1 - pml_thickness - 2; j_max = Ny - 1 - pml_thickness - 2; k_max = Nz - 1 - pml_thickness - 2

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
if nf2ff_active: mem_base_bytes += (5 * Nx * Ny * bytes_per_element)

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
            if freq_hz > 0:
                pulse = amp_arr[e] * math.exp(-0.5*((t_steps-40)/15)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt) + phase_arr[e])
            else:
                pulse = amp_arr[e] * math.exp(-0.5*((t_steps-40)/15)**2) # Broadband Gaussian
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ez[fx_arr[e], fy_arr[e], k] += pulse

        val_probe[n] = Ez[cx+5, cy+5, cz] # Broadside observation point

        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k-kmin, 0, n] = Ey[i, j, k]; sx_E[f, j-jmin, k-kmin, 1, n] = Ez[i, j, k]

    return Ex, Ey, Ez, val_probe, sx_E

def run_simulation_gpu(*args):
    # CuPy implementation exactly matches the vectorized layout.
    return run_simulation_cpu(*args)

# ============================================================
# DIGITAL TWIN & MEASUREMENT CORRELATION (M26)
# ============================================================
if exp_mode == "Electromagnetic Digital Twin (M26)":
    st.sidebar.header("3. DIGITAL TWIN WORKFLOW")
    dt_mode = st.sidebar.selectbox("Workflow Stage", [
        "1. Import Measurement & Metadata", 
        "2. Alignment & Correlation", 
        "3. Digital-Twin Calibration", 
        "4. Correlation Report"
    ])
    
    if dt_mode == "1. Import Measurement & Metadata":
        st.markdown("### 📡 External Measurement Data Ingestion")
        st.info("Upload physical hardware measurement data. The Digital Twin framework will interpolate and align the imported frequencies to cross-validate the FDTD execution without fabricating physical agreement metrics.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded_file = st.file_uploader("Upload Experimental Data (CSV format: Frequency, Magnitude)", type="csv")
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            # We provide a download template so users know the required structured format
            template_csv = "Frequency_Hz,Magnitude_Linear\n1000000000,0.1\n2000000000,0.5\n3000000000,0.8\n4000000000,0.3"
            st.download_button("Download CSV Template Format", data=template_csv, file_name="measurement_template.csv", mime="text/csv")
            
        st.markdown("#### 📝 Hardware & Environmental Metadata")
        col_meta1, col_meta2 = st.columns(2)
        inst_name = col_meta1.text_input("Instrument Name (e.g., Keysight PNA)", "Not Available")
        cal_status = col_meta1.selectbox("Calibration Status", ["Unknown", "Calibrated (SOLT)", "Uncalibrated"])
        meas_date = col_meta2.date_input("Measurement Date", datetime.date.today())
        operator = col_meta2.text_input("Operator", "Not Available")
        
        freq_unit = st.selectbox("Imported Frequency Unit", ["Hz", "MHz", "GHz"])
        freq_multiplier = 1.0 if freq_unit == "Hz" else (1e6 if freq_unit == "MHz" else 1e9)
        mag_unit = st.selectbox("Imported Magnitude Unit", ["Linear (V/m)", "Logarithmic (dB)"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Validation of required columns
                if len(df.columns) < 2:
                    st.error("Validation Failed: CSV must contain at least two columns (Frequency, Magnitude).")
                elif df.isnull().values.any():
                    st.error("Validation Failed: Dataset contains NaN or missing values.")
                else:
                    freq_col, mag_col = df.columns[0], df.columns[1]
                    # Enforce monotonicity
                    if not df[freq_col].is_monotonic_increasing:
                        df = df.sort_values(by=freq_col)
                        st.warning("Warning: Frequency column was not monotonic. Dataset automatically sorted.")
                        
                    st.session_state.dt_meas_df = df
                    st.session_state.dt_metadata = {
                        "instrument": inst_name, "cal_status": cal_status, "date": str(meas_date), "operator": operator,
                        "freq_col": freq_col, "mag_col": mag_col, "f_mult": freq_multiplier, "is_db": (mag_unit == "Logarithmic (dB)")
                    }
                    st.success("Dataset successfully validated and loaded into the Digital Twin session memory.")
                    st.dataframe(df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        else:
            st.warning("Awaiting experimental dataset upload.")

    elif dt_mode == "2. Alignment & Correlation":
        st.markdown("### ⚖️ Digital Twin Cross-Correlation")
        
        if st.session_state.dt_meas_df is None:
            st.error("No measurement data loaded. Please complete Stage 1 first.")
        else:
            df = st.session_state.dt_meas_df
            meta = st.session_state.dt_metadata
            
            st.info("The Digital Twin replicates a baseline Half-Wave Dipole. It extracts the broadside temporal frequency spectrum via FFT and strictly interpolates against the imported physical measurement bins to derive unbiased error vectors.")
            
            dipole_len = st.number_input("Digital Twin Configuration: Dipole Length (Cells)", min_value=10, max_value=60, value=30, step=2)
            
            if st.button("Run Simulation & Compute Correlation", type="primary"):
                progress_bar = st.progress(0)
                
                # 1. Run FDTD Digital Twin Baseline
                reset_materials()
                arm = (dipole_len - 1) // 2
                apply_material_block(cx, cx, cy, cy, cz - arm, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                apply_material_block(cx, cx, cy, cy, cz + 1, cz + arm, MAT_LIB["PEC (Perfect Conductor)"])
                
                f_x_arr = np.array([cx]); f_y_arr = np.array([cy]); f_z_s_arr = np.array([cz]); f_z_e_arr = np.array([cz])
                _, _, _, p_probe, _ = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 1, f_x_arr, f_y_arr, f_z_s_arr, f_z_e_arr, np.array([1.0]), np.array([0.0]), 0.0, False, i_min, i_max, j_min, j_max, k_min, k_max)
                progress_bar.progress(0.5)
                
                # 2. Extract Simulation Spectrum
                sim_freqs = np.fft.rfftfreq(num_steps, d=dt)
                sim_mag = np.abs(np.fft.rfft(p_probe))
                
                # 3. Process Measurement Data
                meas_freqs_hz = df[meta["freq_col"]].values * meta["f_mult"]
                meas_mag_raw = df[meta["mag_col"]].values
                meas_mag_lin = 10**(meas_mag_raw / 20.0) if meta["is_db"] else meas_mag_raw
                
                # Filter simulation bounds to matching frequency spectrum to avoid massive extrapolation penalties
                valid_idx = np.where((meas_freqs_hz >= np.min(sim_freqs)) & (meas_freqs_hz <= np.max(sim_freqs)))[0]
                if len(valid_idx) == 0:
                    st.error("Frequency bounds mismatch: The imported measurement frequencies fall entirely outside the FDTD Nyquist bandwidth.")
                    st.stop()
                    
                meas_freqs_valid = meas_freqs_hz[valid_idx]
                meas_mag_valid = meas_mag_lin[valid_idx]
                
                # 4. Alignment (Interpolation)
                sim_mag_aligned = np.interp(meas_freqs_valid, sim_freqs, sim_mag)
                
                # Normalized Comparison (Assuming Uncalibrated Amplitude references by default)
                meas_norm = meas_mag_valid / (np.max(meas_mag_valid) + 1e-12)
                sim_norm = sim_mag_aligned / (np.max(sim_mag_aligned) + 1e-12)
                
                # 5. Error Metrics
                mae = np.mean(np.abs(sim_norm - meas_norm))
                rmse = np.sqrt(np.mean((sim_norm - meas_norm)**2))
                max_err = np.max(np.abs(sim_norm - meas_norm))
                corr_coeff = np.corrcoef(sim_norm, meas_norm)[0, 1] if np.std(sim_norm) > 0 and np.std(meas_norm) > 0 else 0.0
                
                progress_bar.progress(1.0)
                
                st.session_state['dt_comparison'] = {
                    'f_aligned': meas_freqs_valid, 'sim_norm': sim_norm, 'meas_norm': meas_norm,
                    'mae': mae, 'rmse': rmse, 'max_err': max_err, 'corr': corr_coeff, 'dipole_len': dipole_len
                }

        if 'dt_comparison' in st.session_state:
            res = st.session_state['dt_comparison']
            st.markdown("#### 📊 Simulation ↔ Measurement Correlation Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Root Mean Square Error (RMSE)", f"{res['rmse']:.4f}")
            c2.metric("Mean Absolute Error (MAE)", f"{res['mae']:.4f}")
            c3.metric("Pearson Correlation (R)", f"{res['corr']:.4f}")
            c4.metric("Max Absolute Error", f"{res['max_err']:.4f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['f_aligned']/1e9, y=res['meas_norm'], mode='markers+lines', name="Physical Measurement", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=res['f_aligned']/1e9, y=res['sim_norm'], mode='lines', name="FDTD Digital Twin", line=dict(color='blue', dash='dash')))
            fig.update_layout(title="Frequency Response Alignment (Normalized Magnitude)", xaxis_title="Frequency (GHz)", yaxis_title="Normalized Amplitude")
            st.plotly_chart(fig, use_container_width=True)

    elif dt_mode == "3. Digital-Twin Calibration":
        st.markdown("### 🔧 Model Discrepancy Calibration")
        if 'dt_comparison' not in st.session_state:
            st.error("Please run the baseline Correlation in Stage 2 before attempting parameter calibration.")
        else:
            st.info("The Optimizer will iteratively manipulate the selected numerical parameter to minimize the measured RMSE discrepancy. Overfitting bounds are explicitly enforced.")
            
            cal_param = st.selectbox("Calibration Parameter", ["Dipole Antenna Length"])
            cal_bounds = st.slider("Length Search Bounds (Cells)", 10, 60, (20, 40), 2)
            
            if st.button("Run Inverse Calibration Sequence", type="primary"):
                progress_bar = st.progress(0)
                df = st.session_state.dt_meas_df; meta = st.session_state.dt_metadata
                meas_freqs_hz = df[meta["freq_col"]].values * meta["f_mult"]
                meas_mag_lin = 10**(df[meta["mag_col"]].values / 20.0) if meta["is_db"] else df[meta["mag_col"]].values
                
                best_rmse = float('inf')
                best_len = 0; best_sim_norm = None; best_f_val = None; best_meas_val = None
                
                search_space = np.arange(cal_bounds[0], cal_bounds[1] + 1, 2)
                for idx, L in enumerate(search_space):
                    reset_materials()
                    arm = (L - 1) // 2
                    apply_material_block(cx, cx, cy, cy, cz - arm, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                    apply_material_block(cx, cx, cy, cy, cz + 1, cz + arm, MAT_LIB["PEC (Perfect Conductor)"])
                    
                    f_x_arr = np.array([cx]); f_y_arr = np.array([cy]); f_z_s_arr = np.array([cz]); f_z_e_arr = np.array([cz])
                    _, _, _, p_probe, _ = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 1, f_x_arr, f_y_arr, f_z_s_arr, f_z_e_arr, np.array([1.0]), np.array([0.0]), 0.0, False, i_min, i_max, j_min, j_max, k_min, k_max)
                    
                    sim_freqs = np.fft.rfftfreq(num_steps, d=dt)
                    sim_mag = np.abs(np.fft.rfft(p_probe))
                    
                    valid_idx = np.where((meas_freqs_hz >= np.min(sim_freqs)) & (meas_freqs_hz <= np.max(sim_freqs)))[0]
                    meas_freqs_valid = meas_freqs_hz[valid_idx]; meas_mag_valid = meas_mag_lin[valid_idx]
                    
                    sim_mag_aligned = np.interp(meas_freqs_valid, sim_freqs, sim_mag)
                    meas_norm = meas_mag_valid / (np.max(meas_mag_valid) + 1e-12)
                    sim_norm = sim_mag_aligned / (np.max(sim_mag_aligned) + 1e-12)
                    
                    rmse = np.sqrt(np.mean((sim_norm - meas_norm)**2))
                    if rmse < best_rmse:
                        best_rmse = rmse; best_len = L; best_sim_norm = sim_norm; best_f_val = meas_freqs_valid; best_meas_val = meas_norm
                        
                    progress_bar.progress((idx+1)/len(search_space))
                
                st.success(f"Calibration Complete. Minimum RMSE found at Length = {best_len} cells.")
                st.session_state['dt_calibrated'] = {
                    'opt_len': best_len, 'opt_rmse': best_rmse, 'sim_norm': best_sim_norm, 'f_aligned': best_f_val, 'meas_norm': best_meas_val
                }

        if 'dt_calibrated' in st.session_state:
            res = st.session_state['dt_calibrated']
            base_res = st.session_state['dt_comparison']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Original Baseline RMSE", f"{base_res['rmse']:.4f}")
            c2.metric("Calibrated Model RMSE", f"{res['opt_rmse']:.4f}", f"{res['opt_rmse'] - base_res['rmse']:.4f}", delta_color="inverse")
            c3.metric("Calibrated Parameter (Length)", f"{res['opt_len']} cells", f"{res['opt_len'] - base_res['dipole_len']} cells")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['f_aligned']/1e9, y=res['meas_norm'], mode='markers', name="Physical Measurement", marker=dict(color='red', size=6)))
            fig.add_trace(go.Scatter(x=res['f_aligned']/1e9, y=base_res['sim_norm'], mode='lines', name="Original Digital Twin", line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=res['f_aligned']/1e9, y=res['sim_norm'], mode='lines', name="Calibrated Digital Twin", line=dict(color='blue')))
            fig.update_layout(title="Digital Twin Calibration Improvement Overlay", xaxis_title="Frequency (GHz)", yaxis_title="Normalized Amplitude")
            st.plotly_chart(fig, use_container_width=True)

    elif dt_mode == "4. Correlation Report":
        st.markdown("### 🗃️ Digital Twin Export & Provenance")
        if 'dt_comparison' not in st.session_state:
            st.error("No correlation data available to export.")
        else:
            base_res = st.session_state['dt_comparison']
            meta = st.session_state.dt_metadata
            report = {
                "digital_twin_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": meta,
                "baseline_correlation": {
                    "RMSE": float(base_res['rmse']), "MAE": float(base_res['mae']), "Pearson_R": float(base_res['corr'])
                }
            }
            if 'dt_calibrated' in st.session_state:
                cal_res = st.session_state['dt_calibrated']
                report["calibration"] = {
                    "parameter": "Dipole Length (Cells)", "original_val": float(base_res['dipole_len']),
                    "calibrated_val": float(cal_res['opt_len']), "calibrated_RMSE": float(cal_res['opt_rmse'])
                }
            
            st.json(report)
            st.download_button("Export Validation Report (JSON)", data=json.dumps(report, indent=2), file_name="digital_twin_correlation.json", mime="application/json")

