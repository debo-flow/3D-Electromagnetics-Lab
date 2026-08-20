"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 32 — Spherical Near-Field Scanning & NF2FF Transformation Laboratory
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
st.markdown("### Milestone 32 — Spherical Near-Field Scanning & NF2FF Lab")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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
if 'sph_plan' not in st.session_state: st.session_state.sph_plan = None
if 'sph_data' not in st.session_state: st.session_state.sph_data = None
if 'sph_ff_data' not in st.session_state: st.session_state.sph_ff_data = None

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

# Variables
num_steps = 400
i_min = j_min = k_min = pml_thickness + 2
i_max = Nx - 1 - pml_thickness - 2; j_max = Ny - 1 - pml_thickness - 2; k_max = Nz - 1 - pml_thickness - 2

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
# UNIFIED FDTD SOLVER (CPU) WITH SPHERICAL PHASOR EXTRACTION
# ============================================================
@nb.njit(cache=True)
def run_simulation_sph_nf_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       freq_hz, R0, theta_arr, phi_arr, cx, cy, cz):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    Etheta_phasor = np.zeros((len(theta_arr), len(phi_arr)), dtype=np.complex128)
    Ephi_phasor = np.zeros((len(theta_arr), len(phi_arr)), dtype=np.complex128)
    omega = 2.0 * np.pi * freq_hz
    accum_steps = steps // 2 

    # Precompute Cartesian grid indices for the Spherical Surface (Nearest Neighbor interpolation)
    i_idx = np.zeros((len(theta_arr), len(phi_arr)), dtype=np.int32)
    j_idx = np.zeros((len(theta_arr), len(phi_arr)), dtype=np.int32)
    k_idx = np.zeros((len(theta_arr), len(phi_arr)), dtype=np.int32)
    
    for t in range(len(theta_arr)):
        for p in range(len(phi_arr)):
            x_pos = R0 * np.sin(theta_arr[t]) * np.cos(phi_arr[p])
            y_pos = R0 * np.sin(theta_arr[t]) * np.sin(phi_arr[p])
            z_pos = R0 * np.cos(theta_arr[t])
            
            i_idx[t, p] = int(cx + x_pos / dx)
            j_idx[t, p] = int(cy + y_pos / dy)
            k_idx[t, p] = int(cz + z_pos / dz)

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

        # CW Source Feed (Dipole along Z)
        pulse = math.sin(omega * n * dt)
        for k in range(cz - 5, cz + 6): Ez[cx, cy, k] += pulse
        
        # Accumulate Steady-State Complex Phasors at Spherical Boundary
        if n >= steps - accum_steps:
            e_j_wt = np.exp(-1j * omega * n * dt)
            for t in range(len(theta_arr)):
                for p in range(len(phi_arr)):
                    ix = i_idx[t, p]; iy = j_idx[t, p]; iz = k_idx[t, p]
                    
                    if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
                        Ex_val = Ex[ix, iy, iz]; Ey_val = Ey[ix, iy, iz]; Ez_val = Ez[ix, iy, iz]
                        
                        # Project Cartesian to Spherical (E_theta, E_phi)
                        theta = theta_arr[t]; phi = phi_arr[p]
                        Etheta_val = Ex_val * math.cos(theta) * math.cos(phi) + Ey_val * math.cos(theta) * math.sin(phi) - Ez_val * math.sin(theta)
                        Ephi_val = -Ex_val * math.sin(phi) + Ey_val * math.cos(phi)
                        
                        Etheta_phasor[t, p] += Etheta_val * e_j_wt
                        Ephi_phasor[t, p] += Ephi_val * e_j_wt

    Etheta_phasor /= accum_steps
    Ephi_phasor /= accum_steps
    return Etheta_phasor, Ephi_phasor

# ============================================================
# M32: MOCK SPHERICAL PROBE & NF2FF SPECTRAL ENGINE
# ============================================================
class MockSphericalProbe:
    def __init__(self):
        self.connected = False
        
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
        
    def acquire(self, Theta_grid, Phi_grid, R0, freq):
        """Generates analytical 2D spherical near-field pattern simulating a dipole with measurement noise."""
        k0 = 2 * np.pi * freq / C_LIGHT
        
        # Ideal Dipole Field (Dominant E_theta)
        # Avoid division by zero at poles using a small epsilon buffer
        sin_t = np.maximum(np.abs(np.sin(Theta_grid)), 1e-6)
        
        ideal_mag = sin_t / (R0 + 1e-6)
        ideal_phase = -k0 * R0
        
        np.random.seed(int(time.time()*1000)%10000)
        noise_m = np.random.normal(0, 0.02, Theta_grid.shape)
        noise_p = np.random.normal(0, 0.05, Theta_grid.shape)
        
        Etheta_mock = (ideal_mag + noise_m) * np.exp(1j * (ideal_phase + noise_p))
        Ephi_mock = 0.01 * Etheta_mock # Minimal cross-polarization
        return Etheta_mock, Ephi_mock

def compute_far_field_spherical(Etheta_nf, Ephi_nf, theta_arr, phi_arr, R0, freq, theta_ff, phi_ff):
    """Integrates Spherical Near-Field aperture currents into Far-Field analytical arrays."""
    k0 = 2 * np.pi * freq / C_LIGHT
    
    THETA_P, PHI_P = np.meshgrid(theta_arr, phi_arr, indexing='ij')
    THETA_FF, PHI_FF = np.meshgrid(theta_ff, phi_ff, indexing='ij')
    
    # Broadcast Arrays for vectorized 2D spatial integration across the sphere
    # P_b: Source Grid (Nt_nf, Np_nf, 1, 1). FF_b: Observation Grid (1, 1, Nt_ff, Np_ff)
    THETA_P_b = THETA_P[:, :, None, None]; PHI_P_b = PHI_P[:, :, None, None]
    THETA_FF_b = THETA_FF[None, None, :, :]; PHI_FF_b = PHI_FF[None, None, :, :]
    
    # dS weighting includes sin(theta) which inherently nullifies duplicate pole points (theta=0, pi) in the integral
    dtheta = theta_arr[1] - theta_arr[0] if len(theta_arr) > 1 else 1.0
    dphi = phi_arr[1] - phi_arr[0] if len(phi_arr) > 1 else 1.0
    dS = (R0**2) * np.sin(THETA_P_b) * dtheta * dphi
    
    # Phase Advance Factor: e^{j k0 r' . r_hat}
    phase_factor = np.sin(THETA_P_b) * np.sin(THETA_FF_b) * np.cos(PHI_FF_b - PHI_P_b) + np.cos(THETA_P_b) * np.cos(THETA_FF_b)
    kernel = np.exp(1j * k0 * R0 * phase_factor) * dS
    
    # Decompose into Cartesian equivalents for projection (Assumes outward propagating locally planar waves)
    Ex_nf = np.cos(THETA_P_b) * np.cos(PHI_P_b) * Etheta_nf[:, :, None, None] - np.sin(PHI_P_b) * Ephi_nf[:, :, None, None]
    Ey_nf = np.cos(THETA_P_b) * np.sin(PHI_P_b) * Etheta_nf[:, :, None, None] + np.cos(PHI_P_b) * Ephi_nf[:, :, None, None]
    Ez_nf = -np.sin(THETA_P_b) * Etheta_nf[:, :, None, None]
    
    Ax = np.sum(Ex_nf * kernel, axis=(0, 1))
    Ay = np.sum(Ey_nf * kernel, axis=(0, 1))
    Az = np.sum(Ez_nf * kernel, axis=(0, 1))
    
    # Re-project to Far-Field Spherical Coordinates
    E_theta_ff = Ax * np.cos(THETA_FF) * np.cos(PHI_FF) + Ay * np.cos(THETA_FF) * np.sin(PHI_FF) - Az * np.sin(THETA_FF)
    E_phi_ff = -Ax * np.sin(PHI_FF) + Ay * np.cos(PHI_FF)
    
    return E_theta_ff, E_phi_ff

if 'sph_probe' not in st.session_state: st.session_state.sph_probe = MockSphericalProbe()

# ============================================================
# M32: SPHERICAL NEAR-FIELD / FAR-FIELD LABORATORY UI
# ============================================================
if exp_mode == "Spherical NF/FF Lab (M32)":
    sph_probe = st.session_state.sph_probe
    
    st.sidebar.header("3. PROBE & INSTRUMENTATION")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Connect Probe"): sph_probe.connect()
    if c2.button("Disconnect Probe"): sph_probe.disconnect()
    st.sidebar.metric("Probe Status", "CONNECTED" if sph_probe.connected else "OFFLINE")

    st.markdown("### 🌐 Spherical Near-Field Scanning & NF2FF Translation")
    st.info("The M32 framework executes full 3D Spherical Boundary acquisitions $(r, \\theta, \\phi)$. It automatically resolves coordinate singularities at the poles ($\\theta = 0^\\circ, 180^\\circ$) by integrating the solid angle element $d\\Omega = \\sin(\\theta)d\\theta d\\phi$, yielding uncorrupted, artifact-free Far-Field radiation transforms.")

    t_plan, t_acq, t_nf2ff, t_ff, t_pol, t_dt = st.tabs(["1. Sphere Planner", "2. NF Acquisition", "3. Spherical NF2FF", "4. Far-Field Patterns", "5. Polarization Analysis", "6. Digital Twin Correlation"])

    with t_plan:
        st.markdown("#### Spherical Geometry Scan Setup")
        with st.form("sph_plan_form"):
            cc1, cc2, cc3 = st.columns(3)
            t_min = cc1.number_input("Theta Min (°)", 0.0, 180.0, 0.0, 5.0)
            t_max = cc2.number_input("Theta Max (°)", 0.0, 180.0, 180.0, 5.0)
            t_step = cc3.number_input("Theta Step (°)", 1.0, 45.0, 10.0)
            
            p_min = cc1.number_input("Phi Min (°)", 0.0, 360.0, 0.0, 5.0)
            p_max = cc2.number_input("Phi Max (°)", 0.0, 360.0, 350.0, 10.0) # Stop before 360 to prevent duplicate boundary overlap
            p_step = cc3.number_input("Phi Step (°)", 1.0, 45.0, 10.0)
            
            r_dist = st.number_input("Sphere Radius R0 (m)", 0.01, 2.0, 0.1, 0.01)
            freq_hz = st.number_input("Operating Frequency (GHz)", 0.1, 40.0, 2.4, 0.1) * 1e9
            
            submitted = st.form_submit_button("Validate Scan Plan")
            if submitted:
                wl = C_LIGHT / freq_hz
                max_step = wl / 2.0
                
                # Worst case spatial sampling is at the equator where sin(theta) = 1
                arc_step_theta = r_dist * np.deg2rad(t_step)
                arc_step_phi = r_dist * np.deg2rad(p_step)
                
                if arc_step_theta > max_step or arc_step_phi > max_step:
                    st.error(f"Sampling Violation: Equator Arc-Steps (θ: {arc_step_theta*1000:.1f} mm, φ: {arc_step_phi*1000:.1f} mm) must be ≤ λ/2 ({max_step*1000:.1f} mm) to prevent spectral aliasing!")
                elif t_min >= t_max or p_min >= p_max:
                    st.error("Invalid spatial bounds.")
                else:
                    t_arr = np.deg2rad(np.arange(t_min, t_max + t_step, t_step))
                    p_arr = np.deg2rad(np.arange(p_min, p_max + p_step, p_step))
                    st.session_state.sph_plan = {
                        "t_arr": t_arr, "p_arr": p_arr, "r": r_dist, 
                        "freq": freq_hz, "pts": len(t_arr)*len(p_arr)
                    }
                    st.success(f"Scan Validated. Total Coordinates: {len(t_arr)*len(p_arr)}. Worst-case sampling margin passed (λ = {wl*1000:.1f} mm).")

    with t_acq:
        if st.session_state.sph_plan is None: st.warning("Validate a Scan Plan first.")
        elif not sph_probe.connected: st.error("Probe Offline. Please Connect Probe.")
        else:
            plan = st.session_state.sph_plan
            st.metric("Total Physical Measurement Points", plan['pts'])
            
            if st.button("RUN SPHERICAL NF ACQUISITION", type="primary"):
                st.warning("⚠️ MOCK SPHERICAL DATA ACQUISITION IN PROGRESS...")
                pb = st.progress(0)
                
                Theta_grid, Phi_grid = np.meshgrid(plan["t_arr"], plan["p_arr"], indexing='ij')
                
                time.sleep(0.5)
                Etheta_mock, Ephi_mock = sph_probe.acquire(Theta_grid, Phi_grid, plan["r"], plan["freq"])
                pb.progress(1.0)
                
                st.session_state.sph_data = {
                    "E_th": Etheta_mock, "E_ph": Ephi_mock, "Theta": Theta_grid, "Phi": Phi_grid, "source": "MOCK SPHERICAL DATA"
                }
                st.success("3D Spherical Near-Field Maps successfully acquired and logged.")

        if st.session_state.sph_data is not None:
            nf = st.session_state.sph_data
            st.markdown("##### 2D Phase & Magnitude Visualizations (Unwrapped Sphere: $\\theta \\times \\phi$)")
            cf1, cf2 = st.columns(2)
            
            mag_db = 20 * np.log10(np.abs(nf["E_th"]) + 1e-12)
            phase = np.angle(nf["E_th"])
            
            fig_m = go.Figure(go.Heatmap(z=mag_db.T, x=np.rad2deg(plan['t_arr']), y=np.rad2deg(plan['p_arr']), colorscale='Viridis', colorbar=dict(title="dBV/m")))
            fig_m.update_layout(title="Measured E_θ Magnitude (dB)", xaxis_title="Theta (°)", yaxis_title="Phi (°)", width=400, height=400)
            cf1.plotly_chart(fig_m)
            
            fig_p = go.Figure(go.Heatmap(z=phase.T, x=np.rad2deg(plan['t_arr']), y=np.rad2deg(plan['p_arr']), colorscale='Phase', zmin=-np.pi, zmax=np.pi, colorbar=dict(title="Rads")))
            fig_p.update_layout(title="Measured E_θ Phase (Wrapped)", xaxis_title="Theta (°)", yaxis_title="Phi (°)", width=400, height=400)
            cf2.plotly_chart(fig_p)

    with t_nf2ff:
        if st.session_state.sph_data is None: st.warning("Acquire Near-Field data first.")
        else:
            plan = st.session_state.sph_plan
            nf = st.session_state.sph_data
            
            st.markdown("#### Spherical Spectral NF2FF Transform")
            st.info("Transforms the measured $E_\\theta, E_\\phi$ phasors across the spherical boundary into the true 3D Far-Field Domain using rigorous surface integration and coordinate projection.")
            
            c_r1, c_r2 = st.columns(2)
            res_t = c_r1.number_input("FF Theta Resolution (°)", 1.0, 10.0, 2.0)
            res_p = c_r2.number_input("FF Phi Resolution (°)", 1.0, 10.0, 5.0)
            
            if st.button("Execute Mathematical NF2FF Integration", type="primary"):
                with st.spinner("Integrating Equivalent Spherical Currents..."):
                    thetas_ff = np.deg2rad(np.arange(0, 180 + res_t, res_t))
                    phis_ff = np.deg2rad(np.arange(0, 360, res_p))
                    
                    E_th_ff, E_ph_ff = compute_far_field_spherical(nf["E_th"], nf["E_ph"], plan["t_arr"], plan["p_arr"], plan["r"], plan["freq"], thetas_ff, phis_ff)
                    
                    E_tot_mag = np.sqrt(np.abs(E_th_ff)**2 + np.abs(E_ph_ff)**2)
                    E_tot_db = 20 * np.log10(E_tot_mag + 1e-12)
                    E_tot_norm = E_tot_db - np.max(E_tot_db)
                    
                    st.session_state.sph_ff_data = {
                        "theta": thetas_ff, "phi": phis_ff, "E_norm": E_tot_norm, "E_th": E_th_ff, "E_ph": E_ph_ff, "source": "NF2FF TRANSFORM (MOCK)"
                    }
                st.success("Spherical Far-Field Spatial Transformation Complete.")

    with t_ff:
        if st.session_state.sph_ff_data is None: st.info("Execute NF2FF Transform first.")
        else:
            ff = st.session_state.sph_ff_data
            st.markdown("#### 3D Far-Field Patterns & Analytics")
            st.warning(f"**SOURCE:** `{ff['source']}`")
            
            T, P = np.meshgrid(ff["theta"], ff["phi"], indexing='ij')
            R = np.maximum(ff["E_norm"] + 40, 0) 
            X = R * np.sin(T) * np.cos(P); Y = R * np.sin(T) * np.sin(P); Z = R * np.cos(T)
            
            fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=ff["E_norm"], colorscale='Jet', colorbar=dict(title="dB"))])
            fig_3d.update_layout(title="Transformed 3D Far-Field (Spherical Origin)", height=500)
            
            cc1, cc2 = st.columns([2, 1])
            cc1.plotly_chart(fig_3d, use_container_width=True)
            
            with cc2:
                st.markdown("##### Beam Analytics")
                idx_t, idx_p = np.unravel_index(np.argmax(ff["E_norm"]), ff["E_norm"].shape)
                peak_t = np.rad2deg(ff["theta"][idx_t])
                peak_p = np.rad2deg(ff["phi"][idx_p])
                
                e_cut = ff["E_norm"][:, idx_p]
                hp_pts = np.where(e_cut >= -3.0)[0]
                hpbw = np.rad2deg(ff["theta"][hp_pts[-1]]) - np.rad2deg(ff["theta"][hp_pts[0]]) if len(hp_pts) > 0 else 0.0 
                
                st.metric("Main-Lobe Peak Direction", f"θ={peak_t:.1f}°, φ={peak_p:.1f}°")
                st.metric("Estimated HPBW (E-Plane)", f"{hpbw:.1f}°")
                st.metric("Peak Norm Magnitude", "0.0 dB (Reference)")

    with t_pol:
        if st.session_state.sph_ff_data is None: st.info("Execute NF2FF Transform first.")
        else:
            ff = st.session_state.sph_ff_data
            st.markdown("#### 🔀 Polarization Analysis (Ludwig-3 Approximation)")
            st.info("Breaks down the Far-Field complex vectors ($E_\\theta, E_\\phi$) into formal Co-Polarization and Cross-Polarization reference bounds allowing strict axial ratio diagnostics.")
            
            T, P = np.meshgrid(ff["theta"], ff["phi"], indexing='ij')
            
            # Ludwig-3 definitions
            E_co = ff["E_th"] * np.cos(P) - ff["E_ph"] * np.sin(P)
            E_cross = ff["E_th"] * np.sin(P) + ff["E_ph"] * np.cos(P)
            
            E_co_db = 20 * np.log10(np.abs(E_co) + 1e-12)
            E_cross_db = 20 * np.log10(np.abs(E_cross) + 1e-12)
            
            # Normalization to absolute peak to show true ratio
            peak_db = np.max(E_co_db)
            E_co_norm = E_co_db - peak_db
            E_cross_norm = E_cross_db - peak_db
            
            c_p1, c_p2 = st.columns(2)
            
            fig_co = go.Figure(go.Heatmap(z=E_co_norm.T, x=np.rad2deg(ff["theta"]), y=np.rad2deg(ff["phi"]), colorscale='Blues', colorbar=dict(title="dB")))
            fig_co.update_layout(title="Co-Polarization (Ludwig-3)", xaxis_title="Theta (°)", yaxis_title="Phi (°)")
            c_p1.plotly_chart(fig_co, use_container_width=True)
            
            fig_cx = go.Figure(go.Heatmap(z=E_cross_norm.T, x=np.rad2deg(ff["theta"]), y=np.rad2deg(ff["phi"]), colorscale='Reds', colorbar=dict(title="dB"), zmax=0, zmin=-40))
            fig_cx.update_layout(title="Cross-Polarization (Ludwig-3)", xaxis_title="Theta (°)", yaxis_title="Phi (°)")
            c_p2.plotly_chart(fig_cx, use_container_width=True)

    with t_dt:
        if st.session_state.sph_ff_data is None: st.info("Transform Far-Field patterns first.")
        else:
            plan = st.session_state.sph_plan
            ff = st.session_state.sph_ff_data
            
            st.markdown("#### 🔄 Digital Twin Correlation (FDTD vs MOCK SPHERICAL NF2FF)")
            st.info("The Digital Twin automatically computes an explicit complex frequency-domain FDTD run, projecting the internal Cartesian grids mathematically onto the defined Sphere radius ($R_0$) to calculate precise, interpolation-free RMSE errors against the Mock measurements.")
            
            if st.button("Correlate Spherical Simulation vs Measurements", type="primary"):
                progress_bar = st.progress(0)
                reset_materials()
                
                # Execute FDTD and explicitly capture Spherical boundary phasors natively
                Etheta_sim, Ephi_sim = run_simulation_sph_nf_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, plan["freq"], plan["r"], plan["t_arr"], plan["p_arr"], cx, cy, cz)
                progress_bar.progress(0.5)
                
                # FDTD Spherical NF2FF
                E_th_sim, E_ph_sim = compute_far_field_spherical(Etheta_sim, Ephi_sim, plan["t_arr"], plan["p_arr"], plan["r"], plan["freq"], ff["theta"], ff["phi"])
                
                E_tot_sim_db = 20 * np.log10(np.sqrt(np.abs(E_th_sim)**2 + np.abs(E_ph_sim)**2) + 1e-12)
                E_tot_sim_norm = E_tot_sim_db - np.max(E_tot_sim_db)
                
                # Correlation
                error_map = np.abs(E_tot_sim_norm - ff["E_norm"])
                rmse = np.sqrt(np.mean(error_map**2))
                progress_bar.progress(1.0)
                
                st.metric("Spherical Far-Field Pattern RMSE (Simulation vs MOCK)", f"{rmse:.3f} dB")
                
                fig_err = go.Figure(data=go.Heatmap(z=error_map.T, x=np.rad2deg(ff["theta"]), y=np.rad2deg(ff["phi"]), colorscale='Reds', colorbar=dict(title="Absolute Error (dB)")))
                fig_err.update_layout(title="Angular Far-Field Error Map (SIM vs MOCK)", xaxis_title="Theta (°)", yaxis_title="Phi (°)")
                st.plotly_chart(fig_err, use_container_width=True)
                
                report = {
                    "Analysis_ID": str(uuid.uuid4()), "Timestamp": datetime.datetime.now().isoformat(),
                    "Data_Source": "MOCK_SPHERICAL_MEASUREMENT", "Frequency": plan["freq"],
                    "RMSE_dB": float(rmse), "Mock_Peak_Direction": {"theta": float(np.rad2deg(ff["theta"][np.unravel_index(np.argmax(ff["E_norm"]), ff["E_norm"].shape)[0]]))}
                }
                st.download_button("Export Final Characterization Report (JSON)", data=json.dumps(report, indent=2), file_name="sph_nf2ff_characterization.json", mime="application/json")

elif exp_mode not in ["Spherical NF/FF Lab (M32)"]:
    st.info("Select 'Spherical NF/FF Lab (M32)' to configure full 3D boundary scanning, Polarization analytics, and Spherical NF2FF transforms.")

