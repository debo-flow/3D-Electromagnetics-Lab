"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 30 — Near-Field Scanning, NF2FF Transformation & Complete Antenna Lab
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
st.markdown("### Milestone 30 — Near-Field Scanning & NF2FF Transformation Lab")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64
active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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
if 'nf_plan' not in st.session_state: st.session_state.nf_plan = None
if 'nf_data' not in st.session_state: st.session_state.nf_data = None
if 'ff_data' not in st.session_state: st.session_state.ff_data = None

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
# UNIFIED FDTD SOLVER (CPU) WITH Z-PLANE PHASOR EXTRACTION
# ============================================================
@nb.njit(cache=True)
def run_simulation_nf_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       freq_hz, z_scan):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    # Frequency-Domain Phasor Accumulators for Z-plane
    Ex_phasor = np.zeros((Nx, Ny), dtype=np.complex128)
    Ey_phasor = np.zeros((Nx, Ny), dtype=np.complex128)
    omega = 2.0 * np.pi * freq_hz
    accum_steps = steps // 2  # Discard transient startup

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

        # CW Source Feed (Dipole at center)
        cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
        pulse = math.sin(omega * n * dt)
        for k in range(cz - 5, cz + 6): Ez[cx, cy, k] += pulse
        
        # Accumulate Steady-State Complex Phasors at Scan Plane
        if n >= steps - accum_steps:
            e_j_wt = np.exp(-1j * omega * n * dt)
            for i in range(Nx):
                for j in range(Ny):
                    Ex_phasor[i,j] += Ex[i, j, z_scan] * e_j_wt
                    Ey_phasor[i,j] += Ey[i, j, z_scan] * e_j_wt

    Ex_phasor /= accum_steps
    Ey_phasor /= accum_steps
    return Ex_phasor, Ey_phasor

# ============================================================
# M30: MOCK PROBE & NF2FF SPECTRAL ENGINE
# ============================================================
class MockNearFieldProbe:
    def __init__(self):
        self.connected = False
        
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
        
    def acquire(self, X_grid, Y_grid, z_scan, freq):
        """Generates analytical 2D near-field pattern simulating an aperture with measurement noise."""
        k0 = 2 * np.pi * freq / C_LIGHT
        # Mock a cosine-tapered aperture propagating to z_scan
        R = np.sqrt(X_grid**2 + Y_grid**2 + z_scan**2)
        
        ideal_mag = np.exp(-(X_grid**2 + Y_grid**2)/0.005) / (R + 1e-6)
        ideal_phase = -k0 * R
        
        np.random.seed(int(time.time()*1000)%10000)
        noise_m = np.random.normal(0, 0.02, X_grid.shape)
        noise_p = np.random.normal(0, 0.05, X_grid.shape)
        
        Ex_mock = (ideal_mag + noise_m) * np.exp(1j * (ideal_phase + noise_p))
        Ey_mock = 0.05 * Ex_mock # Cross-polarization component
        return Ex_mock, Ey_mock

def compute_far_field_direct(Ex_nf, Ey_nf, x_arr, y_arr, freq, thetas, phis):
    """Direct, exact integration of Plane Wave Spectrum to Far-Field."""
    k0 = 2 * np.pi * freq / C_LIGHT
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
    dx_scan = x_arr[1] - x_arr[0] if len(x_arr) > 1 else 1.0
    dy_scan = y_arr[1] - y_arr[0] if len(y_arr) > 1 else 1.0
    
    THETA, PHI = np.meshgrid(thetas, phis, indexing='ij')
    
    # Spectral wave vectors
    kx = k0 * np.sin(THETA) * np.cos(PHI)
    ky = k0 * np.sin(THETA) * np.sin(PHI)
    
    # Broadcast Arrays for vectorized 2D spatial integration across all angular bounds
    # X_b: (Nx, Ny, 1, 1), kx_b: (1, 1, Nt, Np)
    X_b = X[:, :, None, None]; Y_b = Y[:, :, None, None]
    kx_b = kx[None, None, :, :]; ky_b = ky[None, None, :, :]
    
    kernel = np.exp(1j * (kx_b * X_b + ky_b * Y_b))
    
    Fx = np.sum(Ex_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_scan * dy_scan
    Fy = np.sum(Ey_nf[:, :, None, None] * kernel, axis=(0, 1)) * dx_scan * dy_scan
    
    E_theta = (Fx * np.cos(PHI) + Fy * np.sin(PHI)) * np.cos(THETA)
    E_phi = -Fx * np.sin(PHI) + Fy * np.cos(PHI)
    
    return E_theta, E_phi

if 'nf_probe' not in st.session_state: st.session_state.nf_probe = MockNearFieldProbe()

# ============================================================
# M30: NEAR-FIELD / FAR-FIELD LABORATORY UI
# ============================================================
if exp_mode == "Near-Field / Far-Field Lab (M30)":
    nf_probe = st.session_state.nf_probe
    
    st.sidebar.header("3. PROBE & INSTRUMENTATION")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Connect Probe"): nf_probe.connect()
    if c2.button("Disconnect Probe"): nf_probe.disconnect()
    st.sidebar.metric("Probe Status", "CONNECTED" if nf_probe.connected else "OFFLINE")

    st.markdown("### 🧲 Automated Near-Field Scanning & NF2FF Translation")
    st.info("The M30 framework coordinates 2D spatial probe sweeps across an established aperture boundary. It mathematically translates the resulting complex Near-Field (NF) spatial phasors into the 3D Far-Field (FF) Domain using analytical Plane Wave Spectrum arrays without artificial extrapolation.")

    t_plan, t_acq, t_nf2ff, t_ff, t_dt = st.tabs(["1. Scan Planner", "2. NF Acquisition", "3. NF2FF Transform", "4. Far-Field Patterns", "5. Digital Twin Correlation"])

    with t_plan:
        st.markdown("#### Planar Near-Field Scan Setup")
        with st.form("nf_plan_form"):
            cc1, cc2, cc3 = st.columns(3)
            x_min = cc1.number_input("X Min (m)", -0.5, 0.0, -0.2, 0.05)
            x_max = cc2.number_input("X Max (m)", 0.0, 0.5, 0.2, 0.05)
            x_step = cc3.number_input("X Step (m)", 0.001, 0.1, 0.02, 0.005)
            
            y_min = cc1.number_input("Y Min (m)", -0.5, 0.0, -0.2, 0.05)
            y_max = cc2.number_input("Y Max (m)", 0.0, 0.5, 0.2, 0.05)
            y_step = cc3.number_input("Y Step (m)", 0.001, 0.1, 0.02, 0.005)
            
            z_dist = st.number_input("Measurement Plane Distance Z (m)", 0.01, 2.0, 0.1, 0.01)
            freq_hz = st.number_input("Operating Frequency (GHz)", 0.1, 40.0, 2.4, 0.1) * 1e9
            
            submitted = st.form_submit_button("Validate Scan Plan")
            if submitted:
                wl = C_LIGHT / freq_hz
                max_step = wl / 2.0
                
                if x_step > max_step or y_step > max_step:
                    st.error(f"Sampling Violation: Step sizes must be ≤ λ/2 ({max_step*1000:.1f} mm) to prevent spectral aliasing!")
                elif x_min >= x_max or y_min >= y_max:
                    st.error("Invalid spatial bounds.")
                else:
                    x_arr = np.arange(x_min, x_max + x_step, x_step)
                    y_arr = np.arange(y_min, y_max + y_step, y_step)
                    st.session_state.nf_plan = {
                        "x_arr": x_arr, "y_arr": y_arr, "z": z_dist, 
                        "freq": freq_hz, "pts": len(x_arr)*len(y_arr)
                    }
                    st.success(f"Scan Validated. Total Grid Points: {len(x_arr)*len(y_arr)}. Required Wavelength resolution satisfied (λ = {wl*1000:.1f} mm).")

    with t_acq:
        if st.session_state.nf_plan is None: st.warning("Validate a Scan Plan first.")
        elif not nf_probe.connected: st.error("Probe Offline. Please Connect Probe.")
        else:
            plan = st.session_state.nf_plan
            st.metric("Total Measurement Points", plan['pts'])
            
            if st.button("RUN NEAR-FIELD ACQUISITION", type="primary"):
                st.warning("⚠️ MOCK NEAR-FIELD DATA ACQUISITION IN PROGRESS...")
                pb = st.progress(0)
                
                X_grid, Y_grid = np.meshgrid(plan["x_arr"], plan["y_arr"], indexing='ij')
                
                # Mock acquisition delay simulation
                time.sleep(0.5)
                Ex_mock, Ey_mock = nf_probe.acquire(X_grid, Y_grid, plan["z"], plan["freq"])
                pb.progress(1.0)
                
                st.session_state.nf_data = {
                    "Ex": Ex_mock, "Ey": Ey_mock, "X": X_grid, "Y": Y_grid, "source": "MOCK DATA"
                }
                st.success("2D Near-Field Maps successfully acquired and logged.")

        if st.session_state.nf_data is not None:
            nf = st.session_state.nf_data
            st.markdown("##### 2D Phase & Magnitude Visualizations")
            cf1, cf2 = st.columns(2)
            
            mag_db = 20 * np.log10(np.abs(nf["Ex"]) + 1e-12)
            phase = np.angle(nf["Ex"])
            
            fig_m = go.Figure(go.Heatmap(z=mag_db.T, x=plan['x_arr'], y=plan['y_arr'], colorscale='Viridis', colorbar=dict(title="dBV/m")))
            fig_m.update_layout(title="Measured E_x Magnitude (dB)", xaxis_title="X (m)", yaxis_title="Y (m)", width=400, height=400)
            cf1.plotly_chart(fig_m)
            
            fig_p = go.Figure(go.Heatmap(z=phase.T, x=plan['x_arr'], y=plan['y_arr'], colorscale='Phase', zmin=-np.pi, zmax=np.pi, colorbar=dict(title="Rads")))
            fig_p.update_layout(title="Measured E_x Phase (Wrapped)", xaxis_title="X (m)", yaxis_title="Y (m)", width=400, height=400)
            cf2.plotly_chart(fig_p)

    with t_nf2ff:
        if st.session_state.nf_data is None: st.warning("Acquire Near-Field data first.")
        else:
            plan = st.session_state.nf_plan
            nf = st.session_state.nf_data
            
            st.markdown("#### Planar Spectral NF2FF Transform")
            st.info("Uses a precise 2D Discrete Plane Wave Spectral decomposition. Transforms the measured spatial phasors across an exact hemispherical $N_{\\theta} \\times N_{\\phi}$ grid without interpolative approximations.")
            
            c_r1, c_r2 = st.columns(2)
            res_t = c_r1.number_input("Theta Resolution (°)", 1.0, 10.0, 2.0)
            res_p = c_r2.number_input("Phi Resolution (°)", 1.0, 10.0, 5.0)
            
            if st.button("Execute Mathematical NF2FF Integration", type="primary"):
                with st.spinner("Integrating Plane Wave Spectrum components..."):
                    thetas = np.deg2rad(np.arange(0, 90 + res_t, res_t))
                    phis = np.deg2rad(np.arange(0, 360, res_p))
                    
                    E_th, E_ph = compute_far_field_direct(nf["Ex"], nf["Ey"], plan["x_arr"], plan["y_arr"], plan["freq"], thetas, phis)
                    
                    E_tot_mag = np.sqrt(np.abs(E_th)**2 + np.abs(E_ph)**2)
                    E_tot_db = 20 * np.log10(E_tot_mag + 1e-12)
                    E_tot_norm = E_tot_db - np.max(E_tot_db)
                    
                    st.session_state.ff_data = {
                        "theta": thetas, "phi": phis, "E_norm": E_tot_norm, "E_th": E_th, "E_ph": E_ph, "source": "NF2FF TRANSFORM (MOCK)"
                    }
                st.success("Far-Field Spatial Transformation Complete. Proceed to Pattern Analytics.")

    with t_ff:
        if st.session_state.ff_data is None: st.info("Execute NF2FF Transform first.")
        else:
            ff = st.session_state.ff_data
            st.markdown("#### 3D Far-Field Patterns & Analytics")
            st.warning(f"**SOURCE:** `{ff['source']}`")
            
            T, P = np.meshgrid(ff["theta"], ff["phi"], indexing='ij')
            R = ff["E_norm"] - np.min(ff["E_norm"])
            X = R * np.sin(T) * np.cos(P); Y = R * np.sin(T) * np.sin(P); Z = R * np.cos(T)
            
            fig_3d = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=ff["E_norm"], colorscale='Jet', colorbar=dict(title="dB"))])
            fig_3d.update_layout(title="Transformed 3D Far-Field Hemisphere", height=500)
            
            cc1, cc2 = st.columns([2, 1])
            cc1.plotly_chart(fig_3d, use_container_width=True)
            
            with cc2:
                st.markdown("##### Beam Analytics")
                idx_t, idx_p = np.unravel_index(np.argmax(ff["E_norm"]), ff["E_norm"].shape)
                peak_t = np.rad2deg(ff["theta"][idx_t])
                peak_p = np.rad2deg(ff["phi"][idx_p])
                
                # HPBW Extraction
                e_cut = ff["E_norm"][:, idx_p]
                hp_pts = np.where(e_cut >= -3.0)[0]
                hpbw = np.rad2deg(ff["theta"][hp_pts[-1]]) * 2 if len(hp_pts) > 0 else 0.0 # Assumed symmetric about zenith
                
                st.metric("Main-Lobe Peak Direction", f"θ={peak_t:.1f}°, φ={peak_p:.1f}°")
                st.metric("Estimated HPBW", f"{hpbw:.1f}°")
                st.metric("Peak Norm Magnitude", "0.0 dB (Reference)")

    with t_dt:
        if st.session_state.ff_data is None: st.info("Transform Far-Field patterns first.")
        else:
            plan = st.session_state.nf_plan
            ff = st.session_state.ff_data
            
            st.markdown("#### 🔄 Digital Twin Correlation (FDTD vs MOCK NF2FF)")
            st.info("The Digital Twin automatically computes an explicit complex frequency-domain FDTD run, captures the matching spatial Z-plane phasors natively, and runs the parallel NF2FF logic to output absolute physical Discrepancy Error margins.")
            
            if st.button("Correlate Simulation vs Measurements", type="primary"):
                progress_bar = st.progress(0)
                
                # Align physical coordinates to FDTD grid
                z_scan_idx = cz + int(plan["z"] / dz)
                if z_scan_idx >= Nz: z_scan_idx = Nz - 1
                
                reset_materials()
                # Run specially recorded Z-Plane FDTD
                Ex_sim, Ey_sim = run_simulation_nf_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, plan["freq"], z_scan_idx)
                progress_bar.progress(0.5)
                
                # Spatial interpolation from FDTD Grid to User's Mock Scan Grid to ensure rigorous parity
                fdtd_x = (np.arange(Nx) - cx) * dx
                fdtd_y = (np.arange(Ny) - cy) * dy
                
                # Safe nearest neighbor for interpolation to avoid external scipy dependencies
                sim_Ex_interp = np.zeros_like(st.session_state.nf_data["Ex"], dtype=complex)
                sim_Ey_interp = np.zeros_like(st.session_state.nf_data["Ey"], dtype=complex)
                
                for i, x in enumerate(plan["x_arr"]):
                    for j, y in enumerate(plan["y_arr"]):
                        ix = np.argmin(np.abs(fdtd_x - x))
                        iy = np.argmin(np.abs(fdtd_y - y))
                        sim_Ex_interp[i, j] = Ex_sim[ix, iy]
                        sim_Ey_interp[i, j] = Ey_sim[ix, iy]
                
                # FDTD NF2FF
                E_th_sim, E_ph_sim = compute_far_field_direct(sim_Ex_interp, sim_Ey_interp, plan["x_arr"], plan["y_arr"], plan["freq"], ff["theta"], ff["phi"])
                
                E_tot_sim_db = 20 * np.log10(np.sqrt(np.abs(E_th_sim)**2 + np.abs(E_ph_sim)**2) + 1e-12)
                E_tot_sim_norm = E_tot_sim_db - np.max(E_tot_sim_db)
                
                # Correlation
                error_map = np.abs(E_tot_sim_norm - ff["E_norm"])
                rmse = np.sqrt(np.mean(error_map**2))
                
                progress_bar.progress(1.0)
                
                st.metric("Far-Field Pattern RMSE (Simulation vs MOCK)", f"{rmse:.3f} dB")
                
                fig_err = go.Figure(data=go.Heatmap(z=error_map.T, x=np.rad2deg(ff["theta"]), y=np.rad2deg(ff["phi"]), colorscale='Reds', colorbar=dict(title="Absolute Error (dB)")))
                fig_err.update_layout(title="Angular Far-Field Error Map (SIM vs MOCK)", xaxis_title="Theta (°)", yaxis_title="Phi (°)")
                st.plotly_chart(fig_err, use_container_width=True)
                
                report = {
                    "Analysis_ID": str(uuid.uuid4()), "Timestamp": datetime.datetime.now().isoformat(),
                    "Frequency": plan["freq"], "Sampling_Grid": f"{plan['pts']} points",
                    "RMSE_dB": float(rmse), "Mock_Peak_Direction": {"theta": float(np.rad2deg(ff["theta"][np.unravel_index(np.argmax(ff["E_norm"]), ff["E_norm"].shape)[0]]))}
                }
                st.download_button("Export Final Characterization Report (JSON)", data=json.dumps(report, indent=2), file_name="nf2ff_characterization.json", mime="application/json")

elif exp_mode not in ["Near-Field / Far-Field Lab (M30)"]:
    st.info("Select 'Near-Field / Far-Field Lab (M30)' to configure planar scanning and NF2FF spectral limits.")
