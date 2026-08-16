"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 22 — Uncertainty Quantification & Robust Electromagnetic Design
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
st.markdown("### Milestone 22 — Uncertainty Quantification & Robust Design")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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

meta_mode = val_suite = adj_mode = surr_mode = None
if exp_mode == "Metamaterials Laboratory":
    meta_mode = st.sidebar.selectbox("Test Type", ["Effective Medium (Drude NIM Slab)", "Explicit Structured Medium (Wire Array)", "Material Frequency Analyzer"])
elif exp_mode == "Advanced Validation Laboratory":
    val_suite = st.sidebar.selectbox("Validation Suite", ["1. Wave Physics (Velocity)", "2. Boundary & Material", "3. CPU vs GPU", "4. Anisotropic Birefringence"])
elif exp_mode == "Adjoint Optimization & Sensitivity":
    adj_mode = st.sidebar.selectbox("Adjoint Workflow", ["1. Adjoint vs Finite-Difference Check", "2. Adjoint Gradient Descent (Topology)"])
elif exp_mode == "Surrogate & Reduced-Order Modeling":
    surr_mode = st.sidebar.selectbox("Surrogate / ROM Workflow", ["1. Dataset Generation & Surrogate Training", "2. Surrogate-Assisted Optimization", "3. Reduced-Order Modeling (POD/PCA)"])

# ============================================================
# GRID & DOMAIN SETUP
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
if exp_mode in ["Adaptive Mesh Refinement (AMR)", "Metamaterials Laboratory", "Inverse Design & Optimization", "Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity", "Surrogate & Reduced-Order Modeling", "Uncertainty Quantification (UQ)"]:
    Nx = Ny = 40; Nz = 40 if exp_mode in ["Inverse Design & Optimization", "Multi-Objective Pareto Optimization", "Surrogate & Reduced-Order Modeling", "Uncertainty Quantification (UQ)"] else 140
    dx = dy = dz = 0.005 
elif exp_mode == "Multi-Objective Pareto Optimization":
    Nx = 50; Ny = 80; Nz = 40; dx = dy = dz = 0.005
else:
    Nx = Ny = Nz = 80; dx = dy = dz = 0.005

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
pml_thickness = 10; dt = 0.9 * (1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)))

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

def apply_material_block(x1, x2, y1, y2, z1, z2, mat):
    sig = mat.get("sigma", 0.0); mur = mat.get("mur", 1.0); is_disp = mat.get("is_dispersive", False)
    er_x = mat.get("er_x", mat.get("er", 1.0)); er_y = mat.get("er_y", mat.get("er", 1.0)); er_z = mat.get("er_z", mat.get("er", 1.0))
    c1x, c2x, c3x, p1x, p2x = get_mat_coeffs(er_x, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, dt)
    ce1_x[x1:x2+1, y1:y2+1, z1:z2+1] = c1x; ce2_x[x1:x2+1, y1:y2+1, z1:z2+1] = c2x; ce3_x[x1:x2+1, y1:y2+1, z1:z2+1] = c3x; cp1_x[x1:x2+1, y1:y2+1, z1:z2+1] = p1x; cp2_x[x1:x2+1, y1:y2+1, z1:z2+1] = p2x
    c1y, c2y, c3y, p1y, p2y = get_mat_coeffs(er_y, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, dt)
    ce1_y[x1:x2+1, y1:y2+1, z1:z2+1] = c1y; ce2_y[x1:x2+1, y1:y2+1, z1:z2+1] = c2y; ce3_y[x1:x2+1, y1:y2+1, z1:z2+1] = c3y; cp1_y[x1:x2+1, y1:y2+1, z1:z2+1] = p1y; cp2_y[x1:x2+1, y1:y2+1, z1:z2+1] = p2y
    c1z, c2z, c3z, p1z, p2z = get_mat_coeffs(er_z, sig, mat.get("tau",0.0), mat.get("er_s",1.0), mat.get("er_inf",1.0), is_disp, dt)
    ce1_z[x1:x2+1, y1:y2+1, z1:z2+1] = c1z; ce2_z[x1:x2+1, y1:y2+1, z1:z2+1] = c2z; ce3_z[x1:x2+1, y1:y2+1, z1:z2+1] = c3z; cp1_z[x1:x2+1, y1:y2+1, z1:z2+1] = p1z; cp2_z[x1:x2+1, y1:y2+1, z1:z2+1] = p2z
    ch2[x1:x2+1, y1:y2+1, z1:z2+1] = dt / (mur * MU_0)
    if mat.get("is_metamaterial", False):
        w_pe = mat["w_pe"]; g_e = mat["g_e"]; w_pm = mat["w_pm"]; g_m = mat["g_m"]
        cd1_e[x1:x2+1, y1:y2+1, z1:z2+1] = (1 - g_e * dt / 2) / (1 + g_e * dt / 2); cd2_e[x1:x2+1, y1:y2+1, z1:z2+1] = (EPS_0 * w_pe**2 * dt) / (1 + g_e * dt / 2)
        cd1_m[x1:x2+1, y1:y2+1, z1:z2+1] = (1 - g_m * dt / 2) / (1 + g_m * dt / 2); cd2_m[x1:x2+1, y1:y2+1, z1:z2+1] = (MU_0 * w_pm**2 * dt) / (1 + g_m * dt / 2)

def reset_materials():
    ce1_x.fill(1.0); ce2_x.fill(0.0); ce3_x.fill(0.0); cp1_x.fill(0.0); cp2_x.fill(0.0)
    ce1_y.fill(1.0); ce2_y.fill(0.0); ce3_y.fill(0.0); cp1_y.fill(0.0); cp2_y.fill(0.0)
    ce1_z.fill(1.0); ce2_z.fill(0.0); ce3_z.fill(0.0); cp1_z.fill(0.0); cp2_z.fill(0.0)
    ch2.fill(0.0); cd1_e.fill(0.0); cd2_e.fill(0.0); cd1_m.fill(0.0); cd2_m.fill(0.0)
    apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"])

reset_materials()

# Variables
num_steps = 300 if exp_mode in ["Inverse Design & Optimization", "Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity", "Multi-Objective Pareto Optimization", "Surrogate & Reduced-Order Modeling", "Uncertainty Quantification (UQ)"] else 600
freq_hz = 2.4e9 if exp_mode in ["Surrogate & Reduced-Order Modeling", "Uncertainty Quantification (UQ)"] else 5e9
nf2ff_active = False; num_elements = 1
feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy]); feed_z_s_arr = np.array([30]); feed_z_e_arr = np.array([30])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])
i_min = j_min = k_min = pml_thickness + 2
i_max = Nx - 1 - pml_thickness - 2; j_max = Ny - 1 - pml_thickness - 2; k_max = Nz - 1 - pml_thickness - 2

# ============================================================
# SURROGATE MATH HELPERS (M21/M22 INTEGRATION)
# ============================================================
def poly_features_2d(X):
    N = X.shape[0]; out = np.ones((N, 6))
    out[:, 1] = X[:, 0]; out[:, 2] = X[:, 1]
    out[:, 3] = X[:, 0]**2; out[:, 4] = X[:, 1]**2
    out[:, 5] = X[:, 0] * X[:, 1]
    return out

def ridge_fit(X, y, alpha=1e-3):
    return np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

# ============================================================
# UQ & ROBUST DESIGN CONFIGURATION
# ============================================================
if exp_mode == "Uncertainty Quantification (UQ)":
    st.sidebar.header("3. UQ & ROBUSTNESS CONFIG")
    num_mc_samples = st.sidebar.number_input("Monte Carlo Samples", min_value=100, max_value=50000, value=10000, step=1000)
    target_angle = st.sidebar.slider("Target Beam Angle (H-Plane φ°)", -90, 90, 45, 5)
    target_gain_threshold = st.sidebar.number_input("Gain Yield Threshold", value=0.6)
    
    st.sidebar.subheader("Uncertain Parameters")
    st.sidebar.markdown("**Param 1: Array Spacing (λ)**")
    u1_nom = st.sidebar.number_input("P1 Nominal", value=0.5)
    u1_std = st.sidebar.number_input("P1 Std Dev (σ)", value=0.05, format="%.3f")
    
    st.sidebar.markdown("**Param 2: Array Phase (°)**")
    u2_nom = st.sidebar.number_input("P2 Nominal", value=45.0)
    u2_std = st.sidebar.number_input("P2 Std Dev (σ)", value=10.0, format="%.2f")
    
    num_elements = 2
    wavelength = C_LIGHT / freq_hz
    nf2ff_active = True

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

def compute_cpml(N, d_pml, delta, dt, m=3, R_err=1e-4, alpha_max=0.05):
    b_e = np.zeros(N, dtype=dtype_np); c_e = np.zeros(N, dtype=dtype_np); b_h = np.zeros(N, dtype=dtype_np); c_h = np.zeros(N, dtype=dtype_np)
    sigma_max = - (m + 1) * math.log(R_err) / (2.0 * Z_0 * (d_pml * delta)) if d_pml > 0 else 0
    for i in range(N):
        if d_pml == 0: continue
        d_e = (d_pml - i)*delta if i < d_pml else (i - (N - 1 - d_pml))*delta if i > N - 1 - d_pml else 0.0
        d_h = (d_pml - i - 0.5)*delta if i < d_pml else (i + 0.5 - (N - 1 - d_pml))*delta if i > N - 2 - d_pml else 0.0
        d_h = max(0.0, d_h)
        if d_e > 0:
            s_e = sigma_max * (d_e / (d_pml * delta))**m; a_e = alpha_max * (1.0 - d_e / (d_pml * delta))**m
            b_e[i] = math.exp(-(s_e + a_e * EPS_0 / dt) * (dt / EPS_0)); c_e[i] = s_e / (s_e + a_e * EPS_0 / dt) * (b_e[i] - 1.0) / delta
        if d_h > 0:
            s_h = sigma_max * (d_h / (d_pml * delta))**m; a_h = alpha_max * (1.0 - d_h / (d_pml * delta))**m
            b_h[i] = math.exp(-(s_h + a_h * EPS_0 / dt) * (dt / EPS_0)); c_h[i] = s_h / (s_h + a_h * EPS_0 / dt) * (b_h[i] - 1.0) / delta
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

        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k-kmin, 0, n] = Ey[i, j, k]; sx_E[f, j-jmin, k-kmin, 1, n] = Ez[i, j, k]
    return Ex, Ey, Ez, sx_E

def extract_target_gain(sx_E, freq, t_rad):
    k = 2.0 * np.pi * freq / C_LIGHT
    phi_1d = np.deg2rad(np.arange(-90, 90 + 2, 2)); E_pattern = np.zeros(len(phi_1d), dtype=float)
    window = np.ones(num_steps); freqs = np.fft.rfftfreq(num_steps, d=dt); bin_idx = np.argmin(np.abs(freqs - freq))
    px_E = np.fft.rfft(sx_E * window, axis=-1)[..., bin_idx] * (2.0 / num_steps)
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
# UNCERTAINTY QUANTIFICATION (UQ) & SOBOL SENSITIVITY
# ============================================================
if exp_mode == "Uncertainty Quantification (UQ)":
    run_uq_btn = st.button("Run Surrogate-Assisted UQ Pipeline", type="primary")

    if run_uq_btn:
        st.markdown("### 🧬 Robust Design & Uncertainty Quantification Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        target_rad = math.radians(target_angle)
        
        # 1. Dataset Generation & Surrogate Training (Self-Contained for UQ)
        status_text.text("Generating Full-Wave Baseline Dataset for UQ Surrogate...")
        num_train = 20
        np.random.seed(42)
        X_train = np.zeros((num_train, 2))
        X_train[:, 0] = np.random.uniform(0.2, 1.0, num_train)
        X_train[:, 1] = np.random.uniform(-180, 180, num_train)
        y_train = np.zeros(num_train)
        
        train_start = time.time()
        for i in range(num_train):
            reset_materials()
            spacing_cells = int((X_train[i, 0] * wavelength) / dy)
            f_y_arr = np.array([cy - spacing_cells//2, cy + spacing_cells//2])
            p_arr = np.array([0.0, math.radians(X_train[i, 1])])
            for n in range(2):
                apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz - 5, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz + 1, cz + 5, MAT_LIB["PEC (Perfect Conductor)"])
            
            _, _, _, sx_E = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 2, np.array([cx, cx]), f_y_arr, np.array([cz, cz]), np.array([cz, cz]), np.array([1.0, 1.0]), p_arr, freq_hz, True, i_min, i_max, j_min, j_max, k_min, k_max)
            y_train[i] = extract_target_gain(sx_E, freq_hz, target_rad)
            progress_bar.progress(0.2 * (i+1)/num_train)
            
        # Surrogate Fitting
        X_min, X_max = np.min(X_train, axis=0), np.max(X_train, axis=0)
        X_scaled = (X_train - X_min) / (X_max - X_min + 1e-12)
        X_poly = poly_features_2d(X_scaled)
        weights = ridge_fit(X_poly, y_train, alpha=1e-3)
        train_time = time.time() - train_start
        
        # 2. Monte Carlo Sampling on Surrogate
        status_text.text(f"Running Monte Carlo Sampling ({num_mc_samples} runs) via Surrogate...")
        mc_start = time.time()
        
        # Generate Normal Distributions based on UI uncertainties
        A_samples = np.zeros((num_mc_samples, 2))
        A_samples[:, 0] = np.random.normal(u1_nom, u1_std, num_mc_samples)
        A_samples[:, 1] = np.random.normal(u2_nom, u2_std, num_mc_samples)
        A_samples[:, 0] = np.clip(A_samples[:, 0], 0.1, 2.0)
        A_samples[:, 1] = np.clip(A_samples[:, 1], -180.0, 180.0)
        
        # Predict Base MC
        A_scaled = (A_samples - X_min) / (X_max - X_min + 1e-12)
        y_mc = poly_features_2d(A_scaled) @ weights
        
        # 3. Sobol Sensitivity Analysis (Saltelli Method)
        status_text.text("Calculating Sobol Sensitivity Indices (Global)...")
        B_samples = np.zeros((num_mc_samples, 2))
        B_samples[:, 0] = np.random.normal(u1_nom, u1_std, num_mc_samples)
        B_samples[:, 1] = np.random.normal(u2_nom, u2_std, num_mc_samples)
        B_samples[:, 0] = np.clip(B_samples[:, 0], 0.1, 2.0)
        B_samples[:, 1] = np.clip(B_samples[:, 1], -180.0, 180.0)
        B_scaled = (B_samples - X_min) / (X_max - X_min + 1e-12)
        y_B = poly_features_2d(B_scaled) @ weights
        
        var_Y = np.var(np.concatenate([y_mc, y_B]))
        S_i = np.zeros(2); ST_i = np.zeros(2)
        
        for p_idx in range(2):
            AB_i = A_scaled.copy(); AB_i[:, p_idx] = B_scaled[:, p_idx]
            y_AB_i = poly_features_2d(AB_i) @ weights
            S_i[p_idx] = (np.mean(y_mc * y_AB_i) - np.mean(y_mc)**2) / var_Y
            ST_i[p_idx] = (1.0 / (2 * num_mc_samples * var_Y)) * np.sum((y_mc - y_AB_i)**2)
            
        mc_time = time.time() - mc_start
        
        # 4. Statistical Outputs & Yield
        y_mean = np.mean(y_mc); y_std = np.std(y_mc)
        y_p5, y_median, y_p95 = np.percentile(y_mc, [5, 50, 95])
        yield_pct = np.sum(y_mc >= target_gain_threshold) / num_mc_samples * 100.0
        
        worst_idx = np.argmin(y_mc)
        worst_X = A_samples[worst_idx]
        worst_pred = y_mc[worst_idx]
        
        # 5. Full-Wave Validation of Nominal and Worst-Case
        status_text.text("Executing Full-Wave Validations on Extremes...")
        val_start = time.time()
        
        def run_fw(spacing, phase):
            reset_materials()
            spacing_cells = int((spacing * wavelength) / dy)
            f_y_arr = np.array([cy - spacing_cells//2, cy + spacing_cells//2])
            p_arr = np.array([0.0, math.radians(phase)])
            for n in range(2):
                apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz - 5, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz + 1, cz + 5, MAT_LIB["PEC (Perfect Conductor)"])
            _, _, _, sx_E = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 2, np.array([cx, cx]), f_y_arr, np.array([cz, cz]), np.array([cz, cz]), np.array([1.0, 1.0]), p_arr, freq_hz, True, i_min, i_max, j_min, j_max, k_min, k_max)
            return extract_target_gain(sx_E, freq_hz, target_rad)
            
        fw_nom = run_fw(u1_nom, u2_nom)
        fw_worst = run_fw(worst_X[0], worst_X[1])
        val_time = time.time() - val_start
        
        progress_bar.progress(1.0)
        status_text.text("Uncertainty Quantification Complete.")
        
        st.session_state['uq_res'] = {
            'y_mc': y_mc, 'y_mean': y_mean, 'y_std': y_std, 'y_p5': y_p5, 'y_median': y_median, 'y_p95': y_p95,
            'yield_pct': yield_pct, 'S_i': S_i, 'ST_i': ST_i, 'worst_X': worst_X, 'worst_pred': worst_pred,
            'fw_nom': fw_nom, 'fw_worst': fw_worst, 'mc_time': mc_time, 'train_time': train_time
        }

# ============================================================
# ANALYSIS & VISUALIZATION (M22 UQ RESULTS)
# ============================================================
if 'uq_res' in st.session_state and exp_mode == "Uncertainty Quantification (UQ)":
    r = st.session_state['uq_res']
    
    st.markdown("### 🎯 Robust Design & Uncertainty Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MC Sample Size", f"{num_mc_samples}")
    c2.metric("Mean Target Gain", f"{r['y_mean']:.4f}", f"± {r['y_std']:.4f} σ", delta_color="off")
    c3.metric("Estimated Manufacturing Yield", f"{r['yield_pct']:.1f}%", f"Threshold: {target_gain_threshold}", delta_color="normal")
    c4.metric("Worst Sampled Case", f"{r['worst_pred']:.4f}", f"Space: {r['worst_X'][0]:.2f}λ, {r['worst_X'][1]:.1f}°", delta_color="inverse")

    t1, t2, t3 = st.tabs(["Statistical Distributions", "Global Sobol Sensitivity", "Full-Wave Validation Checks"])

    with t1:
        st.info("The Monte Carlo simulation isolates standard manufacturing/hardware uncertainty variances against final electromagnetic efficiency goals using thousands of sub-millisecond surrogate evaluations.")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=r['y_mc'], nbinsx=50, marker_color='blue', name="Gain Distribution"))
        fig.add_vline(x=r['y_p5'], line_dash="dash", line_color="red", annotation_text="5th Percentile")
        fig.add_vline(x=r['y_p95'], line_dash="dash", line_color="green", annotation_text="95th Percentile")
        fig.add_vline(x=target_gain_threshold, line_dash="solid", line_color="black", annotation_text="Yield Threshold")
        fig.update_layout(title="Electromagnetic Target Gain (Manufacturing Distribution)", xaxis_title="Array Gain (Linear)", yaxis_title="Sample Count")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.info("Saltelli variance-based global sensitivity analysis computes the fractional contribution of each uncertain physical parameter. $S_i$ evaluates isolated effects, while $S_{T_i}$ evaluates combined non-linear interaction bounds.")
        fig2 = go.Figure(data=[
            go.Bar(name='First-Order Index (S_i)', x=['Array Spacing (λ)', 'Array Phase (°)'], y=r['S_i']),
            go.Bar(name='Total-Order Index (S_T_i)', x=['Array Spacing (λ)', 'Array Phase (°)'], y=r['ST_i'])
        ])
        fig2.update_layout(title="Sobol Sensitivity Indices", barmode='group', yaxis_title="Variance Contribution Ratio", yaxis_range=[0, 1.1])
        st.plotly_chart(fig2, use_container_width=True)
        
    with t3:
        st.markdown("#### 🔬 FDTD Verification of Statistical Extremes")
        st.markdown("To ensure scientific honesty, the surrogate's projected Nominal and Worst-Case bounding extremes are re-solved using isolated explicit Full-Wave FDTD configurations to certify that surrogate predictive variance has not drifted from Maxwell's limits.")
        
        val_data = {
            "Design Case": ["Nominal Parameter State", "Worst Sampled Defect State"],
            "Spacing (λ)": [f"{u1_nom:.2f}", f"{r['worst_X'][0]:.2f}"],
            "Phase (°)": [f"{u2_nom:.1f}", f"{r['worst_X'][1]:.1f}"],
            "Surrogate Prediction": [f"N/A (Anchor)", f"{r['worst_pred']:.4f}"],
            "Full-Wave FDTD Validated": [f"{r['fw_nom']:.4f}", f"{r['fw_worst']:.4f}"]
        }
        st.table(pd.DataFrame(val_data))
        st.write(f"**Surrogate Boundary Relative Error:** `{abs(r['worst_pred'] - r['fw_worst'])/r['fw_worst']*100:.2f}%`")
