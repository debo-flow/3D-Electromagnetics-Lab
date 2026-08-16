"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 20 — Multi-Objective Electromagnetic Optimization & Pareto Analysis
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
st.markdown("### Milestone 20 — Multi-Objective Electromagnetic Optimization & Pareto Analysis")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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

meta_mode = val_suite = adj_mode = None
if exp_mode == "Metamaterials Laboratory":
    meta_mode = st.sidebar.selectbox("Test Type", ["Effective Medium (Drude NIM Slab)", "Explicit Structured Medium (Wire Array)", "Material Frequency Analyzer"])
elif exp_mode == "Advanced Validation Laboratory":
    val_suite = st.sidebar.selectbox("Validation Suite", ["1. Wave Physics (Velocity)", "2. Boundary & Material", "3. CPU vs GPU", "4. Anisotropic Birefringence"])
elif exp_mode == "Adjoint Optimization & Sensitivity":
    adj_mode = st.sidebar.selectbox("Adjoint Workflow", ["1. Adjoint vs Finite-Difference Check", "2. Adjoint Gradient Descent (Topology)"])

# ============================================================
# GRID & DOMAIN SETUP
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
if exp_mode in ["Adaptive Mesh Refinement (AMR)", "Metamaterials Laboratory", "Inverse Design & Optimization", "Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity"]:
    Nx = Ny = 40; Nz = 40 if exp_mode in ["Inverse Design & Optimization", "Multi-Objective Pareto Optimization"] else 140
    dx = dy = dz = 0.005 
elif exp_mode == "Multi-Objective Pareto Optimization":
    Nx = 50; Ny = 80; Nz = 40; dx = dy = dz = 0.005 # Wider for Array processing
else:
    Nx = Ny = Nz = 80; dx = dy = dz = 0.005

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
pml_thickness = 10; dt = 0.9 * (1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)))

# Unified Material Arrays
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
num_steps = 300 if exp_mode in ["Inverse Design & Optimization", "Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity", "Multi-Objective Pareto Optimization"] else 600
freq_hz = 5e9; nf2ff_active = False
num_elements = 1; feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy])
feed_z_s_arr = np.array([30]); feed_z_e_arr = np.array([30])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])
i_min = j_min = k_min = pml_thickness + 2
i_max = Nx - 1 - pml_thickness - 2; j_max = Ny - 1 - pml_thickness - 2; k_max = Nz - 1 - pml_thickness - 2

# ============================================================
# MULTI-OBJECTIVE NSGA-II CONFIGURATION
# ============================================================
if exp_mode == "Multi-Objective Pareto Optimization":
    st.sidebar.header("3. MULTI-OBJECTIVE CONFIG")
    st.sidebar.markdown("**Algorithm:** `NSGA-II`")
    pop_size = st.sidebar.number_input("Population Size", min_value=4, max_value=20, value=6, step=2)
    max_gen = st.sidebar.number_input("Maximum Generations", min_value=2, max_value=10, value=3)
    
    st.sidebar.subheader("Objectives")
    st.sidebar.markdown("1. **Maximize** Main-Lobe Directivity")
    st.sidebar.markdown("2. **Minimize** Peak Sidelobe Level (SLL)")
    target_angle = st.sidebar.slider("Target Beam Angle (H-Plane φ°)", -90, 90, 45, 5)
    
    st.sidebar.subheader("Design Variables (2-Element Array)")
    st.sidebar.markdown("Var 1: **Element Spacing** ($0.25\lambda - 1.0\lambda$)")
    st.sidebar.markdown("Var 2: **Progressive Phase** ($-180^\circ - 180^\circ$)")
    
    num_elements = 2
    freq_hz = 2.4e9; wavelength = C_LIGHT / freq_hz
    nf2ff_active = True

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
if nf2ff_active: mem_base_bytes += (5 * Nx * Ny * bytes_per_element)
if exp_mode == "Adjoint Optimization & Sensitivity":
    mem_base_bytes += ((Nx-20) * (Ny-20) * (20) * num_steps * bytes_per_element * 3)

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

        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k-kmin, 0, n] = Ey[i, j, k]; sx_E[f, j-jmin, k-kmin, 1, n] = Ez[i, j, k]

    return Ex, Ey, Ez, val_probe, sx_E

# ============================================================
# NSGA-II MULTI-OBJECTIVE FUNCTIONS
# ============================================================
def extract_array_objectives(sx_E, freq, t_rad):
    # Calculates Objective 1 (Gain/Directivity at Target) and Objective 2 (Peak Sidelobe Level)
    k = 2.0 * np.pi * freq / C_LIGHT
    phi_1d = np.deg2rad(np.arange(-90, 90 + 2, 2))
    E_pattern = np.zeros(len(phi_1d), dtype=float)
    
    window = np.ones(num_steps); freqs = np.fft.rfftfreq(num_steps, d=dt); bin_idx = np.argmin(np.abs(freqs - freq))
    px_E = np.fft.rfft(sx_E * window, axis=-1)[..., bin_idx] * (2.0 / num_steps)
    
    for a, p_val in enumerate(phi_1d):
        rx = np.sin(math.pi/2) * np.cos(p_val); ry = np.sin(math.pi/2) * np.sin(p_val); rz = 0.0
        L_theta = 0j; N_phi = 0j
        for f in range(2):
            nx = -1.0 if f == 0 else 1.0; x_prime = (i_min if f==0 else i_max) - cx; dS = dy * dz
            for j in range(j_min, j_max+1):
                y_prime = j - cy
                for k_idx in range(k_min, k_max+1):
                    z_prime = k_idx - cz
                    exp_phase = np.exp(1j * k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz))
                    Ey_val = px_E[f, j-j_min, k_idx-k_min, 0]; Ez_val = px_E[f, j-j_min, k_idx-k_min, 1]
                    L_theta += (nx * Ey_val) * exp_phase * dS; N_phi += (-nx * Ez_val) * exp_phase * dS
        E_pattern[a] = np.abs(L_theta) + np.abs(N_phi)
    
    # Obj 1: Maximize Main Lobe Directivity at Target Angle
    target_idx = np.argmin(np.abs(phi_1d - t_rad))
    obj_gain = E_pattern[target_idx]
    
    # Obj 2: Minimize Peak Sidelobe Level (SLL)
    # Mask out the main beam (+/- 20 degrees from peak)
    peak_idx = np.argmax(E_pattern)
    mask_start = max(0, peak_idx - 10)
    mask_end = min(len(E_pattern), peak_idx + 10)
    masked_pattern = np.copy(E_pattern)
    masked_pattern[mask_start:mask_end] = 0.0
    
    obj_sll = np.max(masked_pattern) if np.max(masked_pattern) > 0 else 1e-6
    sll_db = 20 * np.log10(obj_sll / np.max(E_pattern)) # Relative dB
    
    # NSGA-II conventionally minimizes all objectives internally.
    # Return: (-Gain, SLL_dB)
    return -obj_gain, sll_db, E_pattern

def dominates(obj1, obj2):
    return (obj1[0] <= obj2[0] and obj1[1] <= obj2[1]) and (obj1[0] < obj2[0] or obj1[1] < obj2[1])

def fast_non_dominated_sort(objectives):
    N = len(objectives)
    fronts = [[]]; S = [[] for _ in range(N)]; n = np.zeros(N, dtype=int)
    for p in range(N):
        for q in range(N):
            if dominates(objectives[p], objectives[q]): S[p].append(q)
            elif dominates(objectives[q], objectives[p]): n[p] += 1
        if n[p] == 0: fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0: Q.append(q)
        i += 1
        if len(Q) > 0: fronts.append(Q)
    return fronts[:-1]

def crowding_distance_assignment(front, objectives):
    l = len(front)
    distances = np.zeros(l)
    if l == 0: return distances
    if l <= 2:
        distances[:] = float('inf')
        return distances
    for m in range(2):
        sorted_indices = np.argsort([objectives[i][m] for i in front])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        obj_min = objectives[front[sorted_indices[0]]][m]
        obj_max = objectives[front[sorted_indices[-1]]][m]
        norm = obj_max - obj_min if obj_max > obj_min else 1.0
        for i in range(1, l - 1):
            distances[sorted_indices[i]] += (objectives[front[sorted_indices[i+1]]][m] - objectives[front[sorted_indices[i-1]]][m]) / norm
    return distances

def nsga2_generate_offspring(pop, bounds):
    offspring = []
    for _ in range(len(pop)):
        p1, p2 = random.sample(pop, 2)
        # Crossover
        c1 = [(p1[i] + p2[i])/2 for i in range(len(p1))]
        # Mutation
        for i in range(len(c1)):
            if random.random() < 0.3:
                c1[i] += random.uniform(-0.1, 0.1) * (bounds[i][1] - bounds[i][0])
            c1[i] = np.clip(c1[i], bounds[i][0], bounds[i][1])
        offspring.append(c1)
    return offspring

# ============================================================
# MULTI-OBJECTIVE OPTIMIZER LOOP
# ============================================================
if exp_mode == "Multi-Objective Pareto Optimization":
    run_moo_btn = st.button("Start Multi-Objective Pareto Optimization", type="primary")

    if run_moo_btn:
        st.markdown("### 🧬 Pareto Optimization Live Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        target_rad = math.radians(target_angle)
        bounds = [(0.25, 1.0), (-180.0, 180.0)] # Spacing (lambda), Phase (deg)
        
        # Initialize Population
        population = [[random.uniform(b[0], b[1]) for b in bounds] for _ in range(pop_size)]
        
        # Include baseline [0.5 lambda, 0 deg]
        population[0] = [0.5, 0.0]
        
        all_evals = [] # Store unique evaluations to avoid FDTD repeats
        eval_cache = {}

        start_opt = time.time()
        for gen in range(max_gen):
            status_text.text(f"Generation [{gen+1}/{max_gen}] | Evaluating Candidates via FDTD...")
            
            pop_objs = []
            for i, ind in enumerate(population):
                key = (round(ind[0], 3), round(ind[1], 1))
                if key not in eval_cache:
                    reset_materials()
                    spacing_cells = int((ind[0] * wavelength) / dy)
                    f_y_arr = np.array([cy - spacing_cells//2, cy + spacing_cells//2])
                    p_arr = np.array([0.0, math.radians(ind[1])])
                    
                    for n in range(2):
                        apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz - 5, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
                        apply_material_block(cx, cx, f_y_arr[n], f_y_arr[n], cz + 1, cz + 5, MAT_LIB["PEC (Perfect Conductor)"])
                        
                    _, _, _, _, sx_E = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, 2, np.array([cx, cx]), f_y_arr, np.array([cz, cz]), np.array([cz, cz]), np.array([1.0, 1.0]), p_arr, freq_hz, True, i_min, i_max, j_min, j_max, k_min, k_max)
                    
                    obj1, obj2, pat = extract_array_objectives(sx_E, freq_hz, target_rad)
                    eval_cache[key] = (obj1, obj2, pat)
                    all_evals.append({"Spacing (L)": key[0], "Phase (deg)": key[1], "Gain_Raw": -obj1, "SLL (dB)": obj2})
                
                pop_objs.append((eval_cache[key][0], eval_cache[key][1]))
                progress_bar.progress(((gen * pop_size) + i + 1) / (max_gen * pop_size))
            
            # NSGA-II Sorting & Selection
            fronts = fast_non_dominated_sort(pop_objs)
            new_pop = []
            for front in fronts:
                distances = crowding_distance_assignment(front, pop_objs)
                front_sorted = [front[i] for i in np.argsort(-distances)] # descending diversity
                for idx in front_sorted:
                    if len(new_pop) < pop_size:
                        new_pop.append(population[idx])
            
            # Generate next population
            if gen < max_gen - 1:
                offspring = nsga2_generate_offspring(new_pop, bounds)
                population = new_pop + offspring # Allow elite retention next gen
            else:
                population = new_pop

        opt_time = time.time() - start_opt
        
        # Extract Final Pareto Front (Front 0)
        final_objs = []
        for ind in population:
            key = (round(ind[0], 3), round(ind[1], 1))
            final_objs.append((eval_cache[key][0], eval_cache[key][1]))
            
        final_fronts = fast_non_dominated_sort(final_objs)
        pareto_front = [population[i] for i in final_fronts[0]]
        pareto_objs = [final_objs[i] for i in final_fronts[0]]
        
        # Estimate Hypervolume (2D Area wrt Nadir point)
        nadir = [np.max([o[0] for o in pareto_objs]), np.max([o[1] for o in pareto_objs])]
        sorted_pareto = sorted(pareto_objs, key=lambda x: x[0])
        hv = 0.0
        for i in range(len(sorted_pareto)):
            width = nadir[0] - sorted_pareto[i][0]
            height = nadir[1] - sorted_pareto[i][1] if i == 0 else sorted_pareto[i-1][1] - sorted_pareto[i][1]
            hv += width * height
            
        # Baseline Data
        base_key = (0.5, 0.0)
        base_obj1, base_obj2, _ = eval_cache[base_key]

        st.session_state['moo_res'] = {
            'pareto_front': pareto_front, 'pareto_objs': pareto_objs, 'hv': hv, 'opt_time': opt_time,
            'all_evals': pd.DataFrame(all_evals), 'base_objs': (base_obj1, base_obj2), 'eval_cache': eval_cache
        }

# ============================================================
# ANALYSIS & VISUALIZATION (M20 MULTI-OBJECTIVE RESULTS)
# ============================================================
if 'moo_res' in st.session_state and exp_mode == "Multi-Objective Pareto Optimization":
    r = st.session_state['moo_res']
    
    st.markdown("### 🎯 Multi-Objective Pareto Validation Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimization Runtime", f"{r['opt_time']:.1f} s")
    c2.metric("Total Unique FDTD Evaluations", f"{len(r['all_evals'])}")
    c3.metric("Non-Dominated Solutions Found", f"{len(r['pareto_front'])}")
    c4.metric("Estimated Hypervolume Diversity", f"{r['hv']:.2e}")

    t1, t2, t3 = st.tabs(["2D Pareto Front (Trade-off)", "Design Candidates Table", "Select Compromise Solution"])

    with t1:
        st.info("Scatter plot showing the non-dominated Pareto front. The NSGA-II algorithm explicitly maximized Target-Angle Gain while simultaneously minimizing Peak Sidelobe Levels (SLL). The ideal unattainable point sits at the bottom-right corner.")
        gains = [-o[0] for o in r['pareto_objs']] # Convert back to positive Gain
        slls = [o[1] for o in r['pareto_objs']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gains, y=slls, mode='markers+lines', name='Estimated Pareto Front', marker=dict(size=10, color='blue')))
        fig.add_trace(go.Scatter(x=[-r['base_objs'][0]], y=[r['base_objs'][1]], mode='markers', name='Baseline (0.5λ, 0°)', marker=dict(size=12, color='red', symbol='x')))
        fig.update_layout(title="Gain vs Peak Sidelobe Level (Trade-off Analysis)", xaxis_title="Objective 1: Maximize Main Lobe Gain (Raw E-Field Magnitude)", yaxis_title="Objective 2: Minimize Sidelobe Level (dB relative to Main)")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        st.dataframe(r['all_evals'].drop_duplicates().sort_values(by="Gain_Raw", ascending=False), use_container_width=True)

    with t3:
        st.markdown("#### 🔬 Decision Making & Re-Simulation")
        selected_idx = st.selectbox("Select a Pareto Solution to Verify:", range(len(r['pareto_front'])), format_func=lambda i: f"Gain: {-r['pareto_objs'][i][0]:.3f} | SLL: {r['pareto_objs'][i][1]:.1f} dB  (Spacing: {r['pareto_front'][i][0]:.2f}λ, Phase: {r['pareto_front'][i][1]:.1f}°)")
        
        if st.button("Validate Selected Design (Re-Simulate)"):
            ind = r['pareto_front'][selected_idx]
            key = (round(ind[0], 3), round(ind[1], 1))
            pat = r['eval_cache'][key][2]
            
            fig_p = go.Figure(go.Scatterpolar(r=pat/np.max(pat), theta=np.arange(-90, 92, 2), mode='lines', line_color='green'))
            fig_p.update_layout(title=f"Normalized H-Plane Pattern Validation (Target: {target_angle}°)", polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(fig_p, use_container_width=True)

elif exp_mode not in ["Multi-Objective Pareto Optimization"]:
    st.info("Select 'Multi-Objective Pareto Optimization' mode to run NSGA-II frameworks.")
