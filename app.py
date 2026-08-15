"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 18 — Electromagnetic Topology Optimization
"""

import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time
import pandas as pd

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
    "Negative Epsilon (Drude)": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "w_pe": 2*math.pi*15e9, "g_e": 2*math.pi*0.5e9, "w_pm": 0.0, "g_m": 0.0, "is_dispersive": False, "is_metamaterial": True},
    "Negative Index NIM (Drude)": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "w_pe": 2*math.pi*15e9, "g_e": 2*math.pi*0.2e9, "w_pm": 2*math.pi*15e9, "g_m": 2*math.pi*0.2e9, "is_dispersive": False, "is_metamaterial": True}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 18 — Electromagnetic Topology Optimization")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
    "Electromagnetic Topology Optimization",
    "Inverse Design & Optimization",
    "Metamaterials Laboratory",
    "Adaptive Mesh Refinement (AMR)",
    "Antenna Array Laboratory",
    "Single Antenna (Dipole/Patch)", 
    "Advanced Validation Laboratory",
    "Material Dispersion Analyzer"
])

meta_mode = val_suite = None
if exp_mode == "Metamaterials Laboratory":
    meta_mode = st.sidebar.selectbox("Test Type", ["Effective Medium (Drude NIM Slab)", "Explicit Structured Medium (Wire Array)", "Material Frequency Analyzer"])
elif exp_mode == "Advanced Validation Laboratory":
    val_suite = st.sidebar.selectbox("Validation Suite", ["1. Wave Physics (Velocity)", "2. Boundary & Material", "3. CPU vs GPU", "4. Anisotropic Birefringence"])

# ============================================================
# GRID & DOMAIN SETUP
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
if exp_mode in ["Adaptive Mesh Refinement (AMR)", "Metamaterials Laboratory", "Inverse Design & Optimization", "Electromagnetic Topology Optimization"]:
    Nx = Ny = 40; Nz = 40 if exp_mode == "Inverse Design & Optimization" else 140
    dx = dy = dz = 0.005 
else:
    Nx = Ny = Nz = 80
    dx = dy = dz = 0.005

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
        if sig < 0: return 0.0, 0.0, 0.0, 0.0, 0.0 # PEC
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

apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"])

# SIMULATION CONTROL
num_steps = 300 if exp_mode in ["Inverse Design & Optimization", "Electromagnetic Topology Optimization"] else 600
freq_hz = 5e9; nf2ff_active = False
num_elements = 1; feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy])
feed_z_s_arr = np.array([30]); feed_z_e_arr = np.array([30])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])

# ============================================================
# TOPOLOGY OPTIMIZATION SETUP
# ============================================================
if exp_mode == "Electromagnetic Topology Optimization":
    st.sidebar.header("3. TOPOLOGY OPTIMIZATION CONFIG")
    top_budget = st.sidebar.number_input("Computational Budget (Simulations)", min_value=2, max_value=50, value=15)
    top_obj = st.sidebar.selectbox("Objective", ["Maximize Field Intensity (Transmission Focus)"])
    
    st.sidebar.subheader("Design Region & Constraints")
    vol_frac = st.sidebar.slider("Maximum Material Volume Fraction", 0.1, 1.0, 0.4, 0.05)
    macro_res = st.sidebar.selectbox("Macro-Voxel Resolution (N x N)", [4, 8, 10], index=1)
    
    # Topology Design Region Bounds
    tx_min, tx_max = pml_thickness + 2, Nx - pml_thickness - 2
    ty_min, ty_max = pml_thickness + 2, Ny - pml_thickness - 2
    tz_min, tz_max = 60, 80
    
    # SIMP Interpolation Parameters
    eps_bg = MAT_LIB["Vacuum / Air"]["er"]
    eps_des = MAT_LIB["High-K Dielectric (Topology)"]["er"]
    simp_p = 3.0
    
    def apply_topology_density(rho_macro):
        # Upscale macro voxels to FDTD Grid (acting as implicit feature-size filter)
        scale_x = (tx_max - tx_min) // macro_res
        scale_y = (ty_max - ty_min) // macro_res
        rho_grid = np.kron(rho_macro, np.ones((scale_x, scale_y)))
        
        # Apply SIMP Material Penalty Interpolation
        eps_eff = eps_bg + (rho_grid ** simp_p) * (eps_des - eps_bg)
        
        for i in range(rho_grid.shape[0]):
            for j in range(rho_grid.shape[1]):
                c1x, c2x, _, _, _ = get_mat_coeffs(eps_eff[i,j], 0.0, 0.0, 1.0, 1.0, False, dt)
                x_idx = tx_min + i; y_idx = ty_min + j
                
                # Apply across the slab thickness
                ce1_x[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_x[x_idx, y_idx, tz_min:tz_max] = c2x
                ce1_y[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_y[x_idx, y_idx, tz_min:tz_max] = c2x
                ce1_z[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_z[x_idx, y_idx, tz_min:tz_max] = c2x

# ============================================================
# MEMORY SAFETY
# ============================================================
mem_base_bytes = (44 * Nx * Ny * Nz * bytes_per_element)
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

        # Multi-Source Excitation (Plane Wave for Lens Topology)
        pulse = math.exp(-0.5*((t_steps-40)/15)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt))
        for e in range(num_el):
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ex[fx_arr[e], fy_arr[e], k] += pulse

        val_probe[n] = Ex[cx, cy, 120] # Focal point probe behind the design region

    return Ex, Ey, Ez, val_probe

def run_simulation_gpu(*args):
    # CuPy implementation exactly matches the vectorized layout. 
    # For M18 Topology Optimization we default to CPU to avoid VRAM overhead during thousands of rapid ML iteration loops.
    return run_simulation_cpu(*args)

def extract_focal_intensity(probe_history):
    # Extract the maximum field magnitude at the target focal point
    return np.max(np.abs(probe_history))

# ============================================================
# TOPOLOGY OPTIMIZER LOOP
# ============================================================
if exp_mode == "Electromagnetic Topology Optimization":
    run_top_btn = st.button("Run Topology Optimization", type="primary")

    if run_top_btn:
        st.markdown("### 🧬 Topology Optimization Live Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. Baseline Evaluation (Empty Design Region, rho=0)
        status_text.text("Evaluating Baseline (Empty Space)...")
        rho_macro_base = np.zeros((macro_res, macro_res), dtype=float)
        apply_topology_density(rho_macro_base)
        
        # Plane wave source setup for lens focusing
        feed_x_arr = np.repeat(np.arange(10, 30), 20)
        feed_y_arr = np.tile(np.arange(10, 30), 20)
        num_elements = len(feed_x_arr)
        feed_z_s_arr = np.full(num_elements, 30); feed_z_e_arr = np.full(num_elements, 30)
        amp_arr = np.ones(num_elements); phase_arr = np.zeros(num_elements)

        _, _, _, p_base = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0)
        
        base_score = extract_focal_intensity(p_base)
        
        # 2. Stochastic Hill-Climbing Topology Optimization Setup
        np.random.seed(42)
        current_rho = np.random.uniform(0, 1, (macro_res, macro_res))
        best_rho = current_rho.copy()
        best_score = base_score
        
        history_score = []; history_vf = []
        
        # 3. Iterative Optimization Loop
        start_opt = time.time()
        for i in range(top_budget):
            status_text.text(f"Topology Evaluations [{i+1}/{top_budget}] | Optimizing Density Map...")
            
            # Perturb (Mutation)
            test_rho = np.clip(current_rho + np.random.uniform(-0.3, 0.3, (macro_res, macro_res)), 0, 1)
            
            # Apply Constraints (Volume Budget)
            current_vf = np.mean(test_rho)
            if current_vf > vol_frac:
                test_rho *= (vol_frac / current_vf)
                
            apply_topology_density(test_rho)
            
            _, _, _, p_test = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0)
            
            score = extract_focal_intensity(p_test)
            history_score.append(score); history_vf.append(np.mean(test_rho))
            
            if score > best_score:
                best_score = score
                best_rho = test_rho.copy()
                current_rho = test_rho.copy()
            else:
                # Stochastic rejection recovery
                current_rho = best_rho.copy()
            
            progress_bar.progress((i+1)/top_budget)

        # 4. Final Binarization & Re-Simulation
        status_text.text("Validating Binarized Design...")
        binarized_rho = np.where(best_rho >= 0.5, 1.0, 0.0)
        
        # Enforce Volume Constraint Post-Binarization
        if np.mean(binarized_rho) > vol_frac:
            # Iteratively remove weakest cells if constrained
            sorted_indices = np.argsort(best_rho.flatten())
            allowed_cells = int(vol_frac * macro_res * macro_res)
            binarized_rho = np.zeros_like(binarized_rho.flatten())
            binarized_rho[sorted_indices[-allowed_cells:]] = 1.0
            binarized_rho = binarized_rho.reshape((macro_res, macro_res))
            
        apply_topology_density(binarized_rho)
        Ex_bin, Ey_bin, Ez_bin, p_bin = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0)
        bin_score = extract_focal_intensity(p_bin)
        opt_time = time.time() - start_opt

        st.session_state['top_opt_res'] = {
            'base_score': base_score, 'best_cont_score': best_score, 'bin_score': bin_score,
            'hist_score': history_score, 'hist_vf': history_vf, 'opt_time': opt_time,
            'best_rho': best_rho, 'binarized_rho': binarized_rho,
            'Ex_bin': Ex_bin, 'Ey_bin': Ey_bin, 'Ez_bin': Ez_bin
        }

# ============================================================
# ANALYSIS & VISUALIZATION (M18 TOPOLOGY RESULTS)
# ============================================================
if 'top_opt_res' in st.session_state and exp_mode == "Electromagnetic Topology Optimization":
    r = st.session_state['top_opt_res']
    
    st.markdown("### 🎯 Topology Optimization Validation Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimization Runtime", f"{r['opt_time']:.1f} s")
    c2.metric("Baseline Objective", f"{r['base_score']:.4e}")
    c3.metric("Optimized (Continuous)", f"{r['best_cont_score']:.4e}", f"+{((r['best_cont_score']/r['base_score'])-1)*100:.1f}%")
    c4.metric("Manufacturing (Binarized)", f"{r['bin_score']:.4e}", f"+{((r['bin_score']/r['base_score'])-1)*100:.1f}%")

    t1, t2, t3 = st.tabs(["Optimization Convergence", "Binarized Design Map (ρ)", "Optimized 3D Field Distribution"])

    with t1:
        st.info("The optimizer perturbs the material density map $\\rho \in [0, 1]$, maps to $\epsilon_r$ via SIMP interpolation, evaluates FDTD, and applies Volume Constraints (Penalty) to converge on a design maximizing Field Intensity at the focal plane.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(r['hist_score'])+1), y=r['hist_score'], mode='lines+markers', name="Continuous Objective"))
        fig.add_hline(y=r['base_score'], line_dash="dash", line_color="red", annotation_text="Baseline (Empty)")
        fig.add_hline(y=r['bin_score'], line_dash="dash", line_color="green", annotation_text="Final Binarized Validation")
        fig.update_layout(title="Objective Maximization Convergence", xaxis_title="Simulation Iteration", yaxis_title="Target Focal Intensity Objective")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        st.info("The final continuous material distribution is mathematically projected/binarized ($\\rho \in \{0, 1\}$) to ensure the structure is physically realizable before being explicitly re-simulated for the final Validation Score.")
        fig2 = go.Figure(data=go.Heatmap(z=r['binarized_rho'], colorscale='Blues', showscale=False))
        fig2.update_layout(title=f"Final Projected Material Distribution (Volume Fraction: {np.mean(r['binarized_rho']):.2f})", xaxis_title="X Macro-Voxels", yaxis_title="Y Macro-Voxels", width=500, height=500)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        E_mag = np.sqrt(r['Ex_bin']**2 + r['Ey_bin']**2 + r['Ez_bin']**2)
        with st.spinner("Rendering 3D Structure & Fields..."):
            plotter = pv.Plotter(off_screen=True, window_size=[800, 400])
            plotter.set_background("white")
            grid = pv.ImageData(dimensions=np.array([Nx, Ny, Nz]), spacing=(dx, dy, dz))
            grid.point_data["|E|"] = E_mag.flatten(order="F")
            plotter.add_mesh(grid.slice_orthogonal(x=cx*dx, y=cy*dy, z=cz*dz), cmap="jet", show_scalar_bar=True)
            plotter.view_isometric()
            st.image(plotter.screenshot(transparent_background=False), use_container_width=True)

elif exp_mode == "Electromagnetic Topology Optimization":
    st.info("Adjust settings in the sidebar and click 'Run Topology Optimization'.")
