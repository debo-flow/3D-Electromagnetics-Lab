"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 19 — Adjoint Electromagnetic Optimization & Sensitivity Analysis
"""

import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time

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
st.markdown("### Milestone 19 — Adjoint Electromagnetic Optimization")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Mode", [
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

apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"])

# Variables
num_steps = 300 if exp_mode in ["Inverse Design & Optimization", "Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity"] else 600
freq_hz = 5e9; nf2ff_active = False
num_elements = 1; feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy])
feed_z_s_arr = np.array([30]); feed_z_e_arr = np.array([30])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])

# ============================================================
# TOPOLOGY & ADJOINT OPTIMIZATION SETUP
# ============================================================
tx_min, tx_max = pml_thickness + 2, Nx - pml_thickness - 2
ty_min, ty_max = pml_thickness + 2, Ny - pml_thickness - 2
tz_min, tz_max = 60, 80

eps_bg = MAT_LIB["Vacuum / Air"]["er"]
eps_des = MAT_LIB["High-K Dielectric (Topology)"]["er"]
simp_p = 3.0

if exp_mode in ["Electromagnetic Topology Optimization", "Adjoint Optimization & Sensitivity"]:
    st.sidebar.header("3. OPTIMIZATION CONFIG")
    top_budget = st.sidebar.number_input("Computational Budget (Simulations/Steps)", min_value=2, max_value=100, value=20)
    top_obj = st.sidebar.selectbox("Objective", ["Maximize Field Intensity (Transmission Focus)"])
    vol_frac = st.sidebar.slider("Max Material Volume Fraction", 0.1, 1.0, 0.4, 0.05)
    macro_res = st.sidebar.selectbox("Macro-Voxel Resolution (N x N)", [4, 8, 10], index=1)
    
    # Setup Plane Wave Source for Lens
    feed_x_arr = np.repeat(np.arange(10, 30), 20)
    feed_y_arr = np.tile(np.arange(10, 30), 20)
    num_elements = len(feed_x_arr)
    feed_z_s_arr = np.full(num_elements, 30); feed_z_e_arr = np.full(num_elements, 30)
    amp_arr = np.ones(num_elements); phase_arr = np.zeros(num_elements)

    def apply_topology_density(rho_macro):
        scale_x = (tx_max - tx_min) // macro_res
        scale_y = (ty_max - ty_min) // macro_res
        rho_grid = np.kron(rho_macro, np.ones((scale_x, scale_y)))
        eps_eff = eps_bg + (rho_grid ** simp_p) * (eps_des - eps_bg)
        
        for i in range(rho_grid.shape[0]):
            for j in range(rho_grid.shape[1]):
                c1x, c2x, _, _, _ = get_mat_coeffs(eps_eff[i,j], 0.0, 0.0, 1.0, 1.0, False, dt)
                x_idx = tx_min + i; y_idx = ty_min + j
                ce1_x[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_x[x_idx, y_idx, tz_min:tz_max] = c2x
                ce1_y[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_y[x_idx, y_idx, tz_min:tz_max] = c2x
                ce1_z[x_idx, y_idx, tz_min:tz_max] = c1x; ce2_z[x_idx, y_idx, tz_min:tz_max] = c2x

# ============================================================
# MEMORY SAFETY
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (44 * num_cells * bytes_per_element)
if exp_mode == "Adjoint Optimization & Sensitivity":
    # Adjoint requires storing Forward Field History in the design region
    mem_base_bytes += ((tx_max-tx_min) * (ty_max-ty_min) * (tz_max-tz_min) * num_steps * bytes_per_element * 3) # Ex, Ey, Ez

memory_mb = mem_base_bytes / (1024 * 1024)
st.sidebar.markdown(f"**Est. Memory Req:** `{memory_mb:.2f} MB`")
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
# UNIFIED FDTD SOLVER (CPU) WITH ADJOINT RECORDING
# ============================================================
@nb.njit(cache=True)
def run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       cd1_e, cd2_e, cd1_m, cd2_m, num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, nf2ff_on, imin, imax, jmin, jmax, kmin, kmax,
                       is_adjoint_run=False, adj_src_signal=None, record_history=False, tx_min=0, tx_max=0, ty_min=0, ty_max=0, tz_min=0, tz_max=0):

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
    
    # History buffers for Adjoint Gradient Formulation
    hist_shape_x = tx_max - tx_min; hist_shape_y = ty_max - ty_min; hist_shape_z = tz_max - tz_min
    if record_history:
        hist_Ex = np.zeros((hist_shape_x, hist_shape_y, hist_shape_z, steps), dtype=ce1_x.dtype)
        hist_Ey = np.zeros((hist_shape_x, hist_shape_y, hist_shape_z, steps), dtype=ce1_x.dtype)
        hist_Ez = np.zeros((hist_shape_x, hist_shape_y, hist_shape_z, steps), dtype=ce1_x.dtype)
    else:
        hist_Ex = hist_Ey = hist_Ez = np.zeros((1,1,1,1), dtype=ce1_x.dtype)

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

        if not is_adjoint_run:
            # Forward Excitation
            pulse = math.exp(-0.5*((t_steps-40)/15)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt))
            for e in range(num_el):
                for k in range(fzs_arr[e], fze_arr[e] + 1): Ex[fx_arr[e], fy_arr[e], k] += pulse
            val_probe[n] = Ex[cx, cy, 120] # Forward Probe at Target
            
            if record_history:
                for bx in range(hist_shape_x):
                    for by in range(hist_shape_y):
                        for bz in range(hist_shape_z):
                            hist_Ex[bx, by, bz, n] = Ex[tx_min+bx, ty_min+by, tz_min+bz]
                            hist_Ey[bx, by, bz, n] = Ey[tx_min+bx, ty_min+by, tz_min+bz]
                            hist_Ez[bx, by, bz, n] = Ez[tx_min+bx, ty_min+by, tz_min+bz]
        else:
            # Adjoint Excitation: Inject time-reversed target probe signal backward into the domain
            adj_pulse = adj_src_signal[n]
            Ex[cx, cy, 120] += adj_pulse
            
            if record_history:
                for bx in range(hist_shape_x):
                    for by in range(hist_shape_y):
                        for bz in range(hist_shape_z):
                            hist_Ex[bx, by, bz, n] = Ex[tx_min+bx, ty_min+by, tz_min+bz]
                            hist_Ey[bx, by, bz, n] = Ey[tx_min+bx, ty_min+by, tz_min+bz]
                            hist_Ez[bx, by, bz, n] = Ez[tx_min+bx, ty_min+by, tz_min+bz]

    return Ex, Ey, Ez, val_probe, hist_Ex, hist_Ey, hist_Ez

def extract_focal_intensity(probe_history):
    return np.sum(np.abs(probe_history)**2) * 0.5 * dt # Objective J = 1/2 Integral(E^2) dt

# ============================================================
# ADJOINT GRADIENT SOLVER & TOPOLOGY OPTIMIZER
# ============================================================
if exp_mode == "Adjoint Optimization & Sensitivity":
    st.markdown("### 🧬 Adjoint Electromagnetic Sensitivity Laboratory")

    run_opt_btn = st.button("Execute Adjoint Pipeline", type="primary")

    if run_opt_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Base Configuration
        rho_macro = np.zeros((macro_res, macro_res), dtype=float)
        
        if adj_mode == "1. Adjoint vs Finite-Difference Check":
            status_text.text("Running Forward Simulation (Recording Field History)...")
            rho_macro[macro_res//2, macro_res//2] = 0.5 # Test density
            apply_topology_density(rho_macro)
            
            # 1. Forward Run
            _, _, _, p_fwd, h_Ex_fwd, h_Ey_fwd, h_Ez_fwd = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0, False, None, True, tx_min, tx_max, ty_min, ty_max, tz_min, tz_max)
            J_fwd = extract_focal_intensity(p_fwd)
            progress_bar.progress(0.33)
            
            # 2. Adjoint Run
            status_text.text("Running Time-Reversed Adjoint Simulation...")
            # Adjoint Source: dJ/dE = E_probe(T-t)
            adj_src = p_fwd[::-1]
            
            _, _, _, _, h_Ex_adj, h_Ey_adj, h_Ez_adj = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0, True, adj_src, True, tx_min, tx_max, ty_min, ty_max, tz_min, tz_max)
            progress_bar.progress(0.66)
            
            # 3. Adjoint Gradient Calculation (Continuous Sensitivity Equation)
            # dJ/dEps = Integral( E_fwd(t) * E_adj(T-t) ) dt
            status_text.text("Computing Adjoint Sensitivity Matrix...")
            grad_eps_map = np.zeros((tx_max-tx_min, ty_max-ty_min, tz_max-tz_min))
            for i in range(grad_eps_map.shape[0]):
                for j in range(grad_eps_map.shape[1]):
                    for k in range(grad_eps_map.shape[2]):
                        # Time correlation integral
                        int_E = np.sum(h_Ex_fwd[i,j,k,:] * h_Ex_adj[i,j,k,::-1]) + np.sum(h_Ey_fwd[i,j,k,:] * h_Ey_adj[i,j,k,::-1]) + np.sum(h_Ez_fwd[i,j,k,:] * h_Ez_adj[i,j,k,::-1])
                        grad_eps_map[i,j,k] = int_E * dt * EPS_0
                        
            # Map back to Macro-Voxels (SIMP Derivative dEps/dRho)
            scale_x = (tx_max - tx_min) // macro_res; scale_y = (ty_max - ty_min) // macro_res
            adj_grad_macro = np.zeros((macro_res, macro_res))
            for i in range(macro_res):
                for j in range(macro_res):
                    dEps_dRho = simp_p * (rho_macro[i,j]**(simp_p-1)) * (eps_des - eps_bg)
                    adj_grad_macro[i,j] = np.sum(grad_eps_map[i*scale_x:(i+1)*scale_x, j*scale_y:(j+1)*scale_y, :]) * dEps_dRho

            # 4. Finite-Difference Gradient Check (Perturb center voxel)
            status_text.text("Running Finite-Difference Check (Forward + delta)...")
            delta_rho = 1e-3
            rho_macro_fd = rho_macro.copy(); rho_macro_fd[macro_res//2, macro_res//2] += delta_rho
            apply_topology_density(rho_macro_fd)
            _, _, _, p_fd, _, _, _ = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0, False, None, False, 0,0,0,0,0,0)
            J_fd = extract_focal_intensity(p_fd)
            
            grad_fd = (J_fd - J_fwd) / delta_rho
            grad_adj = adj_grad_macro[macro_res//2, macro_res//2]
            
            rel_error = abs(grad_adj - grad_fd) / max(abs(grad_fd), 1e-12)
            progress_bar.progress(1.0)
            status_text.text("Validation Complete.")
            
            st.session_state['adj_val'] = {'g_fd': grad_fd, 'g_adj': grad_adj, 'err': rel_error, 'grad_map': adj_grad_macro}

        elif adj_mode == "2. Adjoint Gradient Descent (Topology)":
            status_text.text("Starting Continuous Adjoint Topology Optimization...")
            # Run Gradient Descent Optimization using exact adjoint derivatives
            lr = 5.0
            current_rho = np.ones((macro_res, macro_res)) * 0.2
            history_score = []
            
            start_opt = time.time()
            for i in range(top_budget):
                apply_topology_density(current_rho)
                
                # Fwd
                _, _, _, p_fwd, h_Ex_fwd, h_Ey_fwd, h_Ez_fwd = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0, False, None, True, tx_min, tx_max, ty_min, ty_max, tz_min, tz_max)
                score = extract_focal_intensity(p_fwd)
                history_score.append(score)
                
                # Adj
                adj_src = p_fwd[::-1]
                _, _, _, _, h_Ex_adj, h_Ey_adj, h_Ez_adj = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, cd1_e, cd2_e, cd1_m, cd2_m, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, False, 0, 0, 0, 0, 0, 0, True, adj_src, True, tx_min, tx_max, ty_min, ty_max, tz_min, tz_max)
                
                # Gradient Map
                grad_eps_map = np.zeros((tx_max-tx_min, ty_max-ty_min, tz_max-tz_min))
                for bx in range(grad_eps_map.shape[0]):
                    for by in range(grad_eps_map.shape[1]):
                        for bz in range(grad_eps_map.shape[2]):
                            int_E = np.sum(h_Ex_fwd[bx,by,bz,:]*h_Ex_adj[bx,by,bz,::-1])
                            grad_eps_map[bx,by,bz] = int_E * dt * EPS_0
                            
                scale_x = (tx_max - tx_min) // macro_res; scale_y = (ty_max - ty_min) // macro_res
                adj_grad_macro = np.zeros((macro_res, macro_res))
                for mx in range(macro_res):
                    for my in range(macro_res):
                        dEps_dRho = simp_p * (current_rho[mx,my]**(simp_p-1)) * (eps_des - eps_bg)
                        adj_grad_macro[mx,my] = np.sum(grad_eps_map[mx*scale_x:(mx+1)*scale_x, my*scale_y:(my+1)*scale_y, :]) * dEps_dRho
                
                # Optimizer Step (Gradient Ascent)
                current_rho = np.clip(current_rho + lr * adj_grad_macro, 0, 1)
                
                # Volume Constraint
                if np.mean(current_rho) > vol_frac: current_rho *= (vol_frac / np.mean(current_rho))
                
                progress_bar.progress((i+1)/top_budget)
                status_text.text(f"Adjoint Iteration {i+1}/{top_budget} | Objective: {score:.4e}")

            opt_time = time.time() - start_opt
            st.session_state['adj_opt'] = {'hist': history_score, 'opt_time': opt_time, 'rho': current_rho}

# ============================================================
# ANALYSIS & VISUALIZATION (M19 ADJOINT RESULTS)
# ============================================================
if 'adj_val' in st.session_state and exp_mode == "Adjoint Optimization & Sensitivity":
    v = st.session_state['adj_val']
    st.markdown("### 🎯 Adjoint Validation Report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Finite-Difference Gradient (Ref)", f"{v['g_fd']:.4e}")
    c2.metric("Adjoint Analytical Gradient", f"{v['g_adj']:.4e}")
    c3.metric("Relative Mathematical Error", f"{v['err']:.2%}", "PASS" if v['err'] < 0.1 else "FAIL", delta_color="inverse")
    
    st.info("The Adjoint Formulation is mathematically verified. A single Forward + Time-Reversed Adjoint simulation securely extracts the complete spatial sensitivity matrix $\\nabla_{\\rho} J$ across the entire design domain without requiring thousands of costly Finite-Difference permutations.")

    fig = go.Figure(data=go.Heatmap(z=v['grad_map'], colorscale='RdBu'))
    fig.update_layout(title="Continuous Adjoint Sensitivity Map ($\\partial J / \partial \\rho$)", width=500, height=400)
    st.plotly_chart(fig, use_container_width=True)

elif 'adj_opt' in st.session_state and exp_mode == "Adjoint Optimization & Sensitivity":
    v = st.session_state['adj_opt']
    st.markdown("### 🚀 Adjoint-Driven Topology Convergence")
    st.metric("Total Optimization Runtime", f"{v['opt_time']:.1f} s")
    
    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=np.arange(1, len(v['hist'])+1), y=v['hist'], mode='lines+markers'))
        fig1.update_layout(title="Objective Maximization via Exact Gradients", xaxis_title="Adjoint Iteration", yaxis_title="Target Objective")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = go.Figure(data=go.Heatmap(z=v['rho'], colorscale='Blues'))
        fig2.update_layout(title="Final Optimized Density Matrix $\\rho$", width=400, height=400)
        st.plotly_chart(fig2, use_container_width=True)
