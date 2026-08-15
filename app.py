"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 14 — Antenna Arrays & Beamforming
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
    "Vacuum / Air": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "is_dispersive": False, "is_anisotropic": False},
    "FR-4 (Lossy)": {"er": 4.4, "mur": 1.0, "sigma": 0.005, "is_dispersive": False, "is_anisotropic": False},
    "PEC (Perfect Conductor)": {"er": 1.0, "mur": 1.0, "sigma": -1.0, "is_dispersive": False, "is_anisotropic": False},
    "Anisotropic Sapphire (Tensor)": {"er_x": 9.3, "er_y": 11.5, "er_z": 9.3, "mur": 1.0, "sigma": 0.0, "is_dispersive": False, "is_anisotropic": True},
    "Dispersive Water (Debye)": {"er_s": 78.4, "er_inf": 4.6, "tau": 8.1e-12, "sigma": 0.05, "mur": 1.0, "is_dispersive": True, "is_anisotropic": False}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 14 — Antenna Arrays & Beamforming")

st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

active_backend = "GPU" if (backend_mode in ["Auto", "GPU"] and GPU_AVAILABLE) else "CPU"
st.sidebar.markdown(f"**Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Experiment", [
    "Antenna Array Laboratory",
    "Single Antenna (Dipole/Patch)", 
    "Advanced Validation Laboratory",
    "Material Dispersion Analyzer"
])

val_suite = None
if exp_mode == "Advanced Validation Laboratory":
    val_suite = st.sidebar.selectbox("Validation Suite", [
        "1. Wave Physics (Velocity)", "2. Boundary & Material", "3. CPU vs GPU", "4. Anisotropic Birefringence"
    ])

# ============================================================
# GRID & DOMAIN SETUP
# ============================================================
st.sidebar.header("2. GRID & DOMAIN")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=20, value=80, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=20, value=120 if exp_mode == "Antenna Array Laboratory" else 80, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=20, value=80, step=10)
dx = dy = dz = 0.005
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
pml_thickness = 10; dt = 0.9 * (1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)))

# Material Arrays
ce1_x = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cp1_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_x = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ce1_y = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cp1_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_y = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ce1_z = np.ones((Nx, Ny, Nz), dtype=dtype_np); ce2_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np); ce3_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cp1_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np); cp2_z = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ch2 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)

def get_mat_coeffs(er, sig, tau, eps_s, eps_inf, is_disp):
    if is_disp:
        d_eps = eps_s - eps_inf
        K1 = (2*tau - dt) / (2*tau + dt); K2 = (EPS_0 * d_eps * dt) / (2*tau + dt)
        A = (EPS_0 * eps_inf / dt) + (K2 / dt) + (sig / 2)
        B = (EPS_0 * eps_inf / dt) - (K2 / dt) - (sig / 2)
        return B/A, 1.0/A, (1.0 - K1)/(A*dt), K1, K2
    else:
        if sig < 0: return 0.0, 0.0, 0.0, 0.0, 0.0 # PEC
        A = (er * EPS_0 / dt) + (sig / 2); B = (er * EPS_0 / dt) - (sig / 2)
        return B/A, 1.0/A, 0.0, 0.0, 0.0

def apply_material_block(x1, x2, y1, y2, z1, z2, mat):
    sig = mat.get("sigma", 0.0); mur = mat.get("mur", 1.0); is_disp = mat.get("is_dispersive", False)
    er_x = mat.get("er_x", mat.get("er", 1.0)); er_y = mat.get("er_y", mat.get("er", 1.0)); er_z = mat.get("er_z", mat.get("er", 1.0))
    tau = mat.get("tau", 0.0); eps_s = mat.get("er_s", 1.0); eps_inf = mat.get("er_inf", 1.0)
    
    c1x, c2x, c3x, p1x, p2x = get_mat_coeffs(er_x, sig, tau, eps_s, eps_inf, is_disp)
    c1y, c2y, c3y, p1y, p2y = get_mat_coeffs(er_y, sig, tau, eps_s, eps_inf, is_disp)
    c1z, c2z, c3z, p1z, p2z = get_mat_coeffs(er_z, sig, tau, eps_s, eps_inf, is_disp)

    ce1_x[x1:x2+1, y1:y2+1, z1:z2+1] = c1x; ce2_x[x1:x2+1, y1:y2+1, z1:z2+1] = c2x; ce3_x[x1:x2+1, y1:y2+1, z1:z2+1] = c3x; cp1_x[x1:x2+1, y1:y2+1, z1:z2+1] = p1x; cp2_x[x1:x2+1, y1:y2+1, z1:z2+1] = p2x
    ce1_y[x1:x2+1, y1:y2+1, z1:z2+1] = c1y; ce2_y[x1:x2+1, y1:y2+1, z1:z2+1] = c2y; ce3_y[x1:x2+1, y1:y2+1, z1:z2+1] = c3y; cp1_y[x1:x2+1, y1:y2+1, z1:z2+1] = p1y; cp2_y[x1:x2+1, y1:y2+1, z1:z2+1] = p2y
    ce1_z[x1:x2+1, y1:y2+1, z1:z2+1] = c1z; ce2_z[x1:x2+1, y1:y2+1, z1:z2+1] = c2z; ce3_z[x1:x2+1, y1:y2+1, z1:z2+1] = c3z; cp1_z[x1:x2+1, y1:y2+1, z1:z2+1] = p1z; cp2_z[x1:x2+1, y1:y2+1, z1:z2+1] = p2z
    ch2[x1:x2+1, y1:y2+1, z1:z2+1] = dt / (mur * MU_0)

apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"])

nf2ff_active = False; box_encloses = True; freq_hz = 1e9
num_elements = 1
feed_x_arr = np.array([cx]); feed_y_arr = np.array([cy])
feed_z_s_arr = np.array([cz]); feed_z_e_arr = np.array([cz])
amp_arr = np.array([1.0]); phase_arr = np.array([0.0])

if exp_mode == "Antenna Array Laboratory":
    st.sidebar.header("3. ARRAY CONFIGURATION")
    num_elements = st.sidebar.selectbox("Number of Elements", [1, 2, 4, 8, 16], index=2)
    freq_ghz = st.sidebar.number_input("Target Frequency (GHz)", min_value=0.1, value=2.4)
    freq_hz = freq_ghz * 1e9; wavelength = C_LIGHT / freq_hz
    
    spacing_wl = st.sidebar.slider("Element Spacing (λ)", 0.1, 2.0, 0.5, 0.05)
    spacing_m = spacing_wl * wavelength
    spacing_cells = int(spacing_m / dy)
    
    st.sidebar.subheader("Beam Steering")
    steer_angle_deg = st.sidebar.slider("Steering Angle (φ°, H-Plane)", -90.0, 90.0, 0.0, 5.0)
    steer_angle_rad = math.radians(steer_angle_deg)
    
    # Grating Lobe Check
    if spacing_wl > 1.0 / (1.0 + abs(math.sin(steer_angle_rad))):
        st.sidebar.warning("⚠️ Warning: Grating Lobes may emerge at this spacing/angle combination!")
    
    dipole_cells = int((wavelength/2) / dz); arm_cells = (dipole_cells - 1) // 2
    
    # Initialize Arrays
    feed_x_arr = np.full(num_elements, cx)
    feed_y_arr = np.zeros(num_elements, dtype=int)
    feed_z_s_arr = np.full(num_elements, cz)
    feed_z_e_arr = np.full(num_elements, cz)
    amp_arr = np.ones(num_elements, dtype=float)
    phase_arr = np.zeros(num_elements, dtype=float)
    
    y_start = cy - (num_elements - 1) * spacing_cells // 2
    
    for n in range(num_elements):
        feed_y_arr[n] = y_start + n * spacing_cells
        # Progressive phase for beam steering: \beta = -k * d * sin(\theta_0)
        phase_arr[n] = -n * (2 * math.pi / wavelength) * spacing_m * math.sin(steer_angle_rad)
        
        # Apply PEC to each dipole element
        apply_material_block(cx, cx, feed_y_arr[n], feed_y_arr[n], cz - arm_cells, cz - 1, MAT_LIB["PEC (Perfect Conductor)"])
        apply_material_block(cx, cx, feed_y_arr[n], feed_y_arr[n], cz + 1, cz + arm_cells, MAT_LIB["PEC (Perfect Conductor)"])

    if min(feed_y_arr) < pml_thickness + 5 or max(feed_y_arr) > Ny - pml_thickness - 5:
        st.error("Array bounds exceed safe PML distance! Increase 'Ny (Cells)' in Domain settings.")
        st.stop()
        
    nf2ff_active = st.sidebar.checkbox("Enable Full-Wave NF2FF", value=True)

# SIMULATION CONTROL
num_steps = st.sidebar.number_input("Timesteps", value=800 if exp_mode == "Antenna Array Laboratory" else 500, step=100)

# ============================================================
# MEMORY SAFETY
# ============================================================
mem_base_bytes = (36 * Nx * Ny * Nz * bytes_per_element)
if nf2ff_active: mem_base_bytes += (5 * Nx * Ny * bytes_per_element)
memory_mb = mem_base_bytes / (1024 * 1024)
st.sidebar.markdown(f"**Est. Memory Req:** `{memory_mb:.2f} MB`")
if active_backend == "GPU" and memory_mb > (GPU_MEM_MB * 0.9): st.stop()
elif active_backend == "CPU" and memory_mb > 3000: st.stop()

def compute_cpml(N, d_pml, delta, dt, m=3, R_err=1e-4, alpha_max=0.05):
    b_e = np.zeros(N, dtype=dtype_np); c_e = np.zeros(N, dtype=dtype_np)
    b_h = np.zeros(N, dtype=dtype_np); c_h = np.zeros(N, dtype=dtype_np)
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

b_e_x, c_e_x, b_h_x, c_h_x = compute_cpml(Nx, pml_thickness, dx, dt)
b_e_y, c_e_y, b_h_y, c_h_y = compute_cpml(Ny, pml_thickness, dy, dt)
b_e_z, c_e_z, b_h_z, c_h_z = compute_cpml(Nz, pml_thickness, dz, dt)

i_min = j_min = k_min = pml_thickness + 4
i_max = Nx - 1 - pml_thickness - 4; j_max = Ny - 1 - pml_thickness - 4; k_max = Nz - 1 - pml_thickness - 4
sx_E = np.zeros((2, j_max-j_min+1, k_max-k_min+1, 2, num_steps), dtype=dtype_np) if nf2ff_active else np.zeros((1,1,1,1,1), dtype=dtype_np)
sy_E = np.zeros((2, i_max-i_min+1, k_max-k_min+1, 2, num_steps), dtype=dtype_np) if nf2ff_active else np.zeros((1,1,1,1,1), dtype=dtype_np)
sz_E = np.zeros((2, i_max-i_min+1, j_max-j_min+1, 2, num_steps), dtype=dtype_np) if nf2ff_active else np.zeros((1,1,1,1,1), dtype=dtype_np)

# ============================================================
# FDTD SOLVER — CPU (NUMBA REFERENCE)
# ============================================================
@nb.njit(cache=True)
def run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, 
                       num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, nf2ff_on, imin, imax, jmin, jmax, kmin, kmax, sx_E):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1_x.dtype)

    for n in range(steps):
        t_steps = float(n)
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                for k in range(Nz - 1):
                    dEz_dy = Ez[i, j+1, k] - Ez[i, j, k]; dEy_dz = Ey[i, j, k+1] - Ey[i, j, k]
                    dEx_dz = Ex[i, j, k+1] - Ex[i, j, k]; dEz_dx = Ez[i+1, j, k] - Ez[i, j, k]
                    dEy_dx = Ey[i+1, j, k] - Ey[i, j, k]; dEx_dy = Ex[i, j+1, k] - Ex[i, j, k]

                    psi_ey_hx[i,j,k] = b_h_y[j] * psi_ey_hx[i,j,k] + c_h_y[j] * dEz_dy * dy
                    psi_ez_hx[i,j,k] = b_h_z[k] * psi_ez_hx[i,j,k] + c_h_z[k] * dEy_dz * dz
                    psi_ez_hy[i,j,k] = b_h_x[i] * psi_ez_hy[i,j,k] + c_h_x[i] * dEx_dz * dz
                    psi_ex_hy[i,j,k] = b_h_z[k] * psi_ex_hy[i,j,k] + c_h_z[k] * dEz_dx * dx
                    psi_ex_hz[i,j,k] = b_h_x[i] * psi_ex_hz[i,j,k] + c_h_x[i] * dEy_dx * dx
                    psi_ey_hz[i,j,k] = b_h_y[j] * psi_ey_hz[i,j,k] + c_h_y[j] * dEx_dy * dy

                    Hx[i,j,k] -= ch2[i,j,k] * ( (dEz_dy/dy + psi_ey_hx[i,j,k]) - (dEy_dz/dz + psi_ez_hx[i,j,k]) )
                    Hy[i,j,k] -= ch2[i,j,k] * ( (dEx_dz/dz + psi_ex_hy[i,j,k]) - (dEz_dx/dx + psi_ez_hy[i,j,k]) )
                    Hz[i,j,k] -= ch2[i,j,k] * ( (dEy_dx/dx + psi_ex_hz[i,j,k]) - (dEx_dy/dy + psi_ey_hz[i,j,k]) )

        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    dHz_dy = Hz[i, j, k] - Hz[i, j-1, k]; dHy_dz = Hy[i, j, k] - Hy[i, j, k-1]
                    dHx_dz = Hx[i, j, k] - Hx[i, j, k-1]; dHz_dx = Hz[i, j, k] - Hz[i-1, j, k]
                    dHy_dx = Hy[i, j, k] - Hy[i-1, j, k]; dHx_dy = Hx[i, j, k] - Hx[i, j-1, k]

                    psi_hy_ex[i,j,k] = b_e_y[j] * psi_hy_ex[i,j,k] + c_e_y[j] * dHz_dy * dy
                    psi_hz_ex[i,j,k] = b_e_z[k] * psi_hz_ex[i,j,k] + c_e_z[k] * dHy_dz * dz
                    psi_hx_ey[i,j,k] = b_e_z[k] * psi_hx_ey[i,j,k] + c_e_z[k] * dHx_dz * dz
                    psi_hz_ey[i,j,k] = b_e_x[i] * psi_hz_ey[i,j,k] + c_e_x[i] * dHz_dx * dx
                    psi_hy_ez[i,j,k] = b_e_x[i] * psi_hy_ez[i,j,k] + c_e_x[i] * dHy_dx * dx
                    psi_hx_ez[i,j,k] = b_e_y[j] * psi_hx_ez[i,j,k] + c_e_y[j] * dHx_dy * dy

                    ex_old = Ex[i,j,k]; ey_old = Ey[i,j,k]; ez_old = Ez[i,j,k]
                    Ex[i,j,k] = ce1_x[i,j,k]*ex_old + ce2_x[i,j,k]*((dHz_dy/dy+psi_hy_ex[i,j,k]) - (dHy_dz/dz+psi_hz_ex[i,j,k])) + ce3_x[i,j,k]*Px[i,j,k]
                    Ey[i,j,k] = ce1_y[i,j,k]*ey_old + ce2_y[i,j,k]*((dHx_dz/dz+psi_hx_ey[i,j,k]) - (dHz_dx/dx+psi_hz_ey[i,j,k])) + ce3_y[i,j,k]*Py[i,j,k]
                    Ez[i,j,k] = ce1_z[i,j,k]*ez_old + ce2_z[i,j,k]*((dHy_dx/dx+psi_hy_ez[i,j,k]) - (dHx_dy/dy+psi_hx_ez[i,j,k])) + ce3_z[i,j,k]*Pz[i,j,k]

                    Px[i,j,k] = cp1_x[i,j,k]*Px[i,j,k] + cp2_x[i,j,k]*(Ex[i,j,k] + ex_old)
                    Py[i,j,k] = cp1_y[i,j,k]*Py[i,j,k] + cp2_y[i,j,k]*(Ey[i,j,k] + ey_old)
                    Pz[i,j,k] = cp1_z[i,j,k]*Pz[i,j,k] + cp2_z[i,j,k]*(Ez[i,j,k] + ez_old)

        # Array Execution
        for e in range(num_el):
            pulse = amp_arr[e] * math.exp(-0.5*((t_steps-60)/20)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt) + phase_arr[e])
            for k in range(fzs_arr[e], fze_arr[e] + 1): Ez[fx_arr[e], fy_arr[e], k] += pulse

        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k-kmin, 0, n] = Ey[i, j, k]; sx_E[f, j-jmin, k-kmin, 1, n] = Ez[i, j, k]

    return Ex, Ey, Ez, sx_E

# ============================================================
# FDTD SOLVER — GPU (CUPY ACCELERATED)
# ============================================================
def run_simulation_gpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_x_np, ce2_x_np, ce3_x_np, cp1_x_np, cp2_x_np, ce1_y_np, ce2_y_np, ce3_y_np, cp1_y_np, cp2_y_np, ce1_z_np, ce2_z_np, ce3_z_np, cp1_z_np, cp2_z_np, ch2_np, 
                       num_el, fx_arr, fy_arr, fzs_arr, fze_arr, amp_arr, phase_arr, freq_hz, nf2ff_on, imin, imax, jmin, jmax, kmin, kmax, sx_E_np):
    dtype_cp = cp.float32 if precision == "float32" else cp.float64
    Ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    Hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    Px = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Py = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Pz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)

    psi_ey_hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ez_hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ez_hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_ex_hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ex_hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ey_hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_hy_ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hz_ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hx_ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_hz_ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hy_ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hx_ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)

    ce1_x = cp.asarray(ce1_x_np, dtype=dtype_cp); ce2_x = cp.asarray(ce2_x_np, dtype=dtype_cp); ce3_x = cp.asarray(ce3_x_np, dtype=dtype_cp); cp1_x = cp.asarray(cp1_x_np, dtype=dtype_cp); cp2_x = cp.asarray(cp2_x_np, dtype=dtype_cp)
    ce1_y = cp.asarray(ce1_y_np, dtype=dtype_cp); ce2_y = cp.asarray(ce2_y_np, dtype=dtype_cp); ce3_y = cp.asarray(ce3_y_np, dtype=dtype_cp); cp1_y = cp.asarray(cp1_y_np, dtype=dtype_cp); cp2_y = cp.asarray(cp2_y_np, dtype=dtype_cp)
    ce1_z = cp.asarray(ce1_z_np, dtype=dtype_cp); ce2_z = cp.asarray(ce2_z_np, dtype=dtype_cp); ce3_z = cp.asarray(ce3_z_np, dtype=dtype_cp); cp1_z = cp.asarray(cp1_z_np, dtype=dtype_cp); cp2_z = cp.asarray(cp2_z_np, dtype=dtype_cp)
    ch2 = cp.asarray(ch2_np, dtype=dtype_cp)
    
    b_h_y_3d = cp.asarray(b_h_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, :-1, :]; c_h_y_3d = cp.asarray(c_h_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, :-1, :]
    b_h_z_3d = cp.asarray(b_h_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, :-1]; c_h_z_3d = cp.asarray(c_h_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, :-1]
    b_h_x_3d = cp.asarray(b_h_x, dtype=dtype_cp).reshape(Nx, 1, 1)[:-1, :, :]; c_h_x_3d = cp.asarray(c_h_x, dtype=dtype_cp).reshape(Nx, 1, 1)[:-1, :, :]
    b_e_y_3d = cp.asarray(b_e_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, 1:-1, :]; c_e_y_3d = cp.asarray(c_e_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, 1:-1, :]
    b_e_z_3d = cp.asarray(b_e_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, 1:-1]; c_e_z_3d = cp.asarray(c_e_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, 1:-1]
    b_e_x_3d = cp.asarray(b_e_x, dtype=dtype_cp).reshape(Nx, 1, 1)[1:-1, :, :]; c_e_x_3d = cp.asarray(c_e_x, dtype=dtype_cp).reshape(Nx, 1, 1)[1:-1, :, :]

    sx_E = cp.zeros_like(sx_E_np, dtype=dtype_cp) if nf2ff_on else None
    s0 = slice(None, -1); s1 = slice(1, None); sC = slice(1, -1); s0E = slice(None, -2)

    for n in range(steps):
        t_steps = float(n)
        dEz_dy = Ez[s0, s1, s0] - Ez[s0, s0, s0]; dEy_dz = Ey[s0, s0, s1] - Ey[s0, s0, s0]; dEx_dz = Ex[s0, s0, s1] - Ex[s0, s0, s0]
        dEz_dx = Ez[s1, s0, s0] - Ez[s0, s0, s0]; dEy_dx = Ey[s1, s0, s0] - Ey[s0, s0, s0]; dEx_dy = Ex[s0, s1, s0] - Ex[s0, s0, s0]

        psi_ey_hx[s0,s0,s0] = b_h_y_3d * psi_ey_hx[s0,s0,s0] + c_h_y_3d * dEz_dy * dy
        psi_ez_hx[s0,s0,s0] = b_h_z_3d * psi_ez_hx[s0,s0,s0] + c_h_z_3d * dEy_dz * dz
        psi_ez_hy[s0,s0,s0] = b_h_x_3d * psi_ez_hy[s0,s0,s0] + c_h_x_3d * dEx_dz * dz
        psi_ex_hy[s0,s0,s0] = b_h_z_3d * psi_ex_hy[s0,s0,s0] + c_h_z_3d * dEz_dx * dx
        psi_ex_hz[s0,s0,s0] = b_h_x_3d * psi_ex_hz[s0,s0,s0] + c_h_x_3d * dEy_dx * dx
        psi_ey_hz[s0,s0,s0] = b_h_y_3d * psi_ey_hz[s0,s0,s0] + c_h_y_3d * dEx_dy * dy

        Hx[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEz_dy/dy + psi_ey_hx[s0,s0,s0]) - (dEy_dz/dz + psi_ez_hx[s0,s0,s0]))
        Hy[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEx_dz/dz + psi_ex_hy[s0,s0,s0]) - (dEz_dx/dx + psi_ez_hy[s0,s0,s0]))
        Hz[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEy_dx/dx + psi_ex_hz[s0,s0,s0]) - (dEx_dy/dy + psi_ey_hz[s0,s0,s0]))

        dHz_dy = Hz[sC, sC, sC] - Hz[sC, s0E, sC]; dHy_dz = Hy[sC, sC, sC] - Hy[sC, sC, s0E]; dHx_dz = Hx[sC, sC, sC] - Hx[sC, sC, s0E]
        dHz_dx = Hz[sC, sC, sC] - Hz[s0E, sC, sC]; dHy_dx = Hy[sC, sC, sC] - Hy[s0E, sC, sC]; dHx_dy = Hx[sC, sC, sC] - Hx[sC, s0E, sC]

        psi_hy_ex[sC,sC,sC] = b_e_y_3d * psi_hy_ex[sC,sC,sC] + c_e_y_3d * dHz_dy * dy
        psi_hz_ex[sC,sC,sC] = b_e_z_3d * psi_hz_ex[sC,sC,sC] + c_e_z_3d * dHy_dz * dz
        psi_hx_ey[sC,sC,sC] = b_e_z_3d * psi_hx_ey[sC,sC,sC] + c_e_z_3d * dHx_dz * dz
        psi_hz_ey[sC,sC,sC] = b_e_x_3d * psi_hz_ey[sC,sC,sC] + c_e_x_3d * dHz_dx * dx
        psi_hy_ez[sC,sC,sC] = b_e_x_3d * psi_hy_ez[sC,sC,sC] + c_e_x_3d * dHy_dx * dx
        psi_hx_ez[sC,sC,sC] = b_e_y_3d * psi_hx_ez[sC,sC,sC] + c_e_y_3d * dHx_dy * dy

        Ex_old = Ex[sC,sC,sC].copy(); Ey_old = Ey[sC,sC,sC].copy(); Ez_old = Ez[sC,sC,sC].copy()

        Ex[sC,sC,sC] = ce1_x[sC,sC,sC]*Ex_old + ce2_x[sC,sC,sC]*((dHz_dy/dy+psi_hy_ex[sC,sC,sC]) - (dHy_dz/dz+psi_hz_ex[sC,sC,sC])) + ce3_x[sC,sC,sC]*Px[sC,sC,sC]
        Ey[sC,sC,sC] = ce1_y[sC,sC,sC]*Ey_old + ce2_y[sC,sC,sC]*((dHx_dz/dz+psi_hx_ey[sC,sC,sC]) - (dHz_dx/dx+psi_hz_ey[sC,sC,sC])) + ce3_y[sC,sC,sC]*Py[sC,sC,sC]
        Ez[sC,sC,sC] = ce1_z[sC,sC,sC]*Ez_old + ce2_z[sC,sC,sC]*((dHy_dx/dx+psi_hy_ez[sC,sC,sC]) - (dHx_dy/dy+psi_hx_ez[sC,sC,sC])) + ce3_z[sC,sC,sC]*Pz[sC,sC,sC]

        Px[sC,sC,sC] = cp1_x[sC,sC,sC]*Px[sC,sC,sC] + cp2_x[sC,sC,sC]*(Ex[sC,sC,sC] + Ex_old)
        Py[sC,sC,sC] = cp1_y[sC,sC,sC]*Py[sC,sC,sC] + cp2_y[sC,sC,sC]*(Ey[sC,sC,sC] + Ey_old)
        Pz[sC,sC,sC] = cp1_z[sC,sC,sC]*Pz[sC,sC,sC] + cp2_z[sC,sC,sC]*(Ez[sC,sC,sC] + Ez_old)

        for e in range(num_el):
            pulse = amp_arr[e] * math.exp(-0.5*((t_steps-60)/20)**2) * math.cos(2.0*math.pi*freq_hz*(n*dt) + phase_arr[e])
            Ez[fx_arr[e], fy_arr[e], fzs_arr[e]:fze_arr[e]+1] += pulse

        if nf2ff_on:
            sx_E[0, :, :, 0, n] = Ey[imin, jmin:jmax+1, kmin:kmax+1]
            sx_E[1, :, :, 0, n] = Ey[imax, jmin:jmax+1, kmin:kmax+1]
            sx_E[0, :, :, 1, n] = Ez[imin, jmin:jmax+1, kmin:kmax+1]
            sx_E[1, :, :, 1, n] = Ez[imax, jmin:jmax+1, kmin:kmax+1]

    return Ex.get(), Ey.get(), Ez.get(), sx_E.get() if nf2ff_on else sx_E_np

# ============================================================
# EXECUTION & LOGIC
# ============================================================
if exp_mode == "Antenna Array Laboratory":
    run_btn = st.button("Run Coupled Array Simulation", type="primary")

    if run_btn:
        with st.spinner(f"Executing Array FDTD on {active_backend}..."):
            start_t = time.time()
            if active_backend == "GPU":
                Ex, Ey, Ez, sx_E = run_simulation_gpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, nf2ff_active, i_min, i_max, j_min, j_max, k_min, k_max, sx_E)
            else:
                Ex, Ey, Ez, sx_E = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1_x, ce2_x, ce3_x, cp1_x, cp2_x, ce1_y, ce2_y, ce3_y, cp1_y, cp2_y, ce1_z, ce2_z, ce3_z, cp1_z, cp2_z, ch2, num_elements, feed_x_arr, feed_y_arr, feed_z_s_arr, feed_z_e_arr, amp_arr, phase_arr, freq_hz, nf2ff_active, i_min, i_max, j_min, j_max, k_min, k_max, sx_E)
            
            # Analytical Array Factor (Broadside is X-axis, Array along Y-axis)
            theta_1d = np.arange(0, 180 + 2, 2)
            phi_1d = np.arange(0, 360 + 2, 2)
            T_mesh, P_mesh = np.meshgrid(np.deg2rad(theta_1d), np.deg2rad(phi_1d), indexing='ij')
            
            # Element Pattern (Dipole along Z)
            cos_th = np.cos(T_mesh); sin_th = np.sin(T_mesh) + 1e-12
            EP = np.abs(np.cos(np.pi/2 * cos_th) / sin_th)
            
            # Array Factor (AF) sum
            AF_cplx = np.zeros_like(T_mesh, dtype=complex)
            for n in range(num_elements):
                yn = (feed_y_arr[n] - cy) * dy
                AF_cplx += amp_arr[n] * np.exp(1j * ((2*np.pi/wavelength) * yn * np.sin(T_mesh) * np.sin(P_mesh) + phase_arr[n]))
            AF = np.abs(AF_cplx)
            AF_norm = AF / np.max(AF)
            Total_Pattern = EP * AF_norm
            
            # Cut Analysis (H-Plane: Theta=90)
            mid_th = len(theta_1d)//2
            H_cut = Total_Pattern[mid_th, :]
            main_lobe_idx = np.argmax(H_cut)
            main_lobe_angle = phi_1d[main_lobe_idx]
            
            # HPBW calculation
            hp_val = np.max(H_cut) / np.sqrt(2)
            above_hp = np.where(H_cut >= hp_val)[0]
            if len(above_hp) > 0:
                hpbw = phi_1d[above_hp[-1]] - phi_1d[above_hp[0]]
                if hpbw > 180: hpbw = 360 - hpbw
            else: hpbw = 0.0

            st.session_state['array_res'] = {
                'calc_time': time.time() - start_t, 'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
                'AF': AF_norm, 'Total_Pattern': Total_Pattern, 'H_cut': H_cut, 'phi_1d': phi_1d,
                'main_angle': main_lobe_angle, 'hpbw': hpbw
            }

# ============================================================
# ANALYSIS & VISUALIZATION
# ============================================================
if 'array_res' in st.session_state and exp_mode == "Antenna Array Laboratory":
    res = st.session_state['array_res']
    st.markdown("### 📊 Array Beamforming Validation")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Execution Time", f"{res['calc_time']:.2f} s")
    c2.metric("Target Steering Angle", f"{steer_angle_deg}°")
    c3.metric("Measured Main-Lobe Direction", f"{res['main_angle'] if res['main_angle']<=180 else res['main_angle']-360}°")
    c4.metric("Half-Power Beamwidth (HPBW)", f"{res['hpbw']}°")

    t1, t2 = st.tabs(["Analytical Array Factor Pattern", "Full-Wave FDTD Near-Field"])

    with t1:
        st.info("The plot below shows the Analytical Element Pattern $\\times$ Array Factor calculation. Full FDTD far-field extraction includes mutual coupling deviations inherently.")
        fig = go.Figure(go.Scatterpolar(r=res['H_cut'], theta=res['phi_1d'], mode='lines', line_color='blue'))
        fig.update_layout(title=f"H-Plane Pattern (Analytical Array Factor)", polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
        with st.spinner("Rendering 3D Array Near-Fields..."):
            plotter = pv.Plotter(off_screen=True, window_size=[800, 400])
            plotter.set_background("white")
            grid = pv.ImageData(dimensions=np.array([Nx, Ny, Nz]), spacing=(dx, dy, dz))
            grid.point_data["|E|"] = E_mag.flatten(order="F")
            plotter.add_mesh(grid.slice_orthogonal(x=cx*dx, y=cy*dy, z=cz*dz), cmap="jet", show_scalar_bar=True)
            
            # Draw array elements
            for n in range(num_elements):
                plotter.add_mesh(pv.Box(bounds=(cx*dx-dx, cx*dx+dx, feed_y_arr[n]*dy-dy, feed_y_arr[n]*dy+dy, (cz-arm_cells)*dz, (cz+arm_cells)*dz)), color='silver')
            
            plotter.view_isometric()
            st.image(plotter.screenshot(transparent_background=False), use_container_width=True)
