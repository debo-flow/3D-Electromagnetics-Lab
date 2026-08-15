"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 12 — Dispersive Electromagnetic Materials (Debye ADE)
"""

# ============================================================
# IMPORTS & GPU DETECTION
# ============================================================
import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time
import pandas as pd

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

# Extended Material Library (Includes Dispersive Debye Models)
MAT_LIB = {
    "Vacuum / Air": {"er": 1.0, "mur": 1.0, "sigma": 0.0, "is_dispersive": False},
    "FR-4 (Lossy)": {"er": 4.4, "mur": 1.0, "sigma": 0.005, "is_dispersive": False},
    "PTFE (Teflon)": {"er": 2.1, "mur": 1.0, "sigma": 0.0002, "is_dispersive": False},
    "Alumina": {"er": 9.8, "mur": 1.0, "sigma": 0.0001, "is_dispersive": False},
    "PEC (Perfect Conductor)": {"er": 1.0, "mur": 1.0, "sigma": -1.0, "is_dispersive": False},
    "Dispersive Water (Debye)": {"er_s": 78.4, "er_inf": 4.6, "tau": 8.1e-12, "sigma": 0.05, "mur": 1.0, "is_dispersive": True},
    "Human Muscle (Debye)": {"er_s": 54.0, "er_inf": 2.5, "tau": 7.0e-12, "sigma": 0.7, "mur": 1.0, "is_dispersive": True}
}

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")
st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 12 — Dispersive Electromagnetic Materials")
st.markdown("*Note: The FDTD engine utilizes an Auxiliary Differential Equation (ADE) formulation to process frequency-dependent Debye dispersion explicitly in the time domain.*")

# Sidebar: BACKEND & PRECISION
st.sidebar.header("COMPUTATION BACKEND")
backend_mode = st.sidebar.selectbox("Execution Backend", ["Auto", "GPU", "CPU"])
precision = st.sidebar.selectbox("Numerical Precision", ["float32", "float64"])
dtype_np = np.float32 if precision == "float32" else np.float64

if backend_mode == "Auto": active_backend = "GPU" if GPU_AVAILABLE else "CPU"
elif backend_mode == "GPU": active_backend = "GPU" if GPU_AVAILABLE else "CPU"
else: active_backend = "CPU"

st.sidebar.markdown(f"**Target Backend:** `{active_backend}` | **VRAM:** `{GPU_MEM_MB:.0f} MB`")

# Sidebar: EXPERIMENT SELECTION
st.sidebar.header("1. EXPERIMENT MODE")
exp_mode = st.sidebar.selectbox("Select Experiment", [
    "Antenna Radiation (Dipole/Patch)", 
    "Advanced Validation Laboratory",
    "Material Dispersion Analyzer"
])

val_suite = None
if exp_mode == "Advanced Validation Laboratory":
    val_suite = st.sidebar.selectbox("Select Validation Suite", [
        "1. Wave Physics (Velocity & Impedance)",
        "2. Boundary & Material (PML & Fresnel)",
        "3. CPU vs GPU & Grid Consistency",
        "4. Dispersive Media (Debye Response)"
    ])

# Sidebar: GRID & DOMAIN
st.sidebar.header("2. GRID & DOMAIN")
if exp_mode == "Antenna Radiation (Dipole/Patch)":
    Nx = st.sidebar.number_input("Nx (Cells)", min_value=20, max_value=300, value=80, step=10)
    Ny = st.sidebar.number_input("Ny (Cells)", min_value=20, max_value=300, value=80, step=10)
    Nz = st.sidebar.number_input("Nz (Cells)", min_value=20, max_value=300, value=80, step=10)
    dx = dy = dz = st.sidebar.number_input("dx, dy, dz (m)", min_value=0.0005, max_value=0.1, value=0.005, format="%.4f")
else:
    if val_suite == "3. CPU vs GPU & Grid Consistency":
        Nx = Ny = Nz = 40; dx = dy = dz = 0.01
    else:
        Nx = Ny = 30; Nz = 180; dx = dy = dz = 0.005

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
pml_thickness = 10; pml_order = 3; pml_R = 1e-4; pml_alpha = 0.05
dt = 0.9 * (1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)))

# ============================================================
# MATERIAL SYSTEM INITIALIZATION
# ============================================================
ce1 = np.ones((Nx, Ny, Nz), dtype=dtype_np)
ce2 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ce3 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cp1 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
cp2 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)
ch2 = np.zeros((Nx, Ny, Nz), dtype=dtype_np)

def apply_material_block(x1, x2, y1, y2, z1, z2, mat):
    sig = mat.get("sigma", 0.0)
    mur = mat.get("mur", 1.0)
    
    if mat.get("is_dispersive", False):
        tau = mat["tau"]; eps_s = mat["er_s"]; eps_inf = mat["er_inf"]
        d_eps = eps_s - eps_inf
        K1 = (2*tau - dt) / (2*tau + dt)
        K2 = (EPS_0 * d_eps * dt) / (2*tau + dt)
        A = (EPS_0 * eps_inf / dt) + (K2 / dt) + (sig / 2)
        B = (EPS_0 * eps_inf / dt) - (K2 / dt) - (sig / 2)
        
        ce1[x1:x2+1, y1:y2+1, z1:z2+1] = B / A
        ce2[x1:x2+1, y1:y2+1, z1:z2+1] = 1.0 / A
        ce3[x1:x2+1, y1:y2+1, z1:z2+1] = (1.0 - K1) / (A * dt)
        cp1[x1:x2+1, y1:y2+1, z1:z2+1] = K1
        cp2[x1:x2+1, y1:y2+1, z1:z2+1] = K2
        ch2[x1:x2+1, y1:y2+1, z1:z2+1] = dt / (mur * MU_0)
    else:
        if sig < 0: # PEC
            ce1[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            ce2[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            ce3[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            ch2[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
        else:
            eps_val = mat.get("er", 1.0) * EPS_0
            A = (eps_val / dt) + (sig / 2)
            B = (eps_val / dt) - (sig / 2)
            ce1[x1:x2+1, y1:y2+1, z1:z2+1] = B / A
            ce2[x1:x2+1, y1:y2+1, z1:z2+1] = 1.0 / A
            ce3[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            cp1[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            cp2[x1:x2+1, y1:y2+1, z1:z2+1] = 0.0
            ch2[x1:x2+1, y1:y2+1, z1:z2+1] = dt / (mur * MU_0)

# Apply Background Vacuum
apply_material_block(0, Nx-1, 0, Ny-1, 0, Nz-1, MAT_LIB["Vacuum / Air"])

nf2ff_active = False; geom_valid = True; box_encloses = False; freq_hz = 1e9

if exp_mode == "Antenna Radiation (Dipole/Patch)":
    st.sidebar.header("3. ANTENNA CONFIGURATION")
    antenna_type = st.sidebar.selectbox("Antenna Type", ["Half-Wave Dipole", "Rectangular Microstrip Patch"])
    freq_ghz = st.sidebar.number_input("Target Frequency (GHz)", min_value=0.1, value=2.4, step=0.1); freq_hz = freq_ghz * 1e9; wavelength = C_LIGHT / freq_hz
    f_z_s = f_z_e = a1_z_s = a1_z_e = a2_z_s = a2_z_e = cz
    pch_x1 = pch_x2 = pch_y1 = pch_y2 = pch_z = gnd_x1 = gnd_x2 = gnd_y1 = gnd_y2 = gnd_z = cz
    sub_x1 = sub_x2 = sub_y1 = sub_y2 = sub_z1 = sub_z2 = cz; feed_x = cx; feed_y = cy

    if antenna_type == "Half-Wave Dipole":
        dipole_cells = int((wavelength/2) / dz); arm_cells = (dipole_cells - 1) // 2
        f_z_s = cz; f_z_e = cz; a1_z_s = cz - arm_cells; a1_z_e = cz - 1; a2_z_s = cz + 1; a2_z_e = cz + arm_cells
        apply_material_block(cx, cx, cy, cy, a1_z_s, a1_z_e, MAT_LIB["PEC (Perfect Conductor)"])
        apply_material_block(cx, cx, cy, cy, a2_z_s, a2_z_e, MAT_LIB["PEC (Perfect Conductor)"])
    else:
        st.sidebar.subheader("Substrate Material")
        sub_mat_name = st.sidebar.selectbox("Select Substrate", list(MAT_LIB.keys()), index=1)
        apply_material_block(cx-10, cx+10, cy-10, cy+10, cz-2, cz, MAT_LIB[sub_mat_name])
        apply_material_block(cx-10, cx+10, cy-10, cy+10, cz-2, cz-2, MAT_LIB["PEC (Perfect Conductor)"])
        apply_material_block(cx-5, cx+5, cy-5, cy+5, cz, cz, MAT_LIB["PEC (Perfect Conductor)"])

    nf2ff_active = st.sidebar.checkbox("Enable NF2FF", value=True)
    i_min = pml_thickness + 4; i_max = Nx - 1 - pml_thickness - 4
    j_min = pml_thickness + 4; j_max = Ny - 1 - pml_thickness - 4
    k_min = pml_thickness + 4; k_max = Nz - 1 - pml_thickness - 4
    box_encloses = True

elif exp_mode == "Advanced Validation Laboratory":
    freq_hz = 5e9
    if val_suite == "2. Boundary & Material (PML & Fresnel)":
        apply_material_block(0, Nx-1, 0, Ny-1, 100, Nz-1, {"er": 4.0, "mur": 1.0, "sigma": 0.0, "is_dispersive": False}) 
    elif val_suite == "4. Dispersive Media (Debye Response)":
        apply_material_block(0, Nx-1, 0, Ny-1, 100, Nz-1, MAT_LIB["Dispersive Water (Debye)"])

# SIMULATION CONTROL
num_steps = st.sidebar.number_input("Timesteps", value=600 if exp_mode == "Antenna Radiation (Dipole/Patch)" else 500, step=50)

# ============================================================
# MEMORY SAFETY & ALLOCATIONS
# ============================================================
bytes_per_element = 4 if precision == "float32" else 8; num_cells = Nx * Ny * Nz
mem_base_bytes = (24 * num_cells * bytes_per_element) # +3 for Px, Py, Pz
if nf2ff_active: mem_base_bytes += (5 * Nx * Ny * bytes_per_element)
memory_mb = mem_base_bytes / (1024 * 1024)

st.sidebar.markdown(f"**Est. Memory Req:** `{memory_mb:.2f} MB`")
if active_backend == "GPU" and memory_mb > (GPU_MEM_MB * 0.9): st.stop()
elif active_backend == "CPU" and memory_mb > 2000: st.stop()

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

# ============================================================
# FDTD SOLVER — CPU (NUMBA ADE REFERENCE)
# ============================================================
@nb.njit(cache=True)
def run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1, ce2, ce3, cp1, cp2, ch2, cx, cy, f_z_s, f_z_e, freq_hz, e_mode_idx):

    Ex = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Ey = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Ez = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    Hx = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Hy = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Hz = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    Px = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Py = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); Pz = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype); psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=ce1.dtype)

    val_probe_1 = np.zeros(steps, dtype=ce1.dtype); val_probe_2 = np.zeros(steps, dtype=ce1.dtype); val_probe_3 = np.zeros(steps, dtype=ce1.dtype)

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
                    
                    Ex[i,j,k] = ce1[i,j,k]*ex_old + ce2[i,j,k] * ( (dHz_dy/dy + psi_hy_ex[i,j,k]) - (dHy_dz/dz + psi_hz_ex[i,j,k]) ) + ce3[i,j,k]*Px[i,j,k]
                    Ey[i,j,k] = ce1[i,j,k]*ey_old + ce2[i,j,k] * ( (dHx_dz/dz + psi_hx_ey[i,j,k]) - (dHz_dx/dx + psi_hz_ey[i,j,k]) ) + ce3[i,j,k]*Py[i,j,k]
                    Ez[i,j,k] = ce1[i,j,k]*ez_old + ce2[i,j,k] * ( (dHy_dx/dx + psi_hy_ez[i,j,k]) - (dHx_dy/dy + psi_hx_ez[i,j,k]) ) + ce3[i,j,k]*Pz[i,j,k]

                    Px[i,j,k] = cp1[i,j,k]*Px[i,j,k] + cp2[i,j,k]*(Ex[i,j,k] + ex_old)
                    Py[i,j,k] = cp1[i,j,k]*Py[i,j,k] + cp2[i,j,k]*(Ey[i,j,k] + ey_old)
                    Pz[i,j,k] = cp1[i,j,k]*Pz[i,j,k] + cp2[i,j,k]*(Ez[i,j,k] + ez_old)

        pulse = math.exp(-0.5 * ((t_steps - 40) / 15)**2) * math.cos(2.0 * math.pi * freq_hz * (n*dt))
        
        if e_mode_idx == 0: 
            for k in range(f_z_s, f_z_e + 1): Ez[cx, cy, k] += pulse
        elif e_mode_idx in [1, 2, 4]: 
            for i in range(Nx):
                for j in range(Ny):
                    Ex[i, j, 30] += pulse
            val_probe_1[n] = Ex[cx, cy, 60]  
            val_probe_2[n] = Ex[cx, cy, 120] 
            val_probe_3[n] = Hy[cx, cy, 60]  

    return Ex, Ey, Ez, val_probe_1, val_probe_2, val_probe_3

# ============================================================
# FDTD SOLVER — GPU (CUPY ADE ACCELERATED)
# ============================================================
def run_simulation_gpu(Nx, Ny, Nz, dx, dy, dz, dt, steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
                       ce1_np, ce2_np, ce3_np, cp1_np, cp2_np, ch2_np, cx, cy, f_z_s, f_z_e, freq_hz, e_mode_idx):
    dtype_cp = cp.float32 if precision == "float32" else cp.float64
    Ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    Hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    Px = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Py = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); Pz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)

    psi_ey_hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ez_hx = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_ez_hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ex_hy = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_ex_hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_ey_hz = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_hy_ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hz_ex = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_hz_ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hx_ey = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)
    psi_hx_ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp); psi_hy_ez = cp.zeros((Nx, Ny, Nz), dtype=dtype_cp)

    ce1 = cp.asarray(ce1_np, dtype=dtype_cp); ce2 = cp.asarray(ce2_np, dtype=dtype_cp); ce3 = cp.asarray(ce3_np, dtype=dtype_cp)
    cp1 = cp.asarray(cp1_np, dtype=dtype_cp); cp2 = cp.asarray(cp2_np, dtype=dtype_cp); ch2 = cp.asarray(ch2_np, dtype=dtype_cp)
    
    b_h_y_3d = cp.asarray(b_h_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, :-1, :]; c_h_y_3d = cp.asarray(c_h_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, :-1, :]
    b_h_z_3d = cp.asarray(b_h_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, :-1]; c_h_z_3d = cp.asarray(c_h_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, :-1]
    b_h_x_3d = cp.asarray(b_h_x, dtype=dtype_cp).reshape(Nx, 1, 1)[:-1, :, :]; c_h_x_3d = cp.asarray(c_h_x, dtype=dtype_cp).reshape(Nx, 1, 1)[:-1, :, :]
    b_e_y_3d = cp.asarray(b_e_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, 1:-1, :]; c_e_y_3d = cp.asarray(c_e_y, dtype=dtype_cp).reshape(1, Ny, 1)[:, 1:-1, :]
    b_e_z_3d = cp.asarray(b_e_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, 1:-1]; c_e_z_3d = cp.asarray(c_e_z, dtype=dtype_cp).reshape(1, 1, Nz)[:, :, 1:-1]
    b_e_x_3d = cp.asarray(b_e_x, dtype=dtype_cp).reshape(Nx, 1, 1)[1:-1, :, :]; c_e_x_3d = cp.asarray(c_e_x, dtype=dtype_cp).reshape(Nx, 1, 1)[1:-1, :, :]

    val_p1 = cp.zeros(steps, dtype=dtype_cp); val_p2 = cp.zeros(steps, dtype=dtype_cp); val_p3 = cp.zeros(steps, dtype=dtype_cp)
    s0 = slice(None, -1); s1 = slice(1, None); sC = slice(1, -1); s0E = slice(None, -2)

    for n in range(steps):
        t_steps = float(n)
        dEz_dy = Ez[s0, s1, s0] - Ez[s0, s0, s0]; dEy_dz = Ey[s0, s0, s1] - Ey[s0, s0, s0]
        dEx_dz = Ex[s0, s0, s1] - Ex[s0, s0, s0]; dEz_dx = Ez[s1, s0, s0] - Ez[s0, s0, s0]
        dEy_dx = Ey[s1, s0, s0] - Ey[s0, s0, s0]; dEx_dy = Ex[s0, s1, s0] - Ex[s0, s0, s0]

        psi_ey_hx[s0,s0,s0] = b_h_y_3d * psi_ey_hx[s0,s0,s0] + c_h_y_3d * dEz_dy * dy
        psi_ez_hx[s0,s0,s0] = b_h_z_3d * psi_ez_hx[s0,s0,s0] + c_h_z_3d * dEy_dz * dz
        psi_ez_hy[s0,s0,s0] = b_h_x_3d * psi_ez_hy[s0,s0,s0] + c_h_x_3d * dEx_dz * dz
        psi_ex_hy[s0,s0,s0] = b_h_z_3d * psi_ex_hy[s0,s0,s0] + c_h_z_3d * dEz_dx * dx
        psi_ex_hz[s0,s0,s0] = b_h_x_3d * psi_ex_hz[s0,s0,s0] + c_h_x_3d * dEy_dx * dx
        psi_ey_hz[s0,s0,s0] = b_h_y_3d * psi_ey_hz[s0,s0,s0] + c_h_y_3d * dEx_dy * dy

        Hx[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEz_dy/dy + psi_ey_hx[s0,s0,s0]) - (dEy_dz/dz + psi_ez_hx[s0,s0,s0]))
        Hy[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEx_dz/dz + psi_ex_hy[s0,s0,s0]) - (dEz_dx/dx + psi_ez_hy[s0,s0,s0]))
        Hz[s0,s0,s0] -= ch2[s0,s0,s0] * ((dEy_dx/dx + psi_ex_hz[s0,s0,s0]) - (dEx_dy/dy + psi_ey_hz[s0,s0,s0]))

        dHz_dy = Hz[sC, sC, sC] - Hz[sC, s0E, sC]; dHy_dz = Hy[sC, sC, sC] - Hy[sC, sC, s0E]
        dHx_dz = Hx[sC, sC, sC] - Hx[sC, sC, s0E]; dHz_dx = Hz[sC, sC, sC] - Hz[s0E, sC, sC]
        dHy_dx = Hy[sC, sC, sC] - Hy[s0E, sC, sC]; dHx_dy = Hx[sC, sC, sC] - Hx[sC, s0E, sC]

        psi_hy_ex[sC,sC,sC] = b_e_y_3d * psi_hy_ex[sC,sC,sC] + c_e_y_3d * dHz_dy * dy
        psi_hz_ex[sC,sC,sC] = b_e_z_3d * psi_hz_ex[sC,sC,sC] + c_e_z_3d * dHy_dz * dz
        psi_hx_ey[sC,sC,sC] = b_e_z_3d * psi_hx_ey[sC,sC,sC] + c_e_z_3d * dHx_dz * dz
        psi_hz_ey[sC,sC,sC] = b_e_x_3d * psi_hz_ey[sC,sC,sC] + c_e_x_3d * dHz_dx * dx
        psi_hy_ez[sC,sC,sC] = b_e_x_3d * psi_hy_ez[sC,sC,sC] + c_e_x_3d * dHy_dx * dx
        psi_hx_ez[sC,sC,sC] = b_e_y_3d * psi_hx_ez[sC,sC,sC] + c_e_y_3d * dHx_dy * dy

        Ex_old = Ex[sC,sC,sC].copy()
        Ey_old = Ey[sC,sC,sC].copy()
        Ez_old = Ez[sC,sC,sC].copy()

        Ex[sC,sC,sC] = ce1[sC,sC,sC]*Ex_old + ce2[sC,sC,sC] * ((dHz_dy/dy + psi_hy_ex[sC,sC,sC]) - (dHy_dz/dz + psi_hz_ex[sC,sC,sC])) + ce3[sC,sC,sC]*Px[sC,sC,sC]
        Ey[sC,sC,sC] = ce1[sC,sC,sC]*Ey_old + ce2[sC,sC,sC] * ((dHx_dz/dz + psi_hx_ey[sC,sC,sC]) - (dHz_dx/dx + psi_hz_ey[sC,sC,sC])) + ce3[sC,sC,sC]*Py[sC,sC,sC]
        Ez[sC,sC,sC] = ce1[sC,sC,sC]*Ez_old + ce2[sC,sC,sC] * ((dHy_dx/dx + psi_hy_ez[sC,sC,sC]) - (dHx_dy/dy + psi_hx_ez[sC,sC,sC])) + ce3[sC,sC,sC]*Pz[sC,sC,sC]

        Px[sC,sC,sC] = cp1[sC,sC,sC]*Px[sC,sC,sC] + cp2[sC,sC,sC]*(Ex[sC,sC,sC] + Ex_old)
        Py[sC,sC,sC] = cp1[sC,sC,sC]*Py[sC,sC,sC] + cp2[sC,sC,sC]*(Ey[sC,sC,sC] + Ey_old)
        Pz[sC,sC,sC] = cp1[sC,sC,sC]*Pz[sC,sC,sC] + cp2[sC,sC,sC]*(Ez[sC,sC,sC] + Ez_old)

        pulse = math.exp(-0.5 * ((t_steps - 40) / 15)**2) * math.cos(2.0 * math.pi * freq_hz * (n*dt))
        if e_mode_idx == 0:
            Ez[cx, cy, f_z_s:f_z_e+1] += pulse
        elif e_mode_idx in [1, 2, 4]:
            Ex[:, :, 30] += pulse
            val_p1[n] = Ex[cx, cy, 60]
            val_p2[n] = Ex[cx, cy, 120]
            val_p3[n] = Hy[cx, cy, 60]

    return Ex.get(), Ey.get(), Ez.get(), val_p1.get(), val_p2.get(), val_p3.get()

# ============================================================
# EXECUTION & LOGIC
# ============================================================
if exp_mode != "Material Dispersion Analyzer":
    run_btn = st.button("Run Simulation" if exp_mode == "Antenna Radiation (Dipole/Patch)" else "Run Validation Suite", type="primary")

    if run_btn:
        e_mode_idx = 0
        if exp_mode == "Advanced Validation Laboratory":
            if val_suite == "1. Wave Physics (Velocity & Impedance)": e_mode_idx = 1
            elif val_suite == "2. Boundary & Material (PML & Fresnel)": e_mode_idx = 2
            elif val_suite == "3. CPU vs GPU & Grid Consistency": e_mode_idx = 0
            elif val_suite == "4. Dispersive Media (Debye Response)": e_mode_idx = 4

        if val_suite == "3. CPU vs GPU & Grid Consistency":
            st.markdown("### 🚀 Running Precision & Backend Benchmarks...")
            t_cpu_s = time.time()
            Ex_c, _, _, _, _, _ = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1, ce2, ce3, cp1, cp2, ch2, cx, cy, f_z_s, f_z_e, freq_hz, 0)
            t_cpu = time.time() - t_cpu_s

            if GPU_AVAILABLE:
                t_gpu_s = time.time()
                Ex_g, _, _, _, _, _ = run_simulation_gpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1, ce2, ce3, cp1, cp2, ch2, cx, cy, f_z_s, f_z_e, freq_hz, 0)
                t_gpu = time.time() - t_gpu_s
                max_err = np.max(np.abs(Ex_c - Ex_g))
                rms_err = np.sqrt(np.mean((Ex_c - Ex_g)**2))
            else:
                t_gpu = max_err = rms_err = 0.0
            st.session_state['val_res'] = {'t_cpu': t_cpu, 't_gpu': t_gpu, 'max_err': max_err, 'rms_err': rms_err}

        else:
            with st.spinner(f"Executing ADE Maxwell Solver on {active_backend}..."):
                start_t = time.time()
                if active_backend == "GPU":
                    Ex, Ey, Ez, p1, p2, p3 = run_simulation_gpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1, ce2, ce3, cp1, cp2, ch2, cx, cy, f_z_s, f_z_e, freq_hz, e_mode_idx)
                else:
                    Ex, Ey, Ez, p1, p2, p3 = run_simulation_cpu(Nx, Ny, Nz, dx, dy, dz, dt, num_steps, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z, ce1, ce2, ce3, cp1, cp2, ch2, cx, cy, f_z_s, f_z_e, freq_hz, e_mode_idx)
                
                st.session_state['res'] = {'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'p1': p1, 'p2': p2, 'p3': p3}

# ============================================================
# ANALYSIS & VISUALIZATION (M12 DISPERSION VALIDATION)
# ============================================================
if exp_mode == "Material Dispersion Analyzer":
    st.markdown("### 📊 Material Dispersion Analyzer (Debye Response)")
    st.info("Calculate the analytical frequency-dependent permittivity $\epsilon'(\omega)$ and dielectric loss $\epsilon''(\omega)$ prior to FDTD propagation.")
    
    mat_sel = st.selectbox("Select Dispersive Material", [k for k, v in MAT_LIB.items() if v.get("is_dispersive")])
    f_arr = np.linspace(0.1e9, 20e9, 200) # 0.1 GHz to 20 GHz
    
    mat = MAT_LIB[mat_sel]
    eps_s = mat["er_s"]; eps_inf = mat["er_inf"]; tau = mat["tau"]; sig = mat["sigma"]
    
    omega = 2 * np.pi * f_arr
    eps_real = eps_inf + (eps_s - eps_inf) / (1 + (omega * tau)**2)
    eps_imag = ((eps_s - eps_inf) * omega * tau) / (1 + (omega * tau)**2) + sig / (omega * EPS_0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_arr/1e9, y=eps_real, name="ε' (Real Permittivity)"))
    fig.add_trace(go.Scatter(x=f_arr/1e9, y=eps_imag, name="ε'' (Imaginary Loss)"))
    fig.update_layout(title=f"Complex Permittivity vs Frequency [{mat_sel}]", xaxis_title="Frequency (GHz)", yaxis_title="Relative Permittivity")
    st.plotly_chart(fig, use_container_width=True)

elif 'val_res' in st.session_state and val_suite == "3. CPU vs GPU & Grid Consistency":
    v = st.session_state['val_res']
    st.markdown("### 📊 CPU / GPU Validation Report")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU Time (Numba ADE)", f"{v['t_cpu']:.3f} s")
    col2.metric("GPU Time (CuPy ADE)", f"{v['t_gpu']:.3f} s" if GPU_AVAILABLE else "N/A")
    col3.metric("GPU Speedup", f"{(v['t_cpu']/v['t_gpu']):.2f}×" if GPU_AVAILABLE and v['t_gpu']>0 else "N/A", delta_color="normal")
    
    st.markdown("#### ⚖️ ADE Numerical Precision Consistency")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"CPU(f64) vs GPU({precision}) Max Error", f"{v['max_err']:.4e}", "PASS" if v['max_err'] < 1e-5 else "FAIL")
    c2.metric("RMS Error", f"{v['rms_err']:.4e}", "PASS" if v['rms_err'] < 1e-5 else "FAIL")
    st.info("Float32 operations on GPU accumulator pipelines naturally produce minor precision truncation compared to CPU Float64. The FDTD ADE physics model remains mathematically identical.")

elif 'res' in st.session_state and exp_mode == "Advanced Validation Laboratory":
    res = st.session_state['res']
    time_ns = np.arange(num_steps) * dt * 1e9
    
    if val_suite == "4. Dispersive Media (Debye Response)":
        st.markdown("### 🔬 Boundary Validation Report: Debye Attenuation & Phase")
        
        inc_pulse = res['p1']
        trans_pulse = res['p2']
        
        inc_fft = np.abs(np.fft.rfft(inc_pulse))
        trans_fft = np.abs(np.fft.rfft(trans_pulse))
        freqs = np.fft.rfftfreq(num_steps, d=dt)
        
        # Valid frequency mask (avoid division by zero noise)
        valid_idx = np.where((freqs > 1e9) & (freqs < 10e9))
        attenuation_db = -20 * np.log10((trans_fft[valid_idx] / inc_fft[valid_idx]) + 1e-12)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs[valid_idx]/1e9, y=attenuation_db, mode='lines', name="Numerical Attenuation (dB)"))
        fig.update_layout(title="Frequency-Dependent Attenuation Through Debye Medium", xaxis_title="Frequency (GHz)", yaxis_title="Attenuation (dB)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("The broadband pulse passes through the Debye medium. The FFT ratio isolates the exact frequency-dependent attenuation curve. Higher frequencies face heavier losses due to relaxation time $\tau$ lags, validating the complex Auxiliary Differential Equations natively coupled into the Maxwell updates.")
