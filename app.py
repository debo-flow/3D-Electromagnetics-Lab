"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 5 — Near-to-Far-Field Transformation (NF2FF)
"""

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import numpy as np
import numba as nb
import pyvista as pv
import plotly.graph_objects as go
import math
import time
import pandas as pd

# PyVista configuration for headless Streamlit environments
pv.global_theme.jupyter_backend = 'static'
pv.global_theme.anti_aliasing = 'fxaa'

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
C_LIGHT = 299792458.0              
MU_0 = 4.0 * math.pi * 1e-7        
EPS_0 = 1.0 / (MU_0 * C_LIGHT**2)  
Z_0 = math.sqrt(MU_0 / EPS_0)      

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")

st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 5 — Near-to-Far-Field Transformation")
st.markdown("*Note: NF2FF transforms sampled near-field electromagnetic data into an estimated far-field solution using the equivalence principle. Results are computational estimates limited by grid discretization.*")

# Sidebar: GRID & DOMAIN
st.sidebar.header("1. GRID & DOMAIN")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=10, max_value=200, value=60, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=10, max_value=200, value=60, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=10, max_value=200, value=80, step=10)
dx = st.sidebar.number_input("dx (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dy = st.sidebar.number_input("dy (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dz = st.sidebar.number_input("dz (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")

# Sidebar: BOUNDARY CONDITIONS
st.sidebar.header("2. BOUNDARY CONDITIONS")
boundary_type = st.sidebar.selectbox("Boundary Type", ["CPML (Absorbing)", "Basic Numerical (PEC)"])
if boundary_type == "CPML (Absorbing)":
    pml_thickness = st.sidebar.number_input("PML Thickness (Cells)", min_value=2, max_value=20, value=10)
    pml_order = 3; pml_R = 1e-4; pml_alpha = 0.05
else:
    pml_thickness = 0; pml_order = 3; pml_R = 1e-4; pml_alpha = 0.05

if pml_thickness >= Nx//2 or pml_thickness >= Ny//2 or pml_thickness >= Nz//2:
    st.sidebar.error("PML thickness must be strictly less than half the domain size.")
    st.stop()

# Sidebar: ANTENNA CONFIGURATION
st.sidebar.header("3. ANTENNA CONFIGURATION")
antenna_active = st.sidebar.checkbox("Enable Half-Wave Dipole", value=True)
freq_ghz = st.sidebar.number_input("Operating Frequency (GHz)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
freq_hz = freq_ghz * 1e9
wavelength = C_LIGHT / freq_hz

dipole_length = st.sidebar.number_input("Dipole Length (m)", min_value=0.01, max_value=5.0, value=round(wavelength/2, 3), step=0.01)
feed_gap_cells = st.sidebar.number_input("Feed Gap (Cells)", min_value=1, max_value=5, value=1)
dipole_cells = int(dipole_length / dz)
arm_cells = (dipole_cells - feed_gap_cells) // 2

# Sidebar: SOURCE
st.sidebar.header("4. SOURCE & EXCITATION")
waveform_type = st.sidebar.selectbox("Source Waveform", ["Modulated Gaussian (Narrowband)", "Gaussian (Broadband)"])
amplitude = st.sidebar.number_input("Feed Amplitude (V/m)", value=1.0)
pulse_width = st.sidebar.number_input("Pulse Width (timesteps)", min_value=5, max_value=500, value=40)
pulse_delay = st.sidebar.number_input("Pulse Delay (timesteps)", min_value=0, max_value=1000, value=120)

# Sidebar: SIMULATION & NEAR-FIELD
st.sidebar.header("5. SIMULATION & NEAR-FIELD")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=5000, value=350, step=50)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
calc_time_avg = st.sidebar.checkbox("Calculate Time-Averaged <S>", value=True)
avg_window = st.sidebar.number_input("Averaging Window (Steps)", min_value=10, max_value=2000, value=100)

# Sidebar: FAR-FIELD TRANSFORMATION (NF2FF)
st.sidebar.header("6. FAR-FIELD TRANSFORMATION (NF2FF)")
nf2ff_active = st.sidebar.checkbox("Enable NF2FF Transformation", value=True)
surf_margin = st.sidebar.number_input("Equivalence Surface Margin (Cells)", min_value=2, max_value=20, value=5)
fft_window = st.sidebar.selectbox("FFT Temporal Window", ["Rectangular", "Hann"])
sweep_type = st.sidebar.selectbox("Observation Sweep", ["Theta Sweep (constant Phi)", "Phi Sweep (constant Theta)", "Single Point"])
obs_distance = st.sidebar.number_input("Observation Distance (m)", min_value=1.0, value=100.0)

if sweep_type == "Theta Sweep (constant Phi)":
    sweep_start = 0; sweep_end = 180; sweep_step = st.sidebar.selectbox("Angular Resolution (deg)", [1, 2, 5])
    const_angle = st.sidebar.number_input("Constant Phi (deg)", value=0.0)
    theta_arr = np.arange(sweep_start, sweep_end + sweep_step, sweep_step)
    phi_arr = np.full_like(theta_arr, const_angle)
elif sweep_type == "Phi Sweep (constant Theta)":
    sweep_start = 0; sweep_end = 360; sweep_step = st.sidebar.selectbox("Angular Resolution (deg)", [1, 2, 5])
    const_angle = st.sidebar.number_input("Constant Theta (deg)", value=90.0)
    phi_arr = np.arange(sweep_start, sweep_end + sweep_step, sweep_step)
    theta_arr = np.full_like(phi_arr, const_angle)
else:
    t_val = st.sidebar.number_input("Theta (deg)", value=90.0)
    p_val = st.sidebar.number_input("Phi (deg)", value=0.0)
    theta_arr = np.array([t_val]); phi_arr = np.array([p_val])

theta_rad = np.deg2rad(theta_arr)
phi_rad = np.deg2rad(phi_arr)

# Sidebar: VISUALIZATION CONTROLS
st.sidebar.header("7. VISUALIZATION CONTROLS")
vis_field = st.sidebar.selectbox("3D Field Quantity", ["|E|", "|H|", "Energy Density (u)", "|<S>| (Time-Avg Poynting)", "Ex"])
plane_select = st.sidebar.selectbox("2D Sampling Plane", ["YZ Plane", "XZ Plane", "XY Plane"])
show_pml = st.sidebar.checkbox("Show PML Region Bounds", value=True)

# ============================================================
# RESOLUTION, VALIDATION & MEMORY CALCULATION
# ============================================================
cells_per_wl = wavelength / max(dx, dy, dz)
dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

# NF2FF Box bounds
i_min = pml_thickness + surf_margin
i_max = Nx - 1 - pml_thickness - surf_margin
j_min = pml_thickness + surf_margin
j_max = Ny - 1 - pml_thickness - surf_margin
k_min = pml_thickness + surf_margin
k_max = Nz - 1 - pml_thickness - surf_margin

# Antenna bounds
if antenna_active:
    feed_z_start = cz - feed_gap_cells // 2
    feed_z_end = feed_z_start + feed_gap_cells - 1
    arm1_z_start = feed_z_start - arm_cells
    arm1_z_end = feed_z_start - 1
    arm2_z_start = feed_z_end + 1
    arm2_z_end = feed_z_end + arm_cells
else:
    feed_z_start = feed_z_end = cz
    arm1_z_start = arm1_z_end = arm2_z_start = arm2_z_end = cz

st.sidebar.markdown(f"**Wavelength ($\lambda$):** `{wavelength:.4f} m`")
st.sidebar.markdown(f"**Resolution:** `{cells_per_wl:.1f} Cells / $\lambda$`")

box_valid = (i_max > i_min) and (j_max > j_min) and (k_max > k_min)
box_encloses = (i_min < cx < i_max) and (j_min < cy < j_max) and (k_min < arm1_z_start) and (k_max > arm2_z_end)

if nf2ff_active and (not box_valid or not box_encloses):
    st.sidebar.error("⚠️ Equivalence surface intersects PML or Antenna. Increase grid size or decrease margin.")
    st.stop()

# Memory Check
num_cells = Nx * Ny * Nz
num_arrays = 6
if pml_thickness > 0: num_arrays += 12 
if calc_time_avg: num_arrays += 3 
memory_mb = (num_arrays * num_cells * 4) / (1024 * 1024)

if nf2ff_active:
    surf_cells = 2*((j_max-j_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(j_max-j_min+1))
    nf2ff_mem = (surf_cells * 4 * num_steps * 4) / (1024 * 1024) # 4 Tangential fields
    memory_mb += nf2ff_mem
    st.sidebar.markdown(f"**NF2FF Surface Cells:** `{surf_cells}`")
    st.sidebar.markdown(f"**Est. NF2FF Memory:** `{nf2ff_mem:.2f} MB`")

st.sidebar.markdown(f"**Total Est. Memory:** `{memory_mb:.2f} MB`")
if memory_mb > 500:
    st.error(f"Requested configuration requires {memory_mb:.2f} MB of RAM, exceeding the 500 MB limit.")
    st.stop()

# ============================================================
# CPML & NF2FF ALLOCATIONS
# ============================================================
def compute_cpml_1d(N, d_pml, delta, dt, m, R_err, alpha_max):
    b_e = np.zeros(N, dtype=np.float32); c_e = np.zeros(N, dtype=np.float32)
    b_h = np.zeros(N, dtype=np.float32); c_h = np.zeros(N, dtype=np.float32)
    if d_pml == 0: return b_e, c_e, b_h, c_h
    d_thickness = d_pml * delta
    sigma_max = - (m + 1) * math.log(R_err) / (2.0 * Z_0 * d_thickness)
    for i in range(N):
        dist_e = (d_pml - i) * delta if i < d_pml else (i - (N - 1 - d_pml)) * delta if i > N - 1 - d_pml else 0.0
        dist_h = (d_pml - i - 0.5) * delta if i < d_pml else (i + 0.5 - (N - 1 - d_pml)) * delta if i > N - 2 - d_pml else 0.0
        dist_h = max(0.0, dist_h)
        if dist_e > 0:
            s_e = sigma_max * (dist_e / d_thickness)**m; a_e = alpha_max * (1.0 - dist_e / d_thickness)**m
            b_e[i] = math.exp(-(s_e + a_e * EPS_0 / dt) * (dt / EPS_0))
            c_e[i] = s_e / (s_e + a_e * EPS_0 / dt) * (b_e[i] - 1.0) / delta
        if dist_h > 0:
            s_h = sigma_max * (dist_h / d_thickness)**m; a_h = alpha_max * (1.0 - dist_h / d_thickness)**m
            b_h[i] = math.exp(-(s_h + a_h * EPS_0 / dt) * (dt / EPS_0))
            c_h[i] = s_h / (s_h + a_h * EPS_0 / dt) * (b_h[i] - 1.0) / delta
    return b_e, c_e, b_h, c_h

b_e_x, c_e_x, b_h_x, c_h_x = compute_cpml_1d(Nx, pml_thickness, dx, dt, pml_order, pml_R, pml_alpha)
b_e_y, c_e_y, b_h_y, c_h_y = compute_cpml_1d(Ny, pml_thickness, dy, dt, pml_order, pml_R, pml_alpha)
b_e_z, c_e_z, b_h_z, c_h_z = compute_cpml_1d(Nz, pml_thickness, dz, dt, pml_order, pml_R, pml_alpha)

# Allocate Surface arrays safely
if nf2ff_active:
    sx_E = np.zeros((2, j_max-j_min+1, k_max-k_min+1, 2, num_steps), dtype=np.float32)
    sx_H = np.zeros((2, j_max-j_min+1, k_max-k_min+1, 2, num_steps), dtype=np.float32)
    sy_E = np.zeros((2, i_max-i_min+1, k_max-k_min+1, 2, num_steps), dtype=np.float32)
    sy_H = np.zeros((2, i_max-i_min+1, k_max-k_min+1, 2, num_steps), dtype=np.float32)
    sz_E = np.zeros((2, i_max-i_min+1, j_max-j_min+1, 2, num_steps), dtype=np.float32)
    sz_H = np.zeros((2, i_max-i_min+1, j_max-j_min+1, 2, num_steps), dtype=np.float32)
else:
    sx_E = sx_H = sy_E = sy_H = sz_E = sz_H = np.zeros((1,1,1,1,1), dtype=np.float32)

# ============================================================
# FDTD SOLVER (NUMBA)
# ============================================================
@nb.njit(cache=True)
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, use_pml, 
                   be_x, ce_x, bh_x, ch_x, be_y, ce_y, bh_y, ch_y, be_z, ce_z, bh_z, ch_z,
                   has_antenna, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
                   freq_hz, amp, delay, width, w_type, calc_avg, avg_start,
                   nf2ff_on, imin, imax, jmin, jmax, kmin, kmax,
                   sx_E, sx_H, sy_E, sy_H, sz_E, sz_H):

    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    Sx_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32); Sy_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32); Sz_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    energy_hist = np.zeros(steps, dtype=np.float32)
    chx = dt / MU_0; cex = dt / EPS_0
    avg_count = 0.0

    for n in range(steps):
        t = float(n) * dt; t_steps = float(n)

        # Update H-field
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                for k in range(Nz - 1):
                    dEz_dy = Ez[i, j+1, k] - Ez[i, j, k]; dEy_dz = Ey[i, j, k+1] - Ey[i, j, k]
                    dEx_dz = Ex[i, j, k+1] - Ex[i, j, k]; dEz_dx = Ez[i+1, j, k] - Ez[i, j, k]
                    dEy_dx = Ey[i+1, j, k] - Ey[i, j, k]; dEx_dy = Ex[i, j+1, k] - Ex[i, j, k]

                    if use_pml:
                        psi_ey_hx[i,j,k] = bh_y[j] * psi_ey_hx[i,j,k] + ch_y[j] * dEz_dy * dy
                        psi_ez_hx[i,j,k] = bh_z[k] * psi_ez_hx[i,j,k] + ch_z[k] * dEy_dz * dz
                        psi_ez_hy[i,j,k] = bh_x[i] * psi_ez_hy[i,j,k] + ch_x[i] * dEx_dz * dz
                        psi_ex_hy[i,j,k] = bh_z[k] * psi_ex_hy[i,j,k] + ch_z[k] * dEz_dx * dx
                        psi_ex_hz[i,j,k] = bh_x[i] * psi_ex_hz[i,j,k] + ch_x[i] * dEy_dx * dx
                        psi_ey_hz[i,j,k] = bh_y[j] * psi_ey_hz[i,j,k] + ch_y[j] * dEx_dy * dy
                        Hx[i,j,k] -= chx * ( (dEz_dy/dy + psi_ey_hx[i,j,k]) - (dEy_dz/dz + psi_ez_hx[i,j,k]) )
                        Hy[i,j,k] -= chx * ( (dEx_dz/dz + psi_ex_hy[i,j,k]) - (dEz_dx/dx + psi_ez_hy[i,j,k]) )
                        Hz[i,j,k] -= chx * ( (dEy_dx/dx + psi_ex_hz[i,j,k]) - (dEx_dy/dy + psi_ey_hz[i,j,k]) )
                    else:
                        Hx[i, j, k] -= chx * (dEz_dy/dy - dEy_dz/dz)
                        Hy[i, j, k] -= chx * (dEx_dz/dz - dEz_dx/dx)
                        Hz[i, j, k] -= chx * (dEy_dx/dx - dEx_dy/dy)

        # Update E-field
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    dHz_dy = Hz[i, j, k] - Hz[i, j-1, k]; dHy_dz = Hy[i, j, k] - Hy[i, j, k-1]
                    dHx_dz = Hx[i, j, k] - Hx[i, j, k-1]; dHz_dx = Hz[i, j, k] - Hz[i-1, j, k]
                    dHy_dx = Hy[i, j, k] - Hy[i-1, j, k]; dHx_dy = Hx[i, j, k] - Hx[i, j-1, k]

                    if use_pml:
                        psi_hy_ex[i,j,k] = be_y[j] * psi_hy_ex[i,j,k] + ce_y[j] * dHz_dy * dy
                        psi_hz_ex[i,j,k] = be_z[k] * psi_hz_ex[i,j,k] + ce_z[k] * dHy_dz * dz
                        psi_hx_ey[i,j,k] = be_z[k] * psi_hx_ey[i,j,k] + ce_z[k] * dHx_dz * dz
                        psi_hz_ey[i,j,k] = be_x[i] * psi_hz_ey[i,j,k] + ce_x[i] * dHz_dx * dx
                        psi_hy_ez[i,j,k] = be_x[i] * psi_hy_ez[i,j,k] + ce_x[i] * dHy_dx * dx
                        psi_hx_ez[i,j,k] = be_y[j] * psi_hx_ez[i,j,k] + ce_y[j] * dHx_dy * dy
                        Ex[i,j,k] += cex * ( (dHz_dy/dy + psi_hy_ex[i,j,k]) - (dHy_dz/dz + psi_hz_ex[i,j,k]) )
                        Ey[i,j,k] += cex * ( (dHx_dz/dz + psi_hx_ey[i,j,k]) - (dHz_dx/dx + psi_hz_ey[i,j,k]) )
                        Ez[i,j,k] += cex * ( (dHy_dx/dx + psi_hy_ez[i,j,k]) - (dHx_dy/dy + psi_hx_ez[i,j,k]) )
                    else:
                        Ex[i, j, k] += cex * (dHz_dy/dy - dHy_dz/dz)
                        Ey[i, j, k] += cex * (dHx_dz/dz - dHz_dx/dx)
                        Ez[i, j, k] += cex * (dHy_dx/dx - dHx_dy/dy)

        # Apply Dipole Geometry
        if has_antenna:
            for k_idx in range(a1_z_s, a1_z_e + 1): Ez[cx, cy, k_idx] = 0.0
            for k_idx in range(a2_z_s, a2_z_e + 1): Ez[cx, cy, k_idx] = 0.0

        # Inject Feed Excitation
        gauss = math.exp(-0.5 * ((t_steps - delay) / width)**2)
        pulse = amp * gauss * math.cos(2.0 * math.pi * freq_hz * t) if w_type == 0 else amp * gauss
        for k_idx in range(f_z_s, f_z_e + 1):
            Ez[cx, cy, k_idx] += pulse

        # Record NF2FF Tangential Fields
        if nf2ff_on:
            for f, i in enumerate([imin, imax]):
                for j in range(jmin, jmax+1):
                    for k_idx in range(kmin, kmax+1):
                        sx_E[f, j-jmin, k_idx-kmin, 0, n] = Ey[i, j, k_idx]; sx_E[f, j-jmin, k_idx-kmin, 1, n] = Ez[i, j, k_idx]
                        sx_H[f, j-jmin, k_idx-kmin, 0, n] = Hy[i, j, k_idx]; sx_H[f, j-jmin, k_idx-kmin, 1, n] = Hz[i, j, k_idx]
            for f, j in enumerate([jmin, jmax]):
                for i in range(imin, imax+1):
                    for k_idx in range(kmin, kmax+1):
                        sy_E[f, i-imin, k_idx-kmin, 0, n] = Ex[i, j, k_idx]; sy_E[f, i-imin, k_idx-kmin, 1, n] = Ez[i, j, k_idx]
                        sy_H[f, i-imin, k_idx-kmin, 0, n] = Hx[i, j, k_idx]; sy_H[f, i-imin, k_idx-kmin, 1, n] = Hz[i, j, k_idx]
            for f, k_idx in enumerate([kmin, kmax]):
                for i in range(imin, imax+1):
                    for j in range(jmin, jmax+1):
                        sz_E[f, i-imin, j-jmin, 0, n] = Ex[i, j, k_idx]; sz_E[f, i-imin, j-jmin, 1, n] = Ey[i, j, k_idx]
                        sz_H[f, i-imin, j-jmin, 0, n] = Hx[i, j, k_idx]; sz_H[f, i-imin, j-jmin, 1, n] = Hy[i, j, k_idx]

        if calc_avg and n >= avg_start:
            avg_count += 1.0
            for i in range(Nx):
                for j in range(Ny):
                    for k_idx in range(Nz):
                        Sx_avg[i,j,k_idx] += Ey[i,j,k_idx]*Hz[i,j,k_idx] - Ez[i,j,k_idx]*Hy[i,j,k_idx]
                        Sy_avg[i,j,k_idx] += Ez[i,j,k_idx]*Hx[i,j,k_idx] - Ex[i,j,k_idx]*Hz[i,j,k_idx]
                        Sz_avg[i,j,k_idx] += Ex[i,j,k_idx]*Hy[i,j,k_idx] - Ey[i,j,k_idx]*Hx[i,j,k_idx]

        e_sum = 0.0
        for i in range(0, Nx, 2):
            for j in range(0, Ny, 2):
                for k_idx in range(0, Nz, 2):
                    e_sum += 0.5 * (EPS_0 * (Ex[i,j,k_idx]**2 + Ey[i,j,k_idx]**2 + Ez[i,j,k_idx]**2) + MU_0 * (Hx[i,j,k_idx]**2 + Hy[i,j,k_idx]**2 + Hz[i,j,k_idx]**2))
        energy_hist[n] = e_sum * (dx * dy * dz * 8)

    if calc_avg and avg_count > 0: Sx_avg /= avg_count; Sy_avg /= avg_count; Sz_avg /= avg_count

    return Ex, Ey, Ez, Hx, Hy, Hz, Sx_avg, Sy_avg, Sz_avg, energy_hist

# ============================================================
# FAR-FIELD TRANSFORMATION (NUMBA)
# ============================================================
@nb.njit(cache=True)
def compute_farfield(freq, theta_arr, phi_arr, r_obs,
                     px_E, px_H, py_E, py_H, pz_E, pz_H,
                     imin, imax, jmin, jmax, kmin, kmax, dx, dy, dz, cx, cy, cz):
    
    k = 2.0 * np.pi * freq / C_LIGHT
    num_angles = len(theta_arr)
    E_theta = np.zeros(num_angles, dtype=np.complex64)
    E_phi = np.zeros(num_angles, dtype=np.complex64)

    for a in range(num_angles):
        theta = theta_arr[a]; phi = phi_arr[a]
        rx = np.sin(theta) * np.cos(phi)
        ry = np.sin(theta) * np.sin(phi)
        rz = np.cos(theta)

        Nx_val, Ny_val, Nz_val = 0j, 0j, 0j
        Lx, Ly, Lz = 0j, 0j, 0j

        # X Faces (Normal +/- x)
        for f in range(2):
            nx = -1.0 if f == 0 else 1.0
            x_prime = (imin if f==0 else imax) - cx
            dS = dy * dz
            for j in range(jmin, jmax+1):
                y_prime = j - cy
                for k_idx in range(kmin, kmax+1):
                    z_prime = k_idx - cz
                    phase = k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz)
                    exp_phase = np.cos(phase) + 1j * np.sin(phase)

                    Ey = px_E[f, j-jmin, k_idx-kmin, 0]; Ez = px_E[f, j-jmin, k_idx-kmin, 1]
                    Hy = px_H[f, j-jmin, k_idx-kmin, 0]; Hz = px_H[f, j-jmin, k_idx-kmin, 1]

                    Js_y = nx * Hz; Js_z = -nx * Hy
                    Ms_y = -nx * Ez; Ms_z = nx * Ey

                    Ny_val += Js_y * exp_phase * dS; Nz_val += Js_z * exp_phase * dS
                    Ly += Ms_y * exp_phase * dS; Lz += Ms_z * exp_phase * dS

        # Y Faces (Normal +/- y)
        for f in range(2):
            ny = -1.0 if f == 0 else 1.0
            y_prime = (jmin if f==0 else jmax) - cy
            dS = dx * dz
            for i in range(imin, imax+1):
                x_prime = i - cx
                for k_idx in range(kmin, kmax+1):
                    z_prime = k_idx - cz
                    phase = k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz)
                    exp_phase = np.cos(phase) + 1j * np.sin(phase)

                    Ex = py_E[f, i-imin, k_idx-kmin, 0]; Ez = py_E[f, i-imin, k_idx-kmin, 1]
                    Hx = py_H[f, i-imin, k_idx-kmin, 0]; Hz = py_H[f, i-imin, k_idx-kmin, 1]

                    Js_x = -ny * Hz; Js_z = ny * Hx
                    Ms_x = ny * Ez; Ms_z = -ny * Ex

                    Nx_val += Js_x * exp_phase * dS; Nz_val += Js_z * exp_phase * dS
                    Lx += Ms_x * exp_phase * dS; Lz += Ms_z * exp_phase * dS

        # Z Faces (Normal +/- z)
        for f in range(2):
            nz = -1.0 if f == 0 else 1.0
            z_prime = (kmin if f==0 else kmax) - cz
            dS = dx * dy
            for i in range(imin, imax+1):
                x_prime = i - cx
                for j in range(jmin, jmax+1):
                    y_prime = j - cy
                    phase = k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz)
                    exp_phase = np.cos(phase) + 1j * np.sin(phase)

                    Ex = pz_E[f, i-imin, j-jmin, 0]; Ey = pz_E[f, i-imin, j-jmin, 1]
                    Hx = pz_H[f, i-imin, j-jmin, 0]; Hy = pz_H[f, i-imin, j-jmin, 1]

                    Js_x = nz * Hy; Js_y = -nz * Hx
                    Ms_x = -nz * Ey; Ms_y = nz * Ex

                    Nx_val += Js_x * exp_phase * dS; Ny_val += Js_y * exp_phase * dS
                    Lx += Ms_x * exp_phase * dS; Ly += Ms_y * exp_phase * dS

        # Convert to Spherical
        N_theta = Nx_val * np.cos(theta)*np.cos(phi) + Ny_val * np.cos(theta)*np.sin(phi) - Nz_val * np.sin(theta)
        N_phi = -Nx_val * np.sin(phi) + Ny_val * np.cos(phi)
        L_theta = Lx * np.cos(theta)*np.cos(phi) + Ly * np.cos(theta)*np.sin(phi) - Lz * np.sin(theta)
        L_phi = -Lx * np.sin(phi) + Ly * np.cos(phi)

        # Compute Asymptotic E_theta, E_phi
        coeff = (1j * k * np.exp(-1j * k * r_obs)) / (4 * np.pi * r_obs)
        E_theta[a] = -coeff * (L_phi + Z_0 * N_theta)
        E_phi[a] = coeff * (L_theta - Z_0 * N_phi)

    return E_theta, E_phi

# ============================================================
# EXECUTION
# ============================================================
run_btn = st.button("Run Simulation & Analysis", type="primary")

if run_btn:
    with st.spinner("Executing FDTD Maxwell Solver..."):
        wt_idx = 0 if waveform_type == "Modulated Gaussian (Narrowband)" else 1
        avg_start = num_steps - avg_window if calc_time_avg else num_steps

        start_t = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, Sx, Sy, Sz, e_hist = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, boundary_type == "CPML (Absorbing)", 
            b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
            antenna_active, cx, cy, feed_z_start, feed_z_end, arm1_z_start, arm1_z_end, arm2_z_start, arm2_z_end,
            freq_hz, amplitude, pulse_delay, pulse_width, wt_idx, calc_time_avg, avg_start,
            nf2ff_active, i_min, i_max, j_min, j_max, k_min, k_max,
            sx_E, sx_H, sy_E, sy_H, sz_E, sz_H
        )
        calc_time = time.time() - start_t
        u_dens = 0.5 * (EPS_0 * (Ex**2 + Ey**2 + Ez**2) + MU_0 * (Hx**2 + Hy**2 + Hz**2))

        # Perform NF2FF Frequency-Domain Transformation
        nf2ff_res = None
        if nf2ff_active:
            with st.spinner("Performing FFT and NF2FF Integration..."):
                # Windowing
                if fft_window == "Hann":
                    window = np.hanning(num_steps)
                else:
                    window = np.ones(num_steps)
                
                # FFT to extract phasors
                freqs = np.fft.rfftfreq(num_steps, d=dt)
                bin_idx = np.argmin(np.abs(freqs - freq_hz))
                
                def extract_phasor(arr):
                    return np.fft.rfft(arr * window, axis=-1)[..., bin_idx] * (2.0 / num_steps)

                px_E = extract_phasor(sx_E); px_H = extract_phasor(sx_H)
                py_E = extract_phasor(sy_E); py_H = extract_phasor(sy_H)
                pz_E = extract_phasor(sz_E); pz_H = extract_phasor(sz_H)

                # Compute Transformation
                Eth, Eph = compute_farfield(
                    freq_hz, theta_rad, phi_rad, obs_distance,
                    px_E, px_H, py_E, py_H, pz_E, pz_H,
                    i_min, i_max, j_min, j_max, k_min, k_max, dx, dy, dz, cx, cy, cz
                )
                nf2ff_res = {"Eth": Eth, "Eph": Eph, "bin_freq": freqs[bin_idx]}

        st.session_state['res'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'Sx': Sx, 'Sy': Sy, 'Sz': Sz, 'u_dens': u_dens, 'e_hist': e_hist,
            'calc_time': calc_time, 'nf2ff': nf2ff_res
        }

# ============================================================
# ANALYSIS & VISUALIZATION
# ============================================================
if 'res' in st.session_state:
    res = st.session_state['res']
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)
    S_mag = np.sqrt(res['Sx']**2 + res['Sy']**2 + res['Sz']**2) if calc_time_avg else np.zeros_like(E_mag)

    # --- VALIDATION PANEL ---
    st.markdown("### Numerical Validation & Diagnostics")
    finite_pass = not (np.isnan(res['Ex']).any() or np.isinf(res['Ex']).any())
    ff_valid = False
    if nf2ff_active and res['nf2ff'] is not None:
        ff_valid = not (np.isnan(res['nf2ff']['Eth']).any() or np.isinf(res['nf2ff']['Eth']).any())

    col_v1, col_v2, col_v3, col_v4, col_v5, col_v6 = st.columns(6)
    col_v1.metric("CFL Stability", "PASS" if dt <= dt_max else "FAIL")
    col_v2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    col_v3.metric("Equivalence Surface", "PASS" if (nf2ff_active and box_encloses) else ("NOT RUN" if not nf2ff_active else "FAIL"))
    col_v4.metric("Frequency Data", "PASS" if nf2ff_active else "NOT RUN")
    col_v5.metric("Phase Data", "PASS" if nf2ff_active else "NOT RUN")
    col_v6.metric("Far-Field Finite", "PASS" if ff_valid else ("NOT RUN" if not nf2ff_active else "FAIL"))
    st.divider()

    # --- TAB LAYOUT ---
    t1, t2, t3 = st.tabs(["3D Near-Field & Geometry", "2D Field Maps", "Far-Field NF2FF Analysis"])

    # 1. 3D VISUALIZATION
    with t1:
        field_map = {"|E|": E_mag, "|H|": H_mag, "Energy Density (u)": res['u_dens'], "|<S>| (Time-Avg Poynting)": S_mag, "Ex": res['Ex']}
        with st.spinner("Rendering..."):
            grid = pv.ImageData(dimensions=np.array([Nx, Ny, Nz]), spacing=(dx, dy, dz))
            grid.point_data[vis_field] = field_map[vis_field].flatten(order="F")
            plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
            plotter.set_background("white")
            
            slc_x, slc_y, slc_z = (cx*dx if plane_select == "YZ Plane" else None, cy*dy if plane_select == "XZ Plane" else None, cz*dz if plane_select == "XY Plane" else None)
            plotter.add_mesh(grid.slice_orthogonal(x=slc_x, y=slc_y, z=slc_z), cmap="jet", show_scalar_bar=True)

            if show_pml and pml_thickness > 0:
                plotter.add_mesh(pv.Box(bounds=(pml_thickness*dx, (Nx-pml_thickness)*dx, pml_thickness*dy, (Ny-pml_thickness)*dy, pml_thickness*dz, (Nz-pml_thickness)*dz)), style='wireframe', color='red')
            
            if antenna_active:
                plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm1_z_start*dz, arm1_z_end*dz)), color='silver')
                plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm2_z_start*dz, arm2_z_end*dz)), color='silver')

            if nf2ff_active:
                nf2ff_bounds = (i_min*dx, i_max*dx, j_min*dy, j_max*dy, k_min*dz, k_max*dz)
                plotter.add_mesh(pv.Box(bounds=nf2ff_bounds), style='wireframe', color='green', line_width=3, label='Equivalence Surface')

            plotter.view_isometric()
            st.image(plotter.screenshot(transparent_background=False), use_container_width=True)

    # 2. 2D NEAR-FIELD MAPS
    with t2:
        c1, c2 = st.columns([1, 4])
        map_qty = c1.selectbox("Map Quantity", ["|E|", "Energy Density", "|<S>|"])
        md = {"|E|": E_mag, "Energy Density": res['u_dens'], "|<S>|": S_mag}[map_qty]

        fig_2d = go.Figure()
        if plane_select == "YZ Plane":
            slice_2d = md[cx, pml_thickness:Ny-pml_thickness, pml_thickness:Nz-pml_thickness].T
            fig_2d.add_trace(go.Heatmap(z=slice_2d, x=np.arange(pml_thickness, Ny-pml_thickness)*dy, y=np.arange(pml_thickness, Nz-pml_thickness)*dz, colorscale='Jet'))
            fig_2d.update_layout(xaxis_title="Y (m)", yaxis_title="Z (m)")
        elif plane_select == "XZ Plane":
            slice_2d = md[pml_thickness:Nx-pml_thickness, cy, pml_thickness:Nz-pml_thickness].T
            fig_2d.add_trace(go.Heatmap(z=slice_2d, x=np.arange(pml_thickness, Nx-pml_thickness)*dx, y=np.arange(pml_thickness, Nz-pml_thickness)*dz, colorscale='Jet'))
            fig_2d.update_layout(xaxis_title="X (m)", yaxis_title="Z (m)")
        
        fig_2d.update_layout(title=f"2D Map on {plane_select}", height=600)
        c2.plotly_chart(fig_2d, use_container_width=True)

    # 3. FAR-FIELD NF2FF ANALYSIS
    with t3:
        if not nf2ff_active:
            st.info("Enable Far-Field Transformation (NF2FF) in the sidebar to view results.")
        else:
            ff = res['nf2ff']
            st.markdown(f"**FFT Extracted Frequency:** `{ff['bin_freq']:.4f} GHz` | **Observation Distance:** `{obs_distance} m`")
            
            E_total_mag = np.sqrt(np.abs(ff['Eth'])**2 + np.abs(ff['Eph'])**2)
            E_norm = E_total_mag / (np.max(E_total_mag) + 1e-12)

            fig_ff = go.Figure()
            if sweep_type == "Theta Sweep (constant Phi)":
                fig_ff.add_trace(go.Scatter(x=theta_arr, y=E_norm, mode='lines', name='Normalized |E|'))
                fig_ff.update_layout(title=f"Far-Field Magnitude vs Theta (Phi = {const_angle}°)", xaxis_title="Theta (deg)", yaxis_title="Normalized |E|")
            elif sweep_type == "Phi Sweep (constant Theta)":
                fig_ff.add_trace(go.Scatter(x=phi_arr, y=E_norm, mode='lines', name='Normalized |E|'))
                fig_ff.update_layout(title=f"Far-Field Magnitude vs Phi (Theta = {const_angle}°)", xaxis_title="Phi (deg)", yaxis_title="Normalized |E|")
            else:
                st.metric(f"|E| at (Theta={theta_arr[0]}°, Phi={phi_arr[0]}°)", f"{E_total_mag[0]:.4e} V/m")

            if sweep_type != "Single Point":
                st.plotly_chart(fig_ff, use_container_width=True)

            # Data Export
            df = pd.DataFrame({
                "Theta_deg": theta_arr, "Phi_deg": phi_arr,
                "E_theta_mag": np.abs(ff['Eth']), "E_theta_phase": np.angle(ff['Eth']),
                "E_phi_mag": np.abs(ff['Eph']), "E_phi_phase": np.angle(ff['Eph']),
                "E_total_mag": E_total_mag, "Normalized_E": E_norm
            })
            csv = df.to_csv(index=False)
            st.download_button(label="Download NF2FF Data (CSV)", data=csv, file_name='nf2ff_data.csv', mime='text/csv')
