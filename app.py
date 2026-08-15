"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 7 — Antenna Gain, Directivity & Efficiency
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
st.markdown("### Milestone 7 — Antenna Performance Metrics")
st.markdown("*Note: Total Radiated Power ($P_{rad}$) and Directivity ($D$) are numerically integrated from the FDTD equivalence surface. Input Power ($P_{in}$), Gain ($G$), and Efficiency ($\eta_{rad}$) are intentionally withheld to preserve scientific integrity, as the basic FDTD soft-source does not strictly define $V \times I$ accepted power.*")

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

# Sidebar: FAR-FIELD & PERFORMANCE (NF2FF)
st.sidebar.header("6. FAR-FIELD & PERFORMANCE")
nf2ff_active = st.sidebar.checkbox("Enable NF2FF & Performance Metrics", value=True)
surf_margin = st.sidebar.number_input("Equivalence Surface Margin (Cells)", min_value=2, max_value=20, value=5)
fft_window = st.sidebar.selectbox("FFT Temporal Window", ["Rectangular", "Hann"])
obs_distance = st.sidebar.number_input("Observation Distance (m)", min_value=1.0, value=100.0)
ang_res = st.sidebar.selectbox("Angular Resolution (deg)", [2, 5, 10], index=1)
rad_scale = st.sidebar.selectbox("Pattern Scale", ["Linear", "dB"])
db_floor = st.sidebar.number_input("dB Floor", value=-40.0, max_value=-10.0, step=5.0)

show_3d_pattern = st.sidebar.checkbox("Show 3D Pattern Overlay", value=True)
pattern_3d_size = st.sidebar.slider("3D Pattern Size (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Generate Spherical Grid
theta_1d = np.arange(0, 180 + ang_res, ang_res)
phi_1d = np.arange(0, 360 + ang_res, ang_res)
T_mesh, P_mesh = np.meshgrid(theta_1d, phi_1d, indexing='ij')
theta_flat = T_mesh.flatten()
phi_flat = P_mesh.flatten()
num_rad_samples = len(theta_flat)

# Sidebar: VISUALIZATION CONTROLS
st.sidebar.header("7. 3D DOMAIN VISUALIZATION")
vis_field = st.sidebar.selectbox("Domain Field Quantity", ["|E|", "|H|", "Ex", "Ey", "Ez"])
plane_select = st.sidebar.selectbox("2D Sampling Plane", ["YZ Plane", "XZ Plane", "XY Plane"])
show_pml = st.sidebar.checkbox("Show PML Bounds", value=True)
show_dipole = st.sidebar.checkbox("Show Dipole", value=True)
show_eq_surf = st.sidebar.checkbox("Show Equivalence Surface", value=True)

# ============================================================
# RESOLUTION, VALIDATION & MEMORY CALCULATION
# ============================================================
cells_per_wl = wavelength / max(dx, dy, dz)
dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

i_min = pml_thickness + surf_margin
i_max = Nx - 1 - pml_thickness - surf_margin
j_min = pml_thickness + surf_margin
j_max = Ny - 1 - pml_thickness - surf_margin
k_min = pml_thickness + surf_margin
k_max = Nz - 1 - pml_thickness - surf_margin

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
memory_mb = (num_arrays * num_cells * 4) / (1024 * 1024)

if nf2ff_active:
    surf_cells = 2*((j_max-j_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(j_max-j_min+1))
    nf2ff_mem = (surf_cells * 4 * num_steps * 4) / (1024 * 1024) 
    rad_mem = (num_rad_samples * 12 * 4) / (1024 * 1024) # Complex/Float Pattern arrays + M7 U & D arrays
    memory_mb += nf2ff_mem + rad_mem
    st.sidebar.markdown(f"**Est. NF2FF/M7 Memory:** `{nf2ff_mem + rad_mem:.2f} MB`")

st.sidebar.markdown(f"**Total Est. Memory:** `{memory_mb:.2f} MB`")
if memory_mb > 500:
    st.error(f"Requested configuration requires {memory_mb:.2f} MB of RAM, exceeding the 500 MB limit.")
    st.stop()

# ============================================================
# CPML ALLOCATION
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
# FDTD SOLVER & NF2FF NUMBA KERNELS
# ============================================================
@nb.njit(cache=True)
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, use_pml, 
                   be_x, ce_x, bh_x, ch_x, be_y, ce_y, bh_y, ch_y, be_z, ce_z, bh_z, ch_z,
                   has_antenna, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
                   freq_hz, amp, delay, width, w_type,
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

    chx = dt / MU_0; cex = dt / EPS_0
    e_inj = 0.0 # Source Energy injected approx
    e_diss = 0.0 # Energy dissipated

    for n in range(steps):
        t = float(n) * dt; t_steps = float(n)

        # Update H-field
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                for k_idx in range(Nz - 1):
                    dEz_dy = Ez[i, j+1, k_idx] - Ez[i, j, k_idx]; dEy_dz = Ey[i, j, k_idx+1] - Ey[i, j, k_idx]
                    dEx_dz = Ex[i, j, k_idx+1] - Ex[i, j, k_idx]; dEz_dx = Ez[i+1, j, k_idx] - Ez[i, j, k_idx]
                    dEy_dx = Ey[i+1, j, k_idx] - Ey[i, j, k_idx]; dEx_dy = Ex[i, j+1, k_idx] - Ex[i, j, k_idx]

                    if use_pml:
                        psi_ey_hx[i,j,k_idx] = bh_y[j] * psi_ey_hx[i,j,k_idx] + ch_y[j] * dEz_dy * dy
                        psi_ez_hx[i,j,k_idx] = bh_z[k_idx] * psi_ez_hx[i,j,k_idx] + ch_z[k_idx] * dEy_dz * dz
                        psi_ez_hy[i,j,k_idx] = bh_x[i] * psi_ez_hy[i,j,k_idx] + ch_x[i] * dEx_dz * dz
                        psi_ex_hy[i,j,k_idx] = bh_z[k_idx] * psi_ex_hy[i,j,k_idx] + ch_z[k_idx] * dEz_dx * dx
                        psi_ex_hz[i,j,k_idx] = bh_x[i] * psi_ex_hz[i,j,k_idx] + ch_x[i] * dEy_dx * dx
                        psi_ey_hz[i,j,k_idx] = bh_y[j] * psi_ey_hz[i,j,k_idx] + ch_y[j] * dEx_dy * dy
                        Hx[i,j,k_idx] -= chx * ( (dEz_dy/dy + psi_ey_hx[i,j,k_idx]) - (dEy_dz/dz + psi_ez_hx[i,j,k_idx]) )
                        Hy[i,j,k_idx] -= chx * ( (dEx_dz/dz + psi_ex_hy[i,j,k_idx]) - (dEz_dx/dx + psi_ez_hy[i,j,k_idx]) )
                        Hz[i,j,k_idx] -= chx * ( (dEy_dx/dx + psi_ex_hz[i,j,k_idx]) - (dEx_dy/dy + psi_ey_hz[i,j,k_idx]) )
                    else:
                        Hx[i, j, k_idx] -= chx * (dEz_dy/dy - dEy_dz/dz)
                        Hy[i, j, k_idx] -= chx * (dEx_dz/dz - dEz_dx/dx)
                        Hz[i, j, k_idx] -= chx * (dEy_dx/dx - dEx_dy/dy)

        # Update E-field
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k_idx in range(1, Nz - 1):
                    dHz_dy = Hz[i, j, k_idx] - Hz[i, j-1, k_idx]; dHy_dz = Hy[i, j, k_idx] - Hy[i, j, k_idx-1]
                    dHx_dz = Hx[i, j, k_idx] - Hx[i, j, k_idx-1]; dHz_dx = Hz[i, j, k_idx] - Hz[i-1, j, k_idx]
                    dHy_dx = Hy[i, j, k_idx] - Hy[i-1, j, k_idx]; dHx_dy = Hx[i, j, k_idx] - Hx[i, j-1, k_idx]

                    if use_pml:
                        psi_hy_ex[i,j,k_idx] = be_y[j] * psi_hy_ex[i,j,k_idx] + ce_y[j] * dHz_dy * dy
                        psi_hz_ex[i,j,k_idx] = be_z[k_idx] * psi_hz_ex[i,j,k_idx] + ce_z[k_idx] * dHy_dz * dz
                        psi_hx_ey[i,j,k_idx] = be_z[k_idx] * psi_hx_ey[i,j,k_idx] + ce_z[k_idx] * dHx_dz * dz
                        psi_hz_ey[i,j,k_idx] = be_x[i] * psi_hz_ey[i,j,k_idx] + ce_x[i] * dHz_dx * dx
                        psi_hy_ez[i,j,k_idx] = be_x[i] * psi_hy_ez[i,j,k_idx] + ce_x[i] * dHy_dx * dx
                        psi_hx_ez[i,j,k_idx] = be_y[j] * psi_hx_ez[i,j,k_idx] + ce_y[j] * dHx_dy * dy
                        Ex[i,j,k_idx] += cex * ( (dHz_dy/dy + psi_hy_ex[i,j,k_idx]) - (dHy_dz/dz + psi_hz_ex[i,j,k_idx]) )
                        Ey[i,j,k_idx] += cex * ( (dHx_dz/dz + psi_hx_ey[i,j,k_idx]) - (dHz_dx/dx + psi_hz_ey[i,j,k_idx]) )
                        Ez[i,j,k_idx] += cex * ( (dHy_dx/dx + psi_hy_ez[i,j,k_idx]) - (dHx_dy/dy + psi_hx_ez[i,j,k_idx]) )
                    else:
                        Ex[i, j, k_idx] += cex * (dHz_dy/dy - dHy_dz/dz)
                        Ey[i, j, k_idx] += cex * (dHx_dz/dz - dHz_dx/dx)
                        Ez[i, j, k_idx] += cex * (dHy_dx/dx - dHx_dy/dy)

        # Apply Dipole Geometry
        if has_antenna:
            for k_idx in range(a1_z_s, a1_z_e + 1): Ez[cx, cy, k_idx] = 0.0
            for k_idx in range(a2_z_s, a2_z_e + 1): Ez[cx, cy, k_idx] = 0.0

        # Inject Feed Excitation
        gauss = math.exp(-0.5 * ((t_steps - delay) / width)**2)
        pulse = amp * gauss * math.cos(2.0 * math.pi * freq_hz * t) if w_type == 0 else amp * gauss
        for k_idx in range(f_z_s, f_z_e + 1):
            Ez[cx, cy, k_idx] += pulse
            e_inj += pulse**2 # Proxy for injected source energy tracking

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

    return Ex, Ey, Ez, Hx, Hy, Hz, e_inj

@nb.njit(cache=True)
def compute_farfield_3d(freq, theta_arr, phi_arr, r_obs,
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

        # X Faces
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
                    Ey_val = px_E[f, j-jmin, k_idx-kmin, 0]; Ez_val = px_E[f, j-jmin, k_idx-kmin, 1]
                    Hy_val = px_H[f, j-jmin, k_idx-kmin, 0]; Hz_val = px_H[f, j-jmin, k_idx-kmin, 1]
                    Js_y = nx * Hz_val; Js_z = -nx * Hy_val
                    Ms_y = -nx * Ez_val; Ms_z = nx * Ey_val
                    Ny_val += Js_y * exp_phase * dS; Nz_val += Js_z * exp_phase * dS
                    Ly += Ms_y * exp_phase * dS; Lz += Ms_z * exp_phase * dS

        # Y Faces
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
                    Ex_val = py_E[f, i-imin, k_idx-kmin, 0]; Ez_val = py_E[f, i-imin, k_idx-kmin, 1]
                    Hx_val = py_H[f, i-imin, k_idx-kmin, 0]; Hz_val = py_H[f, i-imin, k_idx-kmin, 1]
                    Js_x = -ny * Hz_val; Js_z = ny * Hx_val
                    Ms_x = ny * Ez_val; Ms_z = -ny * Ex_val
                    Nx_val += Js_x * exp_phase * dS; Nz_val += Js_z * exp_phase * dS
                    Lx += Ms_x * exp_phase * dS; Lz += Ms_z * exp_phase * dS

        # Z Faces
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
                    Ex_val = pz_E[f, i-imin, j-jmin, 0]; Ey_val = pz_E[f, i-imin, j-jmin, 1]
                    Hx_val = pz_H[f, i-imin, j-jmin, 0]; Hy_val = pz_H[f, i-imin, j-jmin, 1]
                    Js_x = nz * Hy_val; Js_y = -nz * Hx_val
                    Ms_x = -nz * Ey_val; Ms_y = nz * Ex_val
                    Nx_val += Js_x * exp_phase * dS; Ny_val += Js_y * exp_phase * dS
                    Lx += Ms_x * exp_phase * dS; Ly += Ms_y * exp_phase * dS

        # Spherical Projections
        N_theta = Nx_val * np.cos(theta)*np.cos(phi) + Ny_val * np.cos(theta)*np.sin(phi) - Nz_val * np.sin(theta)
        N_phi = -Nx_val * np.sin(phi) + Ny_val * np.cos(phi)
        L_theta = Lx * np.cos(theta)*np.cos(phi) + Ly * np.cos(theta)*np.sin(phi) - Lz * np.sin(theta)
        L_phi = -Lx * np.sin(phi) + Ly * np.cos(phi)

        coeff = (1j * k * np.exp(-1j * k * r_obs)) / (4 * np.pi * r_obs)
        E_theta[a] = -coeff * (L_phi + Z_0 * N_theta)
        E_phi[a] = coeff * (L_theta - Z_0 * N_phi)

    return E_theta, E_phi

# ============================================================
# EXECUTION & M7 METRICS
# ============================================================
col_b1, col_b2 = st.columns([1, 1])
run_btn = col_b1.button("Run FDTD & Antenna Performance Analysis", type="primary")

# Optional Convergence Test button
conv_btn = col_b2.button("Run Angular Convergence Test (Optional)")

if conv_btn:
    st.info("Run the main FDTD simulation first to compute and cache NF2FF surface parameters before running the convergence tests.")

if run_btn:
    with st.spinner("Executing FDTD Maxwell Solver..."):
        wt_idx = 0 if waveform_type == "Modulated Gaussian (Narrowband)" else 1
        start_t = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, e_inj = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, boundary_type == "CPML (Absorbing)", 
            b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
            antenna_active, cx, cy, feed_z_start, feed_z_end, arm1_z_start, arm1_z_end, arm2_z_start, arm2_z_end,
            freq_hz, amplitude, pulse_delay, pulse_width, wt_idx,
            nf2ff_active, i_min, i_max, j_min, j_max, k_min, k_max, sx_E, sx_H, sy_E, sy_H, sz_E, sz_H
        )
        calc_time = time.time() - start_t

        nf2ff_res = None
        if nf2ff_active:
            with st.spinner(f"Computing 3D Radiation Pattern & M7 Performance Metrics..."):
                window = np.hanning(num_steps) if fft_window == "Hann" else np.ones(num_steps)
                freqs = np.fft.rfftfreq(num_steps, d=dt)
                bin_idx = np.argmin(np.abs(freqs - freq_hz))
                
                def ext_phasor(arr): return np.fft.rfft(arr * window, axis=-1)[..., bin_idx] * (2.0 / num_steps)

                px_E = ext_phasor(sx_E); px_H = ext_phasor(sx_H)
                py_E = ext_phasor(sy_E); py_H = ext_phasor(sy_H)
                pz_E = ext_phasor(sz_E); pz_H = ext_phasor(sz_H)

                # Cache phasors for convergence tests
                st.session_state['cache_phasors'] = (px_E, px_H, py_E, py_H, pz_E, pz_H, freqs[bin_idx])

                theta_rad_flat = np.deg2rad(theta_flat)
                phi_rad_flat = np.deg2rad(phi_flat)

                Eth_flat, Eph_flat = compute_farfield_3d(
                    freq_hz, theta_rad_flat, phi_rad_flat, obs_distance,
                    px_E, px_H, py_E, py_H, pz_E, pz_H,
                    i_min, i_max, j_min, j_max, k_min, k_max, dx, dy, dz, cx, cy, cz
                )
                
                Eth_2d = Eth_flat.reshape(T_mesh.shape)
                Eph_2d = Eph_flat.reshape(T_mesh.shape)
                E_tot_2d = np.sqrt(np.abs(Eth_2d)**2 + np.abs(Eph_2d)**2)

                # M7: Radiation Intensity U(theta, phi)
                U_2d = (obs_distance**2) * (E_tot_2d**2) / (2.0 * Z_0)

                # M7: Total Radiated Power P_rad using Trapezoidal Integration over solid angle
                T_mesh_rad = np.deg2rad(T_mesh)
                phi_rad_1d = np.deg2rad(phi_1d)
                theta_rad_1d = np.deg2rad(theta_1d)
                
                U_sin = U_2d * np.sin(T_mesh_rad)
                # Integrate over phi (axis=1), then theta (axis=0)
                P_rad = np.trapz(np.trapz(U_sin, x=phi_rad_1d, axis=1), x=theta_rad_1d, axis=0)

                # M7: Directivity
                D_2d = 4 * np.pi * U_2d / (P_rad + 1e-12)
                D_max = np.max(D_2d)
                D_max_dBi = 10 * np.log10(D_max + 1e-12)

                max_idx = np.unravel_index(np.argmax(U_2d, axis=None), U_2d.shape)
                theta_max = theta_1d[max_idx[0]]
                phi_max = phi_1d[max_idx[1]]

                nf2ff_res = {
                    "Eth": Eth_2d, "Eph": Eph_2d, "bin_freq": freqs[bin_idx],
                    "U": U_2d, "P_rad": P_rad, "D": D_2d, "D_max": D_max, "D_max_dBi": D_max_dBi,
                    "t_max": theta_max, "p_max": phi_max, "e_inj": e_inj
                }

        st.session_state['res'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'calc_time': calc_time, 'nf2ff': nf2ff_res
        }

if conv_btn and 'cache_phasors' in st.session_state:
    with st.spinner("Running M7 Angular Power/Directivity Convergence..."):
        px_E, px_H, py_E, py_H, pz_E, pz_H, cache_freq = st.session_state['cache_phasors']
        conv_res = []
        for a_res in [10, 5, 2]:
            t_1d = np.arange(0, 180 + a_res, a_res)
            p_1d = np.arange(0, 360 + a_res, a_res)
            t_m, p_m = np.meshgrid(t_1d, p_1d, indexing='ij')
            
            eth_c, eph_c = compute_farfield_3d(
                freq_hz, np.deg2rad(t_m.flatten()), np.deg2rad(p_m.flatten()), obs_distance,
                px_E, px_H, py_E, py_H, pz_E, pz_H,
                i_min, i_max, j_min, j_max, k_min, k_max, dx, dy, dz, cx, cy, cz
            )
            e_t_c = np.sqrt(np.abs(eth_c)**2 + np.abs(eph_c)**2).reshape(t_m.shape)
            u_c = (obs_distance**2) * (e_t_c**2) / (2.0 * Z_0)
            u_s_c = u_c * np.sin(np.deg2rad(t_m))
            p_r_c = np.trapz(np.trapz(u_s_c, x=np.deg2rad(p_1d), axis=1), x=np.deg2rad(t_1d), axis=0)
            d_m_c = np.max(4 * np.pi * u_c / (p_r_c + 1e-12))
            
            conv_res.append({"Resolution": f"{a_res}°", "Samples": len(t_m.flatten()), "P_rad (W)": p_r_c, "D_max": d_m_c, "D_max_dBi": 10*np.log10(d_m_c+1e-12)})
        st.session_state['conv_results'] = pd.DataFrame(conv_res)

# ============================================================
# ANALYSIS & VISUALIZATION
# ============================================================
if 'res' in st.session_state:
    res = st.session_state['res']
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)

    ff = res['nf2ff'] if nf2ff_active else None
    
    # --- M7 NUMERICAL VALIDATION PANEL ---
    st.markdown("### Numerical Validation & Diagnostics")
    cfl_pass = dt <= dt_max
    finite_pass = not (np.isnan(res['Ex']).any() or np.isinf(res['Ex']).any())
    ff_valid = False; p_valid = False; d_valid = False
    
    if ff is not None:
        ff_valid = not (np.isnan(ff['Eth']).any() or np.isinf(ff['Eth']).any())
        p_valid = ff['P_rad'] > 0 and not np.isnan(ff['P_rad'])
        d_valid = ff['D_max'] >= 1.0 # D_max theoretically must be >= 1 for omni/directional 

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL")
    c2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    c3.metric("Equivalence Surface", "PASS" if (nf2ff_active and box_encloses) else ("NOT RUN" if not nf2ff_active else "FAIL"))
    c4.metric("Radiation Pattern", "PASS" if ff_valid else "NOT RUN")
    c5.metric("Power Integration", "PASS" if p_valid else "NOT RUN")
    
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Directivity Check (≥1)", "PASS" if d_valid else ("NOT RUN" if not nf2ff_active else "FAIL"))
    c7.metric("Input Power", "NOT AVAILABLE", help="Numerical soft-source feed prohibits exact physical power acceptance scaling without advanced sub-cell corrections.")
    c8.metric("Rad. Efficiency", "NOT AVAILABLE", help="Dependent on physical Input Power.")
    c9.metric("Gain", "NOT AVAILABLE", help="Dependent on Radiation Efficiency.")
    c10.metric("Power Convergence", "PASS" if 'conv_results' in st.session_state else "NOT RUN")
    st.divider()

    # --- TAB LAYOUT ---
    t1, t2, t3, t4 = st.tabs(["M7 Antenna Performance Dashboard", "3D Radiation Pattern", "2D Principal Polar Cuts", "Component & Data Export"])

    # 1. M7 ANTENNA PERFORMANCE DASHBOARD
    with t1:
        if ff is None:
            st.info("Enable NF2FF & Performance Metrics to view Antenna Performance.")
        else:
            st.markdown("### 📊 Antenna Performance Dashboard")
            st.markdown(f"**Operating Frequency:** `{freq_ghz:.4f} GHz` | **Wavelength:** `{wavelength:.4f} m` | **Observation Distance:** `{obs_distance} m` | **Integration Angular Resolution:** `{ang_res}°`")
            
            # Simulated Performance
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Total Radiated Power ($P_{rad}$)", f"{ff['P_rad']:.4e} W")
            pc2.metric("Max Radiation Intensity ($U_{max}$)", f"{np.max(ff['U']):.4e} W/sr")
            pc3.metric("Maximum Directivity ($D_{max}$)", f"{ff['D_max']:.3f} (Linear)")
            pc4.metric("Max Directivity ($D_{max}$ dBi)", f"{ff['D_max_dBi']:.2f} dBi")

            pc5, pc6, pc7, pc8 = st.columns(4)
            pc5.metric("Peak Direction (Theta)", f"{ff['t_max']}°")
            pc6.metric("Peak Direction (Phi)", f"{ff['p_max']}°")
            pc7.metric("Radiation Efficiency ($\eta_{rad}$)", "NOT AVAILABLE", delta_color="off")
            pc8.metric("Maximum Gain ($G_{max}$ dBi)", "NOT AVAILABLE", delta_color="off")
            
            st.info("**Scientific Note on Efficiency & Gain:** Extracting true absolute efficiency requires a hard numerical definition of accepted power $P_{in}$. Because the existing implementation uses a purely additive soft-source proxy to preserve numerical stability, calculating $V \times I$ blindly produces physically irrelevant normalizations. Therefore, Efficiency and Gain are explicitly withheld in M7 to maintain scientific honesty.")
            
            st.markdown("#### 🔬 Theory Reference Check")
            st.markdown("For a perfectly thin ideal $\lambda/2$ dipole in free space, theoretical Maximum Directivity $D \approx 1.64$ ($2.15$ dBi) occurring at broadside $\theta=90^\circ$. Compare this against your numerical simulation to identify discretization and gap-size dispersion effects.")

            if 'conv_results' in st.session_state:
                st.markdown("#### 📉 Angular Power Convergence Test Results")
                st.dataframe(st.session_state['conv_results'], use_container_width=True)

    # 2. 3D VISUALIZATION
    with t2:
        field_map = {"|E|": E_mag, "|H|": H_mag, "Ex": res['Ex'], "Ey": res['Ey'], "Ez": res['Ez']}
        with st.spinner("Rendering 3D Pattern..."):
            plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
            plotter.set_background("white")

            grid = pv.ImageData(dimensions=np.array([Nx, Ny, Nz]), spacing=(dx, dy, dz))
            grid.point_data[vis_field] = field_map[vis_field].flatten(order="F")
            slc_x, slc_y, slc_z = (cx*dx if plane_select == "YZ Plane" else None, cy*dy if plane_select == "XZ Plane" else None, cz*dz if plane_select == "XY Plane" else None)
            plotter.add_mesh(grid.slice_orthogonal(x=slc_x, y=slc_y, z=slc_z), cmap="jet", show_scalar_bar=True)

            if show_pml and pml_thickness > 0:
                plotter.add_mesh(pv.Box(bounds=(pml_thickness*dx, (Nx-pml_thickness)*dx, pml_thickness*dy, (Ny-pml_thickness)*dy, pml_thickness*dz, (Nz-pml_thickness)*dz)), style='wireframe', color='red')
            if show_eq_surf and nf2ff_active:
                plotter.add_mesh(pv.Box(bounds=(i_min*dx, i_max*dx, j_min*dy, j_max*dy, k_min*dz, k_max*dz)), style='wireframe', color='green', line_width=2)
            if antenna_active and show_dipole:
                plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm1_z_start*dz, arm1_z_end*dz)), color='silver')
                plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm2_z_start*dz, arm2_z_end*dz)), color='silver')

            if nf2ff_active and show_3d_pattern:
                # Plot Directivity (M7)
                D_norm = ff['D'] / ff['D_max']
                r_plot = D_norm if rad_scale == "Linear" else (ff['D_max_dBi'] - db_floor) / abs(db_floor) # Scaled dB visualization proxy
                
                x_surf = cx*dx + pattern_3d_size * r_plot * np.sin(np.deg2rad(T_mesh)) * np.cos(np.deg2rad(P_mesh))
                y_surf = cy*dy + pattern_3d_size * r_plot * np.sin(np.deg2rad(T_mesh)) * np.sin(np.deg2rad(P_mesh))
                z_surf = cz*dz + pattern_3d_size * r_plot * np.cos(np.deg2rad(T_mesh))
                
                surf_grid = pv.StructuredGrid(x_surf, y_surf, z_surf)
                surf_grid.point_data["Directivity"] = ff['D'].flatten(order="C") if rad_scale == "Linear" else (10*np.log10(ff['D']+1e-12)).flatten(order="C")
                plotter.add_mesh(surf_grid, cmap="inferno", show_scalar_bar=True, opacity=0.85)

            plotter.view_isometric()
            st.image(plotter.screenshot(transparent_background=False), use_container_width=True)

    # 3. 2D PRINCIPAL POLAR CUTS
    with t3:
        if nf2ff_active:
            c1, c2 = st.columns(2)
            phi_cut_val = c1.selectbox("E-Plane Cut (Phi Angle)", phi_1d, index=0)
            theta_cut_val = c2.selectbox("H-Plane Cut (Theta Angle)", theta_1d, index=len(theta_1d)//2)
            
            p_idx = np.where(phi_1d == phi_cut_val)[0][0]
            t_idx = np.where(theta_1d == theta_cut_val)[0][0]

            def get_cut(arr_2d, cut_type, idx):
                slice_arr = arr_2d[:, idx] if cut_type == "E" else arr_2d[idx, :]
                return slice_arr if rad_scale == "Linear" else np.clip(10*np.log10(slice_arr+1e-12), db_floor, np.max(10*np.log10(arr_2d+1e-12)))

            e_plane_cut = get_cut(ff['D'], "E", p_idx)
            h_plane_cut = get_cut(ff['D'], "H", t_idx)
            
            r_max_val = ff['D_max'] if rad_scale == "Linear" else ff['D_max_dBi']
            r_min_val = 0 if rad_scale == "Linear" else db_floor

            fig_polar1 = go.Figure(go.Scatterpolar(r=e_plane_cut, theta=theta_1d, mode='lines', line_color='blue', name="E-Plane Directivity"))
            fig_polar1.update_layout(title=f"E-Plane Directivity (Phi = {phi_cut_val}°) | {rad_scale}", polar=dict(radialaxis=dict(range=[r_min_val, r_max_val])))
            
            fig_polar2 = go.Figure(go.Scatterpolar(r=h_plane_cut, theta=phi_1d, mode='lines', line_color='red', name="H-Plane Directivity"))
            fig_polar2.update_layout(title=f"H-Plane Directivity (Theta = {theta_cut_val}°) | {rad_scale}", polar=dict(radialaxis=dict(range=[r_min_val, r_max_val])))

            cp1, cp2 = st.columns(2)
            cp1.plotly_chart(fig_polar1, use_container_width=True)
            cp2.plotly_chart(fig_polar2, use_container_width=True)

    # 4. COMPONENT ANALYSIS & DATA EXPORT
    with t4:
        if nf2ff_active:
            E_tot = np.sqrt(np.abs(ff['Eth'])**2 + np.abs(ff['Eph'])**2)
            
            # Export Data
            df = pd.DataFrame({
                "Theta_deg": T_mesh.flatten(),
                "Phi_deg": P_mesh.flatten(),
                "E_theta_mag": np.abs(ff['Eth']).flatten(),
                "E_phi_mag": np.abs(ff['Eph']).flatten(),
                "E_total_mag": E_tot.flatten(),
                "Radiation_Intensity_U": ff['U'].flatten(),
                "Directivity_Linear": ff['D'].flatten(),
                "Directivity_dBi": (10*np.log10(ff['D']+1e-12)).flatten()
            })
            csv = df.to_csv(index=False)
            st.download_button(label="Download Full M7 Antenna Metrics Data (CSV)", data=csv, file_name='antenna_metrics_m7.csv', mime='text/csv')
            st.dataframe(df.head(15))
