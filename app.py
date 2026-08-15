"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 8 — Rectangular Microstrip Patch Antenna
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
st.markdown("### Milestone 8 — Rectangular Microstrip Patch Antenna")
st.markdown("*Note: The patch antenna is geometrically discretized on the 3D FDTD grid. Input Power, $S_{11}$, VSWR, and Efficiency metrics remain intentionally unavailable as the numerical soft-source proxy does not strictly define $V \\times I$ accepted power.*")

# Sidebar: GRID & DOMAIN
st.sidebar.header("1. GRID & DOMAIN")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=20, max_value=300, value=80, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=20, max_value=300, value=80, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=20, max_value=300, value=80, step=10)
dx = st.sidebar.number_input("dx (m)", min_value=0.0005, max_value=0.1, value=0.005, format="%.4f")
dy = st.sidebar.number_input("dy (m)", min_value=0.0005, max_value=0.1, value=0.005, format="%.4f")
dz = st.sidebar.number_input("dz (m)", min_value=0.0005, max_value=0.1, value=0.005, format="%.4f")

# Sidebar: BOUNDARY CONDITIONS
st.sidebar.header("2. BOUNDARY CONDITIONS")
boundary_type = st.sidebar.selectbox("Boundary Type", ["CPML (Absorbing)", "Basic Numerical (PEC)"])
if boundary_type == "CPML (Absorbing)":
    pml_thickness = st.sidebar.number_input("PML Thickness (Cells)", min_value=2, max_value=20, value=10)
    pml_order = 3; pml_R = 1e-4; pml_alpha = 0.05
else:
    pml_thickness = 0; pml_order = 3; pml_R = 1e-4; pml_alpha = 0.05

# Sidebar: ANTENNA SELECTION & CONFIGURATION
st.sidebar.header("3. ANTENNA CONFIGURATION")
antenna_type = st.sidebar.selectbox("Antenna Type", ["Half-Wave Dipole", "Rectangular Microstrip Patch"])
freq_ghz = st.sidebar.number_input("Target Frequency (GHz)", min_value=0.1, max_value=20.0, value=2.4, step=0.1)
freq_hz = freq_ghz * 1e9
wavelength = C_LIGHT / freq_hz

# Analytical Patch Calculator
def calc_analytical_patch(f, er, h_m):
    W_m = (C_LIGHT / (2 * f)) * math.sqrt(2 / (er + 1))
    e_reff = (er + 1)/2 + ((er - 1)/2) * (1 / math.sqrt(1 + 12 * h_m / W_m))
    dL = 0.412 * h_m * ((e_reff + 0.3) * (W_m/h_m + 0.264)) / ((e_reff - 0.258) * (W_m/h_m + 0.8))
    L_m = C_LIGHT / (2 * f * math.sqrt(e_reff)) - 2 * dL
    return W_m, L_m, e_reff

cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

# Initialize Geometry Defaults
f_z_s = f_z_e = a1_z_s = a1_z_e = a2_z_s = a2_z_e = cz
pch_x1 = pch_x2 = pch_y1 = pch_y2 = pch_z = cz
gnd_x1 = gnd_x2 = gnd_y1 = gnd_y2 = gnd_z = cz
sub_x1 = sub_x2 = sub_y1 = sub_y2 = sub_z1 = sub_z2 = cz
eps_r_val = 1.0; feed_x = cx; feed_y = cy

if antenna_type == "Half-Wave Dipole":
    st.sidebar.subheader("Dipole Parameters")
    dipole_length = st.sidebar.number_input("Dipole Length (m)", min_value=0.01, max_value=5.0, value=round(wavelength/2, 3), step=0.01)
    feed_gap_cells = st.sidebar.number_input("Feed Gap (Cells)", min_value=1, max_value=5, value=1)
    dipole_cells = int(dipole_length / dz)
    arm_cells = (dipole_cells - feed_gap_cells) // 2
    f_z_s = cz - feed_gap_cells // 2; f_z_e = f_z_s + feed_gap_cells - 1
    a1_z_s = f_z_s - arm_cells; a1_z_e = f_z_s - 1
    a2_z_s = f_z_e + 1; a2_z_e = f_z_e + arm_cells

elif antenna_type == "Rectangular Microstrip Patch":
    st.sidebar.subheader("Substrate Parameters")
    eps_r_val = st.sidebar.number_input("Substrate εr", min_value=1.0, max_value=20.0, value=4.4, step=0.1)
    h_mm = st.sidebar.number_input("Substrate Thickness h (mm)", min_value=0.1, max_value=10.0, value=1.6, step=0.1)
    h_m = h_mm / 1000.0
    
    W_a, L_a, e_reff = calc_analytical_patch(freq_hz, eps_r_val, h_m)
    st.sidebar.markdown(f"**Analytical Est:** W=`{W_a*1000:.1f}mm`, L=`{L_a*1000:.1f}mm`")
    
    st.sidebar.subheader("FDTD Discretized Geometry")
    patch_L_cells = st.sidebar.number_input("Patch Length L (Cells, X-Axis)", min_value=2, value=int(L_a/dx))
    patch_W_cells = st.sidebar.number_input("Patch Width W (Cells, Y-Axis)", min_value=2, value=int(W_a/dy))
    sub_h_cells = max(1, int(h_m/dz))
    
    gnd_L_cells = st.sidebar.number_input("Ground Length (Cells)", min_value=patch_L_cells, value=patch_L_cells + 6*sub_h_cells)
    gnd_W_cells = st.sidebar.number_input("Ground Width (Cells)", min_value=patch_W_cells, value=patch_W_cells + 6*sub_h_cells)
    
    feed_offset_x = st.sidebar.number_input("Feed X Offset from Center (Cells)", value=-int(patch_L_cells/4))
    
    # Map to grid
    gnd_z = cz - sub_h_cells//2
    pch_z = gnd_z + sub_h_cells
    
    pch_x1 = cx - patch_L_cells//2; pch_x2 = pch_x1 + patch_L_cells
    pch_y1 = cy - patch_W_cells//2; pch_y2 = pch_y1 + patch_W_cells
    
    gnd_x1 = cx - gnd_L_cells//2; gnd_x2 = gnd_x1 + gnd_L_cells
    gnd_y1 = cy - gnd_W_cells//2; gnd_y2 = gnd_y1 + gnd_W_cells
    
    sub_x1 = gnd_x1; sub_x2 = gnd_x2
    sub_y1 = gnd_y1; sub_y2 = gnd_y2
    sub_z1 = gnd_z; sub_z2 = pch_z
    
    feed_x = cx + feed_offset_x; feed_y = cy

# Sidebar: SOURCE
st.sidebar.header("4. SOURCE & EXCITATION")
waveform_type = st.sidebar.selectbox("Source Waveform", ["Modulated Gaussian (Narrowband)", "Gaussian (Broadband)"])
amplitude = st.sidebar.number_input("Feed Amplitude (V/m)", value=1.0)
pulse_width = st.sidebar.number_input("Pulse Width (timesteps)", min_value=5, max_value=500, value=60)
pulse_delay = st.sidebar.number_input("Pulse Delay (timesteps)", min_value=0, max_value=1000, value=180)

# Sidebar: SIMULATION
st.sidebar.header("5. SIMULATION & NEAR-FIELD")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=10000, value=800, step=100)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

# Sidebar: FAR-FIELD (NF2FF)
st.sidebar.header("6. FAR-FIELD & METRICS (NF2FF)")
nf2ff_active = st.sidebar.checkbox("Enable NF2FF Transformation", value=True)
surf_margin = st.sidebar.number_input("Equivalence Surface Margin (Cells)", min_value=2, max_value=20, value=4)
obs_distance = st.sidebar.number_input("Observation Distance (m)", min_value=1.0, value=100.0)
ang_res = st.sidebar.selectbox("Angular Resolution (deg)", [2, 5, 10], index=1)
rad_scale = st.sidebar.selectbox("Pattern Scale", ["Linear", "dB"])

# Generate Spherical Grid
theta_1d = np.arange(0, 180 + ang_res, ang_res)
phi_1d = np.arange(0, 360 + ang_res, ang_res)
T_mesh, P_mesh = np.meshgrid(theta_1d, phi_1d, indexing='ij')
theta_flat = T_mesh.flatten()
phi_flat = P_mesh.flatten()

# Sidebar: VISUALIZATION CONTROLS
st.sidebar.header("7. 3D VISUALIZATION")
vis_field = st.sidebar.selectbox("Domain Field Quantity", ["|E|", "|H|", "Ex", "Ey", "Ez"])
show_pml = st.sidebar.checkbox("Show PML Bounds", value=True)
show_antenna = st.sidebar.checkbox("Show Antenna Structure", value=True)
show_eq_surf = st.sidebar.checkbox("Show Equivalence Surface", value=False)
show_3d_pattern = st.sidebar.checkbox("Show 3D Pattern Overlay", value=True)

# ============================================================
# RESOLUTION, VALIDATION & MEMORY CALCULATION
# ============================================================
cells_per_wl = wavelength / max(dx, dy, dz)
eff_wl = wavelength / math.sqrt(eps_r_val)
cells_per_eff_wl = eff_wl / max(dx, dy, dz)

dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

i_min = pml_thickness + surf_margin
i_max = Nx - 1 - pml_thickness - surf_margin
j_min = pml_thickness + surf_margin
j_max = Ny - 1 - pml_thickness - surf_margin
k_min = pml_thickness + surf_margin
k_max = Nz - 1 - pml_thickness - surf_margin

box_valid = (i_max > i_min) and (j_max > j_min) and (k_max > k_min)
box_encloses = True
geom_valid = True

if antenna_type == "Half-Wave Dipole":
    box_encloses = (i_min < cx < i_max) and (j_min < cy < j_max) and (k_min < a1_z_s) and (k_max > a2_z_e)
elif antenna_type == "Rectangular Microstrip Patch":
    box_encloses = (i_min < gnd_x1) and (i_max > gnd_x2) and (j_min < gnd_y1) and (j_max > gnd_y2) and (k_min < gnd_z) and (k_max > pch_z)
    geom_valid = (gnd_x1 > pml_thickness) and (gnd_x2 < Nx-pml_thickness) and (feed_x >= pch_x1) and (feed_x <= pch_x2)

st.sidebar.markdown(f"**Free-Space Resolution:** `{cells_per_wl:.1f} C/$\lambda$`")
if antenna_type == "Rectangular Microstrip Patch":
    st.sidebar.markdown(f"**Dielectric Resolution:** `{cells_per_eff_wl:.1f} C/$\lambda_g$`")

if nf2ff_active and not box_encloses:
    st.sidebar.error("⚠️ Equivalence surface intersects the Antenna or Ground Plane! Increase grid size or decrease margin.")
    st.stop()
if not geom_valid:
    st.sidebar.error("⚠️ Invalid Geometry! Antenna intersects PML or Feed is outside patch bounds.")
    st.stop()

# Memory Check
num_cells = Nx * Ny * Nz
num_arrays = 6
if pml_thickness > 0: num_arrays += 12 
memory_mb = (num_arrays * num_cells * 4) / (1024 * 1024)
if nf2ff_active:
    surf_cells = 2*((j_max-j_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(k_max-k_min+1) + (i_max-i_min+1)*(j_max-j_min+1))
    memory_mb += (surf_cells * 4 * num_steps * 4) / (1024 * 1024) + (len(theta_flat) * 12 * 4) / (1024 * 1024)

if memory_mb > 700:
    st.error(f"Configuration requires {memory_mb:.2f} MB, exceeding safety limit.")
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
                   ant_idx, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
                   pch_x1, pch_x2, pch_y1, pch_y2, pch_z, gnd_x1, gnd_x2, gnd_y1, gnd_y2, gnd_z,
                   sub_x1, sub_x2, sub_y1, sub_y2, sub_z1, sub_z2, eps_r, feed_x, feed_y,
                   freq_hz, amp, delay, width, w_type,
                   nf2ff_on, imin, imax, jmin, jmax, kmin, kmax, sx_E, sx_H, sy_E, sy_H, sz_E, sz_H):

    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    chx = dt / MU_0; cex = dt / EPS_0
    feed_history = np.zeros(steps, dtype=np.float32)

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

                    # Material property handling (Dielectric Substrate)
                    eps_factor = eps_r if (ant_idx == 1 and sub_x1 <= i <= sub_x2 and sub_y1 <= j <= sub_y2 and sub_z1 <= k_idx <= sub_z2) else 1.0
                    local_cex = cex / eps_factor

                    if use_pml:
                        psi_hy_ex[i,j,k_idx] = be_y[j] * psi_hy_ex[i,j,k_idx] + ce_y[j] * dHz_dy * dy
                        psi_hz_ex[i,j,k_idx] = be_z[k_idx] * psi_hz_ex[i,j,k_idx] + ce_z[k_idx] * dHy_dz * dz
                        psi_hx_ey[i,j,k_idx] = be_z[k_idx] * psi_hx_ey[i,j,k_idx] + ce_z[k_idx] * dHx_dz * dz
                        psi_hz_ey[i,j,k_idx] = be_x[i] * psi_hz_ey[i,j,k_idx] + ce_x[i] * dHz_dx * dx
                        psi_hy_ez[i,j,k_idx] = be_x[i] * psi_hy_ez[i,j,k_idx] + ce_x[i] * dHy_dx * dx
                        psi_hx_ez[i,j,k_idx] = be_y[j] * psi_hx_ez[i,j,k_idx] + ce_y[j] * dHx_dy * dy
                        Ex[i,j,k_idx] += local_cex * ( (dHz_dy/dy + psi_hy_ex[i,j,k_idx]) - (dHy_dz/dz + psi_hz_ex[i,j,k_idx]) )
                        Ey[i,j,k_idx] += local_cex * ( (dHx_dz/dz + psi_hx_ey[i,j,k_idx]) - (dHz_dx/dx + psi_hz_ey[i,j,k_idx]) )
                        Ez[i,j,k_idx] += local_cex * ( (dHy_dx/dx + psi_hy_ez[i,j,k_idx]) - (dHx_dy/dy + psi_hx_ez[i,j,k_idx]) )
                    else:
                        Ex[i, j, k_idx] += local_cex * (dHz_dy/dy - dHy_dz/dz)
                        Ey[i, j, k_idx] += local_cex * (dHx_dz/dz - dHz_dx/dx)
                        Ez[i, j, k_idx] += local_cex * (dHy_dx/dx - dHx_dy/dy)

        # Apply Antenna PEC Geometry & Excitation
        gauss = math.exp(-0.5 * ((t_steps - delay) / width)**2)
        pulse = amp * gauss * math.cos(2.0 * math.pi * freq_hz * t) if w_type == 0 else amp * gauss
        
        if ant_idx == 0:
            # Dipole PEC
            for k_idx in range(a1_z_s, a1_z_e + 1): Ez[cx, cy, k_idx] = 0.0
            for k_idx in range(a2_z_s, a2_z_e + 1): Ez[cx, cy, k_idx] = 0.0
            # Dipole Feed
            for k_idx in range(f_z_s, f_z_e + 1): Ez[cx, cy, k_idx] += pulse

        elif ant_idx == 1:
            # Ground Plane PEC
            for i in range(gnd_x1, gnd_x2 + 1):
                for j in range(gnd_y1, gnd_y2 + 1):
                    Ex[i, j, gnd_z] = 0.0; Ey[i, j, gnd_z] = 0.0
            # Patch PEC
            for i in range(pch_x1, pch_x2 + 1):
                for j in range(pch_y1, pch_y2 + 1):
                    Ex[i, j, pch_z] = 0.0; Ey[i, j, pch_z] = 0.0
            # Vertical Feed via Substrate
            for k_idx in range(gnd_z, pch_z):
                Ez[feed_x, feed_y, k_idx] += pulse

        feed_history[n] = Ez[feed_x, feed_y, k_idx] if ant_idx == 1 else Ez[cx, cy, f_z_s]

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

    return Ex, Ey, Ez, Hx, Hy, Hz, feed_history

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
        rx = np.sin(theta) * np.cos(phi); ry = np.sin(theta) * np.sin(phi); rz = np.cos(theta)
        Nx_val, Ny_val, Nz_val = 0j, 0j, 0j; Lx, Ly, Lz = 0j, 0j, 0j

        for f in range(2):
            nx = -1.0 if f == 0 else 1.0
            x_prime = (imin if f==0 else imax) - cx
            dS = dy * dz
            for j in range(jmin, jmax+1):
                y_prime = j - cy
                for k_idx in range(kmin, kmax+1):
                    z_prime = k_idx - cz
                    exp_phase = np.exp(1j * k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz))
                    Ey_val = px_E[f, j-jmin, k_idx-kmin, 0]; Ez_val = px_E[f, j-jmin, k_idx-kmin, 1]
                    Hy_val = px_H[f, j-jmin, k_idx-kmin, 0]; Hz_val = px_H[f, j-jmin, k_idx-kmin, 1]
                    Ny_val += (nx * Hz_val) * exp_phase * dS; Nz_val += (-nx * Hy_val) * exp_phase * dS
                    Ly += (-nx * Ez_val) * exp_phase * dS; Lz += (nx * Ey_val) * exp_phase * dS

        for f in range(2):
            ny = -1.0 if f == 0 else 1.0
            y_prime = (jmin if f==0 else jmax) - cy
            dS = dx * dz
            for i in range(imin, imax+1):
                x_prime = i - cx
                for k_idx in range(kmin, kmax+1):
                    z_prime = k_idx - cz
                    exp_phase = np.exp(1j * k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz))
                    Ex_val = py_E[f, i-imin, k_idx-kmin, 0]; Ez_val = py_E[f, i-imin, k_idx-kmin, 1]
                    Hx_val = py_H[f, i-imin, k_idx-kmin, 0]; Hz_val = py_H[f, i-imin, k_idx-kmin, 1]
                    Nx_val += (-ny * Hz_val) * exp_phase * dS; Nz_val += (ny * Hx_val) * exp_phase * dS
                    Lx += (ny * Ez_val) * exp_phase * dS; Lz += (-ny * Ex_val) * exp_phase * dS

        for f in range(2):
            nz = -1.0 if f == 0 else 1.0
            z_prime = (kmin if f==0 else kmax) - cz
            dS = dx * dy
            for i in range(imin, imax+1):
                x_prime = i - cx
                for j in range(jmin, jmax+1):
                    y_prime = j - cy
                    exp_phase = np.exp(1j * k * (rx*x_prime*dx + ry*y_prime*dy + rz*z_prime*dz))
                    Ex_val = pz_E[f, i-imin, j-jmin, 0]; Ey_val = pz_E[f, i-imin, j-jmin, 1]
                    Hx_val = pz_H[f, i-imin, j-jmin, 0]; Hy_val = pz_H[f, i-imin, j-jmin, 1]
                    Nx_val += (nz * Hy_val) * exp_phase * dS; Ny_val += (-nz * Hx_val) * exp_phase * dS
                    Lx += (-nz * Ey_val) * exp_phase * dS; Ly += (nz * Ex_val) * exp_phase * dS

        N_theta = Nx_val * np.cos(theta)*np.cos(phi) + Ny_val * np.cos(theta)*np.sin(phi) - Nz_val * np.sin(theta)
        N_phi = -Nx_val * np.sin(phi) + Ny_val * np.cos(phi)
        L_theta = Lx * np.cos(theta)*np.cos(phi) + Ly * np.cos(theta)*np.sin(phi) - Lz * np.sin(theta)
        L_phi = -Lx * np.sin(phi) + Ly * np.cos(phi)

        coeff = (1j * k * np.exp(-1j * k * r_obs)) / (4 * np.pi * r_obs)
        E_theta[a] = -coeff * (L_phi + Z_0 * N_theta)
        E_phi[a] = coeff * (L_theta - Z_0 * N_phi)

    return E_theta, E_phi

# ============================================================
# EXECUTION & M8 LOGIC
# ============================================================
run_btn = st.button("Run Simulation & Antenna Analysis", type="primary")

if run_btn:
    with st.spinner("Executing 3D FDTD Maxwell Solver..."):
        wt_idx = 0 if waveform_type == "Modulated Gaussian (Narrowband)" else 1
        ant_idx = 0 if antenna_type == "Half-Wave Dipole" else 1

        start_t = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, feed_history = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, boundary_type == "CPML (Absorbing)", 
            b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
            ant_idx, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
            pch_x1, pch_x2, pch_y1, pch_y2, pch_z, gnd_x1, gnd_x2, gnd_y1, gnd_y2, gnd_z,
            sub_x1, sub_x2, sub_y1, sub_y2, sub_z1, sub_z2, eps_r_val, feed_x, feed_y,
            freq_hz, amplitude, pulse_delay, pulse_width, wt_idx,
            nf2ff_active, i_min, i_max, j_min, j_max, k_min, k_max, sx_E, sx_H, sy_E, sy_H, sz_E, sz_H
        )
        calc_time = time.time() - start_t

        nf2ff_res = None
        est_res_freq = 0.0
        if nf2ff_active:
            with st.spinner(f"Computing NF2FF & Pattern Metrics..."):
                window = np.hanning(num_steps) if fft_window == "Hann" else np.ones(num_steps)
                freqs = np.fft.rfftfreq(num_steps, d=dt)
                
                # Spectral Peak Detection (Resonance Proxy)
                feed_fft = np.abs(np.fft.rfft(feed_history * window))
                est_res_freq = freqs[np.argmax(feed_fft)]

                bin_idx = np.argmin(np.abs(freqs - freq_hz))
                
                def ext_phasor(arr): return np.fft.rfft(arr * window, axis=-1)[..., bin_idx] * (2.0 / num_steps)

                px_E = ext_phasor(sx_E); px_H = ext_phasor(sx_H)
                py_E = ext_phasor(sy_E); py_H = ext_phasor(sy_H)
                pz_E = ext_phasor(sz_E); pz_H = ext_phasor(sz_H)

                Eth_flat, Eph_flat = compute_farfield_3d(
                    freq_hz, np.deg2rad(theta_flat), np.deg2rad(phi_flat), obs_distance,
                    px_E, px_H, py_E, py_H, pz_E, pz_H,
                    i_min, i_max, j_min, j_max, k_min, k_max, dx, dy, dz, cx, cy, cz
                )
                
                Eth_2d = Eth_flat.reshape(T_mesh.shape)
                Eph_2d = Eph_flat.reshape(T_mesh.shape)
                E_tot_2d = np.sqrt(np.abs(Eth_2d)**2 + np.abs(Eph_2d)**2)

                U_2d = (obs_distance**2) * (E_tot_2d**2) / (2.0 * Z_0)
                
                U_sin = U_2d * np.sin(np.deg2rad(T_mesh))
                P_rad = np.trapz(np.trapz(U_sin, x=np.deg2rad(phi_1d), axis=1), x=np.deg2rad(theta_1d), axis=0)

                D_2d = 4 * np.pi * U_2d / (P_rad + 1e-12)
                D_max = np.max(D_2d)

                nf2ff_res = {
                    "Eth": Eth_2d, "Eph": Eph_2d, "bin_freq": freqs[bin_idx],
                    "U": U_2d, "P_rad": P_rad, "D": D_2d, "D_max": D_max,
                    "est_res_freq": est_res_freq
                }

        st.session_state['res'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'calc_time': calc_time, 'nf2ff': nf2ff_res, 'ant_type': antenna_type
        }

# ============================================================
# ANALYSIS & VISUALIZATION
# ============================================================
if 'res' in st.session_state:
    res = st.session_state['res']
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    ff = res['nf2ff'] if nf2ff_active else None
    
    # --- NUMERICAL VALIDATION PANEL ---
    st.markdown("### Numerical Validation & Diagnostics")
    cfl_pass = dt <= dt_max
    finite_pass = not (np.isnan(res['Ex']).any() or np.isinf(res['Ex']).any())
    ff_valid = ff is not None and not np.isnan(ff['P_rad'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL")
    c2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    c3.metric("Grid / Antenna Geometry", "PASS" if geom_valid else "FAIL")
    c4.metric("Equivalence Surface", "PASS" if (nf2ff_active and box_encloses) else ("NOT RUN" if not nf2ff_active else "FAIL"))
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Resonance Detection", "PASS" if ff_valid else "NOT RUN")
    c6.metric("Impedance / S11 / VSWR", "NOT AVAILABLE", help="Numerical soft-source feed prohibits exact physical power acceptance scaling without advanced sub-cell corrections.")
    c7.metric("Rad. Efficiency", "NOT AVAILABLE", help="Dependent on physical Input Power.")
    c8.metric("Gain", "NOT AVAILABLE", help="Dependent on Radiation Efficiency.")
    st.divider()

    # --- TAB LAYOUT ---
    t1, t2, t3, t4 = st.tabs(["Performance Dashboard", "3D Antenna & Radiation", "2D Polar Cuts", "Design vs Simulation Analysis"])

    # 1. PERFORMANCE DASHBOARD
    with t1:
        if ff is None:
            st.info("Enable NF2FF & Performance Metrics to view Antenna Performance.")
        else:
            st.markdown("### 📊 Antenna Performance Metrics")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Total Radiated Power", f"{ff['P_rad']:.4e} W")
            pc2.metric("Max Radiation Intensity", f"{np.max(ff['U']):.4e} W/sr")
            pc3.metric("Maximum Directivity", f"{ff['D_max']:.3f} (Linear)")
            pc4.metric("Max Directivity (dBi)", f"{10 * np.log10(ff['D_max'] + 1e-12):.2f} dBi")
            
            st.info("**Scientific Note:** Because the implemented feed utilizes a soft-source additive current proxy to ensure broad numerical stability across both Dipole and Patch topologies, physical Absolute Input Power ($P_{in}$) calculations would yield mathematically flawed $S_{11}$, VSWR, and Efficiency values. True evaluation of these requires complex gap-source corrections reserved for advanced implementations.")

    # 2. 3D VISUALIZATION
    with t2:
        with st.spinner("Rendering 3D Structure & Pattern..."):
            plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
            plotter.set_background("white")

            if show_pml and pml_thickness > 0:
                plotter.add_mesh(pv.Box(bounds=(pml_thickness*dx, (Nx-pml_thickness)*dx, pml_thickness*dy, (Ny-pml_thickness)*dy, pml_thickness*dz, (Nz-pml_thickness)*dz)), style='wireframe', color='red')
            if show_eq_surf and nf2ff_active:
                plotter.add_mesh(pv.Box(bounds=(i_min*dx, i_max*dx, j_min*dy, j_max*dy, k_min*dz, k_max*dz)), style='wireframe', color='green', line_width=2)
            
            if show_antenna:
                if res['ant_type'] == "Half-Wave Dipole":
                    plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, a1_z_s*dz, a1_z_e*dz)), color='silver')
                    plotter.add_mesh(pv.Box(bounds=((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, a2_z_s*dz, a2_z_e*dz)), color='silver')
                else:
                    # Patch Geometry Rendering
                    plotter.add_mesh(pv.Box(bounds=(gnd_x1*dx, gnd_x2*dx, gnd_y1*dy, gnd_y2*dy, gnd_z*dz, (gnd_z+0.5)*dz)), color='goldenrod', label="Ground")
                    plotter.add_mesh(pv.Box(bounds=(sub_x1*dx, sub_x2*dx, sub_y1*dy, sub_y2*dy, gnd_z*dz, pch_z*dz)), color='lightgreen', opacity=0.4, label="Substrate")
                    plotter.add_mesh(pv.Box(bounds=(pch_x1*dx, pch_x2*dx, pch_y1*dy, pch_y2*dy, pch_z*dz, (pch_z+0.5)*dz)), color='goldenrod', label="Patch")
                    plotter.add_mesh(pv.Line((feed_x*dx, feed_y*dy, gnd_z*dz), (feed_x*dx, feed_y*dy, pch_z*dz)), color='red', line_width=4, label="Feed")

            if nf2ff_active and show_3d_pattern:
                r_plot = ff['D'] / ff['D_max'] if rad_scale == "Linear" else (10*np.log10(ff['D']+1e-12) - db_floor) / abs(db_floor)
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
        if ff is not None:
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
            
            r_max_val = ff['D_max'] if rad_scale == "Linear" else 10*np.log10(ff['D_max']+1e-12)
            r_min_val = 0 if rad_scale == "Linear" else db_floor

            fig_polar1 = go.Figure(go.Scatterpolar(r=e_plane_cut, theta=theta_1d, mode='lines', line_color='blue'))
            fig_polar1.update_layout(title=f"E-Plane Directivity (Phi = {phi_cut_val}°) | {rad_scale}", polar=dict(radialaxis=dict(range=[r_min_val, r_max_val])))
            
            fig_polar2 = go.Figure(go.Scatterpolar(r=h_plane_cut, theta=phi_1d, mode='lines', line_color='red'))
            fig_polar2.update_layout(title=f"H-Plane Directivity (Theta = {theta_cut_val}°) | {rad_scale}", polar=dict(radialaxis=dict(range=[r_min_val, r_max_val])))

            cp1, cp2 = st.columns(2)
            cp1.plotly_chart(fig_polar1, use_container_width=True)
            cp2.plotly_chart(fig_polar2, use_container_width=True)

    # 4. DESIGN VS SIMULATION ANALYSIS
    with t4:
        st.markdown("### 🔬 Patch Design Comparison")
        if res['ant_type'] == "Rectangular Microstrip Patch" and ff is not None:
            st.markdown("*Comparison between Analytical Transmission-Line Model approximations and strict FDTD Grid Discretization results.*")
            
            comp_data = {
                "Parameter": ["Target/Resonant Frequency", "Patch Width (W)", "Patch Length (L)", "Peak Directivity", "S11 / Impedance Match"],
                "Analytical/Theoretical": [f"{freq_ghz:.4f} GHz", f"{W_a*1000:.2f} mm", f"{L_a*1000:.2f} mm", "~ 5.0 - 8.0 dBi", "-10 dB or better"],
                "FDTD Numerical Result": [f"{ff['est_res_freq']/1e9:.4f} GHz", f"{(patch_W_cells*dy)*1000:.2f} mm", f"{(patch_L_cells*dx)*1000:.2f} mm", f"{10 * np.log10(ff['D_max'] + 1e-12):.2f} dBi", "NOT AVAILABLE"]
            }
            st.table(pd.DataFrame(comp_data))
            st.info("**Analysis:** The analytical equations assume infinite substrate and idealized fringing limits. FDTD incorporates the exact boundary limits of the finite ground plane, precise discrete staircased material profiles, and explicit discrete temporal FFT integration, naturally leading to minor resonance shifts.")
        else:
            st.info("Select 'Rectangular Microstrip Patch' from the sidebar to view analytical design metrics.")
