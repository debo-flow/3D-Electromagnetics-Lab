"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 4 — Near-Field Electromagnetic Analysis
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
st.markdown("### Milestone 4 — Near-Field Electromagnetic Analysis")
st.markdown("*Note: Near-field data is directly sampled from the numerical FDTD grid. Real-world exact resonance and near-field boundaries may vary due to grid discretization and finite computational domain limits.*")

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

# Sidebar: SIMULATION & NEAR-FIELD ANALYSIS
st.sidebar.header("5. SIMULATION & NEAR-FIELD")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=5000, value=350, step=50)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
calc_time_avg = st.sidebar.checkbox("Calculate Time-Averaged Poynting Vector <S>", value=True)
avg_window = st.sidebar.number_input("Time-Averaging Window (Steps)", min_value=10, max_value=2000, value=100)

# Sidebar: VISUALIZATION
st.sidebar.header("6. VISUALIZATION CONTROLS")
vis_field = st.sidebar.selectbox("3D Field Quantity", ["|E|", "|H|", "Energy Density (u)", "|<S>| (Time-Avg Poynting)", "Ex", "Ey", "Ez"])
plane_select = st.sidebar.selectbox("2D Sampling Plane", ["YZ Plane", "XZ Plane", "XY Plane"])
show_pml = st.sidebar.checkbox("Show PML Region Bounds", value=True)

# ============================================================
# RESOLUTION, STABILITY & MEMORY VALIDATION
# ============================================================
cells_per_wl = wavelength / max(dx, dy, dz)
L_lambda_ratio = dipole_length / wavelength
dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

st.sidebar.markdown(f"**Wavelength ($\lambda$):** `{wavelength:.4f} m`")
st.sidebar.markdown(f"**Resolution:** `{cells_per_wl:.1f} Cells / $\lambda$`")

if antenna_active and cells_per_wl < 10:
    st.sidebar.error("⚠️ Resolution too coarse! Cells/wavelength < 10 causes severe numerical dispersion.")

num_cells = Nx * Ny * Nz
num_arrays = 6 # E/H
if pml_thickness > 0: num_arrays += 12 # CPML
if calc_time_avg: num_arrays += 3 # Sx, Sy, Sz averages
memory_mb = (num_arrays * num_cells * 4) / (1024 * 1024) + 2.0 # +2MB for probes/FFT

st.sidebar.markdown(f"**Estimated Memory:** `{memory_mb:.2f} MB`")
if memory_mb > 500:
    st.error(f"Requested grid requires {memory_mb:.2f} MB of RAM, exceeding the 500 MB limit.")
    st.stop()

# ============================================================
# DIPOLE GEOMETRY & PROBES
# ============================================================
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

if antenna_active:
    feed_z_start = cz - feed_gap_cells // 2
    feed_z_end = feed_z_start + feed_gap_cells - 1
    arm1_z_start = feed_z_start - arm_cells
    arm1_z_end = feed_z_start - 1
    arm2_z_start = feed_z_end + 1
    arm2_z_end = feed_z_end + arm_cells
else:
    feed_z_start = feed_z_end = cz
    arm1_z_start = arm1_z_end = arm2_z_start = arm2_z_end = -1

# Probes Setup: (x, y, z)
# 0: Near Feed, 1: Dipole Axis, 2: Perpendicular, 3: Intermediate Radial
px_arr = np.array([cx + 2, cx, cx + int(wavelength/(4*dx)), cx + int(wavelength/(8*dx))])
py_arr = np.array([cy, cy, cy, cy + int(wavelength/(8*dy))])
pz_arr = np.array([cz, arm2_z_end + 2, cz, cz + int(wavelength/(8*dz))])

# Ensure probes are within bounds
px_arr = np.clip(px_arr, 0, Nx-1); py_arr = np.clip(py_arr, 0, Ny-1); pz_arr = np.clip(pz_arr, 0, Nz-1)

# ============================================================
# CPML CALCULATION
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

        def calc_bc(x):
            if x <= 0: return 0.0, 0.0
            sigma = sigma_max * (x / d_thickness)**m
            alpha = alpha_max * (1.0 - x / d_thickness)**m
            b = math.exp(-(sigma + alpha * EPS_0 / dt) * (dt / EPS_0))
            c = sigma / (sigma + alpha * EPS_0 / dt) * (b - 1.0) / delta
            return b, c

        b_e[i], c_e[i] = calc_bc(dist_e)
        b_h[i], c_h[i] = calc_bc(dist_h)
    return b_e, c_e, b_h, c_h

b_e_x, c_e_x, b_h_x, c_h_x = compute_cpml_1d(Nx, pml_thickness, dx, dt, pml_order, pml_R, pml_alpha)
b_e_y, c_e_y, b_h_y, c_h_y = compute_cpml_1d(Ny, pml_thickness, dy, dt, pml_order, pml_R, pml_alpha)
b_e_z, c_e_z, b_h_z, c_h_z = compute_cpml_1d(Nz, pml_thickness, dz, dt, pml_order, pml_R, pml_alpha)

# ============================================================
# FDTD SOLVER (NUMBA)
# ============================================================
@nb.njit(cache=True)
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, use_pml, 
                   be_x, ce_x, bh_x, ch_x, be_y, ce_y, bh_y, ch_y, be_z, ce_z, bh_z, ch_z,
                   has_antenna, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
                   freq_hz, amp, delay, width, w_type, px, py, pz, calc_avg, avg_start):

    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # Time Averaged Poynting Vector
    Sx_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Sy_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Sz_avg = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # Probe Arrays (4 probes, 6 components)
    num_probes = 4
    probe_E = np.zeros((num_probes, 3, steps), dtype=np.float32) # 0:x, 1:y, 2:z
    probe_H = np.zeros((num_probes, 3, steps), dtype=np.float32)

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
            for k in range(a1_z_s, a1_z_e + 1): Ez[cx, cy, k] = 0.0
            for k in range(a2_z_s, a2_z_e + 1): Ez[cx, cy, k] = 0.0

        # Inject Feed Excitation
        gauss = math.exp(-0.5 * ((t_steps - delay) / width)**2)
        pulse = amp * gauss * math.cos(2.0 * math.pi * freq_hz * t) if w_type == 0 else amp * gauss
        for k in range(f_z_s, f_z_e + 1):
            Ez[cx, cy, k] += pulse

        # Record Probes
        for p in range(num_probes):
            probe_E[p, 0, n] = Ex[px[p], py[p], pz[p]]
            probe_E[p, 1, n] = Ey[px[p], py[p], pz[p]]
            probe_E[p, 2, n] = Ez[px[p], py[p], pz[p]]
            probe_H[p, 0, n] = Hx[px[p], py[p], pz[p]]
            probe_H[p, 1, n] = Hy[px[p], py[p], pz[p]]
            probe_H[p, 2, n] = Hz[px[p], py[p], pz[p]]

        # Time-Average Poynting Vector
        if calc_avg and n >= avg_start:
            avg_count += 1.0
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        Sx_avg[i,j,k] += Ey[i,j,k] * Hz[i,j,k] - Ez[i,j,k] * Hy[i,j,k]
                        Sy_avg[i,j,k] += Ez[i,j,k] * Hx[i,j,k] - Ex[i,j,k] * Hz[i,j,k]
                        Sz_avg[i,j,k] += Ex[i,j,k] * Hy[i,j,k] - Ey[i,j,k] * Hx[i,j,k]

        # Domain Energy (subsampled for speed)
        e_sum = 0.0
        for i in range(0, Nx, 2):
            for j in range(0, Ny, 2):
                for k in range(0, Nz, 2):
                    e_sum += 0.5 * (EPS_0 * (Ex[i,j,k]**2 + Ey[i,j,k]**2 + Ez[i,j,k]**2) + MU_0 * (Hx[i,j,k]**2 + Hy[i,j,k]**2 + Hz[i,j,k]**2))
        energy_hist[n] = e_sum * (dx * dy * dz * 8)

    if calc_avg and avg_count > 0:
        Sx_avg /= avg_count; Sy_avg /= avg_count; Sz_avg /= avg_count

    return Ex, Ey, Ez, Hx, Hy, Hz, probe_E, probe_H, Sx_avg, Sy_avg, Sz_avg, energy_hist

# ============================================================
# EXECUTION
# ============================================================
run_btn = st.button("Run Near-Field FDTD Simulation", type="primary")

if run_btn:
    with st.spinner("Executing Maxwell Solver & Near-Field Sampling..."):
        is_pml = boundary_type == "CPML (Absorbing)"
        wt_idx = 0 if waveform_type == "Modulated Gaussian (Narrowband)" else 1
        avg_start = num_steps - avg_window if calc_time_avg else num_steps

        start_t = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, pE, pH, Sx, Sy, Sz, e_hist = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, is_pml, 
            b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
            antenna_active, cx, cy, feed_z_start, feed_z_end, arm1_z_start, arm1_z_end, arm2_z_start, arm2_z_end,
            freq_hz, amplitude, pulse_delay, pulse_width, wt_idx, px_arr, py_arr, pz_arr, calc_time_avg, avg_start
        )
        calc_time = time.time() - start_t

        # Calculate Energy Density (u) at final step
        u_dens = 0.5 * (EPS_0 * (Ex**2 + Ey**2 + Ez**2) + MU_0 * (Hx**2 + Hy**2 + Hz**2))

        st.session_state['res'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'pE': pE, 'pH': pH, 'Sx': Sx, 'Sy': Sy, 'Sz': Sz, 'u_dens': u_dens,
            'e_hist': e_hist, 'calc_time': calc_time
        }

# ============================================================
# ANALYSIS & VISUALIZATION (TABS)
# ============================================================
if 'res' in st.session_state:
    res = st.session_state['res']
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)
    S_mag = np.sqrt(res['Sx']**2 + res['Sy']**2 + res['Sz']**2) if calc_time_avg else np.zeros_like(E_mag)

    # --- VALIDATION PANEL ---
    st.markdown("### Numerical Validation & Diagnostics")
    cfl_pass = dt <= dt_max
    finite_pass = not (np.isnan(res['Ex']).any() or np.isinf(res['Ex']).any())
    res_pass = cells_per_wl >= 10
    feed_pass = np.max(np.abs(res['pE'][0, 2, :])) > 0 # Ez near feed
    prop_pass = np.max(np.abs(res['pE'][2, 2, :])) > (amplitude * 1e-6) # Perpendicular probe
    s_pass = np.max(S_mag) > 0 if calc_time_avg else False
    
    col_v1, col_v2, col_v3, col_v4, col_v5, col_v6 = st.columns(6)
    col_v1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL")
    col_v2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    col_v3.metric("Grid Resolution", "PASS" if res_pass else "WARN")
    col_v4.metric("Dipole Excitation", "PASS" if feed_pass else "FAIL")
    col_v5.metric("Probe Validity", "PASS" if prop_pass else "FAIL")
    col_v6.metric("Poynting Vector", "PASS" if s_pass else "NOT RUN")
    st.divider()

    # --- TAB LAYOUT ---
    t1, t2, t3, t4, t5 = st.tabs(["3D Near-Field Visualization", "2D Near-Field Maps", "Time-Domain Probes", "Frequency (FFT) Analysis", "Spatial Field Decay"])

    # 1. 3D VISUALIZATION
    with t1:
        st.markdown(f"**Visualizing:** `{vis_field}`")
        field_map = {"|E|": E_mag, "|H|": H_mag, "Energy Density (u)": res['u_dens'], "|<S>| (Time-Avg Poynting)": S_mag, "Ex": res['Ex'], "Ey": res['Ey'], "Ez": res['Ez']}
        plot_data = field_map[vis_field]

        with st.spinner("Rendering 3D Domain..."):
            grid = pv.ImageData()
            grid.dimensions = np.array([Nx, Ny, Nz])
            grid.spacing = (dx, dy, dz)
            grid.point_data[vis_field] = plot_data.flatten(order="F")

            plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
            plotter.set_background("white")
            
            # Auto-slice based on selected plane
            slc_x, slc_y, slc_z = None, None, None
            if plane_select == "YZ Plane": slc_x = cx*dx
            elif plane_select == "XZ Plane": slc_y = cy*dy
            elif plane_select == "XY Plane": slc_z = cz*dz
            slices = grid.slice_orthogonal(x=slc_x, y=slc_y, z=slc_z)
            plotter.add_mesh(slices, cmap="jet", show_scalar_bar=True)

            if show_pml and pml_thickness > 0:
                bnd = (pml_thickness*dx, (Nx-pml_thickness)*dx, pml_thickness*dy, (Ny-pml_thickness)*dy, pml_thickness*dz, (Nz-pml_thickness)*dz)
                plotter.add_mesh(pv.Box(bounds=bnd), style='wireframe', color='red', label="CPML Boundary")

            if antenna_active:
                a1 = ((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm1_z_start*dz, arm1_z_end*dz)
                a2 = ((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm2_z_start*dz, arm2_z_end*dz)
                plotter.add_mesh(pv.Box(bounds=a1), color='silver')
                plotter.add_mesh(pv.Box(bounds=a2), color='silver')

            plotter.view_isometric()
            st.image(plotter.screenshot(transparent_background=False), use_container_width=True)

    # 2. 2D NEAR-FIELD MAPS
    with t2:
        c1, c2 = st.columns([1, 4])
        map_qty = c1.selectbox("Map Quantity", ["|E|", "|H|", "Energy Density", "|<S>|", "Ez", "Hx"])
        map_data_map = {"|E|": E_mag, "|H|": H_mag, "Energy Density": res['u_dens'], "|<S>|": S_mag, "Ez": res['Ez'], "Hx": res['Hx']}
        md = map_data_map[map_qty]

        fig_2d = go.Figure()
        if plane_select == "YZ Plane":
            slice_2d = md[cx, pml_thickness:Ny-pml_thickness, pml_thickness:Nz-pml_thickness].T
            x_ax = np.arange(pml_thickness, Ny-pml_thickness)*dy
            y_ax = np.arange(pml_thickness, Nz-pml_thickness)*dz
            fig_2d.add_trace(go.Heatmap(z=slice_2d, x=x_ax, y=y_ax, colorscale='Jet'))
            fig_2d.update_layout(xaxis_title="Y (m)", yaxis_title="Z (m)")
        elif plane_select == "XZ Plane":
            slice_2d = md[pml_thickness:Nx-pml_thickness, cy, pml_thickness:Nz-pml_thickness].T
            x_ax = np.arange(pml_thickness, Nx-pml_thickness)*dx
            y_ax = np.arange(pml_thickness, Nz-pml_thickness)*dz
            fig_2d.add_trace(go.Heatmap(z=slice_2d, x=x_ax, y=y_ax, colorscale='Jet'))
            fig_2d.update_layout(xaxis_title="X (m)", yaxis_title="Z (m)")
        elif plane_select == "XY Plane":
            slice_2d = md[pml_thickness:Nx-pml_thickness, pml_thickness:Ny-pml_thickness, cz].T
            x_ax = np.arange(pml_thickness, Nx-pml_thickness)*dx
            y_ax = np.arange(pml_thickness, Ny-pml_thickness)*dy
            fig_2d.add_trace(go.Heatmap(z=slice_2d, x=x_ax, y=y_ax, colorscale='Jet'))
            fig_2d.update_layout(xaxis_title="X (m)", yaxis_title="Y (m)")
        
        fig_2d.update_layout(title=f"2D Map of {map_qty} on {plane_select} (Inside PML)", height=600)
        c2.plotly_chart(fig_2d, use_container_width=True)

    # 3. TIME-DOMAIN PROBES
    with t3:
        st.markdown("*Observe the transient field response separating from the later-time quasi-steady state.*")
        time_ns = np.arange(num_steps) * dt * 1e9
        probe_names = ["0: Near Feed", "1: Dipole Axis", "2: Perpendicular", "3: Intermediate Radial"]
        sel_probe = st.selectbox("Select Probe", range(4), format_func=lambda x: probe_names[x])
        
        fig_time = go.Figure()
        Ez_hist = res['pE'][sel_probe, 2, :]
        Hx_hist = res['pH'][sel_probe, 0, :]
        fig_time.add_trace(go.Scatter(x=time_ns, y=Ez_hist, name="Ez (Electric)"))
        fig_time.add_trace(go.Scatter(x=time_ns, y=Hx_hist * Z_0, name="Hx * Z_0 (Magnetic Scaled)", line=dict(dash='dash')))
        fig_time.update_layout(xaxis_title="Time (ns)", yaxis_title="Amplitude", hovermode="x unified")
        st.plotly_chart(fig_time, use_container_width=True)

    # 4. FREQUENCY (FFT) ANALYSIS
    with t4:
        st.markdown("*FFT uses a rectangular window (exact for decaying transients) to compute continuous-time frequency spectrum.*")
        fft_probe = st.selectbox("Select Probe for FFT", range(4), format_func=lambda x: probe_names[x], key="fft")
        
        sig = res['pE'][fft_probe, 2, :] # Analyze Ez
        freqs = np.fft.rfftfreq(num_steps, d=dt) / 1e9 # GHz
        spectrum = np.abs(np.fft.rfft(sig))
        phase = np.angle(np.fft.rfft(sig))

        c_fft1, c_fft2 = st.columns(2)
        fig_fft = go.Figure(go.Scatter(x=freqs, y=spectrum, mode='lines', fill='tozeroy'))
        fig_fft.update_layout(title=f"Frequency Spectrum E_z(f)", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude", xaxis_range=[0, freq_ghz*3])
        c_fft1.plotly_chart(fig_fft, use_container_width=True)

        fig_ph = go.Figure(go.Scatter(x=freqs, y=phase, mode='markers', marker=dict(size=4)))
        fig_ph.update_layout(title=f"Phase Analysis (Radians)", xaxis_title="Frequency (GHz)", yaxis_title="Phase", xaxis_range=[0, freq_ghz*3])
        c_fft2.plotly_chart(fig_ph, use_container_width=True)

    # 5. SPATIAL FIELD DECAY
    with t5:
        st.markdown("*Numerical diagnostic showing reactive and radiating near-field amplitude decay.*")
        decay_axis = st.selectbox("Observation Direction", ["X-Axis (Perpendicular)", "Z-Axis (Along Dipole)"])
        
        fig_decay = go.Figure()
        if decay_axis == "X-Axis (Perpendicular)":
            dist = np.arange(0, Nx - cx - pml_thickness) * dx
            e_decay = E_mag[cx:Nx-pml_thickness, cy, cz]
            fig_decay.add_trace(go.Scatter(x=dist, y=e_decay, mode='lines+markers', name="|E| vs r"))
            fig_decay.update_layout(xaxis_title="Distance from Feed (m)", yaxis_title="|E| Amplitude")
        else:
            dist = np.arange(0, Nz - arm2_z_end - pml_thickness) * dz
            e_decay = E_mag[cx, cy, arm2_z_end:Nz-pml_thickness]
            fig_decay.add_trace(go.Scatter(x=dist, y=e_decay, mode='lines+markers', name="|E| vs z"))
            fig_decay.update_layout(xaxis_title="Distance from Antenna Tip (m)", yaxis_title="|E| Amplitude")

        st.plotly_chart(fig_decay, use_container_width=True)

