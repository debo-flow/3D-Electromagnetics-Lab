"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 3 — Half-Wave Dipole Antenna
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
C_LIGHT = 299792458.0              # Speed of light in vacuum [m/s]
MU_0 = 4.0 * math.pi * 1e-7        # Vacuum permeability [H/m]
EPS_0 = 1.0 / (MU_0 * C_LIGHT**2)  # Vacuum permittivity [F/m]
Z_0 = math.sqrt(MU_0 / EPS_0)      # Vacuum impedance [Ohms]

# ============================================================
# CONFIGURATION & STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="3D EM Laboratory", layout="wide")

st.title("3D Electromagnetics & Antenna Radiation Laboratory")
st.markdown("### Milestone 3 — Half-Wave Dipole Antenna")
st.markdown("*Note: The numerical half-wave dipole model is an FDTD approximation. Real-world physical resonance may vary due to grid discretization, feed gap, and numerical dispersion.*")

# Sidebar: GRID
st.sidebar.header("GRID & DOMAIN")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=10, max_value=200, value=60, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=10, max_value=200, value=60, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=10, max_value=200, value=80, step=10)
dx = st.sidebar.number_input("dx (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dy = st.sidebar.number_input("dy (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dz = st.sidebar.number_input("dz (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")

# Sidebar: BOUNDARY CONDITIONS
st.sidebar.header("BOUNDARY CONDITIONS")
boundary_type = st.sidebar.selectbox("Boundary Type", ["CPML (Absorbing)", "Basic Numerical (PEC)"])

if boundary_type == "CPML (Absorbing)":
    pml_thickness = st.sidebar.number_input("PML Thickness (Cells)", min_value=2, max_value=20, value=10)
    pml_order = st.sidebar.number_input("PML Polynomial Order", min_value=1, max_value=5, value=3)
    pml_R = st.sidebar.selectbox("Target Reflection Coeff", [1e-3, 1e-4, 1e-5, 1e-6], index=1)
    pml_alpha = st.sidebar.number_input("Max Alpha (Conductivity Factor)", min_value=0.0, max_value=0.1, value=0.05, format="%.3f")
else:
    pml_thickness = 0
    pml_order = 3
    pml_R = 1e-4
    pml_alpha = 0.05

if pml_thickness >= Nx//2 or pml_thickness >= Ny//2 or pml_thickness >= Nz//2:
    st.sidebar.error("PML thickness must be strictly less than half the domain size.")
    st.stop()

# Sidebar: ANTENNA CONFIGURATION
st.sidebar.header("ANTENNA CONFIGURATION")
antenna_active = st.sidebar.checkbox("Enable Half-Wave Dipole", value=True)
freq_ghz = st.sidebar.number_input("Operating Frequency (GHz)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
freq_hz = freq_ghz * 1e9
wavelength = C_LIGHT / freq_hz

dipole_length = st.sidebar.number_input("Dipole Length (m)", min_value=0.01, max_value=5.0, value=round(wavelength/2, 3), step=0.01)
feed_gap_cells = st.sidebar.number_input("Feed Gap (Cells)", min_value=1, max_value=5, value=1)
# Grid mapping for antenna
dipole_cells = int(dipole_length / dz)
arm_cells = (dipole_cells - feed_gap_cells) // 2

# Sidebar: SOURCE
st.sidebar.header("SOURCE & EXCITATION")
waveform_type = st.sidebar.selectbox("Source Waveform", ["Modulated Gaussian (Narrowband)", "Gaussian (Broadband)"])
amplitude = st.sidebar.number_input("Feed Amplitude (V/m)", value=1.0)
pulse_width = st.sidebar.number_input("Pulse Width (timesteps)", min_value=5, max_value=500, value=40)
pulse_delay = st.sidebar.number_input("Pulse Delay (timesteps)", min_value=0, max_value=1000, value=120)

# Sidebar: SIMULATION
st.sidebar.header("SIMULATION RUN")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=5000, value=300, step=50)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

# Sidebar: VISUALIZATION
st.sidebar.header("VISUALIZATION")
vis_field = st.sidebar.selectbox("Field to Visualize", ["|E|", "|H|", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
show_pml = st.sidebar.checkbox("Show PML Region Bounds", value=True)
show_dipole = st.sidebar.checkbox("Show Dipole Geometry", value=True)

# ============================================================
# RESOLUTION & STABILITY VALIDATION
# ============================================================
cells_per_wl = wavelength / max(dx, dy, dz)
L_lambda_ratio = dipole_length / wavelength

dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

st.sidebar.markdown(f"**Wavelength ($\lambda$):** `{wavelength:.4f} m`")
st.sidebar.markdown(f"**Dipole Length / $\lambda$:** `{L_lambda_ratio:.3f}`")
st.sidebar.markdown(f"**Resolution:** `{cells_per_wl:.1f} Cells / $\lambda$`")

if antenna_active:
    if cells_per_wl < 10:
        st.sidebar.error("⚠️ Resolution is too coarse! Cells/wavelength < 10 causes severe numerical dispersion.")
    elif cells_per_wl < 20:
        st.sidebar.warning("⚠️ Resolution is marginal. Aim for ≥ 20 cells/wavelength for accurate antenna behaviour.")
    
    if arm_cells <= 0:
        st.sidebar.error("Dipole is too small to be resolved on this grid. Increase length or decrease dz.")
        st.stop()

# Memory Check
num_cells = Nx * Ny * Nz
bytes_per_array = num_cells * 4
num_arrays = 6
if pml_thickness > 0: num_arrays += 12
memory_mb = (num_arrays * bytes_per_array) / (1024 * 1024)
# Add small overhead for probe arrays
memory_mb += 1.0 

if memory_mb > 500:
    st.error(f"Requested grid requires {memory_mb:.2f} MB of RAM, exceeding the 500 MB safety limit.")
    st.stop()

# ============================================================
# DIPOLE GEOMETRY MAPPING
# ============================================================
# Default placement: center of the domain, aligned with Z-axis
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2

if antenna_active:
    # Dipole Indices
    feed_z_start = cz - feed_gap_cells // 2
    feed_z_end = feed_z_start + feed_gap_cells - 1
    
    arm1_z_start = feed_z_start - arm_cells
    arm1_z_end = feed_z_start - 1
    
    arm2_z_start = feed_z_end + 1
    arm2_z_end = feed_z_end + arm_cells
else:
    # Fallback to single cell source if antenna disabled
    feed_z_start = feed_z_end = cz
    arm1_z_start = arm1_z_end = -1
    arm2_z_start = arm2_z_end = -1

# Probes Setup
probes = {
    "Feed Near-Field": (cx + 2, cy, cz),
    "Dipole Axis (Top)": (cx, cy, arm2_z_end + 2),
    "Perpendicular": (cx + int(wavelength/(4*dx)), cy, cz) # lambda/4 away
}

# ============================================================
# CPML CALCULATION
# ============================================================
def compute_cpml_1d(N, d_pml, delta, dt, m, R_err, alpha_max):
    b_e = np.zeros(N, dtype=np.float32)
    c_e = np.zeros(N, dtype=np.float32)
    b_h = np.zeros(N, dtype=np.float32)
    c_h = np.zeros(N, dtype=np.float32)

    if d_pml == 0: return b_e, c_e, b_h, c_h

    d_thickness = d_pml * delta
    sigma_max = - (m + 1) * math.log(R_err) / (2.0 * Z_0 * d_thickness)

    for i in range(N):
        dist_e = 0.0
        if i < d_pml: dist_e = (d_pml - i) * delta
        elif i > N - 1 - d_pml: dist_e = (i - (N - 1 - d_pml)) * delta

        dist_h = 0.0
        if i < d_pml: dist_h = (d_pml - i - 0.5) * delta
        elif i > N - 2 - d_pml: dist_h = (i + 0.5 - (N - 1 - d_pml)) * delta
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
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, 
                   use_pml, be_x, ce_x, bh_x, ch_x, be_y, ce_y, bh_y, ch_y, be_z, ce_z, bh_z, ch_z,
                   has_antenna, cx, cy, f_z_s, f_z_e, a1_z_s, a1_z_e, a2_z_s, a2_z_e,
                   freq_hz, amp, delay, width, w_type, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3):

    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # CPML Arrays
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32); psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # Diagnostics
    feed_sig_hist = np.zeros(steps, dtype=np.float32)
    current_proxy = np.zeros(steps, dtype=np.float32)
    energy_hist = np.zeros(steps, dtype=np.float32)
    probe1_E = np.zeros(steps, dtype=np.float32)
    probe2_E = np.zeros(steps, dtype=np.float32)
    probe3_E = np.zeros(steps, dtype=np.float32)

    chx = dt / MU_0
    cex = dt / EPS_0

    for n in range(steps):
        t = float(n) * dt
        t_steps = float(n)

        # 1. Update H-field
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

        # 2. Update E-field
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

        # 3. Apply Dipole Geometry (PEC Arms)
        if has_antenna:
            for k in range(a1_z_s, a1_z_e + 1):
                Ez[cx, cy, k] = 0.0 # Bottom arm
            for k in range(a2_z_s, a2_z_e + 1):
                Ez[cx, cy, k] = 0.0 # Top arm

        # 4. Inject Center Feed Excitation (Soft Source)
        gauss = math.exp(-0.5 * ((t_steps - delay) / width)**2)
        if w_type == 0:
            pulse = amp * gauss * math.cos(2.0 * math.pi * freq_hz * t) # Modulated
        else:
            pulse = amp * gauss # Baseband
            
        for k in range(f_z_s, f_z_e + 1):
            Ez[cx, cy, k] += pulse

        # 5. Diagnostics
        feed_sig_hist[n] = pulse
        # Current Proxy: Circulating H-field around feed (Ampere's law approx, simplified)
        current_proxy[n] = (Hx[cx, cy, f_z_s] - Hx[cx, cy-1, f_z_s]) - (Hy[cx, cy, f_z_s] - Hy[cx-1, cy, f_z_s])
        
        probe1_E[n] = math.sqrt(Ex[px1,py1,pz1]**2 + Ey[px1,py1,pz1]**2 + Ez[px1,py1,pz1]**2)
        probe2_E[n] = math.sqrt(Ex[px2,py2,pz2]**2 + Ey[px2,py2,pz2]**2 + Ez[px2,py2,pz2]**2)
        if 0 <= px3 < Nx and 0 <= py3 < Ny and 0 <= pz3 < Nz:
            probe3_E[n] = math.sqrt(Ex[px3,py3,pz3]**2 + Ey[px3,py3,pz3]**2 + Ez[px3,py3,pz3]**2)

        e_sum = 0.0
        for i in range(0, Nx, 2):
            for j in range(0, Ny, 2):
                for k in range(0, Nz, 2):
                    e_sum += 0.5 * (EPS_0 * (Ex[i,j,k]**2 + Ey[i,j,k]**2 + Ez[i,j,k]**2) + 
                                    MU_0 * (Hx[i,j,k]**2 + Hy[i,j,k]**2 + Hz[i,j,k]**2))
        energy_hist[n] = e_sum * (dx * dy * dz * 8) 

    return Ex, Ey, Ez, Hx, Hy, Hz, feed_sig_hist, current_proxy, probe1_E, probe2_E, probe3_E, energy_hist

# ============================================================
# EXECUTION
# ============================================================
run_btn = st.button("Run Antenna Simulation", type="primary")

if run_btn:
    with st.spinner("Calculating 3D FDTD Maxwell updates & Antenna Interaction..."):
        is_pml = boundary_type == "CPML (Absorbing)"
        wt_idx = 0 if waveform_type == "Modulated Gaussian (Narrowband)" else 1
        
        p1 = probes["Feed Near-Field"]
        p2 = probes["Dipole Axis (Top)"]
        p3 = probes["Perpendicular"]

        start_t = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, feed_hist, I_hist, p1_hist, p2_hist, p3_hist, e_hist = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, is_pml, 
            b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z,
            antenna_active, cx, cy, feed_z_start, feed_z_end, arm1_z_start, arm1_z_end, arm2_z_start, arm2_z_end,
            freq_hz, amplitude, pulse_delay, pulse_width, wt_idx,
            p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2]
        )
        calc_time = time.time() - start_t

        st.session_state['results'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'feed_hist': feed_hist, 'I_hist': I_hist, 'e_hist': e_hist,
            'p1': p1_hist, 'p2': p2_hist, 'p3': p3_hist,
            'calc_time': calc_time
        }

# ============================================================
# RESULTS & VALIDATION UI
# ============================================================
if 'results' in st.session_state:
    res = st.session_state['results']
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)

    st.markdown("### Numerical Validation & Antenna Sanity Check")

    cfl_pass = dt <= dt_max
    finite_pass = not (np.isnan(res['Ex']).any() or np.isinf(res['Ex']).any())
    prop_pass = np.max(res['p3']) > (amplitude * 1e-5) # Wave reached perpendicular probe
    res_pass = cells_per_wl >= 10
    feed_pass = np.max(np.abs(res['feed_hist'])) > 0
    sym_diff = np.abs(res['Ex'][cx+2, cy, cz] - res['Ex'][cx-2, cy, cz])
    sym_pass = sym_diff < 1e-4 # Check symmetry on X axis

    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
    col_v1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL", delta_color="off" if cfl_pass else "inverse")
    col_v2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    col_v3.metric("Grid Resolution", "PASS" if res_pass else "WARN")
    col_v4.metric("Feed Excitation", "PASS" if feed_pass else "FAIL")
    col_v5.metric("Wave Propagation", "PASS" if prop_pass else "FAIL")

    st.markdown(f"**Diagnostics:** Max Energy = `{np.max(res['e_hist']):.4e} J` | Symmetric Error = `{sym_diff:.2e}` | Solve Time = `{res['calc_time']:,.2f} s`")
    
    st.divider()

    # ============================================================
    # 3D VISUALIZATION
    # ============================================================
    st.markdown(f"### 3D Field & Geometry Visualization: {vis_field}")
    field_map = {"|E|": E_mag, "|H|": H_mag, "Ex": res['Ex'], "Ey": res['Ey'], "Ez": res['Ez'], "Hx": res['Hx'], "Hy": res['Hy'], "Hz": res['Hz']}
    plot_data = field_map[vis_field]

    with st.spinner("Rendering..."):
        grid = pv.ImageData()
        grid.dimensions = np.array([Nx, Ny, Nz])
        grid.spacing = (dx, dy, dz)
        grid.point_data[vis_field] = plot_data.flatten(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
        plotter.set_background("white")

        slices = grid.slice_orthogonal(x=cx*dx, y=cy*dy, z=cz*dz)
        plotter.add_mesh(slices, cmap="jet", show_scalar_bar=True)

        if show_pml and pml_thickness > 0:
            bounds = (pml_thickness*dx, (Nx-pml_thickness)*dx, pml_thickness*dy, (Ny-pml_thickness)*dy, pml_thickness*dz, (Nz-pml_thickness)*dz)
            plotter.add_mesh(pv.Box(bounds=bounds), style='wireframe', color='red', line_width=2, label="CPML Boundary")

        if show_dipole and antenna_active:
            arm1_bounds = ((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm1_z_start*dz, arm1_z_end*dz)
            arm2_bounds = ((cx-0.5)*dx, (cx+0.5)*dx, (cy-0.5)*dy, (cy+0.5)*dy, arm2_z_start*dz, arm2_z_end*dz)
            plotter.add_mesh(pv.Box(bounds=arm1_bounds), color='silver', label="Dipole Arm")
            plotter.add_mesh(pv.Box(bounds=arm2_bounds), color='silver')

        plotter.view_isometric()
        img = plotter.screenshot(transparent_background=False)
        st.image(img, use_container_width=True)

    st.divider()

    # ============================================================
    # 2D TEMPORAL ANALYSIS
    # ============================================================
    st.markdown("### Dipole Feed & Near-Field Temporal Analysis")
    st.markdown("*Observe the transient excitation, the induced structural current proxy, and the radiated field propagating to the probes.*")
    
    time_axis = np.arange(num_steps) * dt * 1e9 # ns

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_axis, y=res['feed_hist'], mode='lines', name='Feed Signal (V/m)'))
    # Scale current proxy purely for visual overlay comparison with voltage feed
    scaled_I = res['I_hist'] * (np.max(np.abs(res['feed_hist'])) / (np.max(np.abs(res['I_hist'])) + 1e-12))
    fig1.add_trace(go.Scatter(x=time_axis, y=scaled_I, mode='lines', name='Current Proxy (Scaled)', line=dict(dash='dash')))
    fig1.update_layout(title="Antenna Feed Diagnostics", xaxis_title="Time (ns)", yaxis_title="Amplitude")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_axis, y=res['p1'], mode='lines', name='Near-Field (Feed)'))
    fig2.add_trace(go.Scatter(x=time_axis, y=res['p3'], mode='lines', name='Perpendicular Field'))
    fig2.update_layout(title="Electromagnetic Field Propagation (|E|)", xaxis_title="Time (ns)", yaxis_title="|E| (V/m)")
    st.plotly_chart(fig2, use_container_width=True)
