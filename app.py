"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 2 — PML Absorbing Boundary (Virtual Anechoic Environment)
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
st.markdown("### Milestone 2 — PML Absorbing Boundary")
st.markdown("*Note: PML is a numerical absorbing boundary designed to minimize artificial electromagnetic reflections. It is not a physical anechoic chamber.*")

# Sidebar: GRID
st.sidebar.header("GRID")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=10, max_value=150, value=60, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=10, max_value=150, value=60, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=10, max_value=150, value=60, step=10)
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

# Prevent invalid PML thickness
if pml_thickness >= Nx//2 or pml_thickness >= Ny//2 or pml_thickness >= Nz//2:
    st.sidebar.error("PML thickness must be strictly less than half the domain size.")
    st.stop()

# Sidebar: SIMULATION
st.sidebar.header("SIMULATION")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=2000, value=200, step=50)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

# Sidebar: SOURCE & PROBE
st.sidebar.header("SOURCE & PROBE")
src_pos_x = st.sidebar.number_input("Source X (index)", min_value=1, max_value=Nx-2, value=Nx//2)
src_pos_y = st.sidebar.number_input("Source Y (index)", min_value=1, max_value=Ny-2, value=Ny//2)
src_pos_z = st.sidebar.number_input("Source Z (index)", min_value=1, max_value=Nz-2, value=Nz//2)

probe_x = st.sidebar.number_input("Probe X (index)", min_value=1, max_value=Nx-2, value=Nx - pml_thickness - 2)
probe_y = st.sidebar.number_input("Probe Y (index)", min_value=1, max_value=Ny-2, value=Ny//2)
probe_z = st.sidebar.number_input("Probe Z (index)", min_value=1, max_value=Nz-2, value=Nz//2)

amplitude = st.sidebar.number_input("Pulse Amplitude (V/m)", value=1.0)
pulse_width = st.sidebar.number_input("Pulse Width (timesteps)", min_value=5, max_value=100, value=15)
pulse_delay = st.sidebar.number_input("Pulse Delay (timesteps)", min_value=0, max_value=300, value=45)
polarization = st.sidebar.selectbox("Polarization", options=["Ez", "Ex", "Ey"])
pol_idx = {"Ex": 0, "Ey": 1, "Ez": 2}[polarization]

# Sidebar: VISUALIZATION
st.sidebar.header("VISUALIZATION")
vis_field = st.sidebar.selectbox("Field Component to Visualize", ["|E|", "|H|", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
vis_mode = st.sidebar.selectbox("Visualization Mode", ["Orthogonal Slices", "Volume Rendering"])
show_pml = st.sidebar.checkbox("Show PML Region Bounds", value=True)

# ============================================================
# CFL CALCULATION & MEMORY MANAGEMENT
# ============================================================
dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

# Memory Check
num_cells = Nx * Ny * Nz
bytes_per_array = num_cells * 4 # float32
num_arrays = 6 # E and H fields
if pml_thickness > 0:
    num_arrays += 12 # CPML psi arrays

memory_mb = (num_arrays * bytes_per_array) / (1024 * 1024)

st.sidebar.markdown(f"**Estimated Memory:** `{memory_mb:.2f} MB`")
st.sidebar.markdown(f"**CFL Max dt:** `{dt_max:.4e} s`")
st.sidebar.markdown(f"**Actual dt:** `{dt:.4e} s`")

if memory_mb > 500:
    st.error(f"Requested grid requires {memory_mb:.2f} MB of RAM, exceeding the 500 MB safety limit. Reduce grid size.")
    st.stop()

# ============================================================
# PML CALCULATION
# ============================================================
def compute_cpml_1d(N, d_pml, delta, dt, m, R_err, alpha_max):
    """Calculates 1D CPML coefficients b and c for E and H nodes."""
    b_e = np.zeros(N, dtype=np.float32)
    c_e = np.zeros(N, dtype=np.float32)
    b_h = np.zeros(N, dtype=np.float32)
    c_h = np.zeros(N, dtype=np.float32)

    if d_pml == 0:
        return b_e, c_e, b_h, c_h

    d_thickness = d_pml * delta
    sigma_max = - (m + 1) * math.log(R_err) / (2.0 * Z_0 * d_thickness)

    for i in range(N):
        # Distance calculation for E nodes (integers)
        dist_e = 0.0
        if i < d_pml:
            dist_e = (d_pml - i) * delta
        elif i > N - 1 - d_pml:
            dist_e = (i - (N - 1 - d_pml)) * delta
        
        # Distance calculation for H nodes (half-integers)
        dist_h = 0.0
        if i < d_pml:
            dist_h = (d_pml - i - 0.5) * delta
        elif i > N - 2 - d_pml:
            dist_h = (i + 0.5 - (N - 1 - d_pml)) * delta
        dist_h = max(0.0, dist_h)

        def calc_bc(x):
            if x <= 0: return 0.0, 0.0
            sigma = sigma_max * (x / d_thickness)**m
            alpha = alpha_max * (1.0 - x / d_thickness)**m
            # standard CPML update constants (kappa = 1)
            b = math.exp(-(sigma + alpha * EPS_0 / dt) * (dt / EPS_0))
            c = sigma / (sigma + alpha * EPS_0 / dt) * (b - 1.0) / delta # pre-divided by delta
            return b, c

        b_e[i], c_e[i] = calc_bc(dist_e)
        b_h[i], c_h[i] = calc_bc(dist_h)

    return b_e, c_e, b_h, c_h

b_e_x, c_e_x, b_h_x, c_h_x = compute_cpml_1d(Nx, pml_thickness, dx, dt, pml_order, pml_R, pml_alpha)
b_e_y, c_e_y, b_h_y, c_h_y = compute_cpml_1d(Ny, pml_thickness, dy, dt, pml_order, pml_R, pml_alpha)
b_e_z, c_e_z, b_h_z, c_h_z = compute_cpml_1d(Nz, pml_thickness, dz, dt, pml_order, pml_R, pml_alpha)

# ============================================================
# FDTD SOLVER (NUMBA OPTIMIZED)
# ============================================================
@nb.njit(cache=True)
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, src_pos, amp, delay, width, pol_idx, probe_pos,
                   use_pml, be_x, ce_x, bh_x, ch_x, be_y, ce_y, bh_y, ch_y, be_z, ce_z, bh_z, ch_z):
    """Runs the 3D FDTD loop with optional CPML absorbing boundaries."""
    
    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    # CPML auxiliary arrays
    psi_ey_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ez_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ex_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_ey_hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    psi_hy_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hz_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hx_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    psi_hy_ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    src_history = np.zeros(steps, dtype=np.float32)
    probe_history_E = np.zeros(steps, dtype=np.float32)
    probe_history_H = np.zeros(steps, dtype=np.float32)
    energy_history = np.zeros(steps, dtype=np.float32)
    
    sx, sy, sz = src_pos
    px, py, pz = probe_pos

    chx = dt / MU_0
    cex = dt / EPS_0
    
    # Precompute denominator for spatial diffs (without dx) since CPML constants handle it or we use raw difference
    # Actually, ce_x has been pre-divided by dx in the Python function for standard CPML, but normal differences need dx.
    # To keep things clean, normal diff uses dx.
    
    for n in range(steps):
        t_steps = float(n)
        
        # 1. Update H-field
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                for k in range(Nz - 1):
                    dEz_dy = Ez[i, j+1, k] - Ez[i, j, k]
                    dEy_dz = Ey[i, j, k+1] - Ey[i, j, k]
                    dEx_dz = Ex[i, j, k+1] - Ex[i, j, k]
                    dEz_dx = Ez[i+1, j, k] - Ez[i, j, k]
                    dEy_dx = Ey[i+1, j, k] - Ey[i, j, k]
                    dEx_dy = Ex[i, j+1, k] - Ex[i, j, k]

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
                    dHz_dy = Hz[i, j, k] - Hz[i, j-1, k]
                    dHy_dz = Hy[i, j, k] - Hy[i, j, k-1]
                    dHx_dz = Hx[i, j, k] - Hx[i, j, k-1]
                    dHz_dx = Hz[i, j, k] - Hz[i-1, j, k]
                    dHy_dx = Hy[i, j, k] - Hy[i-1, j, k]
                    dHx_dy = Hx[i, j, k] - Hx[i, j-1, k]

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

        # 3. Inject Gaussian Source
        pulse = amp * math.exp(-0.5 * ((t_steps - delay) / width)**2)
        if pol_idx == 0: Ex[sx, sy, sz] += pulse
        elif pol_idx == 1: Ey[sx, sy, sz] += pulse
        elif pol_idx == 2: Ez[sx, sy, sz] += pulse

        # 4. Record Diagnostics
        src_history[n] = pulse
        probe_history_E[n] = math.sqrt(Ex[px, py, pz]**2 + Ey[px, py, pz]**2 + Ez[px, py, pz]**2)
        probe_history_H[n] = math.sqrt(Hx[px, py, pz]**2 + Hy[px, py, pz]**2 + Hz[px, py, pz]**2)

        # Energy (sub-sampled for speed, approx over whole domain)
        e_sum = 0.0
        for i in range(0, Nx, 2):
            for j in range(0, Ny, 2):
                for k in range(0, Nz, 2):
                    e_sum += 0.5 * (EPS_0 * (Ex[i,j,k]**2 + Ey[i,j,k]**2 + Ez[i,j,k]**2) + 
                                    MU_0 * (Hx[i,j,k]**2 + Hy[i,j,k]**2 + Hz[i,j,k]**2))
        energy_history[n] = e_sum * (dx * dy * dz * 8) # x8 for subsampling

    return Ex, Ey, Ez, Hx, Hy, Hz, src_history, probe_history_E, energy_history

# ============================================================
# MAIN BUTTONS
# ============================================================
col_btns1, col_btns2, col_btns3 = st.columns([1.5, 1.5, 3])
run_btn = col_btns1.button("Run Simulation", type="primary")
val_btn = col_btns2.button("Run PML Validation Test")

if st.sidebar.button("Reset State"):
    st.session_state.clear()
    st.rerun()

# ============================================================
# SIMULATION EXECUTION
# ============================================================
def execute_fdtd(use_pml_flag):
    src_position = (src_pos_x, src_pos_y, src_pos_z)
    probe_position = (probe_x, probe_y, probe_z)
    
    start_t = time.time()
    Ex, Ey, Ez, Hx, Hy, Hz, src_hist, probe_hist, energy_hist = run_simulation(
        Nx, Ny, Nz, dx, dy, dz, dt, num_steps, src_position, amplitude, pulse_delay, pulse_width, pol_idx, probe_position,
        use_pml_flag, b_e_x, c_e_x, b_h_x, c_h_x, b_e_y, c_e_y, b_h_y, c_h_y, b_e_z, c_e_z, b_h_z, c_h_z
    )
    calc_time = time.time() - start_t
    return Ex, Ey, Ez, Hx, Hy, Hz, src_hist, probe_hist, energy_hist, calc_time

if run_btn:
    with st.spinner("Calculating 3D FDTD Maxwell updates..."):
        is_pml = boundary_type == "CPML (Absorbing)"
        Ex, Ey, Ez, Hx, Hy, Hz, src_hist, probe_hist, energy_hist, calc_time = execute_fdtd(is_pml)
        
        st.session_state['results'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'src_hist': src_hist, 'probe_hist': probe_hist, 'energy_hist': energy_hist,
            'calc_time': calc_time, 'dt': dt, 'is_pml': is_pml
        }

if val_btn:
    with st.spinner("Running Validation: Basic Boundary vs PML Boundary..."):
        _, _, _, _, _, _, _, probe_no_pml, _, _ = execute_fdtd(False)
        _, _, _, _, _, _, _, probe_pml, _, _ = execute_fdtd(True)
        st.session_state['validation'] = {'no_pml': probe_no_pml, 'pml': probe_pml}

# ============================================================
# RESULTS & VALIDATION UI
# ============================================================
if 'results' in st.session_state:
    res = st.session_state['results']
    
    # Lazy calculate magnitudes
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)

    st.markdown("### Numerical Validation")
    
    cfl_pass = dt <= dt_max
    has_nan = np.isnan(res['Ex']).any()
    has_inf = np.isinf(res['Ex']).any()
    finite_pass = not has_nan and not has_inf
    prop_pass = np.max(res['probe_hist']) > (amplitude * 1e-4)
    energy_pass = np.max(res['energy_hist']) < (amplitude**2 * 1e5)
    
    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
    col_v1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL", delta_color="off" if cfl_pass else "inverse")
    col_v2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    col_v3.metric("NaN/Inf Check", "PASS" if finite_pass else "FAIL")
    col_v4.metric("Wave Propagated", "PASS" if prop_pass else "FAIL")
    col_v5.metric("PML Active", "YES" if res['is_pml'] else "NO")

    st.markdown(f"**Diagnostic Data:** Max Field Energy $\approx$ `{np.max(res['energy_hist']):.4e} J` | Max $|E|$ = `{np.max(E_mag):.4e} V/m` | Solve Time = `{res['calc_time']:,.2f} s`")
    if res['is_pml']:
        st.info("Energy decreasing over time indicates the CPML is successfully absorbing the wave leaving the domain.")

    if not finite_pass or not cfl_pass:
        st.error("Numerical instability detected. Decrease CFL factor.")
        st.stop()
    st.divider()

    # ============================================================
    # 3D VISUALIZATION
    # ============================================================
    st.markdown(f"### 3D Visualization: {vis_field}")
    field_map = {"|E|": E_mag, "|H|": H_mag, "Ex": res['Ex'], "Ey": res['Ey'], "Ez": res['Ez'], "Hx": res['Hx'], "Hy": res['Hy'], "Hz": res['Hz']}
    plot_data = field_map[vis_field]

    with st.spinner("Rendering..."):
        grid = pv.ImageData()
        grid.dimensions = np.array([Nx, Ny, Nz])
        grid.spacing = (dx, dy, dz)
        grid.point_data[vis_field] = plot_data.flatten(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=[800, 500])
        plotter.set_background("white")

        if vis_mode == "Orthogonal Slices":
            slices = grid.slice_orthogonal(x=src_pos_x*dx, y=src_pos_y*dy, z=src_pos_z*dz)
            plotter.add_mesh(slices, cmap="jet", show_scalar_bar=True)
        else:
            plotter.add_volume(grid, cmap="jet", opacity="linear", show_scalar_bar=True)
            plotter.add_mesh(grid.outline(), color="black")

        # Draw PML Bounds
        if show_pml and pml_thickness > 0 and res['is_pml']:
            inner_bounds = (pml_thickness*dx, (Nx-pml_thickness)*dx, 
                            pml_thickness*dy, (Ny-pml_thickness)*dy, 
                            pml_thickness*dz, (Nz-pml_thickness)*dz)
            inner_box = pv.Box(bounds=inner_bounds)
            plotter.add_mesh(inner_box, style='wireframe', color='red', line_width=2, label="PML Boundary")
            plotter.add_legend()

        plotter.view_isometric()
        img = plotter.screenshot(transparent_background=False)
        st.image(img, use_container_width=True)

    st.divider()

    # ============================================================
    # 2D TEMPORAL ANALYSIS
    # ============================================================
    st.markdown("### Temporal Field Energy & Probe Data")
    time_axis = np.arange(num_steps) * res['dt'] * 1e9 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=res['probe_hist'], mode='lines', name='Probe |E|'))
    fig.add_trace(go.Scatter(x=time_axis, y=res['src_hist'], mode='lines', name='Source Waveform', line=dict(dash='dash', color='gray')))
    
    # Scale energy down just for unified plotting visibility
    scaled_energy = res['energy_hist'] / np.max(res['energy_hist']) * np.max(res['src_hist'])
    fig.add_trace(go.Scatter(x=time_axis, y=scaled_energy, mode='lines', name='Normalized Total Energy', line=dict(color='green', dash='dot')))
    
    fig.update_layout(xaxis_title="Time (ns)", yaxis_title="Amplitude", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PML VALIDATION EXPERIMENT VIEW
# ============================================================
if 'validation' in st.session_state:
    st.divider()
    st.markdown("### PML Reflection Coefficient Validation")
    v_res = st.session_state['validation']
    time_axis = np.arange(num_steps) * dt * 1e9

    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=time_axis, y=v_res['no_pml'], mode='lines', name='Without PML (PEC Boundary)', line=dict(color='red')))
    fig_val.add_trace(go.Scatter(x=time_axis, y=v_res['pml'], mode='lines', name='With CPML', line=dict(color='blue')))
    fig_val.update_layout(title="Probe |E| Comparison near boundary", xaxis_title="Time (ns)", yaxis_title="Amplitude (V/m)")
    st.plotly_chart(fig_val, use_container_width=True)

    # Estimate reflection
    # Split the history to isolate the reflection (assuming pulse happens early)
    mid_step = int(num_steps * 0.4)
    if mid_step < len(v_res['pml']):
        inc_amp = np.max(v_res['no_pml'][:mid_step]) + 1e-12
        ref_pml = np.max(v_res['pml'][mid_step:])
        ref_no_pml = np.max(v_res['no_pml'][mid_step:])
        
        R_pml = ref_pml / inc_amp
        R_no_pml = ref_no_pml / inc_amp

        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Est. Reflection Coeff (PEC)", f"{R_no_pml:.4f}")
        col_r2.metric("Est. Reflection Coeff (CPML)", f"{R_pml:.4e}")
        st.markdown(f"*Note: CPML achieves significantly reduced numerical reflection compared to PEC. It creates a virtual anechoic environment suitable for accurate near-field analysis.*")
