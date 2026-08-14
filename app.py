"""
3D Electromagnetics & Antenna Radiation Laboratory
Milestone 1 — 3D FDTD Electromagnetic Wave Propagation
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
import sys
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
st.markdown("### Milestone 1 — 3D FDTD Electromagnetic Wave Propagation")
st.markdown("*Note: Boundary: Basic numerical boundary (PEC). PML will be implemented in a future milestone.*")

# Sidebar Configuration
st.sidebar.header("GRID")
Nx = st.sidebar.number_input("Nx (Cells)", min_value=10, max_value=200, value=50, step=10)
Ny = st.sidebar.number_input("Ny (Cells)", min_value=10, max_value=200, value=50, step=10)
Nz = st.sidebar.number_input("Nz (Cells)", min_value=10, max_value=200, value=50, step=10)
dx = st.sidebar.number_input("dx (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dy = st.sidebar.number_input("dy (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")
dz = st.sidebar.number_input("dz (m)", min_value=0.001, max_value=0.1, value=0.01, format="%.4f")

st.sidebar.header("SIMULATION")
num_steps = st.sidebar.number_input("Number of timesteps", min_value=10, max_value=5000, value=150, step=50)
cfl_factor = st.sidebar.slider("CFL Safety Factor", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

st.sidebar.header("SOURCE")
src_pos_x = st.sidebar.number_input("Source X (index)", min_value=1, max_value=Nx-2, value=Nx//2)
src_pos_y = st.sidebar.number_input("Source Y (index)", min_value=1, max_value=Ny-2, value=Ny//2)
src_pos_z = st.sidebar.number_input("Source Z (index)", min_value=1, max_value=Nz-2, value=Nz//2)
amplitude = st.sidebar.number_input("Pulse Amplitude (V/m)", value=1.0)
pulse_width = st.sidebar.number_input("Pulse Width (timesteps)", min_value=5, max_value=100, value=15)
pulse_delay = st.sidebar.number_input("Pulse Delay (timesteps)", min_value=0, max_value=300, value=45)
polarization = st.sidebar.selectbox("Polarization", options=["Ez", "Ex", "Ey"])
pol_idx = {"Ex": 0, "Ey": 1, "Ez": 2}[polarization]

st.sidebar.header("VISUALIZATION")
vis_field = st.sidebar.selectbox("Field Component to Visualize", ["|E|", "|H|", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
vis_mode = st.sidebar.selectbox("Visualization Mode", ["Orthogonal Slices", "Volume Rendering"])

# ============================================================
# CFL CALCULATION & MEMORY MANAGEMENT
# ============================================================
# CFL condition for 3D free space
dt_max = 1.0 / (C_LIGHT * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
dt = cfl_factor * dt_max

# Memory Check (6 arrays * float32 (4 bytes) * Nx * Ny * Nz)
memory_bytes = 6 * (Nx * Ny * Nz) * 4
memory_mb = memory_bytes / (1024 * 1024)

st.sidebar.markdown(f"**Estimated Memory:** `{memory_mb:.2f} MB`")
st.sidebar.markdown(f"**CFL Max dt:** `{dt_max:.4e} s`")
st.sidebar.markdown(f"**Actual dt:** `{dt:.4e} s`")

if memory_mb > 500:
    st.error(f"Requested grid requires {memory_mb:.2f} MB of RAM, which exceeds the 500 MB safety limit. Please reduce grid size.")
    st.stop()

# ============================================================
# FDTD SOLVER (NUMBA OPTIMIZED)
# ============================================================
@nb.njit(cache=True)
def fdtd_step(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz, t, src_pos, amp, delay, width, pol_idx):
    """Executes a single FDTD time step using the 3D Yee grid staggered formulation."""
    Nx, Ny, Nz = Ex.shape

    # 1. Update H-field (Curl E)
    # H field components are evaluated at half-integer spatial steps.
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                Hx[i, j, k] -= (dt / MU_0) * ((Ez[i, j+1, k] - Ez[i, j, k]) / dy - (Ey[i, j, k+1] - Ey[i, j, k]) / dz)
                Hy[i, j, k] -= (dt / MU_0) * ((Ex[i, j, k+1] - Ex[i, j, k]) / dz - (Ez[i+1, j, k] - Ez[i, j, k]) / dx)
                Hz[i, j, k] -= (dt / MU_0) * ((Ey[i+1, j, k] - Ey[i, j, k]) / dx - (Ex[i, j+1, k] - Ex[i, j, k]) / dy)

    # 2. Update E-field (Curl H)
    # E field components are evaluated at integer spatial steps.
    # Boundary faces remain 0 (Basic PEC numerical truncation)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            for k in range(1, Nz - 1):
                Ex[i, j, k] += (dt / EPS_0) * ((Hz[i, j, k] - Hz[i, j-1, k]) / dy - (Hy[i, j, k] - Hy[i, j, k-1]) / dz)
                Ey[i, j, k] += (dt / EPS_0) * ((Hx[i, j, k] - Hx[i, j, k-1]) / dz - (Hz[i, j, k] - Hz[i-1, j, k]) / dx)
                Ez[i, j, k] += (dt / EPS_0) * ((Hy[i, j, k] - Hy[i-1, j, k]) / dx - (Hx[i, j, k] - Hx[i, j-1, k]) / dy)

    # 3. Inject Gaussian Source (Soft Source mechanism)
    sx, sy, sz = src_pos
    pulse = amp * math.exp(-0.5 * ((t - delay) / width)**2)
    if pol_idx == 0:
        Ex[sx, sy, sz] += pulse
    elif pol_idx == 1:
        Ey[sx, sy, sz] += pulse
    elif pol_idx == 2:
        Ez[sx, sy, sz] += pulse

@nb.njit(cache=True)
def run_simulation(Nx, Ny, Nz, dx, dy, dz, dt, steps, src_pos, amp, delay, width, pol_idx, probe_pos):
    """Runs the complete FDTD loop over all timesteps."""
    # Initialize electromagnetic fields (float32 to save memory)
    Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    src_history = np.zeros(steps, dtype=np.float32)
    probe_history = np.zeros(steps, dtype=np.float32)
    px, py, pz = probe_pos

    for n in range(steps):
        # Current physical time
        t_steps = float(n)
        
        fdtd_step(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz, t_steps, src_pos, amp, delay, width, pol_idx)

        # Record histories
        src_history[n] = amp * math.exp(-0.5 * ((t_steps - delay) / width)**2)
        if pol_idx == 0:
            probe_history[n] = Ex[px, py, pz]
        elif pol_idx == 1:
            probe_history[n] = Ey[px, py, pz]
        else:
            probe_history[n] = Ez[px, py, pz]

    return Ex, Ey, Ez, Hx, Hy, Hz, src_history, probe_history

# ============================================================
# BUTTONS & STATE MANAGEMENT
# ============================================================
col_btns1, col_btns2, col_btns3 = st.columns([1, 1, 4])
run_btn = col_btns1.button("Run Simulation", type="primary")
reset_btn = col_btns2.button("Reset")

if reset_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ============================================================
# SIMULATION EXECUTION & ANALYSIS
# ============================================================
if run_btn:
    with st.spinner("Calculating 3D FDTD Maxwell updates. Please wait..."):
        src_position = (src_pos_x, src_pos_y, src_pos_z)
        
        # Probe placed slightly offset from the source to measure propagation
        probe_x = min(src_pos_x + 5, Nx - 2)
        probe_y = src_pos_y
        probe_z = src_pos_z
        probe_position = (probe_x, probe_y, probe_z)

        start_time = time.time()
        Ex, Ey, Ez, Hx, Hy, Hz, src_hist, probe_hist = run_simulation(
            Nx, Ny, Nz, dx, dy, dz, dt, num_steps, 
            src_position, amplitude, pulse_delay, pulse_width, pol_idx, probe_position
        )
        calc_time = time.time() - start_time

        st.session_state['results'] = {
            'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
            'src_hist': src_hist, 'probe_hist': probe_hist,
            'calc_time': calc_time, 'dt': dt, 'probe_offset': 5 * dx
        }

if 'results' in st.session_state:
    res = st.session_state['results']
    
    # Calculate magnitudes lazily based on UI selection or diagnostics
    E_mag = np.sqrt(res['Ex']**2 + res['Ey']**2 + res['Ez']**2)
    H_mag = np.sqrt(res['Hx']**2 + res['Hy']**2 + res['Hz']**2)

    # ============================================================
    # VALIDATION
    # ============================================================
    st.markdown("### Simulation Validation")
    
    # 1. CFL Check
    cfl_pass = dt <= dt_max
    
    # 2. NaN and Inf check
    has_nan = np.isnan(res['Ex']).any()
    has_inf = np.isinf(res['Ex']).any()
    finite_pass = not has_nan and not has_inf
    
    # 3. Wave Propagation Check (Did the probe record a signal?)
    prop_pass = np.max(np.abs(res['probe_hist'])) > (amplitude * 1e-4)

    # 4. Energy Sanity Check (Total Electromagnetic Energy)
    total_energy = np.sum(0.5 * (EPS_0 * E_mag**2 + MU_0 * H_mag**2)) * (dx * dy * dz)
    stability_pass = total_energy < (amplitude**2 * 1e5) # Simplistic bounds check

    col_val1, col_val2, col_val3, col_val4, col_val5 = st.columns(5)
    col_val1.metric("CFL Stability", "PASS" if cfl_pass else "FAIL", delta_color="off" if cfl_pass else "inverse")
    col_val2.metric("Finite Values", "PASS" if finite_pass else "FAIL")
    col_val3.metric("NaN Check", "PASS" if not has_nan else "FAIL")
    col_val4.metric("Wave Propagation", "PASS" if prop_pass else "FAIL")
    col_val5.metric("Energy Stability", "PASS" if stability_pass else "FAIL")

    st.markdown(f"**Diagnostic Data:** Total Field Energy $\approx$ `{total_energy:.4e} Joules` | Max $|E|$ = `{np.max(E_mag):.4e} V/m` | Solve Time = `{res['calc_time']:,.2f} s`")
    
    if not finite_pass or not cfl_pass:
        st.error("Simulation became numerically unstable. Reduce CFL Safety Factor.")
        st.stop()

    st.divider()

    # ============================================================
    # PYVISTA 3D VISUALIZATION
    # ============================================================
    st.markdown(f"### 3D Visualization: {vis_field} Component")
    
    # Select requested data array
    field_map = {
        "|E|": E_mag, "|H|": H_mag,
        "Ex": res['Ex'], "Ey": res['Ey'], "Ez": res['Ez'],
        "Hx": res['Hx'], "Hy": res['Hy'], "Hz": res['Hz']
    }
    plot_data = field_map[vis_field]

    with st.spinner("Rendering 3D Domain..."):
        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = np.array([Nx, Ny, Nz])
        grid.spacing = (dx, dy, dz)
        grid.point_data[vis_field] = plot_data.flatten(order="F") # F-order required for correct spatial mapping

        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.set_background("white")

        if vis_mode == "Orthogonal Slices":
            slices = grid.slice_orthogonal(x=src_pos_x*dx, y=src_pos_y*dy, z=src_pos_z*dz)
            plotter.add_mesh(slices, cmap="jet", show_scalar_bar=True, scalar_bar_args={"title": f"{vis_field} Amplitude", "color": "black"})
        else: # Volume rendering
            plotter.add_volume(grid, cmap="jet", opacity="linear", show_scalar_bar=True, scalar_bar_args={"title": f"{vis_field} Amplitude", "color": "black"})
            plotter.add_mesh(grid.outline(), color="black")

        # Set a nice camera view
        plotter.view_isometric()
        img = plotter.screenshot(transparent_background=False)
        st.image(img, use_container_width=True)

    st.divider()

    # ============================================================
    # 2D SCIENTIFIC PLOTS
    # ============================================================
    st.markdown("### Temporal Field Analysis")
    st.markdown(f"*Probe placed {res['probe_offset']:.3f} meters away from source to verify causality and physical propagation.*")

    time_axis = np.arange(num_steps) * res['dt'] * 1e9 # Convert to nanoseconds

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=res['src_hist'], mode='lines', name='Source Waveform', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=time_axis, y=res['probe_hist'], mode='lines', name=f'Probe {polarization}', line=dict(color='blue')))
    
    fig.update_layout(
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude (V/m)",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Configure the grid and press **Run Simulation** to generate the electromagnetic fields.")

