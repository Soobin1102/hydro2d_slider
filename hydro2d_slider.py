"""
hydro2d_full_fixed.py - 2D Hydrostatic Simulator (Matplotlib)
FIXED: Pressure vectors, pressure graph, and centroid marker now update correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import pi, sqrt

# --- PARAMETERS ---
R_default = 0.45          # Radius (m)
zc_default = 0.6          # Center z (m) - water surface at z=0.0
rho_default = 1000.0      # Density (kg/m^3)
g_default = 9.81          # Gravity (m/s^2)
WATER_LEVEL = 0.0

# Drawing bounds
X_LEFT, X_RIGHT = -1.6, 1.6
Z_MIN, Z_MAX = -1.6, 1.6

# Resolution
N_ARC_PTS = 300
N_INTEGRATION = 2000
N_VECTORS = 35

# --- PHYSICS HELPERS ---
def submerged_area_centroid(zc, R, n_slices=N_INTEGRATION):
    z_bot, z_top = zc - R, zc + R
    z_sub_top = min(z_top, WATER_LEVEL)
    if z_sub_top <= z_bot: return 0.0, None

    zs = np.linspace(z_bot, z_sub_top, n_slices)
    dz = zs[1] - zs[0]
    widths = 2.0 * np.sqrt(np.maximum(0.0, R**2 - (zs - zc)**2))
    dA = widths * dz
    A_sub = np.sum(dA)
    return float(A_sub), float(np.sum(zs * dA) / A_sub) if A_sub > 1e-9 else None

def circular_segment_area_analytic(R, h):
    if h <= 0: return 0.0
    if h >= 2*R: return pi * R**2
    a = R - h
    theta = 2.0 * np.arccos(a / R)
    return 0.5 * R**2 * (theta - np.sin(theta))

def circle_xy(zc, R, n=N_ARC_PTS):
    t = np.linspace(0, 2*np.pi, n)
    return R * np.cos(t), zc + R * np.sin(t), t

# --- SETUP FIGURE AND AXES ---
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.18)

ax_scene = fig.add_subplot(1, 2, 1)
ax_pressure = fig.add_subplot(2, 2, 2)
ax_stats = fig.add_subplot(2, 2, 4)
ax_stats.axis('off')

# Global variables for dynamic artists (must be managed carefully with ax.cla())
line_circle = None
cb_marker = None
quiver = None
submerged_poly_list = []

# --- DRAW AND UPDATE FUNCTION ---
def update_scene(zc_val, R_val, rho_val, g_val):
    global line_circle, cb_marker, quiver, submerged_poly_list

    # 1. Redraw Tank & Water Background (Clears ALL artists from ax_scene)
    ax_scene.cla()
    ax_scene.set_xlim(X_LEFT, X_RIGHT); ax_scene.set_ylim(Z_MIN, Z_MAX)
    ax_scene.set_xlabel('x (m)'); ax_scene.set_ylabel('z (m)')
    ax_scene.set_title('2D Tank — Circle (use sliders)')
    ax_scene.set_aspect('equal', adjustable='box')
    
    # Tank Boundary and Water
    ax_scene.plot([X_LEFT, X_RIGHT, X_RIGHT, X_LEFT, X_LEFT], [Z_MIN, Z_MIN, Z_MAX, Z_MAX, Z_MIN], color='k', linewidth=0.9)
    ax_scene.axhspan(Z_MIN, WATER_LEVEL, color='#d7f0ff', alpha=1.0)
    ax_scene.axhline(WATER_LEVEL, color='deepskyblue', linewidth=2)
    ax_scene.text(X_RIGHT * 0.95, WATER_LEVEL + 0.05, 'Water surface (z=0)', ha='right', va='bottom', color='deepskyblue', fontsize=9, weight='bold')

    # 2. Circle outline and physics
    x_circ, z_circ, _ = circle_xy(zc_val, R_val)
    line_circle, = ax_scene.plot(x_circ, z_circ, color='black', linewidth=2) # Redraw line_circle
    
    A_sub, z_cb = submerged_area_centroid(zc_val, R_val)
    A_analytic = circular_segment_area_analytic(R_val, max(0.0, min(2*R_val, R_val - (zc_val - WATER_LEVEL))))
    Fb = rho_val * g_val * A_sub

    # 3. Submerged Region Shading
    if A_sub > 0:
        mask = z_circ <= WATER_LEVEL + 1e-12
        if mask.any():
            xs_sub, zs_sub = x_circ[mask], z_circ[mask]
            
            # Chord calculation
            term = R_val**2 - (WATER_LEVEL - zc_val)**2
            dx = sqrt(term) if term >= 0 else 0
            
            poly_x = np.concatenate(([-dx], xs_sub, [dx]))
            poly_z = np.concatenate(([WATER_LEVEL], zs_sub, [WATER_LEVEL]))
            # ax_scene.fill returns a list of Polygon objects
            submerged_poly_list = ax_scene.fill(poly_x, poly_z, color='deepskyblue', alpha=0.6)

        # Quiver (Pressure Vectors)
        indices = np.where(z_circ <= WATER_LEVEL + 1e-12)[0]
        if indices.size > 0:
            spaced_idx = np.linspace(indices[0], indices[-1], min(N_VECTORS, indices.size)).astype(int)
            x_pts, z_pts = x_circ[spaced_idx], z_circ[spaced_idx]
            nx, nz = -(x_pts), -(z_pts - zc_val) 
            norms = np.hypot(nx, nz); norms[norms==0] = 1.0
            nxu, nzu = nx / norms, nz / norms
            depths = np.maximum(0.0, WATER_LEVEL - z_pts)
            pressures = rho_val * g_val * depths
            
            # Scaling: 20% of R_val per max pressure (rho*g*2R) for visual clarity
            typical_p = rho_val * g_val * (2 * R_val)
            scale_factor = 0.20 * R_val / (typical_p if typical_p > 0 else 1.0)
            lengths = pressures * scale_factor
            
            quiver = ax_scene.quiver(x_pts, z_pts, nxu * lengths, nzu * lengths, 
                                     angles='xy', scale_units='xy', scale=1, 
                                     color='navy', alpha=0.9, label='Pressure Vectors')

    # 4. Center of Buoyancy Marker
    if z_cb is not None:
        cb_marker, = ax_scene.plot([0.0], [z_cb], marker='o', color='red', markersize=6, label='Center of Buoyancy') # Redraw marker
    
    # 5. Update Pressure Plot (Clears ALL artists from ax_pressure)
    ax_pressure.cla() 
    max_depth_plot = max(2.0 * R_val + abs(zc_val), 2.0)
    depths_p = np.linspace(0.0, max_depth_plot, 300)
    pressures_p = rho_val * g_val * depths_p
    
    # Redraw plot elements
    ax_pressure.plot(depths_p, pressures_p, color='tab:blue', linewidth=2, label=r'$p = \rho g d$')
    ax_pressure.set_xlabel('Depth below water (m)')
    ax_pressure.set_ylabel('Pressure (Pa)')
    ax_pressure.set_title(r'Pressure vs Depth ($p = \rho g d$)')
    ax_pressure.grid(True)
    
    if z_cb is not None:
        depth_centroid = max(0.0, WATER_LEVEL - z_cb)
        ax_pressure.plot([depth_centroid], [rho_val * g_val * depth_centroid], marker='o', color='red', label='Centroid Depth')
    
    ax_pressure.legend(loc='upper left')

    # 6. Update Stats Text
    ax_stats.cla(); ax_stats.axis('off')
    lines = [f"Inputs:",
             f"  Radius R          = {R_val:.4f} m",
             f"  Circle center z   = {zc_val:.4f} m (water at z=0.0)",
             f"  Density rho       = {rho_val:.1f} kg/m^3",
             f"  gravity g         = {g_val:.3f} m/s^2", ""]
    
    if A_sub <= 0.0:
        lines.extend(["Object fully above water.", "Submerged area = 0.0 m^2/m", "Buoyant force = 0.0 N/m"])
    else:
        lines.extend([f"Full circle area  = {pi * R_val**2:.6f} m^2/m",
                      f"Submerged area (num)      = {A_sub:.6f} m^2/m",
                      f"Submerged area (analytic) = {A_analytic:.6f} m^2/m",
                      f"Buoyant force (per unit length) = {Fb:.3f} N/m"])
        if z_cb is not None:
            lines.extend([f"Center of buoyancy $z_{{cb}}$ = {z_cb:.4f} m",
                          f"Depth of centroid = {max(0.0, WATER_LEVEL - z_cb):.4f} m"])

    ax_stats.text(0.01, 0.98, "\n".join(lines), va='top', ha='left', family='monospace', fontsize=10)
    fig.canvas.draw_idle()

# --- WIDGETS: SLIDERS ---
# Placement
slider_dims = [0.30, 0.03]
slider_x_1 = 0.10
slider_x_2 = slider_x_1 + slider_dims[0] + 0.12
slider_y_1 = 0.06
slider_y_2 = slider_y_1 - 0.055

# Sliders
slider_zc = Slider(fig.add_axes([slider_x_1, slider_y_1, *slider_dims]), "Center z (m)", Z_MIN + 0.2, Z_MAX - 0.2, valinit=zc_default, valstep=0.005)
slider_R = Slider(fig.add_axes([slider_x_2, slider_y_1, *slider_dims]), "Radius R (m)", 0.05, 1.2, valinit=R_default, valstep=0.005)
slider_rho = Slider(fig.add_axes([slider_x_1, slider_y_2, *slider_dims]), "Density ρ (kg/m³)", 100.0, 2000.0, valinit=rho_default, valstep=10.0)
slider_g = Slider(fig.add_axes([slider_x_2, slider_y_2, *slider_dims]), "g (m/s²)", 1.0, 20.0, valinit=g_default, valstep=0.01)

# Slider Callback
def sliders_on_changed(val):
    update_scene(slider_zc.val, slider_R.val, slider_rho.val, slider_g.val)

slider_zc.on_changed(sliders_on_changed)
slider_R.on_changed(sliders_on_changed)
slider_rho.on_changed(sliders_on_changed)
slider_g.on_changed(sliders_on_changed)

# --- START ---
update_scene(zc_default, R_default, rho_default, g_default)
plt.show()