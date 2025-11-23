"""
hydro2d_full.py

2D Hydrostatic Simulator (Matplotlib) — Side-by-side layout (Option C)

Features:
- Left: 2D tank + circle (ball). Waterline and tank boundary labeled.
- Submerged area shaded; circle always visible.
- Pressure vectors shown on submerged arc (normal inward, magnitude = rho*g*depth).
- Center of buoyancy (centroid of submerged area) marked.
- Right top: Pressure vs depth (p = rho*g*d); marker for centroid depth.
- Right bottom: Numeric readout of values.
- Sliders: circle center z, radius, rho, g.

Units: meters, kg, seconds. Forces are per unit length into the page (N/m).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from math import pi, sqrt

# -------------------------
# Default parameters
# -------------------------
R_default = 0.45            # circle radius (m)
zc_default = 0.6            # circle center z coordinate (m) -- water surface at z = 0.0
rho_default = 1000.0        # fluid density (kg/m^3)
g_default = 9.81            # gravity (m/s^2)
water_level = 0.0

# drawing bounds
x_left, x_right = -1.6, 1.6
z_min_default = -1.6
z_max_default = 1.6

# numerical resolution
N_arc_pts = 300    # points for circle plotting
N_integration = 2000  # slices for submerged area integration and centroid
N_pressure_vectors = 35  # how many arrows to plot around submerged arc (<= N_arc_pts)

# -------------------------
# Physics helpers
# -------------------------
def submerged_area_and_centroid(zc, R, water_level=0.0, n_slices=N_integration):
    """
    Numerically compute submerged area (per unit length) and centroid z-coordinate (center of buoyancy).
    z axis is vertical; water_level is the free surface z.
    Returns (A_sub, z_centroid) where z_centroid is None if A_sub == 0.
    """
    z_bot = zc - R
    z_top = zc + R
    z_sub_top = min(z_top, water_level)
    if z_sub_top <= z_bot:
        return 0.0, None

    zs = np.linspace(z_bot, z_sub_top, n_slices)
    dz = zs[1] - zs[0]
    half_chord = np.sqrt(np.maximum(0.0, R**2 - (zs - zc)**2))
    widths = 2.0 * half_chord
    dA = widths * dz
    A_sub = float(np.sum(dA))
    if A_sub <= 0.0:
        return 0.0, None
    z_moment = float(np.sum(zs * dA))
    z_centroid = z_moment / A_sub
    return A_sub, z_centroid

def circular_segment_area_analytic(R, h):
    """
    Analytic area of a circular segment of height h (0 <= h <= 2R).
    h is segment depth measured from top of segment downward.
    """
    if h <= 0:
        return 0.0
    if h >= 2*R:
        return pi * R**2
    a = R - h  # distance from center to chord
    theta = 2.0 * np.arccos(a / R)
    area = 0.5 * R**2 * (theta - np.sin(theta))
    return float(area)

def buoyant_force_per_length(rho, g, A_sub):
    """Buoyant force per unit length into page (N/m)."""
    return rho * g * A_sub

# -------------------------
# Plot helpers
# -------------------------
def circle_xy(xc, zc, R, n=N_arc_pts):
    t = np.linspace(0, 2*np.pi, n)
    x = xc + R * np.cos(t)
    z = zc + R * np.sin(t)
    return x, z, t

# -------------------------
# Build figure and axes (side-by-side layout)
# -------------------------
fig = plt.figure(figsize=(12, 6))
# leave room at bottom for sliders
plt.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.18)

# Left: tank scene
ax_scene = fig.add_subplot(1, 2, 1)
ax_scene.set_xlim(x_left, x_right)
ax_scene.set_ylim(z_min_default, z_max_default)
ax_scene.set_aspect('equal', adjustable='box')
ax_scene.set_xlabel('x (m)')
ax_scene.set_ylabel('z (m)')
ax_scene.set_title('2D Tank — Circle (use sliders)')

# Right top: pressure vs depth
ax_pressure = fig.add_subplot(2, 2, 2)
ax_pressure.set_title('Pressure vs Depth')
ax_pressure.set_xlabel('Depth below water (m)')
ax_pressure.set_ylabel('Pressure (Pa)')
ax_pressure.grid(True)

# Right bottom: stats text
ax_stats = fig.add_subplot(2, 2, 4)
ax_stats.axis('off')

# draw static tank boundary and waterline (we will redraw when necessary)
def draw_tank_and_water(ax, water_level=0.0):
    ax.cla()
    # tank rectangle
    ax.plot([x_left, x_right, x_right, x_left, x_left],
            [z_min_default, z_min_default, z_max_default, z_max_default, z_min_default],
            color='k', linewidth=0.9)
    # water shading below water level
    ax.axhspan(z_min_default, water_level, color='#d7f0ff', alpha=1.0)
    # waterline label
    ax.axhline(water_level, color='deepskyblue', linewidth=2)
    ax.text(x_right - 0.05*(x_right-x_left), water_level + 0.03*(z_max_default - z_min_default),
            'Water surface (z=0)', ha='right', va='bottom', color='deepskyblue', fontsize=9, weight='bold')

# -------------------------
# Initial params and artists
# -------------------------
R = R_default
zc = zc_default
rho = rho_default
g = g_default

# circle outline patch (we'll not use Circle patch for filling because we shade submerged region separately)
x_circ, z_circ, theta = circle_xy(0.0, zc, R)
line_circle, = ax_scene.plot(x_circ, z_circ, color='black', linewidth=2)

# central marker for center of buoyancy (plotted as filled circle)
cb_marker, = ax_scene.plot([], [], marker='o', color='red', markersize=6, label='Center of Buoyancy')

# pressure vectors (quiver)
quiver = None  # will be created dynamically

# submerged polygon patch drawn with fill
submerged_poly = None

# centroid depth marker on pressure plot
centroid_depth_marker, = ax_pressure.plot([], [], marker='o', color='red', label='centroid depth')

# pressure line (we'll create once)
pressure_line, = ax_pressure.plot([], [], color='tab:blue', linewidth=2, label='p = rho*g*d')

# legend
ax_pressure.legend(loc='upper left')

# -------------------------
# Update function (responsible to redraw everything)
# -------------------------
def update_scene(zc_val, R_val, rho_val, g_val, redraw_pressure=True):
    global line_circle, cb_marker, quiver, submerged_poly, centroid_depth_marker, pressure_line
    zc_val = float(zc_val)
    R_val = float(R_val)
    rho_val = float(rho_val)
    g_val = float(g_val)

    # Draw tank & water background
    draw_tank_and_water(ax_scene, water_level=water_level)

    # Circle coordinates
    x_circ, z_circ, theta = circle_xy(0.0, zc_val, R_val, n=N_arc_pts)
    line_circle.set_data(x_circ, z_circ)
    ax_scene.add_line(line_circle)

    # compute submerged area & centroid
    A_sub, z_cb = submerged_area_and_centroid(zc_val, R_val, water_level=water_level, n_slices=N_integration)
    A_analytic = circular_segment_area_analytic(R_val, max(0.0, min(2*R_val, R_val - (zc_val - water_level))))
    Fb = buoyant_force_per_length(rho_val, g_val, A_sub)

    # shade submerged region
    global submerged_poly
    if submerged_poly is not None:
        try:
            submerged_poly.remove()
        except Exception:
            pass
        submerged_poly = None

    if A_sub > 0:
        # build polygon: submerged arc left->right then chord back
        mask = z_circ <= water_level + 1e-12
        xs_sub = x_circ[mask]
        zs_sub = z_circ[mask]
        # If no mask points due to sampling, fallback to arc construction
        if xs_sub.size > 0:
            # chord endpoints
            term = R_val**2 - (water_level - zc_val)**2
            if term >= 0:
                dx = sqrt(term)
                x_left_chord = -dx
                x_right_chord = dx
                poly_x = np.concatenate(([x_left_chord], xs_sub, [x_right_chord]))
                poly_z = np.concatenate(([water_level], zs_sub, [water_level]))
                submerged_poly = ax_scene.fill(poly_x, poly_z, color='deepskyblue', alpha=0.6)[0]
        else:
            # fallback arc
            term = R_val**2 - (water_level - zc_val)**2
            if term >= 0:
                dx = sqrt(term)
                x_left_chord = -dx
                x_right_chord = dx
                angle_left = np.arctan2(water_level - zc_val, x_left_chord)
                angle_right = np.arctan2(water_level - zc_val, x_right_chord)
                if angle_right < angle_left:
                    angle_right += 2*np.pi
                arc_theta = np.linspace(angle_left, angle_right, 200)
                arc_x = 0.0 + R_val * np.cos(arc_theta)
                arc_z = zc_val + R_val * np.sin(arc_theta)
                poly_x = np.concatenate(([x_left_chord], arc_x, [x_right_chord]))
                poly_z = np.concatenate(([water_level], arc_z, [water_level]))
                submerged_poly = ax_scene.fill(poly_x, poly_z, color='deepskyblue', alpha=0.6)[0]
    else:
        # nothing submerged
        pass

    # draw center of buoyancy marker
    if z_cb is not None:
        cb_marker.set_data([0.0], [z_cb])
    else:
        cb_marker.set_data([], [])

    # draw pressure vectors (quiver) on submerged arc
    # remove old quiver
    if quiver is not None:
        try:
            quiver.remove()
        except Exception:
            pass

    # compute vectors along submerged part
    if A_sub > 0:
        # select evenly spaced points along submerged arc
        # find indices where z <= water_level
        indices = np.where(z_circ <= water_level + 1e-12)[0]
        if indices.size > 0:
            # pick evenly spaced subset for vectors
            spaced_idx = np.linspace(indices[0], indices[-1], min(N_pressure_vectors, indices.size)).astype(int)
            x_pts = x_circ[spaced_idx]
            z_pts = z_circ[spaced_idx]
            # normal vectors (pointing inward to circle center)
            # radial vector from surface point to center (0, zc)
            nx = - (x_pts - 0.0)  # inward x component
            nz = - (z_pts - zc_val)  # inward z component
            norms = np.sqrt(nx**2 + nz**2)
            norms[norms==0] = 1.0
            nxu = nx / norms
            nzu = nz / norms
            # pressure magnitude = rho * g * depth (depth = water_level - z_point)
            depths = np.maximum(0.0, water_level - z_pts)
            pressures = rho_val * g_val * depths  # Pa
            # scale arrows to be visible: arrow length = k * pressure / (rho*g*R) times R
            # where typical max pressure = rho*g*(2R)
            typical = rho_val * g_val * (2*R_val)
            # avoid division by zero
            if typical <= 0:
                typical = 1.0
            k = 0.20  # visual scaling factor
            lengths = k * (pressures / typical) * R_val
            U = nxu * lengths
            V = nzu * lengths
            quiver = ax_scene.quiver(x_pts, z_pts, U, V, angles='xy', scale_units='xy', scale=1, color='navy', alpha=0.9)
    else:
        quiver = None

    # update pressure plot: depths from 0 .. max_depth
    if redraw_pressure:
        max_depth_plot = max(2.0 * R_val + abs(zc_val), 2.0)
        depths = np.linspace(0.0, max_depth_plot, 300)
        pressures = rho_val * g_val * depths
        ax_pressure.cla()
        ax_pressure.plot(depths, pressures, color='tab:blue', linewidth=2)
        ax_pressure.set_xlabel('Depth below water (m)')
        ax_pressure.set_ylabel('Pressure (Pa)')
        ax_pressure.set_title('Pressure vs Depth (p = ρ g d)')
        ax_pressure.grid(True)
        # if centroid exists, mark its depth (below water)
        if z_cb is not None:
            depth_centroid = max(0.0, water_level - z_cb)
            ax_pressure.plot([depth_centroid], [rho_val * g_val * depth_centroid], marker='o', color='red', label='centroid depth')
            ax_pressure.legend(loc='upper left')

    # stats text
    ax_stats.cla()
    ax_stats.axis('off')
    lines = []
    lines.append("Inputs:")
    lines.append(f"  Radius R         = {R_val:.4f} m")
    lines.append(f"  Circle center z  = {zc_val:.4f} m (water at z=0.0)")
    lines.append(f"  Density rho      = {rho_val:.1f} kg/m^3")
    lines.append(f"  gravity g        = {g_val:.3f} m/s^2")
    lines.append("")
    if A_sub <= 0.0:
        lines.append("Object fully above water.")
        lines.append("Submerged area = 0.0 m^2 (per unit length)")
        lines.append("Buoyant force = 0.0 N (per unit length)")
    else:
        Vfull = pi * R_val**2
        Fb = buoyant_force_per_length(rho_val, g_val, A_sub)
        lines.append(f"Full circle area = {Vfull:.6f} m^2 (per unit length)")
        lines.append(f"Submerged area  (num) = {A_sub:.6f} m^2")
        lines.append(f"Submerged area (analytic) = {A_analytic:.6f} m^2")
        lines.append(f"Buoyant force (per unit length) = {Fb:.3f} N/m")
        if z_cb is not None:
            lines.append(f"Center of buoyancy z_cb = {z_cb:.4f} m (above origin)")
            lines.append(f"Depth of centroid (below water) = {max(0.0, water_level - z_cb):.4f} m")
    ax_stats.text(0.01, 0.98, "\n".join(lines), va='top', ha='left', family='monospace', fontsize=10)

    # redraw canvas
    fig.canvas.draw_idle()

# initial draw
update_scene(zc, R, rho, g, redraw_pressure=True)

# -------------------------
# Widgets: sliders
# -------------------------
# placement coords
slider_height = 0.03
slider_left = 0.10
slider_width = 0.30
slider_gap = 0.12
bottom_y = 0.06

# slider: zc (center)
ax_zc = fig.add_axes([slider_left, bottom_y, slider_width, slider_height])
slider_zc = Slider(ax_zc, "Center z (m)", z_min_default + 0.2, z_max_default - 0.2, valinit=zc, valstep=0.005)

# slider: Radius
ax_R = fig.add_axes([slider_left + slider_width + slider_gap, bottom_y, slider_width, slider_height])
slider_R = Slider(ax_R, "Radius R (m)", 0.05, 1.2, valinit=R, valstep=0.005)

# slider: density rho
ax_rho = fig.add_axes([slider_left, bottom_y - 0.055, slider_width, slider_height])
slider_rho = Slider(ax_rho, "Density ρ (kg/m³)", 100.0, 2000.0, valinit=rho, valstep=10.0)

# slider: gravity g
ax_g = fig.add_axes([slider_left + slider_width + slider_gap, bottom_y - 0.055, slider_width, slider_height])
slider_g = Slider(ax_g, "g (m/s²)", 1.0, 20.0, valinit=g, valstep=0.01)

# -------------------------
# Slider callbacks
# -------------------------
def sliders_on_changed(val):
    cur_zc = slider_zc.val
    cur_R = slider_R.val
    cur_rho = slider_rho.val
    cur_g = slider_g.val
    update_scene(cur_zc, cur_R, cur_rho, cur_g, redraw_pressure=True)

slider_zc.on_changed(sliders_on_changed)
slider_R.on_changed(sliders_on_changed)
slider_rho.on_changed(sliders_on_changed)
slider_g.on_changed(sliders_on_changed)

# -------------------------
# Start UI
# -------------------------
plt.show()
