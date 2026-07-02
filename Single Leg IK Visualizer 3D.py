import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

# mpl_toolkits.mplot3d is required for 3D projection in older versions,
# and implicitly used in newer matplotlib when projection='3d' is called.

# --------------------------------------------------
# Fixed lengths
# --------------------------------------------------

LS1 = 8.0
LS2 = 12.0

# --------------------------------------------------
# Boundary limits (for the local extension distance)
# --------------------------------------------------
MIN_EXT = abs(LS1 - LS2) + 0.01
MAX_EXT = LS1 + LS2 - 0.01
INITIAL_EXT = max(MIN_EXT, min(10.0, MAX_EXT))

# --------------------------------------------------
# Joint Constraints
# --------------------------------------------------
# LS1 Absolute Rotation Limits (in local plane)
LS1_ANGLE_MIN = 210.0
LS1_ANGLE_MAX = 30.0

# Beta (Angle at C, between LS1 and LS2) Limits
BETA_MIN = 20.0
BETA_MAX = 150.0

# Roll Axis Limits (Around the X axis)
# -30 = Inward, +90 = Outward
ROLL_MIN = -30.0
ROLL_MAX = 90.0

# --------------------------------------------------
# Figure setup
# --------------------------------------------------

fig = plt.subplots(figsize=(12, 10))
fig = plt.gcf()
# Add 3D subplot
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# --------------------------------------------------
# UI Widgets (Two Rows for breathing room)
# --------------------------------------------------
# Row 1: Sliders
ax_ext = plt.axes([0.10, 0.12, 0.20, 0.03])
extension_slider = Slider(ax_ext, "Extension", MIN_EXT, MAX_EXT, valinit=INITIAL_EXT)

ax_angle = plt.axes([0.40, 0.12, 0.20, 0.03])
angle_slider = Slider(ax_angle, "Pitch (°)", -180, 180, valinit=0)

ax_roll = plt.axes([0.70, 0.12, 0.20, 0.03])
roll_slider = Slider(ax_roll, "Roll (°)", ROLL_MIN, ROLL_MAX, valinit=0)

# Row 2: Coordinate Controls
ax_box_start = plt.axes([0.20, 0.04, 0.15, 0.04])
text_start = TextBox(ax_box_start, 'Start (X,Y,Z): ', initial="0, 0, -15")

ax_box_stop = plt.axes([0.45, 0.04, 0.15, 0.04])
text_stop = TextBox(ax_box_stop, 'Stop (X,Y,Z): ', initial="10, 8, -5")

ax_button = plt.axes([0.65, 0.04, 0.15, 0.04])
btn_perform = Button(ax_button, 'Animate 3D')

# --------------------------------------------------
# Global state
# --------------------------------------------------
active_branch = None
WORKSPACE_COLORS = None  # Caches the 2D workspace map color array
first_draw = True


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def norm_angle(a):
    return a % 360.0


def is_angle_in_range(phi, min_ang, max_ang):
    min_ang = norm_angle(min_ang)
    max_ang = norm_angle(max_ang)
    phi = norm_angle(phi)

    if min_ang <= max_ang:
        return min_ang <= phi <= max_ang
    else:
        return phi >= min_ang or phi <= max_ang


def rotate_to_3d(x_loc, z_loc, roll_deg):
    """Rotates local 2D planar coordinates into global 3D space"""
    gamma = math.radians(roll_deg)
    x_3d = x_loc
    y_3d = -z_loc * math.sin(gamma)
    z_3d = z_loc * math.cos(gamma)
    return x_3d, y_3d, z_3d


def find_apex_candidates(A, B, r1, r2):
    d = np.linalg.norm(B - A)
    if d == 0 or d >= r1 + r2 or d <= abs(r1 - r2): return []
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = r1 ** 2 - a ** 2
    h = np.sqrt(max(0, h_sq))
    P = A + a * (B - A) / d
    perp = np.array([-(B[1] - A[1]), B[0] - A[0]]) / d
    return [P + h * perp, P - h * perp]


def find_valid_apex(A, B, r1, r2):
    global active_branch
    candidates = find_apex_candidates(A, B, r1, r2)
    if not candidates: return None, []

    d = np.linalg.norm(B - A)
    cos_beta = (r1 ** 2 + r2 ** 2 - d ** 2) / (2 * r1 * r2)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))

    if not (BETA_MIN <= beta <= BETA_MAX): return None, candidates

    if active_branch is not None:
        c = candidates[active_branch]
        phi = norm_angle(math.degrees(math.atan2(c[1] - A[1], c[0] - A[0])))
        if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
            return c, candidates
        else:
            return None, candidates
    else:
        for i, c in enumerate(candidates):
            phi = norm_angle(math.degrees(math.atan2(c[1] - A[1], c[0] - A[0])))
            if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
                active_branch = i
                return c, candidates
        return None, candidates


def parse_coordinate_3d(text_val):
    try:
        clean = text_val.replace('(', '').replace(')', '').strip()
        parts = clean.split(',')
        if len(parts) != 3: return None
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def generate_arc_3d(center, pt1, pt2, radius, roll_deg):
    """Generates 3D line points for an arc (used since Matplotlib 2D Arc fails in 3D)"""
    ang1 = math.atan2(pt1[1] - center[1], pt1[0] - center[0])
    ang2 = math.atan2(pt2[1] - center[1], pt2[0] - center[0])

    diff = (ang2 - ang1) % (2 * math.pi)
    if diff > math.pi: diff -= 2 * math.pi

    start = ang1
    end = ang1 + diff
    if end < start: start, end = end, start

    t = np.linspace(start, end, 20)
    x_loc = center[0] + radius * np.cos(t)
    z_loc = center[1] + radius * np.sin(t)

    # Map local arc to 3D
    x_3d, y_3d, z_3d = rotate_to_3d(x_loc, z_loc, roll_deg)

    # Find mid point for label
    mid = start + (end - start) / 2
    lx, ly, lz = rotate_to_3d(center[0] + radius * 1.35 * math.cos(mid),
                              center[1] + radius * 1.35 * math.sin(mid),
                              roll_deg)
    return x_3d, y_3d, z_3d, lx, ly, lz


# --------------------------------------------------
# Draw
# --------------------------------------------------

def draw(extension, angle_deg, roll_deg):
    global WORKSPACE_COLORS, first_draw

    # Store camera view before clearing
    if not first_draw:
        elev = ax.elev
        azim = ax.azim

    ax.clear()

    # Check roll bounds
    if not (ROLL_MIN <= roll_deg <= ROLL_MAX):
        ax.text2D(0.5, 0.5, f"Roll Limit Exceeded\nMust be between {ROLL_MIN}° and {ROLL_MAX}°",
                  fontsize=13, ha='center', va='center', transform=ax.transAxes, color='darkred')
        fig.canvas.draw_idle()
        return

    # Local 2D Kinematics (X and Z axis)
    A_loc = np.array([0.0, 0.0])
    theta = np.radians(angle_deg - 90)
    B_loc = np.array([extension * np.cos(theta), extension * np.sin(theta)])

    limit = LS1 + LS2 + 2
    C_loc, candidates = find_valid_apex(A_loc, B_loc, LS1, LS2)

    # --------------------------------------------------
    # 3D Workspace Plane Generation
    # --------------------------------------------------
    grid_res = 120
    x_val = np.linspace(-limit, limit, grid_res)
    z_val = np.linspace(-limit, limit, grid_res)
    X_loc_grid, Z_loc_grid = np.meshgrid(x_val, z_val)

    if WORKSPACE_COLORS is None and active_branch is not None:
        # Calculate validity colors ONCE in the local frame
        R = np.hypot(X_loc_grid, Z_loc_grid)
        R_safe = np.where(R == 0, 1e-6, R)

        mask_ext = (R >= MIN_EXT) & (R <= MAX_EXT)

        cos_beta_grid = (LS1 ** 2 + LS2 ** 2 - R ** 2) / (2 * LS1 * LS2)
        beta_grid = np.degrees(np.arccos(np.clip(cos_beta_grid, -1.0, 1.0)))
        mask_beta = (beta_grid >= BETA_MIN) & (beta_grid <= BETA_MAX)

        a = (LS1 ** 2 - LS2 ** 2 + R ** 2) / (2 * R_safe)
        h_sq = LS1 ** 2 - a ** 2
        h = np.sqrt(np.maximum(0, h_sq))

        Px = a * X_loc_grid / R_safe
        Pz = a * Z_loc_grid / R_safe
        perpx = -Z_loc_grid / R_safe
        perpz = X_loc_grid / R_safe

        if active_branch == 0:
            Cx_g, Cz_g = Px + h * perpx, Pz + h * perpz
        else:
            Cx_g, Cz_g = Px - h * perpx, Pz - h * perpz

        phi = np.degrees(np.arctan2(Cz_g, Cx_g)) % 360.0

        if LS1_ANGLE_MIN <= LS1_ANGLE_MAX:
            mask_phi = (phi >= LS1_ANGLE_MIN) & (phi <= LS1_ANGLE_MAX)
        else:
            mask_phi = (phi >= LS1_ANGLE_MIN) | (phi <= LS1_ANGLE_MAX)

        valid_space = mask_ext & mask_beta & mask_phi

        WORKSPACE_COLORS = np.zeros((grid_res, grid_res, 4))
        WORKSPACE_COLORS[~valid_space] = [1.0, 0.0, 0.0, 0.08]  # Red, high transparency
        WORKSPACE_COLORS[valid_space] = [0.0, 1.0, 0.0, 0.20]  # Green, visible

    if WORKSPACE_COLORS is not None:
        # Rotate the grid points dynamically to form the 3D surface
        X_3d_grid, Y_3d_grid, Z_3d_grid = rotate_to_3d(X_loc_grid, Z_loc_grid, roll_deg)
        ax.plot_surface(X_3d_grid, Y_3d_grid, Z_3d_grid, facecolors=WORKSPACE_COLORS, shade=False, zorder=0)

    # --------------------------------------------------
    # Draw Leg Mechanics in 3D
    # --------------------------------------------------
    if C_loc is not None:
        # Convert local 2D joints to 3D
        A_3d = rotate_to_3d(A_loc[0], A_loc[1], roll_deg)
        B_3d = rotate_to_3d(B_loc[0], B_loc[1], roll_deg)
        C_3d = rotate_to_3d(C_loc[0], C_loc[1], roll_deg)

        # Plot Links
        ax.plot3D([A_3d[0], C_3d[0]], [A_3d[1], C_3d[1]], [A_3d[2], C_3d[2]], lw=4, color='blue', label=f'LS1 ({LS1})')
        ax.plot3D([C_3d[0], B_3d[0]], [C_3d[1], B_3d[1]], [C_3d[2], B_3d[2]], lw=4, color='green', label=f'LS2 ({LS2})')
        ax.plot3D([A_3d[0], B_3d[0]], [A_3d[1], B_3d[1]], [A_3d[2], B_3d[2]], lw=2, color='black', linestyle='--')

        # Plot Joints
        ax.scatter3D(*A_3d, color='red', s=80, zorder=5)
        ax.scatter3D(*C_3d, color='red', s=80, zorder=5)
        ax.scatter3D(*B_3d, color='orange', s=140, zorder=6, label='End Effector B')

        ax.text(A_3d[0], A_3d[1], A_3d[2] + 0.5, " A", weight='bold')
        ax.text(C_3d[0], C_3d[1], C_3d[2] + 0.5, " C", weight='bold')
        ax.text(B_3d[0], B_3d[1], B_3d[2] + 0.5, " B", weight='bold')

        # Calculate Angles
        cos_alpha = (LS1 ** 2 + extension ** 2 - LS2 ** 2) / (2 * LS1 * extension)
        alpha = math.degrees(math.acos(np.clip(cos_alpha, -1.0, 1.0)))

        cos_beta = (LS1 ** 2 + LS2 ** 2 - extension ** 2) / (2 * LS1 * LS2)
        beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))

        # Draw Arcs in 3D
        arc_x, arc_y, arc_z, lx, ly, lz = generate_arc_3d(A_loc, C_loc, B_loc, min(LS1, extension) * 0.3, roll_deg)
        ax.plot3D(arc_x, arc_y, arc_z, color='purple', lw=2)
        ax.text(lx, ly, lz, f'α={alpha:.1f}°', color='purple', weight='bold')

        arc_x, arc_y, arc_z, lx, ly, lz = generate_arc_3d(C_loc, A_loc, B_loc, min(LS1, LS2) * 0.3, roll_deg)
        ax.plot3D(arc_x, arc_y, arc_z, color='darkorange', lw=2)
        ax.text(lx, ly, lz, f'β={beta:.1f}°', color='darkorange', weight='bold')

        ls1_angle = norm_angle(math.degrees(math.atan2(C_loc[1] - A_loc[1], C_loc[0] - A_loc[0])))

        info = (
            f"Ext = {extension:.2f} | Pitch = {angle_deg:.2f}° | Roll = {roll_deg:.2f}°\n"
            f"B Coords: X={B_3d[0]:.1f}, Y={B_3d[1]:.1f}, Z={B_3d[2]:.1f}\n"
            f"Alpha: {alpha:.1f}° | Beta: {beta:.1f}° (lim {BETA_MIN:.0f}°-{BETA_MAX:.0f}°)\n"
            f"LS1 Dir: {ls1_angle:.1f}° (lim {LS1_ANGLE_MIN:.0f}°-{LS1_ANGLE_MAX:.0f}°)"
        )
    else:
        ax.text2D(0.5, 0.5, "Impossible Geometry\nKinematic limit exceeded.",
                  fontsize=13, ha='center', va='center', transform=ax.transAxes, color='darkred')
        info = "Status: Invalid Configuration"

    # Screen UI Text
    ax.text2D(0.02, 0.98, info, transform=ax.transAxes, verticalalignment='top',
              bbox=dict(facecolor='white', alpha=0.85))
    ax.set_title("3D Dog Leg Simulator (Left Click & Drag to rotate Camera)")

    # Format 3D Box perfectly cubic
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X (Forward/Back)")
    ax.set_ylabel("Y (Left/Right)")
    ax.set_zlabel("Z (Up/Down)")

    if C_loc is not None: ax.legend(loc='upper right', fontsize=8)

    # Restore Camera Angle
    if first_draw:
        ax.view_init(elev=20, azim=-35)  # Good default isometric view
        first_draw = False
    else:
        ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw_idle()


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

def slider_update(val):
    draw(extension_slider.val, angle_slider.val, roll_slider.val)


extension_slider.on_changed(slider_update)
angle_slider.on_changed(slider_update)
roll_slider.on_changed(slider_update)


def animate_motion(event):
    start_coord = parse_coordinate_3d(text_start.text)
    stop_coord = parse_coordinate_3d(text_stop.text)

    if start_coord is None or stop_coord is None:
        print("Invalid coordinates! Use format 'X, Y, Z' (e.g., '10, 5, -10')")
        return

    frames = 40
    x_vals = np.linspace(start_coord[0], stop_coord[0], frames)
    y_vals = np.linspace(start_coord[1], stop_coord[1], frames)
    z_vals = np.linspace(start_coord[2], stop_coord[2], frames)

    for x, y, z in zip(x_vals, y_vals, z_vals):
        # Inverse mapping: Global 3D to Local 2D + Roll
        # Z_local must be negative since leg points downward
        z_loc = -np.sqrt(y ** 2 + z ** 2)
        x_loc = x

        # Calculate roll based on requested Y out-of-plane and Z depth
        # Handles edge cases at zero
        if y == 0 and z == 0:
            roll = 0
        else:
            roll = np.degrees(np.arctan2(y, -z))

        roll = np.clip(roll, ROLL_MIN, ROLL_MAX)

        extension = np.hypot(x_loc, z_loc)
        extension = np.clip(extension, MIN_EXT, MAX_EXT)

        angle = np.degrees(np.arctan2(z_loc, x_loc)) + 90

        # Updating sliders drives the `draw()`
        extension_slider.set_val(extension)
        angle_slider.set_val(angle)
        roll_slider.set_val(roll)

        plt.pause(0.01)


btn_perform.on_clicked(animate_motion)

# --------------------------------------------------
# Initial draw
# --------------------------------------------------

draw(extension_slider.val, angle_slider.val, roll_slider.val)
plt.show()
