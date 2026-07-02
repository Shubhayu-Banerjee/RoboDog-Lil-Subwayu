import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.patches import Arc, Wedge

# --------------------------------------------------
# Fixed lengths
# --------------------------------------------------

LS1 = 8.0
LS2 = 12.0

# --------------------------------------------------
# Boundary limits (for the AB "extension" distance)
# --------------------------------------------------
MIN_EXT = abs(LS1 - LS2) + 0.01
MAX_EXT = LS1 + LS2 - 0.01
INITIAL_EXT = max(MIN_EXT, min(10.0, MAX_EXT))

# --------------------------------------------------
# Joint Constraints
# --------------------------------------------------
# LS1 Absolute Rotation Limits
LS1_ANGLE_MIN = 210.0
LS1_ANGLE_MAX = 30.0

# Beta (Angle at C, between LS1 and LS2) Limits
BETA_MIN = 20.0
BETA_MAX = 150.0

# --------------------------------------------------
# Figure setup
# --------------------------------------------------

# Widened figure to comfortably fit all widgets in a single row
fig, ax = plt.subplots(figsize=(13, 10))
# Reduced bottom margin since we only have one row of widgets now
plt.subplots_adjust(bottom=0.18) 

# --------------------------------------------------
# UI Widgets (Aligned in one horizontal line)
# Format: [left, bottom, width, height]
# --------------------------------------------------
Y_POS = 0.08
HEIGHT = 0.03

ax_ext = plt.axes([0.10, Y_POS, 0.15, HEIGHT])
extension_slider = Slider(
    ax_ext, "Extension", MIN_EXT, MAX_EXT, valinit=INITIAL_EXT
)

ax_angle = plt.axes([0.35, Y_POS, 0.15, HEIGHT])
angle_slider = Slider(
    ax_angle, "Angle (deg)", -180, 180, valinit=0
)

ax_box_start = plt.axes([0.58, Y_POS, 0.08, HEIGHT])
text_start = TextBox(ax_box_start, 'Start (X,Y): ', initial="0, -15")

ax_box_stop = plt.axes([0.73, Y_POS, 0.08, HEIGHT])
text_stop = TextBox(ax_box_stop, 'Stop (X,Y): ', initial="15, 0")

ax_button = plt.axes([0.83, Y_POS, 0.12, HEIGHT])
btn_perform = Button(ax_button, 'Perform Motion')

# --------------------------------------------------
# Global state
# --------------------------------------------------

dragging_B = False
B_point = None

# We use this to lock the kinematic branch (elbow state) permanently.
active_branch = None  

# Global variable to hold our pre-computed background map
WORKSPACE_IMG = None  

# Flag to handle zooming/panning preservation
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

def find_apex_candidates(A, B, r1, r2):
    d = np.linalg.norm(B - A)
    if d == 0 or d >= r1 + r2 or d <= abs(r1 - r2):
        return []

    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = r1 ** 2 - a ** 2
    if h_sq < 0:
        h_sq = 0.0
    h = np.sqrt(h_sq)
    P = A + a * (B - A) / d

    perp = np.array([-(B[1] - A[1]), B[0] - A[0]]) / d
    
    C1 = P + h * perp
    C2 = P - h * perp
    return [C1, C2]


def find_valid_apex(A, B, r1, r2):
    global active_branch
    candidates = find_apex_candidates(A, B, r1, r2)
    if not candidates:
        return None, []

    d = np.linalg.norm(B - A)
    cos_beta = (r1 ** 2 + r2 ** 2 - d ** 2) / (2 * r1 * r2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = math.degrees(math.acos(cos_beta))

    if not (BETA_MIN <= beta <= BETA_MAX):
        return None, candidates

    if active_branch is not None:
        intended_c = candidates[active_branch]
        phi = norm_angle(math.degrees(math.atan2(intended_c[1] - A[1], intended_c[0] - A[0])))
        if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
            return intended_c, candidates
        else:
            return None, candidates
    else:
        for i, c in enumerate(candidates):
            phi = norm_angle(math.degrees(math.atan2(c[1] - A[1], c[0] - A[0])))
            if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
                active_branch = i 
                return c, candidates
        return None, candidates


def parse_coordinate(text_val):
    try:
        clean = text_val.replace('(', '').replace(')', '').strip()
        parts = clean.split(',')
        if len(parts) != 2: return None
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def draw_angle_arc(vertex, dir1_pt, dir2_pt, radius, color, label):
    ang1 = norm_angle(math.degrees(math.atan2(dir1_pt[1] - vertex[1], dir1_pt[0] - vertex[0])))
    ang2 = norm_angle(math.degrees(math.atan2(dir2_pt[1] - vertex[1], dir2_pt[0] - vertex[0])))

    diff = (ang2 - ang1) % 360
    if diff > 180: diff -= 360 

    start = ang1
    end = ang1 + diff
    if end < start: start, end = end, start

    arc = Arc(vertex, 2 * radius, 2 * radius, angle=0,
              theta1=start, theta2=end, color=color, lw=2)
    ax.add_patch(arc)

    mid_angle = math.radians(start + (end - start) / 2)
    label_r = radius * 1.35
    lx = vertex[0] + label_r * math.cos(mid_angle)
    ly = vertex[1] + label_r * math.sin(mid_angle)
    ax.text(lx, ly, label, color=color, fontsize=10, fontweight='bold', ha='center', va='center')


# --------------------------------------------------
# Draw
# --------------------------------------------------

def draw(extension, angle_deg):
    global B_point, WORKSPACE_IMG, first_draw
    
    # Save current zoom/pan limits before clearing the axis
    if not first_draw:
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()

    ax.clear()

    A = np.array([0.0, 0.0])
    theta = np.radians(angle_deg - 90)

    B = np.array([
        extension * np.cos(theta),
        extension * np.sin(theta)
    ])

    B_point = B.copy()
    limit = LS1 + LS2 + 5

    wedge_max = LS1_ANGLE_MAX if LS1_ANGLE_MAX >= LS1_ANGLE_MIN else LS1_ANGLE_MAX + 360.0
    wedge = Wedge(A, LS1 * 1.15, LS1_ANGLE_MIN, wedge_max,
                  facecolor='blue', alpha=0.08, edgecolor='blue',
                  linestyle='--', linewidth=1)
    ax.add_patch(wedge)

    C, candidates = find_valid_apex(A, B, LS1, LS2)

    # Calculate workspace background once
    if WORKSPACE_IMG is None and active_branch is not None:
        grid_res = 400
        x_val = np.linspace(-limit, limit, grid_res)
        y_val = np.linspace(-limit, limit, grid_res)
        X, Y = np.meshgrid(x_val, y_val)
        
        R = np.hypot(X, Y)
        R_safe = np.where(R == 0, 1e-6, R)
        
        mask_ext = (R >= MIN_EXT) & (R <= MAX_EXT)
        
        cos_beta_grid = (LS1**2 + LS2**2 - R**2) / (2 * LS1 * LS2)
        beta_grid = np.degrees(np.arccos(np.clip(cos_beta_grid, -1.0, 1.0)))
        mask_beta = (beta_grid >= BETA_MIN) & (beta_grid <= BETA_MAX)
        
        a = (LS1**2 - LS2**2 + R**2) / (2 * R_safe)
        h_sq = LS1**2 - a**2
        h = np.sqrt(np.maximum(0, h_sq))
        
        Px = a * X / R_safe
        Py = a * Y / R_safe
        perpx = -Y / R_safe
        perpy = X / R_safe
        
        if active_branch == 0:
            Cx = Px + h * perpx
            Cy = Py + h * perpy
        else:
            Cx = Px - h * perpx
            Cy = Py - h * perpy
            
        phi = np.degrees(np.arctan2(Cy, Cx)) % 360.0
        
        if LS1_ANGLE_MIN <= LS1_ANGLE_MAX:
            mask_phi = (phi >= LS1_ANGLE_MIN) & (phi <= LS1_ANGLE_MAX)
        else:
            mask_phi = (phi >= LS1_ANGLE_MIN) | (phi <= LS1_ANGLE_MAX)
            
        valid_space = mask_ext & mask_beta & mask_phi
        
        WORKSPACE_IMG = np.zeros((grid_res, grid_res, 4))
        WORKSPACE_IMG[~valid_space] = [1.0, 0.0, 0.0, 0.12]
        WORKSPACE_IMG[valid_space]  = [0.0, 1.0, 0.0, 0.15]

    if WORKSPACE_IMG is not None:
        ax.imshow(WORKSPACE_IMG, extent=(-limit, limit, -limit, limit), origin='lower', zorder=0)

    # Geometry calculations for textual output and arcs
    if C is not None:
        cos_alpha = (LS1 ** 2 + extension ** 2 - LS2 ** 2) / (2 * LS1 * extension)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = math.degrees(math.acos(cos_alpha))

        cos_beta = (LS1 ** 2 + LS2 ** 2 - extension ** 2) / (2 * LS1 * LS2)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = math.degrees(math.acos(cos_beta))

        ax.plot([A[0], C[0]], [A[1], C[1]], lw=3, color='blue', label=f'LS1 ({LS1})')
        ax.plot([C[0], B[0]], [C[1], B[1]], lw=3, color='green', label=f'LS2 ({LS2})')
        ax.plot([A[0], B[0]], [A[1], B[1]], lw=2, color='black', label='Extension')

        ax.scatter(A[0], A[1], color='red', s=80, zorder=5)
        ax.scatter(C[0], C[1], color='red', s=80, zorder=5)
        ax.scatter(B[0], B[1], color='orange', s=140, zorder=6, label='Drag B')

        ax.text(A[0] - 0.4, A[1] - 0.4, "A")
        ax.text(B[0] + 0.2, B[1] + 0.2, "B")
        ax.text(C[0] + 0.2, C[1] + 0.2, "C")

        arc_radius_alpha = min(LS1, extension) * 0.3
        arc_radius_beta = min(LS1, LS2) * 0.3
        
        draw_angle_arc(A, C, B, arc_radius_alpha, 'purple', f'α={alpha:.1f}°')
        draw_angle_arc(C, A, B, arc_radius_beta, 'darkorange', f'β={beta:.1f}°')

        ls1_angle = norm_angle(math.degrees(math.atan2(C[1] - A[1], C[0] - A[0])))
        info = (
            f"Extension = {extension:.2f}\n"
            f"Angle = {angle_deg:.2f}°\n"
            f"X = {B[0]:.2f}, Y = {B[1]:.2f}\n"
            f"Alpha (at A) = {alpha:.2f}°\n"
            f"Beta (at C) = {beta:.2f}° (limit {BETA_MIN:.0f}°-{BETA_MAX:.0f}°)\n"
            f"LS1 direction = {ls1_angle:.2f}° (limit {LS1_ANGLE_MIN:.0f}°-{LS1_ANGLE_MAX:.0f}°)"
        )
    else:
        if not candidates:
            msg = "Impossible Geometry\n(links can't reach: check extension)"
        else:
            msg = ("Impossible Geometry\n"
                   f"Kinematic constraint violated\n"
                   f"(Check LS1 limits or Beta {BETA_MIN}°-{BETA_MAX}° limits)")
        ax.text(0.5, 0.5, msg, fontsize=13, ha='center', va='center', transform=ax.transAxes, color='darkred')
        info = "Status: Invalid Configuration"

    ax.text(
        0.02, 0.98, info, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.85)
    )

    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title("Background Shading: Green = Accessible, Red = Inaccessible")
    if C is not None: ax.legend(loc='upper right', fontsize=8)

    # Restore the zoom/pan limits, or set defaults if it's the very first draw
    if first_draw:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        first_draw = False
    else:
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)

    fig.canvas.draw_idle()


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

def slider_update(val):
    draw(extension_slider.val, angle_slider.val)

extension_slider.on_changed(slider_update)
angle_slider.on_changed(slider_update)


def animate_motion(event):
    start_coord = parse_coordinate(text_start.text)
    stop_coord = parse_coordinate(text_stop.text)

    if start_coord is None or stop_coord is None:
        print("Invalid coordinates! Please use format 'X, Y' (e.g., '10, -5')")
        return

    frames = 40 

    x_vals = np.linspace(start_coord[0], stop_coord[0], frames)
    y_vals = np.linspace(start_coord[1], stop_coord[1], frames)

    for x, y in zip(x_vals, y_vals):
        extension = np.sqrt(x * x + y * y)
        extension = np.clip(extension, MIN_EXT, MAX_EXT)
        angle = np.degrees(np.arctan2(y, x)) + 90

        extension_slider.set_val(extension)
        angle_slider.set_val(angle)

        plt.pause(0.01)

btn_perform.on_clicked(animate_motion)


# --------------------------------------------------
# Mouse events (Dragging)
# --------------------------------------------------

def on_press(event):
    global dragging_B
    # Prevent dragging if the user is using the pan/zoom tool!
    if fig.canvas.manager.toolbar.mode != '': return
    
    if event.inaxes != ax: return
    if B_point is None: return

    click = np.array([event.xdata, event.ydata])
    distance = np.linalg.norm(click - B_point)
    
    # Scale drag sensitivity based on current zoom level
    xlim = ax.get_xlim()
    current_view_width = xlim[1] - xlim[0]
    drag_tolerance = current_view_width * 0.03
    
    if distance < drag_tolerance: dragging_B = True


def on_release(event):
    global dragging_B
    dragging_B = False


def on_motion(event):
    global dragging_B
    if not dragging_B: return
    if event.xdata is None or event.ydata is None: return

    x, y = event.xdata, event.ydata
    extension = np.sqrt(x * x + y * y)
    extension = np.clip(extension, MIN_EXT, MAX_EXT)
    angle = np.degrees(np.arctan2(y, x)) + 90

    extension_slider.set_val(extension)
    angle_slider.set_val(angle)


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# --------------------------------------------------
# Initial draw
# --------------------------------------------------

draw(extension_slider.val, angle_slider.val)
plt.show()
