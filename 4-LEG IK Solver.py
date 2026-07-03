import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --------------------------------------------------
# Fixed Joint & Roll Constraints (From original solver)
# --------------------------------------------------
LS1 = 8.0
LS2 = 12.0

LS1_ANGLE_MIN = 210.0
LS1_ANGLE_MAX = 30.0

BETA_MIN = 20.0
BETA_MAX = 150.0

ROLL_MIN = -30.0
ROLL_MAX = 90.0

# Chassis Dimensions
CHASSIS_W = 10.0  
CHASSIS_L = 16.0  
CHASSIS_H = 4.0   

MIN_EXT = abs(LS1 - LS2) + 0.01
MAX_EXT = LS1 + LS2 - 0.01

# --------------------------------------------------
# Figure & Axis Setup
# --------------------------------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# UI Sliders: One for Height (Z), one for Lateral Shift (Y)
ax_height = plt.axes([0.25, 0.12, 0.50, 0.03])
height_slider = Slider(ax_height, "Chassis Height (Z)", MIN_EXT + 0.5, MAX_EXT - 0.5, valinit=11.0)

ax_sideway = plt.axes([0.25, 0.06, 0.50, 0.03])
sideway_slider = Slider(ax_sideway, "Side-to-Side (Y)", -8.0, 8.0, valinit=0.0)

first_draw = True

# --------------------------------------------------
# Kinematics & Transformation Helpers
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

def get_chassis_vertices(center_y, center_z):
    """Calculates 3D box vertices tracking both Y and Z shifts"""
    dx, dy, dz = CHASSIS_L / 2, CHASSIS_W / 2, CHASSIS_H / 2
    return np.array([
        [-dx, center_y - dy, center_z - dz], [ dx, center_y - dy, center_z - dz],
        [ dx, center_y + dy, center_z - dz], [-dx, center_y + dy, center_z - dz],
        [-dx, center_y - dy, center_z + dz], [ dx, center_y - dy, center_z + dz],
        [ dx, center_y + dy, center_z + dz], [-dx, center_y + dy, center_z + dz]
    ])

def solve_leg_local_ik(x_loc, z_loc):
    """Solves original 2D local plane kinematics with Thigh and Beta limits"""
    A = np.array([0.0, 0.0])
    B = np.array([x_loc, z_loc])
    d = np.linalg.norm(B)
    
    if d == 0 or d >= LS1 + LS2 or d <= abs(LS1 - LS2): 
        return None

    cos_beta = (LS1**2 + LS2**2 - d**2) / (2 * LS1 * LS2)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))
    if not (BETA_MIN <= beta <= BETA_MAX): 
        return None

    a = (LS1**2 - LS2**2 + d**2) / (2 * d)
    h_sq = LS1**2 - a**2
    h = np.sqrt(max(0, h_sq))
    
    P = a * B / d
    perp = np.array([-B[1], B[0]]) / d
    
    candidates = [P + h * perp, P - h * perp]
    
    for c in candidates:
        phi = norm_angle(math.degrees(math.atan2(c[1], c[0])))
        if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
            return c  
            
    return None

# --------------------------------------------------
# Main Renderer Core
# --------------------------------------------------
def draw(body_y, body_height):
    global first_draw
    
    if not first_draw:
        elev, azim = ax.elev, ax.azim
    
    ax.clear()
    
    # 1. Ground Plane Grid
    grid_lim = 20
    g_x, g_y = np.meshgrid(np.linspace(-grid_lim, grid_lim, 5), np.linspace(-grid_lim, grid_lim, 5))
    g_z = np.zeros_like(g_x)
    ax.plot_wireframe(g_x, g_y, g_z, color='gray', alpha=0.2, linestyle=':')
    
    # 2. Draw Chassis Box
    v = get_chassis_vertices(body_y, body_height)
    edges = [
        (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), 
        (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
    ]
    for edge in edges:
        ax.plot3D([v[edge[0]][0], v[edge[1]][0]], 
                  [v[edge[0]][1], v[edge[1]][1]], 
                  [v[edge[0]][2], v[edge[1]][2]], color='black', lw=3)
                  
    # 3. Setup Floating Hips vs Static Foot Floor Anchors
    dx, dy = CHASSIS_L / 2, CHASSIS_W / 2
    
    # Hips move with body_y translation
    hips = {
        "FR": np.array([ dx, body_y - dy, body_height]),
        "FL": np.array([ dx, body_y + dy, body_height]),
        "RR": np.array([-dx, body_y - dy, body_height]),
        "RL": np.array([-dx, body_y + dy, body_height])
    }
    
    # Feet stay hard-snapped to their natural neutral layout at Z=0
    feet = {
        "FR": np.array([ dx, -dy, 0.0]),
        "FL": np.array([ dx,  dy, 0.0]),
        "RR": np.array([-dx, -dy, 0.0]),
        "RL": np.array([ -dx,  dy, 0.0])
    }
    
    any_failed = False

    for name, hip_pos in hips.items():
        foot_pos = feet[name]
        
        # Determine current lateral roll offsets
        delta_y = foot_pos[1] - hip_pos[1]
        delta_z = foot_pos[2] - hip_pos[2] # Negative value downwards
        
        # Compute exact Roll Angle (around the X-axis) required to reach the foot
        # Sign handling aligns with original solver: outward vs inward tilt
        roll_rad = math.atan2(delta_y, -delta_z)
        roll_deg = math.degrees(roll_rad)
        
        # Verify calculated roll satisfies structural limitations
        if not (ROLL_MIN <= roll_deg <= ROLL_MAX):
            any_failed = True
            ax.plot3D([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]], [hip_pos[2], foot_pos[2]], color='purple', linestyle=':', alpha=0.6)
            continue
            
        # Extract the remaining length in the leg's swung planar workspace
        # Local Z is the absolute distance on the rotated Y-Z plane
        x_loc = foot_pos[0] - hip_pos[0] # remains 0 because X doesn't translate
        z_loc = -math.sqrt(delta_y**2 + delta_z**2)
        
        # Solve local 2D IK under strict link constraints
        C_loc = solve_leg_local_ik(x_loc, z_loc)
        
        if C_loc is not None:
            # Transform local 2D knee coordinate [C_loc[0], C_loc[1]] back into 3D using the roll angle
            # Matching the original rotate_to_3d matrix physics:
            # y_3d = -z_local * sin(roll) ; z_3d = z_local * cos(roll)
            k_x = C_loc[0]
            k_y = -C_loc[1] * math.sin(roll_rad)
            k_z =  C_loc[1] * math.cos(roll_rad)
            
            knee_3d = hip_pos + np.array([k_x, k_y, k_z])
            
            # Draw valid linkages
            ax.plot3D([hip_pos[0], knee_3d[0]], [hip_pos[1], knee_3d[1]], [hip_pos[2], knee_3d[2]], color='blue', lw=4)
            ax.plot3D([knee_3d[0], foot_pos[0]], [knee_3d[1], foot_pos[1]], [knee_3d[2], foot_pos[2]], color='green', lw=4)
            
            ax.scatter3D(*hip_pos, color='darkred', s=50)
            ax.scatter3D(*knee_3d, color='red', s=40)
            ax.scatter3D(*foot_pos, color='orange', s=60, zorder=5)
        else:
            any_failed = True
            ax.plot3D([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]], [hip_pos[2], foot_pos[2]], color='red', linestyle='--', alpha=0.5)

    if any_failed:
        ax.text2D(0.5, 0.45, "Kinematic / Roll Limits Exceeded!", color='darkred', 
                  transform=ax.transAxes, ha='center', va='center', weight='bold', fontsize=12)

    # Presentation Aesthetics
    ax.set_title("Quadruped Solver: Dynamic Height (Z) & Lateral Shift (Y)")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 20)
    ax.set_xlabel("X (Forward)")
    ax.set_ylabel("Y (Sideways)")
    ax.set_zlabel("Z (Height)")
    
    if first_draw:
        ax.view_init(elev=20, azim=-50)
        first_draw = False
    else:
        ax.view_init(elev=elev, azim=azim)
        
    fig.canvas.draw_idle()

# --------------------------------------------------
# Realtime Updates Slider Connections
# --------------------------------------------------
def update(val):
    draw(sideway_slider.val, height_slider.val)

height_slider.on_changed(update)
sideway_slider.on_changed(update)

# Trigger entry frame
draw(sideway_slider.val, height_slider.val)
plt.show()
