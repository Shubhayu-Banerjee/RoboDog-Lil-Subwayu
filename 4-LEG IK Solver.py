import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.spatial import ConvexHull

# --------------------------------------------------
# Fixed Joint & Roll Constraints
# --------------------------------------------------
LS1 = 12.0
LS2 = 14.0

LS1_ANGLE_MIN = 210.0
LS1_ANGLE_MAX = 30.0

BETA_MIN = 20.0
BETA_MAX = 150.0

ROLL_MIN = -45.0
ROLL_MAX = 45.0

CHASSIS_W = 15.0
CHASSIS_L = 35.0
CHASSIS_H = 10.0

MIN_EXT = abs(LS1 - LS2) + 0.01
MAX_EXT = LS1 + LS2 - 0.01

# --------------------------------------------------
# Leg Workspace / Stepping Behavior
# --------------------------------------------------
WORKSPACE_TRIGGER_RADIUS = CHASSIS_L / 5.0
STEP_TARGET_RADIUS_FRACTION = 0.85
BODY_STEP_SIZE = 2.5
STEP_ANIM_FRAMES = 15
STEP_ANIM_PAUSE = 0.015
STEP_LIFT_HEIGHT = 3.0

OFFSET_X = 0.85
OFFSET_Y = 1.25

# --------------------------------------------------
# Figure & Axis Setup
# --------------------------------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.30)

ax_height = plt.axes([0.12, 0.06, 0.45, 0.03])
height_slider = Slider(ax_height, "Chassis Height (Z)", MIN_EXT + 0.5, MAX_EXT - 0.5, valinit=11.0)

ax_fwd = plt.axes([0.85, 0.1475, 0.08, 0.05])
ax_back = plt.axes([0.85, 0.0275, 0.08, 0.05])
ax_left = plt.axes([0.76, 0.0875, 0.08, 0.05])
ax_right = plt.axes([0.94, 0.0875, 0.08, 0.05])
ax_stance = plt.axes([0.85, 0.0875, 0.08, 0.05])

btn_fwd = Button(ax_fwd, "Fwd")
btn_back = Button(ax_back, "Back")
btn_left = Button(ax_left, "Left")
btn_right = Button(ax_right, "Right")
btn_stance = Button(ax_stance, "Stance")

first_draw = True
stability_warning = False

body = {'x': 0.0, 'y': 0.0, 'z': 11.0}

attach_dx = (CHASSIS_L / 2) * OFFSET_X
attach_dy = (CHASSIS_W / 2) * OFFSET_Y

foot_positions = {
    "FR": np.array([attach_dx, -attach_dy, 0.0]),
    "FL": np.array([attach_dx, attach_dy, 0.0]),
    "RR": np.array([-attach_dx, -attach_dy, 0.0]),
    "RL": np.array([-attach_dx, attach_dy, 0.0]),
}


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


def get_chassis_vertices(center_x, center_y, center_z):
    dx, dy, dz = CHASSIS_L / 2, CHASSIS_W / 2, CHASSIS_H / 2
    return np.array([
        [center_x - dx, center_y - dy, center_z - dz], [center_x + dx, center_y - dy, center_z - dz],
        [center_x + dx, center_y + dy, center_z - dz], [center_x - dx, center_y + dy, center_z - dz],
        [center_x - dx, center_y - dy, center_z + dz], [center_x + dx, center_y - dy, center_z + dz],
        [center_x + dx, center_y + dy, center_z + dz], [center_x - dx, center_y + dy, center_z + dz]
    ])


def get_hip_positions(bx, by, bz):
    return {
        "FR": np.array([bx + attach_dx, by - attach_dy, bz]),
        "FL": np.array([bx + attach_dx, by + attach_dy, bz]),
        "RR": np.array([bx - attach_dx, by - attach_dy, bz]),
        "RL": np.array([bx - attach_dx, by + attach_dy, bz]),
    }


def solve_leg_local_ik(x_loc, z_loc):
    B = np.array([x_loc, z_loc])
    d = np.linalg.norm(B)

    if d == 0 or d >= LS1 + LS2 or d <= abs(LS1 - LS2):
        return None

    cos_beta = (LS1 ** 2 + LS2 ** 2 - d ** 2) / (2 * LS1 * LS2)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))
    if not (BETA_MIN <= beta <= BETA_MAX):
        return None

    a = (LS1 ** 2 - LS2 ** 2 + d ** 2) / (2 * d)
    h_sq = LS1 ** 2 - a ** 2
    h = np.sqrt(max(0, h_sq))

    P = a * B / d
    perp = np.array([-B[1], B[0]]) / d

    candidates = [P + h * perp, P - h * perp]

    for c in candidates:
        phi = norm_angle(math.degrees(math.atan2(c[1], c[0])))
        if is_angle_in_range(phi, LS1_ANGLE_MIN, LS1_ANGLE_MAX):
            return c

    return None


def get_triangle_margin(p, a, b, c):
    """
    Computes the shortest distance from point p to the edges of triangle abc.
    Returns >0 if inside, <0 if outside.
    """
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    is_ccw = cross > 0

    def sign_dist(pt, v1, v2):
        num = (pt[0] - v1[0]) * (v2[1] - v1[1]) - (pt[1] - v1[1]) * (v2[0] - v1[0])
        den = np.linalg.norm(v2 - v1)
        dist = num / den if den > 1e-6 else 0.0
        return dist if is_ccw else -dist

    d1 = sign_dist(p, a, b)
    d2 = sign_dist(p, b, c)
    d3 = sign_dist(p, c, a)
    return min(d1, d2, d3)


# --------------------------------------------------
# Animation & Motion Helpers
# --------------------------------------------------
def shift_body_to(target_x, target_y, frames=10):
    start_x, start_y = body['x'], body['y']
    for frame in range(frames + 1):
        t = frame / frames
        body['x'] = start_x + (target_x - start_x) * t
        body['y'] = start_y + (target_y - start_y) * t
        draw(body['x'], body['y'], body['z'])
        plt.pause(STEP_ANIM_PAUSE)


def step_leg(chosen, target):
    start = foot_positions[chosen][:2].copy()
    for frame in range(STEP_ANIM_FRAMES + 1):
        t = frame / STEP_ANIM_FRAMES
        eased_t = 0.5 - 0.5 * math.cos(math.pi * t)
        lift_t = math.sin(math.pi * t)

        xy = start + (target - start) * eased_t
        foot_positions[chosen] = np.array([xy[0], xy[1], STEP_LIFT_HEIGHT * lift_t])

        draw(body['x'], body['y'], body['z'])
        plt.pause(STEP_ANIM_PAUSE)

    foot_positions[chosen] = np.array([target[0], target[1], 0.0])
    draw(body['x'], body['y'], body['z'])


def compute_step_target(hip_xy, foot_xy, cog_xy, radius_fraction=STEP_TARGET_RADIUS_FRACTION):
    outward = hip_xy - cog_xy
    dist = np.linalg.norm(outward)
    if dist < 1e-6:
        direction = foot_xy - hip_xy
        d2 = np.linalg.norm(direction)
        unit = direction / d2 if d2 > 1e-6 else np.array([1.0, 0.0])
    else:
        unit = outward / dist
    return hip_xy + unit * (WORKSPACE_TRIGGER_RADIUS * radius_fraction)


def try_pre_position_leg(chosen, target_cog_xy, hips, curr_cog):
    """
    Actively scans the perimeters of the remaining legs' workspaces.
    If it finds a foot placement that secures the COG inside the triangle,
    it takes the step and returns True.
    """
    others = [n for n in foot_positions if n != chosen]

    best_helper = None
    best_target = None
    best_margin = 0.05  # Require at least a tiny physical buffer to consider it safe

    for helper in others:
        helper_others = [foot_positions[n][:2] for n in foot_positions if n != helper]

        # 1. Can we safely lift this helper leg to move it?
        helper_margin = get_triangle_margin(curr_cog, helper_others[0], helper_others[1], helper_others[2])
        if helper_margin < 0.05:
            continue

        hip_xy = hips[helper][:2]
        scan_radius = WORKSPACE_TRIGGER_RADIUS * 0.95

        # 2. Radially scan 16 points around the edge of the helper's workspace
        for angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            test_target = hip_xy + np.array([math.cos(angle) * scan_radius, math.sin(angle) * scan_radius])

            # Simulate placing the helper at this test target
            simulated_others = []
            for n in others:
                if n == helper:
                    simulated_others.append(test_target)
                else:
                    simulated_others.append(foot_positions[n][:2])

            # 3. Does this new configuration solve the original problem for 'chosen'?
            sim_margin = get_triangle_margin(curr_cog, simulated_others[0], simulated_others[1], simulated_others[2])

            # Keep track of the placement that provides the deepest, safest COG margin
            if sim_margin > best_margin:
                best_margin = sim_margin
                best_helper = helper
                best_target = test_target

    if best_helper is not None:
        step_leg(best_helper, best_target)
        return True

    return False


# --------------------------------------------------
# Core Movement Logic
# --------------------------------------------------
def animate_leg_steps():
    global stability_warning
    stability_warning = False

    target_body_x = body['x']
    target_body_y = body['y']
    target_cog_xy = np.array([target_body_x, target_body_y])

    intended_hips = get_hip_positions(target_body_x, target_body_y, body['z'])
    hips = get_hip_positions(body['x'], body['y'], body['z'])
    curr_cog = np.array([body['x'], body['y']])

    pending = []
    for name, hip in hips.items():
        hip_xy = hip[:2]
        foot_xy = foot_positions[name][:2]

        overshoot_workspace = np.linalg.norm(foot_xy - hip_xy) - WORKSPACE_TRIGGER_RADIUS
        ideal_target = compute_step_target(hip_xy, foot_xy, curr_cog, radius_fraction=STEP_TARGET_RADIUS_FRACTION)
        overshoot_ideal = np.linalg.norm(foot_xy - ideal_target) - WORKSPACE_TRIGGER_RADIUS

        overshoot = max(overshoot_workspace, overshoot_ideal)

        if overshoot > 0:
            pending.append(name)

    if not pending:
        draw(body['x'], body['y'], body['z'])
        return

    gait_order = {"FL": 1, "RR": 2, "FR": 3, "RL": 4}
    pending.sort(key=lambda n: gait_order[n])

    EDGE_THRESHOLD = 0.5

    for chosen in pending:
        cog_xy = np.array([body['x'], body['y']])
        others_xy = [foot_positions[n][:2] for n in foot_positions if n != chosen]

        margin = get_triangle_margin(cog_xy, others_xy[0], others_xy[1], others_xy[2])

        if margin < EDGE_THRESHOLD:
            if margin >= 0:
                # COG is technically inside, but right on the edge. Nudge in 20%.
                stability_warning = True
                centroid = np.mean(others_xy, axis=0)
                safe_cog = cog_xy + (centroid - cog_xy) * 0.2
                shift_body_to(safe_cog[0], safe_cog[1])
            else:
                # COG is outside (unstable). Try an active radial scan pre-positioning!
                hips = get_hip_positions(body['x'], body['y'], body['z'])
                success = try_pre_position_leg(chosen, target_cog_xy, hips, cog_xy)

                if not success:
                    # Deadlocked (no geometric solution found). Revert to compensatory sway.
                    stability_warning = True
                    centroid = np.mean(others_xy, axis=0)
                    safe_cog = cog_xy + (centroid - cog_xy) * 0.6
                    shift_body_to(safe_cog[0], safe_cog[1])

        # Step the chosen leg
        target = compute_step_target(intended_hips[chosen][:2], foot_positions[chosen][:2], target_cog_xy)
        step_leg(chosen, target)

    # Restore Body to Intended Target
    curr_cog = np.array([body['x'], body['y']])
    if np.linalg.norm(curr_cog - target_cog_xy) > 0.01:
        shift_body_to(target_cog_xy[0], target_cog_xy[1])


def on_stance(event):
    global stability_warning
    stability_warning = False
    gait_order = ["FL", "RR", "FR", "RL"]

    target_body_x = body['x']
    target_body_y = body['y']
    target_cog_xy = np.array([target_body_x, target_body_y])
    intended_hips = get_hip_positions(target_body_x, target_body_y, body['z'])

    EDGE_THRESHOLD = 0.5

    for chosen in gait_order:
        cog_xy = np.array([body['x'], body['y']])
        others_xy = [foot_positions[n][:2] for n in foot_positions if n != chosen]

        margin = get_triangle_margin(cog_xy, others_xy[0], others_xy[1], others_xy[2])

        if margin < EDGE_THRESHOLD:
            centroid = np.mean(others_xy, axis=0)
            if margin >= 0:
                safe_cog = cog_xy + (centroid - cog_xy) * 0.2
            else:
                safe_cog = cog_xy + (centroid - cog_xy) * 0.6
            shift_body_to(safe_cog[0], safe_cog[1])

        target = compute_step_target(intended_hips[chosen][:2], foot_positions[chosen][:2], target_cog_xy,
                                     radius_fraction=0.95)
        step_leg(chosen, target)

    all_feet = [foot_positions[n][:2] for n in foot_positions]
    center_cog = np.mean(all_feet, axis=0)
    shift_body_to(center_cog[0], center_cog[1])


# --------------------------------------------------
# Main Renderer Core
# --------------------------------------------------
def draw(body_x, body_y, body_height):
    global first_draw
    if not first_draw:
        elev, azim = ax.elev, ax.azim
    ax.clear()

    grid_lim = 25
    g_x, g_y = np.meshgrid(np.linspace(-grid_lim, grid_lim, 7), np.linspace(-grid_lim, grid_lim, 7))
    g_z = np.zeros_like(g_x)
    ax.plot_wireframe(g_x, g_y, g_z, color='gray', alpha=0.1, linestyle=':')

    v = get_chassis_vertices(body_x, body_y, body_height)
    cog = np.mean(v, axis=0)
    ax.plot3D([cog[0], cog[0]], [cog[1], cog[1]], [cog[2], 0], color='red', linestyle=':', lw=2, label='COG')

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for edge in edges:
        ax.plot3D([v[edge[0]][0], v[edge[1]][0]], [v[edge[0]][1], v[edge[1]][1]], [v[edge[0]][2], v[edge[1]][2]],
                  color='black', lw=2)

    hips = get_hip_positions(body_x, body_y, body_height)
    valid_feet = []
    any_failed = False
    theta = np.linspace(0, 2 * np.pi, 40)

    for name, hip_pos in hips.items():
        foot_pos = foot_positions[name]

        circ_x = hip_pos[0] + WORKSPACE_TRIGGER_RADIUS * np.cos(theta)
        circ_y = hip_pos[1] + WORKSPACE_TRIGGER_RADIUS * np.sin(theta)
        ax.plot3D(circ_x, circ_y, np.zeros_like(circ_x), color='steelblue', alpha=0.25, lw=1)
        ax.plot3D([hip_pos[0], hip_pos[0]], [hip_pos[1], hip_pos[1]], [hip_pos[2], 0],
                  color='steelblue', linestyle='--', alpha=0.35, lw=1)

        delta_y = foot_pos[1] - hip_pos[1]
        delta_z = foot_pos[2] - hip_pos[2]

        roll_rad = math.atan2(delta_y, -delta_z)
        roll_deg = math.degrees(roll_rad)

        if not (ROLL_MIN <= roll_deg <= ROLL_MAX):
            any_failed = True
            ax.plot3D([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]], [hip_pos[2], foot_pos[2]], color='purple',
                      linestyle=':', alpha=0.5)
            continue

        x_loc = foot_pos[0] - hip_pos[0]
        z_loc = -math.sqrt(delta_y ** 2 + delta_z ** 2)
        C_loc = solve_leg_local_ik(x_loc, z_loc)

        if C_loc is not None:
            k_x = C_loc[0]
            k_y = -C_loc[1] * math.sin(roll_rad)
            k_z = C_loc[1] * math.cos(roll_rad)

            knee_3d = hip_pos + np.array([k_x, k_y, k_z])

            ax.plot3D([hip_pos[0], knee_3d[0]], [hip_pos[1], knee_3d[1]], [hip_pos[2], knee_3d[2]], 'blue', lw=3)
            ax.plot3D([knee_3d[0], foot_pos[0]], [knee_3d[1], foot_pos[1]], [knee_3d[2], foot_pos[2]], 'green', lw=3)

            ax.scatter3D(*hip_pos, color='darkred', s=40)
            ax.scatter3D(*knee_3d, color='red', s=30)
            foot_color = 'gold' if foot_pos[2] > 0.05 else 'orange'
            ax.scatter3D(*foot_pos, color=foot_color, s=50, zorder=5)

            if foot_pos[2] < 0.05:
                valid_feet.append(foot_pos)
        else:
            any_failed = True
            ax.plot3D([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]], [hip_pos[2], foot_pos[2]], 'red',
                      linestyle='--', alpha=0.5)

    if len(valid_feet) >= 3:
        pts = np.array([f[:2] for f in valid_feet])
        hull = ConvexHull(pts)
        for simplex in hull.simplices:
            ax.plot3D(pts[simplex, 0], pts[simplex, 1], [0, 0], 'orange', lw=2.5, alpha=0.8)

    if any_failed:
        ax.text2D(0.5, 0.45, "Kinematic / Roll Limits Exceeded!", color='darkred',
                  transform=ax.transAxes, ha='center', va='center', weight='bold', fontsize=12)

    if stability_warning:
        ax.text2D(0.5, 0.40, "Compensatory Sway Active (COG shifted)", color='goldenrod',
                  transform=ax.transAxes, ha='center', va='center', weight='bold', fontsize=10)

    ax.set_title(f"Quadruped Solver  |  Body: ({body_x:.1f}, {body_y:.1f}, {body_height:.1f})")
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 60)

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
# Control Callbacks
# --------------------------------------------------
def on_height_change(val):
    body['z'] = val
    draw(body['x'], body['y'], body['z'])


def push_body(dx, dy):
    body['x'] += dx
    body['y'] += dy
    animate_leg_steps()


def on_fwd(event):
    push_body(BODY_STEP_SIZE, 0.0)


def on_back(event):
    push_body(-BODY_STEP_SIZE, 0.0)


def on_left(event):
    push_body(0.0, -BODY_STEP_SIZE)


def on_right(event):
    push_body(0.0, BODY_STEP_SIZE)


height_slider.on_changed(on_height_change)
btn_fwd.on_clicked(on_fwd)
btn_back.on_clicked(on_back)
btn_left.on_clicked(on_left)
btn_right.on_clicked(on_right)
btn_stance.on_clicked(on_stance)

draw(body['x'], body['y'], body['z'])
plt.show()
