import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.spatial import ConvexHull

# --------------------------------------------------
# Fixed Joint Constraints
# --------------------------------------------------
LS1 = 12.0
LS2 = 14.0

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
# Workspace & Stepping Parameters
# --------------------------------------------------
WORKSPACE_TRIGGER_RADIUS = CHASSIS_L / 5.0
STEP_TARGET_RADIUS_FRACTION = 0.85
BODY_STEP_SIZE = 2.5
STEP_ANIM_FRAMES = 12
STEP_ANIM_PAUSE = 0.01
STEP_LIFT_HEIGHT = 4.0

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

# UI Layout
ax_fwd = plt.axes([0.85, 0.1475, 0.08, 0.05])
ax_back = plt.axes([0.85, 0.0275, 0.08, 0.05])
ax_left = plt.axes([0.76, 0.0875, 0.08, 0.05])
ax_right = plt.axes([0.94, 0.0875, 0.08, 0.05])
ax_stance = plt.axes([0.85, 0.0875, 0.08, 0.05])
ax_walk = plt.axes([0.65, 0.1475, 0.08, 0.05])

btn_fwd = Button(ax_fwd, "Fwd")
btn_back = Button(ax_back, "Back")
btn_left = Button(ax_left, "Left")
btn_right = Button(ax_right, "Right")
btn_stance = Button(ax_stance, "Stance")
btn_walk = Button(ax_walk, "Walk", color='lightgreen')

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
    if d == 0 or d >= LS1 + LS2 or d <= abs(LS1 - LS2): return None

    cos_beta = (LS1 ** 2 + LS2 ** 2 - d ** 2) / (2 * LS1 * LS2)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))
    if not (BETA_MIN <= beta <= BETA_MAX): return None

    a = (LS1 ** 2 - LS2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(0, LS1 ** 2 - a ** 2))
    P = a * B / d
    perp = np.array([-B[1], B[0]]) / d

    candidates = [P + h * perp, P - h * perp]

    # CORE FIX: Force the knee to bend backwards.
    # By picking the candidate with the smallest (most negative) X coordinate,
    # the knee joint will always point behind the extension line.
    candidates.sort(key=lambda c: c[0])
    return candidates[0]


def get_triangle_margin(p, a, b, c):
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    is_ccw = cross > 0

    def sign_dist(pt, v1, v2):
        num = (pt[0] - v1[0]) * (v2[1] - v1[1]) - (pt[1] - v1[1]) * (v2[0] - v1[0])
        den = np.linalg.norm(v2 - v1)
        dist = num / den if den > 1e-6 else 0.0
        return dist if is_ccw else -dist

    return min(sign_dist(p, a, b), sign_dist(p, b, c), sign_dist(p, c, a))


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


def glide_body(body_dx, body_dy, frames=STEP_ANIM_FRAMES):
    """Translates the body at a constant velocity without lifting any legs."""
    start_body_x = body['x']
    start_body_y = body['y']
    for frame in range(frames + 1):
        t = frame / frames
        body['x'] = start_body_x + body_dx * t
        body['y'] = start_body_y + body_dy * t
        draw(body['x'], body['y'], body['z'])
        plt.pause(STEP_ANIM_PAUSE)


def animate_step_continuous(chosen_leg, target_foot_xy, body_dx, body_dy, frames=STEP_ANIM_FRAMES):
    """Translates the body linearly WHILE the leg executes an eased swing curve."""
    start_foot = foot_positions[chosen_leg][:2].copy()
    start_body_x = body['x']
    start_body_y = body['y']

    for frame in range(frames + 1):
        t = frame / frames
        eased_t = 0.5 - 0.5 * math.cos(math.pi * t)

        # Linear body progression (constant velocity)
        body['x'] = start_body_x + body_dx * t
        body['y'] = start_body_y + body_dy * t

        # Eased leg lift
        lift_t = math.sin(math.pi * t)
        foot_positions[chosen_leg][0] = start_foot[0] + (target_foot_xy[0] - start_foot[0]) * eased_t
        foot_positions[chosen_leg][1] = start_foot[1] + (target_foot_xy[1] - start_foot[1]) * eased_t
        foot_positions[chosen_leg][2] = STEP_LIFT_HEIGHT * lift_t

        draw(body['x'], body['y'], body['z'])
        plt.pause(STEP_ANIM_PAUSE)

    foot_positions[chosen_leg][2] = 0.0
    draw(body['x'], body['y'], body['z'])


def step_leg(chosen, target):
    animate_step_continuous(chosen, target, 0.0, 0.0)


# --------------------------------------------------
# Locomotion Controller (Continuous Flow Gait)
# --------------------------------------------------
def execute_walk_cycle(stride_length=12.0, cycles=2):
    global stability_warning
    stability_warning = False

    sequence = ["RL", "FL", "RR", "FR"]
    body_chunk = stride_length / 4.0

    for _ in range(cycles):
        for leg in sequence:

            # PHASE 1: 4-Leg Support Phase (Glide & Prepare Sway)
            glide_dx = body_chunk * 0.4

            future_body_x = body['x'] + glide_dx
            future_body_y = body['y']

            cog_xy = np.array([future_body_x, future_body_y])
            others_xy = [foot_positions[n][:2] for n in foot_positions if n != leg]
            margin = get_triangle_margin(cog_xy, others_xy[0], others_xy[1], others_xy[2])

            if margin < 0.5:
                centroid = np.mean(others_xy, axis=0)
                future_body_y = body['y'] + (centroid[1] - body['y']) * 0.5

            glide_dy = future_body_y - body['y']

            # The body keeps moving forward smoothly before the leg lifts
            glide_body(glide_dx, glide_dy, frames=int(STEP_ANIM_FRAMES * 0.5))

            # PHASE 2: 3-Leg Support Phase (Swing)
            swing_dx = body_chunk * 0.6
            swing_dy = (0.0 - body['y']) * 0.3  # Gradually center the Y axis

            future_hips = get_hip_positions(body['x'] + swing_dx, body['y'] + swing_dy, body['z'])
            hip_xy = future_hips[leg][:2]

            outward_vector = hip_xy - np.array([body['x'] + swing_dx, body['y'] + swing_dy])
            outward_unit = outward_vector / np.linalg.norm(outward_vector)
            base_target = hip_xy + outward_unit * (WORKSPACE_TRIGGER_RADIUS * 0.4)

            # Ensure the foot is planted sufficiently forward of the *future* hip
            target = base_target + np.array([stride_length * 0.65, 0.0])

            # Leg swings while body continues seamlessly
            animate_step_continuous(leg, target, swing_dx, swing_dy, frames=STEP_ANIM_FRAMES)


# --------------------------------------------------
# Reactive Stance Controller
# --------------------------------------------------
def compute_step_target(hip_xy, foot_xy, cog_xy, radius_fraction=STEP_TARGET_RADIUS_FRACTION):
    outward = hip_xy - cog_xy
    dist = np.linalg.norm(outward)
    unit = outward / dist if dist > 1e-6 else np.array([1.0, 0.0])
    return hip_xy + unit * (WORKSPACE_TRIGGER_RADIUS * radius_fraction)


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
        if overshoot_workspace > 0:
            pending.append(name)

    if not pending:
        draw(body['x'], body['y'], body['z'])
        return

    gait_order = {"FL": 1, "RR": 2, "FR": 3, "RL": 4}
    pending.sort(key=lambda n: gait_order[n])

    for chosen in pending:
        cog_xy = np.array([body['x'], body['y']])
        others_xy = [foot_positions[n][:2] for n in foot_positions if n != chosen]
        margin = get_triangle_margin(cog_xy, others_xy[0], others_xy[1], others_xy[2])

        if margin < 0.5:
            stability_warning = True
            centroid = np.mean(others_xy, axis=0)
            safe_cog = cog_xy + (centroid - cog_xy) * (0.2 if margin >= 0 else 0.6)
            shift_body_to(safe_cog[0], safe_cog[1])

        target = compute_step_target(intended_hips[chosen][:2], foot_positions[chosen][:2], target_cog_xy)
        step_leg(chosen, target)

    curr_cog = np.array([body['x'], body['y']])
    if np.linalg.norm(curr_cog - target_cog_xy) > 0.01:
        shift_body_to(target_cog_xy[0], target_cog_xy[1])


# --------------------------------------------------
# Main Renderer Core
# --------------------------------------------------
def draw(body_x, body_y, body_height):
    global first_draw
    if not first_draw:
        elev, azim = ax.elev, ax.azim
    ax.clear()

    grid_lim = 35
    g_x, g_y = np.meshgrid(np.linspace(-grid_lim, grid_lim, 10), np.linspace(-grid_lim, grid_lim, 10))
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
        ax.text2D(0.5, 0.45, "Kinematic Limits Exceeded!", color='darkred', transform=ax.transAxes, ha='center',
                  va='center', weight='bold', fontsize=12)
    if stability_warning:
        ax.text2D(0.5, 0.40, "Compensatory Sway Active", color='goldenrod', transform=ax.transAxes, ha='center',
                  va='center', weight='bold', fontsize=10)

    ax.set_title(f"Quadruped Gait Engine  |  Body: ({body_x:.1f}, {body_y:.1f}, {body_height:.1f})")
    ax.set_box_aspect([1, 1, 1])

    cam_offset = 35
    ax.set_xlim(body_x - cam_offset, body_x + cam_offset)
    ax.set_ylim(-cam_offset, cam_offset)
    ax.set_zlim(0, 60)

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


def on_fwd(event): push_body(BODY_STEP_SIZE, 0.0)


def on_back(event): push_body(-BODY_STEP_SIZE, 0.0)


def on_left(event): push_body(0.0, -BODY_STEP_SIZE)


def on_right(event): push_body(0.0, BODY_STEP_SIZE)


def on_walk(event):
    execute_walk_cycle(stride_length=12.0, cycles=2)


def on_stance(event):
    global stability_warning
    stability_warning = False
    gait_order = ["FL", "RR", "FR", "RL"]
    target_cog_xy = np.array([body['x'], body['y']])
    intended_hips = get_hip_positions(body['x'], body['y'], body['z'])

    for chosen in gait_order:
        cog_xy = np.array([body['x'], body['y']])
        others_xy = [foot_positions[n][:2] for n in foot_positions if n != chosen]
        margin = get_triangle_margin(cog_xy, others_xy[0], others_xy[1], others_xy[2])

        if margin < 0.5:
            centroid = np.mean(others_xy, axis=0)
            safe_cog = cog_xy + (centroid - cog_xy) * 0.4
            shift_body_to(safe_cog[0], safe_cog[1])

        target = compute_step_target(intended_hips[chosen][:2], foot_positions[chosen][:2], target_cog_xy,
                                     radius_fraction=0.95)
        step_leg(chosen, target)

    all_feet = [foot_positions[n][:2] for n in foot_positions]
    center_cog = np.mean(all_feet, axis=0)
    shift_body_to(center_cog[0], center_cog[1])


height_slider.on_changed(on_height_change)
btn_fwd.on_clicked(on_fwd)
btn_back.on_clicked(on_back)
btn_left.on_clicked(on_left)
btn_right.on_clicked(on_right)
btn_stance.on_clicked(on_stance)
btn_walk.on_clicked(on_walk)

draw(body['x'], body['y'], body['z'])
plt.show()
