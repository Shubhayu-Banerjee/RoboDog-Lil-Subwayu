import time
import math
import numpy as np
import pybullet as pyb
import pybullet_data
import multiprocessing as mp
import threading
from collections import deque



# ==================== ACTUATOR TELEMETRY ====================
# ============================================================
# ADAPTIVE LEG GEOMETRY
# Change these two values for different physical leg lengths.
# The gait/body height are derived from them automatically.
# ============================================================
LEG_THIGH_LENGTH = 0.15   # metres
LEG_CALF_LENGTH  = 0.20   # metres

# Desired neutral knee configuration.  This is a joint-space target,
# so changing leg length does not force the robot into the old stance.
NEUTRAL_KNEE_ANGLE = 1.00  # radians

# Fraction of each gait cycle spent in stance.  The remainder is swing.
STANCE_FRACTION = 0.60

# Keep the original nominal horizontal reach unless the robot geometry
# itself is changed.
NOMINAL_FOOT_X = 0.12

ACTUATOR_MAX_TORQUE = 2.45       # N*m (25 kg*cm servo)
ACTUATOR_MAX_CURRENT = 3.70      # A, maximum/stall current per servo
SERVO_SUPPLY_VOLTAGE = 6.0       # V, used only for estimated electrical power


# Telemetry architecture:
#   PyBullet/main thread -> tiny raw-torque queue
#   calculation thread   -> current/power calculation + smoothing
#   matplotlib process    -> graph rendering
TELEMETRY_CALC_QUEUE_SIZE = 4
TELEMETRY_PLOT_QUEUE_SIZE = 2000
TELEMETRY_PLOT_HZ = 20.0
TELEMETRY_HISTORY_SECONDS = 30.0
TELEMETRY_SMOOTH_ALPHA = 0.10


# ==================== GAIT TRANSITION SETTINGS ====================

# How quickly commanded velocity is allowed to change.
# Smaller = faster response, larger = smoother.
COMMAND_RAMP_TIME = 0.15       # seconds

# How quickly the robot stops its gait after command reaches zero.
STOP_RAMP_TIME = 0.20          # seconds

# How quickly swing height comes on/off.
STEP_HEIGHT_RISE_TIME = 0.12   # seconds
STEP_HEIGHT_FALL_TIME = 0.16   # seconds

# Full swing height.
MAX_STEP_HEIGHT = 0.05         # meters

# Minimum active command before considering the robot stopped.
COMMAND_DEADZONE = 0.008

# Phase considered "safe" for final standing reset.
# We don't actually need to land exactly here because the gait amplitude
# is already being collapsed toward zero, but this provides a clean reset.
PHASE_RESET_WINDOW = 0.10      # radians

TWO_PI = 2.0 * math.pi


def actuator_calculation_worker(raw_queue, plot_queue, stop_event):
    """Calculate total estimated current/power off the PyBullet thread."""
    smooth_current = None

    while not stop_event.is_set():
        try:
            timestamp, torques, velocity = raw_queue.get(timeout=0.2)
        except Exception:
            continue

        torques = np.asarray(torques, dtype=np.float64)

        # Torque -> proportional estimated current.
        # This is an approximation, not a physical servo current model.
        utilization = np.abs(torques) / ACTUATOR_MAX_TORQUE
        estimated_currents = utilization * ACTUATOR_MAX_CURRENT
        total_current = float(np.sum(estimated_currents))

        # Smooth only the aggregate value so the graph remains readable.
        if smooth_current is None:
            smooth_current = total_current
        else:
            smooth_current += TELEMETRY_SMOOTH_ALPHA * (
                total_current - smooth_current
            )

        item = (
            float(timestamp),
            float(smooth_current),
            float(velocity)
        )

        try:
            plot_queue.put_nowait(item)
        except Exception:
            # Plot process is behind; discard this point rather than blocking
            # the simulation/calculation pipeline.
            pass


def actuator_plot_worker(plot_queue, stop_event):
    """Dedicated matplotlib dashboard. It never touches PyBullet."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plot_period = 1.0 / TELEMETRY_PLOT_HZ

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="black")
    fig.canvas.manager.set_window_title("Robo Dog - Power & Velocity")

    ax.set_facecolor("black")
    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_yticks([])
    ax.set_xlabel("Estimated Total Servo Current (A)",
                   color="white", fontsize=12)
    ax.tick_params(axis="x", colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")

    # Threshold regions.
    ax.axvspan(0, 30, color="green", alpha=0.12)
    ax.axvspan(30, 45, color="orange", alpha=0.12)
    ax.axvspan(45, 60, color="red", alpha=0.12)
    ax.axvline(30, color="orange", linewidth=1.5, alpha=0.8)
    ax.axvline(45, color="red", linewidth=1.5, alpha=0.8)

    # Main current meter.
    meter = ax.barh(
        [0.5], [0.0], height=0.55,
        color="green", edgecolor="white", linewidth=1.2
    )

    current_text = ax.text(
        30.0, 0.5, "0.00 A",
        ha="center", va="center",
        color="white", fontsize=30, fontweight="bold"
    )

    velocity_text = ax.text(
        30.0, 0.18, "Velocity: 0.00 m/s",
        ha="center", va="center",
        color="white", fontsize=17
    )

    velocity_avg_text = ax.text(
        30.0, 0.08, "5s Avg: 0.00 m/s",
        ha="center", va="center",
        color="#aaaaaa", fontsize=14
    )

    # Rolling velocity history: (timestamp, velocity)
    velocity_history = deque()

    status_text = ax.text(
        30.0, 0.86, "SAFE",
        ha="center", va="center",
        color="green", fontsize=15, fontweight="bold"
    )

    ax.set_title(
        "ROBO DOG — ACTUATOR POWER",
        color="white", fontsize=16, fontweight="bold", pad=14
    )

    def update(_frame):
        latest = None

        # Drain the queue and keep only the newest telemetry sample.
        while True:
            try:
                latest = plot_queue.get_nowait()
            except Exception:
                break

        if latest is None:
            return (
                meter[0],
                current_text,
                velocity_text,
                velocity_avg_text,
                status_text
            )

        # New format:
        # (timestamp, current, velocity)
        timestamp, current, velocity = latest

        current = max(0.0, min(60.0, float(current)))
        velocity = max(0.0, float(velocity))

        # Maintain a rolling 5-second velocity window.
        velocity_history.append((float(timestamp), velocity))
        cutoff = float(timestamp) - 5.0

        while velocity_history and velocity_history[0][0] < cutoff:
            velocity_history.popleft()

        if velocity_history:
            velocity_average = sum(
                sample_velocity
                for _, sample_velocity in velocity_history
            ) / len(velocity_history)
        else:
            velocity_average = 0.0

        if current >= 45.0:
            state_color = "red"
            state_text = "HIGH CURRENT"
        elif current >= 30.0:
            state_color = "orange"
            state_text = "WARNING"
        else:
            state_color = "green"
            state_text = "SAFE"

        meter[0].set_width(current)
        meter[0].set_color(state_color)

        current_text.set_text(f"{current:.2f} A")
        velocity_text.set_text(f"Velocity: {velocity:.2f} m/s")
        velocity_avg_text.set_text(
            f"5s Avg: {velocity_average:.2f} m/s"
        )
        status_text.set_text(state_text)
        status_text.set_color(state_color)

        fig.canvas.draw_idle()
        return (
            meter[0],
            current_text,
            velocity_text,
            velocity_avg_text,
            status_text
        )

    def on_close(_event):
        stop_event.set()

    fig.canvas.mpl_connect("close_event", on_close)

    animation = FuncAnimation(
        fig,
        update,
        interval=1000.0 / TELEMETRY_PLOT_HZ,
        blit=False,
        cache_frame_data=False
    )

    while not stop_event.is_set():
        plt.pause(plot_period)

    plt.close(fig)


class PureMathQuadruped:
    """Pure Mathematical Controller using a 3D CPG Oscillator and Analytical Inverse Kinematics"""

    def __init__(self):
        # Initialize PyBullet GUI
        self.client = pyb.connect(pyb.GUI)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Camera state parameters for interactive rotations
        self.cam_distance = 0.9
        self.cam_yaw = 50.0
        self.cam_pitch = -25.0

        pyb.resetDebugVisualizerCamera(
            cameraDistance=self.cam_distance,
            cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch,
            cameraTargetPosition=[0, 0, 0.18]
        )

        # Robot Geometry
        self.l1 = LEG_THIGH_LENGTH
        self.l2 = LEG_CALF_LENGTH
        self.hip_offset_y = 0.055

        # Interactive UI Sliders for Body Pose & Height
        self.height_slider = pyb.addUserDebugParameter(
            "Body Height", -0.30, -0.05, -0.17
        )

        self.roll_slider = pyb.addUserDebugParameter(
            "Body Roll", -0.4, 0.4, 0.0
        )

        self.pitch_slider = pyb.addUserDebugParameter(
            "Body Pitch", -0.4, 0.4, 0.0
        )

        self.yaw_slider = pyb.addUserDebugParameter(
            "Static Body Yaw", -0.5, 0.5, 0.0
        )

        self.x_stride_slider = pyb.addUserDebugParameter(
            "X Stride (x)", 1.0, 2.0, 1.0
        )

        self.y_stride_slider = pyb.addUserDebugParameter(
            "Y Stride (x)", 1.0, 2.0, 1.0
        )

        # ==================== CPG ====================
        self.cpg_time = 0.0
        self.gait_frequency = 2.0

        # ==================== CPG ====================
        self.cpg_time = 0.0
        self.gait_frequency = 2.0

        # Actual filtered command used by the gait.
        # self.commands remains the user's requested command.
        self.active_command = np.zeros(3, dtype=np.float32)

        # Current smooth swing height.
        self.current_step_height = 0.0

        # Gait state:
        # RUNNING  = normal CPG walking
        # STOPPING = smoothly collapsing gait to standing
        # STANDING = fully stopped / neutral
        self.gait_state = "STANDING"

        # Used to track stopping progression.
        self.stop_timer = 0.0

        # Interactive Vector Commands:
        # [Forward/Backward, Left/Right Strafe, Dynamic Yaw]
        self.commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # --- JUMP STATE MACHINE VARIABLES ---
        self.jump_state = "GROUNDED"
        self.jump_timer = 0.0
        self.jump_pitch = 0.0
        self.jump_z_front = 0.0
        self.jump_z_rear = 0.0
        self.motor_force = ACTUATOR_MAX_TORQUE

        self._setup_world()

        # ==================== NON-BLOCKING TELEMETRY ====================
        self.telemetry_raw_queue = mp.Queue(
            maxsize=TELEMETRY_CALC_QUEUE_SIZE
        )

        self.telemetry_plot_queue = mp.Queue(
            maxsize=TELEMETRY_PLOT_QUEUE_SIZE
        )

        self.telemetry_stop_event = mp.Event()

        self.telemetry_calc_thread = threading.Thread(
            target=actuator_calculation_worker,
            args=(
                self.telemetry_raw_queue,
                self.telemetry_plot_queue,
                self.telemetry_stop_event
            ),
            daemon=True,
            name="ActuatorCalculation"
        )

        self.telemetry_plot_process = None
        self.telemetry_start_time = time.perf_counter()

        self.telemetry_calc_thread.start()

    def _create_wedge_mesh(self, length, width, height):
        """Generates raw triangular mesh vertices with a tiny lip to eliminate Z-fighting"""
        lip = 0.002

        vertices = [
            [0, -width / 2, lip],
            [0, width / 2, lip],
            [length, -width / 2, lip],
            [length, width / 2, lip],
            [length, -width / 2, height],
            [length, width / 2, height]
        ]

        indices = [
            0, 2, 1,
            1, 2, 3,

            0, 1, 4,
            1, 5, 4,

            0, 4, 2,
            1, 3, 5,

            2, 4, 3,
            3, 4, 5
        ]

        return vertices, indices

    def _setup_world(self):
        pyb.resetSimulation(physicsClientId=self.client)
        pyb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pyb.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = pyb.loadURDF(
            "plane.urdf",
            physicsClientId=self.client
        )

        # Progressive ramp network layout
        ramps_config = [
            {"size": [0.8, 0.6, 0.08], "pos": [0.4, 0.0, 0.0]},
            {"size": [1.0, 0.6, 0.15], "pos": [0.4, 2.1, 0.0]},
            {"size": [1.5, 0.7, 0.30], "pos": [1.6, -4.05, 0.0]},
            {"size": [0.8, 0.7, 0.30], "pos": [0.3, 4.0, 0.0]},
            {"size": [2.0, 0.8, 0.60], "pos": [1.3, -2.0, 0.0]}
        ]

        for config in ramps_config:
            l, w, h = config["size"]

            verts, idxs = self._create_wedge_mesh(
                l,
                w,
                h
            )

            collision_id = pyb.createCollisionShape(
                pyb.GEOM_MESH,
                vertices=verts,
                indices=idxs,
                physicsClientId=self.client
            )

            visual_id = pyb.createVisualShape(
                pyb.GEOM_MESH,
                vertices=verts,
                indices=idxs,
                rgbaColor=[0.4, 0.4, 0.45, 1.0],
                physicsClientId=self.client
            )

            ramp_id = pyb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=config["pos"],
                physicsClientId=self.client
            )

            pyb.changeDynamics(
                ramp_id,
                -1,
                lateralFriction=2.5,
                physicsClientId=self.client
            )

        # Spawn robot
        start_pos = [0, 0,0.42]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = pyb.loadURDF(
            "robot.urdf",
            start_pos,
            start_ori,
            useFixedBase=False,
        )

        for joint in range(12):
            pyb.changeDynamics(
                self.robot_id,
                joint,
                lateralFriction=1.8,
                physicsClientId=self.client
            )

        pyb.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.8,
            physicsClientId=self.client
        )

    def analytical_ik(self, leg_index, target_xyz):
        x, y, z = target_xyz

        is_left_side = 1.0 if leg_index in [0, 2] else -1.0

        y_hip = y - (is_left_side * self.hip_offset_y)

        d = math.sqrt(y_hip ** 2 + z ** 2)

        if d == 0:
            return 0.0, 0.0, 0.0

        hip_roll = math.atan2(y_hip, -z)

        z_proj = -math.sqrt(d ** 2)

        r_sq = x ** 2 + z_proj ** 2
        r = math.sqrt(r_sq)

        cos_calf = (
            self.l1 ** 2 +
            self.l2 ** 2 -
            r_sq
        ) / (
            2.0 *
            self.l1 *
            self.l2
        )

        cos_calf = np.clip(
            cos_calf,
            -1.0,
            1.0
        )

        calf_knee = math.pi - math.acos(cos_calf)

        alpha = math.atan2(
            x,
            -z_proj
        )

        cos_beta = (
            self.l1 ** 2 +
            r_sq -
            self.l2 ** 2
        ) / (
            2.0 *
            self.l1 *
            r
        )

        cos_beta = np.clip(
            cos_beta,
            -1.0,
            1.0
        )

        beta = math.acos(cos_beta)

        thigh_pitch = alpha - beta

        return hip_roll, thigh_pitch, calf_knee

    def _process_jump_state(self):
        """Manages jumping dynamics with flat pitch profiles and inverted axis stroke mapping"""

        dt = 1.0 / 60.0

        if self.jump_state == "GROUNDED":
            self.jump_z_front = 0.0
            self.jump_z_rear = 0.0
            self.jump_pitch = 0.0

        elif self.jump_state == "WIND_DOWN":
            self.jump_timer += dt

            t_ratio = min(
                self.jump_timer / 0.20,
                1.0
            )

            crouch = 0.07 * t_ratio

            self.jump_z_front = crouch
            self.jump_z_rear = crouch
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.20:
                self.jump_state = "STABILIZE"
                self.jump_timer = 0.0

        elif self.jump_state == "STABILIZE":
            self.jump_timer += dt

            self.jump_z_front = 0.07
            self.jump_z_rear = 0.07
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.50:
                self.jump_state = "LAUNCH"
                self.jump_timer = 0.0

        elif self.jump_state == "LAUNCH":
            self.jump_timer += dt

            self.jump_z_rear = -0.09
            self.jump_z_front = -0.02
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.08:
                self.jump_state = "FLIGHT"
                self.jump_timer = 0.0

        elif self.jump_state == "FLIGHT":
            self.jump_timer += dt

            self.jump_z_front = 0.02
            self.jump_z_rear = 0.02
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.45:
                self.jump_state = "GROUNDED"
                self.jump_timer = 0.0

    # ============================================================
    # GAIT TRANSITION HELPERS
    # ============================================================

    @staticmethod
    def _smoothstep(x):
        """
        Cubic smoothstep.

        0 -> 0
        1 -> 1

        Zero slope at both ends, avoiding sudden acceleration.
        """
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _ramp_vector(self, current, target, dt, ramp_time):
        """
        Smoothly move current command toward target.

        This is deliberately independent of the CPG itself.
        """
        if ramp_time <= 0.0:
            return target.copy()

        alpha = min(
            dt / ramp_time,
            1.0
        )

        # Smooth exponential-ish response.
        # This avoids a hard discontinuity.
        return current + (target - current) * alpha

    def _update_command_ramp(self, dt):
        """
        Move the actual gait command toward the requested command.

        self.commands = what the user asks for.
        self.active_command = what the CPG actually receives.
        """

        target = self.commands.copy()

        # During jumping, don't allow locomotion commands to leak
        # into the jump controller.
        if self.jump_state != "GROUNDED":
            target[:] = 0.0

        self.active_command = self._ramp_vector(
            self.active_command,
            target,
            dt,
            COMMAND_RAMP_TIME
        )

        # Kill tiny numerical residuals.
        for i in range(3):
            if abs(self.active_command[i]) < COMMAND_DEADZONE:
                self.active_command[i] = 0.0

    def _update_step_height(self, dt):
        """
        Smoothly ramp swing height.

        This prevents:
            0 mm -> 50 mm

        from happening in a single frame.
        """

        command_magnitude = max(
            abs(float(self.active_command[0])),
            abs(float(self.active_command[1])),
            abs(float(self.active_command[2]))
        )

        if (
            command_magnitude > COMMAND_DEADZONE
            and self.jump_state == "GROUNDED"
        ):
            target_height = MAX_STEP_HEIGHT
        else:
            target_height = 0.0

        if target_height > self.current_step_height:
            ramp_time = STEP_HEIGHT_RISE_TIME
        else:
            ramp_time = STEP_HEIGHT_FALL_TIME

        if ramp_time <= 0.0:
            self.current_step_height = target_height
            return

        alpha = min(
            dt / ramp_time,
            1.0
        )

        self.current_step_height += (
            target_height - self.current_step_height
        ) * alpha

        if abs(self.current_step_height) < 0.0001:
            self.current_step_height = 0.0

    def _handle_gait_state(self, dt):
        """
        Handles RUNNING -> STOPPING -> STANDING.

        Important behavior:

        1. User releases movement.
        2. active_command smoothly approaches zero.
        3. CPG continues running during the transition.
        4. As the stride amplitude collapses, the feet naturally converge
           toward their neutral positions.
        5. Once almost stationary, the oscillator phase is reset cleanly.
        """

        active_mag = max(
            abs(float(self.active_command[0])),
            abs(float(self.active_command[1])),
            abs(float(self.active_command[2]))
        )

        requested_mag = max(
            abs(float(self.commands[0])),
            abs(float(self.commands[1])),
            abs(float(self.commands[2]))
        )

        # ------------------------------------------------------------
        # START WALKING
        # ------------------------------------------------------------
        if requested_mag > COMMAND_DEADZONE:

            self.gait_state = "RUNNING"
            self.stop_timer = 0.0

            return

        # ------------------------------------------------------------
        # ALREADY STANDING
        # ------------------------------------------------------------
        if (
            self.gait_state == "STANDING"
            and active_mag <= COMMAND_DEADZONE
        ):
            return

        # ------------------------------------------------------------
        # START STOPPING
        # ------------------------------------------------------------
        if self.gait_state == "RUNNING":
            self.gait_state = "STOPPING"
            self.stop_timer = 0.0

        # ------------------------------------------------------------
        # STOPPING
        # ------------------------------------------------------------
        if self.gait_state == "STOPPING":

            self.stop_timer += dt

            # Once command is basically zero and enough time has passed,
            # wait for the gait amplitude to become negligible.
            if (
                active_mag <= COMMAND_DEADZONE
                and
                self.stop_timer >= STOP_RAMP_TIME
            ):
                # We are now effectively standing.
                #
                # Reset phase to a known point instead of leaving the CPG
                # frozen at an arbitrary point in its cycle.
                self.cpg_time = 0.0

                self.active_command[:] = 0.0
                self.current_step_height = 0.0

                self.gait_state = "STANDING"
                self.stop_timer = 0.0

    def _foot_trajectory(self, phase):
        """Generate a length-independent stance/swing foot trajectory.

        The phase is 0..2π.  Stance keeps the foot on the ground while it
        travels backward; swing smoothly lifts, moves forward, and returns
        to exactly the ground height.  Because the vertical profile is
        defined in normalized phase rather than joint angle, changing leg
        length does not change ground-contact timing.
        """
        u = (phase % (2.0 * math.pi)) / (2.0 * math.pi)

        # Stance: foot is planted and moves backward.
        if u < STANCE_FRACTION:
            t = u / STANCE_FRACTION
            x = 1.0 - 2.0 * t
            z = 0.0
        else:
            # Swing: smooth 0 -> 1 -> 0 lift with zero vertical velocity
            # at both ground-contact events.
            t = (u - STANCE_FRACTION) / (1.0 - STANCE_FRACTION)
            s = 0.5 - 0.5 * math.cos(math.pi * t)
            x = -1.0 + 2.0 * s
            z = math.sin(math.pi * t) ** 2

        return x, z

    def update_gait(self):
        dt = 1.0 / 60.0

        # ============================================================
        # PHASE UPDATE
        # ============================================================

        # Convert the CPG to an explicitly bounded phase.
        #
        # Instead of allowing cpg_time to grow forever, keep it in
        # seconds but periodically wrap it.
        self.cpg_time += dt

        gait_period = 1.0 / self.gait_frequency

        if self.cpg_time >= gait_period:
            self.cpg_time -= gait_period

        phase = TWO_PI * self.gait_frequency * self.cpg_time

        # ============================================================
        # JUMP
        # ============================================================

        self._process_jump_state()

        # ============================================================
        # COMMAND RAMP
        # ============================================================

        self._update_command_ramp(dt)

        # ============================================================
        # GAIT STATE MACHINE
        # ============================================================

        self._handle_gait_state(dt)

        # ============================================================
        # STEP HEIGHT RAMP
        # ============================================================

        self._update_step_height(dt)

        # ============================================================
        # BODY POSE
        # ============================================================

        body_h = pyb.readUserDebugParameter(
            self.height_slider
        )

        b_roll = pyb.readUserDebugParameter(
            self.roll_slider
        )

        b_pitch = (
            pyb.readUserDebugParameter(self.pitch_slider)
            + self.jump_pitch
        )

        b_yaw = pyb.readUserDebugParameter(
            self.yaw_slider
        )

        # ============================================================
        # ACTIVE COMMAND
        # ============================================================

        if self.jump_state == "GROUNDED":
            v_x = float(self.active_command[0])
            v_y = float(self.active_command[1])
            w_yaw = float(self.active_command[2])
        else:
            v_x = 0.0
            v_y = 0.0
            w_yaw = 0.0

        # ============================================================
        # STRIDE
        # ============================================================

        x_stride_scale = float(
            pyb.readUserDebugParameter(self.x_stride_slider)
        )

        y_stride_scale = float(
            pyb.readUserDebugParameter(self.y_stride_slider)
        )

        x_stride_scale = float(np.clip(x_stride_scale, 1.0, 2.0))
        y_stride_scale = float(np.clip(y_stride_scale, 1.0, 2.0))

        stride_x = v_x * 0.12 * x_stride_scale
        stride_y = v_y * 0.10 * y_stride_scale

        # IMPORTANT:
        #
        # Step height is now smoothly ramped rather than:
        #
        #     moving = TRUE  ->  0.05
        #     moving = FALSE ->  0.00
        #
        step_height = self.current_step_height

        # ============================================================
        # DIAGONAL CPG
        # ============================================================

        phases = [
            phase,
            phase + math.pi,
            phase + math.pi,
            phase
        ]

        base_x = [
            0.12,
            0.12,
            -0.12,
            -0.12
        ]

        base_y = [
            self.hip_offset_y,
            -self.hip_offset_y,
            self.hip_offset_y,
            -self.hip_offset_y
        ]

        # ============================================================
        # LEG GENERATION
        # ============================================================

        for i in range(4):

            p = phases[i]

            # --------------------------------------------------------
            # HORIZONTAL CPG MOTION
            # --------------------------------------------------------

            cos_p = math.cos(p)
            sin_p = math.sin(p)

            x_step = stride_x * cos_p
            y_step = stride_y * cos_p

            # --------------------------------------------------------
            # SWING TRAJECTORY
            # --------------------------------------------------------
            #
            # Still using your original positive-half sine trajectory.
            # Phase 1-3 intentionally does NOT replace the CPG here.
            #
            # The difference is that step_height itself is now smooth.
            #

            z_step = step_height * max(
                0.0,
                sin_p
            )

            # --------------------------------------------------------
            # DYNAMIC YAW
            # --------------------------------------------------------

            dynamic_yaw_rx = base_y[i]
            dynamic_yaw_ry = -base_x[i]

            x_step += (
                w_yaw *
                dynamic_yaw_rx *
                0.4 *
                cos_p
            )

            y_step += (
                w_yaw *
                dynamic_yaw_ry *
                0.4 *
                cos_p
            )

            # --------------------------------------------------------
            # BODY PITCH / ROLL
            # --------------------------------------------------------

            pitch_z_extension = (
                base_x[i] *
                math.tan(b_pitch)
            )

            roll_z_extension = (
                -base_y[i] *
                math.tan(b_roll)
            )

            # --------------------------------------------------------
            # STATIC BODY YAW
            # --------------------------------------------------------

            cos_yaw = math.cos(b_yaw)
            sin_yaw = math.sin(b_yaw)

            static_yaw_x = -(
                base_x[i] *
                (cos_yaw - 1.0)
                -
                base_y[i] *
                sin_yaw
            )

            static_yaw_y = -(
                base_x[i] *
                sin_yaw
                +
                base_y[i] *
                (cos_yaw - 1.0)
            )

            # --------------------------------------------------------
            # FINAL XY TARGET
            # --------------------------------------------------------

            ik_x = x_step + static_yaw_x
            ik_y = y_step + static_yaw_y

            # --------------------------------------------------------
            # FINAL Z TARGET
            # --------------------------------------------------------

            if self.jump_state != "GROUNDED":

                # Legs 0 & 1 are REAR (-X)
                # Legs 2 & 3 are FRONT (+X)

                current_jump_z = (
                    self.jump_z_rear
                    if i in [0, 1]
                    else self.jump_z_front
                )

                ik_z = (
                    z_step
                    +
                    pitch_z_extension
                    +
                    roll_z_extension
                    +
                    (-0.17 + current_jump_z)
                )

            else:

                ik_z = (
                    body_h
                    +
                    z_step
                    +
                    pitch_z_extension
                    +
                    roll_z_extension
                )

            # --------------------------------------------------------
            # ANALYTICAL IK
            # --------------------------------------------------------

            hip, thigh, calf = self.analytical_ik(
                i,
                (ik_x, ik_y, ik_z)
            )

            # --------------------------------------------------------
            # SERVO TARGETS
            # --------------------------------------------------------

            pyb.setJointMotorControl2(
                self.robot_id,
                i * 3,
                pyb.POSITION_CONTROL,
                targetPosition=hip,
                force=self.motor_force
            )

            pyb.setJointMotorControl2(
                self.robot_id,
                i * 3 + 1,
                pyb.POSITION_CONTROL,
                targetPosition=thigh,
                force=self.motor_force
            )

            pyb.setJointMotorControl2(
                self.robot_id,
                i * 3 + 2,
                pyb.POSITION_CONTROL,
                targetPosition=calf,
                force=self.motor_force
            )

    def update_actuator_telemetry(self):
        """Fast telemetry producer: read torques and hand them off."""

        timestamp = (
            time.perf_counter()
            -
            self.telemetry_start_time
        )

        torques = []

        for joint_index in range(12):

            joint_state = pyb.getJointState(
                self.robot_id,
                joint_index,
                physicsClientId=self.client
            )

            torques.append(
                float(joint_state[3])
            )

        # Actual simulated body velocity, rather than commanded velocity.
        linear_velocity, _ = pyb.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.client
        )

        # Horizontal ground speed only.
        velocity = math.sqrt(
            float(linear_velocity[0]) ** 2 +
            float(linear_velocity[1]) ** 2
        )

        try:

            self.telemetry_raw_queue.put_nowait(
                (
                    timestamp,
                    torques,
                    velocity
                )
            )

        except Exception:

            # Never let telemetry block the physics loop.
            pass

    def run_loop(self):

        print(
            "\n=== PURE MATHEMATICAL JOYSTICK CONTROLLER ACTIVE ==="
        )

        print("Locomotion Controls:")
        print("  [K] -> Forward (-X)")
        print("  [N] -> Backward (+X)")
        print("  [B] -> Strafe Left")
        print("  [M] -> Strafe Right")
        print("  [J] -> Rotate CCW")
        print("  [L] -> Rotate CW")
        print("  [X] -> Stop")
        print("  [SPACEBAR] -> Level Stance Stabilized Jump Sequence")

        print("\nGait:")
        print(f"  CPG frequency       : {self.gait_frequency:.2f} Hz")
        print(f"  Command ramp        : {COMMAND_RAMP_TIME:.2f} s")
        print(f"  Stop ramp           : {STOP_RAMP_TIME:.2f} s")
        print(f"  Max step height     : {MAX_STEP_HEIGHT * 1000:.0f} mm")
        print(f"  Step height rise    : {STEP_HEIGHT_RISE_TIME:.2f} s")
        print(f"  Step height fall    : {STEP_HEIGHT_FALL_TIME:.2f} s")

        print("\nCamera View Adjustments:")
        print("  [UP ARROW] / [DOWN ARROW]   -> Adjust Camera Pitch")
        print("  [LEFT ARROW] / [RIGHT ARROW] -> Adjust Camera Yaw")

        print("====================================================")

        print("\nTelemetry:")
        print("  Separate calculation thread + separate matplotlib process")
        print(f"  Servo torque limit: {ACTUATOR_MAX_TORQUE:.2f} N*m")
        print(f"  Servo current limit: {ACTUATOR_MAX_CURRENT:.2f} A")
        print("  Telemetry quantity: total estimated actuator current")
        print("====================================================\n")

        # Start plotting in a completely separate process.
        self.telemetry_plot_process = mp.Process(
            target=actuator_plot_worker,
            args=(
                self.telemetry_plot_queue,
                self.telemetry_stop_event
            ),
            daemon=True,
            name="ActuatorPowerPlot"
        )

        self.telemetry_plot_process.start()

        try:

            while True:

                # ----------------------------------------------------
                # PHYSICS
                # ----------------------------------------------------

                pyb.stepSimulation(
                    physicsClientId=self.client
                )

                # ----------------------------------------------------
                # KEYBOARD
                # ----------------------------------------------------

                keys = pyb.getKeyboardEvents()

                # ----------------------------------------------------
                # SPACEBAR JUMP
                # ----------------------------------------------------

                if (
                    32 in keys
                    and
                    self.jump_state == "GROUNDED"
                ):
                    self.jump_state = "WIND_DOWN"
                    self.jump_timer = 0.0

                # ----------------------------------------------------
                # FORWARD / BACKWARD
                # ----------------------------------------------------

                if 107 in keys:
                    self.commands[0] = max(
                        self.commands[0] - 0.02,
                        -0.5
                    )

                elif 110 in keys:
                    self.commands[0] = min(
                        self.commands[0] + 0.02,
                        0.40
                    )

                # ----------------------------------------------------
                # STRAFE
                # ----------------------------------------------------

                if 98 in keys:
                    self.commands[1] = min(
                        self.commands[1] + 0.02,
                        0.25
                    )

                elif 109 in keys:
                    self.commands[1] = max(
                        self.commands[1] - 0.02,
                        -0.25
                    )

                # ----------------------------------------------------
                # YAW
                # ----------------------------------------------------

                if 106 in keys:
                    self.commands[2] = min(
                        self.commands[2] + 0.04,
                        0.50
                    )

                elif 108 in keys:
                    self.commands[2] = max(
                        self.commands[2] - 0.04,
                        -0.50
                    )

                # ----------------------------------------------------
                # HARD STOP REQUEST
                # ----------------------------------------------------
                #
                # This no longer directly freezes the CPG.
                #
                # It only sets the requested command to zero.
                # The gait controller then performs its smooth
                # RUNNING -> STOPPING -> STANDING transition.
                #

                if 120 in keys:
                    self.commands[:] = 0.0

                # ----------------------------------------------------
                # NATURAL COMMAND DECAY
                # ----------------------------------------------------
                #
                # Keep the original behavior, but the gait itself
                # still has an additional smooth ramp.
                #

                if not keys:
                    self.commands[0] *= 0.95
                    self.commands[1] *= 0.95
                    self.commands[2] *= 0.90

                # ----------------------------------------------------
                # CAMERA
                # ----------------------------------------------------

                if pyb.B3G_LEFT_ARROW in keys:
                    self.cam_yaw -= 1.5

                if pyb.B3G_RIGHT_ARROW in keys:
                    self.cam_yaw += 1.5

                if pyb.B3G_UP_ARROW in keys:
                    self.cam_pitch = min(
                        self.cam_pitch + 1.0,
                        -5.0
                    )

                if pyb.B3G_DOWN_ARROW in keys:
                    self.cam_pitch = max(
                        self.cam_pitch - 1.0,
                        -75.0
                    )

                # ----------------------------------------------------
                # GAIT
                # ----------------------------------------------------

                self.update_gait()

                # ----------------------------------------------------
                # TELEMETRY
                # ----------------------------------------------------

                self.update_actuator_telemetry()

                # ----------------------------------------------------
                # CAMERA TARGET
                # ----------------------------------------------------

                base_pos, _ = pyb.getBasePositionAndOrientation(
                    self.robot_id,
                    physicsClientId=self.client
                )

                pyb.resetDebugVisualizerCamera(
                    cameraDistance=self.cam_distance,
                    cameraYaw=self.cam_yaw,
                    cameraPitch=self.cam_pitch,
                    cameraTargetPosition=base_pos,
                    physicsClientId=self.client
                )

                # ----------------------------------------------------
                # 60 Hz CONTROLLER
                # ----------------------------------------------------

                time.sleep(1.0 / 60.0)

        except KeyboardInterrupt:
            pass

        finally:

            self.telemetry_stop_event.set()

            if (
                self.telemetry_plot_process is not None
                and
                self.telemetry_plot_process.is_alive()
            ):
                self.telemetry_plot_process.join(
                    timeout=2.0
                )

            if (
                self.telemetry_plot_process is not None
                and
                self.telemetry_plot_process.is_alive()
            ):
                self.telemetry_plot_process.terminate()

                self.telemetry_plot_process.join(
                    timeout=1.0
                )

            # Close multiprocessing queues cleanly on Windows
            try:

                self.telemetry_raw_queue.close()
                self.telemetry_plot_queue.close()

            except Exception:
                pass


if __name__ == "__main__":

    mp.freeze_support()

    dog_pilot = PureMathQuadruped()

    dog_pilot.run_loop()
