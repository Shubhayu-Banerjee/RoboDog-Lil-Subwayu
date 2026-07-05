import time
import math
import numpy as np
import pybullet as pyb
import pybullet_data


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
        self.l1 = 0.10
        self.l2 = 0.12
        self.hip_offset_y = 0.055

        # Interactive UI Sliders for Body Pose & Height
        self.height_slider = pyb.addUserDebugParameter("Body Height", -0.25, -0.05, -0.17)
        self.roll_slider = pyb.addUserDebugParameter("Body Roll", -0.4, 0.4, 0.0)
        self.pitch_slider = pyb.addUserDebugParameter("Body Pitch", -0.4, 0.4, 0.0)
        self.yaw_slider = pyb.addUserDebugParameter("Static Body Yaw", -0.5, 0.5, 0.0)

        self.cpg_time = 0.0
        self.gait_frequency = 2.0

        # Interactive Vector Commands: [Forward/Backward, Left/Right Strafe, Dynamic Yaw]
        self.commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # --- JUMP STATE MACHINE VARIABLES ---
        self.jump_state = "GROUNDED"  # States: GROUNDED, WIND_DOWN, STABILIZE, LAUNCH, FLIGHT
        self.jump_timer = 0.0
        self.jump_pitch = 0.0
        self.jump_z_front = 0.0  # Front legs extension target (Legs 2 & 3)
        self.jump_z_rear = 0.0  # Rear legs extension target (Legs 0 & 1)
        self.motor_force = 5.88  # Strictly locked to original physics limit

        self._setup_world()

    def _create_wedge_mesh(self, length, width, height):
        """Generates raw triangular mesh vertices with a tiny lip to eliminate Z-fighting"""
        lip = 0.002
        vertices = [
            [0, -width / 2, lip],  # 0: Front Left
            [0, width / 2, lip],  # 1: Front Right
            [length, -width / 2, lip],  # 2: Back Left Base
            [length, width / 2, lip],  # 3: Back Right Base
            [length, -width / 2, height],  # 4: Back Left Peak
            [length, width / 2, height]  # 5: Back Right Peak
        ]
        indices = [
            0, 2, 1, 1, 2, 3,  # Bottom face
            0, 1, 4, 1, 5, 4,  # Sloped ramp face
            0, 4, 2,  # Left triangle side
            1, 3, 5,  # Right triangle side
            2, 4, 3, 3, 4, 5  # Back vertical wall
        ]
        return vertices, indices

    def _setup_world(self):
        pyb.resetSimulation(physicsClientId=self.client)
        pyb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pyb.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = pyb.loadURDF("plane.urdf", physicsClientId=self.client)

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
            verts, idxs = self._create_wedge_mesh(l, w, h)

            collision_id = pyb.createCollisionShape(pyb.GEOM_MESH, vertices=verts, indices=idxs,
                                                    physicsClientId=self.client)
            visual_id = pyb.createVisualShape(pyb.GEOM_MESH, vertices=verts, indices=idxs,
                                              rgbaColor=[0.4, 0.4, 0.45, 1.0], physicsClientId=self.client)
            ramp_id = pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id,
                                          baseVisualShapeIndex=visual_id, basePosition=config["pos"],
                                          physicsClientId=self.client)
            pyb.changeDynamics(ramp_id, -1, lateralFriction=2.5, physicsClientId=self.client)

        # Spawn robot
        start_pos = [0, 0, 0.19]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyb.loadURDF("robot.urdf", start_pos, start_ori, useFixedBase=False,
                                     physicsClientId=self.client)

        for joint in range(12):
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.8, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.8, physicsClientId=self.client)

    def analytical_ik(self, leg_index, target_xyz):
        x, y, z = target_xyz
        is_left_side = 1.0 if leg_index in [0, 2] else -1.0
        y_hip = y - (is_left_side * self.hip_offset_y)

        d = math.sqrt(y_hip ** 2 + z ** 2)
        if d == 0: return 0.0, 0.0, 0.0
        hip_roll = math.atan2(y_hip, -z)

        z_proj = -math.sqrt(d ** 2)
        r_sq = x ** 2 + z_proj ** 2
        r = math.sqrt(r_sq)

        cos_calf = (self.l1 ** 2 + self.l2 ** 2 - r_sq) / (2.0 * self.l1 * self.l2)
        cos_calf = np.clip(cos_calf, -1.0, 1.0)
        calf_knee = math.pi - math.acos(cos_calf)

        alpha = math.atan2(x, -z_proj)
        cos_beta = (self.l1 ** 2 + r_sq - self.l2 ** 2) / (2.0 * self.l1 * r)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
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
            t_ratio = min(self.jump_timer / 0.20, 1.0)
            # Symmetrical flat vertical compression
            crouch = 0.07 * t_ratio
            self.jump_z_front = crouch
            self.jump_z_rear = crouch
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.20:
                self.jump_state = "STABILIZE"
                self.jump_timer = 0.0

        elif self.jump_state == "STABILIZE":
            self.jump_timer += dt
            # Hold steady for 500ms to eliminate crouch inertia
            self.jump_z_front = 0.07
            self.jump_z_rear = 0.07
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.50:
                self.jump_state = "LAUNCH"
                self.jump_timer = 0.0

        elif self.jump_state == "LAUNCH":
            self.jump_timer += dt
            # --- ASYMMETRICAL EXTENSION CORRECTED FOR -X AXIS ---
            # Legs 0 & 1 are at the structural REAR (-X) -> get massive -0.09m push to punch COG up.
            # Legs 2 & 3 are at the structural FRONT (+X) -> get small -0.02m push to stop nose wheelies.
            self.jump_z_rear = -0.09
            self.jump_z_front = -0.02
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.08:
                self.jump_state = "FLIGHT"
                self.jump_timer = 0.0

        elif self.jump_state == "FLIGHT":
            self.jump_timer += dt
            # Uniform high-clearance tuck in mid-air
            self.jump_z_front = 0.02
            self.jump_z_rear = 0.02
            self.jump_pitch = 0.0

            if self.jump_timer >= 0.45:
                self.jump_state = "GROUNDED"
                self.jump_timer = 0.0

    def update_gait(self):
        self.cpg_time += 1.0 / 60.0
        omega = 2.0 * math.pi * self.gait_frequency

        self._process_jump_state()

        body_h = pyb.readUserDebugParameter(self.height_slider)
        b_roll = pyb.readUserDebugParameter(self.roll_slider)
        b_pitch = pyb.readUserDebugParameter(self.pitch_slider) + self.jump_pitch
        b_yaw = pyb.readUserDebugParameter(self.yaw_slider)

        if self.jump_state == "GROUNDED":
            v_x, v_y, w_yaw = self.commands[0], self.commands[1], self.commands[2]
        else:
            v_x, v_y, w_yaw = 0.0, 0.0, 0.0

        stride_x = v_x * 0.12
        stride_y = v_y * 0.10
        step_height = 0.05 if (abs(v_x) > 0.01 or abs(v_y) > 0.01 or abs(w_yaw) > 0.01) else 0.0

        phases = [omega * self.cpg_time, omega * self.cpg_time + math.pi,
                  omega * self.cpg_time + math.pi, omega * self.cpg_time]

        base_x = [0.12, 0.12, -0.12, -0.12]
        base_y = [self.hip_offset_y, -self.hip_offset_y, self.hip_offset_y, -self.hip_offset_y]

        for i in range(4):
            p = phases[i]
            x_step = stride_x * math.cos(p)
            y_step = stride_y * math.cos(p)
            z_step = step_height * max(0.0, math.sin(p))

            dynamic_yaw_rx = base_y[i]
            dynamic_yaw_ry = -base_x[i]
            x_step += w_yaw * dynamic_yaw_rx * 0.4 * math.cos(p)
            y_step += w_yaw * dynamic_yaw_ry * 0.4 * math.cos(p)

            pitch_z_extension = base_x[i] * math.tan(b_pitch)
            roll_z_extension = -base_y[i] * math.tan(b_roll)

            static_yaw_x = -(base_x[i] * (math.cos(b_yaw) - 1.0) - base_y[i] * math.sin(b_yaw))
            static_yaw_y = -(base_x[i] * math.sin(b_yaw) + base_y[i] * (math.cos(b_yaw) - 1.0))

            ik_x = x_step + static_yaw_x
            ik_y = y_step + static_yaw_y

            # --- MAP THE GEOMETRIC COMPENSATIONS BASED ON REVERSE FRAME ---
            if self.jump_state != "GROUNDED":
                # Legs 0 & 1 are REAR (-X) -> receive jump_z_rear
                # Legs 2 & 3 are FRONT (+X) -> receive jump_z_front
                current_jump_z = self.jump_z_rear if i in [0, 1] else self.jump_z_front
                ik_z = z_step + pitch_z_extension + roll_z_extension + (-0.17 + current_jump_z)
            else:
                ik_z = body_h + z_step + pitch_z_extension + roll_z_extension

            hip, thigh, calf = self.analytical_ik(i, (ik_x, ik_y, ik_z))

            pyb.setJointMotorControl2(self.robot_id, i * 3, pyb.POSITION_CONTROL, targetPosition=hip,
                                      force=self.motor_force)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 1, pyb.POSITION_CONTROL, targetPosition=thigh,
                                      force=self.motor_force)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 2, pyb.POSITION_CONTROL, targetPosition=calf,
                                      force=self.motor_force)

    def run_loop(self):
        print("\n=== PURE MATHEMATICAL JOYSTICK CONTROLLER ACTIVE ===")
        print("Locomotion Controls:")
        print("  [K] -> Forward (-X)   [N] -> Backward (+X)")
        print("  [B] -> Strafe Left    [M] -> Strafe Right")
        print("  [J] -> Rotate CCW     [L] -> Rotate CW")
        print("  [SPACEBAR] -> Level Stance Stabilized Jump Sequence")
        print("\nCamera View Adjustments:")
        print("  [UP ARROW] / [DOWN ARROW]   -> Adjust Camera Pitch")
        print("  [LEFT ARROW] / [RIGHT ARROW] -> Adjust Camera Yaw")
        print("====================================================\n")

        while True:
            pyb.stepSimulation(physicsClientId=self.client)
            keys = pyb.getKeyboardEvents()

            # Spacebar jump activation
            if 32 in keys and self.jump_state == "GROUNDED":
                self.jump_state = "WIND_DOWN"
                self.jump_timer = 0.0

            # Locomotion Key Processing
            if 107 in keys:
                self.commands[0] = max(self.commands[0] - 0.02, -0.5)  # K
            elif 110 in keys:
                self.commands[0] = min(self.commands[0] + 0.02, 0.40)  # N

            if 98 in keys:
                self.commands[1] = min(self.commands[1] + 0.02, 0.25)  # B
            elif 109 in keys:
                self.commands[1] = max(self.commands[1] - 0.02, -0.25)  # M

            if 106 in keys:
                self.commands[2] = min(self.commands[2] + 0.04, 0.50)  # J
            elif 108 in keys:
                self.commands[2] = max(self.commands[2] - 0.04, -0.50)  # L

            if 120 in keys: self.commands = np.zeros(3, dtype=np.float32)  # X

            if not keys:
                self.commands[0] *= 0.95
                self.commands[1] *= 0.95
                self.commands[2] *= 0.90

            # Dynamic Arrow Camera processing
            if pyb.B3G_LEFT_ARROW in keys: self.cam_yaw -= 1.5
            if pyb.B3G_RIGHT_ARROW in keys: self.cam_yaw += 1.5
            if pyb.B3G_UP_ARROW in keys: self.cam_pitch = min(self.cam_pitch + 1.0, -5.0)
            if pyb.B3G_DOWN_ARROW in keys: self.cam_pitch = max(self.cam_pitch - 1.0, -75.0)

            self.update_gait()

            base_pos, _ = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
            pyb.resetDebugVisualizerCamera(
                cameraDistance=self.cam_distance, cameraYaw=self.cam_yaw, cameraPitch=self.cam_pitch,
                cameraTargetPosition=base_pos, physicsClientId=self.client
            )

            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    dog_pilot = PureMathQuadruped()
    dog_pilot.run_loop()
