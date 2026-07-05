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

        # Configure camera view focused on the robot center
        pyb.resetDebugVisualizerCamera(
            cameraDistance=0.9, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.18]
        )

        # Robot Geometry
        self.l1 = 0.10  # Thigh link length (meters)
        self.l2 = 0.12  # Shin link length (meters)
        self.hip_offset_y = 0.055  # Lateral width separation of hip joint

        # Interactive UI Sliders for Body Pose & Height
        self.height_slider = pyb.addUserDebugParameter("Body Height", -0.25, -0.05, -0.17)
        self.roll_slider = pyb.addUserDebugParameter("Body Roll", -0.4, 0.4, 0.0)
        self.pitch_slider = pyb.addUserDebugParameter("Body Pitch", -0.4, 0.4, 0.0)
        self.yaw_slider = pyb.addUserDebugParameter("Static Body Yaw", -0.5, 0.5, 0.0)

        self.cpg_time = 0.0
        self.gait_frequency = 2.0  # 2.0 Hz stepping rate

        # Interactive Vector Commands: [Forward/Backward, Left/Right Strafe, Dynamic Yaw]
        self.commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._setup_world()

    def _setup_world(self):
        pyb.resetSimulation(physicsClientId=self.client)
        pyb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pyb.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = pyb.loadURDF("plane.urdf", physicsClientId=self.client)

        # Spawn robot body comfortably flat above ground grid
        start_pos = [0, 0, 0.19]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyb.loadURDF("robot.urdf", start_pos, start_ori, useFixedBase=False,
                                     physicsClientId=self.client)

        # Set friction to maximize foot traction profiles
        for joint in range(12):
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.8, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.8, physicsClientId=self.client)

    def analytical_ik(self, leg_index, target_xyz):
        """Translates 3D foot position coordinates into precise joint angles using trigonometry"""
        x, y, z = target_xyz

        # Account for mechanical side inversion mirroring (Left vs Right legs)
        is_left_side = 1.0 if leg_index in [0, 2] else -1.0
        y_hip = y - (is_left_side * self.hip_offset_y)

        # 1. Calculate Hip Roll Angle (Abduction/Adduction)
        d = math.sqrt(y_hip ** 2 + z ** 2)
        if d == 0:
            return 0.0, 0.0, 0.0
        hip_roll = math.atan2(y_hip, -z)

        # Project coordinates onto the sagittal 2D plane of the thigh/calf link path
        z_proj = -math.sqrt(d ** 2)

        # 2. Calculate Calf Knee Angle using the Law of Cosines
        r_sq = x ** 2 + z_proj ** 2
        r = math.sqrt(r_sq)

        cos_calf = (self.l1 ** 2 + self.l2 ** 2 - r_sq) / (2.0 * self.l1 * self.l2)
        cos_calf = np.clip(cos_calf, -1.0, 1.0)
        calf_knee = math.pi - math.acos(cos_calf)

        # 3. Calculate Thigh Pitch Angle
        alpha = math.atan2(x, -z_proj)
        cos_beta = (self.l1 ** 2 + r_sq - self.l2 ** 2) / (2.0 * self.l1 * r)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = math.acos(cos_beta)
        thigh_pitch = alpha - beta

        return hip_roll, thigh_pitch, calf_knee

    def update_gait(self):
        """CPG Engine handles walking paths, with Z-axis extension overrides for body pose"""
        self.cpg_time += 1.0 / 60.0
        omega = 2.0 * math.pi * self.gait_frequency

        # Read Slider States for Body Pose
        body_h = pyb.readUserDebugParameter(self.height_slider)
        b_roll = pyb.readUserDebugParameter(self.roll_slider)
        b_pitch = pyb.readUserDebugParameter(self.pitch_slider)
        b_yaw = pyb.readUserDebugParameter(self.yaw_slider)

        v_x = self.commands[0]
        v_y = self.commands[1]
        w_yaw = self.commands[2]

        stride_x = v_x * 0.12
        stride_y = v_y * 0.10
        step_height = 0.05 if (abs(v_x) > 0.01 or abs(v_y) > 0.01 or abs(w_yaw) > 0.01) else 0.0

        # Leg phase standard: 0=FL, 1=FR, 2=BL, 3=BR
        phases = [
            omega * self.cpg_time,
            omega * self.cpg_time + math.pi,
            omega * self.cpg_time + math.pi,
            omega * self.cpg_time
        ]

        base_x = [0.12, 0.12, -0.12, -0.12]
        base_y = [self.hip_offset_y, -self.hip_offset_y, self.hip_offset_y, -self.hip_offset_y]

        for i in range(4):
            p = phases[i]

            # Dynamic walking trajectories
            x_step = stride_x * math.cos(p)
            y_step = stride_y * math.cos(p)
            z_step = step_height * max(0.0, math.sin(p))

            # Dynamic turning arcs (for moving)
            dynamic_yaw_rx = base_y[i]
            dynamic_yaw_ry = -base_x[i]
            x_step += w_yaw * dynamic_yaw_rx * 0.4 * math.cos(p)
            y_step += w_yaw * dynamic_yaw_ry * 0.4 * math.cos(p)

            # --- THE EXTENSION FIX ---
            # 1. Pitch & Roll: Modify ONLY the Z-axis extension so the legs don't push horizontally
            pitch_z_extension = base_x[i] * math.tan(b_pitch)
            roll_z_extension = -base_y[i] * math.tan(b_roll)

            # 2. Static Yaw: Since legs can't extend sideways, we must calculate the exact counter-shift
            # to keep the feet planted perfectly in place while the hips rotate around the COG.
            static_yaw_x = -(base_x[i] * (math.cos(b_yaw) - 1.0) - base_y[i] * math.sin(b_yaw))
            static_yaw_y = -(base_x[i] * math.sin(b_yaw) + base_y[i] * (math.cos(b_yaw) - 1.0))

            # Combine all coordinates into the local hip frame
            ik_x = x_step + static_yaw_x
            ik_y = y_step + static_yaw_y
            ik_z = body_h + z_step + pitch_z_extension + roll_z_extension

            # Pass calculated lengths through Inverse Kinematics
            hip, thigh, calf = self.analytical_ik(i, (ik_x, ik_y, ik_z))

            pyb.setJointMotorControl2(self.robot_id, i * 3, pyb.POSITION_CONTROL, targetPosition=hip, force=5.88)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 1, pyb.POSITION_CONTROL, targetPosition=thigh, force=5.88)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 2, pyb.POSITION_CONTROL, targetPosition=calf, force=5.88)

    def run_loop(self):
        print("\n=== PURE MATHEMATICAL JOYSTICK CONTROLLER ACTIVE ===")
        print("Click on the PyBullet simulation window to pilot:")
        print("  [K] -> Shuffle Forward          [N] -> Shuffle Backward")
        print("  [B] -> Strafe Left              [M] -> Strafe Right")
        print("  [J] -> Rotate Counter-Clockwise [L] -> Rotate Clockwise")
        print("  [X] -> Full Stationary Brake / Pause Stance")
        print("  (Use UI Sliders to adjust Body Height, Pitch, Roll, and Yaw)")
        print("====================================================\n")

        while True:
            pyb.stepSimulation(physicsClientId=self.client)
            keys = pyb.getKeyboardEvents()

            # Forward / Backward mapping (K/N)
            if 107 in keys:  # 'K'
                self.commands[0] = max(self.commands[0] - 0.02, -0.5)
            elif 110 in keys:  # 'N'
                self.commands[0] = min(self.commands[0] + 0.02, 0.40)

            # Lateral Strafe mapping (B/M)
            if 98 in keys:  # 'B'
                self.commands[1] = min(self.commands[1] + 0.02, 0.25)
            elif 109 in keys:  # 'M'
                self.commands[1] = max(self.commands[1] - 0.02, -0.25)

            # Yaw Rotational turn mapping (J/L)
            if 106 in keys:  # 'J'
                self.commands[2] = min(self.commands[2] + 0.04, 0.50)
            elif 108 in keys:  # 'L'
                self.commands[2] = max(self.commands[2] - 0.04, -0.50)

            # Emergency Stop / Full Reset
            if 120 in keys:  # 'X'
                self.commands = np.zeros(3, dtype=np.float32)

            # Decelerate commands smoothly toward zero
            if not keys:
                self.commands[0] *= 0.95
                self.commands[1] *= 0.95
                self.commands[2] *= 0.90

            self.update_gait()

            base_pos, _ = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.9, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=base_pos,
                physicsClientId=self.client
            )

            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    dog_pilot = PureMathQuadruped()
    dog_pilot.run_loop()
