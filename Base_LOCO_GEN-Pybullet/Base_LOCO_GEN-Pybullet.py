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

        # Robot Geometry (As requested: 10cm Thigh, 12cm Shin)
        self.l1 = 0.10  # Thigh link length (meters)
        self.l2 = 0.12  # Shin link length (meters)
        self.hip_offset_y = 0.055  # Lateral width separation of hip joint

        # Stance configurations
        self.default_z = -0.17  # Default standing height relative to hip axis
        self.cpg_time = 0.0
        self.gait_frequency = 2.0  # 2.0 Hz stepping rate (cycles per second)

        # Interactive Vector Commands: [Forward/Backward, Left/Right Strafe, Yaw Turn Rate]
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
        """CPG Engine calculates clean alternating elliptical paths in all vector directions"""
        self.cpg_time += 1.0 / 60.0  # Increment master loop clock execution time
        omega = 2.0 * math.pi * self.gait_frequency

        # Extract targeted velocities from vector command registry
        v_x = self.commands[0]
        v_y = self.commands[1]
        w_yaw = self.commands[2]

        # Dynamically scale trajectory limits based on speed parameters
        stride_x = v_x * 0.12
        stride_y = v_y * 0.10
        step_height = 0.05 if (abs(v_x) > 0.01 or abs(v_y) > 0.01 or abs(w_yaw) > 0.01) else 0.0

        # Define 180-degree phase shift splits for diagonal trot pacing pairs
        # Leg ordering standard: 0=FL, 1=FR, 2=BL, 3=BR
        phases = [
            omega * self.cpg_time,  # FL Pair A
            omega * self.cpg_time + math.pi,  # FR Pair B
            omega * self.cpg_time + math.pi,  # BL Pair B
            omega * self.cpg_time  # BR Pair A
        ]

        # Coordinates defining nominal resting centers for each hip link bracket
        # FL, FR, BL, BR respectively
        base_x = [0.12, 0.12, -0.12, -0.12]
        base_y = [self.hip_offset_y, -self.hip_offset_y, self.hip_offset_y, -self.hip_offset_y]

        for i in range(4):
            p = phases[i]

            # Calculate base tracking paths along the forward and lateral axes
            x_target = stride_x * math.cos(p)
            y_target = stride_y * math.cos(p)

            # Vertical foot clearance lifting parabola profile
            z_target = self.default_z + (step_height * max(0.0, math.sin(p)))

            # APPLY DIFFERENTIAL YAW: Mixes turning angles directly into the foot vectors
            # Inner/Outer legs scale proportionally to trace a clean geometric arc path
            yaw_radius_x = base_y[i]
            yaw_radius_y = -base_x[i]
            x_target += w_yaw * yaw_radius_x * 0.4 * math.cos(p)
            y_target += w_yaw * yaw_radius_y * 0.4 * math.cos(p)

            # Map the clean Cartesian position paths straight through Inverse Kinematics
            hip, thigh, calf = self.analytical_ik(i, (x_target, y_target, z_target))

            # Send position coordinates down to the high-torque hardware servos
            pyb.setJointMotorControl2(self.robot_id, i * 3, pyb.POSITION_CONTROL, targetPosition=hip, force=5.88)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 1, pyb.POSITION_CONTROL, targetPosition=thigh, force=5.88)
            pyb.setJointMotorControl2(self.robot_id, i * 3 + 2, pyb.POSITION_CONTROL, targetPosition=calf, force=5.88)

    def run_loop(self):
        print("\n=== PURE MATHEMATICAL JOYSTICK CONTROLLER ACTIVE ===")
        print("Click on the PyBullet simulation window to pilot:")
        print("  [W] -> Shuffle Forward          [S] -> Shuffle Backward")
        print("  [A] -> Strafe Left              [D] -> Strafe Right")
        print("  [Q] -> Rotate Counter-Clockwise [E] -> Rotate Clockwise")
        print("  [X] -> Full Stationary Brake / Pause Stance")
        print("====================================================\n")

        while True:
            # Step the structural physics solver inside PyBullet environment
            pyb.stepSimulation(physicsClientId=self.client)

            # Read keyboard inputs directly
            keys = pyb.getKeyboardEvents()

            # Forward / Backward mapping
            if 119 in keys:  # 'W'
                self.commands[0] = max(self.commands[0] - 0.02, -0.30)
            elif 115 in keys:  # 'S'
                self.commands[0] = min(self.commands[0] + 0.02, 0.40)

            # Lateral Strafe mapping
            if 97 in keys:  # 'A'
                self.commands[1] = min(self.commands[1] + 0.02, 0.25)
            elif 100 in keys:  # 'D'
                self.commands[1] = max(self.commands[1] - 0.02, -0.25)

            # Yaw Rotational turn mapping
            if 113 in keys:  # 'Q'
                self.commands[2] = min(self.commands[2] + 0.04, 0.50)
            elif 101 in keys:  # 'E'
                self.commands[2] = max(self.commands[2] - 0.04, -0.50)

            # Emergency Stop / Full Reset
            if 120 in keys:  # 'X'
                self.commands = np.zeros(3, dtype=np.float32)

            # Decelerate commands smoothly toward zero if no key is actively held down
            if not keys:
                self.commands[0] *= 0.95
                self.commands[1] *= 0.95
                self.commands[2] *= 0.90

            # Execute the mathematical coordinate calculations
            self.update_gait()

            # Dynamic Camera tracking preserves focus directly behind the robot chassis position
            base_pos, _ = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
            pyb.resetDebugVisualizerCamera(
                cameraDistance=0.9, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=base_pos,
                physicsClientId=self.client
            )

            # Precision frequency locking at a steady 60 Hz execution window
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    dog_pilot = PureMathQuadruped()
    dog_pilot.run_loop()
