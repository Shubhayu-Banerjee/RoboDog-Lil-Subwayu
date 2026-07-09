import time
import math
import gymnasium as gym
import numpy as np
import pybullet as pyb
import pybullet_data
from stable_baselines3 import PPO


class QuadrupedPrecisionWalkingEnv(gym.Env):
    """Evaluation environment matching the high-precision 34-observation setup"""

    def __init__(self, render_mode="human"):
        super(QuadrupedPrecisionWalkingEnv, self).__init__()

        self.render_mode = render_mode
        self.client = pyb.connect(pyb.GUI if render_mode == "human" else pyb.DIRECT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

        self.robot_id = None
        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.smoothed_action = np.zeros(12, dtype=np.float32)

        self.neutral_stance = [
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6,
            0.0, -0.3, 0.6
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.smoothed_action = np.zeros(12, dtype=np.float32)

        pyb.resetSimulation(physicsClientId=self.client)
        pyb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pyb.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = pyb.loadURDF("plane.urdf", physicsClientId=self.client)
        start_pos = [0, 0, 0.19]
        start_ori = pyb.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = pyb.loadURDF("robot.urdf", start_pos, start_ori, useFixedBase=False,
                                     physicsClientId=self.client)

        for joint in range(12):
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.6, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.6, physicsClientId=self.client)

        for i, angle in enumerate(self.neutral_stance):
            pyb.resetJointState(self.robot_id, i, angle, physicsClientId=self.client)

        if self.render_mode == "human":
            pyb.resetDebugVisualizerCamera(
                cameraDistance=1.0, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.18],
                physicsClientId=self.client
            )
            # Draw a green line down the center of the target global track (Y=0)
            pyb.addUserDebugLine([-10.0, 0.0, 0.01], [10.0, 0.0, 0.01], [0, 1, 0], lineWidth=2,
                                 physicsClientId=self.client)

            # OPTIONAL VISUAL AID: Draw thin red lines marking the absolute 7cm boundaries
            pyb.addUserDebugLine([-10.0, 0.07, 0.01], [10.0, 0.07, 0.01], [1, 0, 0], lineWidth=1,
                                 physicsClientId=self.client)
            pyb.addUserDebugLine([-10.0, -0.07, 0.01], [10.0, -0.07, 0.01], [1, 0, 0], lineWidth=1,
                                 physicsClientId=self.client)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        self.smoothed_action = 0.8 * self.smoothed_action + 0.2 * action
        action_scale = 0.4
        target_angles = np.array(self.neutral_stance) + (self.smoothed_action * action_scale)

        for i in range(12):
            pyb.setJointMotorControl2(
                self.robot_id, i, pyb.POSITION_CONTROL,
                targetPosition=target_angles[i],
                force=5.88,
                physicsClientId=self.client
            )

        for _ in range(4):
            pyb.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_vel, _ = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = pyb.getEulerFromQuaternion(base_quat)

        # --- GAIT PROGRESSION VARIABLES ---
        distance_x_achieved = -base_pos[0]
        global_forward_vel = -base_vel[0]
        global_lateral_vel = base_vel[1]
        lateral_displacement = base_pos[1]

        # --- MATCHED STAGE 3 TUNING RESTRICTIONS ---
        terminated = bool(
            base_pos[2] < 0.12 or
            abs(roll) > 0.45 or
            abs(pitch) > 0.45 or
            abs(yaw) > 0.10 or  # Updated from 0.30 -> 0.20 rad
            abs(base_pos[1]) > 0.07 or  # ENFORCED UPDATED 7CM CORRIDOR
            global_forward_vel < -0.05
        )
        truncated = bool(self.step_count >= 6000)  # Extended to match 6k step training parameters

        # Pack telemetry dictionary to pipe into evaluation runtime logs
        info = {
            "dist_x": distance_x_achieved,
            "vel_x": global_forward_vel,
            "vel_y": global_lateral_vel,
            "drift_y": lateral_displacement
        }

        return obs, 0.0, terminated, truncated, info

    def _get_obs(self):
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        _, base_ang_vel = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        foot_indices = [2, 5, 8, 11]
        contacts = []
        for leg_idx in foot_indices:
            cp = pyb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=leg_idx,
                                      physicsClientId=self.client)
            contacts.append(1.0 if len(cp) > 0 else 0.0)

        return np.concatenate([
            [roll, pitch, yaw],
            base_ang_vel,
            joint_pos,
            joint_vel,
            contacts
        ]).astype(np.float32)

    def close(self):
        pyb.disconnect(self.client)


if __name__ == "__main__":
    checkpoint_path = "ppo_robodog_walking.zip"

    print(f"Booting evaluation sequence from '{checkpoint_path}'...")
    env = QuadrupedPrecisionWalkingEnv(render_mode="human")

    try:
        model = PPO.load(checkpoint_path, env=env)
        print("Weights bound successfully. Telemetry tracker active.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{checkpoint_path}' missing.")
        env.close()
        exit()

    while True:
        obs, _ = env.reset()
        done = False

        print("\n======================= GAIT ANALYSIS LIVE =======================")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # Print refreshing terminal diagnostic line
            print(
                f"Progress [+X]: {info['dist_x']:.3f}m | "
                f"Vel [X]: {info['vel_x']:.3f}m/s | "
                f"Drift [Y]: {info['drift_y']:.3f}m | "
                f"Vel [Y]: {info['vel_y']:.3f}m/s",
                end="\r"
            )

            time.sleep(1.0 / 60.0)

            if terminated:
                print(f"\n[Reset Triggered]: Breached high-precision corridor boundary parameters.")
            if truncated:
                print(f"\n[Success]: High-precision path target run finished flawlessly.")