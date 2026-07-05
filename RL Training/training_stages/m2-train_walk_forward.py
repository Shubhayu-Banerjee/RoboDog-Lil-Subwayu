import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pyb
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


class QuadrupedEnv(gym.Env):
    """Custom Environment for Stage 2: Corrected X-Axis Direction and Extended Timeline"""

    def __init__(self, render_mode="direct"):
        super(QuadrupedEnv, self).__init__()

        self.render_mode = render_mode
        self.client = pyb.connect(pyb.GUI if render_mode == "human" else pyb.DIRECT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

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
            pyb.changeDynamics(self.robot_id, joint, lateralFriction=1.5, physicsClientId=self.client)
        pyb.changeDynamics(self.plane_id, -1, lateralFriction=1.5, physicsClientId=self.client)

        for i, angle in enumerate(self.neutral_stance):
            pyb.resetJointState(self.robot_id, i, angle, physicsClientId=self.client)

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
                force=5.88,  # 60kg-cm hardware ceiling
                physicsClientId=self.client
            )

        for _ in range(4):
            pyb.stepSimulation(physicsClientId=self.client)

        obs = self._get_obs()

        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        base_vel, base_ang_vel = pyb.getBaseVelocity(self.robot_id, physicsClientId=self.client)
        roll, pitch, yaw = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_vels = np.array([state[1] for state in joint_states])

        # FIX 1: Inverted local forward calculation to perfectly match your URDF alignment
        forward_vel = -(base_vel[0] * math.cos(yaw) + base_vel[1] * math.sin(yaw))

        contacts = 0
        for leg_idx in [2, 5, 8, 11]:
            contact_points = pyb.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=leg_idx,
                                                  physicsClientId=self.client)
            if len(contact_points) > 0:
                contacts += 1

        # --------------------------------------------------
        # LOCOMOTION REWARD MESH
        # --------------------------------------------------
        if forward_vel > 0:
            reward_vel = forward_vel * 25.0  # Increased driver payout
        else:
            reward_vel = forward_vel * 50.0  # Punish backward tracking heavily

        reward_height = 4.0 if (0.16 <= base_pos[2] <= 0.22) else 0.0
        penalty_orientation = (abs(roll) + abs(pitch)) * 2.0
        penalty_rocking = np.sum(np.square(base_ang_vel)) * 0.4
        penalty_airborne = 2.0 if contacts == 0 else 0.0
        penalty_joint_speed = np.sum(np.square(joint_vels)) * 0.02
        penalty_flicker = np.sum(np.abs(action - self.prev_action)) * 0.1

        penalty_lazy_legs = 0.0
        for i in range(0, 12, 3):
            hip_act = abs(joint_vels[i]) + abs(joint_vels[i + 1])
            calf_act = abs(joint_vels[i + 2])
            if calf_act > (hip_act * 1.5) or (hip_act + calf_act < 0.02):
                penalty_lazy_legs += 0.2

        reward = (reward_vel + reward_height - penalty_orientation - penalty_rocking -
                  penalty_airborne - penalty_joint_speed - penalty_flicker - penalty_lazy_legs)

        self.prev_action = np.copy(action)

        terminated = bool(
            base_pos[2] < 0.13 or
            abs(roll) > 0.45 or
            abs(pitch) > 0.45 or
            forward_vel < -0.02  # Instantly catches genuine backward failures
        )

        # FIX 2: Extended time limit to 4000 steps to allow full gait discovery
        truncated = bool(self.step_count >= 4000)

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        base_pos, base_quat = pyb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        roll, pitch, _ = pyb.getEulerFromQuaternion(base_quat)

        joint_states = pyb.getJointStates(self.robot_id, range(12), physicsClientId=self.client)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]

        return np.concatenate([[roll, pitch, base_pos[2]], joint_pos, joint_vel]).astype(np.float32)

    def close(self):
        pyb.disconnect(self.client)


def make_env(rank, seed=0):
    def _init():
        env = QuadrupedEnv(render_mode="direct")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    num_cpu = 8
    print(f"Setting up {num_cpu} Parallel CPU Environments for Walking Task...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    standing_checkpoint = "ppo_robodog_standing.zip"
    walking_checkpoint = "ppo_robodog_walking.zip"

    # Delete or rename any incomplete ppo_robodog_walking.zip to force a clean direction splice
    if os.path.exists(walking_checkpoint):
        print(f"Found existing walking checkpoint. Resuming training...")
        model = PPO.load(walking_checkpoint, env=vec_env, learning_rate=2e-4)
    elif os.path.exists(standing_checkpoint):
        print(f"Splicing weights from Milestone 1 Balance Checkpoint ('{standing_checkpoint}')...")
        model = PPO.load(standing_checkpoint, env=vec_env, learning_rate=2e-4)
    else:
        print("CRITICAL WARNING: No standing checkpoint found!")
        model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=2e-4, batch_size=256)

    print("Launching Walking Stride Training Loop...")
    try:
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
        model.save("ppo_robodog_walking")
        print("Training Complete! Walking policy saved as 'ppo_robodog_walking.zip'")
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving progress...")
        model.save(walking_checkpoint)
        print(f"Progress safely logged to '{walking_checkpoint}'.")